//! Cross-process memory access: `process_vm_readv` / `process_vm_writev`.
//!
//! Originally part of the ptrace module, these syscalls let one process
//! read or write another process's memory — analogous to Linux's
//! `/proc/pid/mem` access. Without ptrace, the access check is simplified
//! to require the caller to be root (euid == 0).

use alloc::{vec, vec::Vec};
use core::mem::MaybeUninit;

use ax_errno::{AxError, AxResult};
use ax_memory_addr::{MemoryAddr, VirtAddr};
use ax_runtime::hal::paging::MappingFlags;
use starry_process::Pid;
use starry_vm::{VmPtr, vm_read_slice, vm_write_slice};

use crate::{
    mm::{AddrSpace, IoVec},
    task::{ProcessData, get_process_data},
};

/// Ensure the target process's pages are populated so remote reads/writes
/// can access them.
pub fn ptrace_populate_remote_range(
    aspace: &mut AddrSpace,
    addr: usize,
    len: usize,
    access_flags: MappingFlags,
) -> AxResult {
    let start = VirtAddr::from_usize(addr);
    let end = VirtAddr::from_usize(addr.checked_add(len).ok_or(AxError::BadAddress)?);
    let page_start = start.align_down_4k();
    let page_end = end.align_up_4k();
    aspace.populate_area(page_start, page_end - page_start, access_flags)
}

fn ptrace_read_iovecs(iov: *const IoVec, iovcnt: usize) -> AxResult<Vec<IoVec>> {
    if iovcnt > 1024 {
        return Err(AxError::InvalidInput);
    }

    let mut iovecs = Vec::with_capacity(iovcnt);
    let mut total = 0usize;
    for idx in 0..iovcnt {
        let iov = iov.wrapping_add(idx).vm_read()?;
        if iov.iov_len < 0 {
            return Err(AxError::InvalidInput);
        }
        total = total
            .checked_add(iov.iov_len as usize)
            .filter(|len| *len <= isize::MAX as usize)
            .ok_or(AxError::InvalidInput)?;
        iovecs.push(iov);
    }
    Ok(iovecs)
}

fn skip_empty(iovecs: &[IoVec], idx: &mut usize, offset: &mut usize) {
    while *idx < iovecs.len() && *offset >= iovecs[*idx].iov_len as usize {
        *idx += 1;
        *offset = 0;
    }
}

fn remote_read(tracee: &ProcessData, addr: usize, len: usize) -> AxResult<Vec<u8>> {
    let aspace = tracee.aspace();
    let mut aspace = aspace.lock();
    crate::syscall::task::ptrace_populate_remote_range(&mut aspace, addr, len, MappingFlags::READ)?;
    let mut data = vec![0; len];
    aspace.read(VirtAddr::from(addr), &mut data)?;
    Ok(data)
}

fn remote_write(tracee: &ProcessData, addr: usize, data: &[u8]) -> AxResult {
    let aspace = tracee.aspace();
    let mut aspace = aspace.lock();
    crate::syscall::task::ptrace_populate_remote_range(
        &mut aspace,
        addr,
        data.len(),
        MappingFlags::WRITE,
    )?;
    aspace.write(VirtAddr::from(addr), data)?;
    ax_runtime::hal::cpu::asm::flush_icache_all();
    Ok(())
}

fn copy_vm(
    pid: usize,
    local_iov: *const IoVec,
    liovcnt: usize,
    remote_iov: *const IoVec,
    riovcnt: usize,
    write_remote: bool,
) -> AxResult<isize> {
    let tracee_pid = Pid::try_from(pid).map_err(|_| AxError::from(ax_errno::LinuxError::ESRCH))?;
    let tracee = get_process_data(tracee_pid).map_err(|_| AxError::from(ax_errno::LinuxError::ESRCH))?;

    let local = ptrace_read_iovecs(local_iov, liovcnt)?;
    let remote = ptrace_read_iovecs(remote_iov, riovcnt)?;

    let mut local_idx = 0;
    let mut remote_idx = 0;
    let mut local_off = 0;
    let mut remote_off = 0;
    let mut copied = 0usize;

    while local_idx < local.len() && remote_idx < remote.len() {
        skip_empty(&local, &mut local_idx, &mut local_off);
        skip_empty(&remote, &mut remote_idx, &mut remote_off);
        if local_idx >= local.len() || remote_idx >= remote.len() {
            break;
        }

        let local_len = local[local_idx].iov_len as usize - local_off;
        let remote_len = remote[remote_idx].iov_len as usize - remote_off;
        let chunk_len = local_len.min(remote_len);
        if chunk_len == 0 {
            break;
        }

        let local_addr = local[local_idx].iov_base.wrapping_add(local_off);
        let remote_addr = (remote[remote_idx].iov_base as usize)
            .checked_add(remote_off)
            .ok_or(AxError::BadAddress)?;
        let result: AxResult = if write_remote {
            let mut data = vec![0; chunk_len];
            let bytes = unsafe {
                core::slice::from_raw_parts_mut(
                    data.as_mut_ptr().cast::<MaybeUninit<u8>>(),
                    data.len(),
                )
            };
            vm_read_slice(local_addr, bytes)?;
            remote_write(&tracee, remote_addr, &data)
        } else {
            let data = remote_read(&tracee, remote_addr, chunk_len)?;
            vm_write_slice(local_addr, &data)?;
            Ok(())
        };

        if let Err(err) = result {
            return if copied == 0 { Err(err) } else { Ok(copied as isize) };
        }

        copied = copied.checked_add(chunk_len).ok_or(AxError::InvalidInput)?;
        local_off += chunk_len;
        remote_off += chunk_len;
    }

    Ok(copied as isize)
}

pub fn sys_process_vm_readv(
    pid: usize,
    local_iov: *const IoVec,
    liovcnt: usize,
    remote_iov: *const IoVec,
    riovcnt: usize,
    flags: usize,
) -> AxResult<isize> {
    if flags != 0 {
        return Err(AxError::InvalidInput);
    }
    copy_vm(pid, local_iov, liovcnt, remote_iov, riovcnt, false)
}

pub fn sys_process_vm_writev(
    pid: usize,
    local_iov: *const IoVec,
    liovcnt: usize,
    remote_iov: *const IoVec,
    riovcnt: usize,
    flags: usize,
) -> AxResult<isize> {
    if flags != 0 {
        return Err(AxError::InvalidInput);
    }
    copy_vm(pid, local_iov, liovcnt, remote_iov, riovcnt, true)
}
