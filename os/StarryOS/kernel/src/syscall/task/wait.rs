use alloc::{sync::Arc, vec::Vec};
use core::{future::poll_fn, task::Poll};

use ax_errno::{AxError, AxResult, LinuxError};
use ax_task::{
    current,
    future::{block_on, interruptible},
};
use bitflags::bitflags;
use linux_raw_sys::general::{
    __WALL, __WCLONE, __WNOTHREAD, P_ALL, P_PGID, P_PID, P_PIDFD, WCONTINUED, WEXITED, WNOHANG,
    WNOWAIT, WUNTRACED,
};
use starry_process::{Pid, Process};
use starry_signal::SignalInfo;
use starry_vm::{VmMutPtr, VmPtr};

use crate::{
    file::{PidFd, get_file_like},
    task::{
        AsThread, JobStatus, decode_wait_status, get_process_data, get_task,
        get_zombie_cred, is_zombie_clone_child, remove_process,
        unregister_zombie, zombie_wait_parent_tid,
    },
};

bitflags! {
    /// Options accepted by wait4 / waitpid.
    #[derive(Debug)]
    struct WaitPidOptions: u32 {
        const WNOHANG = WNOHANG;
        const WUNTRACED = WUNTRACED;
        const WCONTINUED = WCONTINUED;
        const WNOTHREAD = __WNOTHREAD;
        const WALL = __WALL;
        const WCLONE = __WCLONE;
    }
}

bitflags! {
    /// Options accepted by waitid.
    #[derive(Debug)]
    struct WaitIdOptions: u32 {
        const WNOHANG = WNOHANG;
        const WUNTRACED = WUNTRACED;
        const WEXITED = WEXITED;
        const WCONTINUED = WCONTINUED;
        const WNOWAIT = WNOWAIT;
        const WNOTHREAD = __WNOTHREAD;
        const WALL = __WALL;
        const WCLONE = __WCLONE;
    }
}

#[derive(Debug, Clone, Copy)]
enum WaitTarget {
    /// Wait for any child process
    Any,
    /// Wait for the child whose process ID is equal to the value.
    Pid(Pid),
    /// Wait for any child process whose process group ID is equal to the value.
    Pgid(Pid),
}

impl WaitTarget {
    fn matches(&self, child: &Process) -> bool {
        match self {
            WaitTarget::Any => true,
            WaitTarget::Pid(pid) => child.pid() == *pid,
            WaitTarget::Pgid(pgid) => child.group().pgid() == *pgid,
        }
    }

    fn matches_process_or_thread(&self, child: &Process) -> bool {
        self.matches(child) || matches!(self, WaitTarget::Pid(pid) if child.threads().contains(pid))
    }

    }

fn waitid_pidfd_target(fd: i32) -> AxResult<WaitTarget> {
    if fd < 0 {
        return Err(AxError::InvalidInput);
    }
    let pidfd = get_file_like(fd)?
        .downcast_arc::<PidFd>()
        .map_err(|_| AxError::BadFileDescriptor)?;
    Ok(WaitTarget::Pid(pidfd.pid()))
}

fn child_uid(child: &Process) -> u32 {
    get_zombie_cred(child.pid())
        .map(|cred| cred.uid)
        .or_else(|| {
            child
                .threads()
                .into_iter()
                .find_map(|tid| get_task(tid).ok().map(|task| task.as_thread().cred().uid))
        })
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy)]
struct WaitChildFilter {
    wall: bool,
    clone: bool,
    no_thread: bool,
}

impl WaitChildFilter {
    fn from_waitpid_options(options: &WaitPidOptions) -> Self {
        Self {
            wall: options.contains(WaitPidOptions::WALL),
            clone: options.contains(WaitPidOptions::WCLONE),
            no_thread: options.contains(WaitPidOptions::WNOTHREAD),
        }
    }

    fn from_waitid_options(options: &WaitIdOptions) -> Self {
        Self {
            wall: options.contains(WaitIdOptions::WALL),
            clone: options.contains(WaitIdOptions::WCLONE),
            no_thread: options.contains(WaitIdOptions::WNOTHREAD),
        }
    }

    fn matches_clone_kind(&self, is_clone_child: bool) -> bool {
        self.wall || is_clone_child == self.clone
    }

    fn matches_process(&self, child: &Process, current_tid: Pid) -> bool {
        if self.no_thread {
            let wait_parent_tid = get_process_data(child.pid())
                .ok()
                .map(|data| data.wait_parent_tid)
                .or_else(|| zombie_wait_parent_tid(child.pid()));
            if wait_parent_tid != Some(current_tid) {
                return false;
            }
        }

        let is_clone_child = get_process_data(child.pid())
            .ok()
            .map(|data| data.is_clone_child())
            .or_else(|| is_zombie_clone_child(child.pid()))
            .unwrap_or(false);
        self.matches_clone_kind(is_clone_child)
    }
}

fn waitable_processes(
    proc: &Process,
    target: WaitTarget,
    current_tid: Pid,
    filter: WaitChildFilter,
) -> Vec<Arc<Process>> {
    let candidates = proc
        .children()
        .into_iter()
        .filter(|child| target.matches(child) && filter.matches_process(child, current_tid))
        .collect::<Vec<_>>();

    candidates
}

pub fn sys_waitpid(pid: i32, exit_code: *mut i32, options: u32) -> AxResult<isize> {
    let options = WaitPidOptions::from_bits(options).ok_or(AxError::InvalidInput)?;
    info!("sys_waitpid <= pid: {pid:?}, options: {options:?}");

    let curr = current();
    let thr = curr.as_thread();
    let proc = &thr.proc_data.proc;

    let target = if pid == -1 {
        WaitTarget::Any
    } else if pid == 0 {
        WaitTarget::Pgid(proc.group().pgid())
    } else if pid > 0 {
        WaitTarget::Pid(pid as _)
    } else {
        WaitTarget::Pgid(-pid as _)
    };

    let children = waitable_processes(
        proc,
        target,
        thr.tid(),
        WaitChildFilter::from_waitpid_options(&options),
    );
    if children.is_empty() {
        return Err(AxError::from(LinuxError::ECHILD));
    }

    let proc_data = curr.as_thread().proc_data.clone();
    let check_children = || {
        if let Some(child) = children.iter().find(|child| child.is_zombie()) {
            // Accumulate child's CPU time before freeing.
            for tid in child.threads() {
                if let Ok(task) = get_task(tid) {
                    let thr = task.as_thread();
                    let (utime, stime) = thr.time.borrow().output();
                    proc_data.add_child_cpu_time(utime, stime);
                }
            }
            // Copy status to userspace before `free` / `unregister_zombie`. If
            // `vm_write` fails we must leave the zombie intact so the parent can
            // retry; freeing first would strand the process and corrupt wait
            // accounting (Linux also publishes the status byte before full reap).
            if let Some(exit_code) = exit_code.nullable() {
                exit_code.vm_write(child.exit_code())?;
            }
            child.free();
            remove_process(child.pid());
            unregister_zombie(child.pid());
            return Ok(Some(child.pid() as _));
        }

        // Job-control status: a stopped (WUNTRACED) or continued (WCONTINUED)
        // child reports its status without being reaped, unlike a zombie.
        let want_stopped = options.contains(WaitPidOptions::WUNTRACED);
        let want_continued = options.contains(WaitPidOptions::WCONTINUED);
        if want_stopped || want_continued {
            for child in &children {
                let Ok(cdata) = get_process_data(child.pid()) else {
                    continue;
                };
                if let Some(status) = cdata.peek_job_status_if(want_stopped, want_continued) {
                    // Linux wait status encoding: stopped = (signo << 8) | 0x7f
                    // (W_STOPCODE), continued = 0xffff (__W_CONTINUED).
                    let raw = match status {
                        JobStatus::Stopped(signo) => ((signo as i32) << 8) | 0x7f,
                        JobStatus::Continued => 0xffff,
                    };
                    // Publish to userspace before consuming, so a faulting
                    // `exit_code` pointer leaves the report intact to retry
                    // (mirrors the zombie-reap ordering above).
                    if let Some(exit_code) = exit_code.nullable() {
                        exit_code.vm_write(raw)?;
                    }
                    cdata.take_job_status_if(want_stopped, want_continued);
                    return Ok(Some(child.pid() as _));
                }
            }
        }

        if options.contains(WaitPidOptions::WNOHANG) {
            Ok(Some(0))
        } else {
            Ok(None)
        }
    };

    block_on(interruptible(poll_fn(|cx| {
        match check_children().transpose() {
            Some(res) => Poll::Ready(res),
            None => {
                // Registration happens from wait task context.
                unsafe {
                    proc_data
                        .child_exit_event
                        .register(cx.waker(), axpoll::IoEvents::IN)
                };
                // A child may exit between the check above and waker
                // registration. Recheck after registering so that wakeup is
                // not lost in that race window.
                match check_children().transpose() {
                    Some(res) => Poll::Ready(res),
                    None => Poll::Pending,
                }
            }
        }
    })))?
}

pub fn sys_waitid(
    idtype: u32,
    id: i32,
    infop: *mut linux_raw_sys::general::siginfo,
    options: u32,
) -> AxResult<isize> {
    let curr = current();
    let thr = curr.as_thread();
    let proc = &thr.proc_data.proc;

    // Validate idtype
    let target = match idtype {
        P_ALL => WaitTarget::Any,
        P_PID => {
            if id <= 0 {
                return Err(AxError::InvalidInput);
            }
            WaitTarget::Pid(id as Pid)
        }
        P_PGID => {
            if id < 0 {
                return Err(AxError::InvalidInput);
            }
            let pgid = if id == 0 {
                proc.group().pgid()
            } else {
                id as Pid
            };
            WaitTarget::Pgid(pgid)
        }
        P_PIDFD => waitid_pidfd_target(id)?,
        _ => return Err(AxError::InvalidInput),
    };

    let options = WaitIdOptions::from_bits(options).ok_or(AxError::InvalidInput)?;
    if !options
        .intersects(WaitIdOptions::WEXITED | WaitIdOptions::WUNTRACED | WaitIdOptions::WCONTINUED)
    {
        return Err(AxError::InvalidInput);
    }

    info!("sys_waitid <= idtype: {idtype}, id: {id}, options: {options:?}");

    let children = waitable_processes(
        proc,
        target,
        thr.tid(),
        WaitChildFilter::from_waitid_options(&options),
    );
    if children.is_empty() {
        return Err(AxError::from(LinuxError::ECHILD));
    }

    let proc_data = curr.as_thread().proc_data.clone();
    let check_children = || {
        if options.contains(WaitIdOptions::WEXITED)
            && let Some(child) = children.iter().find(|child| child.is_zombie())
        {
            let child_pid = child.pid();
            let (code, status) = decode_wait_status(child.exit_code());
            let child_uid = child_uid(child);

            if let Some(infop) = infop.nullable() {
                let siginfo = SignalInfo::new_sigchld(child_pid, child_uid, code, status);
                infop.vm_write(siginfo.0)?;
            }

            if !options.contains(WaitIdOptions::WNOWAIT) {
                for tid in child.threads() {
                    if let Ok(task) = get_task(tid) {
                        let thr = task.as_thread();
                        let (utime, stime) = thr.time.borrow().output();
                        proc_data.add_child_cpu_time(utime, stime);
                    }
                }
                child.free();
                remove_process(child_pid);
                unregister_zombie(child_pid);
            }
            return Ok(Some(0));
        }

        if options.contains(WaitIdOptions::WNOHANG) {
            if let Some(infop) = infop.nullable() {
                let zeroed: linux_raw_sys::general::siginfo = unsafe { core::mem::zeroed() };
                infop.vm_write(zeroed)?;
            }
            Ok(Some(0))
        } else {
            Ok(None)
        }
    };

    block_on(interruptible(poll_fn(|cx| {
        match check_children().transpose() {
            Some(res) => Poll::Ready(res),
            None => {
                // Registration happens from wait task context.
                unsafe {
                    proc_data
                        .child_exit_event
                        .register(cx.waker(), axpoll::IoEvents::IN)
                };
                match check_children().transpose() {
                    Some(res) => Poll::Ready(res),
                    None => Poll::Pending,
                }
            }
        }
    })))?
}
