use ax_runtime::hal::cpu::uspace::{ExceptionInfo, ExceptionKind, ReturnReason, UserContext};
use ax_task::TaskInner;
use starry_process::Pid;
use starry_signal::{SEGV_ACCERR, SEGV_MAPERR, SignalInfo, Signo};
use starry_vm::{VmMutPtr, VmPtr};

use super::{
    AsThread, SyscallRestartInfo, TimerState, check_signals, poll_process_timer,
    raise_signal_fatal, set_timer_state,
    unblock_next_signal,
};
use crate::syscall::{handle_syscall, syscall_allows_signal_restart};

/// Create a new user task.
pub fn new_user_task(name: &str, mut uctx: UserContext, set_child_tid: usize) -> TaskInner {
    TaskInner::new(
        move || {
            let curr = ax_task::current();

            if let Some(tid) = (set_child_tid as *mut Pid).nullable() {
                tid.vm_write(curr.as_thread().tid() as Pid).ok();
            }

            info!("Enter user space: ip={:#x}, sp={:#x}", uctx.ip(), uctx.sp());

            let thr = curr.as_thread();
            while !thr.pending_exit() {
                let reason = uctx.run();

                set_timer_state(&curr, TimerState::Kernel);

                let saved_a0 = uctx.arg0();
                let saved_sysno = uctx.sysno();
                let is_syscall = matches!(reason, ReturnReason::Syscall);

                match reason {
                    ReturnReason::Syscall => {
                        handle_syscall(&mut uctx);
                    }
                    ReturnReason::PageFault(addr, flags) => {
                        // Classify si_code while holding the aspace lock: an
                        // existing mapping that rejected the access is a
                        // permission violation (SEGV_ACCERR), otherwise the
                        // address is unmapped (SEGV_MAPERR) — matching Linux's
                        // do_user_addr_fault().
                        let si_code = {
                            let aspace = thr.proc_data.aspace();
                            let mut aspace = aspace.lock();
                            if aspace.handle_page_fault(addr, flags) {
                                None
                            } else if aspace.find_area(addr).is_some() {
                                Some(SEGV_ACCERR)
                            } else {
                                Some(SEGV_MAPERR)
                            }
                        };
                        if let Some(si_code) = si_code {
                            warn!(
                                "{:?}: segmentation fault at {:#x} {:?}",
                                thr.proc_data.proc, addr, flags
                            );
                            // POSIX: a synchronous SIGSEGV must carry the
                            // faulting address in si_addr so handlers can
                            // classify and recover from guard-page / implicit-
                            // null-check faults.
                            raise_signal_fatal(
                                SignalInfo::new_fault(Signo::SIGSEGV, si_code, addr.as_usize()),
                                &uctx,
                            )
                            .expect("Failed to send SIGSEGV");
                        }
                    }
                    ReturnReason::Interrupt => {}
                    #[allow(unused_labels)]
                    ReturnReason::Exception(exc_info) => 'exc: {
                        let kind = exc_info.kind();
                        // A uprobe plants an `int3` in user text (delivered as a
                        // #BP / Breakpoint exception) and completes its
                        // out-of-line single-step via a #DB / Debug exception.
                        // Route both to this process' uprobe manager before any
                        // ptrace / signal handling: if a uprobe owns the
                        // faulting address it fixes up `uctx` (sets the
                        // out-of-line PC + single-step, or restores PC after the
                        // step) and we resume directly. If not, fall through.
                        match kind {
                            ExceptionKind::Breakpoint
                                if crate::uprobe::break_uprobe_handler(&mut uctx).is_some() =>
                            {
                                break 'exc;
                            }
                            // x86_64 completes the out-of-line single-step via a
                            // #DB; other arches handle stepping inside the
                            // breakpoint path, so the debug hook is x86_64-only.
                            #[cfg(target_arch = "x86_64")]
                            ExceptionKind::Debug
                                if crate::uprobe::debug_uprobe_handler(&mut uctx).is_some() =>
                            {
                                break 'exc;
                            }
                            _ => {}
                        }
                        warn!(
                            "user exception: ip={:#x}, fault_addr={:#x}, kind={:?}, esr={:#x}, \
                             ec={:#x}, iss={:#x}, info={:?}",
                            uctx.ip(),
                            exception_fault_addr(&exc_info),
                            kind,
                            exception_esr_value(&exc_info),
                            exception_ec_value(&exc_info),
                            exception_iss_value(&exc_info),
                            exc_info
                        );
                        let signo = match kind {
                            ExceptionKind::Misaligned => {
                                #[cfg(target_arch = "loongarch64")]
                                if unsafe { uctx.emulate_unaligned() }.is_ok() {
                                    break 'exc;
                                }
                                Signo::SIGBUS
                            }
                            ExceptionKind::Breakpoint => Signo::SIGTRAP,
                            ExceptionKind::IllegalInstruction => {
                                // AArch64 EL0 reads of ID_AA64*_EL1 (CPU feature
                                // detection, e.g. the Go runtime) trap as EC=0 /
                                // IllegalInstruction. Emulate them like Linux
                                // instead of killing the program with SIGILL.
                                #[cfg(target_arch = "aarch64")]
                                if unsafe { uctx.emulate_mrs_id_reg() } {
                                    break 'exc;
                                }
                                Signo::SIGILL
                            }
                            _ => Signo::SIGTRAP,
                        };
                        raise_signal_fatal(SignalInfo::new_kernel(signo), &uctx)
                            .expect("Failed to send SIGTRAP");
                    }
                    r => {
                        warn!("Unexpected return reason: {r:?}");
                        raise_signal_fatal(SignalInfo::new_kernel(Signo::SIGSEGV), &uctx)
                            .expect("Failed to send SIGSEGV");
                    }
                }

                if !unblock_next_signal() {
                    // POSIX timers are also driven by the alarm task, but polling
                    // here closes the window where an expired timer is only noticed
                    // after the current syscall returns to userspace.
                    poll_process_timer(thr.proc_data.proc.pid());

                    let eintr_code = -(ax_errno::LinuxError::EINTR.code() as isize);
                    let restart = if is_syscall
                        && (uctx.retval() as isize) == eintr_code
                        && syscall_allows_signal_restart(saved_sysno)
                    {
                        Some(SyscallRestartInfo {
                            saved_a0,
                            saved_sysno,
                        })
                    } else {
                        None
                    };
                    // Single-shot: the first delivered signal decides
                    // whether to restart. Subsequent signals in the same
                    // loop must not re-apply the decision.
                    let mut pending_restart = restart.as_ref();
                    while check_signals(thr, &mut uctx, None, pending_restart) {
                        pending_restart = None;
                    }
                }

                set_timer_state(&curr, TimerState::User);
                curr.clear_interrupt();
            }
        },
        name.into(),
        crate::config::KERNEL_STACK_SIZE,
    )
}

#[cfg(target_arch = "aarch64")]
fn exception_fault_addr(exc_info: &ExceptionInfo) -> usize {
    exc_info.far
}

#[cfg(target_arch = "aarch64")]
fn exception_esr_value(exc_info: &ExceptionInfo) -> u64 {
    exc_info.esr_value()
}

#[cfg(target_arch = "aarch64")]
fn exception_ec_value(exc_info: &ExceptionInfo) -> u64 {
    exc_info.ec_value()
}

#[cfg(target_arch = "aarch64")]
fn exception_iss_value(exc_info: &ExceptionInfo) -> u64 {
    exc_info.iss_value()
}

#[cfg(target_arch = "riscv64")]
fn exception_fault_addr(exc_info: &ExceptionInfo) -> usize {
    exc_info.stval
}

#[cfg(target_arch = "riscv64")]
fn exception_esr_value(_exc_info: &ExceptionInfo) -> u64 {
    0
}

#[cfg(target_arch = "riscv64")]
fn exception_ec_value(_exc_info: &ExceptionInfo) -> u64 {
    0
}

#[cfg(target_arch = "riscv64")]
fn exception_iss_value(_exc_info: &ExceptionInfo) -> u64 {
    0
}

#[cfg(target_arch = "loongarch64")]
fn exception_fault_addr(exc_info: &ExceptionInfo) -> usize {
    exc_info.badv
}

#[cfg(target_arch = "loongarch64")]
fn exception_esr_value(_exc_info: &ExceptionInfo) -> u64 {
    0
}

#[cfg(target_arch = "loongarch64")]
fn exception_ec_value(_exc_info: &ExceptionInfo) -> u64 {
    _exc_info.ecode as u64
}

#[cfg(target_arch = "loongarch64")]
fn exception_iss_value(_exc_info: &ExceptionInfo) -> u64 {
    _exc_info.esubcode as u64
}

#[cfg(target_arch = "x86_64")]
fn exception_fault_addr(exc_info: &ExceptionInfo) -> usize {
    exc_info.cr2
}

#[cfg(target_arch = "x86_64")]
fn exception_esr_value(_exc_info: &ExceptionInfo) -> u64 {
    0
}

#[cfg(target_arch = "x86_64")]
fn exception_ec_value(_exc_info: &ExceptionInfo) -> u64 {
    0
}

#[cfg(target_arch = "x86_64")]
fn exception_iss_value(_exc_info: &ExceptionInfo) -> u64 {
    0
}
