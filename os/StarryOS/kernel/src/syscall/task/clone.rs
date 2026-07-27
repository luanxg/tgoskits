use alloc::sync::Arc;

use ax_errno::{AxError, AxResult};
use ax_fs_ng::vfs::FS_CONTEXT;
use ax_kspin::SpinNoIrq;
use ax_runtime::hal::cpu::uspace::UserContext;
use ax_task::{AxTaskExt, current, spawn_task};
use bitflags::bitflags;
use linux_raw_sys::general::*;
use starry_process::Pid;
use starry_signal::Signo;
use starry_vm::VmMutPtr;

use crate::{
    file::{FD_TABLE, FileLike, PidFd, close_file_like},
    mm::copy_from_kernel,
    task::{AsThread, ProcessData, ProcessImage, Thread, add_task_to_table, new_user_task},
};

bitflags! {
    /// Options for use with [`sys_clone`] and [`sys_clone3`].
    #[derive(Debug, Clone, Copy, Default)]
    pub struct CloneFlags: u64 {
        /// 调用进程与子进程运行在同一内存空间中。
        const VM = CLONE_VM as u64;
        /// 调用者与子进程共享相同的文件系统信息。
        const FS = CLONE_FS as u64;
        /// 调用进程与子进程共享相同的文件描述符表。
        const FILES = CLONE_FILES as u64;
        /// 调用进程与子进程共享相同的信号处理器表。
        const SIGHAND = CLONE_SIGHAND as u64;
        /// 将 pidfd 设置为子进程的 PID 文件描述符。
        const PIDFD = CLONE_PIDFD as u64;
        /// 调用进程的执行将被挂起，直到子进程通过调用 execve(2) 或 _exit(2)
        /// 释放其虚拟内存资源（与 vfork(2) 的行为一致）。
        const VFORK = CLONE_VFORK as u64;
        /// 新子进程的父进程（由 getppid(2) 返回）将与调用进程的父进程相同。
        const PARENT = CLONE_PARENT as u64;
        /// 子进程被放入与调用进程相同的线程组中。
        const THREAD = CLONE_THREAD as u64;
        /// 子进程被放入与调用进程相同的线程组中。
        const SYSVSEM = CLONE_SYSVSEM as u64;
        /// 将 TLS（线程局部存储）描述符设置为 tls。
        const SETTLS = CLONE_SETTLS as u64;
        /// 将子线程 ID 存储到父进程的内存中。
        const PARENT_SETTID = CLONE_PARENT_SETTID as u64;
        /// 在子进程退出时，清除（置零）子进程内存中的子线程 ID，
        /// 并在该地址上的 futex 执行唤醒操作。
        const CHILD_CLEARTID = CLONE_CHILD_CLEARTID as u64;
        /// 将子线程 ID 存储到子进程的内存中。
        const CHILD_SETTID = CLONE_CHILD_SETTID as u64;
        /// 新进程与调用进程共享 I/O 上下文。
        const IO = CLONE_IO as u64;
        /// 在 clone 时清除信号处理器（自 Linux 5.5 起）。
        const CLEAR_SIGHAND = 0x100000000u64;
        /// （已废弃）导致子进程终止时父进程不会收到信号。
        const DETACHED = CLONE_DETACHED as u64;
    }
}


/// Unified arguments for clone/clone3/fork/vfork.
#[derive(Debug, Clone, Copy, Default)]
pub struct CloneArgs {
    pub flags: CloneFlags,
    pub exit_signal: u64,   //子进程退出时向父进程发送的信号编号，如果为0，子进程退出时不发送信号，若同时设置了THREAD或PARENT标志，则exit_signal必须为0（线程退出不发信号给父进程）
    pub stack: usize,       //子进程的用户栈指针，创建线程时（CLONE_VM|CLONE_THREAD）调用者需要手动分配栈内存并将此字段设为栈顶地址，对于fork/vfork，此字段通常为0，内核会将父进程当前的 SP 值作为子进程的初始 SP，子进程拥有父进程栈的 COW 副本。
    pub tls: usize,         //线程局部存储（TLS）描述符。仅在设置 SETTLS 标志时生效，内核会将其写入子进程的 TLS 寄存器（如 aarch64 的 tpidr_el0、x86 的 fs 段基址）。这是 pthread_create 实现线程局部存储的关键。
    pub parent_tid: usize,  //父进程中的一个用户空间内存地址。当设置了 PARENT_SETTID 标志时，内核会将子进程的 TID 写入该地址。这样父进程就能获知新创建的子线程/子进程的 ID，无需额外一次系统调用。
    pub child_tid: usize,   //子进程中的一个用户空间内存地址，有两个用途：1. CHILD_SETTID：子进程创建后，内核将子进程自己的 TID 写入此地址（常用于 pthread_create 让线程知道自己的 TID）。CHILD_CLEARTID：子进程退出时，内核将此地址清零并对该地址做 futex 唤醒。pthread_join 就是在该地址上做 FUTEX_WAIT 来实现等待线程退出的。
    pub pidfd: usize,       //当设置了 PIDFD 标志时，内核会在该地址写入一个指向子进程的 PID 文件描述符（PidFd），父进程可以通过这个 fd 用 poll/select/epoll 等方式异步等待子进程退出，而不需要处理 SIGCHLD 信号。
}

impl CloneArgs {
    fn validate(&self) -> AxResult<()> {
        //获取CloneArgs中的flags和exit_signal
        let Self {
            flags, exit_signal, ..
        } = self;

        //线程退出不发送信号，PARENT创建的进程不是通过信号的方式通知父进程，而是通过其他方式通知
        //CloneFlags::THREAD | CloneFlags::PARENT，二者中存在一个就成立
        if *exit_signal > 0 && flags.intersects(CloneFlags::THREAD | CloneFlags::PARENT) {
            return Err(AxError::InvalidInput);
        }
        //创建线程必须共享地址空间和信号处理器表。
        if flags.contains(CloneFlags::THREAD)
            && !flags.contains(CloneFlags::VM | CloneFlags::SIGHAND)
        {
            return Err(AxError::InvalidInput);
        }
        //共享信号处理器的前提是共享地址空间，反之不必（共享 VM 不需要共享 SIGHAND）。
        if flags.contains(CloneFlags::SIGHAND) && !flags.contains(CloneFlags::VM) {
            return Err(AxError::InvalidInput);
        }
        //VFORK和THREAD不能同时设置，任意一个单独设置没问题
        if flags.contains(CloneFlags::VFORK | CloneFlags::THREAD) {
            return Err(AxError::InvalidInput);
        }
        if flags.contains(CloneFlags::PIDFD | CloneFlags::DETACHED) {
            return Err(AxError::InvalidInput);
        }

        Ok(())
    }

    pub fn do_clone(self, uctx: &UserContext) -> AxResult<isize> {
        self.validate()?;

        let Self {
            flags,
            exit_signal,
            stack,
            tls,
            parent_tid,
            child_tid,
            pidfd,
        } = self;

        debug!(
            "do_clone <= flags: {:?}, exit_signal: {}, stack: {:#x}, tls: {:#x}",
            flags, exit_signal, stack, tls
        );

        let exit_signal = if exit_signal > 0 {
            Some(Signo::from_repr(exit_signal as u8).ok_or(AxError::InvalidInput)?)
        } else {
            None
        };

        // 对于每一个设置了 CLONE_VFORK 的 clone 调用，Linux 都会阻塞父进程，
        // 直到子进程执行 exec 或退出，无论调用者是否传递了子进程栈。
        // BusyBox 中 shell/timeout 等路径依赖这一顺序保证，
        // 即同时使用 CLONE_VM、CLONE_VFORK 和私有的子进程栈。
        let needs_vfork_block = flags.contains(CloneFlags::VFORK);

        let mut new_uctx = *uctx;
        //设置spsr_el0，开启irq
        new_uctx.prepare_clone_child_return_state();

        //sp时栈顶
        if stack != 0 {
            new_uctx.set_sp(stack);
        }

        //设置TLS
        if flags.contains(CloneFlags::SETTLS) {
            new_uctx.set_tls(tls);
        }

        //设置返回值，子进程返回pid = 0
        new_uctx.set_retval(0);

        //child_tid是用户空间的内存地址
        let set_child_tid = if flags.contains(CloneFlags::CHILD_SETTID) {
            child_tid
        } else {
            0
        };

        let curr = current();
        let curr_thread = curr.as_thread();
        let old_proc_data = &curr_thread.proc_data;

        let mut new_task = new_user_task(&curr.name(), new_uctx, set_child_tid);
        let tid = new_task.id().as_u64() as Pid;
        if flags.contains(CloneFlags::PARENT_SETTID) && parent_tid != 0 {
            (parent_tid as *mut Pid).vm_write(tid).ok();
        }


        let new_proc_data = if flags.contains(CloneFlags::THREAD) {
            new_task
                .ctx_mut()
                .set_page_table_root(old_proc_data.aspace().lock().page_table_root());
            old_proc_data.clone()
        } else {
            let proc = if flags.contains(CloneFlags::PARENT) {
                old_proc_data.proc.parent().ok_or(AxError::InvalidInput)?
            } else {
                old_proc_data.proc.clone()
            }
            .fork(tid);

            let aspace = if flags.contains(CloneFlags::VM) {
                old_proc_data.aspace()
            } else {
                let aspace_arc = old_proc_data.aspace();
                let aspace = aspace_arc.lock().try_clone()?;
                copy_from_kernel(&mut aspace.lock())?;
                aspace
            };
            new_task
                .ctx_mut()
                .set_page_table_root(aspace.lock().page_table_root());

            let signal_actions = if flags.contains(CloneFlags::SIGHAND) {
                old_proc_data.signal.actions()
            } else if flags.contains(CloneFlags::CLEAR_SIGHAND) {
                Arc::new(SpinNoIrq::new(Default::default()))
            } else {
                Arc::new(SpinNoIrq::new(
                    old_proc_data.signal.actions().lock().clone(),
                ))
            };

            let proc_data = ProcessData::new(
                proc,
                ProcessImage::new(
                    old_proc_data.exe_path.read().clone(),
                    old_proc_data.cmdline.read().clone(),
                    old_proc_data.auxv.read().clone(),
                ),
                aspace,
                signal_actions,
                exit_signal,
                curr_thread.tid(),
                flags.contains(CloneFlags::VM),
            );
            proc_data.set_umask(old_proc_data.umask());
            proc_data.set_nice(old_proc_data.nice());
            proc_data.set_heap_top(old_proc_data.get_heap_top());
            proc_data.replace_personality(old_proc_data.personality());
            // 继承父进程的 dumpable 标志（PR_SET_DUMPABLE 状态）。
            // Linux 行为：fork/clone 创建的子进程会从父进程拷贝 mm->dumpable；
            // 如果没有这一步，执行 prctl(PR_SET_DUMPABLE, 0) 之后再 fork()，
            // 子进程的 dumpable 会重置为 SUID_DUMP_USER (1)，
            // 从而破坏该 prctl 旨在强制实施的安全语义。
            // 已在 Linux 主机上验证：父进程设为 0 后 fork，子进程的 PR_GET_DUMPABLE 返回 0。
            // 控制的是进程崩溃后是否允许生成 core dump（核心转储）文件。
            proc_data.set_dumpable(old_proc_data.dumpable());
            // 禁用透明大页
            proc_data.set_thp_disable(old_proc_data.thp_disable());

            {
                let mut scope = proc_data.scope.write();
                if flags.contains(CloneFlags::FILES) {
                    // Synchronize with close_all_fds: holding a read lock
                    // ensures close_all_fds either observes our strong_count
                    // increment or blocks on write lock until we release.
                    let _guard = FD_TABLE.read();
                    FD_TABLE.scope_mut(&mut scope).clone_from(&FD_TABLE);
                } else {
                    FD_TABLE
                        .scope_mut(&mut scope)
                        .write()
                        .clone_from(&FD_TABLE.read());
                }

                if flags.contains(CloneFlags::FS) {
                    FS_CONTEXT.scope_mut(&mut scope).clone_from(&FS_CONTEXT);
                } else {
                    let fs_context = FS_CONTEXT.lock().clone();
                    *FS_CONTEXT.scope_mut(&mut scope).lock() = fs_context;
                }
            }

            proc_data
        };

        new_proc_data.proc.add_thread(tid);

        // cred 是 credentials（凭证）的缩写，记录了**"这个线程是谁"以及"能做什么"**的完整身份信息
        let parent_cred = Some(curr_thread.cred());
        let thr = Thread::new(
            tid,
            new_proc_data.clone(),
            parent_cred,
            curr_thread.signal.blocked(),
        );
        if curr_thread.no_new_privs() {
            thr.set_no_new_privs();
        }
        thr.set_seccomp_state(curr_thread.seccomp_state());
        if flags.contains(CloneFlags::CHILD_CLEARTID) {
            thr.set_clear_child_tid(child_tid);
        }
        if flags.contains(CloneFlags::PIDFD) && pidfd != 0 {
            let pidfd_obj = if flags.contains(CloneFlags::THREAD) {
                PidFd::new_thread(&thr, tid)
            } else {
                PidFd::new_process(&new_proc_data)
            };
            let fd = pidfd_obj.add_to_fd_table(true)?;
            if let Err(err) = (pidfd as *mut i32).vm_write(fd) {
                let _ = close_file_like(fd);
                return Err(err.into());
            }
        }
        *new_task.task_ext_mut() = Some(AxTaskExt::from_impl(thr));

        // vfork(2) 和 clone(CLONE_VFORK) 必须让父进程休眠，直到子进程 exec 或退出。
        // 使用 PollSet 机制实现，这样父进程的等待可以被 task.interrupt() 中断唤醒。
        if needs_vfork_block {
            let poll = Arc::new(axpoll::PollSet::new());
            new_proc_data.set_vfork_done(poll);
        }

        //new task添加到运行队列
        let task = spawn_task(new_task);
        add_task_to_table(&task);

        // 在任何可能的 vfork-等待之前触发，这样即使父进程在下方阻塞，
        // 观察者也能看到 fork 事件的发生。
        // 阻塞父进程直到子进程执行 exec 或退出。
        if needs_vfork_block {
            new_proc_data.wait_vfork_done();
        }

        Ok(tid as _)
    }
}


pub fn sys_clone(
    uctx: &UserContext,
    flags: u32,
    stack: usize,
    parent_tid: usize,
    tls: usize,
    child_tid: usize,
) -> AxResult<isize> {
    const FLAG_MASK: u32 = 0xff;
    let clone_flags = CloneFlags::from_bits_truncate((flags & !FLAG_MASK) as u64);
    let exit_signal = (flags & FLAG_MASK) as u64;


    if clone_flags.contains(CloneFlags::PIDFD | CloneFlags::PARENT_SETTID) {
        return Err(AxError::InvalidInput);
    }

    let args = CloneArgs {
        flags: clone_flags,
        exit_signal,
        stack,
        tls,
        parent_tid,
        child_tid,
        // 在 sys_clone 中，当设置了 CLONE_PIDFD 时，parent_tid 字段被复用为 pidfd 的输出地址。
        pidfd: if clone_flags.contains(CloneFlags::PIDFD) {
            parent_tid
        } else {
            0
        },
    };

    args.do_clone(uctx)
}
