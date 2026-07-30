use alloc::{
    format,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::{
    ffi::{c_char, c_int},
    future::poll_fn,
    iter,
    task::Poll,
};

use ax_errno::{AxError, AxResult};
use ax_runtime::hal::cpu::uspace::UserContext;
use ax_sync::Mutex;
use ax_task::{current, future::block_on, yield_now};
use axfs_ng_vfs::Location;
use kernel_elf_parser::AuxType;
use linux_raw_sys::general::{AT_EMPTY_PATH, AT_SYMLINK_NOFOLLOW};
use starry_process::Pid;
use starry_vm::vm_load_until_nul;

use crate::{
    config::USER_HEAP_BASE,
    file::{ResolveAtResult, current_fd_table, memfd::Memfd, resolve_at},
    mm::{copy_from_kernel, load_user_app, new_user_aspace_empty, vm_load_string},
    task::{AsThread, rebind_task_tid, zap_thread},
};

pub fn sys_execve(
    uctx: &mut UserContext,
    path: *const c_char,
    argv: *const *const c_char,
    envp: *const *const c_char,
) -> AxResult<isize> {
    let path = vm_load_string(path)?;
    let loc = current().as_thread().proc_data.fs_context.lock().resolve(&path)?;
    do_execve(uctx, loc, path, argv, envp)
}

/// execveat(2) — like execve, but the program is identified by `dirfd` plus
/// `path` (resolved relative to `dirfd`), or by `dirfd` alone when
/// `AT_EMPTY_PATH` is set and `path` is empty.
pub fn sys_execveat(
    uctx: &mut UserContext,
    dirfd: c_int,
    path: *const c_char,
    argv: *const *const c_char,
    envp: *const *const c_char,
    flags: u32,
) -> AxResult<isize> {
    if flags & !(AT_EMPTY_PATH | AT_SYMLINK_NOFOLLOW) != 0 {
        return Err(AxError::InvalidInput);
    }

    let path = vm_load_string(path)?;

    // Resolve dirfd + path to the `Location` the loader reads from. A regular
    // file yields its filesystem path as the display name; an anonymous memfd
    // has no path but wraps a tmpfs-backed `Location` we can still load — this
    // is systemd's `execveat(memfd, "", AT_EMPTY_PATH)` path. Other anonymous
    // fds (sockets, eventfd, …) are not executable.
    let (loc, disp_path) = match resolve_at(dirfd, Some(path.as_str()), flags)? {
        ResolveAtResult::File(loc) => {
            let disp = loc.absolute_path().map(|p| p.to_string()).unwrap_or(path);
            (loc, disp)
        }
        ResolveAtResult::Other(f) => {
            let memfd = f.downcast_ref::<Memfd>().ok_or_else(|| {
                warn!("sys_execveat: exec from non-memfd anonymous fd is not supported");
                AxError::PermissionDenied
            })?;
            let loc = memfd.inner().inner().location().clone();
            let disp = format!("/memfd:{} (deleted)", memfd.name());
            (loc, disp)
        }
    };

    do_execve(uctx, loc, disp_path, argv, envp)
}

/// execve 共享核心（相当于 Linux 的 `do_execveat_common`）：`sys_execve` 和
/// `sys_execveat` 都将程序解析为 `Location`，然后将它连同原始的用户态 `argv` /
/// `envp` 指针一并传入此处加载，仅加载一次。`path` 是显示名称（用于独立于 argv0
/// 的 `comm`/`exe_path` 以及加载器的 `.sh`/shebang 处理），不会在文件系统中对其
/// 进行二次解析。
fn do_execve(
    uctx: &mut UserContext,
    loc: Location,
    path: String,
    argv: *const *const c_char,
    envp: *const *const c_char,
) -> AxResult<isize> {
    // ----------------------------------------------------------------
    // 阶段一：所有可能失败的工作 —— 尚未提交任何更改。
    // 此阶段任何一步失败都会返回错误，进程保持完整不受影响。
    // ----------------------------------------------------------------

    // NULL 向量指针被视为空列表：glibc 的 `execl(path, NULL)` 传 NULL
    // 表示"无参数"，Linux 的 `count_strings_kernel` 对 NULL 短路返回空
    // 列表，而非返回 EFAULT。
    let load_vec = |ptr: *const *const c_char| -> AxResult<Vec<String>> {
        if ptr.is_null() {
            Ok(Vec::new())
        } else {
            vm_load_until_nul(ptr)?
                .into_iter()
                .map(vm_load_string)
                .collect::<Result<Vec<_>, _>>()
        }
    };
    //args = Ok(vec!["/bin/sh", "-c", "echo hello"])。
    let mut args = load_vec(argv)?;
    let envs = load_vec(envp)?;

    // Linux 仍会为新映像提供空字符串作为 argv[0]，因此这里将空的 argv 统
    // 一处理为 [""]。
    if args.is_empty() {
        args.push(String::new());
    }

    debug!("do_execve <= path: {path:?}, args: {args:?}, envs: {envs:?}");

    let curr = current();
    let thr = curr.as_thread();
    let proc_data = &thr.proc_data;
    // 此时tid是进程自身的tid，和父进程无关
    let my_tid = thr.tid();
    //读取线程组ID
    let tgid = proc_data.proc.pid();

    // 对来自兄弟线程的并发 execve 进行串行化处理。
    //
    // 如果仅使用 `try_lock`，即使持有锁的线程仍处于*可能失败*的阶段（路径解析 /
    // ELF 加载），竞争失败的线程也会以 EINTR 退出：若持有者随后出错并释放锁，
    // 失败者就错误地放弃了本可以基于自身映像成功执行的 execve。因此我们改为等待
    // 锁，仅在持有者进入不可逆的拆除阶段后才退出 —— 我们通过 `zap_thread` 设置
    // 的 `exit_request` 来观察这一状态。
    //
    // 不能直接使用 `ax_sync::Mutex::lock`：它在 `WaitQueue::wait_until` 上睡眠，
    // 而该等待队列不会被 zap 的 `task.interrupt()` 唤醒；更糟糕的是，锁释放时
    // 失败者会获取互斥锁并基于持有者已提交的新映像继续 execve。使用带
    // `exit_request` 探测的忙等自旋（busy-yield）可以达成：
    //   - 若持有者在提交前失败，则穿透获取锁；
    //   - 若持有者在兄弟线程拆除循环中将我们 zap，则协作退出（EINTR → 返回用户态 → `do_exit(0, false)`）；
    // 且不会消耗返回用户态时 `check_signals` 所需的任何标志位。
    //
    // 注意：我们故意*不*在通用 `task.interrupt()`（信号唤醒）时中止。Linux 的
    // execve 虽然可被杀死，但在通过 `cred_guard_mutex` 串行化期间并不允许被任意
    // 信号中断。
    let _exec_guard = loop {
        if let Some(g) = proc_data.exec_lock.try_lock() {
            break g;
        }
        //检查是否被标记了exit_request
        if thr.has_exit_request() {
            return Err(AxError::Interrupted);
        }
        yield_now();
    };

    // 在触碰任何内容之前，先从已解析的 Location 收集元数据。匿名的 memfd 没有
    // 文件系统路径，因此回退到调用者传入的显示名称（例如 `/memfd:<name> (deleted)`）。
    //new_name = 文件名
    let mut new_name = loc.name().to_string();
    //new_exe_path = "/bin/ls"
    let mut new_exe_path = loc
        .absolute_path()
        .map(|p| p.to_string())
        .unwrap_or_else(|_| path.clone());

    // 在提交之前完整构建新的地址空间。向全新的地址空间加载（而非清空现有地址空
    // 间）可以确保 CLONE_VM 父进程的映射不受任何干扰 —— posix_spawn 使用
    // CLONE_VM|CLONE_VFORK，在父进程地址空间内的栈片段上运行子进程。完全加载好
    // 的地址空间也充当了 bprm 的等价物：可执行文件内容此时已被固定，因此拆除后
    // 的提交阶段无需重新解析路径（在回收兄弟线程期间文件系统可能已经发生变化）。
    // 创建新的用户空间
    let mut new_aspace = new_user_aspace_empty()?;
    copy_from_kernel(&mut new_aspace)?;
    let (entry_point, user_stack_base, auxv) =
        match load_user_app(&mut new_aspace, loc, &path, &args, &envs, &curr.as_thread().proc_data.fs_context) {
            Ok(result) => result,
            //此处加载elf失败，认为可能是xxx.sh脚本文件，将镜像替换为/bin/sh，使用/bin/sh xxx.sh arg1的方式重新加载
            Err(AxError::InvalidExecutable) => {
                // ENOEXEC 回退：通过 /bin/sh 重试。
                // 在 Linux 中此重试由用户态（execvp / busybox）而非内核完成。
                // 这是一个实用的临时方案，待 musl 的 execvp 或 busybox 的
                // ENOEXEC 处理可用后再移除。
                let shell_path = "/bin/sh";
                let shell_loc = curr.as_thread().proc_data.fs_context.lock().resolve(shell_path)?;
                new_name = shell_loc.name().to_string();
                new_exe_path = shell_loc.absolute_path()?.to_string();
                args = iter::once(String::from(shell_path))
                    .chain(args.iter().cloned())
                    .collect();
                load_user_app(&mut new_aspace, shell_loc, shell_path, &args, &envs, &curr.as_thread().proc_data.fs_context)?
            }
            Err(e) => return Err(e),
        };

    // ----------------------------------------------------------------
    // 兄弟线程拆除（仅多线程场景）。
    // zap 每个兄弟线程，使其执行仅线程级别的 `do_exit(0, false)` —— 而非
    // 进程级别的致命 SIGKILL —— 然后等待线程组中仅剩调用者自身，再提交。
    //
    // 等待过程*不可*中断：一旦兄弟线程被 zap，拆除便不可逆，此处若返回 EINTR
    // 会导致进程线程已被部分拆除、却仍运行在旧地址空间上。任何针对调用者的致
    // 命信号将在提交阶段之后通过返回用户态的路径投递。
    //
    // 每次迭代都重新获取快照：兄弟线程可能在我们广播 zap 到其自身退出之间又
    // 生成了新的线程，而该新线程的 tid 在上次遍历时还不可见。
    // ----------------------------------------------------------------
    loop {
        let siblings: Vec<Pid> = proc_data
            .proc
            .threads()
            .into_iter()
            .filter(|tid| *tid != my_tid)
            .collect();
        if siblings.is_empty() {
            break;
        }

        debug!(
            "sys_execve: zapping {} sibling thread(s) before exec",
            siblings.len()
        );
        for tid in &siblings {
            // 尽力而为：目标线程可能已经被回收。
            let _ = zap_thread(*tid);
        }

        block_on(poll_fn(|cx| {
            let remaining = proc_data
                .proc
                .threads()
                .into_iter()
                .filter(|tid| *tid != my_tid)
                .count();
            if remaining == 0 {
                return Poll::Ready(());
            }
            unsafe {
                proc_data
                    .thread_exit_event
                    .register(cx.waker(), axpoll::IoEvents::IN)
            };
            // 注册后重新检查：兄弟线程可能在第一次检查和注册之间已经退出，
            // 那时触发的唤醒会因 waker 集合为空而丢失。
            let remaining = proc_data
                .proc
                .threads()
                .into_iter()
                .filter(|tid| *tid != my_tid)
                .count();
            if remaining == 0 {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        }));
    }

    // 收集需要在兄弟线程拆除*之后*关闭的 CLOEXEC fd。若在拆除之前获取快照，会
    // 遗漏兄弟线程在我们获取快照到其退出之间提升为 CLOEXEC 的 fd（通过
    // `open(... O_CLOEXEC)`、`fcntl(F_SETFD)` 或 `close_range(..., CLOEXEC)`），
    // 导致这些 fd 泄漏到新映像中。当所有兄弟线程被回收后，快照反映的是最终静默
    // 后的 fd 表。下面的关闭过程在同一 `fd_table.write()` 锁保护下执行，因此扫
    // 描和关闭之间不会有新的 fd 出现。
    let fd_table_outer = current_fd_table();
    let mut fd_table = fd_table_outer.write();
    let cloexec_fds: Vec<_> = fd_table
        .ids()
        .filter(|it| fd_table.get(*it).unwrap().cloexec)
        .collect();

    // ----------------------------------------------------------------
    // 阶段二：不可逆点 —— 提交所有更改。
    // 以下任何操作都不允许失败；此处出错将导致进程处于损坏状态。
    // ----------------------------------------------------------------

    // 替换地址空间的 Arc，使父进程共享的 Arc<Mutex<AddrSpace>>（来自 CLONE_VM）
    // 不受影响。父进程的页表寄存器继续指向原有且仍在运行的地址空间。
    let new_pt_root = new_aspace.page_table_root();
    let newaspace_arc = Arc::new(Mutex::new(new_aspace));
    proc_data.replace_aspace(newaspace_arc);
    proc_data.mark_vm_aspace_private_after_exec();

    // 新的地址空间已安装，切换硬件页表。
    curr.switch_page_table(new_pt_root);

    curr.set_name(&new_name);
    *proc_data.exe_path.write() = new_exe_path;
    *proc_data.cmdline.write() = Arc::new(args);
    let auxv_len = auxv.len();
    let has_ldso = auxv.iter().any(|e| e.get_type() == AuxType::BASE);
    *proc_data.auxv.write() = auxv;

    proc_data.set_heap_top(USER_HEAP_BASE);

    // 按照 POSIX/Linux 语义为新映像重置信号状态
    // （参见 Linux 的 `flush_signal_handlers` + `do_execveat_common`）：
    //
    //   - 自定义用户处理函数恢复为 SIG_DFL，标志位和屏蔽字均被清除。
    //   - 显式设置的 `SIG_IGN` 在 exec 后保留（POSIX 要求）；即使信号的默认
    //     动作为 Ignore，处置方式也保持为 `SIG_DFL`。
    //   - 进程级和线程级的未决信号*被保留*：POSIX 要求已入队的信号（包括被阻
    //     塞的信号）在 `execve` 后继续存在，并对新映像的处理函数进行投递。信号
    //     阻塞屏蔽字本身也被保留。
    //   - 通过 `sigaltstack` 注册的备用信号栈被重置，因为其 `ss_sp` 指向旧地
    //     址空间，而该地址空间已不再映射。
    proc_data.signal.reset_actions_for_exec();
    thr.signal.reset_stack();
    proc_data.posix_timers.clear();

    // 线程中缓存的、引用了旧地址空间中用户内存的指针现已悬空。清除它们，确保
    // 后续系统调用和线程退出路径不会解引用已释放的用户页。
    thr.set_clear_child_tid(0);
    thr.set_robust_list_head(0);
    thr.clear_rseq_state();

    // 在拆除后快照的写锁保护下从表中移除 CLOEXEC fd —— 从扫描到关闭之间不
    // 会有新的 fd 被添加或翻转 CLOEXEC 位 —— 但将实际的 `release_locks_on_close`
    // （POSIX 锁释放、OFD waker 唤醒、FileDescriptor 析构）推迟到释放表写锁
    // 之后执行。waker 触发在全局 advisory-lock 等待队列上，可能立即将唤醒的
    // 任务通过 `FD_TABLE` 再次调度回来；在持有写锁时运行它们不仅有锁重入的风
    // 险，还会将临界区扩展到任意析构逻辑中。Linux 的 `do_close_on_exec` 在每次
    // `filp_close` 调用前后分别释放和重新获取 `files->file_lock`，正是出于同
    // 样的原因。我们在释放锁之后批量关闭所有 fd，效果等价：刚清空的槽位中不可
    // 能出现新的 fd，因为此时进程中尚无其他任何代码在运行（兄弟线程已回收，新
    // 映像尚未启动）。
    let mut closing = Vec::with_capacity(cloexec_fds.len());
    for fd in cloexec_fds {
        if let Some(f) = fd_table.remove(fd) {
            closing.push(f);
        }
    }
    drop(fd_table);
    for f in closing {
        crate::file::release_locks_on_close(f);
    }

    // de_thread 领导者转移（仅限非领导者调用者）。
    //
    // 经过上述兄弟线程拆除循环后，线程组中仅剩 `curr` 这一个任务。如果 `curr`
    // 不是原始领导者，Linux 的 `de_thread()` 会通过 `exchange_tids` /
    // `transfer_pid` 将领导者的 TID/TGID 身份转移给调用线程，使得新映像中
    // `gettid() == getpid()` 成立，并且父进程对（仍是原来的）PID 的已有句柄
    // 继续指向该线程，用于 `wait`、`kill`、`tgkill`、`/proc/<pid>` 等操作。
    //
    // 我们在此处通过以下操作来对应这一行为：
    //   - 将 `Thread::tid` 从旧的非领导者值重命名为领导者的 TGID，
    //   - 重新索引全局 TASK_TABLE 条目，
    //   - 重新索引进程级的信号子进程列表，
    //   - 替换 `proc.tg.threads` 中对应的条目。
    //
    // 原始领导者已在上面的逻辑中被 zap（从 `curr` 的视角来看它是一个兄弟线
    // 程），已执行 `do_exit(0, false)`，不再存在于任务表或线程组中，因此目
    // 标 TID 是空闲的。
    if my_tid != tgid {
        thr.set_tid(tgid);
        rebind_task_tid(&curr, my_tid, tgid);
        proc_data.signal.rename_child(my_tid, tgid);
        proc_data.proc.rename_thread(my_tid, tgid);
    }

    // 将所有用户可见的寄存器重置为新进程状态，而不仅仅是 IP/SP。Linux 的
    // `start_thread()` 会清除所有通用寄存器、重置 TLS 指针，并将所有 FP/SIMD
    // 状态置为 ABI 默认值；如果保留部分填充的系统调用陷阱帧，新映像可能会观测
    // 到残留的 argv/envp 指针、exec 前映像设置的过期 TLS 基址等。构建一个新的
    // `UserContext` 与 `entry::run_user_app` 对 init 进程的处理一致 —— 新映像
    // 合法继承的唯一状态是地址空间以及我们在上面显式保留的内核/调度器相关数据。
    *uctx = UserContext::new(entry_point.as_usize(), user_stack_base, 0);

    debug!(
        "execve: path={} entry={:#x} sp={:#x} tp={} auxv_count={} auxv_has_ldso={}",
        new_name,
        entry_point.as_usize(),
        user_stack_base,
        uctx.tls(),
        auxv_len,
        has_ldso,
    );

    // 唤醒正在等待此子进程 exec 的 vfork 父进程。
    // 必须放在最后：此时 CLOEXEC fd 已经关闭，父进程的管道读取将正确收到 EOF。
    proc_data.notify_vfork_done();

    Ok(0)
}
