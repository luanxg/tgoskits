use alloc::{
    string::{String, ToString},
    sync::Arc,
};

use ax_fs_ng::vfs::FS_CONTEXT;
use ax_runtime::hal::cpu::uspace::UserContext;
use ax_sync::Mutex;
use ax_task::{AxTaskExt, spawn_task};
use starry_process::{Pid, Process};

use crate::{
    file::FD_TABLE,
    mm::{copy_from_kernel, load_user_app, new_user_aspace_empty},
    pseudofs::{self, dev::tty::N_TTY},
    task::{ProcessData, ProcessImage, Thread, add_task_to_table, new_user_task, spawn_alarm_task},
    tracepoint::tracepoint_init,
};

/// Initialize and run initproc.
pub fn init(args: &[String], envs: &[String]) {
    //可以注释掉，暂时不使用
    static_keys::global_init();
    //可以注释掉，暂时不使用
    tracepoint_init().expect("Failed to initialize tracepoints");

    //可以注释掉，暂时不使用
    crate::ebpf::init_ebpf();
    //可以注释掉，暂时不使用
    crate::perf::perf_event_init();
    //可以注释掉，暂时不使用
    crate::kmod::init_kmod();

    pseudofs::mount_all().expect("Failed to mount pseudofs");
    spawn_alarm_task();
    //可以注释掉，暂时不使用
    pseudofs::usbfs::start_event_pump();

    //当物理内存分配失败时，分配器优先淘汰page cache中的干净页面
    //暂时也可以不需要
    ax_alloc::register_page_reclaim_fn(ax_fs_ng::vfs::page_cache_reclaim);

    let loc = FS_CONTEXT
        .lock()
        .resolve(&args[0])
        .expect("Failed to resolve executable path");
    let path = loc
        .absolute_path()
        .expect("Failed to get executable absolute path");
    let name = loc.name().into_owned();

    let mut uspace = new_user_aspace_empty()
        .and_then(|mut it| {
            copy_from_kernel(&mut it)?;
            Ok(it)
        })
        .expect("Failed to create user address space");

    let (entry_vaddr, ustack_top, auxv) = load_user_app(&mut uspace, loc, &args[0], args, envs)
        .unwrap_or_else(|e| panic!("Failed to load user app: {}", e));

    let uctx = UserContext::new(entry_vaddr.into(), ustack_top, 0);
    let mut task = new_user_task(&name, uctx, 0);
    task.ctx_mut().set_page_table_root(uspace.page_table_root());

    // PID 1 必须真是 1：init 进程是进程树的根节点，用户态程序
    // （比如 systemd 的 `getpid() == 1` 系统管理器检查）依赖这一点。
    // 调度器的 task id 是一个内部计数器，到我们启动用户态 init 时，
    // 它早就超过 1 了（内核辅助线程已经占用了低编号），因此我们把
    // 用户可见的 pid/tid 固定为 1，同时保持调度器 id 不变。
    // `Thread::tid` 已经与调度器 id 解耦（参见该字段的文档），
    // 所以这里只需要让进程表中的 key 跟随 thread tid 而非
    // `task.id()` 即可。
    const INIT_PID: Pid = 1;
    let pid = INIT_PID;
    let proc = Process::new_init(pid);
    proc.add_thread(pid);

    N_TTY.bind_to(&proc).expect("Failed to bind ntty");

    let proc = ProcessData::new(
        proc,
        ProcessImage::new(path.to_string(), Arc::new(args.to_vec()), auxv),
        Arc::new(Mutex::new(uspace)),
        Arc::default(),
        None,
        pid,
        false,
    );

    {
        let mut scope = proc.scope.write();
        crate::file::add_stdio(&mut FD_TABLE.scope_mut(&mut scope).write())
            .expect("Failed to add stdio");
    }

    let thr = Thread::new(pid, proc, None, starry_signal::SignalSet::default());
    *task.task_ext_mut() = Some(AxTaskExt::from_impl(thr));

    let task = spawn_task(task);
    add_task_to_table(&task);

    // TODO: wait for all processes to finish
    let exit_code = task.join();
    info!("Init process exited with code: {exit_code:?}");

    let cx = FS_CONTEXT.lock();
    cx.root_dir()
        .unmount_all()
        .expect("Failed to unmount all filesystems");
    cx.root_dir()
        .filesystem()
        .flush()
        .expect("Failed to flush rootfs");
}
