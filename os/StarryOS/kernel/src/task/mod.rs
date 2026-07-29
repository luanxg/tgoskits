//! User task management.

mod cred;
pub mod futex;
mod ops;
pub mod posix_timer;
mod resources;
mod seccomp;
mod signal;
mod stat;
mod timer;
mod user;

use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};
use core::{
    cell::RefCell,
    ops::Deref,
    sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicU32, AtomicUsize, Ordering},
};

use ax_errno::AxResult;
use ax_fs_ng::vfs::{FsContext, ROOT_FS_CONTEXT};
use ax_runtime::hal::time::TimeValue;
use ax_sync::{Mutex, spin::SpinNoIrq};
use ax_task::{TaskExt, TaskInner};
use axpoll::{IoEvents, PollSet};
use extern_trait::extern_trait;
use flatten_objects::FlattenObjects;
use kernel_elf_parser::AuxEntry;
use scope_local::{ActiveScope, Scope};
use spin::RwLock;
use starry_process::{Pid, Process};
use starry_signal::{
    SignalSet, Signo,
    api::{ProcessSignalManager, SignalActions, ThreadSignalManager},
};

pub use self::{
    cred::*, futex::*, ops::*, posix_timer::PosixTimerTable, resources::*, seccomp::*, signal::*,
    stat::*, timer::*, user::*,
};

use crate::{
    file::FileDescriptor,
    mm::AddrSpace,
};

/// Size of the syscall instruction for the current architecture.
/// Used by SA_RESTART to back up the program counter.
#[cfg(target_arch = "x86_64")]
pub const SYSCALL_INSN_LEN: usize = 2;
/// Size of the syscall instruction for the current architecture.
/// Used by SA_RESTART to back up the program counter.
#[cfg(not(target_arch = "x86_64"))]
pub const SYSCALL_INSN_LEN: usize = 4;

///  A wrapper type that assumes the inner type is `Sync`.
#[repr(transparent)]
pub struct AssumeSync<T>(pub T);

unsafe impl<T> Sync for AssumeSync<T> {}

impl<T> Deref for AssumeSync<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A one-shot flag that suppresses exactly one signal check.
struct NextSignalCheckBlock(AtomicBool);

impl NextSignalCheckBlock {
    const fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    fn block(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn unblock(&self) -> bool {
        self.0.swap(false, Ordering::AcqRel)
    }
}

/// 线程的内部数据。
pub struct Thread {
    /// 用户可见的线程 ID（即 `gettid` 返回的 `Pid`）。
    ///
    /// 初始值与底层调度器的 `TaskInner::id()` 相同。在非 leader 线程成功
    /// 执行 `execve` 之后，二者会产生分歧：Linux 的 `de_thread` 步骤会将
    /// leader 的 TID/TGID 转移给调用线程，从而在新镜像中满足
    /// `gettid() == getpid()`。我们通过更新此字段来建模这一行为，同时
    /// 保持不可变的调度器 ID 不变。所有面向用户的 TID 查询（`sys_gettid`、
    /// `set_tid_address`、信号子进程注册、`do_exit` 的线程组簿记等）都
    /// 读取此字段而非调度器 ID。
    tid: AtomicU32,

    /// 进程中所有线程共享的进程数据。
    pub proc_data: Arc<ProcessData>,

    /// clear_child_tid 字段
    ///
    /// 参见 <https://manpages.debian.org/unstable/manpages-dev/set_tid_address.2.en.html#clear_child_tid>
    ///
    /// 当线程退出时，如果此地址不为 NULL，内核会将该地址处的字清零。
    clear_child_tid: AtomicUsize,

    /// robust 链表头
    robust_list_head: AtomicUsize,

    /// 线程级信号管理器
    pub signal: Arc<ThreadSignalManager>,

    /// 时间管理器
    ///
    /// 假定其为 `Sync`，因为它仅在上下文切换期间被可变借用，而上下文切换
    /// 是当前线程独占的。
    pub time: AssumeSync<RefCell<TimeManager>>,

    /// OOM 分数调整值。
    oom_score_adj: AtomicI32,

    /// 准备退出
    pub exit: Arc<AtomicBool>,

    /// 表示线程当前是否正在访问用户内存。
    accessing_user_memory: AtomicBool,

    /// 从用户空间信号处理函数返回后，跳过一次信号检查。
    block_next_signal_check: NextSignalCheckBlock,

    /// 自身退出事件
    pub exit_event: Arc<PollSet>,

    /// 由 `sys_execve` 在回收兄弟线程时设置。信号检查路径会将其转换为
    /// 仅针对当前线程的 `do_exit(0, false)`——不触发组退出，不产生致命
    /// 信号级联——从而保持新镜像完整无损。
    exit_request: AtomicBool,

    /// 为可重启序列（`rseq(2)`）注册的 rseq 区域指针（用户地址）。
    rseq_area: AtomicUsize,

    /// 注册时记录的 rseq 签名。
    rseq_signature: AtomicU32,

    /// 当父进程终止时，向该线程发送的信号（PR_SET_PDEATHSIG）。
    pdeathsig: AtomicU32,

    /// PR_SET_NO_NEW_PRIVS：一旦设置，不可撤销。
    no_new_privs: AtomicBool,

    /// seccomp 系统调用过滤状态。
    seccomp: SpinNoIrq<SeccompState>,

    /// 进程凭证（uid、gid 等）。
    cred: SpinNoIrq<Arc<Cred>>,

    /// [`raise_signal_fatal`] 最近强制投递给该线程的同步用户态故障的
    /// 信号编号（以 u8 表示），若为 0 则表示"没有待输出的故障转储"。
    /// [`check_signals`] 仅在即将终止当前线程的信号与此 signo 匹配时，
    /// 才会输出寄存器转储——否则，一个低编号的挂起信号（例如在缺页异常
    /// 产生 SIGSEGV 之前到达的外部 SIGTERM）可能会消费掉该标志，导致
    /// 要么在错误的上下文中输出转储，要么（如果该信号有用户态处理函数）
    /// 吞噬掉转储，使得真正的故障静默终止。
    pub fault_dump_signo: AtomicU8,

    /// 是否已为该线程的用户命名空间写入了 uid_map。
    uid_map_written: AtomicBool,

    /// 是否已为该线程的用户命名空间写入了 gid_map。
    gid_map_written: AtomicBool,

    /// 是否已将该线程的用户命名空间的 setgroups 设置为 "deny"。
    setgroups_deny: AtomicBool,
}

impl Thread {
    /// 创建一个新的 [`Thread`]。
    ///
    /// 若 `parent_cred` 为 `Some`，线程将继承父进程的凭证；
    /// 否则以 root 凭证启动（用于 init 进程）。
    pub fn new(
        tid: u32,
        proc_data: Arc<ProcessData>,
        parent_cred: Option<Arc<Cred>>,
        signal_mask: SignalSet,
    ) -> Box<Self> {
        let cred = parent_cred.unwrap_or_else(|| Arc::new(Cred::root()));
        Box::new(Thread {
            tid: AtomicU32::new(tid),
            signal: ThreadSignalManager::new_with_blocked(
                tid,
                proc_data.signal.clone(),
                signal_mask,
            ),
            proc_data,
            clear_child_tid: AtomicUsize::new(0),
            robust_list_head: AtomicUsize::new(0),
            time: AssumeSync(RefCell::new(TimeManager::new())),
            exit: Arc::new(AtomicBool::new(false)),
            oom_score_adj: AtomicI32::new(200),
            accessing_user_memory: AtomicBool::new(false),
            block_next_signal_check: NextSignalCheckBlock::new(),
            exit_event: Arc::default(),
            exit_request: AtomicBool::new(false),
            rseq_area: AtomicUsize::new(0),
            rseq_signature: AtomicU32::new(0),
            pdeathsig: AtomicU32::new(0),
            no_new_privs: AtomicBool::new(false),
            seccomp: SpinNoIrq::new(SeccompState::default()),
            cred: SpinNoIrq::new(cred),

            fault_dump_signo: AtomicU8::new(0),
            uid_map_written: AtomicBool::new(false),
            gid_map_written: AtomicBool::new(false),
            setgroups_deny: AtomicBool::new(false),
        })
    }

    /// Returns the user-visible TID for this thread.
    ///
    /// See the field doc on [`Thread::tid`] for why this can differ from
    /// the underlying scheduler `TaskInner::id()`.
    pub fn tid(&self) -> u32 {
        self.tid.load(Ordering::Acquire)
    }

    /// Updates the user-visible TID. Called only by `execve`'s de_thread
    /// step to transfer the leader's TID to a non-leader caller.
    pub(crate) fn set_tid(&self, tid: u32) {
        self.tid.store(tid, Ordering::Release);
    }

    /// Get the clear child tid field.
    pub fn clear_child_tid(&self) -> usize {
        self.clear_child_tid.load(Ordering::Relaxed)
    }

    /// Set the clear child tid field.
    pub fn set_clear_child_tid(&self, clear_child_tid: usize) {
        self.clear_child_tid
            .store(clear_child_tid, Ordering::Relaxed);
    }

    /// Get the robust list head.
    pub fn robust_list_head(&self) -> usize {
        self.robust_list_head.load(Ordering::SeqCst)
    }

    /// Set the robust list head.
    pub fn set_robust_list_head(&self, robust_list_head: usize) {
        self.robust_list_head
            .store(robust_list_head, Ordering::SeqCst);
    }

    /// Get the oom score adjustment value.
    pub fn oom_score_adj(&self) -> i32 {
        self.oom_score_adj.load(Ordering::SeqCst)
    }

    /// Set the oom score adjustment value.
    pub fn set_oom_score_adj(&self, value: i32) {
        self.oom_score_adj.store(value, Ordering::SeqCst);
    }

    /// Check if the thread is ready to exit.
    pub fn pending_exit(&self) -> bool {
        self.exit.load(Ordering::Acquire)
    }

    /// Set the thread to exit.
    pub fn set_exit(&self) {
        self.exit.store(true, Ordering::Release);
    }

    /// Consume a pending thread-only exit request, returning whether one
    /// was set. The flag is cleared in the same atomic step so that a
    /// re-entrant `check_signals` (the user loop drains signals in a
    /// while-loop) doesn't fire `do_exit` twice for the same zap.
    pub fn take_exit_request(&self) -> bool {
        self.exit_request.swap(false, Ordering::AcqRel)
    }

    /// Non-consuming probe for a pending thread-only exit request. Used
    /// by in-kernel wait loops that want to abort cooperatively without
    /// stealing the flag from the user-return `check_signals` path.
    pub fn has_exit_request(&self) -> bool {
        self.exit_request.load(Ordering::Acquire)
    }

    /// Request a thread-only exit. Honored by `check_signals` on the next
    /// return to user space, where it routes to `do_exit(0, false)`.
    pub fn set_exit_request(&self) {
        self.exit_request.store(true, Ordering::Release);
    }

    /// Check if the thread is accessing user memory.
    pub fn is_accessing_user_memory(&self) -> bool {
        self.accessing_user_memory.load(Ordering::Acquire)
    }

    /// Set the accessing user memory flag.
    pub fn set_accessing_user_memory(&self, accessing: bool) {
        self.accessing_user_memory
            .store(accessing, Ordering::Release);
    }

    /// Get the pdeathsig value (signal sent to this thread when parent exits).
    pub fn pdeathsig(&self) -> u32 {
        self.pdeathsig.load(Ordering::Relaxed)
    }

    /// Set the pdeathsig value.
    pub fn set_pdeathsig(&self, sig: u32) {
        self.pdeathsig.store(sig, Ordering::Relaxed);
    }

    /// Get the no_new_privs flag.
    pub fn no_new_privs(&self) -> bool {
        self.no_new_privs.load(Ordering::Relaxed)
    }

    /// Set the no_new_privs flag (one-way: once set, cannot be unset).
    pub fn set_no_new_privs(&self) {
        self.no_new_privs.store(true, Ordering::Relaxed);
    }

    /// Get a snapshot of the current seccomp state.
    pub fn seccomp_state(&self) -> SeccompState {
        self.seccomp.lock().clone()
    }

    /// Replace seccomp state. Used by clone inheritance.
    pub fn set_seccomp_state(&self, state: SeccompState) {
        *self.seccomp.lock() = state;
    }

    /// Enable strict seccomp mode.
    pub fn install_seccomp_strict(&self) -> AxResult<()> {
        self.seccomp.lock().install_strict()
    }

    /// Append a seccomp filter. Filters are inherited and evaluated in order.
    pub fn append_seccomp_filter(&self, insns: Vec<SockFilter>) -> AxResult<()> {
        self.seccomp.lock().append_filter(insns)
    }

    /// Get a snapshot of the current credentials (clones the `Arc`).
    pub fn cred(&self) -> Arc<Cred> {
        self.cred.lock().clone()
    }

    /// Replace the credentials with `new_cred` for this thread only.
    /// Prefer `set_cred` for credential-changing syscalls.
    fn set_cred_single(&self, new_cred: Arc<Cred>) {
        *self.cred.lock() = new_cred;
    }

    /// Replace the credentials for ALL threads in the same process.
    ///
    /// POSIX requires that credential changes (setuid, setresuid, etc.)
    /// affect all threads in a process. On Linux, the kernel stores
    /// credentials per-thread and the C library synchronizes via signals.
    /// musl's setxid synchronization does NOT work on StarryOS, so we
    /// implement this at the kernel level instead.
    ///
    /// Lock ordering: threads are updated in ascending TID order to
    /// prevent AB/BA deadlock when two threads call set_cred
    /// concurrently on SMP.
    pub fn set_cred(&self, new_cred: Cred) {
        let new_arc = Arc::new(new_cred);

        // Always update the caller first.  The process thread list is a
        // best-effort snapshot, and credential-changing syscalls must not
        // return before the calling thread observes its own new credentials.
        self.set_cred_single(new_arc.clone());

        // Collect TIDs and sort to establish a consistent lock order.
        let mut tids = self.proc_data.proc.threads();
        tids.sort_unstable();

        for tid in &tids {
            if let Ok(task) = ops::get_task(*tid)
                && let Some(thr) = task.try_as_thread()
            {
                thr.set_cred_single(new_arc.clone());
            }
        }
    }

    /// Get the registered rseq area pointer.
    pub fn rseq_area(&self) -> usize {
        self.rseq_area.load(Ordering::SeqCst)
    }

    /// Get the registered rseq signature.
    pub fn rseq_signature(&self) -> u32 {
        self.rseq_signature.load(Ordering::SeqCst)
    }

    /// Check if uid_map has been written for this thread's user namespace.
    pub fn uid_map_written(&self) -> bool {
        self.uid_map_written.load(Ordering::Relaxed)
    }

    /// Mark uid_map as written.
    pub fn set_uid_map_written(&self, val: bool) {
        self.uid_map_written.store(val, Ordering::Relaxed);
    }

    /// Check if gid_map has been written for this thread's user namespace.
    pub fn gid_map_written(&self) -> bool {
        self.gid_map_written.load(Ordering::Relaxed)
    }

    /// Mark gid_map as written.
    pub fn set_gid_map_written(&self, val: bool) {
        self.gid_map_written.store(val, Ordering::Relaxed);
    }

    /// Check if setgroups has been set to "deny".
    pub fn setgroups_deny(&self) -> bool {
        self.setgroups_deny.load(Ordering::Relaxed)
    }

    /// Set the setgroups deny flag.
    pub fn set_setgroups_deny(&self, val: bool) {
        self.setgroups_deny.store(val, Ordering::Relaxed);
    }

    /// Set the registered rseq area pointer.
    pub fn set_rseq_area(&self, addr: usize) {
        self.rseq_area.store(addr, Ordering::SeqCst);
    }

    /// Set the registered rseq area pointer and signature.
    pub fn set_rseq_state(&self, addr: usize, sig: u32) {
        self.rseq_area.store(addr, Ordering::SeqCst);
        self.rseq_signature.store(sig, Ordering::SeqCst);
    }

    /// Clear the registered rseq state.
    pub fn clear_rseq_state(&self) {
        self.rseq_area.store(0, Ordering::SeqCst);
        self.rseq_signature.store(0, Ordering::SeqCst);
    }

    /// Block the next signal check for this thread.
    pub fn block_next_signal_check(&self) {
        self.block_next_signal_check.block();
    }

    /// Consume and clear the one-shot signal-check block flag.
    pub fn unblock_next_signal_check(&self) -> bool {
        self.block_next_signal_check.unblock()
    }
}

#[extern_trait]
impl TaskExt for Box<Thread> {
    fn on_enter(&self) {
        let scope = self.proc_data.scope.read();
        unsafe { ActiveScope::set(&scope) };
        core::mem::forget(scope);
    }

    fn on_leave(&self) {
        ActiveScope::set_global();
        unsafe { self.proc_data.scope.force_read_decrement() };
    }
}

/// Helper trait to access the thread from a task.
pub trait AsThread {
    /// Try to get the thread from the task.
    fn try_as_thread(&self) -> Option<&Thread>;

    /// Get the thread from the task, panicking if it is a kernel task.
    #[track_caller]
    fn as_thread(&self) -> &Thread {
        self.try_as_thread().expect("kernel task")
    }
}

impl AsThread for TaskInner {
    fn try_as_thread(&self) -> Option<&Thread> {
        self.task_ext()
            .map(|ext| ext.downcast_ref::<Box<Thread>>().as_ref())
    }
}

/// A one-shot completion for vfork synchronization.
///
/// This avoids lost-wakeup races by recording the "done" state under the same
/// lock as the waker set. If the child completes before the parent enters the
/// wait, the parent will see `done == true` and skip waiting.
///
/// We use [`PollSet`] (not `WaitQueue`) so the parent's wait can run inside
/// `block_on(interruptible(...))`: a sibling thread that does `execve` will
/// zap us via `task.interrupt()`, which only wakes futures-based polls, not
/// `WaitQueue::wait_until`. Without this, the execve initiator would deadlock
/// in its sibling-teardown loop waiting for us to exit.
pub struct VforkDone {
    done: bool,
    poll: Arc<PollSet>,
}

impl VforkDone {
    pub fn new(poll: Arc<PollSet>) -> Self {
        Self { done: false, poll }
    }
}

/// A pending job-control status change awaiting report to the parent's
/// `waitpid(WUNTRACED | WCONTINUED)`.
#[derive(Clone, Copy)]
pub enum JobStatus {
    /// The process stopped after receiving the given job-control signal
    /// (`SIGSTOP`/`SIGTSTP`/`SIGTTIN`/`SIGTTOU`).
    Stopped(Signo),
    /// The process continued after receiving `SIGCONT`.
    Continued,
}

/// Job-control state for a process, kept under a single lock so the stop flag
/// and the pending parent report are updated atomically (a concurrent
/// stop/continue on another CPU must not split the two).
///
/// `stopped` and `status` are **intentionally independent** and may legitimately
/// diverge — do not collapse them into one field. `stopped` is the live parked
/// state (cleared only by continue/kill); `status` is a one-shot report the
/// parent's `waitpid` consumes (so `stopped == Some` with `status == None` is
/// valid once the report has been reaped).
#[derive(Default)]
struct JobControl {
    /// `None` = running, `Some(signo)` = stopped by the given job-control
    /// signal. A stopped process parks its threads in the kernel until
    /// `SIGCONT` (or `SIGKILL`) is delivered.
    stopped: Option<Signo>,
    /// Pending status change for the parent's `waitpid`, consumed once
    /// reported. Single-slot: a new stop/continue before the parent reaps the
    /// previous one overwrites it (unlike Linux, which queues each SIGCHLD).
    /// Adequate for the single-threaded job-control this targets.
    status: Option<JobStatus>,
    /// Bumped on every continue. A thread about to park (`set_job_stopped`)
    /// snapshots this; if it changed by the time the thread checks before
    /// parking, a `SIGCONT` raced in after the stop was recorded and the park
    /// is skipped. This closes the STOP-immediately-followed-by-CONT race
    /// (e.g. busybox `killall5 -STOP` then `-CONT`) without having to scrub the
    /// pending-signal queue.
    continue_generation: u64,
}

/// [`Process`]-shared data.
pub struct ProcessImage {
    pub exe_path: String,
    pub cmdline: Arc<Vec<String>>,
    pub auxv: Vec<AuxEntry>,
}

impl ProcessImage {
    pub fn new(exe_path: String, cmdline: Arc<Vec<String>>, auxv: Vec<AuxEntry>) -> Self {
        Self {
            exe_path,
            cmdline,
            auxv,
        }
    }
}

pub struct ProcessData {
    /// 进程。
    pub proc: Arc<Process>,
    /// 可执行文件路径。
    pub exe_path: RwLock<String>,
    /// 命令行参数。
    pub cmdline: RwLock<Arc<Vec<String>>>,
    /// 通过 /proc/[pid]/auxv 暴露的辅助向量条目。
    pub auxv: RwLock<Vec<AuxEntry>>,
    /// 虚拟内存地址空间。
    // TODO: 限定作用域
    aspace: SpinNoIrq<Arc<Mutex<AddrSpace>>>,
    /// 资源作用域。
    pub scope: RwLock<Scope>,
    /// 文件系统上下文（根目录 + 当前工作目录）。
    pub fs_context: Arc<Mutex<FsContext>>,
    /// 文件描述符表。外层 RwLock 用于 close_range(UNSHARE) 替换整张表，
    /// 内层 RwLock 用于常规的增删查操作。
    pub fd_table: RwLock<Arc<RwLock<FlattenObjects<FileDescriptor, AX_FILE_LIMIT>>>>,
    /// 用户堆顶地址。
    heap_top: AtomicUsize,

    /// 资源限制。
    pub rlim: RwLock<Rlimits>,

    /// 子进程退出等待事件。
    pub child_exit_event: Arc<PollSet>,
    /// 自身退出事件。
    pub exit_event: Arc<PollSet>,
    /// 当本进程中的某个线程退出时被唤醒。由执行 execve 的 
    /// 线程用于等待兄弟线程被回收。
    pub thread_exit_event: Arc<PollSet>,
    /// 对进程内的 execve进行串行化。同一时刻只允许一个线程 
    /// 拆除线程组；并发尝试将返回EINTR（失败者反正也即将被清除）。
    pub exec_lock: Mutex<()>,
    /// 线程的退出信号。
    pub exit_signal: Option<Signo>,
    /// 父线程组中创建了本进程的线程。
    ///
    /// Linux 的 `__WNOTHREAD` 等待选项将子进程选择范围限制为
    /// 由调用线程所创建的子进程，而默认的等待则可以回收由
    /// 同一线程组中任意线程创建的子进程。
    pub wait_parent_tid: Pid,

    /// 进程信号管理器。
    pub signal: Arc<ProcessSignalManager>,

    /// futex 表。
    futex_table: Arc<FutexTable>,

    /// 如果本进程由 vfork 创建，此字段跟踪完成状态。 
    /// 父进程等待直到 done 变为 true。使用与等待队列相同的锁 
    /// 保护，以避免唤醒丢失的竞态条件。
    vfork_done: SpinNoIrq<Option<VforkDone>>,

    /// 文件权限的默认掩码（umask）。
    umask: AtomicU32,

    /// 进程的 nice 值，用于 getpriority/setpriority 兼容。
    nice: AtomicI32,

    /// 进程本地的 membarrier(2) 注册状态位掩码。
    membarrier_state: AtomicU32,

    /// PR_GET_DUMPABLE / PR_SET_DUMPABLE 的值（默认 1 = SUID_DUMP_USER）。
    /// 每当通过 setuid / setresuid / setreuid 改变有效 UID/GID 时，清为 0
    /// （SUID_DUMP_DISABLE）（见 man 2 setuid §NOTES：
    /// "If uid is different from the old effective UID, the process will
    /// be forbidden from leaving core dumps"）。
    /// Linux 将此字段保存在 `mm_struct` 上；StarryOS 将其维护为进程级字段。
    dumpable: AtomicI32,

    /// PR_GET_THP_DISABLE / PR_SET_THP_DISABLE 的值。
    /// StarryOS 未实现透明大页，但用户空间可将其作为兼容性提示设置，
    /// 并在之后查询。
    thp_disable: AtomicU32,

    /// 已等待子进程的累计 CPU 时间（utime + stime）。
    /// 在 wait() 回收子进程时更新。
    children_cpu_time: SpinNoIrq<(TimeValue, TimeValue)>,

    /// Linux 进程 personality 标志。StarryOS 尚未对用户空间映射做随机化，
    /// 但调试器仍会探测并设置 ADDR_NO_RANDOMIZE。
    personality: AtomicUsize,

    /// POSIX 每进程间隔定时器（timer_create / timer_settime 等）
    pub posix_timers: Arc<PosixTimerTable>,

    /// 当此进程与父进程/兄弟进程共享 [`AddrSpace`] 时（`CLONE_VM`，例如
    /// vfork / posix_spawn），该值为 `true`。这种情况下，最后一个线程退出时
    /// **不能**清理地址空间——共享者可能仍在运行。
    ///
    /// 对于普通的 `fork()` 子进程以及在 `execve` 成功安装私有地址空间之后，
    /// 该值为 `false`。
    vm_aspace_shared: AtomicBool,

    /// 在 [`Self::release_aspace_slot_if_needed`] 执行后置位，以防止 `Drop`
    /// 对 [`AddrSpace::process_slots`] 重复减量。
    aspace_slot_released: AtomicBool,

    /// 作业控制状态（停止标志 + 待向父进程报告），由同一把锁保护。
    job_control: SpinNoIrq<JobControl>,

    /// 被唤醒以释放因作业控制停止而阻塞的线程。由 `SIGCONT`（继续）和
    /// `SIGKILL`（强制恢复以执行终止）触发。
    cont_event: Arc<PollSet>,
}

impl ProcessData {
    /// Create a new [`ProcessData`].
    pub fn new(
        proc: Arc<Process>,
        image: ProcessImage,
        aspace: Arc<Mutex<AddrSpace>>,
        signal_actions: Arc<SpinNoIrq<SignalActions>>,
        exit_signal: Option<Signo>,
        wait_parent_tid: Pid,
        vm_aspace_shared: bool,
        fs_context: Arc<Mutex<FsContext>>,
        fd_table: Arc<RwLock<FlattenObjects<FileDescriptor, AX_FILE_LIMIT>>>,
    ) -> Arc<Self> {
        let this = Arc::new(Self {
            proc,
            exe_path: RwLock::new(image.exe_path),
            cmdline: RwLock::new(image.cmdline),
            auxv: RwLock::new(image.auxv),
            aspace: SpinNoIrq::new(aspace),
            scope: RwLock::new(Scope::new()),
            fs_context,
            fd_table: RwLock::new(fd_table),
            heap_top: AtomicUsize::new(crate::config::USER_HEAP_BASE),

            rlim: RwLock::default(),

            child_exit_event: Arc::default(),
            exit_event: Arc::default(),
            thread_exit_event: Arc::default(),
            exec_lock: Mutex::new(()),
            exit_signal,
            wait_parent_tid,

            signal: Arc::new(ProcessSignalManager::new(
                signal_actions,
                crate::config::SIGNAL_TRAMPOLINE,
            )),

            futex_table: Arc::new(FutexTable::new()),

            vfork_done: SpinNoIrq::new(None),

            umask: AtomicU32::new(0o022),
            nice: AtomicI32::new(0),
            membarrier_state: AtomicU32::new(0),
            dumpable: AtomicI32::new(1),
            thp_disable: AtomicU32::new(0),

            children_cpu_time: SpinNoIrq::new((TimeValue::ZERO, TimeValue::ZERO)),

            personality: AtomicUsize::new(0),

            posix_timers: Arc::new(PosixTimerTable::default()),

            vm_aspace_shared: AtomicBool::new(vm_aspace_shared),
            aspace_slot_released: AtomicBool::new(false),

            job_control: SpinNoIrq::new(JobControl::default()),
            cont_event: Arc::default(),
        });
        // Clone the Arc in a separate statement: a temporary `SpinNoIrq` guard
        // from `lock()` lives until the end of the statement, so calling
        // `attach_process_slot` (which locks `Mutex<AddrSpace>`) in the same
        // expression would nest a sleepable lock inside atomic context.
        let aspace_arc = this.aspace.lock().clone();
        crate::mm::attach_process_slot(&aspace_arc);
        this
    }

    /// Whether this process shares its VM address space (`CLONE_VM`).
    #[inline]
    pub fn vm_aspace_shared(&self) -> bool {
        self.vm_aspace_shared.load(Ordering::Acquire)
    }

    /// Called after `execve` commits a fresh private address space so exit
    /// teardown may clear VMAs without touching a vfork parent's mappings.
    #[inline]
    pub fn mark_vm_aspace_private_after_exec(&self) {
        self.vm_aspace_shared.store(false, Ordering::Release);
    }

    /// Release this process's [`AddrSpace::process_slots`] entry.
    ///
    /// Invoked from the last-thread exit path so inode-scoped accounting (memfd
    /// shared-writable counts, etc.) is torn down before `waitpid` returns, and
    /// again from `Drop` if not already run. Uses reference counting: only the
    /// last slot holder triggers [`AddrSpace::clear`], so `CLONE_VM` co-owners
    /// are unaffected.
    pub fn release_aspace_slot_if_needed(&self) {
        if self.aspace_slot_released.swap(true, Ordering::AcqRel) {
            return;
        }
        let aspace = self.aspace.lock().clone();
        crate::mm::release_process_slot(&aspace);
    }

    /// Mutate the process scope from the current task.
    ///
    /// `TaskExt::on_enter` leaves the current task's active scope installed by
    /// holding one read count on [`Self::scope`]. A syscall running in that task
    /// must temporarily release that read count before taking the write side.
    pub fn with_current_scope_mut<R>(&self, f: impl FnOnce(&mut Scope) -> R) -> R {
        ActiveScope::set_global();
        unsafe { self.scope.force_read_decrement() };
        let mut scope = self.scope.write();
        let ret = f(&mut scope);
        drop(scope);
        let scope = self.scope.read();
        unsafe { ActiveScope::set(&scope) };
        core::mem::forget(scope);
        ret
    }

    /// Get the top address of the user heap.
    pub fn get_heap_top(&self) -> usize {
        self.heap_top.load(Ordering::Acquire)
    }

    /// Set the top address of the user heap.
    pub fn set_heap_top(&self, top: usize) {
        self.heap_top.store(top, Ordering::Release)
    }

    /// Linux manual: A "clone" child is one which delivers no signal, or a
    /// signal other than SIGCHLD to its parent upon termination.
    pub fn is_clone_child(&self) -> bool {
        self.exit_signal != Some(Signo::SIGCHLD)
    }

    /// Get the umask.
    pub fn umask(&self) -> u32 {
        self.umask.load(Ordering::SeqCst)
    }

    /// Set the umask.
    pub fn set_umask(&self, umask: u32) {
        self.umask.store(umask, Ordering::SeqCst);
    }

    /// Set the umask and return the old value.
    pub fn replace_umask(&self, umask: u32) -> u32 {
        self.umask.swap(umask, Ordering::SeqCst)
    }

    /// Get the process nice value.
    pub fn nice(&self) -> i32 {
        self.nice.load(Ordering::SeqCst)
    }

    /// Set the process nice value.
    pub fn set_nice(&self, nice: i32) {
        self.nice.store(nice, Ordering::SeqCst);
    }

    /// Get the membarrier(2) registration state bitmask.
    pub fn membarrier_state(&self) -> u32 {
        self.membarrier_state.load(Ordering::SeqCst)
    }

    /// Add bits to the membarrier(2) registration state.
    pub fn register_membarrier_state(&self, state: u32) {
        self.membarrier_state.fetch_or(state, Ordering::SeqCst);
    }

    /// Get the dumpable flag (PR_GET_DUMPABLE).
    pub fn dumpable(&self) -> i32 {
        self.dumpable.load(Ordering::SeqCst)
    }

    /// Set the dumpable flag (PR_SET_DUMPABLE).
    /// Valid userspace values are 0 (SUID_DUMP_DISABLE) and 1
    /// (SUID_DUMP_USER). Callers must validate before storing.
    pub fn set_dumpable(&self, dumpable: i32) {
        self.dumpable.store(dumpable, Ordering::SeqCst);
    }

    /// Get the transparent huge page disable state (PR_GET_THP_DISABLE).
    pub fn thp_disable(&self) -> u32 {
        self.thp_disable.load(Ordering::SeqCst)
    }

    /// Set the transparent huge page disable state (PR_SET_THP_DISABLE).
    pub fn set_thp_disable(&self, thp_disable: u32) {
        self.thp_disable.store(thp_disable, Ordering::SeqCst);
    }

    /// Returns true if the process is currently job-control stopped.
    pub fn is_job_stopped(&self) -> bool {
        self.job_control.lock().stopped.is_some()
    }

    /// Mark the process stopped by `signo` and queue a `Stopped` report for the
    /// parent's `waitpid(WUNTRACED)`. Returns `true` if the caller should park.
    ///
    /// Returns `false` (and records nothing) when a `SIGCONT` arrived after the
    /// stop signal was dequeued but before this call — see
    /// [`Self::set_job_continued`] / `continue_generation`. Closing this race at
    /// the stop site lets us avoid scrubbing the pending-signal queue (which
    /// would require modifying `starry-signal`).
    pub fn set_job_stopped(&self, signo: Signo, continue_gen_snapshot: u64) -> bool {
        let mut jc = self.job_control.lock();
        if jc.continue_generation != continue_gen_snapshot {
            // A continue raced in after we observed `continue_gen_snapshot`;
            // honor it and do not stop.
            return false;
        }
        jc.stopped = Some(signo);
        jc.status = Some(JobStatus::Stopped(signo));
        true
    }

    /// Snapshot the continue generation. Taken right after a stop signal is
    /// dequeued and passed to [`Self::set_job_stopped`]; any intervening
    /// `SIGCONT` advances the generation and cancels the stop.
    pub fn continue_generation(&self) -> u64 {
        self.job_control.lock().continue_generation
    }

    /// Continue a stopped process: clear the stop, queue a `Continued` report,
    /// and wake parked threads. Returns true if it had been stopped.
    ///
    /// Always advances `continue_generation` so a concurrent stop in progress
    /// (signal already dequeued, not yet parked) observes the continue and
    /// skips parking.
    pub fn set_job_continued(&self) -> bool {
        let mut jc = self.job_control.lock();
        jc.continue_generation = jc.continue_generation.wrapping_add(1);
        let was_stopped = jc.stopped.take().is_some();
        if was_stopped {
            jc.status = Some(JobStatus::Continued);
            drop(jc);
            // Wake only when a thread was actually parked; avoids spurious
            // wakeups on SIGCONT to an already-running process.
            // Continue state is published before waking stopped threads.
            unsafe { self.cont_event.wake(IoEvents::IN) };
        }
        was_stopped
    }

    /// Force-clear the stop (for `SIGKILL`) so a parked thread re-checks and
    /// proceeds to terminate. Does not queue a `Continued` report.
    pub fn clear_job_stop_for_kill(&self) {
        let was_stopped = self.job_control.lock().stopped.take().is_some();
        if was_stopped {
            // Stop state is cleared before waking stopped threads.
            unsafe { self.cont_event.wake(IoEvents::IN) };
        }
    }

    /// The wait queue woken when the process is continued or killed.
    pub fn cont_event(&self) -> Arc<PollSet> {
        self.cont_event.clone()
    }

    /// Peek the pending job-control status report (without consuming it) if it
    /// matches a kind the caller's `waitpid` flags allow (`WUNTRACED` for
    /// stopped, `WCONTINUED` for continued).
    pub fn peek_job_status_if(
        &self,
        want_stopped: bool,
        want_continued: bool,
    ) -> Option<JobStatus> {
        let jc = self.job_control.lock();
        match jc.status {
            Some(s @ JobStatus::Stopped(_)) if want_stopped => Some(s),
            Some(s @ JobStatus::Continued) if want_continued => Some(s),
            _ => None,
        }
    }

    /// Consume the pending job-control status report if it matches a kind the
    /// caller's `waitpid` flags allow. Mirrors [`Self::peek_job_status_if`] but
    /// clears the slot; call it only after the status has been published to
    /// userspace so a faulting copy leaves the report intact to retry.
    pub fn take_job_status_if(
        &self,
        want_stopped: bool,
        want_continued: bool,
    ) -> Option<JobStatus> {
        let mut jc = self.job_control.lock();
        match jc.status {
            Some(JobStatus::Stopped(_)) if want_stopped => jc.status.take(),
            Some(JobStatus::Continued) if want_continued => jc.status.take(),
            _ => None,
        }
    }

    /// Get the accumulated CPU time of waited children.
    pub fn children_cpu_time(&self) -> (TimeValue, TimeValue) {
        *self.children_cpu_time.lock()
    }

    /// Accumulate a child's CPU time when it is reaped by wait().
    pub fn add_child_cpu_time(&self, utime: TimeValue, stime: TimeValue) {
        let mut time = self.children_cpu_time.lock();
        time.0 += utime;
        time.1 += stime;
    }

    pub fn personality(&self) -> usize {
        self.personality.load(Ordering::Acquire)
    }

    pub fn replace_personality(&self, personality: usize) -> usize {
        self.personality.swap(personality, Ordering::AcqRel)
    }

    /// Returns a clone of the address space Arc.
    pub fn aspace(&self) -> Arc<Mutex<AddrSpace>> {
        self.aspace.lock().clone()
    }

    /// Replace this process's address space with a new one.
    ///
    /// # Why `mem::replace` instead of `*guard = new_aspace`
    ///
    /// `self.aspace` is a `SpinNoIrq<Arc<Mutex<AddrSpace>>>`. Locking it
    /// disables IRQs and increments `preempt_count`, putting us in atomic
    /// context. A plain assignment (`*guard = new_aspace`) would drop the
    /// **old** `Arc<Mutex<AddrSpace>>` while the `SpinNoIrq` guard is still
    /// alive. If that was the last strong reference (e.g. after a
    /// `CLONE_VM` + `execve`), the destructor chain would be:
    ///
    /// ```text
    /// Arc::drop → Mutex<AddrSpace>::drop → AddrSpace::drop
    ///   → self.clear() → areas.clear() → FileBackendInner::drop
    ///     → cache.remove_evict_listener()
    ///       → evict_listeners.lock()        ← sleeping Mutex
    ///         → might_sleep()               ← PANIC (atomic context)
    /// ```
    ///
    /// `mem::replace` moves the old Arc out of the guard so it is dropped
    /// **after** the `SpinNoIrq` guard, in normal preemptible context.
    pub fn replace_aspace(&self, new_aspace: Arc<Mutex<AddrSpace>>) {
        let old = {
            let mut guard = self.aspace.lock();
            core::mem::replace(&mut *guard, new_aspace)
        };
        crate::mm::release_process_slot(&old);
        let aspace_arc = self.aspace.lock().clone();
        crate::mm::attach_process_slot(&aspace_arc);
    }

    /// Set the vfork completion (called on the child after a vfork,
    /// before the child task is spawned).
    pub fn set_vfork_done(&self, poll: Arc<PollSet>) {
        *self.vfork_done.lock() = Some(VforkDone::new(poll));
    }

    /// Wait for vfork completion. Returns immediately if already done.
    /// This should be called by the parent after spawning the vfork child.
    ///
    /// The wait is killable but not arbitrarily signal-interruptible
    /// (mirroring Linux's `wait_for_completion_killable`):
    ///
    ///   - If the child notifies (exec or exit), we return normally.
    ///   - If another thread in this parent process does `execve` it will
    ///     zap us by setting `exit_request`. We bail and let the user-
    ///     return path consume `exit_request` and route to
    ///     `do_exit(0, false)`. Without this, `WaitQueue::wait_until`
    ///     would never observe the zap and the execve initiator would
    ///     deadlock in its sibling-teardown loop.
    ///   - Non-fatal signal wakeups must not unblock us: returning early
    ///     while the child still shares our address space would violate
    ///     the vfork contract. We re-enter the wait in that case.
    pub fn wait_vfork_done(&self) {
        let poll = {
            let guard = self.vfork_done.lock();
            match guard.as_ref() {
                Some(vfork) => vfork.poll.clone(),
                None => return, // No vfork, shouldn't happen but be safe.
            }
        };
        let curr_task = ax_task::current();
        let curr_thr = curr_task.as_thread();
        loop {
            let result = ax_task::future::block_on(ax_task::future::interruptible(
                core::future::poll_fn(|cx| {
                    // Register before re-checking so a notify that fires
                    // between our last check and this register isn't lost.
                    // Registration happens from the vfork parent task context.
                    unsafe { poll.register(cx.waker(), IoEvents::IN) };
                    let done = self
                        .vfork_done
                        .lock()
                        .as_ref()
                        .map(|v| v.done)
                        .unwrap_or(true);
                    if done {
                        core::task::Poll::Ready(())
                    } else {
                        core::task::Poll::Pending
                    }
                }),
            ));
            match result {
                Ok(()) => return,
                Err(_) => {
                    if curr_thr.has_exit_request() {
                        return;
                    }
                    // Spurious wake from a non-fatal signal; keep waiting.
                    continue;
                }
            }
        }
    }

    /// Notify the vfork parent that this child has exec'd or exited.
    /// No-op if this process was not created by vfork.
    pub fn notify_vfork_done(&self) {
        // Set done under the lock, then drop the lock before notifying
        // to avoid lock-order inversion with the poll-set internal lock.
        let poll = {
            let mut guard = self.vfork_done.lock();
            match guard.as_mut() {
                Some(vfork) => {
                    vfork.done = true;
                    vfork.poll.clone()
                }
                None => return,
            }
            // guard dropped here
        };
        // vfork completion is published before waking the parent.
        unsafe { poll.wake(IoEvents::IN) };
    }
}

impl Drop for ProcessData {
    fn drop(&mut self) {
        self.release_aspace_slot_if_needed();
    }
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicBool, Ordering};

    use super::NextSignalCheckBlock;

    #[test]
    fn old_global_signal_check_block_leaks_between_threads() {
        static OLD_BLOCK_NEXT_SIGNAL_CHECK: AtomicBool = AtomicBool::new(false);

        fn block_next_signal() {
            OLD_BLOCK_NEXT_SIGNAL_CHECK.store(true, Ordering::SeqCst);
        }

        fn unblock_next_signal() -> bool {
            OLD_BLOCK_NEXT_SIGNAL_CHECK.swap(false, Ordering::SeqCst)
        }

        // Simulate thread A returning from `rt_sigreturn()`.
        block_next_signal();

        // Simulate thread B reaching the user return path first and incorrectly
        // consuming thread A's one-shot state.
        assert!(
            unblock_next_signal(),
            "the old global flag leaks across logical threads"
        );
        assert!(!unblock_next_signal());
    }

    #[test]
    fn per_thread_signal_check_block_is_isolated() {
        let thread_a = NextSignalCheckBlock::new();
        let thread_b = NextSignalCheckBlock::new();

        thread_a.block();

        assert!(
            !thread_b.unblock(),
            "thread B must not observe thread A's signal-check block"
        );
        assert!(thread_a.unblock());
        assert!(!thread_a.unblock());
    }
}
