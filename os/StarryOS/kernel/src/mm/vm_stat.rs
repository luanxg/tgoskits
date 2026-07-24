//! 每个地址空间的虚拟内存统计 (VmX)。
//!
//! [`ProcessVmStat`] 是所有 VmX 计数器的唯一权威来源。
//! 它内置于 [`super::AddrSpace`] 中，由 `map` / `unmap` / `clear` /
//! `try_clone` 自动维护，因此任何 syscall 处理函数都无需手动操作它。
//!
//! # 计数器分类
//!
//! | 类别 | 字段 | 更新规则 |
//! |---|---|---|
//! | 当前值 (O(1) 原子) | `vss_pages` | map 时 +size, unmap/clear 时 -size |
//! | 高水位线 | `peak_vss_pages`, `peak_rss_pages` | map 时 `fetch_max` |
//! | RSS (Plan2) | `rss_pages` | 保留字段，Plan2 之前始终为 0 |
//!
//! 当前 VSS 使用 `AtomicI64`（有符号）维护，因此重复 unmap
//! 或竞争条件永远不会回绕到 u64::MAX；读取时始终取
//! `max(0, value)`。

use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// 单个地址空间的所有 VmX 统计信息。
///
/// 字段有意设为私有；请使用提供的方法进行读取或更新。
/// 这确保了高水位线的单调性不变量由构造本身来保证。
pub struct ProcessVmStat {
    // ── 当前计数器 (O(1), 每次 map/unmap 时更新) ────────────────────────
    /// 当前虚拟内存大小，以页为单位 (VmSize)。使用有符号类型以捕获下溢 bug。
    vss_pages: AtomicI64,

    // ── 高水位线 (单调非递减) ──────────────────────────────────────────
    /// 虚拟内存大小的历史峰值，以页为单位 (VmPeak)。
    peak_vss_pages: AtomicU64,
    /// 常驻内存大小的历史峰值，以页为单位 (VmHWM)。
    /// Plan1: 镜像 peak_vss。Plan2: 替换为真实的 RSS 追踪。
    peak_rss_pages: AtomicU64,
    // ── RSS 占位符 (Plan2) ─────────────────────────────────────────────
    // Plan2 落地后，在此处添加 `rss_pages: AtomicI64`，
    // 并在缺页/回收路径中更新。届时高水位线应使用真实 RSS 更新。
}

impl ProcessVmStat {
    pub const fn new() -> Self {
        Self {
            vss_pages: AtomicI64::new(0),
            peak_vss_pages: AtomicU64::new(0),
            peak_rss_pages: AtomicU64::new(0),
        }
    }

    // ── Read accessors ────────────────────────────────────────────────────

    /// Current VSS in pages (VmSize).
    #[inline]
    pub fn vss_pages(&self) -> u64 {
        self.vss_pages.load(Ordering::Relaxed).max(0) as u64
    }

    /// Peak VSS in pages (VmPeak).
    #[inline]
    pub fn peak_vss_pages(&self) -> u64 {
        self.peak_vss_pages.load(Ordering::Relaxed)
    }

    /// Peak RSS in pages (VmHWM).
    #[inline]
    pub fn peak_rss_pages(&self) -> u64 {
        self.peak_rss_pages.load(Ordering::Relaxed)
    }

    // ── Mutation (called only by AddrSpace) ───────────────────────────────

    /// Account for `pages` newly mapped pages and update high-water marks.
    ///
    /// Must be called **after** the mapping succeeds so that a failed map does
    /// not advance the watermarks.
    #[inline]
    pub(super) fn on_map(&self, pages: u64) {
        let new_vss = self
            .vss_pages
            .fetch_add(pages as i64, Ordering::Relaxed)
            .max(0) as u64
            + pages;
        // Plan1: RSS == VSS.  Plan2: pass real RSS here instead.
        self.peak_vss_pages.fetch_max(new_vss, Ordering::Relaxed);
        self.peak_rss_pages.fetch_max(new_vss, Ordering::Relaxed);
    }

    /// Account for `pages` unmapped pages.  High-water marks are never lowered.
    #[inline]
    pub(super) fn on_unmap(&self, pages: u64) {
        self.vss_pages.fetch_sub(pages as i64, Ordering::Relaxed);
    }

    /// Reset all counters to zero (exec / address-space teardown).
    ///
    /// High-water marks are also reset: Linux resets VmPeak/VmHWM on `execve`
    /// because the new image starts a fresh `mm_struct`.
    #[inline]
    pub(super) fn on_clear(&self) {
        self.vss_pages.store(0, Ordering::Relaxed);
        self.peak_vss_pages.store(0, Ordering::Relaxed);
        self.peak_rss_pages.store(0, Ordering::Relaxed);
    }

    /// Seed this stat from a parent's snapshot (used when `try_clone` builds
    /// the child address space for `fork`/`clone`).
    ///
    /// The child inherits the parent's current VSS as its starting watermarks,
    /// matching Linux: the child's `mm_struct` starts with `hiwater_vm` set to
    /// the copied `total_vm`.
    #[inline]
    pub(super) fn seed_from(&self, parent: &Self) {
        let vss = parent.vss_pages();
        self.vss_pages.store(vss as i64, Ordering::Relaxed);
        // Child's peaks start at current VSS (the copied address space size).
        self.peak_vss_pages
            .store(parent.peak_vss_pages().max(vss), Ordering::Relaxed);
        self.peak_rss_pages
            .store(parent.peak_rss_pages().max(vss), Ordering::Relaxed);
    }
}

impl Default for ProcessVmStat {
    fn default() -> Self {
        Self::new()
    }
}
