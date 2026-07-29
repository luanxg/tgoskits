//! Virtual memory utilities.
#![no_std]
#![feature(maybe_uninit_as_bytes)]
#![warn(missing_docs)]

use core::{mem::MaybeUninit, slice};

use ax_errno::AxError;
use extern_trait::extern_trait;

/// 虚拟内存操作可能产生的错误。
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum VmError {
    /// 地址无效，例如未对齐到要求的边界、越界（包括空指针）。
    BadAddress,
    /// 操作不被允许，例如尝试写入只读内存。
    AccessDenied,
    /// C 风格字符串或数组过长。
    ///
    /// 当在预定义的搜索限制内未找到空终止符时，
    /// [`vm_load_until_nul`] 会返回此错误。
    #[cfg(feature = "alloc")]
    TooLong,
}

impl From<VmError> for AxError {
    fn from(err: VmError) -> Self {
        match err {
            VmError::BadAddress | VmError::AccessDenied => AxError::BadAddress,
            #[cfg(feature = "alloc")]
            VmError::TooLong => AxError::NameTooLong,
        }
    }
}

/// 虚拟内存操作的结果类型。
pub type VmResult<T = ()> = Result<T, VmError>;

/// 访问虚拟内存的接口。
///
/// # 安全性
///
/// - 实现者必须确保内存访问是安全的，且不违反任何内存安全规则。
/// #[extern_trait(VmImpl)]是因为具体实现的VmIo trait的类型不在不当前的crate中
#[extern_trait(VmImpl)]
pub unsafe trait VmIo {
    /// 创建一个 [`VmIo`] 实例。
    ///
    /// 用于那些可能需要存储一些状态或数据才能执行操作的实现。
    /// 如果不需要任何状态，实现者可以将其留空。
    fn new() -> Self;

    /// 从虚拟内存中 `start` 处开始读取数据到 `buf` 中。
    /// 模拟cory_from_user函数
    fn read(&mut self, start: usize, buf: &mut [MaybeUninit<u8>]) -> VmResult;

    /// 将 `buf` 中的数据写入到虚拟内存中 `start` 处。
    /// 模拟copy_to_user
    fn write(&mut self, start: usize, buf: &[u8]) -> VmResult;
}

/// 从虚拟内存中读取一个切片。
///
/// 用户指针无需对齐到 `align_of::<T>()`。底层的 `user_copy` 在所有架构上
/// 都是字节粒度的（x86 使用 `rep movsb`；aarch64/riscv64/loongarch64 先将
/// 目标地址字节对齐，再进行批量拷贝）——与 Linux `copy_from_user` 完全一致，
/// 后者从不要求用户缓冲区对齐。旧的 `is_aligned()` 检查会错误地拒绝有效的
/// 非对齐用户缓冲区。
pub fn vm_read_slice<T>(ptr: *const T, buf: &mut [MaybeUninit<T>]) -> VmResult {
    VmImpl::new().read(ptr.addr(), buf.as_bytes_mut())
}

/// 将数据写入虚拟内存。
///
/// 无指针对齐要求（与 Linux 保持一致：`copy_to_user` 不关心对齐；
/// 参见 [`vm_read_slice`]）。旧的 `is_aligned()` 检查曾导致 `epoll_pwait`
/// 在 riscv64/loongarch64 上返回 EFAULT：Go 的 `[]epollevent` 是 4 字节
/// 对齐的（`data [8]byte`），而 `struct epoll_event` 在非 x86 架构上是
/// 8 字节对齐的（`u64 data`），因此事件缓冲区无法通过对齐检查，
/// 导致 Go 网络轮询器崩溃（`netpoll failed`）。
pub fn vm_write_slice<T>(ptr: *mut T, buf: &[T]) -> VmResult {
    // 安全性：我们不关心数据的有效性，因为这些字节仅用于写入虚拟内存。
    let bytes = unsafe { slice::from_raw_parts(buf.as_ptr().cast::<u8>(), size_of_val(buf)) };
    VmImpl::new().write(ptr.addr(), bytes)
}

mod thin;
pub use thin::{VmMutPtr, VmPtr};

#[cfg(feature = "alloc")]
mod alloc;
#[cfg(feature = "alloc")]
pub use alloc::{vm_load, vm_load_any, vm_load_until_nul};
