extern crate alloc;

use alloc::vec::Vec;

use bytemuck::{AnyBitPattern, Pod, bytes_of, zeroed};

use crate::{VmError, VmImpl, VmIo, VmResult, vm_read_slice};

/// 从虚拟内存中加载一个元素向量。
///
/// # 安全性
///
/// 调用者必须确保 `ptr` 指向的内存是有效且已初始化的。
pub unsafe fn vm_load_any<T>(ptr: *const T, len: usize) -> VmResult<Vec<T>> {
    let mut buf = Vec::with_capacity(len);
    vm_read_slice(ptr, &mut buf.spare_capacity_mut()[..len])?;
    // 安全性：调用者保证内存是有效且已初始化的。
    unsafe { buf.set_len(len) }
    Ok(buf)
}

/// 从虚拟内存中加载一个元素向量。
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn vm_load<T: AnyBitPattern>(ptr: *const T, len: usize) -> VmResult<Vec<T>> {
    // 安全性：`AnyBitPattern` 约束保证了任意位模式都是有效的。
    unsafe { vm_load_any(ptr, len) }
}

#[inline]
fn is_zero<T: Pod>(value: &T) -> bool {
    bytes_of(value) == bytes_of(&zeroed::<T>())
}

const MAX_BYTES: usize = 131072;

/// 从给定指针加载元素，直到遇到零元素为止。
pub fn vm_load_until_nul<T: Pod>(ptr: *const T) -> VmResult<Vec<T>> {
    if !ptr.is_aligned() {
        return Err(VmError::BadAddress);
    }

    let size = size_of::<T>();
    let mut result = Vec::new();
    let mut vm = VmImpl::new();

    loop {
        const CHUNK_SIZE: usize = 32;

        let start = ptr.addr() + result.len() * size;
        let end = (start + 1).next_multiple_of(CHUNK_SIZE);
        let len = (end - start) / size;

        result.reserve(len);
        let buf = &mut result.spare_capacity_mut()[..len];
        vm.read(start, buf.as_bytes_mut())?;

        // SAFETY: `Pod`
        let buf = unsafe { buf.assume_init_ref() };
        let pos = buf.iter().position(is_zero);

        unsafe { result.set_len(result.len() + pos.unwrap_or(len)) };
        if result.len() >= MAX_BYTES / size {
            return Err(VmError::TooLong);
        }

        if pos.is_some() {
            break;
        }
    }

    Ok(result)
}
