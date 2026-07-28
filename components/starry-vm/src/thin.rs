use core::{mem::MaybeUninit, ptr::NonNull, slice};

use bytemuck::AnyBitPattern;

use crate::{VmResult, vm_read_slice, vm_write_slice};

/// 虚拟内存指针。
pub trait VmPtr: Copy {
    /// 该指针所指向的数据类型。
    type Target;

    #[doc(hidden)]
    fn as_ptr(self) -> *const Self::Target;

    /// 如果指针为空则返回 `None`，否则返回 `Some(self)`。
    fn nullable(self) -> Option<Self> {
        if self.as_ptr().is_null() {
            None
        } else {
            Some(self)
        }
    }

    /// 从此虚拟内存指针读取值。与 [`VmPtr::vm_read`] 不同，
    /// 此方法不要求值必须已初始化。
    fn vm_read_uninit(self) -> VmResult<MaybeUninit<Self::Target>> {
        let mut uninit = MaybeUninit::<Self::Target>::uninit();
        vm_read_slice(self.as_ptr(), slice::from_mut(&mut uninit))?;
        Ok(uninit)
    }

    /// 从此虚拟内存指针读取值。
    fn vm_read(self) -> VmResult<Self::Target>
    where
        Self::Target: AnyBitPattern,
    {
        let uninit = self.vm_read_uninit()?;
        // SAFETY: `AnyBitPattern`
        Ok(unsafe { uninit.assume_init() })
    }
}

impl<T> VmPtr for *const T {
    type Target = T;

    fn as_ptr(self) -> *const T {
        self
    }
}

impl<T> VmPtr for *mut T {
    type Target = T;

    fn as_ptr(self) -> *const T {
        self
    }
}

impl<T> VmPtr for NonNull<T> {
    type Target = T;

    fn as_ptr(self) -> *const T {
        self.as_ptr()
    }
}

/// 可变虚拟内存指针。
pub trait VmMutPtr: VmPtr {
    /// 用给定的值覆盖虚拟内存中的某个位置。
    fn vm_write(self, value: Self::Target) -> VmResult {
        vm_write_slice(self.as_ptr().cast_mut(), slice::from_ref(&value))
    }
}

impl<T> VmMutPtr for *mut T {}

impl<T> VmMutPtr for NonNull<T> {}
