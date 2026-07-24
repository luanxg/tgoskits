//! Architecture-specific configurations.

cfg_if::cfg_if! {
    if #[cfg(target_arch = "aarch64")] {
        #[rustfmt::skip]
        mod aarch64;
        pub use aarch64::*;
    } else {
        compile_error!("Unsupported architecture");
    }
}
