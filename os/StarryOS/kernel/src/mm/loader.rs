//! User address space management.

use alloc::{borrow::ToOwned, collections::VecDeque, string::String, vec, vec::Vec};
use core::{ffi::CStr, iter, mem::size_of};

use ax_errno::{AxError, AxResult};
use ax_fs_ng::vfs::{CachedFile, FileBackend, FsContext};
use ax_memory_addr::{MemoryAddr, PAGE_SIZE_4K, VirtAddr};
use ax_runtime::hal::{
    mem::virt_to_phys,
    paging::{MappingFlags, PageSize},
};
use ax_sync::Mutex;
use axfs_ng_vfs::Location;
use kernel_elf_parser::{AuxEntry, AuxType, ELFHeaders, ELFHeadersBuilder, ELFParser};
use ouroboros::self_referencing;
use uluru::LRUCache;
use zerocopy::IntoBytes;

use crate::{
    config::{USER_SPACE_BASE, USER_SPACE_SIZE},
    mm::aspace::{AddrSpace, Backend},
};

#[cfg(target_arch = "riscv64")]
const RISCV_COMPAT_HWCAP_IMAFDC: usize = (1 << (b'I' - b'A'))
    | (1 << (b'M' - b'A'))
    | (1 << 0)
    | (1 << (b'F' - b'A'))
    | (1 << (b'D' - b'A'))
    | (1 << (b'C' - b'A'));

// RISC-V relocation types
#[cfg(target_arch = "riscv64")]
const R_RISCV_RELATIVE: u32 = 3;
#[cfg(target_arch = "riscv64")]
const R_RISCV_JUMP_SLOT: u32 = 5;
#[cfg(target_arch = "riscv64")]
const R_RISCV_64: u32 = 2;
#[cfg(target_arch = "riscv64")]
const R_RISCV_COPY: u32 = 4;

/// Creates a new empty user address space.
pub fn new_user_aspace_empty() -> AxResult<AddrSpace> {
    AddrSpace::new_empty(VirtAddr::from_usize(USER_SPACE_BASE), USER_SPACE_SIZE)
}

/// If the target architecture requires it, the kernel portion of the address
/// space will be copied to the user address space.
pub fn copy_from_kernel(_aspace: &mut AddrSpace) -> AxResult {
    #[cfg(not(any(target_arch = "aarch64", target_arch = "loongarch64")))]
    {
        // ARMv8 (aarch64) and LoongArch64 use separate page tables for user space
        // (aarch64: TTBR0_EL1, LoongArch64: PGDL), so there is no need to copy the
        // kernel portion to the user page table.
        let kspace = ax_mm::kernel_aspace().lock();
        _aspace.page_table_mut().cursor().copy_from(
            kspace.page_table(),
            kspace.base(),
            kspace.size(),
        );
    }
    Ok(())
}

/// Map the signal trampoline to the user address space.
pub fn map_trampoline(aspace: &mut AddrSpace) -> AxResult {
    let signal_trampoline_paddr =
        virt_to_phys(starry_signal::arch::signal_trampoline_address().into());
    aspace.map_linear(
        crate::config::SIGNAL_TRAMPOLINE.into(),
        signal_trampoline_paddr,
        PAGE_SIZE_4K,
        MappingFlags::READ | MappingFlags::EXECUTE | MappingFlags::USER,
    )?;
    Ok(())
}

fn mapping_flags(flags: xmas_elf::program::Flags) -> MappingFlags {
    let mut mapping_flags = MappingFlags::USER;
    if flags.is_read() {
        mapping_flags |= MappingFlags::READ;
    }
    if flags.is_write() {
        mapping_flags |= MappingFlags::WRITE | MappingFlags::READ;
    }
    if flags.is_execute() {
        mapping_flags |= MappingFlags::EXECUTE;
    }
    mapping_flags
}

fn app_stack_region(args: &[String], envs: &[String], auxv: &[AuxEntry], sp: usize) -> Vec<u8> {
    let mut data = VecDeque::new();
    let mut push = |src: &[u8]| -> usize {
        data.extend(src.iter().copied());
        data.rotate_right(src.len());
        sp - data.len()
    };

    let random_str_pos = push(b"0123456789abcdef");
    let envs_slice: Vec<_> = envs
        .iter()
        .map(|env| {
            push(b"\0");
            push(env.as_bytes())
        })
        .collect();
    let argv_slice: Vec<_> = args
        .iter()
        .map(|arg| {
            push(b"\0");
            push(arg.as_bytes())
        })
        .collect();
    let padding_null = "\0".repeat(size_of::<usize>());
    let sp = push(padding_null.as_bytes());

    push(&b"\0".repeat(sp % 16));

    if (envs.len() + args.len() + 3) & 1 != 0 {
        push(padding_null.as_bytes());
    }

    let has_random = auxv.iter().any(|entry| entry.get_type() == AuxType::RANDOM);
    let has_execfn = auxv.iter().any(|entry| entry.get_type() == AuxType::EXECFN);

    // `push` prepends bytes to the stack image. Push the terminator first so
    // user memory presents auxv as: supplied entries, AT_RANDOM, AT_EXECFN,
    // AT_NULL. Without AT_NULL, musl keeps parsing argv/env padding as auxv
    // and can falsely enable AT_SECURE.
    push(AuxEntry::new(AuxType::NULL, 0).as_bytes());
    if !has_execfn {
        push(AuxEntry::new(AuxType::EXECFN, argv_slice[0]).as_bytes());
    }
    if !has_random {
        push(AuxEntry::new(AuxType::RANDOM, random_str_pos).as_bytes());
    }
    push(auxv.as_bytes());

    push(padding_null.as_bytes());
    push(envs_slice.as_bytes());
    push(padding_null.as_bytes());
    push(argv_slice.as_bytes());
    let sp = push(args.len().as_bytes());

    assert!(sp % 16 == 0);

    let mut result = Vec::with_capacity(data.len());
    let (first, second) = data.as_slices();
    result.extend_from_slice(first);
    result.extend_from_slice(second);
    result
}

/// 将 ELF 文件映射到用户地址空间。
///
/// # 参数
/// - `uspace`: 用户程序的地址空间。
/// - `elf`: ELF 文件。
///
/// # 返回值
/// - 用户程序的入口点。
fn map_elf<'a>(
    uspace: &mut AddrSpace,
    base: usize,
    entry: &'a ElfCacheEntry,
) -> AxResult<ELFParser<'a>> {
    //入口地址就是虚拟地址
    let elf_parser = ELFParser::new(entry.borrow_elf(), base).map_err(|_| AxError::InvalidData)?;
    //相当于一个文件的句柄
    let cache = entry.borrow_cache();

    // PT_TLS 的初始化镜像可能超出最后一个 PT_LOAD 的文件范围。
    // 这里假设 PT_TLS 的文件数据紧跟在最后一个 PT_LOAD 段
    // 的文件末尾之后且与之连续，这是 GNU ld 和 LLVM lld 生成的
    // 标准布局。
    // 计算出所需的最大文件偏移，以便 COW 后端能够为动态链接器
    // 处理 TLS 初始化镜像的缺页请求。
    // tls_max_offset 是 TLS 初始化镜像在文件中的最远结束偏移。
    let tls_max_offset: u64 = elf_parser
        .headers()
        .ph
        .iter()
        .filter(|ph| ph.get_type() == Ok(xmas_elf::program::Type::Tls))
        .map(|ph| {
            debug!(
                "PT_TLS: vaddr={:#x} memsz={:#x} filesz={:#x} offset={:#x}",
                ph.virtual_addr, ph.mem_size, ph.file_size, ph.offset
            );
            ph.offset + ph.file_size
        })
        .max()
        .unwrap_or(0);

    let load_segments: Vec<_> = elf_parser
        .headers()
        .ph
        .iter()
        .filter(|ph| ph.get_type() == Ok(xmas_elf::program::Type::Load))
        .collect();
    let last_load_idx = load_segments.len().wrapping_sub(1);

    for (i, ph) in load_segments.iter().enumerate() {
        let vaddr = ph.virtual_addr as usize + elf_parser.base();
        debug!(
            "Mapping ELF segment: [{:#x?}, {:#x?}) flags: {}",
            vaddr,
            vaddr + ph.mem_size as usize,
            ph.flags
        );
        let seg_pad = vaddr.align_offset_4k();
        assert_eq!(seg_pad, ph.offset as usize % PAGE_SIZE_4K);

        let seg_align_size =
            (ph.mem_size as usize + seg_pad + PAGE_SIZE_4K - 1) & !(PAGE_SIZE_4K - 1);
        let seg_start = VirtAddr::from_usize(vaddr);

        // 注意，此处的 `offset` 可能未按 4K 对齐，正确处理它
        // 是后端的职责。
        let file_end = if i == last_load_idx && tls_max_offset > ph.offset + ph.file_size {
            tls_max_offset
        } else {
            ph.offset + ph.file_size
        };
        let backend = Backend::new_cow(
            seg_start,
            PageSize::Size4K,
            FileBackend::Cached(cache.clone()),
            ph.offset,
            Some(file_end),
            false,
        );
        uspace.map(
            seg_start.align_down_4k(),
            seg_align_size,
            mapping_flags(ph.flags),
            false,
            backend,
        )?;
    }

    // 为 static-pie 二进制文件应用重定位
    // 在非 riscv64 架构上，apply_relocations() 是一个空操作桩函数。
    // 共享库的解析，可以先不管
    if elf_parser.headers().header.pt1.class() == xmas_elf::header::Class::SixtyFour {
        let is_pie = elf_parser.headers().header.pt2.type_().as_type()
            == xmas_elf::header::Type::SharedObject;
        if is_pie {
            apply_relocations(uspace, base, entry.borrow_cache(), &elf_parser.headers().ph)?;
        }
    }

    Ok(elf_parser)
}

/// Stub for non-riscv64 architectures
#[cfg(not(target_arch = "riscv64"))]
fn apply_relocations(
    _uspace: &mut AddrSpace,
    _base: usize,
    _cache: &CachedFile,
    _ph: &[xmas_elf::program::ProgramHeader64],
) -> AxResult {
    Ok(())
}

fn map_elf_error(err: &'static str) -> AxError {
    debug!("Failed to parse ELF file: {err}");
    AxError::InvalidExecutable
}

#[self_referencing]
struct ElfCacheEntry {
    cache: CachedFile,
    data: Vec<u8>,
    #[borrows(data)]
    #[covariant]
    elf: ELFHeaders<'this>,
}

impl ElfCacheEntry {
    fn load(loc: Location) -> AxResult<Result<Self, Vec<u8>>> {
        let cache = CachedFile::get_or_create(loc)?;

        //从文件的偏移量 0 开始，最多读取 4096 字节（4KB）到 data 缓冲区中。
        let mut data = vec![0; 4096];
        let read = cache.read_at(&mut data[..], 0)?;

        //把 data 的长度缩减到实际读取的字节数。
        data.truncate(read);

        match ElfCacheEntry::try_new_or_recover::<AxError>(cache.clone(), data, |data| {
            let builder = ELFHeadersBuilder::new(data).map_err(map_elf_error)?;
            let range = builder.ph_range();
            if range.end as usize <= data.len() {
                builder.build(&data[range.start as usize..range.end as usize])
            } else {
                let mut buf = vec![0; (range.end - range.start) as usize];
                cache.read_at(&mut buf[..], range.start)?;
                builder.build(&buf)
            }
            .map_err(map_elf_error)
        }) {
            Ok(e) => Ok(Ok(e)),
            Err((_, heads)) => Ok(Err(heads.data)),
        }
    }
}

/// The value reported in the `AT_HWCAP` auxiliary vector entry.
///
/// `AT_HWCAP` (auxv type 16) advertises architecture-dependent CPU capability
/// bits to userspace. `getauxval(AT_HWCAP)` reads it, and feature-dispatching
/// runtimes gate optional instruction sets on it.
///
/// Per-arch policy:
/// - **loongarch64**: report the baseline the kernel actually provides. The
///   platform enables LSX (128-bit vectors) and LASX (256-bit vectors) at boot
///   via `EUEN.SXE`/`EUEN.ASXE`, and the task/signal save paths preserve all 256
///   vector bits. Therefore we set `CPUCFG | LAM | UAL | FPU | LSX | LASX`.
///   This matters for feature-dispatching libraries such as OpenSSL and numpy.
/// - **riscv64**: report the baseline ISA bits expected by Linux-compatible
///   user space (`IMAFDC`).
/// - **x86_64 / aarch64**: 0. x86 uses CPUID; aarch64 ASIMD/NEON is mandatory.
const fn hwcap_value() -> usize {
    #[cfg(target_arch = "loongarch64")]
    {
        // Linux loongarch HWCAP bits (uapi/asm/hwcap.h):
        const HWCAP_LOONGARCH_CPUCFG: usize = 1 << 0;
        const HWCAP_LOONGARCH_LAM: usize = 1 << 1;
        const HWCAP_LOONGARCH_UAL: usize = 1 << 2;
        const HWCAP_LOONGARCH_FPU: usize = 1 << 3;
        const HWCAP_LOONGARCH_LSX: usize = 1 << 4;
        const HWCAP_LOONGARCH_LASX: usize = 1 << 5;
        HWCAP_LOONGARCH_CPUCFG
            | HWCAP_LOONGARCH_LAM
            | HWCAP_LOONGARCH_UAL
            | HWCAP_LOONGARCH_FPU
            | HWCAP_LOONGARCH_LSX
            | HWCAP_LOONGARCH_LASX
    }
    #[cfg(target_arch = "riscv64")]
    {
        RISCV_COMPAT_HWCAP_IMAFDC
    }
    #[cfg(not(any(target_arch = "loongarch64", target_arch = "riscv64")))]
    {
        0
    }
}

struct ElfLoader(LRUCache<ElfCacheEntry, 32>);

type LoadResult = Result<(VirtAddr, Vec<AuxEntry>), Vec<u8>>;

impl ElfLoader {
    const fn new() -> Self {
        Self(LRUCache::new())
    }

    fn load(&mut self, uspace: &mut AddrSpace, loc: Location, fs_ctx: &Mutex<FsContext>) -> AxResult<LoadResult> {
        if !self.0.touch(|e| e.borrow_cache().location().ptr_eq(&loc)) {
            match ElfCacheEntry::load(loc)? {
                Ok(e) => {
                    self.0.insert(e);
                }
                Err(data) => {
                    return Ok(Err(data));
                }
            }
        }

        uspace.clear();
        //用户地址空间映射一个信号返回跳板
        map_trampoline(uspace)?;

        let entry = self.0.front().unwrap();

        //动态链接的处理
        let ldso = if let Some(header) = entry
            .borrow_elf()
            .ph
            .iter()
            .find(|ph| ph.get_type() == Ok(xmas_elf::program::Type::Interp))
        {
            let cache = entry.borrow_cache();
            let mut data = vec![0; header.file_size as usize];
            let read = cache.read_at(&mut data[..], header.offset)?;
            assert_eq!(data.len(), read);

            let ldso = CStr::from_bytes_with_nul(&data)
                .ok()
                .and_then(|cstr| cstr.to_str().ok())
                .ok_or(AxError::InvalidInput)?;
            debug!("Loading dynamic linker: {ldso}");
            Some(ldso.to_owned())
        } else {
            None
        };

        //动态链接的处理
        let (elf, ldso) = if let Some(ldso) = ldso {
            let loc = fs_ctx.lock().resolve(ldso)?;
            if !self.0.touch(|e| e.borrow_cache().location().ptr_eq(&loc)) {
                let e = ElfCacheEntry::load(loc)?.map_err(|_| AxError::InvalidInput)?;
                self.0.insert(e);
            }

            let mut iter = self.0.iter();
            let ldso = iter.next().unwrap();
            let elf = iter.next().unwrap();
            (elf, Some(ldso))
        } else {
            (entry, None)
        };

        //映射用户态地址空间
        let elf = map_elf(uspace, crate::config::USER_SPACE_BASE, elf)?;

        //动态链接的处理
        let ldso = if ldso.is_some() {
            let max_end = uspace
                .areas()
                .map(|area| area.end().as_usize())
                .max()
                .unwrap_or(crate::config::USER_SPACE_BASE);
            let interp_base = (max_end + 0x100000 - 1) & !(0x100000 - 1);
            ldso.map(|elf| map_elf(uspace, interp_base, elf))
                .transpose()?
        } else {
            None
        };

        let entry = VirtAddr::from_usize(
            ldso.as_ref()
                .map_or_else(|| elf.entry(), |ldso| ldso.entry()),
        );
        let has_ldso = ldso.is_some();
        //进程辅助变量auxv的处理
        let mut auxv = elf
            .aux_vector(PAGE_SIZE_4K, ldso.map(|elf| elf.base()))
            .collect::<Vec<_>>();
        auxv.push(AuxEntry::new(AuxType::HWCAP, hwcap_value()));
        auxv.push(AuxEntry::new(AuxType::UID, 0));
        auxv.push(AuxEntry::new(AuxType::EUID, 0));
        auxv.push(AuxEntry::new(AuxType::GID, 0));
        auxv.push(AuxEntry::new(AuxType::EGID, 0));
        auxv.push(AuxEntry::new(AuxType::SECURE, 0));

        debug!(
            "loader: entry={:#x} auxv_len={} has_ldso={} auxv_last_type={}",
            entry.as_usize(),
            auxv.len(),
            has_ldso,
            auxv.last()
                .map(|e| e.get_type() as usize)
                .unwrap_or(usize::MAX),
        );

        Ok(Ok((entry, auxv)))
    }
}

static ELF_LOADER: Mutex<ElfLoader> = Mutex::new(ElfLoader::new());

/// 清空 ELF 缓存。
///
/// 用于在内存泄漏检测时排除干扰。
#[cfg(feature = "memtrack")]
pub fn clear_elf_cache() {
    ELF_LOADER.lock().0.clear();
}

/// 将用户程序加载到用户地址空间。
///
/// 可执行文件由一个已解析的 [`Location`] 标识 —— 调用者负责一次性解析并打开它
/// （对应 Linux 的 `do_open_execat`，在单次查找中遵循 `AT_SYMLINK_NOFOLLOW`），
/// 本函数绝不会根据路径名重新解析主可执行文件。通过 `.sh` 重定向或 `#!` shebang
/// 触达的解释器由本函数按路径解析，这对应 Linux 的 `open_exec(interp)`，合法地
/// 跟随符号链接。
///
/// # 参数
/// - `uspace`：用户程序的地址空间。
/// - `loc`：要加载的已解析可执行文件。
/// - `path`：调用该可执行文件时使用的路径名，用于 `.sh` 重定向以及解释器在
///   `argv` 中接收到的脚本名。
/// - `args`：用户程序的参数。
/// - `envs`：用户程序的环境变量。
///
/// # 返回值
/// - 用户程序的入口点。
/// - 用户程序的栈指针。
pub fn load_user_app(
    uspace: &mut AddrSpace,
    loc: Location,
    path: &str,
    args: &[String],
    envs: &[String],
    fs_ctx: &Mutex<FsContext>,
) -> AxResult<(VirtAddr, VirtAddr, Vec<AuxEntry>)> {
    // `/proc/self/exe` 在 procfs 中可用；busybox 可以通过 `readlink` 读取它，
    // 以便在 ENOEXEC 时重新以 shell 身份执行自身，前提是 busybox 编译时包含了
    // 该回退逻辑（Alpine 的预编译二进制可能未包含）。
    if path.ends_with(".sh") {
        let new_args: Vec<String> = iter::once("/bin/sh".to_owned())
            .chain(args.iter().cloned())
            .collect();
        let sh = fs_ctx.lock().resolve("/bin/sh")?;
        return load_user_app(uspace, sh, "/bin/sh", &new_args, envs, fs_ctx);
    }

    let (entry, auxv) = match { ELF_LOADER.lock().load(uspace, loc, fs_ctx)? } {
        Ok((entry, auxv)) => (entry, auxv),
        //解析以#!开头的文件
        Err(data) => {
            if data.starts_with(b"#!") {
                let head = &data[2..data.len().min(256)];
                let pos = head.iter().position(|c| *c == b'\n').unwrap_or(head.len());
                let line = core::str::from_utf8(&head[..pos]).map_err(|_| AxError::InvalidInput)?;

                let new_args: Vec<String> = line
                    .trim()
                    .splitn(2, |c: char| c.is_ascii_whitespace())
                    .map(|s| s.trim_ascii().to_owned())
                    .chain(iter::once(path.to_owned()))
                    .chain(args.iter().skip(1).cloned())
                    .collect();
                // 按路径打开解释器（对应 Linux 中对 shebang 解释器的
                // `open_exec` 调用），并将其作为新的可执行文件加载。
                let interp = fs_ctx.lock().resolve(&new_args[0])?;
                return load_user_app(uspace, interp, &new_args[0], &new_args, envs, fs_ctx);
            }
            return Err(AxError::InvalidExecutable);
        }
    };

    //USER_STACK_TOP = 0x0400_0000_0000 USER_STACK_SIZE = 0x80_0000 = 8MB
    let ustack_top = VirtAddr::from_usize(crate::config::USER_STACK_TOP);
    let ustack_size = crate::config::USER_STACK_SIZE;
    let ustack_start = ustack_top - ustack_size;
    debug!("Mapping user stack: {ustack_start:#x?} -> {ustack_top:#x?}");

    //为用户进程分配并映射栈空间。
    uspace.map(
        ustack_start,
        ustack_size,
        MappingFlags::READ | MappingFlags::WRITE | MappingFlags::USER,
        false,
        Backend::new_alloc(ustack_start, PageSize::Size4K, "[stack]"),
    )?;

    //从栈顶（ustack_top，高地址）到低地址的布局：
    //sp (0x0400_0000_0000)
    //      ┌──────────────┐
    //      │    argc      │  ← pu***s，最后一个 push
    //      ├──────────────┤
    //      │   argv[0]    │
    //      │   argv[1]    │  ← argv 指针数组
    //      │     ...      │
    //      │    NULL      │
    //      ├──────────────┤
    //      │   envp[0]    │
    //      │   envp[1]    │  ← envp 指针数组
    //      │     ...      │
    //      │    NULL      │
    //      ├──────────────┤
    //      │  auxv[0..n]  │  ← 辅助向量数组（AT_PHDR, AT_ENTRY…）
    //      ├──────────────┤
    //      │  AT_RANDOM   │  → 指向下方随机字节的指针
    //      │  AT_EXECFN   │  → 指向 argv[0] 字符串的指针
    //      │   AT_NULL    │  ← 结束标记（musl 依赖它来停止解析）
    //      ├──────────────┤
    //      │ 对齐填充      │  ← sp 保持 16 字节对齐
    //      ├──────────────┤
    //      │ arg 字符串    │  ← "/bin/ls\0", "-l\0"...
    //      ├──────────────┤
    //      │ env 字符串    │  ← "PATH=/usr/bin\0"...
    //      ├──────────────┤
    //      │ 16字节随机值  │  ← AT_RANDOM 指向这里
    //      └──────────────┘
    //ustack_start (0x03FF_FF80_0000)
    let stack_data = app_stack_region(args, envs, &auxv, ustack_top.into());
    let user_sp = ustack_top - stack_data.len();
    let user_sp_aligned = user_sp.align_down_4k();
    //建立PTE映射
    uspace.populate_area(
        user_sp_aligned,
        (ustack_top - user_sp_aligned).align_up_4k(),
        MappingFlags::READ | MappingFlags::WRITE,
    )?;
    //新进程启动时，sp 寄存器指向 user_sp，_start 执行的第一条指令就能正确读到 argc、argv、envp、auxv。
    uspace.write(user_sp, stack_data.as_slice())?;

    let heap_start = VirtAddr::from_usize(crate::config::USER_HEAP_BASE);
    let heap_size = crate::config::USER_HEAP_SIZE;
    uspace.map(
        heap_start,
        heap_size,
        MappingFlags::READ | MappingFlags::WRITE | MappingFlags::USER,
        true,
        Backend::new_alloc(heap_start, PageSize::Size4K, "[heap]"),
    )?;

    Ok((entry, user_sp, auxv))
}
