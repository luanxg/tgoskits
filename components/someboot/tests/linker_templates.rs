#[path = "../build_support/linker.rs"]
mod linker;

use linker::{LinkerArch, LinkerConfig, render_linker_script, source_paths};

const CONFIG: LinkerConfig = LinkerConfig {
    kernel_load_vaddr: 0xffff_ffff_8000_0000,
    kernel_load_paddr: 0x20_0000,
};

fn assert_common_contract(script: &str) {
    assert!(
        script.find(".head.text").unwrap() < script.find("_stext = .;").unwrap(),
        ".head.text must stay before the main text section"
    );
    assert!(script.contains("__kernel_load_end = .;"));
    assert!(script.contains("__bss_start"));
    assert!(script.contains("__bss_stop = .;"));
    assert!(script.contains("__cpu0_stack_top = .;"));
    assert!(script.contains("__kernel_code_end = .;"));
    assert!(
        !script.contains("${"),
        "all template tokens must be rendered"
    );
}

#[test]
fn renders_common_symbols_for_aarch64() {
    let script = render_linker_script(LinkerArch::Aarch64, CONFIG);
    assert_common_contract(&script);
}

#[test]
fn preserves_arch_specific_linker_contracts() {
    let aarch64 = render_linker_script(LinkerArch::Aarch64, CONFIG);
    assert!(aarch64.contains("OUTPUT_ARCH(aarch64)"));
    assert!(aarch64.contains("_kernel_entry = ABSOLUTE(kernel_entry & 0xffffffffffff);"));
    assert!(aarch64.contains("KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*)))"));
}

#[test]
fn tracks_arch_templates_and_shared_fragments_for_cargo_reruns() {
    let paths = source_paths();

    assert!(paths.contains(&"build_support/linker.rs"));
    assert!(paths.contains(&"src/arch/aarch64/link.ld"));
    assert!(paths.contains(&"src/ld/text.ld"));
    assert!(paths.contains(&"src/ld/bss.ld"));
    assert!(paths.contains(&"src/ld/discard-exit.ld"));
}
