#[ax_runtime::hal::cpu::trap::breakpoint_handler]
fn default_breakpoint_handler(_tf: &mut ax_runtime::hal::cpu::TrapFrame) -> bool {
    warn!("unexpected breakpoint trap");
    false
}
