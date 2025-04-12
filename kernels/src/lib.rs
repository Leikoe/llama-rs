#![no_std]
#![feature(stdarch_nvptx)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use core::arch::nvptx;
use core::ops::Add;
use nvptx_std::prelude::*;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn vecadd(a: *const f32, b: *const f32, mut out: *mut f32) {
    let idx = coords_1d();
    *out.wrapping_add(idx) = *a.wrapping_add(idx) + *b.wrapping_add(idx);
}
