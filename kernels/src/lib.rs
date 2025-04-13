#![no_std]
#![feature(stdarch_nvptx)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use core::arch::nvptx;
use core::ops::Add;
use half::bf16;
use nvptx_std::prelude::*;

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn vec_add(
    a: *const bf16,
    b: *const bf16,
    mut out: *mut bf16,
    N: usize,
) {
    let idx = coords_1d();

    if idx < N {
        *out.wrapping_add(idx) = *a.wrapping_add(idx) + *b.wrapping_add(idx);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn embedding(
    embeddings: *const bf16,         // (V, D)
    token_ids: *const u32,           // (B, T)
    mut token_embeddings: *mut bf16, // (B, T, D)
    b: usize,
    t: usize,
    d: usize,
    v: usize,
) {
    let (batch, token) = coords_2d();

    let token_id = unsafe { token_ids.add(batch * t + token).read() } as usize;

    for i in 0..d {
        let emb = unsafe { embeddings.add(token_id * d + i).read() };
        unsafe {
            token_embeddings
                .add(batch * t * d + token * d + i)
                .write(emb);
        }
    }
}
