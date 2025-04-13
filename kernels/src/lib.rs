#![no_std]
#![feature(stdarch_nvptx)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]
#![feature(core_intrinsics)]

use core::arch::nvptx;
use core::ops::Add;
use half::bf16;
use nvptx_std::prelude::*;

mod helpers;
use helpers::*;

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

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn layer_norm(
    weight: *const bf16, // (d)
    bias: *const bf16,   // (d)
    epsilon: f32,
    input: *const bf16, // (b, t, d)
    output: *mut bf16,
    B: usize,
    T: usize,
    D: usize,
) {
    let (batch, token) = coords_2d();

    let mut avg = 0.0f32;
    for i in 0..D {
        avg += bf16_to_fp32(unsafe { *input.add(batch * T * D + token * D + i) });
    }
    avg /= D as f32;

    let mut variance = 0.0;
    for i in 0..D {
        let v = bf16_to_fp32(unsafe { *input.add(batch * T * D + token * D + i) });
        variance += unsafe { core::intrinsics::powif32(v - avg, 2) };
    }
    variance /= D as f32;

    let std_dev = unsafe { core::intrinsics::sqrtf32(variance + epsilon) };

    for i in 0..D {
        let v = bf16_to_fp32(unsafe { *input.add(batch * T * D + token * D + i) });
        let w = bf16_to_fp32(unsafe { *weight.add(i) });
        let b = bf16_to_fp32(unsafe { *bias.add(i) });

        let ln = ((v - avg) / std_dev) * w + b;
        unsafe {
            *output.add(batch * T * D + token * D + i) = fp32_to_bf16(ln);
        }
    }
}
