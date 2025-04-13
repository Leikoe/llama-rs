use core::arch::asm;
use half::bf16;

#[inline]
pub fn bf16_to_fp32(x: bf16) -> f32 {
    let out: f32;
    unsafe {
        asm!(
            "cvt.rn.f32.bf16 {out}, {in};",
            out = out(reg32) out,
            in = in(reg16) x.to_bits()
        )
    }
    out
}

#[inline]
pub fn fp32_to_bf16(x: f32) -> bf16 {
    let out: u16;
    unsafe {
        asm!(
            "cvt.rn.bf16.f32 {out}, {in};",
            out = out(reg16) out,
            in = in(reg32) x
        )
    }
    bf16::from_bits(out)
}
