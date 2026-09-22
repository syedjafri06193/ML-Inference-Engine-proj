//! Convolutions (design doc 13.4 and 6.5). NHWC throughout, batch 1.

use super::gemm::{self, Gemm, Isa};
use crate::quant::{clamp, requantize};

#[derive(Clone, Copy)]
pub struct ConvGeom {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub pt: usize,
    pub pl: usize,
    pub dh: usize,
    pub dw: usize,
    pub oh: usize,
    pub ow: usize,
}

impl ConvGeom {
    /// A 1×1, stride-1, unpadded convolution's input already *is* the GEMM's
    /// A matrix: skip im2col entirely. MobileNetV2 is dominated by these
    /// pointwise convolutions, which makes this the cheapest large win in the
    /// runtime.
    pub fn is_pointwise(&self) -> bool {
        self.kh == 1 && self.kw == 1 && self.sh == 1 && self.sw == 1 && self.pt == 0 && self.pl == 0
    }

    pub fn k(&self) -> usize {
        self.kh * self.kw * self.c
    }

    pub fn m(&self) -> usize {
        self.oh * self.ow
    }

    pub fn im2col_bytes(&self) -> usize {
        if self.is_pointwise() {
            0
        } else {
            self.m() * self.k()
        }
    }
}

/// Output extent: floor((in + pads − dilation·(k−1) − 1) / stride) + 1.
pub fn out_extent(input: usize, pad_a: usize, pad_b: usize, k: usize, stride: usize, dil: usize) -> Option<usize> {
    let span = dil.checked_mul(k.checked_sub(1)?)?.checked_add(1)?;
    let padded = input.checked_add(pad_a)?.checked_add(pad_b)?;
    if padded < span || stride == 0 {
        return None;
    }
    Some((padded - span) / stride + 1)
}

/// Lay out every receptive field as a row. Out-of-image taps are filled with
/// the input zero point — the quantized encoding of 0.0 — not with 0, which
/// would encode `−Z_a · S_a` and shift every border pixel.
///
/// # Safety
/// `x` holds H·W·C bytes; `dst` holds M·K bytes.
pub unsafe fn im2col(x: *const i8, g: &ConvGeom, zero_point: i8, dst: *mut i8) {
    let k = g.k();
    for oy in 0..g.oh {
        for ox in 0..g.ow {
            let row = dst.add((oy * g.ow + ox) * k);
            let mut col = 0;
            for ky in 0..g.kh {
                let iy = (oy * g.sh + ky * g.dh) as isize - g.pt as isize;
                for kx in 0..g.kw {
                    let ix = (ox * g.sw + kx * g.dw) as isize - g.pl as isize;
                    let d = row.add(col);
                    if iy >= 0 && (iy as usize) < g.h && ix >= 0 && (ix as usize) < g.w {
                        let s = x.add((iy as usize * g.w + ix as usize) * g.c);
                        core::ptr::copy_nonoverlapping(s, d, g.c);
                    } else {
                        core::ptr::write_bytes(d, zero_point as u8, g.c);
                    }
                    col += g.c;
                }
            }
        }
    }
}

pub struct ConvArgs {
    pub x: *const i8,
    pub w: *const i8,
    pub bias: *const i32,
    pub mult: *const i32,
    pub shift: *const i32,
    pub y: *mut i8,
    pub n: usize,
    pub x_zero_point: i32,
    pub y_zero_point: i32,
    pub lo: i32,
    pub hi: i32,
}

/// Standard convolution: im2col (unless pointwise) then GEMM.
///
/// # Safety
/// Shapes and scratch size are checked by the graph validator.
pub unsafe fn conv2d(g: &ConvGeom, a: &ConvArgs, scratch: *mut u8, isa: Isa) {
    let k = g.k();
    let (a_mat, gemm_scratch) = if g.is_pointwise() {
        (a.x, scratch)
    } else {
        im2col(a.x, g, a.x_zero_point as i8, scratch as *mut i8);
        let used = gemm::round_up(g.im2col_bytes(), 16);
        (scratch as *const i8, scratch.add(used))
    };
    let gm = Gemm {
        m: g.m(),
        n: a.n,
        k,
        a: a_mat,
        lda: k,
        b: a.w,
        bias: a.bias,
        mult: a.mult,
        shift: a.shift,
        out: a.y,
        ldc: a.n,
        zero_point: a.y_zero_point,
        lo: a.lo,
        hi: a.hi,
    };
    gemm::gemm(&gm, gemm_scratch, isa);
}

/// Depthwise convolution, direct (doc 6.5): one filter per channel, so no
/// im2col — it would inflate memory for very little arithmetic. Vectorized
/// across channels, which are contiguous in NHWC.
///
/// `acc[c] = bias[c] + Σ_valid (x − Z_x) · w`. The zero point is subtracted
/// per tap rather than folded into the bias as for GEMM, because the number
/// of in-image taps varies at the borders and a folded constant would be
/// wrong there.
///
/// Weights are (KH, KW, C), unpacked.
///
/// # Safety
/// Shapes are checked by the graph validator.
pub unsafe fn depthwise(g: &ConvGeom, a: &ConvArgs, isa: Isa) {
    let c = g.c;
    for oy in 0..g.oh {
        for ox in 0..g.ow {
            let out = a.y.add((oy * g.ow + ox) * c);
            let mut ch = 0;
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            {
                if isa != Isa::Scalar {
                    while ch + 16 <= c {
                        depthwise16_simd(g, a, oy, ox, ch, out);
                        ch += 16;
                    }
                }
            }
            let _ = isa;
            while ch < c {
                let mut acc = *a.bias.add(ch);
                for ky in 0..g.kh {
                    let iy = (oy * g.sh + ky * g.dh) as isize - g.pt as isize;
                    if iy < 0 || iy as usize >= g.h {
                        continue;
                    }
                    for kx in 0..g.kw {
                        let ix = (ox * g.sw + kx * g.dw) as isize - g.pl as isize;
                        if ix < 0 || ix as usize >= g.w {
                            continue;
                        }
                        let xv = *a.x.add((iy as usize * g.w + ix as usize) * c + ch) as i32;
                        let wv = *a.w.add((ky * g.kw + kx) * c + ch) as i32;
                        acc += (xv - a.x_zero_point) * wv;
                    }
                }
                *out.add(ch) = requantize(acc, *a.mult.add(ch), *a.shift.add(ch), a.y_zero_point, a.lo, a.hi);
                ch += 1;
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn depthwise16_simd(g: &ConvGeom, a: &ConvArgs, oy: usize, ox: usize, ch: usize, out: *mut i8) {
    use core::arch::wasm32::*;
    let c = g.c;
    let zx = i16x8_splat(a.x_zero_point as i16);
    let mut acc0 = v128_load(a.bias.add(ch) as *const v128);
    let mut acc1 = v128_load(a.bias.add(ch + 4) as *const v128);
    let mut acc2 = v128_load(a.bias.add(ch + 8) as *const v128);
    let mut acc3 = v128_load(a.bias.add(ch + 12) as *const v128);
    for ky in 0..g.kh {
        let iy = (oy * g.sh + ky * g.dh) as isize - g.pt as isize;
        if iy < 0 || iy as usize >= g.h {
            continue;
        }
        for kx in 0..g.kw {
            let ix = (ox * g.sw + kx * g.dw) as isize - g.pl as isize;
            if ix < 0 || ix as usize >= g.w {
                continue;
            }
            let xv = v128_load(a.x.add((iy as usize * g.w + ix as usize) * c + ch) as *const v128);
            let wv = v128_load(a.w.add((ky * g.kw + kx) * c + ch) as *const v128);
            // x − Z_x fits in i16 (−255..=255); x·w then needs i32.
            let xl = i16x8_sub(i16x8_extend_low_i8x16(xv), zx);
            let xh = i16x8_sub(i16x8_extend_high_i8x16(xv), zx);
            let wl = i16x8_extend_low_i8x16(wv);
            let wh = i16x8_extend_high_i8x16(wv);
            acc0 = i32x4_add(acc0, i32x4_extmul_low_i16x8(xl, wl));
            acc1 = i32x4_add(acc1, i32x4_extmul_high_i16x8(xl, wl));
            acc2 = i32x4_add(acc2, i32x4_extmul_low_i16x8(xh, wh));
            acc3 = i32x4_add(acc3, i32x4_extmul_high_i16x8(xh, wh));
        }
    }
    let mut lanes = [0i32; 16];
    let p = lanes.as_mut_ptr() as *mut v128;
    v128_store(p, acc0);
    v128_store(p.add(1), acc1);
    v128_store(p.add(2), acc2);
    v128_store(p.add(3), acc3);
    for i in 0..16 {
        let n = ch + i;
        *out.add(n) = requantize(lanes[i], *a.mult.add(n), *a.shift.add(n), a.y_zero_point, a.lo, a.hi);
    }
}

/// Max and average pooling. Input and output share scale and zero point
/// (the converter guarantees it), so neither needs requantization.
///
/// # Safety
/// Shapes are checked by the graph validator.
pub unsafe fn pool(g: &ConvGeom, x: *const i8, y: *mut i8, average: bool, lo: i32, hi: i32) {
    let c = g.c;
    for oy in 0..g.oh {
        for ox in 0..g.ow {
            let out = y.add((oy * g.ow + ox) * c);
            for ch in 0..c {
                let mut sum = 0i32;
                let mut max = i32::MIN;
                let mut count = 0i32;
                for ky in 0..g.kh {
                    let iy = (oy * g.sh + ky) as isize - g.pt as isize;
                    if iy < 0 || iy as usize >= g.h {
                        continue;
                    }
                    for kx in 0..g.kw {
                        let ix = (ox * g.sw + kx) as isize - g.pl as isize;
                        if ix < 0 || ix as usize >= g.w {
                            continue;
                        }
                        let v = *x.add((iy as usize * g.w + ix as usize) * c + ch) as i32;
                        sum += v;
                        if v > max {
                            max = v;
                        }
                        count += 1;
                    }
                }
                let v = if count == 0 {
                    0
                } else if average {
                    // TFLite's rounding: half away from zero.
                    if sum > 0 {
                        (sum + count / 2) / count
                    } else {
                        (sum - count / 2) / count
                    }
                } else {
                    max
                };
                *out.add(ch) = clamp(v, lo, hi) as i8;
            }
        }
    }
}
