//! Elementwise and data-movement kernels. None of these are hot enough to
//! vectorize by hand; the GEMM and depthwise kernels are 90%+ of runtime.

use crate::quant::{clamp, dequantize_i8, mul_by_quantized_multiplier, quantize_f32, round_half_away};

pub struct AddParams {
    pub left_shift: i32,
    pub m1: i32,
    pub s1: i32,
    pub m2: i32,
    pub s2: i32,
    pub mo: i32,
    pub so: i32,
    pub z1: i32,
    pub z2: i32,
    pub zo: i32,
    pub lo: i32,
    pub hi: i32,
}

/// Quantized add with independent input scales — TFLite's reference
/// algorithm, bit for bit. Both inputs are brought to a common scale with
/// 20 bits of headroom, summed, then rescaled to the output.
///
/// # Safety
/// `a`, `b` and `y` hold `n` elements.
pub unsafe fn add(a: *const i8, b: *const i8, y: *mut i8, n: usize, p: &AddParams) {
    for i in 0..n {
        let x1 = (*a.add(i) as i32 - p.z1) << p.left_shift;
        let x2 = (*b.add(i) as i32 - p.z2) << p.left_shift;
        let s = mul_by_quantized_multiplier(x1, p.m1, p.s1) + mul_by_quantized_multiplier(x2, p.m2, p.s2);
        let v = mul_by_quantized_multiplier(s, p.mo, p.so) + p.zo;
        *y.add(i) = clamp(v, p.lo, p.hi) as i8;
    }
}

/// Quantized multiply. `b` is either the same size as `a`, one value per
/// channel (broadcast over the last axis — the squeeze-excite case), or a
/// single scalar.
///
/// # Safety
/// `a`/`y` hold `n` elements; `b` holds `nb` with `nb ∈ {n, channels, 1}`.
pub unsafe fn mul(a: *const i8, b: *const i8, y: *mut i8, n: usize, nb: usize, z1: i32, z2: i32, zo: i32, mo: i32, so: i32, lo: i32, hi: i32) {
    for i in 0..n {
        let j = if nb == n { i } else if nb == 1 { 0 } else { i % nb };
        let acc = (*a.add(i) as i32 - z1) * (*b.add(j) as i32 - z2);
        let v = mul_by_quantized_multiplier(acc, mo, so) + zo;
        *y.add(i) = clamp(v, lo, hi) as i8;
    }
}

/// # Safety
/// `x` and `y` hold `n` elements.
pub unsafe fn clamp_i8(x: *const i8, y: *mut i8, n: usize, lo: i32, hi: i32) {
    for i in 0..n {
        *y.add(i) = clamp(*x.add(i) as i32, lo, hi) as i8;
    }
}

/// 256-entry lookup table: hard-swish, GELU, sigmoid, tanh (doc 5.4). The
/// converter computes the table for the exact (input, output) quantization
/// pair, so the activation is exact by construction and a byte load at run
/// time.
///
/// # Safety
/// `x`/`y` hold `n` elements; `table` holds 256.
pub unsafe fn lookup(x: *const i8, table: *const i8, y: *mut i8, n: usize) {
    for i in 0..n {
        let idx = (*x.add(i) as i32 + 128) as usize;
        *y.add(i) = *table.add(idx);
    }
}

/// Softmax over the last axis, int8 in, int8 out with the fixed output
/// quantization TFLite uses (scale 1/256, zero point −128).
///
/// `table[j] = exp(S_x · (j − 255))`, precomputed by the converter, so
/// `exp(S_x · (x − max))` is a lookup at index `255 + x − max`. The rest is
/// IEEE f32 arithmetic in a fixed order, which is deterministic across
/// engines and reproduced exactly by the reference implementation.
///
/// # Safety
/// `x`/`y` hold `rows · c` elements; `table` holds 256 f32.
pub unsafe fn softmax(x: *const i8, table: *const f32, y: *mut i8, rows: usize, c: usize) {
    for r in 0..rows {
        let xr = x.add(r * c);
        let yr = y.add(r * c);
        let mut max = -128i32;
        for i in 0..c {
            let v = *xr.add(i) as i32;
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        for i in 0..c {
            sum += *table.add((255 + *xr.add(i) as i32 - max) as usize);
        }
        for i in 0..c {
            let e = *table.add((255 + *xr.add(i) as i32 - max) as usize);
            let q = round_half_away((e / sum) * 256.0) - 128;
            *yr.add(i) = clamp(q, -128, 127) as i8;
        }
    }
}

/// Concatenate along the last axis. Each input is rescaled to the output's
/// quantization with its own multiplier; `mult == 0` means "same
/// quantization, copy".
///
/// # Safety
/// The validator checks every input's outer size matches and the channel
/// counts sum to the output's.
pub unsafe fn concat_input(x: *const i8, y: *mut i8, outer: usize, c_in: usize, c_out: usize, c_off: usize, zin: i32, zout: i32, mult: i32, shift: i32) {
    for o in 0..outer {
        let src = x.add(o * c_in);
        let dst = y.add(o * c_out + c_off);
        if mult == 0 {
            core::ptr::copy_nonoverlapping(src, dst, c_in);
        } else {
            for i in 0..c_in {
                let v = mul_by_quantized_multiplier(*src.add(i) as i32 - zin, mult, shift) + zout;
                *dst.add(i) = clamp(v, -128, 127) as i8;
            }
        }
    }
}

/// # Safety
/// `x` holds `n` f32, `y` `n` i8.
pub unsafe fn quantize(x: *const f32, y: *mut i8, n: usize, scale: f32, zp: i32) {
    for i in 0..n {
        *y.add(i) = quantize_f32(*x.add(i), scale, zp);
    }
}

/// # Safety
/// `x` holds `n` i8, `y` `n` f32.
pub unsafe fn dequantize(x: *const i8, y: *mut f32, n: usize, scale: f32, zp: i32) {
    for i in 0..n {
        *y.add(i) = dequantize_i8(*x.add(i), scale, zp);
    }
}
