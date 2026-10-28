//! Fixed-point requantization (design doc section 5.3).
//!
//! This is the gemmlowp / TFLite scheme, reproduced bit-exactly. Every
//! function here has a twin in `converter/lnn_convert/reference.py`, and the
//! conformance suite asserts the two agree on every output of every test
//! model. A one-LSB disagreement is a bug, not noise: it compounds across
//! fifty layers into "97% accurate and I can't find why".

/// (a * b) / 2^31 rounded to nearest, with saturation — gemmlowp's
/// SaturatingRoundingDoublingHighMul.
///
/// The tie rule is gemmlowp's, not a textbook one: positive ties round up,
/// negative ties round toward zero (the negative nudge is one short of a
/// half, and the division truncates). -2.5 becomes -2.
#[inline(always)]
pub fn srdhm(a: i32, b: i32) -> i32 {
    if a == i32::MIN && b == i32::MIN {
        return i32::MAX;
    }
    let prod = (a as i64) * (b as i64);
    let nudge: i64 = if prod >= 0 { 1 << 30 } else { 1 - (1 << 30) };
    // Division, not an arithmetic shift: gemmlowp divides by 2^31, which
    // truncates toward zero. The design doc's `>> 31` floors instead, and
    // disagrees with gemmlowp by one on every negative tie.
    ((prod + nudge) / (1i64 << 31)) as i32
}

/// Round-half-away-from-zero right shift — gemmlowp's RoundingDivideByPOT.
#[inline(always)]
pub fn rounding_shift_right(x: i32, shift: i32) -> i32 {
    if shift <= 0 {
        return x;
    }
    let mask = ((1i64 << shift) - 1) as i32;
    let remainder = x & mask;
    let threshold = (mask >> 1) + ((x < 0) as i32);
    (x >> shift) + ((remainder > threshold) as i32)
}

/// TFLite's MultiplyByQuantizedMultiplier.
///
/// `shift` is a signed exponent: the real multiplier is
/// `mult / 2^31 * 2^shift`. The design doc's sketch only allows a right
/// shift, which cannot express a multiplier above one — and the residual
/// `Add` produces those routinely.
#[inline(always)]
pub fn mul_by_quantized_multiplier(x: i32, mult: i32, shift: i32) -> i32 {
    let left = if shift > 0 { shift } else { 0 };
    let right = if shift > 0 { 0 } else { -shift };
    rounding_shift_right(srdhm(x.wrapping_mul(1i32 << left), mult), right)
}

/// Requantize an int32 accumulator to int8 with a fused activation clamp.
///
/// ReLU and ReLU6 are not separate passes: they are the `lo`/`hi` bounds
/// here (doc 5.4), computed by the converter.
#[inline(always)]
pub fn requantize(acc: i32, mult: i32, shift: i32, zero_point: i32, lo: i32, hi: i32) -> i8 {
    let v = mul_by_quantized_multiplier(acc, mult, shift) + zero_point;
    clamp(v, lo, hi) as i8
}

#[inline(always)]
pub fn clamp(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// f32 → nearest integer, ties away from zero. `f32::round` lives in std
/// (it needs libm), so this is spelled out; valid for |x| < 2^23, which the
/// callers guarantee by clamping first.
#[inline(always)]
pub fn round_half_away(x: f32) -> i32 {
    let t = x as i32; // truncates toward zero
    let frac = x - t as f32;
    if frac >= 0.5 {
        t + 1
    } else if frac <= -0.5 {
        t - 1
    } else {
        t
    }
}

/// fp32 → int8 at the graph boundary.
#[inline(always)]
pub fn quantize_f32(x: f32, scale: f32, zero_point: i32) -> i8 {
    let v = x / scale;
    // NaN compares false everywhere and would fall through to the cast, which
    // saturates it to 0. Pin it to the zero point instead, explicitly.
    if v.is_nan() {
        return clamp(zero_point, -128, 127) as i8;
    }
    // Not NaN past this point, so f32::clamp cannot panic.
    let v = v.clamp(-1.0e6, 1.0e6);
    clamp(round_half_away(v) + zero_point, -128, 127) as i8
}

#[inline(always)]
pub fn dequantize_i8(q: i8, scale: f32, zero_point: i32) -> f32 {
    scale * ((q as i32 - zero_point) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srdhm_matches_gemmlowp_reference_values() {
        assert_eq!(srdhm(1 << 30, 1 << 30), 1 << 29);
        assert_eq!(srdhm(-(1 << 30), 1 << 30), -(1 << 29));
        assert_eq!(srdhm(i32::MIN, i32::MIN), i32::MAX);
        assert_eq!(srdhm(100, 1 << 30), 50);
        assert_eq!(srdhm(3, 1 << 30), 2); // 1.5 -> 2
        assert_eq!(srdhm(1, 1 << 30), 1); // 0.5 -> 1
        assert_eq!(srdhm(-3, 1 << 30), -1); // -1.5 -> -1
        assert_eq!(srdhm(-1, 1 << 30), 0); // -0.5 -> 0
        assert_eq!(srdhm(-7, 1 << 30), -3); // -3.5 -> -3
        assert_eq!(srdhm(-6, 1 << 30), -3); // exact
    }

    #[test]
    fn negative_ties_differ_from_an_arithmetic_shift() {
        // The case the design doc's `>> 31` gets wrong. -5 * 0.5 = -2.5:
        // gemmlowp gives -2, a floor-based shift gives -3.
        assert_eq!(srdhm(-5, 1 << 30), -2);
        assert_eq!(srdhm(5, 1 << 30), 3);
        let prod = (-5i64) * (1i64 << 30);
        let doc_version = ((prod + (1 - (1i64 << 30))) >> 31) as i32;
        assert_eq!(doc_version, -3);
    }

    #[test]
    fn rounding_shift_right_is_half_away_from_zero() {
        assert_eq!(rounding_shift_right(5, 1), 3); // 2.5 -> 3
        assert_eq!(rounding_shift_right(-5, 1), -3); // -2.5 -> -3
        assert_eq!(rounding_shift_right(4, 1), 2);
        assert_eq!(rounding_shift_right(7, 2), 2); // 1.75 -> 2
        assert_eq!(rounding_shift_right(-7, 2), -2);
        assert_eq!(rounding_shift_right(6, 2), 2); // 1.5 -> 2
        assert_eq!(rounding_shift_right(-6, 2), -2); // -1.5 -> -2
        assert_eq!(rounding_shift_right(9, 0), 9);
        assert_eq!(rounding_shift_right(i32::MAX, 31), 1);
    }

    #[test]
    fn multiplier_above_one_uses_a_left_shift() {
        // M = 0.75 * 2^2 = 3.0
        let m = (0.75 * (1u64 << 31) as f64) as i32;
        assert_eq!(mul_by_quantized_multiplier(10, m, 2), 30);
        assert_eq!(mul_by_quantized_multiplier(-10, m, 2), -30);
    }

    #[test]
    fn requantize_clamps() {
        let half = 1 << 30; // 0.5
        assert_eq!(requantize(1000, half, 0, 0, -128, 127), 127);
        assert_eq!(requantize(-1000, half, 0, 0, -128, 127), -128);
        assert_eq!(requantize(-10, half, 0, 3, 3, 127), 3); // fused ReLU
    }

    #[test]
    fn quantize_rounds_half_away_and_saturates() {
        assert_eq!(quantize_f32(0.25, 0.5, 0), 1);
        assert_eq!(quantize_f32(-0.25, 0.5, 0), -1);
        assert_eq!(quantize_f32(1.0e9, 0.1, 0), 127);
        assert_eq!(quantize_f32(-1.0e9, 0.1, 0), -128);
        assert_eq!(quantize_f32(f32::NAN, 0.1, 5), 5);
        assert_eq!(quantize_f32(f32::INFINITY, 0.1, 0), 127);
        assert_eq!(quantize_f32(f32::NEG_INFINITY, 0.1, 0), -128);
        assert_eq!(quantize_f32(0.0, 0.1, -128), -128);
    }
}
