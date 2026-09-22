"""Quantization arithmetic (design doc section 5).

Everything here runs at conversion time. The runtime only ever sees integers
and precomputed fixed-point multipliers.
"""

from __future__ import annotations

import math

import numpy as np

I8_MIN, I8_MAX = -128, 127


def quantize_multiplier(m: float) -> tuple[int, int]:
    """Real multiplier M -> (int32 multiplier, signed shift).

    M = mult / 2^31 * 2^shift, with mult in [2^30, 2^31). TFLite's
    QuantizeMultiplier, including its rounding and its 2^31 overflow fixup.
    """
    if m == 0.0:
        return 0, 0
    if not math.isfinite(m) or m < 0:
        raise ValueError(f"multiplier must be finite and non-negative, got {m}")
    frac, shift = math.frexp(m)  # m == frac * 2^shift, frac in [0.5, 1)
    q = int(math.floor(frac * (1 << 31) + 0.5))
    if q == (1 << 31):
        q //= 2
        shift += 1
    if shift < -31:
        # Underflow: the result would always be zero.
        return 0, 0
    if shift > 30:
        raise ValueError(f"multiplier {m} is too large to represent")
    return q, shift


def choose_affine_qparams(lo: float, hi: float) -> tuple[float, int]:
    """Asymmetric int8 scale and zero point for a value range.

    The range is widened to include zero, because padding, ReLU clamps and
    the zero-point correction all assume 0.0 is exactly representable.
    """
    lo = float(min(lo, 0.0))
    hi = float(max(hi, 0.0))
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    scale = (hi - lo) / 255.0
    zp = int(round(I8_MIN - lo / scale))
    zp = max(I8_MIN, min(I8_MAX, zp))
    return float(np.float32(scale)), zp


def per_channel_weight_scales(w: np.ndarray, axis: int = 0, seven_bit: bool = False) -> np.ndarray:
    """Symmetric per-output-channel scales (doc 5.1).

    Per channel, not per tensor: depthwise channels have wildly different
    ranges and a shared scale destroys accuracy on them (doc 17.2).
    """
    qmax = 63.0 if seven_bit else 127.0
    axes = tuple(i for i in range(w.ndim) if i != axis)
    amax = np.abs(w).max(axis=axes)
    amax = np.maximum(amax, 1e-12)
    return (amax / qmax).astype(np.float32)


def quantize_weights(w: np.ndarray, scales: np.ndarray, axis: int = 0, seven_bit: bool = False) -> np.ndarray:
    """Symmetric quantization, ties away from zero, clamped.

    In 7-bit mode the range is -63..63 rather than the design doc's -64..63:
    the relaxed-SIMD kernel adds 64 to each weight so the instruction's
    second operand stays in 0..127, and -64 would map to 0, which is fine —
    but keeping the magnitude symmetric costs nothing and keeps the
    dequantized grid centred.
    """
    qmin, qmax = (-63, 63) if seven_bit else (I8_MIN, I8_MAX)
    shape = [1] * w.ndim
    shape[axis] = -1
    q = np.sign(w) * np.floor(np.abs(w) / scales.reshape(shape) + 0.5)
    return np.clip(q, qmin, qmax).astype(np.int8)


def quantize_activation(x: np.ndarray, scale: float, zp: int) -> np.ndarray:
    q = np.sign(x) * np.floor(np.abs(x.astype(np.float64) / scale) + 0.5) + zp
    return np.clip(q, I8_MIN, I8_MAX).astype(np.int8)


def dequantize(q: np.ndarray, scale: float, zp: int) -> np.ndarray:
    return (q.astype(np.float32) - zp) * np.float32(scale)


def fold_bias(bias: np.ndarray, x_scale: float, w_scales: np.ndarray) -> np.ndarray:
    """Bias from float into the accumulator's units: bias / (S_a · S_w)."""
    s = np.float64(x_scale) * w_scales.astype(np.float64)
    q = np.sign(bias) * np.floor(np.abs(bias.astype(np.float64) / s) + 0.5)
    return np.clip(q, -(2**31), 2**31 - 1).astype(np.int32)


def zero_point_correction(qw: np.ndarray, x_zero_point: int, axis: int = 0) -> np.ndarray:
    """−Z_a · Σ_k q_b, the correction folded into the bias (doc 5.2).

    With symmetric weights (Z_b = 0) this is the only correction term left,
    and because it depends on nothing but the weights it is computed once
    here instead of in the inner loop.
    """
    axes = tuple(i for i in range(qw.ndim) if i != axis)
    col_sums = qw.astype(np.int64).sum(axis=axes)
    return (-int(x_zero_point) * col_sums).astype(np.int64)


def requant_params(x_scale: float, w_scales: np.ndarray, y_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (multiplier, shift) for M = S_a · S_w / S_c."""
    m = (np.float64(x_scale) * w_scales.astype(np.float64)) / np.float64(y_scale)
    mult = np.zeros(len(m), dtype=np.int32)
    shift = np.zeros(len(m), dtype=np.int32)
    for i, v in enumerate(m):
        q, s = quantize_multiplier(float(v))
        mult[i], shift[i] = q, s
    return mult, shift


def add_params(s1: float, s2: float, so: float) -> dict:
    """TFLite's quantized add: bring both inputs to a common scale with 20
    bits of headroom, sum, then rescale to the output."""
    left_shift = 20
    twice_max = 2.0 * max(s1, s2)
    m1, sh1 = quantize_multiplier(s1 / twice_max)
    m2, sh2 = quantize_multiplier(s2 / twice_max)
    mo, sho = quantize_multiplier(twice_max / ((1 << left_shift) * so))
    return dict(left_shift=left_shift, m1=m1, s1=sh1, m2=m2, s2=sh2, mo=mo, so=sho)


def lut_for(fn, x_scale: float, x_zp: int, y_scale: float, y_zp: int) -> np.ndarray:
    """A 256-entry int8 lookup table for a nonlinearity (doc 5.4).

    Hard-swish and GELU cannot fold into a clamp, and a transcendental in the
    runtime would cost more bytes than the whole model parser. The table is
    exact by construction: every possible int8 input has its answer computed
    here in float64.
    """
    q = np.arange(-128, 128, dtype=np.int32)
    y = fn((q - x_zp).astype(np.float64) * x_scale)
    out = np.sign(y) * np.floor(np.abs(y / y_scale) + 0.5) + y_zp
    return np.clip(out, I8_MIN, I8_MAX).astype(np.int8)


def softmax_exp_table(x_scale: float) -> np.ndarray:
    """`exp(S · (j − 255))` for j in 0..255, so the runtime's softmax is a
    lookup plus a float divide and never needs `exp`."""
    j = np.arange(256, dtype=np.float64)
    return np.exp((j - 255.0) * x_scale).astype(np.float32)
