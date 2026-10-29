# Quantization

The arithmetic here is the gemmlowp / TFLite scheme, reproduced bit-exactly.
It exists twice — in `crates/lnn/src/quant.rs` and in
`converter/lnn_convert/reference.py` — and the test suite asserts the two
agree on every output of every test model.

## Affine quantization

A real value `r` is stored as an 8-bit integer `q`:

```
r = S · (q − Z)
```

* **Weights** are symmetric (`Z = 0`) and **per output channel**. Per-channel
  matters most on depthwise convolutions, whose channel ranges differ by
  orders of magnitude; one shared scale there costs several points of
  accuracy.
* **Activations** are asymmetric and per tensor, calibrated from data.

The activation range is always widened to include 0.0, because convolution
padding, ReLU clamps and the zero-point correction all assume zero is exactly
representable.

## The integer matrix multiply

For `C = A · B`:

```
q_c = Z_c + M · [ Σ q_a q_b − Z_b Σ q_a − Z_a Σ q_b + K Z_a Z_b ]
```

With symmetric weights `Z_b = 0`, two terms vanish. The third,
`−Z_a · Σ_k q_b`, depends only on the weights, so the converter computes it
and folds it into the bias. What is left in the inner loop is a plain
int8 × int8 → int32 dot product with no offsets, which is the whole point.

Depthwise convolution is the exception: padding is excluded from the sum, so
the number of contributing taps varies at the borders and a folded constant
would be wrong there. That kernel subtracts `Z_x` per tap instead.

## Requantization

`M = S_a · S_b / S_c` is a float, usually in (0, 1). Doing a float multiply
per output element would defeat the purpose, so the converter decomposes it
into `M₀ · 2^e` with `M₀ ∈ [0.5, 1)` stored as an int32 (`M₀ · 2³¹`) and `e`
as a signed shift. At run time:

```rust
fn mul_by_quantized_multiplier(x: i32, mult: i32, shift: i32) -> i32 {
    let left  = max(shift, 0);
    let right = max(-shift, 0);
    rounding_shift_right(srdhm(x << left, mult), right)
}
```

Two details that decide whether your outputs match TFLite's:

**The shift is signed.** The design document's sketch only allows a right
shift, which cannot express `M > 1`. The residual `Add` produces `M > 1`
routinely, so a right-shift-only implementation silently truncates those
layers to zero.

**`srdhm` divides; it does not shift.** gemmlowp computes
`(a*b + nudge) / 2^31`, and C division truncates toward zero. An arithmetic
`>> 31` floors instead, so the two disagree by one on every negative tie:
`-5 · 0.5` is `-2` with the division and `-3` with the shift. One LSB, in
one layer, compounding across fifty.

Rounding rules, exactly:

| Operation | Rule |
|---|---|
| `srdhm` | nearest; positive ties up, negative ties toward zero |
| `rounding_shift_right` | nearest, ties away from zero |
| fp32 → int8 at the boundary | nearest, ties away from zero |
| average pooling | `(sum ± count/2) / count`, away from zero |

## Fused activations

ReLU and ReLU6 are not passes over a tensor; they are the clamp bounds on the
requantization step, computed by the converter as `q(0)` and `q(6)` in the
output's own grid. A separate ReLU pass would be pure memory bandwidth.

Hard-swish, GELU, sigmoid and tanh cannot fold into a clamp, so the converter
emits a **256-entry lookup table** per activation, computed in float64 for
the exact (input, output) quantization pair. The result is exact by
construction and the runtime needs no transcendental functions at all — which
also keeps `core::fmt`-free binary size down.

Softmax works the same way: the converter ships `exp(S_x · (j − 255))` as a
256-entry f32 table, so the kernel is a lookup, a sequential f32 sum and a
divide. Its output uses TFLite's fixed grid (scale 1/256, zero point −128).

## Calibration

> "Calibration matters more than the kernel."

The converter runs the fp32 model over the calibration set with every
intermediate exposed as an output, and takes the 99.99th percentile of each
tensor rather than its absolute range. One outlier activation otherwise
stretches a scale until everything else quantizes into a handful of levels.

If quantized accuracy is bad, the order to check things in is: calibration
data representative → percentile clipping on → per-channel weight scales →
rounding matches gemmlowp → zero-point correction applied → preprocessing
identical to training. The kernels are the last place to look, not the first.

## 7-bit weights

`lnn-convert --relaxed` limits weights to −63..63 so the relaxed-SIMD kernel
can add 64 to every weight byte and keep the instruction's second operand in
0..127, where x86 and ARM agree. The accuracy cost is one bit of weight
precision, which per-channel scales largely absorb; the conformance suite
runs the whole test matrix in both modes.
