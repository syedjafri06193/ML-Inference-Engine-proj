# The `.lnn` format

One file holds everything: the graph, the quantization parameters, the
pre-packed weights and the arena layout. The runtime does no shape inference,
no packing and no allocation planning — all of it happened in the converter.

All integers are little-endian. Every section starts 16-byte aligned so SIMD
loads never straddle a boundary.

```
┌──────────────────────────────────────────────────────┐
│ Header                    64 bytes                   │
├──────────────────────────────────────────────────────┤
│ TensorDesc[n_tensors]     40 bytes each              │
├──────────────────────────────────────────────────────┤
│ NodeDesc[n_nodes]         variable, 4-byte aligned   │
├──────────────────────────────────────────────────────┤
│ Strings                   tensor names, NUL-separated│
├──────────────────────────────────────────────────────┤
│ Weight blob               16-byte aligned            │
└──────────────────────────────────────────────────────┘
```

## Header

| Offset | Field | Notes |
|---|---|---|
| 0 | `magic` u32 | `0x314E4E4C` — "LNN1" |
| 4 | `version` u32 | 1 |
| 8 | `flags` u32 | bit 0: weights pre-packed. bit 1: 7-bit weights |
| 12 | `n_tensors` u32 | |
| 16 | `n_nodes` u32 | |
| 20 | `n_inputs` u32 | |
| 24 | `n_outputs` u32 | |
| 28 | `arena_bytes` u32 | activation arena size |
| 32 | `scratch_bytes` u32 | im2col + GEMM packing scratch |
| 36 | `tensors_off` u32 | |
| 40 | `nodes_off` u32 | |
| 44 | `weights_off` u32 | |
| 48 | `strings_off` u32 | |
| 52 | `nodes_bytes` u32 | length of the node table |
| 56 | reserved | two u32 |

`nodes_bytes` occupies one of the design document's three reserved words.
Node records are variable-length, so without it the table can only be walked
by trusting each record's own length fields — exactly what a parser handed a
hostile file must not do.

## TensorDesc (40 bytes)

| Offset | Field | Notes |
|---|---|---|
| 0 | `dtype` u8 | 0 i8, 1 u8, 2 i32, 3 f32 |
| 1 | `rank` u8 | 1–4 |
| 2 | `kind` u8 | 0 weight, 1 activation, 2 input, 3 output |
| 3 | `n_scales` u8 | 1 per-tensor, C per-channel |
| 4 | `dims` u32×4 | NHWC for rank 4 |
| 20 | `offset` u32 | into the weight blob, or into the arena |
| 24 | `scale_off` u32 | f32 array in the weight blob |
| 28 | `zero_point` i32 | |
| 32 | `name_off` u32 | into the strings section |
| 36 | `nbytes` u32 | bytes actually occupied |

The design document specifies 32 bytes but lists fields that add up to 36.
`nbytes` brings it to 40 and means the parser never has to multiply dims
together and trust the result — a packed weight's size is not `numel` anyway,
since N and K are padded.

Input tensors come first in the table, in order. The runtime checks this so
`lnn_input_ptr(i)` is a table lookup rather than a scan.

## NodeDesc

```
opcode   u16
n_in     u8
n_out    u8
attr_len u16     bytes, always a multiple of 4
pad      u16
inputs   u32 * n_in     tensor indices
outputs  u32 * n_out
attrs    i32 * (attr_len / 4)
```

Nodes are stored in topological order; the runtime executes the table
front to back and never sorts.

## Operators

| Opcode | Op | Attributes (i32 words) |
|---|---|---|
| 0 | `Conv2D` | kh, kw, sh, sw, pad×4, dh, dw, clamp_lo, clamp_hi |
| 1 | `DepthwiseConv2D` | same |
| 2 | `FullyConnected` | clamp_lo, clamp_hi |
| 3 | `MatMul` | *reserved, not implemented* |
| 4 | `Add` | left_shift, m1, s1, m2, s2, mo, so, clamp_lo, clamp_hi |
| 5 | `Mul` | mo, so, clamp_lo, clamp_hi |
| 6 | `MaxPool2D` | kh, kw, sh, sw, pad×4, clamp_lo, clamp_hi |
| 7 | `AvgPool2D` | same |
| 8 | `Softmax` | — (second input is a 256-entry f32 exp table) |
| 9 | `Reshape` | — |
| 10 | `Transpose` | *reserved, not implemented* |
| 11 | `Concat` | per input: mult, shift (0, 0 means "same grid, copy") |
| 12 | `Clamp` | lo, hi |
| 13 | `LookupActivation` | — (second input is a 256-entry i8 table) |
| 14 | `LayerNorm` | *reserved, not implemented* |
| 15 | `Quantize` | — |
| 16 | `Dequantize` | — |

The design document lists Quantize/Dequantize as one opcode; they are two
kernels with no code in common, so they are two opcodes here, and Slice and
Resize keep their reserved numbers.

Convolution and fully-connected nodes take five inputs: activations, packed
weights, bias, per-channel multiplier, per-channel shift. The bias already
contains the zero-point correction (see [quantization.md](quantization.md)).

## Packed weights ("panel layout v1")

N is padded up to a multiple of 8 and K to a multiple of 4, with zeros. For
each 8-column panel, for each group of four k, eight columns of four bytes:

```
panel 0: [c0k0 c0k1 c0k2 c0k3 c1k0 … c7k3][next four k] …
         └────────── 32 bytes = two v128 ──────────┘
```

One layout feeds all three kernels. It is the exact operand shape
`i32x4.relaxed_dot_i8x16_i7x16_add_s` wants, and the baseline kernel gets to
the pair layout `i32x4.dot_i16x8_s` wants with one byte shuffle. A layout per
ISA would make the model file depend on which browser loads it.

Depthwise weights are *not* packed — that kernel reads them directly as
(KH, KW, C).

## Validation

`Model::parse` checks the header, then every tensor (dtype, rank, non-zero
dims, that `numel * elem_bytes <= nbytes`, that the bytes lie inside the
weight blob or the arena, 16-byte alignment, a positive finite scale), then
walks the node table checking every index. `graph::validate` then checks each
node against what its kernel assumes: operand counts, dtypes, shapes,
attribute ranges, and that the declared scratch is large enough.

All of it happens at load. After that the interpreter runs with raw pointer
arithmetic and no checks, which is what makes the hot path fast and a
malformed file safe. `tests/test_robustness.py` mutates and truncates a real
model several thousand ways; every one is rejected or runs without trapping.
