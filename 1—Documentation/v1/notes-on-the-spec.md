# Notes on the design document

Where this implementation departs from `docs/design.md`, and why.
Two kinds of entry: deliberate scope decisions, and places where the
document's own reference material would have produced a bug.

## Bugs in the document's reference code

**1. The 7-bit range for relaxed SIMD is wrong.** The document says to
constrain one operand of `i32x4.relaxed_dot_i8x16_i7x16_add_s` to −64..63.
The instruction's actual rule is that the second operand's **top bit must be
clear** — x86 treats that operand as unsigned, ARM as signed, and they agree
exactly on 0..127. Every value in −64..−1 has the top bit set (−64 is 0xC0),
so the document's range is the non-deterministic case it was trying to avoid:
Chrome on an Intel machine and Chrome on an M-series Mac would disagree.

What this implementation does instead: `--relaxed` limits weights to −63..63,
and the kernel adds 64 to each weight byte before the instruction, so the
operand it sees is 1..127. The extra `64 · Σ_k a` that puts into every
accumulator is removed in the epilogue from row sums computed during A
packing. The i16 intermediate x86 uses cannot saturate either, since
`2 · 128 · 127 = 32512 < 32767`.

**2. `saturating_rounding_doubling_high_mul` shifts where gemmlowp divides.**
The document's version ends `((prod + nudge) >> 31)`. gemmlowp divides by
`2^31`, and C division truncates toward zero while `>>` floors. They disagree
by one on every negative tie: `-5 · 0.5` is `-2` in gemmlowp and `-3` with
the shift. The document is emphatic that this path must be bit-exact, and its
own code is not.

**3. The requantization shift cannot express M > 1.** `requantize` takes
`shift: u32` and only shifts right. The residual `Add` in every MobileNet
block produces multipliers above one routinely; with a right-shift-only
implementation those layers quietly truncate toward zero. TFLite's signed
exponent (left shift when positive, right when negative) is what works.

**4. `TensorDesc` is specified as 32 bytes and its fields add up to 36.**
Resolved at 40 bytes, with `nbytes` added so the parser never computes a
size by multiplying `dims` and trusting the result — a packed weight's size
is not `numel` anyway, because N and K are padded.

**5. The node table has no length.** The header's reserved words are unused
while the variable-length node records can only be walked by trusting each
record's own length fields. One reserved word now holds `nodes_bytes`, so a
hostile file cannot walk the parser off the end of the buffer.

**6. `im2col` padding.** Not stated in the document, easy to get wrong, and
silent when wrong: out-of-image taps must be filled with the input's **zero
point**, not with 0. Filling with 0 encodes the real value `−Z_a · S_a` and
puts a systematic error around every border pixel.

## Bugs this implementation had, and the tests that caught them

Recorded because they say something about which tests were worth writing.

- **The SIMD depthwise kernel wrote to the wrong channels.** It stored at
  `out[i]` instead of `out[ch + i]`, so every 16-channel chunk after the
  first landed on top of the first. Invisible for C ≤ 16 — the case every
  small test uses. Caught by a C = 32 case in the per-op suite.
- **A validator requirement that was simply false.** Depthwise bias was
  required to be padded to a multiple of 16 for the SIMD path, which reads 16
  values at a time — but only when `ch + 16 <= C`, so no padding is ever
  needed. It rejected every 8-channel depthwise model.
- **The M = 1 GEMM was slower packed than naive.** Found in the benchmark
  ladder, not the tests: correct, but the 4×8 microkernel spent three
  quarters of its work on padding rows for batch-1 fully-connected layers.
- **The "naive scalar" benchmark baseline was auto-vectorized by LLVM**,
  making the hand-written kernels look 2× less valuable than they are. Fixed
  by building the lower rungs of the ladder with SIMD disabled.

## Deliberate scope decisions

**Threading (M9) is not implemented.** The design document itself ranks it
last, budgets 2–3 weeks, and calls it the milestone most likely to go
sideways. It also needs COOP/COEP headers, which contradicts the
zero-server-dependency premise. Single-threaded is what ships; `detectFeatures()`
reports whether threads *would* be available so the decision can be revisited
with data.

**Transformer support (M10) is not implemented.** `MatMul` (activation ×
activation), `Transpose` and `LayerNorm` have opcodes reserved in the format
and are rejected by both the converter and the runtime with a clear error,
rather than half-implemented.

**No real ImageNet models.** `onnx/models` and the PyTorch hub are Git LFS /
CDN downloads this build environment cannot make. The converter works on any
ONNX graph using the supported operators — `lnn-convert mobilenetv2.onnx -o
mobilenetv2.lnn --calib imagenet.npz` is the command — but no MobileNetV2
accuracy number is claimed here, because none was measured. The trained model
that *is* included (scikit-learn digits) is small but real, and its numbers
are honest.

**Reshape copies instead of aliasing.** The document calls Reshape
"zero-cost, alias". Aliasing means the arena planner must treat both tensors
as one live range, which is a real complication for a kernel that in these
models runs once, on a 256-byte tensor, at the end of the graph. It copies.

**`-Z build-std` in the build script.** Not a divergence from the design so
much as from its environment: `static.rust-lang.org` is unreachable from this
sandbox, so `rustup target add wasm32-unknown-unknown` fails and `core` is
compiled from source instead. `scripts/build.sh` detects which situation it
is in and uses the prebuilt target when it exists. On a normal machine the
documented `rustup target add` path is what runs.

**Two extra wasm features enabled.** `sign-ext` and
`nontrapping-float-to-int`, which rustc emits by default and `wasm-opt`
rejects unless told about them. Both are baseline in every browser that has
simd128 (Chrome 75+, Firefox 62+, Safari 15+), so nothing is lost.

## Additions

- **A 1×8 microkernel** for M = 1, which every batch-1 classifier head hits.
- **`lnn_tensor_ptr`**, the layer-bisection hook the document recommends in
  §17.1 for debugging — dump any intermediate tensor and compare it to the
  reference to find the *first* node that diverges.
- **`lnn_memory_used`**, so "run() allocates nothing" is a testable claim
  rather than an intention. Both test suites assert it.
- **A second, independent implementation of every kernel** in NumPy
  (`converter/lnn_convert/reference.py`), which is what the conformance suite
  compares against. The document suggests golden vectors from ONNX Runtime;
  ORT's quantized kernels use float requantization internally and are not
  bit-exact with gemmlowp by design, so they can validate accuracy but not
  exactness. A reimplementation of the documented arithmetic can do both.
- **A browser test** (`tests/browser_test.mjs`) driving Chromium through
  Playwright: real feature detection, the Cache API offline path, and the
  demo actually classifying a drawn digit.
