# Benchmarks

Measured with `make bench` on:

```
Intel Xeon @ 2.10 GHz, linux/x64
Node 22.22.2 (V8 12.4) — the same engine Chrome runs
```

V8 in Node is the same wasm engine as Chrome's, so these numbers transfer to
Chrome on comparable hardware. A laptop core will be considerably faster than
this shared cloud vCPU; treat the *ratios* as the durable result and the
absolute figures as a floor. Raw output is in `bench/results/latest.json`.

Every rung is checksummed against the others: a kernel that computes the
wrong thing cannot look fast.

## The GEMM optimization ladder

Int8 GOP/s, higher is better. Each column adds one thing to the one before.

| Shape (M×N×K) | naive scalar | + blocking & packing | + simd128 microkernel | + relaxed-simd dot | total |
|---|---|---|---|---|---|
| pointwise 196×256×64 | 3.66 | 6.13 | 19.75 | **22.14** | 6.1× |
| conv 3×3 196×64×576 | 5.73 | 7.57 | 39.74 | **46.96** | 8.2× |
| classifier 1×1000×256 | 5.78 | 6.57 | 21.10 | **42.96** | 7.4× |
| square 256×256×256 | 5.65 | 6.74 | 35.25 | **36.31** | 6.4× |

Reading the table:

**The first two columns come from a build with SIMD disabled entirely.** This
matters more than it sounds. Measured inside the SIMD build, the "naive"
triple loop runs at 8–11 GOP/s — because LLVM auto-vectorizes it. Quoting
that as a scalar baseline would understate what the hand-written kernels buy
by about 2×. The scalar rungs here are genuinely scalar code.

**Blocking and packing alone buy 15–70%,** not the 2–4× the design document
predicts. Two reasons: these matrices are small enough that much of the
working set was already cache-resident, and packing costs a pass over A that
a naive loop never pays. Its real value shows in the next column — the
microkernel can only be written against a packed layout.

**The baseline SIMD microkernel is the big step**, 3–5×, which is where the
design document said the time would go and it was right.

**Relaxed SIMD adds 3–18% on wide-K shapes** and 2× on the classifier. The
instruction does 16 multiply-accumulates where the baseline path needs a
shuffle, two widenings and two `dot` instructions, so its advantage grows
with K and with how little else the loop is doing.

**The classifier row (M=1) needed its own kernel.** The 4×8 microkernel
computes four rows; a batch-1 fully-connected layer has one, so three
quarters of the work was padding. The 1×8 path took that shape from 6.6 to
43 GOP/s. Every classifier head and every MLP layer at batch 1 hits it.

## Whole-model latency

Median of 200 runs after 30 warmup iterations, so V8 has tiered up from
Liftoff to TurboFan.

| Model | build | kernel used | median | p95 |
|---|---|---|---|---|
| digits MLP (8.3 KiB, 5 nodes) | scalar | scalar | 0.0144 ms | 0.020 ms |
| | simd128 | simd128 | 0.0127 ms | 0.021 ms |
| | relaxed | simd128¹ | 0.0111 ms | 0.017 ms |
| MobileNetV2-shaped CNN (15.5 KiB, 15 nodes) | scalar | scalar | 0.414 ms | 0.439 ms |
| | simd128 | simd128 | **0.176 ms** | 0.215 ms |
| | relaxed | relaxed | 0.178 ms | 0.201 ms |

¹ The digits model was converted without `--relaxed`, so its weights are full
8-bit and the relaxed build correctly steps down to the baseline kernel. The
API reports which kernel actually ran, which is the first thing to check when
"SIMD is no faster".

SIMD is worth 2.4× on the CNN and very little on the MLP: that model is five
tiny nodes, where fixed per-node costs dominate the arithmetic.

## Size

| Build | Raw | Gzipped | Design doc target |
|---|---|---|---|
| scalar | 36.1 KB | **14.9 KB** | 25 KB |
| simd128 | 56.9 KB | **19.5 KB** | 45 KB |
| relaxed | 57.5 KB | **19.6 KB** | 52 KB |

All three are well inside budget, and `make size` fails the build if that
stops being true. For comparison, ONNX Runtime Web's wasm binary is several
megabytes — the size argument is the one this project can actually win, which
is why it leads the README.

What keeps it small: `#![no_std]`, zero dependencies, no `wasm-bindgen`, a
bump allocator instead of `dlmalloc`, numeric error codes instead of
formatted messages (so `core::fmt` never links), and lookup tables instead of
transcendental functions.

## Memory

The MobileNet-shaped model at 32×32 input:

| | Bytes |
|---|---|
| Every activation its own buffer | 183.1 KiB |
| Arena with liveness-based reuse | 111.0 KiB |
| Saving | 1.6× |

The design document predicts 3–5× for MobileNetV2 at 224×224. The gap is
model size, not planner quality: this test network has fewer, smaller
activations, so fewer lifetimes overlap in a way that can be exploited. The
planner's correctness — that two tensors live at the same time never share
bytes — is asserted directly in `tests/test_arena.py` rather than inferred
from the ratio.

`run()` allocates nothing, which is checked by reading the allocator's high
water mark before and after twenty inferences, in both the Python and JS
suites.

## What is not measured here

**No comparison against ONNX Runtime Web or TFLite.** Both would have to be
downloaded, and this build environment has no route to their CDNs. The
comparison the design document asks for — same models, same machine, same
browser session — is a script away (`bench/bench.mjs` already measures the
lnn side) but the numbers are not in this repository, so no claim is made
about them.

**No real MobileNetV2 weights**, for the same reason: `onnx/models` is a Git
LFS download this environment cannot make. The benchmark network has
MobileNetV2's topology, op mix and tensor shapes with random weights, which
exercises the kernels and the arena identically but says nothing about
accuracy. The accuracy numbers that *are* reported come from a real trained
model (scikit-learn's digits, 97.1% fp32 → 97.1% int8, 100% top-1 agreement).

**No threading.** Not implemented; see the README.
