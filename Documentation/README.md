# ML Inference Engine — Design & Build Guide

**Project:** Lightweight WebAssembly neural network runtime running quantized models entirely in-browser
**Language:** Rust → `wasm32-unknown-unknown`
**Status of this document:** planning + reference

---

## Table of contents

1. [Executive summary and scope](#1-executive-summary-and-scope)
2. [Platform reality check](#2-platform-reality-check)
3. [Competitive landscape and what "winning" means](#3-competitive-landscape-and-what-winning-means)
4. [System architecture](#4-system-architecture)
5. [Quantization math](#5-quantization-math)
6. [The SIMD GEMM microkernel](#6-the-simd-gemm-microkernel)
7. [Model format](#7-model-format)
8. [Memory management](#8-memory-management)
9. [Threading](#9-threading)
10. [Toolchain and environment setup](#10-toolchain-and-environment-setup)
11. [Repository layout](#11-repository-layout)
12. [Milestone ladder](#12-milestone-ladder)
13. [Reference implementations](#13-reference-implementations)
14. [Verification strategy](#14-verification-strategy)
15. [Benchmarking](#15-benchmarking)
16. [Size optimization](#16-size-optimization)
17. [Debugging playbook](#17-debugging-playbook)
18. [Stretch goals](#18-stretch-goals)
19. [References](#19-references)

---

## 1. Executive summary and scope

### The original statement

> Lightweight WASM-compiled neural network runtime that runs quantized models in-browser with zero server dependency.

This is buildable and a genuinely good project. Unlike a hardware project, nothing here is physically impossible. But three constraints will shape it more than you'd expect, and it's worth knowing them before you write code rather than after:

1. **"Zero server dependency" and "multi-threaded" are in direct tension.** WASM threads require `SharedArrayBuffer`, which requires cross-origin isolation, which requires the server to send `COOP` and `COEP` headers. A purely static host that can't set headers (GitHub Pages, a plain S3 bucket, a `file://` URL) cannot enable threads on first load. There's a service-worker workaround, but it's a workaround. See section 2.1.
2. **The instruction you most want doesn't exist everywhere.** The int8 dot-product instruction that makes quantized inference fast lives in Relaxed SIMD, which Chrome and Firefox have but **Safari does not**. You need a baseline-SIMD fallback path, which is roughly 2–3× slower on the inner loop. See section 2.2.
3. **Memory is capped lower than you think.** wasm32 gives a 4 GB address space and browsers in practice allocate less. Memory64 lifts this but **Safari doesn't support it either**, and it costs ~10% throughput. There's no `mmap`, so weights cannot be lazily paged — every byte of the model sits in linear memory. See section 2.3.

None of these kill the project. They define its scope.

### Revised project statement

> A dependency-free WebAssembly inference runtime for int8-quantized convolutional and small transformer models, targeting sub-100 KB gzipped binary size and single-digit-millisecond latency on models under 50 MB, with a hand-written SIMD GEMM kernel and a custom model format converted from ONNX.

### Explicit non-goals

Writing these down now will save you from scope creep later.

- **Not an LLM runtime.** A 1B-parameter int8 model is 1 GB in linear memory with no way to page it. WebLLM and wllama exist and solve a different problem with different techniques (KV cache management, streaming weights, WebGPU). Don't compete there.
- **Not a training framework.** Inference only. No autograd, no optimizers.
- **Not a general ONNX runtime.** ONNX has hundreds of operators. You will support perhaps 20. The converter rejects anything else with a clear error.
- **Not dynamic-shape.** Static shapes resolved at conversion time. This buys you the arena allocator (section 8) and most of your "lightweight" claim.
- **Not faster than ONNX Runtime Web.** See section 3.

### Target model set

Pick these up front and benchmark against them throughout:

| Model | Size (int8) | Why |
|---|---|---|
| MNIST MLP | ~100 KB | First thing that runs end-to-end. Trivially verifiable. |
| MobileNetV2 (1.0, 224) | ~3.5 MB | The classic quantized CNN benchmark. Exercises conv, depthwise conv, pointwise, pooling, residual add. |
| MobileNetV3-Small | ~2.5 MB | Adds hard-swish and squeeze-excite. Good stress test for op fusion. |
| all-MiniLM-L6-v2 | ~22 MB | Small transformer for sentence embeddings. Exercises matmul, layernorm, softmax, GELU. Real-world useful (semantic search in-browser). |
| YOLOv8-nano | ~12 MB | Object detection. Good demo material. |
| Keyword spotting (DS-CNN) | ~50 KB | Tiny, real-time, great for showing low-latency streaming inference. |

---

## 2. Platform reality check

### 2.1 Cross-origin isolation: the "zero server dependency" trap

`SharedArrayBuffer` — and therefore WASM threads — is gated behind **cross-origin isolation**. The page must be served with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Check at runtime with `self.crossOriginIsolated`.

This has consequences that are easy to miss:

- **Static hosts that don't let you set headers can't do it.** GitHub Pages cannot set arbitrary response headers. Neither can a plain S3 bucket without CloudFront. Neither can `file://`.
- **Cross-origin isolation breaks embedding.** Under `require-corp`, every cross-origin subresource (images, fonts, scripts, iframes, analytics) must itself send `Cross-Origin-Resource-Policy: cross-origin` or it will be blocked. If your demo page loads a Google Font or an embedded YouTube video, it will break.
- **The workaround is a service worker.** The `coi-serviceworker` pattern registers a service worker that re-serves the page with the isolation headers injected. It works, but **the first page load is never isolated** — the service worker has to install, then the page reloads. Your user sees a flash and a double-load.
- **`performance.now()` is coarsened without isolation.** As a Spectre mitigation, timer resolution is reduced (to ~100 µs in some browsers) when not cross-origin isolated. This means **benchmarking your runtime requires the same headers threading does.** You can work around it by timing thousands of iterations, but it's a real annoyance.

**Recommended stance:** build single-threaded first and make it good. Treat threads as an opt-in enhancement, detected at runtime, with a clean fallback. Document the header requirement prominently. Your default deployment story — a static file you can open anywhere — stays intact.

### 2.2 SIMD: what you can actually depend on

| Feature | Chrome | Firefox | Safari | Verdict |
|---|---|---|---|---|
| **Fixed-width SIMD (128-bit)** | 91 | 89 | 16.4 | **Baseline Widely Available since Sept 2025. Depend on it.** |
| **Relaxed SIMD** | 114 | 146 | ❌ not shipped | **Optional path only.** |
| **Threads / SharedArrayBuffer** | 74 | 79 | 14.1 | Available, but gated on COOP/COEP (see 2.1) |
| **Memory64** | 133 | 134 | ❌ not shipped | Optional; ~10% slower |
| **Bulk memory ops** | 75 | 79 | 14.1 | Depend on it |

Why Relaxed SIMD matters so much: it contains `i32x4.relaxed_dot_i8x16_i7x16_add_s`, which performs **16 int8 multiply-accumulates into 4 int32 lanes in a single instruction**. This maps directly to x86 `VPDPBUSD` / ARM `SDOT`. It is the single most valuable instruction for quantized inference.

Baseline SIMD has no int8 dot product. The best you can do is:

```
i16x8.extmul_low_i8x16_s   / i16x8.extmul_high_i8x16_s   (widen and multiply)
i32x4.dot_i16x8_s                                         (pairwise multiply-add i16→i32)
```

which needs roughly 3–4× more instructions for the same amount of work.

**The "i7" in the instruction name is not a typo and it matters.** One operand must be restricted to 7 bits (range −64..63) for results to be deterministic across hardware, because x86's `VPDPBUSD` saturates its intermediate i16 accumulation while ARM's `SDOT` doesn't. If you feed full-range int8 to both operands, Chrome on an Intel machine and Chrome on an M-series Mac can produce different numbers. Quantize weights to 7 bits (or use the unsigned-activation / signed-weight arrangement the instruction is designed around) if you take this path.

**Design implication:** write the GEMM inner loop behind a trait or function pointer, compile two builds, and dispatch at runtime. See section 10.5.

### 2.3 Memory

- wasm32 linear memory is a single contiguous `ArrayBuffer`, addressed by 32-bit offsets → **4 GB theoretical ceiling**. Browsers typically fail `memory.grow` well before that; budget **~2 GB practical maximum**, less on mobile, much less on iOS Safari.
- **There is no `mmap`.** Native runtimes memory-map weight files so the OS pages them in on demand and shares them across processes. In the browser, every weight byte occupies linear memory for the lifetime of the model.
- **Loading is a transient 2× spike.** You `fetch()` the model into a JS `ArrayBuffer`, then copy it into WASM linear memory. For a moment both exist. A 500 MB model briefly needs 1 GB. Mitigate by fetching in chunks and copying each chunk in, releasing as you go.
- **`memory.grow` can fail and often does so ungracefully.** Handle the null/trap case and surface a real error message rather than a WASM trap.
- Memory64 lifts the ceiling but is Chrome/Firefox only and costs about 10% throughput on pointer-heavy code. Not worth it for your target model sizes.

**Design implication:** allocate one arena up front, sized from the model header. Never grow memory during inference. If the arena doesn't fit, fail early with a clear message.

### 2.4 The performance ceiling

Rough orders of magnitude on a modern laptop core:

| Backend | int8 throughput | Notes |
|---|---|---|
| Naive scalar WASM | ~1–2 GOP/s | Where you start |
| WASM + baseline SIMD, 1 thread | ~8–15 GOP/s | Where a good hand-written kernel lands |
| WASM + relaxed SIMD, 1 thread | ~25–40 GOP/s | The dot-product instruction earns its keep |
| WASM + SIMD + 4 threads | ~80–150 GOP/s | If you can get cross-origin isolation |
| WebGPU | 1–10 TOP/s | Different universe |
| Native (oneDNN/XNNPACK) | ~100–400 GOP/s/core | What you're ~2–3× away from |

**WASM CPU inference wins in a specific regime:** small models, low per-call latency, instant cold start (no GPU shader compilation, which can take hundreds of milliseconds), universal compatibility, no GPU driver lottery, and predictable performance. For MobileNetV2 at 224×224, a good WASM kernel lands somewhere in the 10–30 ms range, which is real-time. For a 7B LLM, it's hopeless.

Be honest about this in your README. "Faster than ONNX Runtime Web" is a claim you will not be able to make. "Runs MobileNetV2 in 18 ms from a 74 KB runtime with no dependencies" is a claim you can make and that people will find interesting.

---

## 3. Competitive landscape and what "winning" means

You are entering a crowded field. Know it:

| Project | What it is | Why you're not beating it |
|---|---|---|
| **ONNX Runtime Web** | Microsoft's ORT compiled to WASM + WebGPU + WebNN backends | Years of engineering, full ONNX op coverage, multiple execution providers |
| **LiteRT.js** | Google's TFLite runtime in WASM, with CPU/GPU/NPU paths | Same C++ runtime shipped on billions of mobile devices; benchmarks well against ORT |
| **transformers.js** | High-level HF pipeline API on top of ORT Web | Not a runtime — a layer above one. Different product. |
| **TensorFlow.js** | JS/WASM/WebGL/WebGPU backends | Mature, huge model zoo |
| **WebLLM / wllama** | In-browser LLM inference | Different problem (KV cache, streaming, WebGPU) |
| **WebNN API** | Browser-native NN API hitting platform NPUs | W3C Candidate Recommendation; Chrome origin trial only, not in Safari or Firefox. Not production-ready, but worth watching — if it ships broadly it changes the landscape. |

### So what's the point?

Three legitimate answers, and you should pick one and lead with it:

**1. Size.** ORT Web's WASM binary is on the order of several megabytes. If your runtime is 70 KB gzipped and runs MobileNetV2, that's a 50× difference and a genuinely useful artifact for size-sensitive deployments (embedded web views, ad-tech, edge widgets, offline PWAs). This is the strongest differentiator and the easiest to measure.

**2. Understanding.** A from-scratch quantized inference engine with a hand-written SIMD GEMM kernel demonstrates that you understand quantization arithmetic, cache blocking, vectorization, memory arenas, and the WASM platform. That is an extremely strong portfolio signal. Nobody hires you for reimplementing ORT; they hire you for being able to explain, in detail, why your kernel does 4×8 register tiling and what the requantization multiplier is doing.

**3. A narrow thing done exactly right.** "The smallest possible runtime for int8 CNNs" is a defensible niche.

**Lead with size. Back it with understanding. Publish the benchmarks honestly, including where you lose.**

---

## 4. System architecture

### 4.1 Layer diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ Browser / JS                                                     │
│                                                                  │
│  ┌────────────┐   ┌──────────────┐   ┌─────────────────────┐    │
│  │ fetch()    │──▶│ Cache API /  │──▶│ loader.js           │    │
│  │ model.lnn  │   │ OPFS         │   │ instantiateStreaming│    │
│  └────────────┘   └──────────────┘   └──────────┬──────────┘    │
│                                                  │               │
│  ┌───────────────────────────────────────────────▼───────────┐  │
│  │ Public API (TypeScript)                                   │  │
│  │   const m = await Engine.load(url)                        │  │
│  │   const out = m.run({ input: Float32Array })              │  │
│  └───────────────────────────┬───────────────────────────────┘  │
└──────────────────────────────┼──────────────────────────────────┘
                               │  copy in / copy out
┌──────────────────────────────▼──────────────────────────────────┐
│ WASM linear memory                                               │
│                                                                  │
│  ┌────────────────┐  ┌──────────────────────────────────────┐   │
│  │ Model region   │  │ Arena (activations)                  │   │
│  │  - header      │  │  reused across nodes via liveness    │   │
│  │  - tensor descs│  │  analysis computed at load time      │   │
│  │  - node descs  │  └──────────────────────────────────────┘   │
│  │  - weight blob │  ┌──────────────────────────────────────┐   │
│  │    (16B aligned│  │ Scratch (im2col / packed panels)     │   │
│  └────────────────┘  └──────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Interpreter: for node in topo_order { dispatch(node) }    │   │
│  └───────────────────────┬──────────────────────────────────┘   │
│                          │                                       │
│  ┌───────────────────────▼──────────────────────────────────┐   │
│  │ Kernels                                                   │   │
│  │  gemm_s8  ◀── the hot path, 80–90% of runtime            │   │
│  │  conv2d (im2col → gemm) │ depthwise │ pool │ softmax     │   │
│  │  layernorm │ gelu │ add │ concat │ transpose │ requant   │   │
│  └───────────────────────┬──────────────────────────────────┘   │
│                          │                                       │
│  ┌───────────────────────▼──────────────────────────────────┐   │
│  │ ISA dispatch: scalar │ simd128 │ relaxed-simd             │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Core design decisions

**Static shapes, resolved at conversion time.** The Python converter computes every intermediate tensor shape and bakes it into the model file. The runtime never does shape inference. This is what enables the arena allocator and removes an enormous amount of code.

**An interpreter, not a compiler.** The graph is a flat, topologically-sorted array of nodes. Executing a model is a `for` loop with a `match` on opcode. No JIT, no codegen, no graph IR at runtime. This keeps the binary small and the code comprehensible.

**Everything funnels into GEMM.** Conv2d becomes im2col + GEMM. Depthwise conv is its own kernel (im2col would be wasteful). Matmul is GEMM. Fully-connected is GEMM. **Optimize one kernel extremely well and the whole runtime gets fast.** Do not spread effort across twenty kernels.

**Fusion happens in the converter, not the runtime.** Conv→BatchNorm→ReLU6 collapses into a single `Conv` node with a fused activation attribute and folded BN parameters *before* the file is written. The runtime sees one node. This is far simpler than runtime graph rewriting and gets most of the benefit.

**No allocation during inference.** All memory is reserved at load time. `run()` performs zero allocations. This makes latency predictable and eliminates a whole class of bugs.

**Zero dependencies, `#![no_std]`.** No `serde`, no `ndarray`, no `wasm-bindgen`. The model format is parsed by casting byte offsets. JS glue is written by hand. This is where the size claim comes from.

### 4.3 Data flow for one inference

```
JS: Float32Array input
  ↓ copy into WASM memory at known input offset
WASM: quantize fp32 → int8 using the input tensor's (scale, zero_point)
  ↓
for each node in topological order:
    read input tensor pointers from arena/weights
    dispatch kernel
    int8 × int8 → int32 accumulate
    requantize int32 → int8 (fixed-point multiply + shift + clamp)
    write to output tensor pointer in arena
  ↓
WASM: dequantize final int8 → fp32
  ↓ JS reads Float32Array view over WASM memory
JS: Float32Array output
```

---

## 5. Quantization math

This is the intellectual core of the project. Get it exactly right; everything else is plumbing.

### 5.1 Affine quantization

A real value `r` is represented by an 8-bit integer `q`:

```
r = S · (q − Z)
```

where `S` is a positive float **scale** and `Z` is an integer **zero point**. For symmetric quantization (used for weights), `Z = 0`, which removes a lot of work. For asymmetric (used for activations, especially after ReLU where the distribution is one-sided), `Z ≠ 0`.

**Per-tensor vs per-channel:** weights should be quantized **per output channel** — each output channel gets its own scale. This costs one float per channel and dramatically improves accuracy on depthwise convolutions, where channel ranges vary wildly. Activations are per-tensor.

### 5.2 Integer matrix multiply

For `C = A · B` where A is M×K activations and B is K×N weights:

```
r_c[m,n] = Σ_k r_a[m,k] · r_b[k,n]

S_c(q_c[m,n] − Z_c) = Σ_k S_a(q_a[m,k] − Z_a) · S_b(q_b[k,n] − Z_b)

q_c[m,n] = Z_c + (S_a·S_b / S_c) · Σ_k (q_a[m,k] − Z_a)(q_b[k,n] − Z_b)
```

Expand the product to avoid subtracting inside the inner loop:

```
Σ_k (q_a − Z_a)(q_b − Z_b)
  = Σ_k q_a·q_b           ← the int8 GEMM, the only part in the hot loop
  − Z_b · Σ_k q_a         ← row sums of A, computed once per row
  − Z_a · Σ_k q_b         ← column sums of B, precomputed at load time
  + K · Z_a · Z_b         ← a constant
```

**This is the key optimization.** The inner loop is a pure int8×int8→int32 dot product with no offsets. The three correction terms are cheap: column sums of B are computed once when the model loads and folded into the bias; row sums of A are computed in a single pass over the activation before the GEMM.

If you use symmetric weights (`Z_b = 0`), two of the four terms vanish entirely.

### 5.3 Requantization

`M = S_a · S_b / S_c` is a float, typically in (0, 1). Doing a float multiply per output element defeats the purpose of integer inference. Instead decompose:

```
M = M₀ · 2^(−n)      where M₀ ∈ [0.5, 1) and n ≥ 0
```

Store `M₀` as a fixed-point int32 (`M₀ · 2³¹`) and `n` as a shift. This is computed **at conversion time**, once per output channel, and stored in the model file. At runtime:

```rust
/// acc: int32 accumulator
/// mult: int32 fixed-point multiplier (M0 * 2^31)
/// shift: right shift amount
#[inline(always)]
fn requantize(acc: i32, mult: i32, shift: u32, zero_point: i32) -> i8 {
    let x = saturating_rounding_doubling_high_mul(acc, mult);
    let y = rounding_shift_right(x, shift);
    (y + zero_point).clamp(-128, 127) as i8
}

/// Computes round((a * b) / 2^31) with saturation.
#[inline(always)]
fn saturating_rounding_doubling_high_mul(a: i32, b: i32) -> i32 {
    if a == i32::MIN && b == i32::MIN {
        return i32::MAX;
    }
    let prod = (a as i64) * (b as i64);
    let nudge: i64 = if prod >= 0 { 1 << 30 } else { 1 - (1 << 30) };
    ((prod + nudge) >> 31) as i32
}

/// Round-half-away-from-zero right shift.
#[inline(always)]
fn rounding_shift_right(x: i32, shift: u32) -> i32 {
    if shift == 0 { return x; }
    let remainder_mask = (1i32 << shift) - 1;
    let remainder = x & remainder_mask;
    let threshold = (remainder_mask >> 1) + ((x < 0) as i32);
    (x >> shift) + ((remainder > threshold) as i32)
}
```

This is the `gemmlowp` / TFLite requantization scheme. **Reproduce it bit-exactly**, because your differential tests against ONNX Runtime or TFLite will compare integer outputs and a rounding difference of 1 LSB will look like a bug. The rounding rules above are not arbitrary — off-by-one differences here are the single most common source of "my model is 97% correct and I can't find why."

### 5.4 Activation fusion

ReLU and ReLU6 become **clamps on the requantization output**, not separate passes:

```
ReLU:   clamp to [Z_c, 127]
ReLU6:  clamp to [Z_c, quantize(6.0, S_c, Z_c)]
```

Fold these into the requantize step. A separate ReLU pass over a large activation tensor is pure memory bandwidth for no reason.

Hard-swish (MobileNetV3) and GELU (transformers) are nonlinear and can't be folded into a clamp. Implement them as **256-entry lookup tables** indexed by the int8 value. One table per (input scale, output scale) pair, computed at load time. This turns a transcendental function into a byte load and is exact by construction.

---

## 6. The SIMD GEMM microkernel

This is where 80–90% of your runtime goes. Budget accordingly — it deserves more of your time than every other kernel combined.

### 6.1 Structure

Three nested levels:

**Packing.** Before the GEMM, copy panels of A and B into contiguous, SIMD-aligned scratch buffers in the exact order the microkernel will read them. This looks like wasted work but is essential — it converts strided, cache-hostile access into pure sequential reads. B (the weights) can be **pre-packed once at model load time** and stored that way, so you pay this cost zero times per inference.

**Blocking.** Tile the M/N/K loops so the working set fits in cache. You have no `cpuid` in WASM and cannot query cache sizes. Pick conservative constants that work on most machines:

```
MC = 64     // rows of A per block
KC = 256    // depth per block
NC = 256    // columns of B per block
```

Working set per block ≈ `MC·KC + KC·NC` bytes of int8 = 16 KB + 64 KB. Tune empirically, but don't over-tune to your own laptop.

**Microkernel.** The innermost loop computing an MR×NR tile of the output, holding accumulators entirely in v128 registers.

### 6.2 Register tiling

WASM `v128` maps to one physical SIMD register (xmm on x86-64, v registers on ARM64). You have effectively **16 usable registers** on x86-64 before the engine starts spilling.

A 4×8 int32 output tile needs `4 × (8/4) = 8` v128 accumulators, leaving 8 registers for operands and addresses. That's the sweet spot. Don't try 8×8 (16 accumulators, zero registers left) — you'll spill to memory on every iteration and it will be slower than the naive version.

```
MR = 4    // output rows per microkernel call
NR = 8    // output cols per microkernel call
→ 8 × v128 accumulators (i32x4 lanes)
```

### 6.3 Baseline SIMD inner loop

With no int8 dot product available, the sequence per 8 K-elements is: load, sign-extend to i16, extended-multiply, pairwise-add into i32.

```rust
use core::arch::wasm32::*;

/// Multiply-accumulate 8 int8 values from `a` and `b` into an i32x4 accumulator.
/// Uses only baseline simd128.
#[inline(always)]
#[target_feature(enable = "simd128")]
unsafe fn madd_8(acc: v128, a: v128, b: v128) -> v128 {
    // Widen the low 8 int8 lanes of each operand to int16.
    let a16 = i16x8_extend_low_i8x16(a);
    let b16 = i16x8_extend_low_i8x16(b);
    // i32x4.dot_i16x8_s: pairwise multiply-add, 8×i16 → 4×i32
    let prod = i32x4_dot_i16x8(a16, b16);
    i32x4_add(acc, prod)
}
```

Sketch of the microkernel (simplified — real code unrolls K and handles all 8 N-columns):

```rust
#[target_feature(enable = "simd128")]
pub unsafe fn gemm_microkernel_4x8_simd128(
    a_packed: *const i8,   // MR × KC, row-major, 16-byte aligned
    b_packed: *const i8,   // KC × NR, col-major panels, 16-byte aligned
    kc: usize,
    c: *mut i32,           // MR × NR accumulators
    ldc: usize,
) {
    // 8 accumulators: 4 rows × 2 (each v128 holds 4 of the 8 output columns)
    let mut acc = [i32x4_splat(0); 8];

    let mut k = 0;
    while k < kc {
        // Broadcast one int8 from each of the 4 A rows
        let a0 = i8x16_splat(*a_packed.add(0 * kc + k));
        let a1 = i8x16_splat(*a_packed.add(1 * kc + k));
        let a2 = i8x16_splat(*a_packed.add(2 * kc + k));
        let a3 = i8x16_splat(*a_packed.add(3 * kc + k));

        // Load 8 int8 weights for this k (one per output column)
        let bv = v128_load64_zero(b_packed.add(k * 8) as *const u64);

        acc[0] = madd_8(acc[0], a0, bv);
        acc[1] = madd_8(acc[1], a0, bv);
        acc[2] = madd_8(acc[2], a1, bv);
        acc[3] = madd_8(acc[3], a1, bv);
        acc[4] = madd_8(acc[4], a2, bv);
        acc[5] = madd_8(acc[5], a2, bv);
        acc[6] = madd_8(acc[6], a3, bv);
        acc[7] = madd_8(acc[7], a3, bv);

        k += 1;
    }

    for r in 0..4 {
        v128_store(c.add(r * ldc) as *mut v128,          acc[r * 2]);
        v128_store(c.add(r * ldc + 4) as *mut v128,      acc[r * 2 + 1]);
    }
}
```

**This sketch is deliberately unoptimized for clarity.** Real gains come from unrolling K by 4 or 8 so multiple independent accumulator chains hide multiply latency, and from loading 16 A-elements at once rather than splatting one at a time. Expect to rewrite this kernel four or five times. That is normal and is where the actual learning is.

### 6.4 Relaxed SIMD inner loop

When available, the whole thing collapses:

```rust
// i32x4.relaxed_dot_i8x16_i7x16_add_s:
//   16 int8 MACs → 4 int32 lanes, accumulating, in ONE instruction.
acc = i32x4_relaxed_dot_i8x16_i7x16_add(a_vec, b_vec, acc);
```

Two caveats:

1. **Constrain one operand to 7 bits** (range −64..63) or results diverge between x86 and ARM. Quantize weights to 7 bits in the converter when building the relaxed path, and accept the small accuracy cost (usually negligible with per-channel scales).
2. **Rust intrinsic availability.** Relaxed-SIMD intrinsics in `core::arch::wasm32` may still be nightly-gated. Check the current status; if they're unstable, you can emit the instruction via a small hand-written `.wat` module linked in, or gate the whole path behind a nightly feature flag.

### 6.5 Depthwise convolution

Depthwise conv has one filter per channel and does **not** become a GEMM — im2col would blow up memory for a tiny amount of arithmetic. Write a direct kernel that vectorizes across the **channel** dimension, which is naturally contiguous in NHWC layout:

```
for each output (y, x):
    for c in (0..C).step_by(16):
        acc = int32 accumulators for 16 channels
        for ky, kx in kernel:
            acc += load_i8x16(input[y+ky][x+kx][c..c+16])
                 * load_i8x16(weights[ky][kx][c..c+16])
        requantize and store 16 channels
```

**Use NHWC layout throughout.** NCHW makes the channel dimension strided, which is hostile to vectorization for depthwise conv and for the requantization step. The converter should transpose all weights to NHWC at conversion time.

---

## 7. Model format

Don't write an ONNX parser. ONNX is protobuf with 190+ operators, versioning, and dynamic shapes. Define your own format, keep the complexity in Python where iteration is cheap, and make the runtime's job trivial.

### 7.1 Layout

Everything 16-byte aligned so SIMD loads never straddle. All integers little-endian.

```
┌─────────────────────────────────────────────────────────┐
│ Header (64 bytes)                                       │
│   magic        u32   0x314E4E4C  "LNN1"                 │
│   version      u32                                      │
│   flags        u32   bit0: weights pre-packed           │
│                      bit1: 7-bit weights (relaxed path) │
│   n_tensors    u32                                      │
│   n_nodes      u32                                      │
│   n_inputs     u32                                      │
│   n_outputs    u32                                      │
│   arena_bytes  u32   activation arena size              │
│   scratch_bytes u32  im2col / packing scratch           │
│   tensors_off  u32                                      │
│   nodes_off    u32                                      │
│   weights_off  u32                                      │
│   strings_off  u32   tensor names, for debugging        │
│   reserved     [u32; 3]                                 │
├─────────────────────────────────────────────────────────┤
│ TensorDesc[n_tensors]  (32 bytes each)                  │
│   dtype       u8    0=i8 1=u8 2=i32 3=f32               │
│   rank        u8                                        │
│   kind        u8    0=weight 1=activation 2=input 3=out │
│   n_scales    u8    1 = per-tensor, C = per-channel     │
│   dims        [u32; 4]                                  │
│   offset      u32   into weight blob, or arena offset   │
│   scale_off   u32   into weight blob (f32 array)        │
│   zero_point  i32                                       │
│   name_off    u32                                       │
├─────────────────────────────────────────────────────────┤
│ NodeDesc[n_nodes]  (variable, 4-byte aligned)           │
│   opcode      u16                                       │
│   n_in        u8                                        │
│   n_out       u8                                        │
│   attr_len    u16                                       │
│   pad         u16                                       │
│   inputs      [u32; n_in]    tensor indices             │
│   outputs     [u32; n_out]                              │
│   attrs       [u8; attr_len]  op-specific, packed       │
├─────────────────────────────────────────────────────────┤
│ Weight blob  (16-byte aligned, pre-packed for GEMM)     │
│   int8 weights, f32 scale arrays, i32 fixed-point       │
│   multipliers, i32 shifts, i32 folded biases,           │
│   precomputed column sums                               │
└─────────────────────────────────────────────────────────┘
```

Nodes are stored in topological order. The runtime does no sorting.

### 7.2 Operator set

Start with the minimum that runs MobileNetV2, then expand:

| Opcode | Op | Notes |
|---|---|---|
| 0 | `Conv2D` | attrs: stride, pad, dilation, groups, fused activation |
| 1 | `DepthwiseConv2D` | |
| 2 | `FullyConnected` | GEMM |
| 3 | `MatMul` | batched GEMM |
| 4 | `Add` | with requantization (different scales on both inputs) |
| 5 | `Mul` | |
| 6 | `MaxPool2D` | |
| 7 | `AvgPool2D` | including global |
| 8 | `Softmax` | |
| 9 | `Reshape` | zero-cost, alias |
| 10 | `Transpose` | |
| 11 | `Concat` | |
| 12 | `Clamp` | usually fused |
| 13 | `LookupActivation` | 256-entry LUT: hard-swish, GELU, sigmoid, tanh |
| 14 | `LayerNorm` | fp32 internally |
| 15 | `Quantize` / `Dequantize` | boundary ops |
| 16 | `Slice` | |
| 17 | `Resize` | nearest / bilinear, for detection heads |

That's 18 operators. It covers MobileNet, small BERT variants, and YOLO-nano. The converter should raise a clear, actionable error on anything else — `Unsupported op 'InstanceNormalization' at node 'blk3/norm'. Supported ops: ...`.

### 7.3 Converter

The Python converter does all the hard work:

```
ONNX model + calibration dataset
  ↓ onnx.shape_inference — resolve all shapes
  ↓ constant folding
  ↓ fuse Conv+BN+ReLU → single Conv with folded params
  ↓ transpose weights NCHW → NHWC
  ↓ calibrate: run N sample inputs, collect activation min/max
  ↓   (percentile-based, e.g. 99.99%, not absolute min/max —
  ↓    outliers wreck the scale)
  ↓ choose per-channel weight scales, per-tensor activation scales
  ↓ compute M = Sa·Sb/Sc per channel → (mult_i32, shift)
  ↓ precompute column sums of B, fold into bias
  ↓ pre-pack weights into GEMM panel order
  ↓ liveness analysis → arena offsets for every activation
  ↓ emit .lnn
```

Two things worth calling out:

**Calibration matters more than the kernel.** A bad activation scale costs you more accuracy than any amount of kernel cleverness gains you speed. Use percentile clipping (99.99% rather than absolute max) and calibrate on at least a few hundred representative samples. If your quantized MobileNetV2 drops more than ~1% top-1 versus fp32, your calibration is wrong, not your math.

**Pre-packing is free performance.** Reordering weights into the microkernel's exact read order at conversion time removes the B-packing step from every single inference. Do it.

---

## 8. Memory management

### 8.1 Arena allocation with liveness analysis

Activation tensors have well-defined lifetimes: a tensor is live from the node that produces it until the last node that consumes it. Compute this at conversion time and pack tensors into a single arena, reusing space for tensors whose lifetimes don't overlap.

```python
def plan_arena(nodes, tensors):
    # 1. lifetime of every activation
    first_use, last_use = {}, {}
    for i, node in enumerate(nodes):
        for t in node.outputs:
            first_use.setdefault(t, i)
        for t in node.inputs:
            last_use[t] = i

    # 2. greedy best-fit over the interval graph
    free_blocks = []          # (offset, size)
    assignments = {}
    high_water = 0

    for i, node in enumerate(nodes):
        for t in node.outputs:
            if tensors[t].kind != ACTIVATION:
                continue
            size = align16(tensors[t].nbytes)
            blk = best_fit(free_blocks, size)
            if blk is None:
                assignments[t] = high_water
                high_water += size
            else:
                assignments[t] = blk.offset
                consume(free_blocks, blk, size)

        # free anything whose last use was this node
        for t in node.inputs:
            if last_use.get(t) == i and tensors[t].kind == ACTIVATION:
                release(free_blocks, assignments[t], align16(tensors[t].nbytes))

    return assignments, high_water
```

For MobileNetV2 at 224×224 this typically cuts peak activation memory by **3–5×** compared to giving every tensor its own buffer. It's also a satisfying thing to put a number on in your README.

### 8.2 The overall memory picture

```
┌──────────────────────────────────────────┐
│ 0                                        │
│   WASM stack (default 1 MB)              │
├──────────────────────────────────────────┤
│   static data (tiny, no_std)             │
├──────────────────────────────────────────┤
│   Model region — weights + metadata      │  read-only after load
│   sized from the file                    │
├──────────────────────────────────────────┤
│   Arena — activations                    │  arena_bytes from header
│   reused per liveness plan               │
├──────────────────────────────────────────┤
│   Scratch — im2col buffer, A packing     │  scratch_bytes from header
├──────────────────────────────────────────┤
│   (unused)                               │
└──────────────────────────────────────────┘
```

Compute the total at load time from the header, call `memory.grow` **once**, and never again. If the grow fails, return a structured error rather than trapping.

### 8.3 Chunked loading

To avoid the transient 2× spike:

```js
async function loadModel(url, wasm) {
  const res = await fetch(url);
  const total = +res.headers.get('content-length');

  // Reserve WASM memory up front by reading just the header first
  const reader = res.body.getReader();
  let offset = wasm.reserve_model(total);   // returns a WASM pointer
  let written = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // value is a Uint8Array chunk; copy straight into linear memory
    new Uint8Array(wasm.memory.buffer, offset + written, value.length)
      .set(value);
    written += value.length;
    onProgress?.(written / total);
  }
  return wasm.finalize_model(offset, total);
}
```

Note that `wasm.memory.buffer` is **detached and replaced** whenever memory grows. Any `Uint8Array` view you hold becomes unusable. Always re-create views after any operation that might grow memory — this is the single most common WASM/JS integration bug.

### 8.4 Offline caching

"Zero server dependency" really means "works offline after first load." Store the model in the **Cache API** (simplest, works with `fetch` semantics) or **OPFS** (better for large files, synchronous access from workers):

```js
const cache = await caches.open('models-v1');
let res = await cache.match(url);
if (!res) {
  res = await fetch(url);
  await cache.put(url, res.clone());
}
```

Add a service worker for full PWA offline behavior, and version your cache key so model updates invalidate cleanly.

---

## 9. Threading

**Build this last, and treat it as optional.** Re-read section 2.1 before you start.

### 9.1 Design

Partition the GEMM by **row blocks of the output** (the M dimension). Each worker gets a contiguous slice of output rows and reads the entire (shared, read-only) B matrix. No synchronization is needed inside the GEMM — workers write disjoint output regions.

```
Main thread
  ├─ allocate SharedArrayBuffer-backed WASM memory
  ├─ spawn N workers, each instantiating the same module over that memory
  ├─ per node: post {node_idx, m_start, m_end} to each worker
  ├─ Atomics.wait on a completion counter
  └─ proceed to next node
```

Use `Atomics.wait` / `Atomics.notify` on a counter in shared memory rather than `postMessage` round-trips — `postMessage` latency (tens of microseconds) will dominate for small layers.

**Don't parallelize small layers.** Below roughly 64 output rows, thread dispatch overhead exceeds the work. Set a threshold and run single-threaded below it.

### 9.2 Setup

```js
// The shared memory must be declared shared at construction
const memory = new WebAssembly.Memory({
  initial: 256,     // 16 MB
  maximum: 32768,   // 2 GB
  shared: true      // requires cross-origin isolation
});
```

Building for threads with Rust requires `atomics` and `bulk-memory` target features and a nightly toolchain with `-Z build-std`:

```
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128" \
cargo +nightly build --target wasm32-unknown-unknown -Z build-std=std,panic_abort --release
```

This is a meaningful jump in toolchain complexity. It's why threading is milestone M8 and not M2.

### 9.3 Detection and fallback

```js
export function threadingAvailable() {
  return typeof SharedArrayBuffer !== 'undefined'
      && self.crossOriginIsolated === true;
}
```

Ship both builds. Load the threaded one only when both conditions hold. Report which path was taken in your API so users can see why they're getting the speed they're getting.

---

## 10. Toolchain and environment setup

### 10.1 Why Rust

| Option | Verdict |
|---|---|
| **Rust → `wasm32-unknown-unknown`** | **Recommended.** No runtime, no JS glue unless you ask for it, `core::arch::wasm32` SIMD intrinsics are stable, excellent size control, `cargo` needs no configuration ceremony. |
| C/C++ → Emscripten | Better pthread ergonomics and a mature SIMD story, but Emscripten injects a large JS runtime that fights your size goal. Use `-sSTANDALONE_WASM` if you go this route. |
| Zig | Genuinely good at freestanding WASM and very small output. Smaller ecosystem, fewer references when you get stuck. |
| AssemblyScript | Too slow for this. No real SIMD intrinsic story. |
| Hand-written WAT | You will not maintain 5,000 lines of it. |

**Important:** use `wasm32-unknown-unknown`, **not** `wasm32-wasi`. You need no filesystem, no clock, no environment. WASI adds imports the browser must polyfill for nothing.

### 10.2 Install

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add wasm32-unknown-unknown
rustup toolchain install nightly                      # only needed for threads
rustup component add rust-src --toolchain nightly     # only needed for threads

# Binaryen (wasm-opt) — the single most valuable size/speed tool
#   macOS:  brew install binaryen
#   Linux:  download a release from github.com/WebAssembly/binaryen/releases
#   or:     npm i -g binaryen

# WABT (wasm2wat, wasm-objdump) — for inspecting output
#   macOS:  brew install wabt

# twiggy — per-function size attribution
cargo install twiggy

# Python side
python -m venv .venv && source .venv/bin/activate
pip install onnx onnxruntime onnxsim numpy pillow
```

### 10.3 Cargo configuration

```toml
# Cargo.toml
[package]
name = "lnn"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[profile.release]
opt-level = 3          # NOT "z" — this is a numeric kernel, speed wins
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true

[dependencies]
# intentionally empty
```

**On `opt-level`:** the usual advice for small WASM is `opt-level = "z"`. Ignore it here. Your binary is dominated by kernel code where loop unrolling and inlining matter enormously, and `"z"` will cost you 2–3× throughput to save perhaps 15 KB. Use `3` and get your size wins from `wasm-opt`, `no_std`, and dead-code elimination instead.

```toml
# .cargo/config.toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128,+bulk-memory"]
```

### 10.4 Build script

```bash
#!/usr/bin/env bash
# scripts/build.sh
set -euo pipefail

OUT=dist
mkdir -p "$OUT"

build() {
  local name=$1 features=$2
  echo "building $name ..."
  RUSTFLAGS="-C target-feature=$features" \
    cargo build --release --target wasm32-unknown-unknown
  wasm-opt -O3 --enable-simd --enable-bulk-memory \
    ${3:-} \
    target/wasm32-unknown-unknown/release/lnn.wasm \
    -o "$OUT/lnn-$name.wasm"
  gzip -9 -c "$OUT/lnn-$name.wasm" > "$OUT/lnn-$name.wasm.gz"
  printf "  %-10s raw %7s  gz %7s\n" "$name" \
    "$(stat -f%z "$OUT/lnn-$name.wasm" 2>/dev/null || stat -c%s "$OUT/lnn-$name.wasm")" \
    "$(stat -f%z "$OUT/lnn-$name.wasm.gz" 2>/dev/null || stat -c%s "$OUT/lnn-$name.wasm.gz")"
}

build simd128 "+simd128,+bulk-memory"
build relaxed "+simd128,+relaxed-simd,+bulk-memory" "--enable-relaxed-simd"
build scalar  "+bulk-memory"
```

Track those three numbers in CI. A size regression should be as visible as a test failure.

### 10.5 Runtime feature detection

You cannot ask the browser "do you support relaxed SIMD?" There is no API. The standard trick is to **validate a tiny module that uses the instruction**:

```js
// Minimal module containing one relaxed-SIMD instruction.
const RELAXED_PROBE = new Uint8Array([
  0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00,   // magic + version
  // ... type section, function section, code section containing
  //     i32x4.relaxed_dot_i8x16_i7x16_add_s
]);

const SIMD_PROBE = new Uint8Array([ /* module using v128.const */ ]);

export function detectFeatures() {
  return {
    simd:    WebAssembly.validate(SIMD_PROBE),
    relaxed: WebAssembly.validate(RELAXED_PROBE),
    threads: typeof SharedArrayBuffer !== 'undefined' && self.crossOriginIsolated,
  };
}
```

Generate the probe bytes with `wat2wasm` from a two-line `.wat` file and check them into the repo as a hex constant with a comment pointing at the source `.wat`. Don't hand-assemble them.

Then pick the build:

```js
const f = detectFeatures();
const url = f.relaxed ? 'lnn-relaxed.wasm'
          : f.simd    ? 'lnn-simd128.wasm'
          :             'lnn-scalar.wasm';
const { instance } = await WebAssembly.instantiateStreaming(fetch(url), imports);
```

`instantiateStreaming` requires the server to send `Content-Type: application/wasm`. If it doesn't, the call fails with a confusing error — fall back to `WebAssembly.instantiate(await res.arrayBuffer(), imports)`.

---

## 11. Repository layout

```
ML-Inference-Engine/
├── README.md
├── docs/
│   ├── design.md               ← this document
│   ├── model-format.md
│   ├── quantization.md
│   └── benchmarks.md
├── crates/
│   └── lnn/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs           ← extern "C" exports, the entire public ABI
│           ├── model.rs         ← zero-copy .lnn parser
│           ├── graph.rs         ← interpreter loop
│           ├── arena.rs
│           ├── tensor.rs
│           ├── quant.rs         ← requantize, LUT construction
│           ├── kernels/
│           │   ├── mod.rs       ← ISA dispatch
│           │   ├── gemm/
│           │   │   ├── scalar.rs
│           │   │   ├── simd128.rs
│           │   │   ├── relaxed.rs
│           │   │   └── pack.rs
│           │   ├── conv.rs      ← im2col + gemm
│           │   ├── depthwise.rs
│           │   ├── pool.rs
│           │   ├── softmax.rs
│           │   ├── layernorm.rs
│           │   └── elementwise.rs
│           └── tests/
├── js/
│   ├── src/
│   │   ├── index.ts             ← public API
│   │   ├── loader.ts            ← streaming instantiate + chunked model load
│   │   ├── detect.ts            ← feature probes
│   │   ├── cache.ts             ← Cache API / OPFS
│   │   └── worker.ts            ← thread pool
│   ├── package.json
│   └── tsconfig.json
├── converter/
│   ├── lnn_convert/
│   │   ├── __init__.py
│   │   ├── cli.py               ← `lnn-convert model.onnx -o model.lnn`
│   │   ├── fuse.py              ← Conv+BN+ReLU folding
│   │   ├── calibrate.py
│   │   ├── quantize.py
│   │   ├── pack.py              ← GEMM panel pre-packing
│   │   ├── arena.py             ← liveness analysis
│   │   └── writer.py
│   └── pyproject.toml
├── tests/
│   ├── golden/                  ← per-op reference vectors from ORT
│   ├── models/
│   └── conformance.py           ← differential test driver
├── bench/
│   ├── bench.html
│   ├── bench.js
│   └── results/
├── examples/
│   ├── image-classify/
│   ├── semantic-search/
│   └── keyword-spotting/
└── scripts/
    ├── build.sh
    ├── size-report.sh
    └── gen-probes.sh
```

---

## 12. Milestone ladder

Each milestone ends with something demonstrable. Resist the urge to build the fast kernel first — you need a correct slow version to test the fast version against.

---

### M0 — WASM hello world and the JS boundary
**Est. 2–3 days**

Build a Rust `cdylib` exporting `add(a: i32, b: i32) -> i32`. Load it in a browser. Then the important part: allocate a buffer in WASM, write a `Float32Array` into it from JS, have WASM modify it, read it back.

**You are learning:** the `wasm32-unknown-unknown` build, raw `extern "C"` exports without `wasm-bindgen`, linear memory, typed-array views, and the buffer-detachment-on-grow gotcha.

**Done when:** you can round-trip a float array through WASM and you understand exactly where the bytes live.

---

### M1 — Naive fp32 GEMM and a correctness harness
**Est. 3–4 days**

Triple-nested-loop `C = A·B` in fp32. No SIMD, no blocking. Then build the harness you'll use for the rest of the project: generate random matrices in Python, compute the reference with NumPy, run yours, compare.

**Why fp32 first:** you're testing the harness, not the math. Introduce quantization as one variable at a time.

**Done when:** random matrices up to 512×512×512 match NumPy within float tolerance, and the harness reports the first mismatched index with both values.

---

### M2 — Model format, converter, and graph executor (fp32)
**Est. 1.5–2 weeks**

- Define the `.lnn` format (section 7)
- Python writer for a hand-built MLP
- Rust zero-copy parser
- Interpreter loop with `FullyConnected`, `Add`, `ReLU`, `Softmax`
- Train a tiny MNIST MLP in PyTorch, export to ONNX, convert, run

**Done when:** a browser page classifies a hand-drawn digit on a canvas. This is your first end-to-end demo and it's worth taking a screenshot of.

---

### M3 — Quantization
**Est. 1.5–2 weeks**

- Calibration in the converter (percentile-based)
- Per-channel symmetric weights, per-tensor asymmetric activations
- Fixed-point multiplier/shift computation
- The bit-exact requantize path in Rust (section 5.3)
- Zero-point correction terms folded into bias

**Done when:** the quantized MNIST MLP matches the fp32 version's predictions on ≥99% of the test set, and your integer outputs match ONNX Runtime's quantized outputs **bit-exactly** on a per-op basis.

That bit-exactness requirement is not optional and not pedantic. If you accept "close enough" here, every subsequent bug will be ambiguous.

---

### M4 — SIMD GEMM ★ **the heart of the project**
**Est. 3–4 weeks**

Iteratively, benchmarking after every change:

1. Naive int8 scalar GEMM → baseline number
2. Cache blocking (MC/KC/NC) → expect 2–4×
3. Panel packing → expect 1.5–2×
4. 4×8 register-tiled SIMD microkernel → expect 3–5×
5. K-loop unrolling with multiple accumulator chains → expect 1.2–1.5×
6. Pre-packed weights (converter side) → removes B-packing from the hot path

Keep a table of GFLOP/s after each step. **This table is the single best artifact this project produces** — it is concrete evidence that you can optimize numeric code, and it makes a great blog post.

**Done when:** you're within ~3× of native XNNPACK on the same matrix shapes, and you can explain every step of the gap.

---

### M5 — Convolution and MobileNetV2
**Est. 2–3 weeks**

- `im2col` + GEMM for standard conv
- Direct depthwise kernel vectorized over channels
- Max/avg/global pooling
- NHWC layout throughout
- Conv+BN+ReLU6 fusion in the converter
- Residual `Add` with dual-scale requantization

**Done when:** MobileNetV2 classifies images correctly in-browser with top-1 accuracy within 1% of the fp32 ONNX model on a 1000-image ImageNet validation subset.

**Expect calibration to be the problem, not the kernels.** If accuracy is bad, check activation scales before you touch any Rust.

---

### M6 — Arena allocator and memory profiling
**Est. 1 week**

Liveness analysis in the converter, arena offsets in the model file, runtime that never allocates during `run()`.

**Done when:** peak activation memory for MobileNetV2 is reported in the README with a before/after number (expect 3–5× reduction), and `run()` provably allocates zero bytes.

---

### M7 — Packaging, API, and the demo
**Est. 1–2 weeks**

- Clean TypeScript API, typed, documented
- Streaming instantiate + chunked model load with progress
- Cache API for offline
- Three example apps (image classify, semantic search, keyword spotting)
- Deploy to static hosting

**Done when:** someone can `npm install` your package, point it at a `.lnn` file, and get predictions in five lines of code. And your demo works offline on a plane.

---

### M8 — Relaxed SIMD path
**Est. 1 week**

Second build, runtime feature detection, 7-bit weight variant in the converter, dispatch.

**Done when:** Chrome picks the relaxed build and shows a measurable speedup, Safari picks the baseline build and still works, and your benchmark page reports which path it took.

---

### M9 — Threads (optional)
**Est. 2–3 weeks**

Nightly toolchain, shared memory, worker pool, `Atomics`-based sync, M-dimension partitioning, COOP/COEP deployment, service-worker fallback.

**Budget generously.** This is the milestone most likely to go sideways, because the toolchain requirements, the header requirements, and the debugging story are all harder than anything before it.

**Done when:** 4 threads give ≥3× speedup on MobileNetV2 under cross-origin isolation, and the single-threaded fallback is automatic and invisible when isolation isn't available.

---

### M10 — Transformer support
**Est. 2–3 weeks**

LayerNorm, GELU (via LUT), batched MatMul, attention, plus a tokenizer on the JS side.

**Done when:** all-MiniLM-L6-v2 produces sentence embeddings whose cosine similarities match the PyTorch reference to within 0.01, and you have a working in-browser semantic search demo.

---

## 13. Reference implementations

Untested sketches to show intended structure, not drop-in code.

### 13.1 The entire public ABI

```rust
#![no_std]
#![allow(clippy::missing_safety_doc)]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! { core::arch::wasm32::unreachable() }

static mut ENGINE: Option<Engine> = None;

/// Reserve `n` bytes in linear memory and return a pointer.
#[no_mangle]
pub unsafe extern "C" fn lnn_alloc(n: usize) -> *mut u8 { /* bump allocator */ }

/// Parse a .lnn buffer already present in linear memory.
/// Returns 0 on success, negative error code otherwise.
#[no_mangle]
pub unsafe extern "C" fn lnn_load(ptr: *const u8, len: usize) -> i32 { /* ... */ }

/// Pointer to the input tensor's buffer, for JS to write into.
#[no_mangle]
pub unsafe extern "C" fn lnn_input_ptr(index: u32) -> *mut u8 { /* ... */ }

#[no_mangle]
pub unsafe extern "C" fn lnn_input_len(index: u32) -> u32 { /* ... */ }

/// Execute the graph. Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn lnn_run() -> i32 { /* ... */ }

#[no_mangle]
pub unsafe extern "C" fn lnn_output_ptr(index: u32) -> *const u8 { /* ... */ }

#[no_mangle]
pub unsafe extern "C" fn lnn_output_len(index: u32) -> u32 { /* ... */ }

/// Which ISA path was selected, for reporting.
#[no_mangle]
pub extern "C" fn lnn_isa() -> u32 { /* 0=scalar 1=simd128 2=relaxed */ }
```

**Nine exported functions. That's the entire interface.** No `wasm-bindgen`, no generated glue, no JS shim to maintain. The whole API surface fits on a screen and costs almost nothing in binary size.

### 13.2 Zero-copy model parsing

```rust
#[repr(C)]
pub struct Header {
    pub magic: u32,
    pub version: u32,
    pub flags: u32,
    pub n_tensors: u32,
    pub n_nodes: u32,
    pub n_inputs: u32,
    pub n_outputs: u32,
    pub arena_bytes: u32,
    pub scratch_bytes: u32,
    pub tensors_off: u32,
    pub nodes_off: u32,
    pub weights_off: u32,
    pub strings_off: u32,
    pub _reserved: [u32; 3],
}

pub struct Model<'a> {
    pub header: &'a Header,
    pub tensors: &'a [TensorDesc],
    pub weights: &'a [u8],
    pub raw: &'a [u8],
}

impl<'a> Model<'a> {
    pub fn parse(buf: &'a [u8]) -> Result<Self, Error> {
        if buf.len() < core::mem::size_of::<Header>() {
            return Err(Error::Truncated);
        }
        if buf.as_ptr() as usize % 16 != 0 {
            return Err(Error::Misaligned);   // SIMD loads require this
        }
        let header = unsafe { &*(buf.as_ptr() as *const Header) };
        if header.magic != 0x314E_4E4C { return Err(Error::BadMagic); }
        if header.version != 1 { return Err(Error::BadVersion); }

        // Validate every offset before constructing any slice.
        let tensors = unsafe {
            core::slice::from_raw_parts(
                buf.as_ptr().add(header.tensors_off as usize) as *const TensorDesc,
                header.n_tensors as usize,
            )
        };
        Ok(Model { header, tensors, weights: &buf[header.weights_off as usize..], raw: buf })
    }
}
```

**Validate every offset and length in the header against the buffer size before you construct a single slice.** A malformed or truncated model file must produce an error code, not a trap or an out-of-bounds read. Fuzz this parser (section 14.4).

### 13.3 The interpreter loop

```rust
pub fn run(model: &Model, arena: &mut [u8], scratch: &mut [u8]) -> Result<(), Error> {
    let mut cursor = NodeCursor::new(model);
    while let Some(node) = cursor.next() {
        match node.opcode {
            OP_CONV2D           => kernels::conv2d(model, node, arena, scratch)?,
            OP_DEPTHWISE_CONV2D => kernels::depthwise(model, node, arena)?,
            OP_FULLY_CONNECTED  => kernels::fully_connected(model, node, arena, scratch)?,
            OP_MATMUL           => kernels::matmul(model, node, arena, scratch)?,
            OP_ADD              => kernels::add(model, node, arena)?,
            OP_MAXPOOL2D        => kernels::maxpool(model, node, arena)?,
            OP_AVGPOOL2D        => kernels::avgpool(model, node, arena)?,
            OP_SOFTMAX          => kernels::softmax(model, node, arena)?,
            OP_LOOKUP_ACT       => kernels::lookup_activation(model, node, arena)?,
            OP_LAYERNORM        => kernels::layernorm(model, node, arena)?,
            OP_RESHAPE          => { /* alias — no work */ }
            OP_TRANSPOSE        => kernels::transpose(model, node, arena)?,
            OP_CONCAT           => kernels::concat(model, node, arena)?,
            _ => return Err(Error::UnsupportedOp(node.opcode)),
        }
    }
    Ok(())
}
```

That's the whole engine. Everything interesting is in `kernels`.

### 13.4 im2col + GEMM convolution

```rust
pub fn conv2d(m: &Model, node: &Node, arena: &mut [u8], scratch: &mut [u8])
    -> Result<(), Error>
{
    let x  = m.tensor(node.inputs[0]);   // NHWC int8
    let w  = m.tensor(node.inputs[1]);   // pre-packed OHWI int8
    let y  = m.tensor(node.outputs[0]);
    let a: ConvAttrs = node.attrs();

    let k  = a.kh * a.kw * x.c;          // GEMM depth
    let mm = y.h * y.w;                  // GEMM rows (spatial positions)
    let n  = y.c;                        // GEMM cols (output channels)

    // 1x1 convolutions need no im2col — the input is already the matrix.
    let a_mat: *const i8 = if a.kh == 1 && a.kw == 1 && a.stride == 1 {
        x.ptr(arena)
    } else {
        im2col_nhwc(x, &a, scratch);
        scratch.as_ptr() as *const i8
    };

    gemm_s8(
        a_mat, k,                        // A: mm × k
        w.ptr(m.weights), k,             // B: k × n, pre-packed
        y.ptr_mut(arena), n,             // C: mm × n
        mm, n, k,
        m.row_sums_scratch(),            // Σ q_a per row
        w.bias(m.weights),               // bias with col-sum correction folded in
        w.multipliers(m.weights),        // per-channel i32 fixed-point
        w.shifts(m.weights),
        y.zero_point,
        a.act_min, a.act_max,            // fused clamp
    );
    Ok(())
}
```

Note the 1×1 fast path. MobileNetV2 is dominated by 1×1 pointwise convolutions, and skipping im2col for them is a large and nearly free win.

### 13.5 Public TypeScript API

```ts
export interface LoadOptions {
  isa?: 'auto' | 'scalar' | 'simd128' | 'relaxed';
  threads?: number | 'auto';
  cache?: boolean;
  onProgress?: (fraction: number) => void;
}

export interface RunStats {
  readonly latencyMs: number;
  readonly isa: 'scalar' | 'simd128' | 'relaxed';
  readonly threads: number;
}

export class Model {
  static async load(url: string, opts?: LoadOptions): Promise<Model>;

  readonly inputs:  ReadonlyArray<TensorInfo>;
  readonly outputs: ReadonlyArray<TensorInfo>;

  run(inputs: Record<string, Float32Array>): Record<string, Float32Array>;
  runWithStats(inputs: Record<string, Float32Array>): [Record<string, Float32Array>, RunStats];

  dispose(): void;
}
```

Usage should be five lines:

```ts
import { Model } from '@you/lnn';

const model = await Model.load('/models/mobilenetv2.lnn');
const { logits } = model.run({ image: preprocessed });
const top = argmax(logits);
```

---

## 14. Verification strategy

Quantized inference fails **quietly**. A wrong shift produces a model that is 94% accurate instead of 96%, and nothing crashes. You cannot find these bugs by looking at outputs. You need differential testing at the operator level.

### 14.1 Per-operator golden vectors

For every operator, generate reference data from ONNX Runtime:

```python
# tests/gen_golden.py
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto

def golden_conv(name, shape_x, shape_w, **attrs):
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, shape_x, dtype=np.int8)
    w = rng.integers(-127, 128, shape_w, dtype=np.int8)
    # build a single-node QLinearConv graph, run it, save inputs+outputs
    ...
    np.savez(f"tests/golden/{name}.npz", x=x, w=w, y=y, **attrs)
```

Then a Rust test loads each `.npz` (converted to a simple binary at build time) and asserts **exact integer equality**.

> Exact. Not `assert_abs_diff_eq!(a, b, epsilon = 1.0)`. If your integers differ by 1, your rounding is wrong, and that error will compound across 50 layers.

Cover, for each op: the identity/trivial case, edge shapes (1×1 spatial, single channel, K not a multiple of 16), saturation cases (values that push the accumulator toward `i32::MAX`), and negative zero-points.

### 14.2 Whole-model differential tests

```python
# tests/conformance.py
def compare_model(onnx_path, lnn_path, n=100):
    sess = ort.InferenceSession(onnx_path)
    engine = LnnHarness(lnn_path)        # runs the wasm via wasmtime
    max_diff, mismatches = 0, 0
    for i in range(n):
        x = sample_input(i)
        ref = sess.run(None, {'input': x})[0]
        got = engine.run(x)
        max_diff = max(max_diff, np.abs(ref - got).max())
        if ref.argmax() != got.argmax():
            mismatches += 1
    return max_diff, mismatches / n
```

Run this in CI on every commit. Track top-1 agreement as a number that must not regress.

### 14.3 Testing outside the browser

Use `wasmtime` or `wasmer` to run your `.wasm` from Python or a Rust integration test. This lets the whole test suite run in CI without a headless browser, which is dramatically faster and less flaky.

```python
from wasmtime import Store, Module, Instance
store = Store()
module = Module.from_file(store.engine, "dist/lnn-simd128.wasm")
instance = Instance(store, module, [])
```

Then add a small set of real-browser tests with Playwright to catch things `wasmtime` can't: feature detection, memory growth behavior, worker setup, the Cache API path.

### 14.4 Fuzz the parser

The model file is untrusted input. A malformed file must return an error code, never trap or read out of bounds.

```rust
// fuzz/fuzz_targets/parse.rs
#![no_main]
libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let _ = lnn::Model::parse(data);   // must never panic or UB
});
```

Run `cargo fuzz run parse` for a few hours once the format stabilizes. This is quick to set up and catches the entire class of "offset field larger than the buffer" bugs.

### 14.5 Accuracy regression

Keep a 1000-image ImageNet validation subset checked in (or a script that downloads it). Report fp32 top-1, quantized top-1, and the delta. If the delta exceeds 1%, the build fails.

---

## 15. Benchmarking

### 15.1 The timer problem

Without cross-origin isolation, `performance.now()` is coarsened as a Spectre mitigation, potentially to ~100 µs. If your model runs in 15 ms that's tolerable; if you're microbenchmarking a GEMM at 200 µs it's useless.

Options:
- Serve the benchmark page with COOP/COEP to get fine-grained timers
- Time N iterations and divide (do this anyway)
- Report the timer resolution you actually measured, so results are interpretable

### 15.2 Methodology

```js
async function bench(fn, { warmup = 20, iters = 200 } = {}) {
  for (let i = 0; i < warmup; i++) fn();     // let the JIT tier up

  const samples = [];
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    fn();
    samples.push(performance.now() - t0);
  }
  samples.sort((a, b) => a - b);
  return {
    min:    samples[0],
    median: samples[iters >> 1],
    p95:    samples[Math.floor(iters * 0.95)],
    mean:   samples.reduce((a, b) => a + b) / iters,
  };
}
```

**Report median and p95, not mean.** GC pauses and thermal throttling produce outliers that make the mean meaningless.

**Warm up properly.** WASM is compiled in tiers — Liftoff (fast compile, slow code) then TurboFan (slow compile, fast code) in V8. The first several calls run unoptimized. Twenty warmup iterations is a reasonable minimum.

### 15.3 What to report

Compare against ORT Web on identical models and inputs, on the same machine, in the same browser session:

| | Runtime size (gz) | MobileNetV2 median | p95 | Peak memory |
|---|---|---|---|---|
| lnn (scalar) | | | | |
| lnn (simd128) | | | | |
| lnn (relaxed) | | | | |
| lnn (simd128, 4t) | | | | |
| ONNX Runtime Web (wasm) | | | | |
| ONNX Runtime Web (webgpu) | | | | |

Report machine, browser version, and whether cross-origin isolation was active. **Publish the losses too.** ORT-WebGPU will beat you by an order of magnitude on throughput; showing that you know this and explaining where your runtime wins anyway (size, cold start, compatibility) is far more credible than a table where you happen to win every row.

Also worth measuring and reporting, because it's where you genuinely win: **time-to-first-inference**, cold, including fetch, compile, and instantiate. WebGPU backends pay a large shader-compilation cost on first run. A 70 KB WASM module streams and compiles in a few milliseconds.

---

## 16. Size optimization

"Lightweight" is in the project title, so it has to be measured and defended.

### 16.1 Where the bytes go

```bash
twiggy top -n 30 dist/lnn-simd128.wasm
twiggy dominators dist/lnn-simd128.wasm
```

Usual culprits in a Rust WASM binary:

| Culprit | Fix |
|---|---|
| `core::fmt` machinery | Never use `format!`, `panic!("{}", x)`, or `Debug` in shipped code. Numeric error codes only. |
| Panic infrastructure | `panic = "abort"` + a `#[panic_handler]` that calls `unreachable()` |
| The default allocator (`dlmalloc`) | Use a bump allocator over the arena. You allocate at load time only. |
| Slice bounds checks | Not worth removing broadly; use `get_unchecked` only in the profiled hot loop, with a safe version behind a debug feature |
| Monomorphization bloat | Watch generic kernels instantiated over many types. Prefer a small number of concrete `i8`/`i32` functions. |
| `wasm-bindgen` glue | Don't use it. Raw `extern "C"` exports. |

### 16.2 Targets

| Build | Raw | Gzipped |
|---|---|---|
| scalar | ~60 KB | ~25 KB |
| simd128 | ~110 KB | ~45 KB |
| simd128 + relaxed | ~130 KB | ~52 KB |
| threaded | ~160 KB | ~65 KB |

These are achievable with `no_std`, no dependencies, and `wasm-opt -O3`. If you're above 300 KB gzipped, run `twiggy` and find out why — it's almost always `core::fmt` sneaking in through a `Debug` impl or an `unwrap()` with a formatted message.

### 16.3 Track it in CI

```yaml
- name: size report
  run: |
    ./scripts/build.sh
    ./scripts/size-report.sh --fail-over simd128=48000 relaxed=56000
```

Fail the build on regression. Size is a feature here; treat it like one.

---

## 17. Debugging playbook

### 17.1 Model output is wrong

Bisect **by layer**, not by guessing.

1. Add a debug export that dumps any intermediate tensor by index into a known buffer.
2. In Python, run the ONNX model with all intermediate outputs exposed (`onnx.utils.extract_model` or add every value_info to the graph outputs).
3. Compare layer by layer, first to last. **The first layer that diverges is your bug.** Everything after it is downstream noise.

This takes an hour to build and will save you weeks. Build it during M3, before you need it.

### 17.2 Quantized accuracy is bad but nothing is "wrong"

Checklist, in order of likelihood:

| Check | Why |
|---|---|
| Calibration data representative? | Calibrating on 10 images, or on the wrong preprocessing, is the #1 cause |
| Using percentile clipping, not absolute min/max? | One outlier activation destroys the scale for the whole tensor |
| Per-channel weight scales on depthwise convs? | Depthwise channels have wildly different ranges; per-tensor scaling murders accuracy here |
| Rounding matches TFLite/gemmlowp exactly? | Round-half-away-from-zero, not round-half-to-even |
| Zero-point correction terms applied? | Missing the `−Z_a·Σq_b` term gives a plausible-looking but systematically shifted output |
| First and last layers quantized? | These are the most sensitive; many pipelines keep them in fp32 or int16 |
| Input preprocessing identical to training? | Mean/std normalization mismatches look exactly like a quantization bug |

### 17.3 WASM traps

An unhandled trap surfaces as `RuntimeError: unreachable executed` with a useless stack. To get something usable:

- Build a debug variant **with** `core::fmt` and a real panic handler that writes a message into a buffer JS can read. Ship the stripped version; debug with the fat one.
- Build with `debug = true` and use `wasm2wat` to map the failing offset back to a function.
- Chrome DevTools supports DWARF in WASM via the C/C++ DevTools Support extension; it works for Rust too and gives real source-level stepping.
- Most traps are (a) out-of-bounds linear memory access from a bad offset in the model file, (b) unaligned SIMD load, (c) integer divide by zero in shape arithmetic.

### 17.4 Memory errors

| Symptom | Cause |
|---|---|
| `RangeError: Array buffer allocation failed` | `memory.grow` failed. Model too large for this browser/device. |
| `TypeError: Cannot perform ... on a detached ArrayBuffer` | You held a typed-array view across a memory growth. Re-create views after any call that can grow memory. |
| Silent wrong results, only with large models | Integer overflow in offset arithmetic. Use `usize` carefully and validate offsets at parse time. |
| Works in Chrome, fails on iOS Safari | Much lower memory ceiling. Test on real iOS hardware, not the simulator. |

### 17.5 "It's slow and I don't know why"

1. **Confirm which build loaded.** Call `lnn_isa()`. A surprising amount of "SIMD is no faster" turns out to be the scalar build being served.
2. **Profile in Chrome DevTools.** The Performance panel attributes time to WASM functions by name if you don't strip.
3. **Check you're not reallocating.** Any allocation inside `run()` is a bug.
4. **Check tier-up.** If you're benchmarking 5 iterations, you're measuring Liftoff, not TurboFan.
5. **Check alignment.** Unaligned SIMD loads are legal in WASM but can be slower. Assert 16-byte alignment on all packed panels.
6. **Isolate the GEMM.** Benchmark the kernel alone on the exact shapes the model uses. If the kernel is fast but the model is slow, you're losing time in im2col, packing, or layout conversion.

---

## 18. Stretch goals

| Feature | Effort | Value |
|---|---|---|
| **int4 weight quantization** | Medium | 2× smaller models; needs unpacking in the kernel |
| **WebGPU backend** | Large | An order of magnitude faster on big models. A whole second project, but shares the converter and model format. |
| **WebNN backend** | Medium | Chrome origin trial only today, and not in Safari or Firefox — but if it reaches Baseline it hits platform NPUs at very low power. Worth watching. |
| **Sparse weights** | Medium | Structured 2:4 sparsity pairs well with the dot-product instruction |
| **Winograd convolution** | Large | Meaningful speedup for 3×3 convs; nontrivial with quantization |
| **Streaming inference** | Small | Ring-buffered audio input for real-time keyword spotting. Great demo. |
| **On-device fine-tuning** | Large | Last-layer-only gradient descent in the browser. Very eye-catching. |
| **Node.js target** | Small | Same `.wasm`, different loader. Nearly free. |
| **Model zoo** | Small | Pre-converted `.lnn` files on a CDN. Massively lowers the barrier for anyone trying your runtime. |

---

## 19. References

### Specifications

| Document | For |
|---|---|
| WebAssembly Core Specification | The base instruction set |
| WebAssembly SIMD proposal | The `v128` instruction list. Keep this open constantly while writing kernels. |
| Relaxed SIMD proposal | `relaxed_dot` semantics and the i7 determinism rule |
| WebAssembly Threads proposal | Atomics, shared memory |
| MDN: `WebAssembly.Memory`, `SharedArrayBuffer`, cross-origin isolation | The COOP/COEP rules |
| W3C Web Neural Network API | Where browser-native ML is heading |

### Papers and technical writing

- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (2017) — the foundational paper for the scheme in section 5
- Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference: A whitepaper* (2018) — practical per-channel vs per-tensor guidance
- Goto & van de Geijn, *Anatomy of High-Performance Matrix Multiplication* — where the packing/blocking/microkernel structure comes from. Read this before writing M4.
- The BLIS papers on microkernel design

### Code worth reading

- **gemmlowp** — Google's low-precision GEMM. The requantization code in `fixedpoint.h` is what section 5.3 reproduces. Read it.
- **XNNPACK** — Google's optimized NN kernels, including a WASM SIMD backend. This is the reference implementation of what you're building. Read the WASM microkernels specifically.
- **ruy** — TFLite's GEMM library, more readable than XNNPACK
- **ONNX Runtime Web** — for the JS-side architecture and feature detection patterns
- **wasm-tools / binaryen** — when you need to understand what `wasm-opt` did to your code

### Tools

`wasm-opt` · `twiggy` · `wasm2wat` · `wasm-objdump` · `cargo-fuzz` · `wasmtime` · `onnxsim` · Chrome DevTools C/C++ extension (works for Rust DWARF)

---

## Appendix A — Decision record

| Decision | Rationale |
|---|---|
| Single-threaded by default, threads opt-in | `SharedArrayBuffer` needs COOP/COEP, which needs server headers, which contradicts "zero server dependency." Service-worker workaround leaves the first load unisolated. |
| Baseline SIMD is the primary path | Relaxed SIMD has the int8 dot product but isn't in Safari. Fixed-width SIMD is Baseline Widely Available since Sept 2025. |
| Two builds + runtime probe | No API exists to query WASM features; `WebAssembly.validate` on a probe module is the standard trick. |
| Target models under ~50 MB | No `mmap` in the browser; every weight byte lives in linear memory. wasm32 caps at 4 GB, practically ~2 GB, less on iOS. Memory64 isn't in Safari and costs ~10%. |
| Not an LLM runtime | A 1B int8 model is 1 GB resident with no paging. Different problem, well served by WebLLM/wllama. |
| Custom model format, not ONNX | ONNX is protobuf with 190+ ops, versioning, and dynamic shapes. Keep that complexity in Python. |
| Static shapes only | Enables the arena allocator and removes shape inference from the runtime. Most of the "lightweight" claim comes from this. |
| Everything funnels into one GEMM | 80–90% of runtime. Optimizing one kernel well beats optimizing twenty kernels adequately. |
| Fusion in the converter, not the runtime | Conv+BN+ReLU folding at conversion time gets most of the benefit for a fraction of the runtime code. |
| NHWC layout | Makes the channel dimension contiguous, which vectorizes depthwise conv and requantization. NCHW does not. |
| `opt-level = 3`, not `"z"` | Conventional WASM size advice is wrong for numeric kernels. Get size from `no_std` + `wasm-opt`, not from de-optimizing the hot loop. |
| Raw `extern "C"`, no `wasm-bindgen` | Nine exported functions. Bindgen glue would cost more bytes than the entire model parser. |
| Bit-exact differential tests | Quantization bugs are silent. "Close enough" makes every later bug ambiguous. |
| Lead the README with size, not speed | ORT Web and LiteRT.js will beat you on throughput. A 50× size difference is real, measurable, and defensible. |

---

## Appendix B — Quick reference card

```
Build targets:     wasm32-unknown-unknown   (NOT wasi)
Rust flags:        -C target-feature=+simd128,+bulk-memory
Profile:           opt-level=3, lto=fat, codegen-units=1,
                   panic=abort, strip=true
Post-process:      wasm-opt -O3 --enable-simd --enable-bulk-memory

Feature support:
  simd128          Chrome 91  Firefox 89   Safari 16.4   ← Baseline, depend on it
  relaxed-simd     Chrome 114 Firefox 146  Safari  ✗     ← optional path
  threads/SAB      Chrome 74  Firefox 79   Safari 14.1   ← needs COOP/COEP
  memory64         Chrome 133 Firefox 134  Safari  ✗     ← skip

Cross-origin isolation headers:
  Cross-Origin-Opener-Policy:   same-origin
  Cross-Origin-Embedder-Policy: require-corp
  check: self.crossOriginIsolated

Quantization:
  r = S(q − Z)
  q_c = Z_c + M · [ Σ q_a q_b − Z_b Σ q_a − Z_a Σ q_b + K Z_a Z_b ]
  M = S_a S_b / S_c = M₀ · 2^(−n),  M₀ ∈ [0.5,1) stored as M₀·2³¹

GEMM tiling:       MR=4, NR=8  → 8 × v128 accumulators
Cache blocks:      MC=64, KC=256, NC=256
Alignment:         16 bytes everywhere

Key instructions:
  baseline   i16x8.extmul_low_i8x16_s + i32x4.dot_i16x8_s
  relaxed    i32x4.relaxed_dot_i8x16_i7x16_add_s
             ↑ one operand must be 7-bit (−64..63) for determinism

Size targets (gz): scalar 25 KB │ simd128 45 KB │ relaxed 52 KB
```
