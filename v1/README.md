# lnn — int8 neural networks in the browser, in 19 KB

A dependency-free WebAssembly inference runtime for quantized convolutional
networks. **19.5 KB gzipped**, no `wasm-bindgen`, no `serde`, no npm
dependency tree — `#![no_std]` Rust with a hand-written SIMD GEMM kernel and
nine exported functions.

```js
import { Model } from 'lnn';

const model = await Model.load('/models/digits.lnn');
const { Y } = model.run({ X: pixels });
```

Copy one `.wasm` and one `.lnn` next to your HTML. It runs from a static
host, from `file://`, and offline after the first load.

## Why it exists

ONNX Runtime Web and LiteRT.js will beat this on throughput — they have years
of engineering and full operator coverage. What they cannot do is be 19 KB.
That is the trade this project makes, and where it wins:

| | lnn | ONNX Runtime Web |
|---|---|---|
| Runtime size (gzipped) | **19.5 KB** | several MB |
| Dependencies | none | — |
| Cold start | a few ms to compile | tens of ms, or shader compilation on WebGPU |
| Operator coverage | 14 | hundreds |
| Peak throughput | 47 GOP/s (int8, 1 thread) | higher |

If you need a full ONNX runtime, use ONNX Runtime Web. If you need a small
CNN in a size-sensitive place — an embedded web view, an ad unit, an offline
PWA, a widget — this is 50× smaller.

## What it does

* **int8 quantized inference**: convolution, depthwise convolution, fully
  connected, add, multiply, max/average pooling, softmax, concat, clamp,
  lookup activations (sigmoid, tanh, hard-swish, hard-sigmoid), reshape, and
  the fp32 boundary ops. 14 operators, chosen to cover MobileNet-class CNNs.
* **Three builds, picked at run time**: scalar, baseline SIMD, and relaxed
  SIMD. The page validates a probe module to decide; Safari gets baseline
  SIMD, Chrome and Firefox get the int8 dot product.
* **A Python converter** from ONNX: batch-norm folding, activation fusion,
  NCHW → NHWC, percentile calibration, per-channel weight scales, fixed-point
  multipliers, weight pre-packing, and arena planning by liveness analysis.
* **Zero allocation during inference.** All memory is reserved at load.

## Numbers

Measured on an Intel Xeon @ 2.10 GHz under Node 22 (V8 12.4 — the same engine
as Chrome). Full tables in [docs/benchmarks.md](../1—Documentation/v1/benchmarks.md).

**The GEMM optimization ladder** (int8 GOP/s, each column adds one thing):

| Shape (M×N×K) | naive scalar | + blocking & packing | + simd128 kernel | + relaxed dot |
|---|---|---|---|---|
| pointwise 196×256×64 | 3.66 | 6.13 | 19.75 | **22.14** |
| conv 3×3 196×64×576 | 5.73 | 7.57 | 39.74 | **46.96** |
| classifier 1×1000×256 | 5.78 | 6.57 | 21.10 | **42.96** |

**Model latency**, MobileNetV2-shaped CNN: 0.414 ms scalar → **0.176 ms**
with SIMD.

**Accuracy**, a real trained classifier (scikit-learn digits):
fp32 97.1% → int8 **97.1%**, with 100% top-1 agreement against the fp32 model.

**Size**: 14.9 KB scalar, 19.5 KB simd128, 19.6 KB relaxed, all gzipped.

## Quick start

```bash
make build      # three .wasm builds + the TypeScript package
make fixtures   # trains the demo model and converts it
make test       # rust + python + js
make serve      # then open http://127.0.0.1:8000/
```

Converting your own model:

```bash
pip install -e converter
lnn-convert mobilenetv2.onnx -o mobilenetv2.lnn --calib samples.npz
lnn-convert mobilenetv2.onnx -o mobilenetv2.lnn --calib samples.npz --relaxed
```

`--calib` takes an `.npz` of a few hundred representative inputs. It is the
single most important input to accuracy: calibrating on the wrong
preprocessing, or on ten images, looks exactly like a kernel bug.
Unsupported operators are refused by name, with the list of what is
supported.

## How it works

```
JS: Float32Array in
  ↓ copied into wasm linear memory
Quantize fp32 → int8
  ↓ for each node in topological order:
      conv → im2col (skipped for 1×1) → packed int8 GEMM
      int8 × int8 → int32 accumulate
      requantize: fixed-point multiply, shift, clamp
  ↓
Dequantize int8 → fp32
JS: Float32Array out
```

Everything funnels into one GEMM, which is where 80–90% of the time goes, so
that is the kernel that got the attention: a 4×8 register tile with eight
`v128` accumulators, weights pre-packed by the converter into a layout all
three ISAs can read, and a separate 1×8 path for batch-1 classifier heads.

The design decisions and their reasoning are in
[docs/quantization.md](../1—Documentation/v1/quantization.md) and
[docs/model-format.md](../1—Documentation/v1/model-format.md).

## Layout

```
crates/lnn/      the runtime: no_std Rust, no dependencies
  src/quant.rs     gemmlowp-exact fixed-point requantization
  src/model.rs     zero-copy .lnn parser, fully validated
  src/graph.rs     the interpreter and the per-operator validator
  src/kernels/     GEMM (scalar / simd128 / relaxed), conv, elementwise
converter/       ONNX → .lnn, plus a NumPy reference interpreter
js/              the TypeScript API, feature probes, loader
tests/           conformance, robustness, arena, browser
bench/           the ladder and model latency
examples/digits/ a static page that classifies what you draw
```

## Testing

69 tests across three suites:

* **Rust** (7): the fixed-point primitives and the packed GEMM against a
  naive loop, on awkward shapes.
* **Python** (45): every operator and every whole model run through all three
  wasm builds via wasmtime and compared to an independent NumPy
  implementation — **exact integer equality**, not a tolerance. Quantization
  bugs are silent, so "close enough" makes every later bug ambiguous. Plus
  robustness: a real model truncated at every length, every header field
  corrupted, and several hundred random byte flips, each of which must be
  rejected or run without trapping.
* **JavaScript** (17): the public API, error paths, and the same
  reference-exact comparison through the TypeScript layer.

Plus a browser suite (`make test-browser`) that drives Chromium: feature
detection, the Cache API offline path, and the demo classifying a drawn
digit.

## Limitations

Worth knowing before you adopt it.

* **Single-threaded.** Threads need `SharedArrayBuffer`, which needs COOP and
  COEP headers, which a static host cannot set — the premise this project is
  built on. `detectFeatures().threads` reports whether they would be
  available.
* **No transformers yet.** `MatMul` between two activations, `Transpose` and
  `LayerNorm` are reserved in the format but not implemented.
* **Static shapes, batch 1.** Resolved at conversion time, which is what
  makes the arena allocator and most of the size claim possible.
* **Models up to roughly 50 MB.** There is no `mmap` in a browser: every
  weight byte lives in linear memory for the model's lifetime.
* **Not an LLM runtime.** A 1B-parameter int8 model is 1 GB resident with no
  way to page it. WebLLM and wllama solve that problem properly.

[docs/notes-on-the-spec.md](../1—Documentation/v1/notes-on-the-spec.md) records where this
departs from the design document it was built from — including four bugs in
the document's own reference code, one of which would have made results
differ between Intel and ARM machines.
