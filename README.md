# ML Inference Engine (`lnn`)

A lightweight WebAssembly runtime that runs int8-quantized neural networks
in the browser with zero server dependency — **19.5 KB gzipped**, no
dependencies, `#![no_std]` Rust with a hand-written SIMD GEMM kernel.

```js
import { Model } from 'lnn';

const model = await Model.load('/models/digits.lnn');
const { Y } = model.run({ X: pixels });
```

> **Trade-off.** ONNX Runtime Web beats this on throughput and operator
> coverage. `lnn` wins on size (~50× smaller) and cold start, for small CNNs in
> size-sensitive places: embedded web views, offline PWAs, widgets.

```
 ONNX model ──► lnn-convert (calibrate, quantize, pack) ──► .lnn file
                                                              │
 browser:  lnn JS API ──► lnn.wasm (int8 kernels, SIMD128 / relaxed-SIMD / scalar) ──► outputs
```

## Repository layout

```
.
├── README.md          ← you are here
├── docs/
│   ├── design.md      ← full design guide (the spec code comments cite)
│   └── design.pdf     ← same guide, PDF
└── v1/                ← first implementation
    ├── crates/lnn/    Rust runtime: graph, quantization, conv/GEMM kernels
    ├── converter/     Python ONNX → .lnn converter (lnn-convert)
    ├── js/            TypeScript loader and public API
    ├── tests/         conformance vs NumPy, robustness, arena, browser tests
    ├── bench/         benchmarks
    ├── examples/      digits demo page
    ├── scripts/       build, size report, feature probes
    └── docs/          quantization, model format, benchmarks, spec errata
```

Each `vN/` directory is a self-contained iteration. Start with
[`v1/README.md`](v1/README.md).

## Quick start

Needs Rust (with the wasm32 target), Python 3 and Node.js:

```bash
cd v1
make build      # three .wasm builds + the TypeScript package
make fixtures   # train and convert the demo model
make test       # rust + python + js
make serve      # then open http://127.0.0.1:8000/
```

Convert your own model:

```bash
pip install -e v1/converter
lnn-convert model.onnx -o model.lnn --calib samples.npz
```

## Versions and feedback

| Version | Summary | Feedback |
|---|---|---|
| [v1](v1/) | int8 WASM runtime, 14 operators, ONNX converter, TS API, browser demo | — |

Add a row per version as new iterations land.
