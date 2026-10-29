# ML Inference Engine (`lnn`)

[![CI](https://github.com/syedjafri06193/ML-Inference-Engine-proj/actions/workflows/ci.yml/badge.svg)](https://github.com/syedjafri06193/ML-Inference-Engine-proj/actions/workflows/ci.yml)

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
├── 1—Documentation/   ← all documentation (index in its README)
│   ├── design.md      ← full design guide (the spec code comments cite)
│   ├── design.pdf     ← same guide, PDF
│   └── v1/            quantization, model format, benchmarks, spec errata
└── v1/                ← first implementation
    ├── crates/lnn/    Rust runtime: graph, quantization, conv/GEMM kernels
    ├── converter/     Python ONNX → .lnn converter (lnn-convert)
    ├── js/            TypeScript loader and public API
    ├── tests/         conformance vs NumPy, robustness, arena, browser tests
    ├── bench/         benchmarks
    ├── examples/      digits demo page
    └── scripts/       build, size report, feature probes
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

## Documentation

All documentation lives in [`1—Documentation/`](1—Documentation/README.md) — start with its index.

<!-- project-metrics:start -->
## Project Metrics

[![Stars](https://img.shields.io/github/stars/syedjafri06193/ML-Inference-Engine-proj?style=flat-square)](https://github.com/syedjafri06193/ML-Inference-Engine-proj/stargazers) [![Forks](https://img.shields.io/github/forks/syedjafri06193/ML-Inference-Engine-proj?style=flat-square)](https://github.com/syedjafri06193/ML-Inference-Engine-proj/forks) [![Open issues](https://img.shields.io/github/issues/syedjafri06193/ML-Inference-Engine-proj?style=flat-square)](https://github.com/syedjafri06193/ML-Inference-Engine-proj/issues) [![Last commit](https://img.shields.io/github/last-commit/syedjafri06193/ML-Inference-Engine-proj/improvements?style=flat-square)](https://github.com/syedjafri06193/ML-Inference-Engine-proj/commits/improvements) [![Commits](https://img.shields.io/github/commit-activity/t/syedjafri06193/ML-Inference-Engine-proj?style=flat-square)](https://github.com/syedjafri06193/ML-Inference-Engine-proj/commits/improvements) [![Top language](https://img.shields.io/github/languages/top/syedjafri06193/ML-Inference-Engine-proj?style=flat-square)](https://github.com/syedjafri06193/ML-Inference-Engine-proj) [![Repo size](https://img.shields.io/github/repo-size/syedjafri06193/ML-Inference-Engine-proj?style=flat-square)](https://github.com/syedjafri06193/ML-Inference-Engine-proj)

### Code

| Metric | Value |
|---|---|
| Lines of code | 4,984 |
| Source files | 43 |
| Tracked files | 60 |
| Documentation | 8 Markdown files (16,723 words), 1 PDF |
| Commits | 10 |
| Contributors | 1 |
| Active | 2026-09-11 → 2026-10-28 |

| Language | Lines | Share |
|---|---:|---:|
| Python | 2,059 | 41.3% |
| Rust | 1,906 | 38.2% |
| JavaScript | 404 | 8.1% |
| TypeScript | 253 | 5.1% |
| HTML+Genshi | 197 | 4.0% |
| Bash | 77 | 1.5% |

### Performance

| Metric | Result |
|---|---|
| Int8 GEMM speed-up over scalar (WASM, 4 shapes) | **6.1–8.2×** |
| Peak throughput, conv 3×3 196×64×576 | **46.96 GOP/s** (relaxed-SIMD) |
| Hardware | Xeon 2.10 GHz vCPU, Node 22 / V8 12.4 |

<sub>Measured figures recorded in [`1—Documentation/v1/benchmarks.md`](1%E2%80%94Documentation/v1/benchmarks.md).</sub>

<sub>Code stats as of 2026-10-28, counted with pygount (comments and blank lines excluded). Badges update live.</sub>
<!-- project-metrics:end -->
