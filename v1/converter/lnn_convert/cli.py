"""`lnn-convert model.onnx -o model.lnn --calib samples.npz`"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import format as fmt
from .convert import convert_model, stats
from .importer import Unsupported


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lnn-convert", description="Convert an ONNX model to .lnn")
    ap.add_argument("model", help="input .onnx file")
    ap.add_argument("-o", "--out", required=True, help="output .lnn file")
    ap.add_argument("--calib", help=".npz of calibration samples, one array per graph input")
    ap.add_argument("--calib-samples", type=int, default=0, help="use at most this many samples")
    ap.add_argument(
        "--percentile", type=float, default=99.99,
        help="activation clipping percentile; absolute min/max (100) lets one outlier wreck a scale",
    )
    ap.add_argument(
        "--relaxed", action="store_true",
        help="quantize weights to 7 bits so the relaxed-SIMD kernel is deterministic across CPUs",
    )
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    import onnx

    model = onnx.load(args.model)

    if args.calib:
        data = np.load(args.calib)
        samples = {k: data[k] for k in data.files}
    else:
        # No calibration set: fall back to Gaussian noise shaped like the
        # input. This produces a runnable model and almost always a bad one,
        # so it says so loudly (doc 17.2 ranks calibration as the #1 cause of
        # quantization accuracy loss).
        print(
            "warning: no --calib given; calibrating on random noise. Expect poor\n"
            "         accuracy. Pass a few hundred representative samples.",
            file=sys.stderr,
        )
        samples = {}
        for vi in model.graph.input:
            names = {t.name for t in model.graph.initializer}
            if vi.name in names:
                continue
            dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
            samples[vi.name] = np.random.default_rng(0).normal(size=[8] + dims[1:]).astype(np.float32)

    if args.calib_samples:
        samples = {k: v[: args.calib_samples] for k, v in samples.items()}

    try:
        blob = convert_model(model, samples, seven_bit=args.relaxed, percentile=args.percentile)
    except Unsupported as e:
        print(f"lnn-convert: {e}", file=sys.stderr)
        return 2

    with open(args.out, "wb") as f:
        f.write(blob)

    if not args.quiet:
        s = stats(blob)
        print(f"wrote {args.out}")
        print(f"  nodes          {s['n_nodes']}")
        print(f"  tensors        {s['n_tensors']}")
        print(f"  weights        {s['weights_bytes'] / 1024:.1f} KiB")
        print(f"  activations    {s['arena_bytes'] / 1024:.1f} KiB arena")
        print(f"  scratch        {s['scratch_bytes'] / 1024:.1f} KiB")
        print(f"  total          {s['total'] / 1024:.1f} KiB")
        if s["flags"] & fmt.FLAG_WEIGHTS_7BIT:
            print("  weights        7-bit (relaxed-SIMD path enabled)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
