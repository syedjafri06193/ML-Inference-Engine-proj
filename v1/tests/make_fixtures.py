"""Build the demo models and the fixtures the JS tests check against.

Writes:
  dist/models/digits-mlp.lnn      the trained 8x8-digit classifier
  dist/models/mobilenet-ish.lnn   a MobileNetV2-shaped CNN, for benchmarks
  js/test/fixtures.json           inputs + expected outputs from the reference
  examples/digits/digits.lnn      a copy for the browser demo
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "converter"))

import models  # noqa: E402
from lnn_convert import Reference, convert_model, stats  # noqa: E402


def main() -> int:
    out_dir = os.path.join(ROOT, "dist", "models")
    os.makedirs(out_dir, exist_ok=True)

    print("training the digits MLP ...")
    model, (xtr, ytr, xte, yte), clf = models.train_digit_mlp()
    blob = convert_model(model, {"X": xtr[:300]})
    mlp_path = os.path.join(out_dir, "digits-mlp.lnn")
    with open(mlp_path, "wb") as f:
        f.write(blob)
    s = stats(blob)
    print(f"  digits-mlp.lnn: {s['total'] / 1024:.1f} KiB, {s['n_nodes']} nodes, "
          f"arena {s['arena_bytes'] / 1024:.1f} KiB")

    ref = Reference(blob)
    cases = []
    for i in range(8):
        x = xte[i : i + 1].astype(np.float32)
        y = ref.run([x])[0].ravel()
        cases.append({"input": x.ravel().tolist(), "expected": y.tolist(), "label": int(yte[i])})

    # A second model, exercising every conv kernel, for the JS tests.
    cnn = models.mobilenet_block_model(c=16, h=16, w=16)
    cx = np.random.default_rng(0).normal(size=(24, 3, 16, 16)).astype(np.float32)
    cnn_blob = convert_model(cnn, {"X": cx}, seven_bit=True)
    cnn_path = os.path.join(out_dir, "mobilenet-ish.lnn")
    with open(cnn_path, "wb") as f:
        f.write(cnn_blob)
    cs = stats(cnn_blob)
    print(f"  mobilenet-ish.lnn: {cs['total'] / 1024:.1f} KiB, {cs['n_nodes']} nodes, "
          f"arena {cs['arena_bytes'] / 1024:.1f} KiB")

    cnn_ref = Reference(cnn_blob)
    cnn_cases = []
    for i in range(3):
        x = cx[i : i + 1]
        cnn_cases.append({
            "input": x.ravel().tolist(),
            "expected": cnn_ref.run([x])[0].ravel().tolist(),
        })

    fixtures = {
        "digits": {
            "model": "digits-mlp.lnn",
            "inputShape": [1, 64],
            "outputName": "Y:f32",
            "cases": cases,
            "fp32Accuracy": float(clf.score(xte, yte)),
        },
        "cnn": {"model": "mobilenet-ish.lnn", "cases": cnn_cases},
    }
    fx = os.path.join(ROOT, "js", "test", "fixtures.json")
    os.makedirs(os.path.dirname(fx), exist_ok=True)
    with open(fx, "w") as f:
        json.dump(fixtures, f)
    print(f"  wrote {os.path.relpath(fx, ROOT)}")

    demo = os.path.join(ROOT, "examples", "digits")
    os.makedirs(demo, exist_ok=True)
    shutil.copy(mlp_path, os.path.join(demo, "digits.lnn"))
    for isa in ("scalar", "simd128", "relaxed"):
        src = os.path.join(ROOT, "dist", f"lnn-{isa}.wasm")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(demo, f"lnn-{isa}.wasm"))
    # The compiled JS, so the demo folder is a self-contained static site.
    for js in ("index.js", "detect.js", "format.js"):
        src = os.path.join(ROOT, "js", "dist", js)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(demo, js))
    print(f"  copied the demo into {os.path.relpath(demo, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
