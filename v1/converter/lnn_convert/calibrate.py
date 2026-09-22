"""Activation-range calibration (design doc 7.3).

"Calibration matters more than the kernel. A bad activation scale costs you
more accuracy than any amount of kernel cleverness gains you speed."

So: percentile clipping rather than absolute min/max, because one outlier
activation destroys the scale for a whole tensor, and enough samples that the
percentile means something.
"""

from __future__ import annotations

import numpy as np


def _expose_all_intermediates(model):
    """Return a copy of the model with every intermediate value as an output."""
    import onnx

    model = onnx.shape_inference.infer_shapes(model)
    g = model.graph
    existing = {o.name for o in g.output}
    initializers = {t.name for t in g.initializer}
    produced = {o for node in g.node for o in node.output}
    for name in sorted(produced | {i.name for i in g.input}):
        if name in existing or name in initializers:
            continue
        g.output.extend([onnx.helper.ValueInfoProto(name=name)])
    return model


def collect_ranges(model, samples: dict, percentile: float = 99.99, progress=None) -> dict:
    """Run the fp32 model over calibration inputs and return per-tensor
    `(lo, hi)` ranges clipped at `percentile`.

    `samples` maps each graph input name to an array whose first axis is the
    sample index.
    """
    import onnxruntime as ort

    exposed = _expose_all_intermediates(model)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(exposed.SerializeToString(), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]

    # A sample array's leading axis is the sample index; the rest must match
    # the graph input's own shape, which is not always batch-1.
    want_shape = {}
    for vi in sess.get_inputs():
        want_shape[vi.name] = [d if isinstance(d, int) else 1 for d in vi.shape]

    n = len(next(iter(samples.values())))
    collected: dict[str, list] = {k: [] for k in names}
    for i in range(n):
        feed = {k: np.asarray(v[i], dtype=np.float32).reshape(want_shape[k]) for k, v in samples.items()}
        outs = sess.run(names, feed)
        for name, arr in zip(names, outs):
            a = np.asarray(arr, dtype=np.float32).ravel()
            if a.size:
                collected[name].append(a)
        if progress:
            progress((i + 1) / n)

    lo_pct = 100.0 - percentile
    ranges = {}
    for name, chunks in collected.items():
        if not chunks:
            continue
        all_v = np.concatenate(chunks)
        lo = float(np.percentile(all_v, lo_pct))
        hi = float(np.percentile(all_v, percentile))
        # Percentiles can invert on a constant tensor; keep the true extremes
        # as a floor so the range is never empty.
        ranges[name] = (min(lo, float(all_v.min())) if lo > hi else lo, max(hi, lo))
    # The graph inputs are fed, not computed, so record them exactly.
    for name, v in samples.items():
        a = np.asarray(v, dtype=np.float32).ravel()
        ranges[name] = (float(a.min()), float(a.max()))
    return ranges
