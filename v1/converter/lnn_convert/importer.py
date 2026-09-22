"""ONNX -> internal graph, with fusion and the NCHW -> NHWC move.

Everything expensive and fiddly happens here rather than in the runtime
(design doc 7.3, 4.2): batch-norm folding, activation fusion, layout
conversion. What comes out is a flat, topologically ordered list of ops whose
activations are all NHWC and whose weights are float — ready to quantize.

Unsupported operators raise `Unsupported` with the node's name and the list
of what is supported, because "Unsupported op 'InstanceNormalization' at node
'blk3/norm'" is a message someone can act on (doc 7.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SUPPORTED = (
    "Conv, Gemm, MatMul(const B), Add, Mul, Relu, Clip, LeakyRelu(no), "
    "MaxPool, AveragePool, GlobalAveragePool, Flatten, Reshape, Squeeze, "
    "Unsqueeze, Concat, Softmax, Sigmoid, Tanh, HardSwish, HardSigmoid, "
    "BatchNormalization (folded), Identity, Dropout"
)


class Unsupported(Exception):
    pass


@dataclass
class Op:
    kind: str
    inputs: list
    output: str
    attrs: dict = field(default_factory=dict)
    # Fused activation bounds in float, or None.
    act: tuple | None = None
    name: str = ""


@dataclass
class Graph:
    ops: list
    shapes: dict  # name -> NHWC (or flat) shape tuple
    inputs: list
    outputs: list
    # Tensors produced by a Flatten of a >1x1 spatial map, and the (H, W, C)
    # they were flattened from. A FullyConnected consuming one needs its
    # weight columns permuted (see `convert.py`).
    flatten_src: dict = field(default_factory=dict)


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == 1:
                return a.f
            if a.type == 2:
                return a.i
            if a.type == 3:
                return a.s.decode()
            if a.type == 6:
                return list(a.floats)
            if a.type == 7:
                return list(a.ints)
    return default


def _nchw_to_nhwc(shape):
    if len(shape) == 4:
        n, c, h, w = shape
        return (n, h, w, c)
    return tuple(shape)


def import_onnx(model) -> Graph:
    import onnx
    from onnx import numpy_helper

    model = onnx.shape_inference.infer_shapes(model)
    g = model.graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}

    # ONNX (NCHW) shapes for every value, from the graph's own inference.
    nchw: dict[str, tuple] = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else 1)
        if dims:
            nchw[vi.name] = tuple(dims)
    for name, arr in init.items():
        nchw[name] = tuple(arr.shape)

    consumers: dict[str, int] = {}
    for node in g.node:
        for i in node.input:
            consumers[i] = consumers.get(i, 0) + 1
    graph_outputs = {o.name for o in g.output}

    ops: list[Op] = []
    produced_by: dict[str, Op] = {}
    alias: dict[str, str] = {}  # Identity/Dropout pass-through
    shapes: dict[str, tuple] = {}
    flatten_src: dict[str, tuple] = {}

    def resolve(n: str) -> str:
        while n in alias:
            n = alias[n]
        return n

    def is_const(n: str) -> bool:
        return resolve(n) in init

    def const(n: str) -> np.ndarray:
        return init[resolve(n)]

    def set_shape(name: str) -> None:
        if name in nchw:
            shapes[name] = _nchw_to_nhwc(nchw[name])

    for vi in g.input:
        if vi.name not in init:
            set_shape(vi.name)

    def only_consumer(name: str) -> bool:
        return consumers.get(name, 0) == 1 and name not in graph_outputs

    def fuse_activation(x: str, lo: float, hi: float, out: str) -> bool:
        """Attach a clamp to the op that produced `x`, if it can take one."""
        src = produced_by.get(resolve(x))
        if src is None or src.act is not None:
            return False
        if src.kind not in ("conv", "depthwise", "fc", "add", "mul"):
            return False
        if not only_consumer(x):
            return False
        src.act = (lo, hi)
        del produced_by[src.output]
        src.output = out
        produced_by[out] = src
        set_shape(out)
        return True

    for node in g.node:
        op_t = node.op_type
        ins = [resolve(i) for i in node.input]
        out = node.output[0] if node.output else ""

        if op_t in ("Identity", "Dropout"):
            alias[out] = ins[0]
            continue

        if op_t == "Constant":
            init[out] = np.array(_attr(node, "value_float", 0.0))
            continue

        if op_t == "BatchNormalization":
            src = produced_by.get(ins[0])
            if src is None or src.kind not in ("conv", "depthwise") or not only_consumer(ins[0]):
                raise Unsupported(
                    f"BatchNormalization at '{node.name or out}' does not follow a "
                    "convolution it can be folded into. Run the model through "
                    "onnxsim, or export with BN already folded."
                )
            gamma, beta, mean, var = (const(i).astype(np.float64) for i in ins[1:5])
            eps = float(_attr(node, "epsilon", 1e-5))
            s = gamma / np.sqrt(var + eps)
            w, b = src.attrs["w"], src.attrs["b"]
            # Weights are (O, KH, KW, C) for conv and (KH, KW, C) for
            # depthwise; the channel axis differs.
            if src.kind == "conv":
                src.attrs["w"] = (w.astype(np.float64) * s.reshape(-1, 1, 1, 1)).astype(np.float32)
            else:
                src.attrs["w"] = (w.astype(np.float64) * s.reshape(1, 1, -1)).astype(np.float32)
            src.attrs["b"] = ((b.astype(np.float64) - mean) * s + beta).astype(np.float32)
            del produced_by[src.output]
            src.output = out
            produced_by[out] = src
            set_shape(out)
            continue

        if op_t == "Relu":
            if fuse_activation(ins[0], 0.0, float("inf"), out):
                continue
            ops.append(Op("clamp", [ins[0]], out, {"lo": 0.0, "hi": float("inf")}, name=node.name))
        elif op_t == "Clip":
            lo, hi = -float("inf"), float("inf")
            if len(ins) > 1 and ins[1] and is_const(ins[1]):
                lo = float(const(ins[1]))
            if len(ins) > 2 and ins[2] and is_const(ins[2]):
                hi = float(const(ins[2]))
            lo = float(_attr(node, "min", lo))
            hi = float(_attr(node, "max", hi))
            if fuse_activation(ins[0], lo, hi, out):
                continue
            ops.append(Op("clamp", [ins[0]], out, {"lo": lo, "hi": hi}, name=node.name))
        elif op_t == "Conv":
            if not is_const(ins[1]):
                raise Unsupported(f"Conv '{node.name or out}' has non-constant weights.")
            w = const(ins[1]).astype(np.float32)  # (M, C/group, kh, kw)
            b = const(ins[2]).astype(np.float32) if len(ins) > 2 and ins[2] else np.zeros(w.shape[0], np.float32)
            group = int(_attr(node, "group", 1))
            kh, kw = _attr(node, "kernel_shape", list(w.shape[2:]))
            sh, sw = _attr(node, "strides", [1, 1])
            dh, dw = _attr(node, "dilations", [1, 1])
            pads = _attr(node, "pads", [0, 0, 0, 0])
            if len(pads) != 4:
                raise Unsupported(f"Conv '{node.name or out}': only 2-D convolution is supported.")
            cin = nchw[ins[0]][1]
            attrs = dict(kh=kh, kw=kw, sh=sh, sw=sw, pads=pads, dh=dh, dw=dw, b=b)
            if group == 1:
                attrs["w"] = np.ascontiguousarray(w.transpose(0, 2, 3, 1))  # OHWI
                ops.append(Op("conv", [ins[0]], out, attrs, name=node.name))
            elif group == cin and w.shape[0] == cin:
                # (C, 1, kh, kw) -> (kh, kw, C)
                attrs["w"] = np.ascontiguousarray(w[:, 0].transpose(1, 2, 0))
                ops.append(Op("depthwise", [ins[0]], out, attrs, name=node.name))
            else:
                raise Unsupported(
                    f"Conv '{node.name or out}' has group={group} with {cin} input "
                    "channels; only group=1 and depthwise (group = channels, "
                    "multiplier 1) are supported."
                )
        elif op_t in ("Gemm", "MatMul"):
            if not is_const(ins[1]):
                raise Unsupported(
                    f"{op_t} '{node.name or out}' multiplies two activations; "
                    "v1 supports MatMul only with a constant right-hand side."
                )
            b_arr = const(ins[1]).astype(np.float32)
            if op_t == "Gemm":
                if int(_attr(node, "transA", 0)):
                    raise Unsupported(f"Gemm '{node.name or out}': transA is not supported.")
                w = b_arr if int(_attr(node, "transB", 0)) else b_arr.T
                bias = const(ins[2]).astype(np.float32) if len(ins) > 2 and ins[2] else np.zeros(w.shape[0], np.float32)
                alpha = float(_attr(node, "alpha", 1.0))
                beta = float(_attr(node, "beta", 1.0))
                w, bias = w * alpha, bias * beta
            else:
                w = b_arr.T  # (K, N) -> (N, K)
                bias = np.zeros(w.shape[0], np.float32)
            ops.append(Op("fc", [ins[0]], out, {"w": w, "b": bias}, name=node.name))
        elif op_t == "Add":
            if is_const(ins[0]) or is_const(ins[1]):
                raise Unsupported(
                    f"Add '{node.name or out}' adds a constant; fold it into the "
                    "preceding layer's bias before converting."
                )
            ops.append(Op("add", [ins[0], ins[1]], out, name=node.name))
        elif op_t == "Mul":
            ops.append(Op("mul", [ins[0], ins[1]], out, {"const": is_const(ins[1])}, name=node.name))
        elif op_t in ("MaxPool", "AveragePool", "GlobalAveragePool"):
            h, w_ = nchw[ins[0]][2], nchw[ins[0]][3]
            if op_t == "GlobalAveragePool":
                kh, kw, sh, sw, pads = h, w_, 1, 1, [0, 0, 0, 0]
            else:
                kh, kw = _attr(node, "kernel_shape")
                sh, sw = _attr(node, "strides", [1, 1])
                pads = _attr(node, "pads", [0, 0, 0, 0])
                if int(_attr(node, "ceil_mode", 0)):
                    raise Unsupported(f"{op_t} '{node.name or out}': ceil_mode is not supported.")
                if op_t == "AveragePool" and int(_attr(node, "count_include_pad", 0)):
                    raise Unsupported(f"AveragePool '{node.name or out}': count_include_pad is not supported.")
            kind = "maxpool" if op_t == "MaxPool" else "avgpool"
            ops.append(Op(kind, [ins[0]], out, dict(kh=kh, kw=kw, sh=sh, sw=sw, pads=pads), name=node.name))
        elif op_t in ("Flatten", "Reshape", "Squeeze", "Unsqueeze"):
            src_shape = nchw.get(ins[0])
            if src_shape and len(src_shape) == 4 and (src_shape[2] > 1 or src_shape[3] > 1):
                # NHWC bytes are ordered h,w,c but ONNX flattens c,h,w.
                flatten_src[out] = (src_shape[2], src_shape[3], src_shape[1])
            ops.append(Op("reshape", [ins[0]], out, name=node.name))
        elif op_t == "Concat":
            axis = int(_attr(node, "axis", 1))
            rank = len(nchw.get(ins[0], (1, 1)))
            if not (axis in (1, -3) and rank == 4 or axis in (1, -1) and rank == 2):
                raise Unsupported(
                    f"Concat '{node.name or out}' on axis {axis}; only the channel "
                    "axis is supported."
                )
            ops.append(Op("concat", list(ins), out, name=node.name))
        elif op_t == "Softmax":
            rank = len(nchw.get(ins[0], (1, 1)))
            axis = int(_attr(node, "axis", -1))
            if axis not in (-1, rank - 1):
                raise Unsupported(f"Softmax '{node.name or out}' on axis {axis}; only the last axis is supported.")
            ops.append(Op("softmax", [ins[0]], out, name=node.name))
        elif op_t in ("Sigmoid", "Tanh", "HardSwish", "HardSigmoid"):
            fns = {
                "Sigmoid": lambda v: 1.0 / (1.0 + np.exp(-v)),
                "Tanh": np.tanh,
                "HardSwish": lambda v: v * np.clip(v / 6.0 + 0.5, 0.0, 1.0),
            }
            if op_t == "HardSigmoid":
                a = float(_attr(node, "alpha", 0.2))
                b_ = float(_attr(node, "beta", 0.5))
                fns["HardSigmoid"] = lambda v, a=a, b_=b_: np.clip(a * v + b_, 0.0, 1.0)
            ops.append(Op("lut", [ins[0]], out, {"fn": fns[op_t], "label": op_t}, name=node.name))
        else:
            raise Unsupported(
                f"Unsupported op '{op_t}' at node '{node.name or out}'. Supported: {SUPPORTED}."
            )

        if ops and ops[-1].output == out:
            produced_by[out] = ops[-1]
        set_shape(out)

    for vi in g.input:
        if vi.name not in init:
            set_shape(vi.name)
    for vi in g.output:
        set_shape(vi.name)

    # Constant operands referenced by the remaining ops (e.g. Mul by a
    # per-channel constant) travel with the op.
    for op in ops:
        if op.kind == "mul" and op.attrs.get("const"):
            op.attrs["value"] = init[op.inputs[1]].astype(np.float32)
            op.inputs = [op.inputs[0]]

    inputs = [vi.name for vi in g.input if vi.name not in init]
    outputs = [resolve(vi.name) for vi in g.output]
    return Graph(ops=ops, shapes=shapes, inputs=inputs, outputs=outputs, flatten_src=flatten_src)
