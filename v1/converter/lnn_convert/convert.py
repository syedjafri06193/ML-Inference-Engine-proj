"""Quantize an imported graph and write the .lnn file."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import format as fmt
from . import pack, quantize as q
from .calibrate import collect_ranges
from .importer import Unsupported, import_onnx


@dataclass
class TensorSpec:
    name: str
    kind: str  # input | output | activation | weight
    dtype: int
    shape: tuple
    scale: float = 1.0
    zero_point: int = 0
    data: np.ndarray | None = None
    scales: np.ndarray | None = None  # per-channel
    nbytes: int = 0
    offset: int = 0
    scale_off: int = 0

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


class WeightBlob:
    """Append-only weight region. Everything lands 16-byte aligned so SIMD
    loads never straddle (doc 7.1)."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def add(self, arr: np.ndarray) -> tuple[int, int]:
        pad = fmt.align_up(len(self.buf)) - len(self.buf)
        self.buf += b"\0" * pad
        off = len(self.buf)
        raw = np.ascontiguousarray(arr).tobytes()
        self.buf += raw
        return off, len(raw)


class Converter:
    def __init__(self, seven_bit: bool = False):
        self.seven_bit = seven_bit
        self.tensors: dict[str, TensorSpec] = {}
        self.nodes: list = []
        self.blob = WeightBlob()
        self.scratch = 0
        self._uid = 0

    # ---------------------------------------------------------------- helpers

    def uid(self, stem: str) -> str:
        self._uid += 1
        return f"{stem}#{self._uid}"

    def act(self, name: str, shape: tuple, scale: float, zp: int) -> TensorSpec:
        t = TensorSpec(name, "activation", fmt.DTYPE_I8, shape, scale, zp)
        t.nbytes = t.numel
        self.tensors[name] = t
        return t

    def weight(self, name: str, arr: np.ndarray, dtype: int, shape=None, scales=None, zp: int = 0) -> str:
        t = TensorSpec(name, "weight", dtype, tuple(shape if shape is not None else arr.shape))
        t.data = arr
        t.scales = scales
        t.zero_point = zp
        self.tensors[name] = t
        return name

    def qparams(self, name: str) -> tuple[float, int]:
        t = self.tensors[name]
        return t.scale, t.zero_point

    def quant_bound(self, v: float, scale: float, zp: int, default: int) -> int:
        if v in (float("inf"), float("-inf")) or v != v:
            return default
        return int(np.clip(round(v / scale) + zp, -128, 127))

    # ------------------------------------------------------------- emission

    def emit_conv(self, op, x: str, y: str, depthwise: bool) -> None:
        a = op.attrs
        xs, xz = self.qparams(x)
        ys, yz = self.qparams(y)
        w = a["w"].astype(np.float32)
        axis = 2 if depthwise else 0
        scales = q.per_channel_weight_scales(w, axis=axis, seven_bit=self.seven_bit)
        qw = q.quantize_weights(w, scales, axis=axis, seven_bit=self.seven_bit)
        n_out = qw.shape[axis]
        bias_q = q.fold_bias(a["b"].astype(np.float32), xs, scales)
        mult, shift = q.requant_params(xs, scales, ys)

        if depthwise:
            # Padding is excluded from the sum, so the zero-point correction
            # varies per output position and cannot be folded into the bias;
            # the kernel subtracts Z_x per tap instead.
            wt = self.weight(self.uid(f"{y}.w"), qw, fmt.DTYPE_I8, shape=(1, qw.shape[0], qw.shape[1], qw.shape[2]), scales=scales)
            bias = bias_q.astype(np.int32)
        else:
            flat = qw.reshape(n_out, -1)
            packed = pack.pack_b(flat)
            wt = self.weight(
                self.uid(f"{y}.w"), packed, fmt.DTYPE_I8,
                shape=(n_out, qw.shape[1], qw.shape[2], qw.shape[3]), scales=scales,
            )
            bias = (bias_q.astype(np.int64) + q.zero_point_correction(flat, xz, axis=0)).astype(np.int32)

        bt = self.weight(self.uid(f"{y}.bias"), bias, fmt.DTYPE_I32)
        mt = self.weight(self.uid(f"{y}.mult"), mult, fmt.DTYPE_I32)
        st = self.weight(self.uid(f"{y}.shift"), shift, fmt.DTYPE_I32)

        lo, hi = (op.act if op.act else (float("-inf"), float("inf")))
        attrs = [
            a["kh"], a["kw"], a["sh"], a["sw"],
            a["pads"][0], a["pads"][1], a["pads"][2], a["pads"][3],
            a["dh"], a["dw"],
            self.quant_bound(lo, ys, yz, -128), self.quant_bound(hi, ys, yz, 127),
        ]
        self.nodes.append((fmt.OP_DEPTHWISE_CONV2D if depthwise else fmt.OP_CONV2D, [x, wt, bt, mt, st], [y], attrs))

        # Scratch: im2col rows, then the GEMM's A panels.
        if not depthwise:
            xh, xw, xc = self.tensors[x].shape[1:]
            yh, yw, _ = self.tensors[y].shape[1:]
            k = a["kh"] * a["kw"] * xc
            pointwise = a["kh"] == 1 and a["kw"] == 1 and a["sh"] == 1 and a["sw"] == 1 and a["pads"][0] == 0 and a["pads"][1] == 0
            im2col = 0 if pointwise else yh * yw * k
            self.scratch = max(self.scratch, fmt.align_up(im2col) + pack.gemm_scratch_bytes(k))

    def emit_fc(self, op, x: str, y: str, flatten_src) -> None:
        xs, xz = self.qparams(x)
        ys, yz = self.qparams(y)
        w = op.attrs["w"].astype(np.float32)  # (N, K)
        if op.inputs[0] in flatten_src:
            h, w_, c = flatten_src[op.inputs[0]]
            if w.shape[1] != h * w_ * c:
                raise Unsupported(
                    f"FullyConnected '{op.name or y}' expects {w.shape[1]} inputs but "
                    f"its NHWC source flattens to {h * w_ * c}."
                )
            # ONNX flattens NCHW as c,h,w; our activations are h,w,c. Permute
            # the weight columns rather than transposing the activation.
            onnx_idx = (
                np.arange(c).reshape(c, 1, 1) * (h * w_)
                + np.arange(h).reshape(1, h, 1) * w_
                + np.arange(w_).reshape(1, 1, w_)
            )
            perm = np.ascontiguousarray(onnx_idx.transpose(1, 2, 0)).ravel()
            w = w[:, perm]

        scales = q.per_channel_weight_scales(w, axis=0, seven_bit=self.seven_bit)
        qw = q.quantize_weights(w, scales, axis=0, seven_bit=self.seven_bit)
        bias_q = q.fold_bias(op.attrs["b"].astype(np.float32), xs, scales)
        bias = (bias_q.astype(np.int64) + q.zero_point_correction(qw, xz, axis=0)).astype(np.int32)
        mult, shift = q.requant_params(xs, scales, ys)

        n, k = qw.shape
        wt = self.weight(self.uid(f"{y}.w"), pack.pack_b(qw), fmt.DTYPE_I8, shape=(n, k), scales=scales)
        bt = self.weight(self.uid(f"{y}.bias"), bias, fmt.DTYPE_I32)
        mt = self.weight(self.uid(f"{y}.mult"), mult, fmt.DTYPE_I32)
        st = self.weight(self.uid(f"{y}.shift"), shift, fmt.DTYPE_I32)
        lo, hi = (op.act if op.act else (float("-inf"), float("inf")))
        attrs = [self.quant_bound(lo, ys, yz, -128), self.quant_bound(hi, ys, yz, 127)]
        self.nodes.append((fmt.OP_FULLY_CONNECTED, [x, wt, bt, mt, st], [y], attrs))
        self.scratch = max(self.scratch, pack.gemm_scratch_bytes(k))

    def emit(self, graph, ranges: dict) -> bytes:
        shapes = graph.shapes

        # --- graph inputs: f32 buffer + Quantize -------------------------
        for name in graph.inputs:
            shape = shapes[name]
            f32 = TensorSpec(f"{name}:f32", "input", fmt.DTYPE_F32, shape)
            f32.nbytes = f32.numel * 4
            self.tensors[f32.name] = f32
            scale, zp = q.choose_affine_qparams(*ranges[name])
            self.act(name, shape, scale, zp)
            self.nodes.append((fmt.OP_QUANTIZE, [f32.name], [name], []))

        def out_qparams(op) -> tuple[float, int]:
            """Quantization for an op's output."""
            if op.kind == "softmax":
                return 1.0 / 256.0, -128
            if op.kind in ("maxpool", "avgpool", "reshape", "clamp"):
                # Same grid as the input: max, average, a reshape and a clamp
                # are all closed over the input's representable values, so a
                # second grid would only add rounding error.
                return self.qparams(op.inputs[0])
            lo, hi = ranges[op.output]
            if op.act:
                lo = max(lo, op.act[0]) if op.act[0] != float("-inf") else lo
                hi = min(hi, op.act[1]) if op.act[1] != float("inf") else hi
            return q.choose_affine_qparams(lo, hi)

        for op in graph.ops:
            y = op.output
            shape = shapes.get(y)
            if shape is None:
                raise Unsupported(f"could not infer a shape for '{y}'; run the model through onnxsim first")
            ys, yz = out_qparams(op)
            self.act(y, shape, ys, yz)
            x = op.inputs[0] if op.inputs else None

            if op.kind == "conv":
                self.emit_conv(op, x, y, depthwise=False)
            elif op.kind == "depthwise":
                self.emit_conv(op, x, y, depthwise=True)
            elif op.kind == "fc":
                self.emit_fc(op, x, y, graph.flatten_src)
            elif op.kind == "add":
                s1, _ = self.qparams(op.inputs[0])
                s2, _ = self.qparams(op.inputs[1])
                p = q.add_params(s1, s2, ys)
                lo, hi = (op.act if op.act else (float("-inf"), float("inf")))
                self.nodes.append((
                    fmt.OP_ADD, [op.inputs[0], op.inputs[1]], [y],
                    [p["left_shift"], p["m1"], p["s1"], p["m2"], p["s2"], p["mo"], p["so"],
                     self.quant_bound(lo, ys, yz, -128), self.quant_bound(hi, ys, yz, 127)],
                ))
            elif op.kind == "mul":
                if op.attrs.get("const"):
                    v = op.attrs["value"].astype(np.float32).ravel()
                    s2 = float(max(np.abs(v).max(), 1e-12) / 127.0)
                    qv = np.clip(np.sign(v) * np.floor(np.abs(v) / s2 + 0.5), -127, 127).astype(np.int8)
                    b_name = self.weight(self.uid(f"{y}.const"), qv, fmt.DTYPE_I8, scales=np.array([s2], np.float32))
                    self.tensors[b_name].scale = s2
                    ins = [op.inputs[0], b_name]
                else:
                    s2, _ = self.qparams(op.inputs[1])
                    ins = [op.inputs[0], op.inputs[1]]
                s1, _ = self.qparams(op.inputs[0])
                mo, so = q.quantize_multiplier(s1 * s2 / ys)
                lo, hi = (op.act if op.act else (float("-inf"), float("inf")))
                self.nodes.append((fmt.OP_MUL, ins, [y], [mo, so, self.quant_bound(lo, ys, yz, -128), self.quant_bound(hi, ys, yz, 127)]))
            elif op.kind in ("maxpool", "avgpool"):
                a = op.attrs
                self.nodes.append((
                    fmt.OP_MAXPOOL2D if op.kind == "maxpool" else fmt.OP_AVGPOOL2D,
                    [x], [y],
                    [a["kh"], a["kw"], a["sh"], a["sw"], a["pads"][0], a["pads"][1], a["pads"][2], a["pads"][3], -128, 127],
                ))
            elif op.kind == "softmax":
                xs, _ = self.qparams(x)
                table = self.weight(self.uid(f"{y}.exp"), q.softmax_exp_table(xs), fmt.DTYPE_F32)
                self.nodes.append((fmt.OP_SOFTMAX, [x, table], [y], []))
            elif op.kind == "reshape":
                self.nodes.append((fmt.OP_RESHAPE, [x], [y], []))
            elif op.kind == "clamp":
                xs, xz = self.qparams(x)
                self.nodes.append((
                    fmt.OP_CLAMP, [x], [y],
                    [self.quant_bound(op.attrs["lo"], xs, xz, -128), self.quant_bound(op.attrs["hi"], xs, xz, 127)],
                ))
            elif op.kind == "lut":
                xs, xz = self.qparams(x)
                table = self.weight(self.uid(f"{y}.lut"), q.lut_for(op.attrs["fn"], xs, xz, ys, yz), fmt.DTYPE_I8)
                self.nodes.append((fmt.OP_LOOKUP, [x, table], [y], []))
            elif op.kind == "concat":
                attrs = []
                for name in op.inputs:
                    s, z = self.qparams(name)
                    if abs(s - ys) < 1e-12 and z == yz:
                        attrs += [0, 0]  # identical grid: copy
                    else:
                        m, sh = q.quantize_multiplier(s / ys)
                        attrs += [m, sh]
                self.nodes.append((fmt.OP_CONCAT, list(op.inputs), [y], attrs))
            else:  # pragma: no cover - importer rejects anything else
                raise Unsupported(f"internal: no emitter for '{op.kind}'")

        # --- graph outputs: Dequantize into an f32 buffer -----------------
        for name in graph.outputs:
            t = self.tensors[name]
            f32 = TensorSpec(f"{name}:f32", "output", fmt.DTYPE_F32, t.shape)
            f32.nbytes = f32.numel * 4
            self.tensors[f32.name] = f32
            self.nodes.append((fmt.OP_DEQUANTIZE, [name], [f32.name], []))

        return self.serialize()

    # ------------------------------------------------------------ container

    def serialize(self) -> bytes:
        # Weights first: their offsets and the scale arrays live in the blob.
        order = list(self.tensors)
        index = {name: i for i, name in enumerate(order)}

        for name in order:
            t = self.tensors[name]
            if t.kind == "weight":
                off, nbytes = self.blob.add(t.data)
                t.offset, t.nbytes = off, nbytes
                scales = t.scales if t.scales is not None else np.array([1.0], np.float32)
                t.scale_off, _ = self.blob.add(scales.astype(np.float32))
                t.n_scales = len(scales)
            else:
                scales = np.array([t.scale], np.float32)
                t.scale_off, _ = self.blob.add(scales)
                t.n_scales = 1

        # Activations: one arena, packed by liveness.
        class _T:
            def __init__(self, kind, nbytes):
                self.kind = kind
                self.nbytes = nbytes

        class _N:
            def __init__(self, ins, outs):
                self.inputs = ins
                self.outputs = outs

        plan_tensors = {n: _T(t.kind, t.nbytes) for n, t in self.tensors.items()}
        plan_nodes = [_N([i for i in ins], list(outs)) for _, ins, outs, _ in self.nodes]
        assign, arena_bytes = pack.plan_arena(plan_nodes, plan_tensors)
        for name, off in assign.items():
            self.tensors[name].offset = off

        strings = bytearray()
        descs = []
        for name in order:
            t = self.tensors[name]
            name_off = len(strings)
            strings += name.encode()[:120] + b"\0"
            kind = {
                "weight": fmt.KIND_WEIGHT,
                "activation": fmt.KIND_ACTIVATION,
                "input": fmt.KIND_INPUT,
                "output": fmt.KIND_OUTPUT,
            }[t.kind]
            descs.append(fmt.TensorDesc(
                name=name, dtype=t.dtype, kind=kind, dims=tuple(int(d) for d in t.shape),
                offset=t.offset, scale_off=t.scale_off, zero_point=t.zero_point,
                n_scales=getattr(t, "n_scales", 1), nbytes=t.nbytes, name_off=name_off,
            ))

        # Inputs must come first, in order (the runtime asserts it).
        def sort_key(i):
            kind = descs[i].kind
            return (0 if kind == fmt.KIND_INPUT else 1, i)

        perm = sorted(range(len(descs)), key=sort_key)
        remap = {old: new for new, old in enumerate(perm)}
        descs = [descs[i] for i in perm]

        nodes = [
            fmt.NodeDesc(op, [remap[index[i]] for i in ins], [remap[index[o]] for o in outs], attrs)
            for op, ins, outs, attrs in self.nodes
        ]

        flags = fmt.FLAG_PREPACKED | (fmt.FLAG_WEIGHTS_7BIT if self.seven_bit else 0)
        return fmt.build(descs, nodes, bytes(self.blob.buf), bytes(strings),
                         arena_bytes, max(self.scratch, 64), flags)


def convert_model(model, samples: dict, seven_bit: bool = False, percentile: float = 99.99, progress=None) -> bytes:
    """ONNX model + calibration samples -> .lnn bytes."""
    graph = import_onnx(model)
    ranges = collect_ranges(model, samples, percentile=percentile, progress=progress)
    missing = [op.output for op in graph.ops if op.output not in ranges]
    if missing:
        raise Unsupported(
            "calibration produced no range for: " + ", ".join(missing[:5])
            + ". This usually means the ONNX graph was rewritten by fusion; "
            "run the model through onnxsim first."
        )
    return Converter(seven_bit=seven_bit).emit(graph, ranges)


def stats(blob: bytes) -> dict:
    """Header fields, for the CLI and the README's memory numbers."""
    import struct

    f = struct.unpack("<16I", blob[: fmt.HEADER_BYTES])
    return dict(
        version=f[1], flags=f[2], n_tensors=f[3], n_nodes=f[4],
        arena_bytes=f[7], scratch_bytes=f[8], weights_bytes=len(blob) - f[11], total=len(blob),
    )
