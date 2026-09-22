"""A NumPy interpreter for .lnn files — the oracle the wasm runtime is
tested against (design doc section 14).

This is deliberately a *second* implementation: it parses the container by
reading offsets out of the bytes (not by reusing the writer's structures) and
recomputes every kernel from the arithmetic in section 5. The conformance
suite asserts the two agree exactly, so a bug has to be made twice, in two
languages, to escape.
"""

from __future__ import annotations

import struct

import numpy as np

from . import format as fmt
from .pack import NR, round_up

I32 = np.int64  # accumulate in int64, saturate to int32 where the runtime does


# ------------------------------------------------------------- fixed point


def srdhm(a, b):
    """gemmlowp SaturatingRoundingDoublingHighMul, elementwise."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    prod = a * b
    nudge = np.where(prod >= 0, 1 << 30, 1 - (1 << 30)).astype(np.int64)
    t = prod + nudge
    # Truncating division, matching C's `/`: numpy's `//` floors.
    out = np.where(t >= 0, t // (1 << 31), -((-t) // (1 << 31)))
    out = np.where((a == -(2**31)) & (b == -(2**31)), 2**31 - 1, out)
    return out.astype(np.int64)


def rounding_shift_right(x, shift):
    x = np.asarray(x, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    mask = (np.int64(1) << shift) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (x < 0).astype(np.int64)
    return (x >> shift) + (remainder > threshold).astype(np.int64)


def mbqm(x, mult, shift):
    """TFLite MultiplyByQuantizedMultiplier with a signed exponent."""
    x = np.asarray(x, dtype=np.int64)
    mult = np.asarray(mult, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    shifted = wrap_i32(x * (np.int64(1) << left))
    return rounding_shift_right(srdhm(shifted, mult), right)


def wrap_i32(x):
    """Two's-complement wrap, matching Rust's `wrapping_mul`."""
    return ((np.asarray(x, dtype=np.int64) + 2**31) % 2**32 - 2**31).astype(np.int64)


def requantize(acc, mult, shift, zp, lo, hi):
    return np.clip(mbqm(acc, mult, shift) + zp, lo, hi).astype(np.int8)


def round_half_away_f32(x):
    """Rust's `round_half_away` on f32: truncate, then nudge."""
    x = np.asarray(x, dtype=np.float32)
    t = x.astype(np.int32)
    frac = x - t.astype(np.float32)
    return (t + (frac >= np.float32(0.5)).astype(np.int32) - (frac <= np.float32(-0.5)).astype(np.int32)).astype(np.int64)


# ---------------------------------------------------------------- container


class Tensor:
    __slots__ = ("index", "dtype", "rank", "kind", "n_scales", "dims", "offset", "scale_off", "zero_point", "name_off", "nbytes", "name")

    @property
    def shape(self):
        return tuple(self.dims[: self.rank])

    @property
    def numel(self):
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


class Node:
    __slots__ = ("opcode", "inputs", "outputs", "attrs")


class Reference:
    def __init__(self, blob: bytes):
        self.blob = blob
        h = struct.unpack("<16I", blob[: fmt.HEADER_BYTES])
        if h[0] != fmt.MAGIC:
            raise ValueError("not a .lnn file")
        if h[1] != fmt.VERSION:
            raise ValueError(f"unsupported version {h[1]}")
        (
            _, _, self.flags, n_tensors, n_nodes, self.n_inputs, self.n_outputs,
            self.arena_bytes, self.scratch_bytes, tensors_off, nodes_off,
            self.weights_off, strings_off, nodes_bytes, _, _,
        ) = h

        self.tensors: list[Tensor] = []
        for i in range(n_tensors):
            o = tensors_off + i * fmt.TENSOR_BYTES
            f = struct.unpack("<4B4I2IiII", blob[o : o + fmt.TENSOR_BYTES])
            t = Tensor()
            t.index = i
            t.dtype, t.rank, t.kind, t.n_scales = f[0], f[1], f[2], f[3]
            t.dims = f[4:8]
            t.offset, t.scale_off, t.zero_point, t.name_off, t.nbytes = f[8], f[9], f[10], f[11], f[12]
            end = blob.index(b"\0", strings_off + t.name_off)
            t.name = blob[strings_off + t.name_off : end].decode()
            self.tensors.append(t)

        self.nodes: list[Node] = []
        pos, end = nodes_off, nodes_off + nodes_bytes
        while pos < end:
            opcode, n_in, n_out, attr_len, _ = struct.unpack("<HBBHH", blob[pos : pos + 8])
            body = pos + 8
            n = Node()
            n.opcode = opcode
            n.inputs = list(struct.unpack(f"<{n_in}I", blob[body : body + 4 * n_in])) if n_in else []
            body += 4 * n_in
            n.outputs = list(struct.unpack(f"<{n_out}I", blob[body : body + 4 * n_out]))
            body += 4 * n_out
            n.attrs = list(struct.unpack(f"<{attr_len // 4}i", blob[body : body + attr_len])) if attr_len else []
            self.nodes.append(n)
            pos = body + attr_len
        if len(self.nodes) != n_nodes:
            raise ValueError("node table length disagrees with the header")

        self.buffers: dict[int, np.ndarray] = {}

    # -------------------------------------------------------------- access

    def weight_bytes(self, t: Tensor) -> bytes:
        start = self.weights_off + t.offset
        return self.blob[start : start + t.nbytes]

    def weight(self, t: Tensor) -> np.ndarray:
        dt = {fmt.DTYPE_I8: np.int8, fmt.DTYPE_U8: np.uint8, fmt.DTYPE_I32: np.int32, fmt.DTYPE_F32: np.float32}[t.dtype]
        return np.frombuffer(self.weight_bytes(t), dtype=dt)

    def scales(self, t: Tensor) -> np.ndarray:
        o = self.weights_off + t.scale_off
        return np.frombuffer(self.blob[o : o + 4 * max(t.n_scales, 1)], dtype=np.float32)

    def scale(self, t: Tensor) -> float:
        return float(self.scales(t)[0])

    def value(self, idx: int) -> np.ndarray:
        t = self.tensors[idx]
        if t.kind == fmt.KIND_WEIGHT:
            return self.weight(t)
        return self.buffers[idx]

    def io_indices(self, kind: int) -> list[int]:
        return [t.index for t in self.tensors if t.kind == kind]

    def unpack_b(self, t: Tensor, n: int, k: int) -> np.ndarray:
        """Inverse of `pack.pack_b` — also the check that the packing is what
        the kernel believes it is."""
        kp, np_ = round_up(k, 4), round_up(n, NR)
        raw = self.weight(t)[: np_ * kp].reshape(np_ // NR, kp // 4, NR, 4)
        full = np.ascontiguousarray(raw.transpose(0, 2, 1, 3)).reshape(np_, kp)
        return full[:n, :k].astype(np.int64)

    # --------------------------------------------------------------- kernels

    def run(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        self.buffers = {}
        for i, idx in enumerate(self.io_indices(fmt.KIND_INPUT)):
            t = self.tensors[idx]
            self.buffers[idx] = np.asarray(inputs[i], dtype=np.float32).reshape(t.shape)

        for node in self.nodes:
            self._exec(node)

        return [self.buffers[i].copy() for i in self.io_indices(fmt.KIND_OUTPUT)]

    def _exec(self, node: Node) -> None:
        op = node.opcode
        out_t = self.tensors[node.outputs[0]]
        a = node.attrs

        if op == fmt.OP_QUANTIZE:
            x = self.value(node.inputs[0]).astype(np.float32)
            v = x / np.float32(self.scale(out_t))
            v = np.clip(v, np.float32(-1e6), np.float32(1e6))
            v = np.where(np.isnan(x), np.float32(0.0), v)
            q = round_half_away_f32(v) + out_t.zero_point
            q = np.where(np.isnan(x).ravel().reshape(q.shape), out_t.zero_point, q)
            self.buffers[node.outputs[0]] = np.clip(q, -128, 127).astype(np.int8).reshape(out_t.shape)
        elif op == fmt.OP_DEQUANTIZE:
            xt = self.tensors[node.inputs[0]]
            x = self.value(node.inputs[0]).astype(np.int32)
            self.buffers[node.outputs[0]] = (np.float32(self.scale(xt)) * (x - xt.zero_point).astype(np.float32)).reshape(out_t.shape)
        elif op in (fmt.OP_CONV2D, fmt.OP_DEPTHWISE_CONV2D):
            self._conv(node, depthwise=op == fmt.OP_DEPTHWISE_CONV2D)
        elif op == fmt.OP_FULLY_CONNECTED:
            self._fc(node)
        elif op == fmt.OP_ADD:
            x1t, x2t = self.tensors[node.inputs[0]], self.tensors[node.inputs[1]]
            x1 = self.value(node.inputs[0]).astype(np.int64).ravel()
            x2 = self.value(node.inputs[1]).astype(np.int64).ravel()
            left = a[0]
            s1 = mbqm((x1 - x1t.zero_point) << left, a[1], a[2])
            s2 = mbqm((x2 - x2t.zero_point) << left, a[3], a[4])
            v = mbqm(s1 + s2, a[5], a[6]) + out_t.zero_point
            self.buffers[node.outputs[0]] = np.clip(v, a[7], a[8]).astype(np.int8).reshape(out_t.shape)
        elif op == fmt.OP_MUL:
            x1t, x2t = self.tensors[node.inputs[0]], self.tensors[node.inputs[1]]
            x1 = self.value(node.inputs[0]).astype(np.int64).ravel()
            x2 = self.value(node.inputs[1]).astype(np.int64).ravel()
            if x2.size != x1.size:
                x2 = np.resize(x2, x1.size) if x2.size > 1 else np.full(x1.size, x2[0], np.int64)
            acc = (x1 - x1t.zero_point) * (x2 - x2t.zero_point)
            v = mbqm(acc, a[0], a[1]) + out_t.zero_point
            self.buffers[node.outputs[0]] = np.clip(v, a[2], a[3]).astype(np.int8).reshape(out_t.shape)
        elif op in (fmt.OP_MAXPOOL2D, fmt.OP_AVGPOOL2D):
            self._pool(node, average=op == fmt.OP_AVGPOOL2D)
        elif op == fmt.OP_SOFTMAX:
            xt = self.tensors[node.inputs[0]]
            x = self.value(node.inputs[0]).astype(np.int32).reshape(-1, xt.dims[xt.rank - 1])
            table = self.weight(self.tensors[node.inputs[1]]).astype(np.float32)
            out = np.zeros(x.shape, np.int8)
            for r in range(x.shape[0]):
                idx = 255 + x[r] - x[r].max()
                e = table[idx]
                s = np.float32(0.0)
                for v in e:  # sequential f32 adds, same order as the runtime
                    s = np.float32(s + v)
                q = round_half_away_f32((e / s).astype(np.float32) * np.float32(256.0)) - 128
                out[r] = np.clip(q, -128, 127).astype(np.int8)
            self.buffers[node.outputs[0]] = out.reshape(out_t.shape)
        elif op == fmt.OP_RESHAPE:
            self.buffers[node.outputs[0]] = self.value(node.inputs[0]).reshape(out_t.shape).copy()
        elif op == fmt.OP_CLAMP:
            x = self.value(node.inputs[0]).astype(np.int32)
            self.buffers[node.outputs[0]] = np.clip(x, a[0], a[1]).astype(np.int8).reshape(out_t.shape)
        elif op == fmt.OP_LOOKUP:
            x = self.value(node.inputs[0]).astype(np.int32).ravel()
            table = self.weight(self.tensors[node.inputs[1]])
            self.buffers[node.outputs[0]] = table[x + 128].reshape(out_t.shape).copy()
        elif op == fmt.OP_CONCAT:
            c_out = out_t.dims[out_t.rank - 1]
            outer = out_t.numel // c_out
            out = np.zeros((outer, c_out), np.int8)
            off = 0
            for i, idx in enumerate(node.inputs):
                xt = self.tensors[idx]
                c = xt.dims[xt.rank - 1]
                x = self.value(idx).astype(np.int64).reshape(outer, c)
                mult, shift = a[2 * i], a[2 * i + 1]
                if mult == 0:
                    out[:, off : off + c] = x.astype(np.int8)
                else:
                    v = mbqm(x - xt.zero_point, mult, shift) + out_t.zero_point
                    out[:, off : off + c] = np.clip(v, -128, 127).astype(np.int8)
                off += c
            self.buffers[node.outputs[0]] = out.reshape(out_t.shape)
        else:
            raise ValueError(f"reference: unimplemented opcode {op} ({fmt.OP_NAMES.get(op, '?')})")

    # ------------------------------------------------------------- helpers

    def _geom(self, node: Node, xt: Tensor, yt: Tensor):
        a = node.attrs
        _, h, w, c = xt.dims[: xt.rank] if xt.rank == 4 else (1, 1, 1, xt.dims[xt.rank - 1])
        _, oh, ow, _ = yt.dims[: yt.rank] if yt.rank == 4 else (1, 1, 1, yt.dims[yt.rank - 1])
        return dict(h=h, w=w, c=c, oh=oh, ow=ow, kh=a[0], kw=a[1], sh=a[2], sw=a[3], pt=a[4], pl=a[5], dh=a[8] if len(a) > 8 else 1, dw=a[9] if len(a) > 9 else 1)

    def _im2col(self, x, g, zp):
        h, w, c, kh, kw = g["h"], g["w"], g["c"], g["kh"], g["kw"]
        padded = np.full((h + 2 * 64, w + 2 * 64, c), zp, np.int64)  # generous pad frame
        padded[64 : 64 + h, 64 : 64 + w] = x
        rows = np.zeros((g["oh"] * g["ow"], kh * kw * c), np.int64)
        for oy in range(g["oh"]):
            for ox in range(g["ow"]):
                patch = []
                for ky in range(kh):
                    iy = 64 + oy * g["sh"] + ky * g["dh"] - g["pt"]
                    for kx in range(kw):
                        ix = 64 + ox * g["sw"] + kx * g["dw"] - g["pl"]
                        patch.append(padded[iy, ix])
                rows[oy * g["ow"] + ox] = np.concatenate(patch)
        return rows

    def _conv(self, node: Node, depthwise: bool) -> None:
        xt, wt = self.tensors[node.inputs[0]], self.tensors[node.inputs[1]]
        yt = self.tensors[node.outputs[0]]
        bias = self.weight(self.tensors[node.inputs[2]]).astype(np.int64)
        mult = self.weight(self.tensors[node.inputs[3]]).astype(np.int64)
        shift = self.weight(self.tensors[node.inputs[4]]).astype(np.int64)
        a = node.attrs
        g = self._geom(node, xt, yt)
        x = self.value(node.inputs[0]).astype(np.int64).reshape(g["h"], g["w"], g["c"])
        n_out = yt.dims[yt.rank - 1]

        if depthwise:
            w = self.weight(wt).astype(np.int64).reshape(g["kh"], g["kw"], g["c"])
            out = np.zeros((g["oh"] * g["ow"], n_out), np.int64)
            for oy in range(g["oh"]):
                for ox in range(g["ow"]):
                    acc = bias[:n_out].copy()
                    for ky in range(g["kh"]):
                        iy = oy * g["sh"] + ky * g["dh"] - g["pt"]
                        if iy < 0 or iy >= g["h"]:
                            continue
                        for kx in range(g["kw"]):
                            ix = ox * g["sw"] + kx * g["dw"] - g["pl"]
                            if ix < 0 or ix >= g["w"]:
                                continue
                            acc += (x[iy, ix] - xt.zero_point) * w[ky, kx]
                    out[oy * g["ow"] + ox] = acc
        else:
            k = g["kh"] * g["kw"] * g["c"]
            rows = self._im2col(x, g, xt.zero_point)
            w = self.unpack_b(wt, n_out, k)
            out = rows @ w.T + bias[:n_out]

        y = requantize(out, mult[:n_out], shift[:n_out], yt.zero_point, a[10], a[11])
        self.buffers[node.outputs[0]] = y.reshape(yt.shape)

    def _fc(self, node: Node) -> None:
        xt, wt = self.tensors[node.inputs[0]], self.tensors[node.inputs[1]]
        yt = self.tensors[node.outputs[0]]
        n, k = wt.dims[0], wt.dims[1]
        x = self.value(node.inputs[0]).astype(np.int64).reshape(-1, k)
        w = self.unpack_b(wt, n, k)
        bias = self.weight(self.tensors[node.inputs[2]]).astype(np.int64)[:n]
        mult = self.weight(self.tensors[node.inputs[3]]).astype(np.int64)[:n]
        shift = self.weight(self.tensors[node.inputs[4]]).astype(np.int64)[:n]
        acc = x @ w.T + bias
        y = requantize(acc, mult, shift, yt.zero_point, node.attrs[0], node.attrs[1])
        self.buffers[node.outputs[0]] = y.reshape(yt.shape)

    def _pool(self, node: Node, average: bool) -> None:
        xt, yt = self.tensors[node.inputs[0]], self.tensors[node.outputs[0]]
        a = node.attrs
        g = self._geom(node, xt, yt)
        g["dh"] = g["dw"] = 1
        x = self.value(node.inputs[0]).astype(np.int64).reshape(g["h"], g["w"], g["c"])
        out = np.zeros((g["oh"] * g["ow"], g["c"]), np.int64)
        for oy in range(g["oh"]):
            for ox in range(g["ow"]):
                vals = []
                for ky in range(g["kh"]):
                    iy = oy * g["sh"] + ky - g["pt"]
                    if iy < 0 or iy >= g["h"]:
                        continue
                    for kx in range(g["kw"]):
                        ix = ox * g["sw"] + kx - g["pl"]
                        if ix < 0 or ix >= g["w"]:
                            continue
                        vals.append(x[iy, ix])
                stack = np.stack(vals) if vals else np.zeros((1, g["c"]), np.int64)
                if average:
                    s = stack.sum(0)
                    cnt = len(vals)
                    v = np.where(s > 0, (s + cnt // 2) // cnt, -((-s + cnt // 2) // cnt))
                else:
                    v = stack.max(0)
                out[oy * g["ow"] + ox] = v
        self.buffers[node.outputs[0]] = np.clip(out, a[8], a[9]).astype(np.int8).reshape(yt.shape)
