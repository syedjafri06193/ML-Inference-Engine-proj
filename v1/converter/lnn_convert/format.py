"""The .lnn container: constants and the writer (design doc section 7).

The runtime parses this by reading fields at fixed offsets, so this file and
`crates/lnn/src/model.rs` have to agree exactly. `tests/test_format.py`
checks the constants against the Rust source so a change to one that isn't
mirrored in the other fails the build rather than the model.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterable

MAGIC = 0x314E4E4C  # "LNN1"
VERSION = 1

HEADER_BYTES = 64
TENSOR_BYTES = 40
NODE_HEADER_BYTES = 8

FLAG_PREPACKED = 1 << 0
FLAG_WEIGHTS_7BIT = 1 << 1

DTYPE_I8, DTYPE_U8, DTYPE_I32, DTYPE_F32 = 0, 1, 2, 3
DTYPE_BYTES = {DTYPE_I8: 1, DTYPE_U8: 1, DTYPE_I32: 4, DTYPE_F32: 4}

KIND_WEIGHT, KIND_ACTIVATION, KIND_INPUT, KIND_OUTPUT = 0, 1, 2, 3

# Opcodes. 0-14 follow the design doc's table; the doc's single
# "Quantize/Dequantize" entry is split into two opcodes because they are two
# kernels, and Slice/Resize keep their numbers reserved but are unimplemented.
OP_CONV2D = 0
OP_DEPTHWISE_CONV2D = 1
OP_FULLY_CONNECTED = 2
OP_MATMUL = 3
OP_ADD = 4
OP_MUL = 5
OP_MAXPOOL2D = 6
OP_AVGPOOL2D = 7
OP_SOFTMAX = 8
OP_RESHAPE = 9
OP_TRANSPOSE = 10
OP_CONCAT = 11
OP_CLAMP = 12
OP_LOOKUP = 13
OP_LAYERNORM = 14
OP_QUANTIZE = 15
OP_DEQUANTIZE = 16

OP_NAMES = {
    OP_CONV2D: "Conv2D",
    OP_DEPTHWISE_CONV2D: "DepthwiseConv2D",
    OP_FULLY_CONNECTED: "FullyConnected",
    OP_MATMUL: "MatMul",
    OP_ADD: "Add",
    OP_MUL: "Mul",
    OP_MAXPOOL2D: "MaxPool2D",
    OP_AVGPOOL2D: "AvgPool2D",
    OP_SOFTMAX: "Softmax",
    OP_RESHAPE: "Reshape",
    OP_TRANSPOSE: "Transpose",
    OP_CONCAT: "Concat",
    OP_CLAMP: "Clamp",
    OP_LOOKUP: "LookupActivation",
    OP_LAYERNORM: "LayerNorm",
    OP_QUANTIZE: "Quantize",
    OP_DEQUANTIZE: "Dequantize",
}

# Error codes returned by lnn_load / lnn_run, mirroring crates/lnn/src/error.rs.
ERRORS = {
    -1: "the file is truncated",
    -2: "the model buffer is not 16-byte aligned",
    -3: "not a .lnn file (bad magic)",
    -4: "unsupported format version",
    -5: "malformed header",
    -6: "malformed tensor descriptor",
    -7: "malformed node",
    -8: "the model uses an operator this build does not implement",
    -9: "shapes are inconsistent",
    -10: "an attribute is out of range",
    -11: "out of memory",
    -12: "no model is loaded",
    -13: "the scratch buffer declared in the header is too small",
    -14: "a tensor has the wrong dtype",
}

ALIGN = 16


def align_up(x: int, to: int = ALIGN) -> int:
    return (x + to - 1) // to * to


@dataclass
class TensorDesc:
    name: str
    dtype: int
    kind: int
    dims: tuple
    offset: int = 0
    scale_off: int = 0
    zero_point: int = 0
    n_scales: int = 0
    nbytes: int = 0
    name_off: int = 0

    def pack(self) -> bytes:
        dims = list(self.dims) + [0] * (4 - len(self.dims))
        return struct.pack(
            "<4B4I2IiII",
            self.dtype,
            len(self.dims),
            self.kind,
            self.n_scales,
            *dims,
            self.offset,
            self.scale_off,
            self.zero_point,
            self.name_off,
            self.nbytes,
        )


@dataclass
class NodeDesc:
    opcode: int
    inputs: list
    outputs: list
    attrs: list = field(default_factory=list)

    def pack(self) -> bytes:
        body = b"".join(struct.pack("<I", i) for i in self.inputs)
        body += b"".join(struct.pack("<I", o) for o in self.outputs)
        attrs = b"".join(struct.pack("<i", int(a)) for a in self.attrs)
        head = struct.pack(
            "<HBBHH", self.opcode, len(self.inputs), len(self.outputs), len(attrs), 0
        )
        return head + body + attrs


def build(
    tensors: Iterable[TensorDesc],
    nodes: Iterable[NodeDesc],
    weight_blob: bytes,
    strings: bytes,
    arena_bytes: int,
    scratch_bytes: int,
    flags: int,
) -> bytes:
    """Assemble a .lnn file. Offsets inside `tensors` are already resolved."""
    tensors = list(tensors)
    nodes = list(nodes)
    n_inputs = sum(1 for t in tensors if t.kind == KIND_INPUT)
    n_outputs = sum(1 for t in tensors if t.kind == KIND_OUTPUT)

    tensor_blob = b"".join(t.pack() for t in tensors)
    node_blob = b"".join(n.pack() for n in nodes)

    tensors_off = HEADER_BYTES
    nodes_off = align_up(tensors_off + len(tensor_blob), 4)
    strings_off = align_up(nodes_off + len(node_blob), 4)
    weights_off = align_up(strings_off + len(strings), ALIGN)

    header = struct.pack(
        "<13I3I",
        MAGIC,
        VERSION,
        flags,
        len(tensors),
        len(nodes),
        n_inputs,
        n_outputs,
        arena_bytes,
        scratch_bytes,
        tensors_off,
        nodes_off,
        weights_off,
        strings_off,
        len(node_blob),  # doc's reserved[0]: the node table has no other length
        0,
        0,
    )
    assert len(header) == HEADER_BYTES

    out = bytearray(header)
    out += tensor_blob
    out += b"\0" * (nodes_off - len(out))
    out += node_blob
    out += b"\0" * (strings_off - len(out))
    out += strings
    out += b"\0" * (weights_off - len(out))
    out += weight_blob
    return bytes(out)
