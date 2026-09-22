"""Arena planning and format agreement.

The arena is where the "lightweight" claim mostly comes from (doc 8.1), so
the planner is tested for the property that matters — two tensors that are
live at the same time never share bytes — rather than for a particular
packing.
"""

from __future__ import annotations

import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "converter"))

import models  # noqa: E402
from lnn_convert import Reference, convert_model  # noqa: E402
from lnn_convert import format as fmt  # noqa: E402
from lnn_convert import pack  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_rust(*parts: str) -> str:
    with open(os.path.join(ROOT, "crates", "lnn", "src", *parts)) as f:
        return f.read()


class ArenaPlanning(unittest.TestCase):
    def build(self, model, shape, seed=0):
        x = np.random.default_rng(seed).normal(size=(12, *shape)).astype(np.float32)
        return convert_model(model, {"X": x})

    def lifetimes(self, ref: Reference):
        """(first write, last read) per activation, from the node table."""
        first, last = {}, {}
        for i, node in enumerate(ref.nodes):
            for t in node.outputs:
                first.setdefault(t, i)
            for t in node.inputs:
                if ref.tensors[t].kind != fmt.KIND_WEIGHT:
                    last[t] = i
        for t in ref.tensors:
            if t.kind == fmt.KIND_INPUT:
                first[t.index] = -1
            if t.kind == fmt.KIND_OUTPUT:
                last[t.index] = len(ref.nodes)
        return first, last

    def test_live_tensors_never_share_bytes(self):
        blob = self.build(models.mobilenet_block_model(c=16, h=16, w=16), (3, 16, 16))
        ref = Reference(blob)
        first, last = self.lifetimes(ref)
        acts = [t for t in ref.tensors if t.kind != fmt.KIND_WEIGHT]
        for i, a in enumerate(acts):
            for b in acts[i + 1:]:
                fa, la = first.get(a.index, 0), last.get(a.index, 0)
                fb, lb = first.get(b.index, 0), last.get(b.index, 0)
                overlap = fa <= lb and fb <= la
                if not overlap:
                    continue
                a_end = a.offset + a.nbytes
                b_end = b.offset + b.nbytes
                self.assertFalse(
                    a.offset < b_end and b.offset < a_end,
                    f"'{a.name}' [{a.offset}, {a_end}) and '{b.name}' [{b.offset}, {b_end}) "
                    "are live at the same time but overlap in the arena",
                )

    def test_every_activation_fits_inside_the_arena(self):
        blob = self.build(models.mobilenet_block_model(c=16, h=16, w=16), (3, 16, 16))
        ref = Reference(blob)
        for t in ref.tensors:
            if t.kind != fmt.KIND_WEIGHT:
                self.assertLessEqual(t.offset + t.nbytes, ref.arena_bytes, t.name)

    def test_reuse_beats_a_buffer_each(self):
        blob = self.build(models.mobilenet_block_model(c=32, h=32, w=32), (3, 32, 32))
        ref = Reference(blob)
        naive = sum(
            pack.round_up(t.nbytes, 16) for t in ref.tensors if t.kind != fmt.KIND_WEIGHT
        )
        print(f"\n  arena: {ref.arena_bytes / 1024:.1f} KiB vs {naive / 1024:.1f} KiB "
              f"unshared ({naive / ref.arena_bytes:.1f}x)")
        self.assertLess(ref.arena_bytes, naive)

    def test_a_chain_reuses_two_buffers(self):
        # A pure chain of same-sized tensors needs only two buffers at a time,
        # whatever its length; if the planner does not find that, it is not
        # reusing anything.
        model = models.mobilenet_block_model(c=8, h=8, w=8)
        ref = Reference(self.build(model, (3, 8, 8)))
        largest = max(t.nbytes for t in ref.tensors if t.kind != fmt.KIND_WEIGHT)
        self.assertLess(ref.arena_bytes, largest * len(ref.nodes))


class FormatAgreement(unittest.TestCase):
    """The Python writer and the Rust parser share no code, so the constants
    they each hardcode are checked against each other here."""

    def rust(self, name: str) -> int:
        src = read_rust("model.rs")
        m = re.search(rf"pub const {name}: \w+ = ([0-9a-fx_A-F<\s]+);", src)
        if not m:
            self.fail(f"{name} not found in model.rs")
        return int(eval(m.group(1).replace("_", "")))

    def test_layout_constants_match(self):
        self.assertEqual(self.rust("MAGIC"), fmt.MAGIC)
        self.assertEqual(self.rust("VERSION"), fmt.VERSION)
        self.assertEqual(self.rust("HEADER_BYTES"), fmt.HEADER_BYTES)
        self.assertEqual(self.rust("TENSOR_BYTES"), fmt.TENSOR_BYTES)
        self.assertEqual(self.rust("NODE_HEADER_BYTES"), fmt.NODE_HEADER_BYTES)
        self.assertEqual(self.rust("FLAG_PREPACKED"), fmt.FLAG_PREPACKED)
        self.assertEqual(self.rust("FLAG_WEIGHTS_7BIT"), fmt.FLAG_WEIGHTS_7BIT)

    def test_opcodes_match(self):
        src = read_rust("graph.rs")
        for name, value in [
            ("OP_CONV2D", fmt.OP_CONV2D), ("OP_DEPTHWISE_CONV2D", fmt.OP_DEPTHWISE_CONV2D),
            ("OP_FULLY_CONNECTED", fmt.OP_FULLY_CONNECTED), ("OP_ADD", fmt.OP_ADD),
            ("OP_MUL", fmt.OP_MUL), ("OP_MAXPOOL2D", fmt.OP_MAXPOOL2D),
            ("OP_AVGPOOL2D", fmt.OP_AVGPOOL2D), ("OP_SOFTMAX", fmt.OP_SOFTMAX),
            ("OP_RESHAPE", fmt.OP_RESHAPE), ("OP_CONCAT", fmt.OP_CONCAT),
            ("OP_CLAMP", fmt.OP_CLAMP), ("OP_LOOKUP", fmt.OP_LOOKUP),
            ("OP_QUANTIZE", fmt.OP_QUANTIZE), ("OP_DEQUANTIZE", fmt.OP_DEQUANTIZE),
        ]:
            m = re.search(rf"pub const {name}: u16 = (\d+);", src)
            self.assertIsNotNone(m, name)
            self.assertEqual(int(m.group(1)), value, name)

    def test_error_codes_match(self):
        src = read_rust("error.rs")
        found = dict(re.findall(r"(\w+) = (-\d+)", src))
        self.assertEqual(len(found), len(fmt.ERRORS))
        for code in found.values():
            self.assertIn(int(code), fmt.ERRORS)

    def test_packing_constants_match(self):
        src = read_rust("kernels", "gemm", "mod.rs")
        for name, value in [("MR", pack.MR), ("NR", pack.NR), ("MC", pack.MC)]:
            m = re.search(rf"pub const {name}: usize = (\d+);", src)
            self.assertEqual(int(m.group(1)), value, name)


class PackRoundTrip(unittest.TestCase):
    def test_pack_then_unpack_is_the_identity(self):
        rng = np.random.default_rng(3)
        for n, k in [(8, 4), (1, 1), (10, 7), (64, 128), (3, 33)]:
            w = rng.integers(-128, 128, size=(n, k)).astype(np.int8)
            packed = pack.pack_b(w)
            self.assertEqual(len(packed), pack.packed_b_bytes(n, k))

            # Unpack the way reference.py does, independently of pack_b.
            kp, np_ = pack.round_up(k, 4), pack.round_up(n, pack.NR)
            raw = packed.reshape(np_ // pack.NR, kp // 4, pack.NR, 4)
            full = np.ascontiguousarray(raw.transpose(0, 2, 1, 3)).reshape(np_, kp)
            np.testing.assert_array_equal(full[:n, :k], w)
            self.assertTrue(np.all(full[n:, :] == 0), "padding must be zero")
            self.assertTrue(np.all(full[:, k:] == 0), "padding must be zero")


if __name__ == "__main__":
    unittest.main(verbosity=2)
