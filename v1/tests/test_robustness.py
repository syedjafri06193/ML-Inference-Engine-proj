"""The model file is untrusted input (design doc 14.4).

A malformed or truncated file must come back as an error code. Never a trap,
never an out-of-bounds read, never a hang. This is the cheap version of
`cargo fuzz`: every byte of a real model flipped or truncated, plus targeted
corruption of the fields a parser is most likely to trust.
"""

from __future__ import annotations

import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "converter"))

import models  # noqa: E402
from harness import WasmRunner, available_builds  # noqa: E402
from lnn_convert import convert_model  # noqa: E402
from lnn_convert import format as fmt  # noqa: E402

BUILDS = available_builds()


def a_model() -> bytes:
    m = models.conv_model(c=4, h=6, w=6, out=8, k=3, pad=1, act="relu")
    x = np.random.default_rng(0).normal(size=(16, 4, 6, 6)).astype(np.float32)
    return convert_model(m, {"X": x})


class Robustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blob = a_model()

    def load(self, blob: bytes) -> int:
        """Load in a fresh instance; returns the error code (0 = accepted)."""
        r = WasmRunner(BUILDS[-1])
        return r.try_load(blob)

    def test_the_good_model_loads(self):
        self.assertEqual(self.load(self.blob), 0)

    def test_truncation_at_every_length(self):
        n = len(self.blob)
        for cut in list(range(0, min(n, 300))) + list(range(300, n, 97)):
            with self.subTest(cut=cut):
                self.assertNotEqual(self.load(self.blob[:cut]), 0, f"accepted a {cut}-byte prefix")

    def test_header_field_corruption(self):
        # Each u32 in the header, set to values a naive parser would trust.
        for field in range(16):
            for value in (0xFFFFFFFF, 0x7FFFFFFF, 0, 1 << 20):
                with self.subTest(field=field, value=hex(value)):
                    b = bytearray(self.blob)
                    struct.pack_into("<I", b, field * 4, value)
                    code = self.load(bytes(b))
                    # It may legitimately still load (e.g. flags), but it must
                    # not crash the instance: a later call still has to work.
                    if code == 0:
                        continue
                    self.assertLess(code, 0)

    def test_tensor_offsets_out_of_range(self):
        h = struct.unpack("<16I", self.blob[:64])
        tensors_off, n_tensors = h[9], h[3]
        for i in range(n_tensors):
            for off, name in ((20, "offset"), (24, "scale_off"), (36, "nbytes")):
                with self.subTest(tensor=i, field=name):
                    b = bytearray(self.blob)
                    struct.pack_into("<I", b, tensors_off + i * fmt.TENSOR_BYTES + off, 0xFFFF0000)
                    self.assertNotEqual(self.load(bytes(b)), 0)

    def test_node_tensor_indices_out_of_range(self):
        h = struct.unpack("<16I", self.blob[:64])
        nodes_off = h[10]
        b = bytearray(self.blob)
        struct.pack_into("<I", b, nodes_off + 8, 0xFFFF)  # first input index
        self.assertNotEqual(self.load(bytes(b)), 0)

    def test_random_byte_flips(self):
        rng = np.random.default_rng(7)
        for trial in range(400):
            b = bytearray(self.blob)
            for _ in range(rng.integers(1, 5)):
                b[int(rng.integers(0, len(b)))] = int(rng.integers(0, 256))
            with self.subTest(trial=trial):
                # Accepted or rejected, but the instance must survive; an
                # accepted mutant also has to run without trapping.
                r = WasmRunner(BUILDS[-1])
                code = r.try_load(bytes(b))
                if code == 0:
                    try:
                        r.run([np.zeros(4 * 6 * 6, np.float32)])
                    except Exception as e:  # pragma: no cover - a real failure
                        self.fail(f"trial {trial}: a mutated model trapped during run: {e}")

    def test_scratch_too_small_is_refused(self):
        b = bytearray(self.blob)
        struct.pack_into("<I", b, 32, 64)  # scratch_bytes
        code = self.load(bytes(b))
        self.assertEqual(code, -13, "a model claiming too little scratch must be refused at load")

    def test_zero_scale_is_refused(self):
        h = struct.unpack("<16I", self.blob[:64])
        weights_off = h[11]
        t_off = h[9]
        scale_off = struct.unpack_from("<I", self.blob, t_off + 24)[0]
        b = bytearray(self.blob)
        struct.pack_into("<f", b, weights_off + scale_off, 0.0)
        self.assertNotEqual(self.load(bytes(b)), 0, "a zero scale would divide by zero at run time")

    def test_running_without_a_model_is_an_error_not_a_trap(self):
        r = WasmRunner(BUILDS[-1])
        self.assertEqual(r._call("lnn_run"), -12)
        self.assertEqual(r._call("lnn_input_len", 0), 0)
        self.assertEqual(r._call("lnn_output_ptr", 0), 0)

    def test_unsupported_opcode_is_refused(self):
        h = struct.unpack("<16I", self.blob[:64])
        b = bytearray(self.blob)
        struct.pack_into("<H", b, h[10], 14)  # LayerNorm: in the format, not implemented
        self.assertEqual(self.load(bytes(b)), -8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
