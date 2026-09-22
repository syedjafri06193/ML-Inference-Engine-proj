"""Differential tests (design doc section 14).

Quantized inference fails quietly: a wrong shift gives you a model that is
94% accurate instead of 96% and nothing crashes. So the assertions here are
exact integer equality against an independent implementation, per operator,
for every ISA build — not `allclose`.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models  # noqa: E402
from harness import WasmRunner, available_builds  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "converter"))
from lnn_convert import Reference, convert_model  # noqa: E402

BUILDS = available_builds()


def calib(shape, n=24, seed=0, scale=1.0):
    return np.random.default_rng(seed).normal(scale=scale, size=(n, *shape)).astype(np.float32)


def convert(model, samples, seven_bit=False):
    return convert_model(model, samples, seven_bit=seven_bit, percentile=99.99)


class ExactAgreement(unittest.TestCase):
    """Every build must agree with the reference, and with each other, to the
    last bit. All three implement the same integer arithmetic; a difference
    is a kernel bug, not a rounding difference."""

    def check(self, model, input_shape, seed=0, seven_bit=False, n_test=4, name=""):
        samples = {"X": calib(input_shape, seed=seed)}
        blob = convert(model, samples, seven_bit=seven_bit)
        ref = Reference(blob)
        tests = calib(input_shape, n=n_test, seed=seed + 100)

        for build in BUILDS:
            runner = WasmRunner(build)
            runner.load(blob)
            for i in range(n_test):
                x = tests[i : i + 1]
                want = ref.run([x])
                got = runner.run([x])
                self.assertEqual(len(want), len(got))
                for w, g in zip(want, got):
                    if not np.array_equal(w.ravel(), g.ravel()):
                        bad = np.flatnonzero(w.ravel() != g.ravel())
                        j = bad[0]
                        self.fail(
                            f"{name or model.graph.name} [{build}] sample {i}: "
                            f"{len(bad)}/{w.size} outputs differ; first at {j}: "
                            f"reference {w.ravel()[j]!r} vs wasm {g.ravel()[j]!r}"
                        )
        return blob

    # ---------------------------------------------------------- per op

    def test_conv_shapes(self):
        cases = [
            dict(c=3, h=8, w=8, out=4, k=3, pad=1),
            dict(c=3, h=8, w=8, out=4, k=3, pad=0),            # no padding
            dict(c=1, h=5, w=5, out=1, k=1, pad=0),            # 1x1x1: pointwise path
            dict(c=8, h=7, w=7, out=16, k=1, pad=0),           # pointwise, wide
            dict(c=5, h=6, w=6, out=7, k=3, pad=1, stride=2),  # odd channels, strided
            dict(c=3, h=9, w=9, out=4, k=3, pad=2, dilation=2),
            dict(c=16, h=4, w=4, out=3, k=3, pad=1),           # K not a multiple of 4
            dict(c=3, h=1, w=1, out=4, k=1, pad=0),            # 1x1 spatial
        ]
        for kw in cases:
            with self.subTest(**kw):
                m = models.conv_model(**kw)
                self.check(m, (kw["c"], kw["h"], kw["w"]), name=f"conv {kw}")

    def test_conv_with_fused_activations(self):
        for act in ("relu", "relu6"):
            with self.subTest(act=act):
                m = models.conv_model(c=4, h=6, w=6, out=6, k=3, pad=1, act=act)
                self.check(m, (4, 6, 6), name=f"conv+{act}")

    def test_conv_without_bias(self):
        m = models.conv_model(c=3, h=5, w=5, out=4, k=3, pad=1, bias=False)
        self.check(m, (3, 5, 5), name="conv no bias")

    def test_depthwise(self):
        for c, k, stride, pad in [(8, 3, 1, 1), (16, 3, 2, 1), (5, 3, 1, 0), (32, 5, 1, 2), (3, 3, 1, 1)]:
            with self.subTest(c=c, k=k, stride=stride, pad=pad):
                m = models.depthwise_model(c=c, h=7, w=7, k=k, stride=stride, pad=pad)
                self.check(m, (c, 7, 7), name=f"depthwise c={c} k={k}")

    def test_pools(self):
        for kind in ("MaxPool", "AveragePool", "GlobalAveragePool"):
            with self.subTest(kind=kind):
                m = models.pool_model(kind=kind, c=4, h=8, w=8)
                self.check(m, (4, 8, 8), name=kind)

    def test_avgpool_odd_window(self):
        # A window that does not divide the input: the count varies at the
        # border, which is where average pooling's rounding shows up.
        m = models.pool_model(kind="AveragePool", c=3, h=7, w=7, k=3, stride=2, pad=1)
        self.check(m, (3, 7, 7), name="avgpool 3x3/2 pad 1")

    def test_fully_connected(self):
        for k, n, m_rows in [(16, 10, 1), (64, 32, 1), (13, 7, 1), (8, 8, 4)]:
            with self.subTest(k=k, n=n, rows=m_rows):
                mod = models.gemm_model(k=k, n=n, m=m_rows)
                self.check(mod, (k,) if m_rows == 1 else (m_rows, k), name=f"fc {k}x{n}")

    def test_fc_with_relu(self):
        self.check(models.gemm_model(k=32, n=16, act="relu"), (32,), name="fc+relu")

    def test_add_two_scales(self):
        self.check(models.add_model(c=4, h=4, w=4), (4, 4, 4), name="add")

    def test_mul_channel_broadcast(self):
        self.check(models.mul_model(c=4, h=3, w=3), (4, 3, 3), name="mul")

    def test_concat(self):
        self.check(models.concat_model(c=3, h=4, w=4), (3, 4, 4), name="concat")

    def test_lookup_activations(self):
        for kind in ("Sigmoid", "Tanh", "HardSwish", "HardSigmoid"):
            with self.subTest(kind=kind):
                self.check(models.activation_model(kind), (4, 3, 3), name=kind)

    def test_softmax(self):
        self.check(models.softmax_model(n=10), (10,), name="softmax")
        self.check(models.softmax_model(n=1000), (1000,), name="softmax wide")

    def test_batchnorm_folding(self):
        self.check(models.batchnorm_model(c=4, h=5, w=5), (4, 5, 5), name="conv+bn+relu")

    def test_flatten_into_fc(self):
        self.check(models.flatten_fc_model(c=3, h=4, w=4, n=5), (3, 4, 4), name="flatten+fc")

    def test_saturating_inputs(self):
        # Values far outside the calibrated range, so the accumulator and the
        # requantization both saturate.
        model = models.conv_model(c=4, h=5, w=5, out=8, k=3, pad=1)
        samples = {"X": calib((4, 5, 5), seed=3)}
        blob = convert(model, samples)
        ref = Reference(blob)
        x = np.full((1, 4, 5, 5), 50.0, np.float32)
        x.ravel()[::3] = -50.0
        for build in BUILDS:
            r = WasmRunner(build)
            r.load(blob)
            np.testing.assert_array_equal(ref.run([x])[0].ravel(), r.run([x])[0].ravel())

    # ------------------------------------------------------ whole models

    def test_mobilenet_block_stack(self):
        m = models.mobilenet_block_model(c=16, h=16, w=16)
        self.check(m, (3, 16, 16), n_test=3, name="mobilenet-ish")

    def test_mobilenet_block_stack_7bit(self):
        m = models.mobilenet_block_model(c=16, h=16, w=16)
        self.check(m, (3, 16, 16), n_test=3, seven_bit=True, name="mobilenet-ish 7-bit")


class RelaxedPath(unittest.TestCase):
    def setUp(self):
        if "relaxed" not in BUILDS:
            self.skipTest("relaxed build not present")

    def test_relaxed_build_uses_the_relaxed_kernel_for_7bit_models(self):
        m = models.conv_model(c=8, h=6, w=6, out=8, k=3, pad=1)
        blob = convert(m, {"X": calib((8, 6, 6))}, seven_bit=True)
        r = WasmRunner("relaxed")
        r.load(blob)
        self.assertEqual(r.isa, "relaxed")

    def test_relaxed_build_steps_down_for_8bit_models(self):
        # Full-range weights are not safe for the relaxed instruction (its
        # second operand must keep the top bit clear), so the runtime must
        # fall back rather than produce CPU-dependent numbers.
        m = models.conv_model(c=8, h=6, w=6, out=8, k=3, pad=1)
        blob = convert(m, {"X": calib((8, 6, 6))}, seven_bit=False)
        r = WasmRunner("relaxed")
        r.load(blob)
        self.assertEqual(r.isa, "simd128")

    def test_7bit_weights_stay_in_range(self):
        from lnn_convert import format as fmt

        m = models.mobilenet_block_model(c=8, h=8, w=8)
        blob = convert(m, {"X": calib((3, 8, 8))}, seven_bit=True)
        ref = Reference(blob)
        self.assertTrue(ref.flags & fmt.FLAG_WEIGHTS_7BIT)
        checked = 0
        for t in ref.tensors:
            if t.kind == fmt.KIND_WEIGHT and t.dtype == fmt.DTYPE_I8 and ".w#" in t.name:
                w = ref.weight(t)
                self.assertGreaterEqual(int(w.min()), -63, t.name)
                self.assertLessEqual(int(w.max()), 63, t.name)
                # +64 must leave the top bit clear, which is the actual
                # determinism condition for the relaxed instruction.
                self.assertTrue(np.all((w.astype(np.int32) + 64) <= 127))
                self.assertTrue(np.all((w.astype(np.int32) + 64) >= 0))
                checked += 1
        self.assertGreater(checked, 0)


class Accuracy(unittest.TestCase):
    """Quantized vs fp32 on a real trained model (doc M3's exit criterion)."""

    @classmethod
    def setUpClass(cls):
        cls.model, (xtr, ytr, xte, yte), cls.clf = models.train_digit_mlp()
        cls.xte, cls.yte = xte, yte
        cls.blob = convert(cls.model, {"X": xtr[:300]})

    def test_quantized_matches_float_predictions(self):
        import onnxruntime as ort

        sess = ort.InferenceSession(self.model.SerializeToString(), providers=["CPUExecutionProvider"])
        runner = WasmRunner(BUILDS[-1])
        runner.load(self.blob)

        agree = 0
        q_correct = 0
        f_correct = 0
        n = len(self.xte)
        for i in range(n):
            x = self.xte[i : i + 1]
            ref = sess.run(None, {"X": x})[0].ravel()
            got = runner.run([x])[0]
            agree += int(ref.argmax() == got.argmax())
            q_correct += int(got.argmax() == self.yte[i])
            f_correct += int(ref.argmax() == self.yte[i])

        q_acc, f_acc = q_correct / n, f_correct / n
        print(f"\n  digits MLP: fp32 {f_acc:.3%}, int8 {q_acc:.3%}, top-1 agreement {agree / n:.3%}")
        self.assertGreaterEqual(agree / n, 0.97)
        self.assertGreaterEqual(q_acc, f_acc - 0.01)

    def test_probabilities_are_close_to_float(self):
        import onnxruntime as ort

        sess = ort.InferenceSession(self.model.SerializeToString(), providers=["CPUExecutionProvider"])
        runner = WasmRunner(BUILDS[-1])
        runner.load(self.blob)
        worst = 0.0
        for i in range(100):
            x = self.xte[i : i + 1]
            ref = sess.run(None, {"X": x})[0].ravel()
            got = runner.run([x])[0]
            worst = max(worst, float(np.abs(ref - got).max()))
        print(f"  worst softmax probability error: {worst:.4f}")
        # 1/256 is the quantization step of the softmax output itself.
        self.assertLess(worst, 0.05)


class Determinism(unittest.TestCase):
    def test_all_builds_agree_with_each_other(self):
        if len(BUILDS) < 2:
            self.skipTest("need at least two builds")
        m = models.mobilenet_block_model(c=8, h=12, w=12)
        blob = convert(m, {"X": calib((3, 12, 12))}, seven_bit=True)
        x = calib((3, 12, 12), n=1, seed=77)
        outs = []
        for build in BUILDS:
            r = WasmRunner(build)
            r.load(blob)
            outs.append(r.run([x])[0])
        for i in range(1, len(outs)):
            np.testing.assert_array_equal(outs[0], outs[i], f"{BUILDS[0]} vs {BUILDS[i]}")

    def test_repeated_runs_are_identical(self):
        m = models.conv_model(c=4, h=6, w=6, out=8, k=3, pad=1, act="relu")
        blob = convert(m, {"X": calib((4, 6, 6))})
        r = WasmRunner(BUILDS[-1])
        r.load(blob)
        x = calib((4, 6, 6), n=1, seed=5)
        first = r.run([x])[0]
        for _ in range(5):
            np.testing.assert_array_equal(first, r.run([x])[0])

    def test_run_allocates_nothing(self):
        # Doc 8.2 and M6: `run()` must not allocate. Memory used after the
        # first inference must not move, ever.
        m = models.mobilenet_block_model(c=8, h=12, w=12)
        blob = convert(m, {"X": calib((3, 12, 12))})
        r = WasmRunner(BUILDS[-1])
        r.load(blob)
        x = calib((3, 12, 12), n=1, seed=1)
        r.run([x])
        before = r.memory_used()
        for _ in range(10):
            r.run([x])
        self.assertEqual(before, r.memory_used())


if __name__ == "__main__":
    unittest.main(verbosity=2)
