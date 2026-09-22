"""Run a .lnn model through the real .wasm, outside a browser.

`wasmtime` rather than a headless browser (doc 14.3): the whole suite runs in
CI in seconds and never flakes. A handful of browser-only behaviours —
feature detection, memory growth, the Cache API — are covered by the Node
tests in `js/test`.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
sys.path.insert(0, os.path.join(ROOT, "converter"))

BUILDS = ("scalar", "simd128", "relaxed")

ERRORS = {
    -1: "truncated", -2: "misaligned", -3: "bad magic", -4: "bad version",
    -5: "bad header", -6: "bad tensor", -7: "bad node", -8: "unsupported op",
    -9: "bad shape", -10: "bad attribute", -11: "out of memory",
    -12: "not loaded", -13: "scratch too small", -14: "bad dtype",
}


class LnnError(RuntimeError):
    def __init__(self, code: int):
        super().__init__(f"lnn error {code}: {ERRORS.get(code, 'unknown')}")
        self.code = code


class WasmRunner:
    """One wasm instance with one model loaded."""

    def __init__(self, build: str = "simd128"):
        from wasmtime import Config, Engine, Instance, Module, Store

        cfg = Config()
        cfg.wasm_simd = True
        try:
            cfg.wasm_relaxed_simd = True
        except AttributeError:  # older wasmtime-py
            if build == "relaxed":
                raise unittest_skip("wasmtime build has no relaxed-SIMD support")
        path = os.path.join(DIST, f"lnn-{build}.wasm")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run scripts/build.sh first")
        self.build = build
        self.store = Store(Engine(cfg))
        self.instance = Instance(self.store, Module.from_file(self.store.engine, path), [])
        self.ex = self.instance.exports(self.store)
        self.memory = self.ex["memory"]

    # -- raw ABI ---------------------------------------------------------

    def _call(self, name, *args):
        return self.ex[name](self.store, *args)

    def write(self, ptr: int, data: bytes) -> None:
        self.memory.write(self.store, data, ptr)

    def read(self, ptr: int, n: int) -> bytes:
        return bytes(self.memory.read(self.store, ptr, ptr + n))

    # -- model -----------------------------------------------------------

    def load(self, blob: bytes) -> None:
        ptr = self._call("lnn_alloc", len(blob))
        if ptr == 0:
            raise LnnError(-11)
        self.write(ptr, blob)
        code = self._call("lnn_load", ptr, len(blob))
        if code != 0:
            raise LnnError(code)

    def try_load(self, blob: bytes) -> int:
        """Load without raising — for the fuzz tests."""
        ptr = self._call("lnn_alloc", len(blob))
        if ptr == 0:
            return -11
        self.write(ptr, blob)
        return self._call("lnn_load", ptr, len(blob))

    @property
    def isa(self) -> str:
        return BUILDS[self._call("lnn_isa")]

    def memory_used(self) -> int:
        return self._call("lnn_memory_used")

    def run(self, inputs) -> list:
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        for i, arr in enumerate(inputs):
            ptr = self._call("lnn_input_ptr", i)
            n = self._call("lnn_input_len", i)
            data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
            if len(data) != n:
                raise ValueError(f"input {i}: model wants {n} bytes, got {len(data)}")
            self.write(ptr, data)
        code = self._call("lnn_run")
        if code != 0:
            raise LnnError(code)
        outs = []
        i = 0
        while True:
            n = self._call("lnn_output_len", i)
            if n == 0:
                break
            ptr = self._call("lnn_output_ptr", i)
            outs.append(np.frombuffer(self.read(ptr, n), dtype=np.float32).copy())
            i += 1
        return outs

    def tensor(self, index: int, nbytes: int, dtype=np.int8) -> np.ndarray:
        """Read any tensor — the layer-bisection hook from doc 17.1."""
        ptr = self._call("lnn_tensor_ptr", index)
        return np.frombuffer(self.read(ptr, nbytes), dtype=dtype).copy()


def unittest_skip(msg):
    import unittest

    return unittest.SkipTest(msg)


def available_builds():
    return [b for b in BUILDS if os.path.exists(os.path.join(DIST, f"lnn-{b}.wasm"))]
