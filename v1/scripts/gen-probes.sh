#!/usr/bin/env bash
# Regenerate the feature-probe byte arrays in js/src/detect.ts.
#
# The arrays are wat2wasm output for scripts/probes/*.wat. Hand-assembling
# them is how you end up with a probe that validates on every engine and
# proves nothing (design doc 10.5).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAT2WASM="${WAT2WASM:-wat2wasm}"
command -v "$WAT2WASM" >/dev/null || { echo "wat2wasm not found (npm i -g wabt)" >&2; exit 1; }

for name in simd relaxed; do
  "$WAT2WASM" --enable-all "$ROOT/scripts/probes/$name.wat" -o "/tmp/lnn-probe-$name.wasm"
  printf '%s: ' "$name"
  od -An -tu1 -v "/tmp/lnn-probe-$name.wasm" | tr -s ' ' | tr ' ' '\n' | grep -v '^$' | paste -sd, -
done

echo
echo "paste these into js/src/detect.ts"
