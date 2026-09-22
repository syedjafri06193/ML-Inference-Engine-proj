#!/usr/bin/env bash
# Build the three ISA variants and report their sizes (design doc 10.4).
#
# Size is a feature here, so every build prints raw and gzipped bytes and
# scripts/size-report.sh can fail CI on a regression.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/dist"
mkdir -p "$OUT"

WASM_OPT="${WASM_OPT:-wasm-opt}"
have_wasm_opt=1
command -v "$WASM_OPT" >/dev/null 2>&1 || have_wasm_opt=0

# Normally `rustup target add wasm32-unknown-unknown` provides a precompiled
# core. Where it cannot be installed, core is built from source instead; both
# produce the same module.
BUILD_STD=()
# `--print target-libdir` answers even for a target that is not installed, so
# check for the compiled core rlib itself.
libdir="$(rustc --print target-libdir --target wasm32-unknown-unknown 2>/dev/null || true)"
if [ -z "$libdir" ] || ! compgen -G "$libdir/libcore-*.rlib" >/dev/null; then
  echo "note: no prebuilt wasm32 std; building core from source (-Zbuild-std)"
  export RUSTC_BOOTSTRAP=1
  BUILD_STD=(-Z build-std=core)
fi

build() {
  local name=$1 features=$2 extra=${3:-} cargo_features=${4:-}
  # shellcheck disable=SC2086
  RUSTFLAGS="-C target-feature=$features" \
    cargo build --quiet --release --manifest-path "$ROOT/crates/lnn/Cargo.toml" \
      --target wasm32-unknown-unknown "${BUILD_STD[@]}" $cargo_features

  local raw="$ROOT/crates/lnn/target/wasm32-unknown-unknown/release/lnn.wasm"
  if [ "$have_wasm_opt" = 1 ]; then
    # shellcheck disable=SC2086
    # sign-ext and nontrapping-float-to-int are emitted by rustc by default.
    # Both are baseline everywhere simd128 is (Chrome 75+, Safari 15+), so
    # enabling them costs no compatibility.
    "$WASM_OPT" -O3 --enable-simd --enable-bulk-memory --enable-sign-ext \
      --enable-nontrapping-float-to-int --enable-mutable-globals \
      $extra "$raw" -o "$OUT/lnn-$name.wasm"
  else
    echo "warning: wasm-opt not found; shipping unoptimized $name" >&2
    cp "$raw" "$OUT/lnn-$name.wasm"
  fi
  gzip -9 -c "$OUT/lnn-$name.wasm" > "$OUT/lnn-$name.wasm.gz"
  printf "  %-9s raw %7d  gz %7d\n" "$name" \
    "$(stat -c%s "$OUT/lnn-$name.wasm")" "$(stat -c%s "$OUT/lnn-$name.wasm.gz")"
}

echo "building lnn:"
build scalar  "+bulk-memory"
build simd128 "+simd128,+bulk-memory"
build relaxed "+simd128,+relaxed-simd,+bulk-memory" "--enable-relaxed-simd"

# Two extra builds for `bench/bench.mjs` only: they export the GEMM ladder.
# The scalar one matters — with simd128 enabled, LLVM auto-vectorizes even the
# "naive" triple loop, so measuring a scalar baseline in a SIMD build measures
# the compiler, not the kernel.
if [ "${WITH_BENCH:-0}" = 1 ]; then
  build bench "+simd128,+relaxed-simd,+bulk-memory" "--enable-relaxed-simd" "--features bench"
  build bench-scalar "+bulk-memory" "" "--features bench"
fi
