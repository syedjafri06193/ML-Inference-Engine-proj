/**
 * Runtime feature detection (design doc 10.5).
 *
 * There is no API that answers "does this engine support relaxed SIMD?", so
 * the standard trick is to validate a tiny module that uses the instruction.
 * The byte arrays below are `wat2wasm` output for the two .wat files in
 * `scripts/probes/`; regenerate them with `scripts/gen-probes.sh` rather than
 * editing by hand.
 */

export interface Features {
  /** Fixed-width 128-bit SIMD. Baseline everywhere since Safari 16.4. */
  simd: boolean;
  /** Relaxed SIMD: the int8 dot product. Not in Safari. */
  relaxed: boolean;
  /** Threads need SharedArrayBuffer, which needs cross-origin isolation. */
  threads: boolean;
  /** Whether the page is cross-origin isolated (COOP + COEP). */
  crossOriginIsolated: boolean;
}

// (module (func (result v128) (v128.const i32x4 1 2 3 4)))
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 7, 5, 1, 1,
  112, 0, 0, 10, 22, 1, 20, 0, 253, 12, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4,
  0, 0, 0, 11,
]);

// (module (func (param v128 v128 v128) (result v128)
//   (i32x4.relaxed_dot_i8x16_i7x16_add_s (local.get 0) (local.get 1) (local.get 2))))
const RELAXED_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 8, 1, 96, 3, 123, 123, 123, 1, 123, 3, 2, 1,
  0, 7, 5, 1, 1, 112, 0, 0, 10, 13, 1, 11, 0, 32, 0, 32, 1, 32, 2, 253, 147, 2,
  11,
]);

let cached: Features | null = null;

export function detectFeatures(): Features {
  if (cached) return cached;
  const isolated =
    typeof globalThis !== 'undefined' &&
    (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
  cached = {
    simd: WebAssembly.validate(SIMD_PROBE),
    relaxed: WebAssembly.validate(RELAXED_PROBE),
    threads: typeof SharedArrayBuffer !== 'undefined' && isolated,
    crossOriginIsolated: isolated,
  };
  return cached;
}

export type Isa = 'scalar' | 'simd128' | 'relaxed';

/** The best build this engine can run. */
export function bestBuild(features = detectFeatures()): Isa {
  if (features.relaxed) return 'relaxed';
  if (features.simd) return 'simd128';
  return 'scalar';
}
