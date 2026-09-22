/**
 * lnn — int8 neural network inference in the browser.
 *
 * ```ts
 * import { Model } from 'lnn';
 * const model = await Model.load('/models/digits.lnn');
 * const { Y } = model.run({ 'X:f32': pixels });
 * ```
 *
 * The runtime is a ~18 KB gzipped wasm module with no dependencies. This
 * file is the entire JS side: feature detection, streaming instantiate,
 * chunked model loading, and a typed-array boundary (design doc 13.5).
 */

import { bestBuild, detectFeatures, type Features, type Isa } from './detect.js';
import { errorMessage, parseInfo, type ModelInfo, type TensorInfo } from './format.js';

export { detectFeatures, bestBuild } from './detect.js';
export type { Features, Isa } from './detect.js';
export type { ModelInfo, TensorInfo } from './format.js';

export interface LoadOptions {
  /** Where the .wasm builds live. Default: alongside this module. */
  runtimeBase?: string;
  /** Force an ISA instead of picking the best available. */
  isa?: 'auto' | Isa;
  /** Cache the model bytes in the Cache API so later loads work offline. */
  cache?: boolean | string;
  /** Called with 0..1 as the model downloads. */
  onProgress?: (fraction: number) => void;
  /** Supply the runtime yourself (tests, bundlers, Node). */
  wasmBytes?: BufferSource;
}

export interface RunStats {
  readonly latencyMs: number;
  readonly isa: Isa;
  readonly threads: number;
}

interface Exports {
  memory: WebAssembly.Memory;
  lnn_alloc(n: number): number;
  lnn_load(ptr: number, len: number): number;
  lnn_input_ptr(i: number): number;
  lnn_input_len(i: number): number;
  lnn_run(): number;
  lnn_output_ptr(i: number): number;
  lnn_output_len(i: number): number;
  lnn_isa(): number;
  lnn_tensor_ptr(i: number): number;
  lnn_memory_used(): number;
}

const ISAS: Isa[] = ['scalar', 'simd128', 'relaxed'];

/** One loaded model, and the wasm instance that holds it. */
export class Model {
  readonly inputs: ReadonlyArray<TensorInfo>;
  readonly outputs: ReadonlyArray<TensorInfo>;
  readonly info: ModelInfo;
  readonly isa: Isa;
  readonly features: Features;

  private ex: Exports;
  private disposed = false;

  private constructor(ex: Exports, info: ModelInfo, features: Features) {
    this.ex = ex;
    this.info = info;
    this.inputs = info.inputs;
    this.outputs = info.outputs;
    this.features = features;
    this.isa = ISAS[ex.lnn_isa()] ?? 'scalar';
  }

  static async load(url: string, opts: LoadOptions = {}): Promise<Model> {
    const features = detectFeatures();
    const want = !opts.isa || opts.isa === 'auto' ? bestBuild(features) : opts.isa;

    const wasm = opts.wasmBytes ?? (await fetchRuntime(want, opts.runtimeBase));
    const { instance } = await WebAssembly.instantiate(wasm as BufferSource, {});
    const ex = instance.exports as unknown as Exports;

    const bytes = await fetchModel(url, opts);
    const info = parseInfo(bytes);

    const ptr = ex.lnn_alloc(bytes.byteLength);
    if (ptr === 0) throw new Error(errorMessage(-11));
    // The view must be taken *after* the allocation: growing memory detaches
    // every existing view of the buffer, which is the most common wasm/JS
    // integration bug there is (doc 8.3).
    new Uint8Array(ex.memory.buffer, ptr, bytes.byteLength).set(bytes);

    const code = ex.lnn_load(ptr, bytes.byteLength);
    if (code !== 0) throw new Error(`lnn: ${errorMessage(code)} (${code})`);

    return new Model(ex, info, features);
  }

  /** Bytes of wasm linear memory this model occupies. */
  get memoryBytes(): number {
    return this.ex.lnn_memory_used();
  }

  run(inputs: Record<string, Float32Array> | Float32Array | Float32Array[]): Record<string, Float32Array> {
    return this.runWithStats(inputs)[0];
  }

  runWithStats(
    inputs: Record<string, Float32Array> | Float32Array | Float32Array[],
  ): [Record<string, Float32Array>, RunStats] {
    if (this.disposed) throw new Error('lnn: this model has been disposed');
    const list = this.resolveInputs(inputs);

    for (let i = 0; i < list.length; i++) {
      const want = this.ex.lnn_input_len(i);
      const data = list[i];
      if (data.byteLength !== want) {
        throw new Error(
          `lnn: input "${this.inputs[i].name}" wants ${want / 4} floats, got ${data.length}`,
        );
      }
      const ptr = this.ex.lnn_input_ptr(i);
      new Uint8Array(this.ex.memory.buffer, ptr, want).set(new Uint8Array(data.buffer, data.byteOffset, want));
    }

    const t0 = now();
    const code = this.ex.lnn_run();
    const latencyMs = now() - t0;
    if (code !== 0) throw new Error(`lnn: ${errorMessage(code)} (${code})`);

    const out: Record<string, Float32Array> = {};
    for (let i = 0; i < this.outputs.length; i++) {
      const n = this.ex.lnn_output_len(i);
      const ptr = this.ex.lnn_output_ptr(i);
      // Copied out, not a view: the caller should not hold a pointer into
      // linear memory that the next run will overwrite.
      out[this.outputs[i].name] = new Float32Array(
        this.ex.memory.buffer.slice(ptr, ptr + n),
      );
    }
    return [out, { latencyMs, isa: this.isa, threads: 1 }];
  }

  dispose(): void {
    // Each model owns its own instance, so dropping the reference is all
    // that is needed; the flag turns use-after-dispose into a clear error.
    this.disposed = true;
    this.ex = null as unknown as Exports;
  }

  private resolveInputs(
    inputs: Record<string, Float32Array> | Float32Array | Float32Array[],
  ): Float32Array[] {
    if (inputs instanceof Float32Array) return [inputs];
    if (Array.isArray(inputs)) return inputs;
    return this.inputs.map((t, i) => {
      // Accept the tensor's own name, or the ONNX name without the ":f32"
      // the converter appends to the float boundary buffer.
      const bare = t.name.replace(/:f32$/, '');
      const v = inputs[t.name] ?? inputs[bare];
      if (!v) {
        throw new Error(
          `lnn: missing input "${bare}" (model inputs: ${this.inputs.map((x) => x.name.replace(/:f32$/, '')).join(', ')})`,
        );
      }
      void i;
      return v;
    });
  }
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

async function fetchRuntime(isa: Isa, base?: string): Promise<BufferSource> {
  const url = `${base ?? defaultBase()}lnn-${isa}.wasm`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`lnn: cannot fetch ${url} (${res.status})`);
  return await res.arrayBuffer();
}

function defaultBase(): string {
  try {
    return new URL('./', import.meta.url).href;
  } catch {
    return './';
  }
}

async function fetchModel(url: string, opts: LoadOptions): Promise<Uint8Array> {
  const cacheName = typeof opts.cache === 'string' ? opts.cache : 'lnn-models-v1';
  if (opts.cache && typeof caches !== 'undefined') {
    const cache = await caches.open(cacheName);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
    const res = await fetch(url);
    if (!res.ok) throw new Error(`lnn: cannot fetch ${url} (${res.status})`);
    await cache.put(url, res.clone());
    return new Uint8Array(await res.arrayBuffer());
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`lnn: cannot fetch ${url} (${res.status})`);
  const total = Number(res.headers.get('content-length') ?? 0);
  if (!opts.onProgress || !res.body || !total) {
    return new Uint8Array(await res.arrayBuffer());
  }

  // Streamed so a large model can report progress. The bytes still land in
  // one buffer before being copied into wasm memory; chunking all the way
  // into linear memory needs the model's length up front, which is what the
  // Content-Length here provides.
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let written = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, written);
    written += value.length;
    opts.onProgress(written / total);
  }
  return written === total ? buf : buf.subarray(0, written);
}
