/**
 * Benchmarks (design doc section 15).
 *
 * Two tables:
 *   1. The GEMM optimization ladder from M4 — the same matrices through
 *      naive loops, packed+blocked scalar, simd128 and relaxed-simd.
 *   2. Whole-model latency per build, plus binary sizes.
 *
 * Methodology: warm up so V8 tiers up from Liftoff to TurboFan, then report
 * median and p95 rather than the mean, because GC pauses and thermal
 * throttling make means meaningless (doc 15.2).
 *
 * Usage: WITH_BENCH=1 ./scripts/build.sh && node bench/bench.mjs
 */

import { execSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const dist = join(root, 'dist');

function bench(fn, { warmup = 20, iters = 100 } = {}) {
  for (let i = 0; i < warmup; i++) fn();
  const samples = [];
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    fn();
    samples.push(performance.now() - t0);
  }
  samples.sort((a, b) => a - b);
  return {
    min: samples[0],
    median: samples[iters >> 1],
    p95: samples[Math.floor(iters * 0.95)],
  };
}

async function instantiate(file) {
  const bytes = readFileSync(join(dist, file));
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

// ------------------------------------------------------------ GEMM ladder

// The first two rungs come from the scalar build and the last two from the
// SIMD build. Running rung 0 inside a SIMD build measures LLVM's
// auto-vectorizer rather than a scalar baseline, which would flatter the
// starting point and understate what the hand-written kernels buy.
const RUNGS = [
  ['naive scalar', 0, 'scalar'],
  ['+ blocking & packing', 1, 'scalar'],
  ['+ simd128 microkernel', 2, 'simd'],
  ['+ relaxed-simd dot', 3, 'simd'],
];

// Shapes taken from the models: a pointwise expansion layer, a 3x3 conv
// after im2col, and a classifier.
const SHAPES = [
  ['pointwise 196x256x64', 196, 256, 64],
  ['conv3x3 196x64x576', 196, 64, 576],
  ['classifier 1x1000x256', 1, 1000, 256],
  ['square 256x256x256', 256, 256, 256],
];

async function gemmLadder() {
  if (!existsSync(join(dist, 'lnn-bench.wasm')) || !existsSync(join(dist, 'lnn-bench-scalar.wasm'))) {
    console.log('skipping the GEMM ladder: build with WITH_BENCH=1 ./scripts/build.sh\n');
    return [];
  }
  const engines = {
    simd: await instantiate('lnn-bench.wasm'),
    scalar: await instantiate('lnn-bench-scalar.wasm'),
  };
  const rows = [];
  console.log('GEMM optimization ladder (int8, GOP/s — higher is better)\n');
  const head = ['shape', ...RUNGS.map(([n]) => n)];
  console.log(head.map((h, i) => (i ? h.padStart(22) : h.padEnd(22))).join(''));

  for (const [label, m, n, k] of SHAPES) {
    for (const ex of Object.values(engines)) {
      if (ex.lnn_bench_setup(m, n, k) !== 0) throw new Error('bench setup failed');
    }
    const ops = 2 * m * n * k; // one multiply and one add per MAC
    const cells = [];
    let checksum = null;
    for (const [, rung, which] of RUNGS) {
      const ex = engines[which];
      if (ex.lnn_bench_gemm(rung) !== 0) {
        cells.push('n/a');
        continue;
      }
      // Every rung must compute the same thing; a fast wrong kernel is not a
      // faster kernel.
      const sum = ex.lnn_bench_checksum();
      if (checksum === null) checksum = sum;
      else if (sum !== checksum) throw new Error(`rung ${rung} computed a different result`);
      const iters = ops > 40e6 ? 20 : 100;
      const t = bench(() => ex.lnn_bench_gemm(rung), { warmup: 5, iters });
      cells.push((ops / (t.median / 1000) / 1e9).toFixed(2));
    }
    rows.push({ shape: label, m, n, k, gops: cells });
    console.log(label.padEnd(22) + cells.map((c) => c.padStart(22)).join(''));
  }
  console.log();
  return rows;
}

// ---------------------------------------------------------- model latency

async function modelLatency() {
  const { Model } = await import(join(root, 'js', 'dist', 'index.js'));
  const builds = ['scalar', 'simd128', 'relaxed'].filter((b) => existsSync(join(dist, `lnn-${b}.wasm`)));
  const modelsDir = join(dist, 'models');
  const rows = [];

  for (const name of ['digits-mlp.lnn', 'mobilenet-ish.lnn']) {
    const path = join(modelsDir, name);
    if (!existsSync(path)) continue;
    const bytes = readFileSync(path);
    console.log(`${name} (${(bytes.length / 1024).toFixed(1)} KiB)\n`);
    console.log('  build      isa        median      p95     memory');

    for (const build of builds) {
      const previous = globalThis.fetch;
      globalThis.fetch = async () => new Response(bytes);
      let model;
      try {
        model = await Model.load(`file:${name}`, { wasmBytes: readFileSync(join(dist, `lnn-${build}.wasm`)) });
      } finally {
        globalThis.fetch = previous;
      }
      const x = new Float32Array(model.inputs[0].size);
      for (let i = 0; i < x.length; i++) x[i] = Math.sin(i) * 0.5;
      const t = bench(() => model.run(x), { warmup: 30, iters: 200 });
      rows.push({ model: name, build, isa: model.isa, ...t, memory: model.memoryBytes });
      console.log(
        `  ${build.padEnd(10)} ${model.isa.padEnd(9)} ${t.median.toFixed(3).padStart(8)}ms ` +
          `${t.p95.toFixed(3).padStart(8)}ms ${(model.memoryBytes / 1024).toFixed(0).padStart(7)} KiB`,
      );
      model.dispose();
    }
    console.log();
  }
  return rows;
}

// ------------------------------------------------------------------ sizes

function sizes() {
  const rows = [];
  console.log('runtime size\n');
  console.log('  build        raw        gzip');
  for (const build of ['scalar', 'simd128', 'relaxed']) {
    const p = join(dist, `lnn-${build}.wasm`);
    if (!existsSync(p)) continue;
    const raw = readFileSync(p);
    const gz = gzipSync(raw, { level: 9 });
    rows.push({ build, raw: raw.length, gzip: gz.length });
    console.log(
      `  ${build.padEnd(9)} ${(raw.length / 1024).toFixed(1).padStart(7)} KiB ` +
        `${(gz.length / 1024).toFixed(1).padStart(7)} KiB`,
    );
  }
  console.log();
  return rows;
}

// -------------------------------------------------------------------- main

function environment() {
  let cpu = 'unknown';
  try {
    cpu = execSync("grep -m1 'model name' /proc/cpuinfo || true").toString().split(':').pop().trim();
  } catch {
    /* not Linux */
  }
  return { node: process.version, v8: process.versions.v8, platform: process.platform, arch: process.arch, cpu };
}

const env = environment();
console.log(`lnn benchmarks\n  ${env.cpu}\n  node ${env.node} (V8 ${env.v8}), ${env.platform}/${env.arch}\n`);

const results = {
  environment: env,
  when: new Date().toISOString(),
  sizes: sizes(),
  gemm: await gemmLadder(),
  models: await modelLatency(),
};

mkdirSync(join(here, 'results'), { recursive: true });
const out = join(here, 'results', 'latest.json');
writeFileSync(out, JSON.stringify(results, null, 2));
console.log(`wrote ${out.replace(root + '/', '')}`);
