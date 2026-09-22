/**
 * Tests for the JS side: the public API, feature detection, the error
 * messages, and — the one that matters — that a model driven through the
 * TypeScript API produces exactly what the Python reference produced.
 *
 * Run: npm test (from js/), after scripts/build.sh and tests/make_fixtures.py.
 */

import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
const dist = join(root, 'dist');

const { Model, detectFeatures, bestBuild } = await import(join(here, '..', 'dist', 'index.js'));
const fixtures = JSON.parse(readFileSync(join(here, 'fixtures.json'), 'utf8'));

const BUILDS = ['scalar', 'simd128', 'relaxed'].filter((b) =>
  existsSync(join(dist, `lnn-${b}.wasm`)),
);

function wasmFor(build) {
  return readFileSync(join(dist, `lnn-${build}.wasm`));
}

function modelBytes(name) {
  return readFileSync(join(dist, 'models', name));
}

/** Load without a network: the API takes the runtime bytes directly, and a
 *  tiny fetch stub serves the model from disk. */
async function load(modelName, build = BUILDS.at(-1), opts = {}) {
  const bytes = modelBytes(modelName);
  const previous = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(bytes, { headers: { 'content-length': String(bytes.length) } });
  try {
    return await Model.load(`file:${modelName}`, { wasmBytes: wasmFor(build), ...opts });
  } finally {
    globalThis.fetch = previous;
  }
}

test('detectFeatures reports this engine honestly', () => {
  const f = detectFeatures();
  assert.equal(typeof f.simd, 'boolean');
  assert.equal(typeof f.relaxed, 'boolean');
  // Node 22 is V8, so SIMD must be there; the probe is wrong if it is not.
  assert.equal(f.simd, true, 'the simd128 probe should validate on Node 22');
  assert.ok(['scalar', 'simd128', 'relaxed'].includes(bestBuild(f)));
  assert.equal(f.threads, typeof SharedArrayBuffer !== 'undefined' && f.crossOriginIsolated);
});

test('a model reports its inputs and outputs', async () => {
  const m = await load(fixtures.digits.model);
  assert.equal(m.inputs.length, 1);
  assert.equal(m.outputs.length, 1);
  assert.deepEqual(m.inputs[0].shape, fixtures.digits.inputShape);
  assert.equal(m.inputs[0].dtype, 'f32');
  assert.equal(m.outputs[0].name, fixtures.digits.outputName);
  assert.equal(m.info.nodes > 0, true);
  m.dispose();
});

test('every build reproduces the reference outputs exactly', async () => {
  for (const build of BUILDS) {
    const m = await load(fixtures.digits.model, build);
    for (const c of fixtures.digits.cases) {
      const out = m.run({ X: Float32Array.from(c.input) });
      const got = out[fixtures.digits.outputName];
      assert.equal(got.length, c.expected.length);
      for (let i = 0; i < got.length; i++) {
        assert.equal(got[i], c.expected[i], `${build}: output ${i} differs`);
      }
    }
    m.dispose();
  }
});

test('the CNN reproduces the reference outputs exactly', async () => {
  for (const build of BUILDS) {
    const m = await load(fixtures.cnn.model, build);
    for (const c of fixtures.cnn.cases) {
      const [out] = Object.values(m.run(Float32Array.from(c.input)));
      for (let i = 0; i < out.length; i++) {
        assert.equal(out[i], c.expected[i], `${build}: output ${i} differs`);
      }
    }
    m.dispose();
  }
});

test('the digit classifier predicts the right labels', async () => {
  const m = await load(fixtures.digits.model);
  let correct = 0;
  for (const c of fixtures.digits.cases) {
    const out = m.run({ X: Float32Array.from(c.input) })[fixtures.digits.outputName];
    let best = 0;
    for (let i = 1; i < out.length; i++) if (out[i] > out[best]) best = i;
    if (best === c.label) correct++;
  }
  assert.ok(correct >= fixtures.digits.cases.length - 1, `${correct} of ${fixtures.digits.cases.length}`);
  m.dispose();
});

test('softmax outputs are a probability distribution', async () => {
  const m = await load(fixtures.digits.model);
  const out = m.run({ X: Float32Array.from(fixtures.digits.cases[0].input) })[fixtures.digits.outputName];
  const sum = out.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 0.02, `probabilities sum to ${sum}`);
  assert.ok(out.every((v) => v >= 0 && v <= 1));
  m.dispose();
});

test('runWithStats reports latency and the ISA actually used', async () => {
  const m = await load(fixtures.cnn.model);
  const [, stats] = m.runWithStats(Float32Array.from(fixtures.cnn.cases[0].input));
  assert.ok(stats.latencyMs >= 0);
  assert.ok(['scalar', 'simd128', 'relaxed'].includes(stats.isa));
  assert.equal(stats.isa, m.isa);
  m.dispose();
});

test('a 7-bit model uses the relaxed kernel when the build has one', async () => {
  if (!BUILDS.includes('relaxed')) return;
  const m = await load(fixtures.cnn.model, 'relaxed');
  assert.equal(m.info.sevenBit, true);
  assert.equal(m.isa, 'relaxed');
  m.dispose();
});

test('an 8-bit model makes the relaxed build step down to simd128', async () => {
  if (!BUILDS.includes('relaxed')) return;
  const m = await load(fixtures.digits.model, 'relaxed');
  assert.equal(m.info.sevenBit, false);
  assert.equal(m.isa, 'simd128');
  m.dispose();
});

test('a wrongly sized input is refused with a useful message', async () => {
  const m = await load(fixtures.digits.model);
  assert.throws(() => m.run({ X: new Float32Array(10) }), /wants 64 floats, got 10/);
  m.dispose();
});

test('a missing input names what was expected', async () => {
  const m = await load(fixtures.digits.model);
  assert.throws(() => m.run({ nope: new Float32Array(64) }), /missing input "X"/);
  m.dispose();
});

test('a corrupt model produces a readable error, not a trap', async () => {
  const bytes = modelBytes(fixtures.digits.model);
  const broken = Uint8Array.from(bytes);
  broken[20] = 0xff; // n_inputs
  broken[21] = 0xff;
  const previous = globalThis.fetch;
  globalThis.fetch = async () => new Response(broken);
  try {
    await assert.rejects(
      Model.load('file:broken.lnn', { wasmBytes: wasmFor(BUILDS.at(-1)) }),
      /lnn: .*\(-\d+\)/,
    );
  } finally {
    globalThis.fetch = previous;
  }
});

test('a file that is not a model at all is rejected before wasm sees it', async () => {
  const previous = globalThis.fetch;
  globalThis.fetch = async () => new Response(new Uint8Array(200));
  try {
    await assert.rejects(
      Model.load('file:nope.lnn', { wasmBytes: wasmFor(BUILDS.at(-1)) }),
      /not a .lnn model file/,
    );
  } finally {
    globalThis.fetch = previous;
  }
});

test('using a disposed model is an error, not a crash', async () => {
  const m = await load(fixtures.digits.model);
  m.dispose();
  assert.throws(() => m.run({ X: new Float32Array(64) }), /disposed/);
});

test('repeated runs do not grow memory', async () => {
  const m = await load(fixtures.cnn.model);
  const x = Float32Array.from(fixtures.cnn.cases[0].input);
  m.run(x);
  const before = m.memoryBytes;
  for (let i = 0; i < 20; i++) m.run(x);
  assert.equal(m.memoryBytes, before, 'run() allocated');
  m.dispose();
});

test('onProgress is called while the model downloads', async () => {
  const bytes = modelBytes(fixtures.digits.model);
  const fractions = [];
  const previous = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(bytes, { headers: { 'content-length': String(bytes.length) } });
  try {
    const m = await Model.load('file:digits.lnn', {
      wasmBytes: wasmFor(BUILDS.at(-1)),
      onProgress: (f) => fractions.push(f),
    });
    assert.ok(fractions.length > 0, 'onProgress was never called');
    assert.equal(fractions.at(-1), 1);
    m.dispose();
  } finally {
    globalThis.fetch = previous;
  }
});

test('two models can be loaded at once without interfering', async () => {
  const a = await load(fixtures.digits.model);
  const b = await load(fixtures.cnn.model);
  const ya = a.run({ X: Float32Array.from(fixtures.digits.cases[0].input) })[fixtures.digits.outputName];
  const [yb] = Object.values(b.run(Float32Array.from(fixtures.cnn.cases[0].input)));
  for (let i = 0; i < ya.length; i++) assert.equal(ya[i], fixtures.digits.cases[0].expected[i]);
  for (let i = 0; i < yb.length; i++) assert.equal(yb[i], fixtures.cnn.cases[0].expected[i]);
  a.dispose();
  b.dispose();
});
