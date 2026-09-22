/**
 * The browser half of the test suite (design doc 14.3).
 *
 * wasmtime covers the kernels; this covers what only a browser can: real
 * feature detection, `WebAssembly.instantiate` over `fetch`, the Cache API,
 * memory growth, and the demo actually classifying a drawn digit — the M2
 * milestone's exit criterion.
 *
 * Usage: node tests/browser_test.mjs  (needs playwright + chromium)
 */

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { existsSync, readdirSync } from 'node:fs';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const demo = join(root, 'examples', 'digits');

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.lnn': 'application/octet-stream',
};

function serve(dir) {
  const server = createServer(async (req, res) => {
    const path = join(dir, normalize(decodeURIComponent(req.url.split('?')[0])).replace(/^(\.\.[/\\])+/, ''));
    const file = path.endsWith('/') ? join(path, 'index.html') : path;
    try {
      const body = await readFile(file);
      // Cross-origin isolation, so crossOriginIsolated is true and the Cache
      // API and fine-grained timers behave as they would in a real deploy.
      res.writeHead(200, {
        'content-type': TYPES[extname(file)] ?? 'application/octet-stream',
        'content-length': body.length,
        'cross-origin-opener-policy': 'same-origin',
        'cross-origin-embedder-policy': 'require-corp',
      });
      res.end(body);
    } catch {
      res.writeHead(404).end('not found');
    }
  });
  return new Promise((resolve) => server.listen(0, () => resolve([server, server.address().port])));
}

let failures = 0;
function check(name, ok, detail = '') {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures++;
}

if (!existsSync(join(demo, 'digits.lnn'))) {
  console.error('run scripts/build.sh and tests/make_fixtures.py first');
  process.exit(1);
}

const { chromium } = await import('playwright');
// Playwright wants the browser revision it shipped with; if that one is not
// installed, fall back to whatever Chromium this machine has. Set
// CHROMIUM_PATH to override.
const executablePath = process.env.CHROMIUM_PATH || findChromium(chromium);

function findChromium(browserType) {
  const candidates = [browserType.executablePath()];
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (base && existsSync(base)) {
    for (const dir of readdirSync(base).filter((d) => d.startsWith('chromium'))) {
      candidates.push(
        join(base, dir, 'chrome-linux', 'chrome'),
        join(base, dir, 'chrome-linux64', 'chrome'),
        join(base, dir, 'chrome-linux', 'headless_shell'),
      );
    }
  }
  const found = candidates.find((p) => p && existsSync(p));
  if (!found) {
    console.error('no chromium found; set CHROMIUM_PATH or run: npx playwright install chromium');
    process.exit(2);
  }
  return found;
}
const [server, port] = await serve(demo);
const browser = await chromium.launch({ executablePath });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));

await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle' });

// --- the page loaded the runtime -------------------------------------------
const isa = await page.textContent('#isa');
check('the demo loads a runtime', isa && !isa.includes('failed'), `isa: ${isa}`);
check('no page errors', errors.length === 0, errors.join(' | '));

// --- feature detection agrees with the engine ------------------------------
const features = await page.evaluate(async () => {
  const { detectFeatures } = await import('./detect.js');
  return { ...detectFeatures(), isolated: crossOriginIsolated };
});
check('simd128 detected in Chromium', features.simd === true);
check('relaxed-simd detected in Chromium', features.relaxed === true);
check('the page is cross-origin isolated', features.isolated === true);
check('threads reported available under isolation', features.threads === true);

// --- draw a digit and read the prediction ----------------------------------
// A "1": a vertical stroke down the middle.
const box = await page.locator('#pad').boundingBox();
await page.mouse.move(box.x + box.width / 2, box.y + box.height * 0.2);
await page.mouse.down();
for (let i = 1; i <= 10; i++) {
  await page.mouse.move(box.x + box.width / 2, box.y + box.height * (0.2 + 0.06 * i));
}
await page.mouse.up();
await page.waitForFunction(() => document.querySelector('#ms').textContent !== '–', null, { timeout: 5000 });

const top = await page.textContent('#top');
const ms = await page.textContent('#ms');
check('the drawn digit is classified', /looks like a \d/.test(top), top.trim());
check('latency is reported', /\d/.test(ms), ms);

const probs = await page.evaluate(() =>
  [...document.querySelectorAll('.bars span')].map((s) => parseFloat(s.textContent)),
);
const sum = probs.reduce((a, b) => a + b, 0);
check('the probabilities sum to ~100%', Math.abs(sum - 100) <= 3, `${sum}%`);
check('one class dominates', Math.max(...probs) >= 40, `max ${Math.max(...probs)}%`);

// --- the same model through the API, against the reference -----------------
const fixtures = JSON.parse(await readFile(join(root, 'js', 'test', 'fixtures.json'), 'utf8'));
const apiResult = await page.evaluate(async (cases) => {
  const { Model } = await import('./index.js');
  const m = await Model.load('./digits.lnn');
  const out = [];
  for (const c of cases) {
    const y = m.run({ X: Float32Array.from(c.input) });
    out.push(Array.from(Object.values(y)[0]));
  }
  const isa = m.isa;
  m.dispose();
  return { out, isa };
}, fixtures.digits.cases);

let exact = true;
apiResult.out.forEach((got, i) => {
  const want = fixtures.digits.cases[i].expected;
  for (let j = 0; j < want.length; j++) if (got[j] !== want[j]) exact = false;
});
check('Chromium reproduces the reference outputs exactly', exact, `isa: ${apiResult.isa}`);

// --- the Cache API path works offline --------------------------------------
const cached = await page.evaluate(async () => {
  const { Model } = await import('./index.js');
  const a = await Model.load('./digits.lnn', { cache: 'lnn-test' });
  a.dispose();
  const keys = await caches.open('lnn-test').then((c) => c.keys());
  // Now break the network: a cached model must still load.
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (u) => {
    if (String(u).endsWith('.lnn')) throw new Error('offline');
    return realFetch(u);
  };
  try {
    const b = await Model.load('./digits.lnn', { cache: 'lnn-test' });
    const ok = b.inputs.length === 1;
    b.dispose();
    return { entries: keys.length, offline: ok };
  } finally {
    globalThis.fetch = realFetch;
  }
});
check('the model is stored in the Cache API', cached.entries > 0);
check('a cached model loads with the network down', cached.offline === true);

await browser.close();
server.close();

console.log(`\n${failures === 0 ? 'all browser checks passed' : `${failures} browser check(s) failed`}`);
process.exit(failures === 0 ? 0 : 1);
