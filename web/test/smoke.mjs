// Headless Chromium check of the demo page. Drives window.__app, never DOM
// clicks. Never a personal browser: this is Playwright's bundled build by
// default.
//
// Headless Chromium cannot reach a real Vulkan/NVIDIA WebGPU adapter (falls
// back to SwiftShader, software); a WebGPU-accuracy run needs headed
// Chromium on a real display, which SMOKE_HEADED/SMOKE_EXECUTABLE/
// SMOKE_DISPLAY below opt into on a machine that has one — never the
// a personal browser, only ever a test machine's desktop session.
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import path from 'node:path';
import { serve } from './serve.mjs';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const port = 8099;
const server = await serve(root, port);

const headed = process.env.SMOKE_HEADED === '1';
// Forces the capability ladder's CPU fallback (engine::Model::to_gpu ->
// no adapter) by disabling the GPU process outright, instead of letting
// SwiftShader's software adapter satisfy requestAdapter() (which it does,
// so the ordinary webgpu flags below are not a CPU-path test by themselves).
const forceCpu = process.env.SMOKE_FORCE_CPU === '1';
if (headed && process.env.SMOKE_DISPLAY) {
  process.env.DISPLAY = process.env.SMOKE_DISPLAY;
}
const launchOpts = { args: forceCpu ? ['--disable-gpu'] : ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU'] };
if (headed) {
  launchOpts.headless = false;
  launchOpts.executablePath = process.env.SMOKE_EXECUTABLE || '/usr/bin/chromium';
  if (!forceCpu) {
    launchOpts.args.push('--use-angle=vulkan', '--use-vulkan=native', '--enable-dawn-features=allow_unsafe_apis', '--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--no-sandbox');
  }
} else {
  launchOpts.channel = 'chromium';
}
const browser = await chromium.launch(launchOpts);
const page = await browser.newPage();
const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

let failure = null;
const check = (cond, msg) => { if (!cond) throw new Error(msg); };
try {
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__app && (window.__app.ready() || window.__app.error), null, { timeout: 60000 });
  const err = await page.evaluate(() => window.__app.error);
  if (err) throw new Error(`engine init failed: ${err}`);

  await page.evaluate(() => window.__app.drop(32, 32));

  // Horizon: centred drop on a 64x64 grid is 32 cells from every wall, so
  // horizon = 32 / c ticks, c = sqrt(g*depth)*dt/dx from the model's meta.
  const meta = JSON.parse(await (await fetch(`http://127.0.0.1:${port}/model/model20/config.json`)).text());
  const c = Math.sqrt(meta.g * meta.depth) * meta.dt / meta.dx;
  const horizon = await page.evaluate(() => window.__app.horizon());
  const expectedHorizon = 32 / c;
  if (Math.abs(horizon - expectedHorizon) > 1) {
    throw new Error(`horizon expected ~${expectedHorizon.toFixed(2)}, got ${horizon}`);
  }
  console.log(`HORIZON: PASS (${horizon.toFixed(2)} ticks)`);

  // Mask: read from the loaded checkpoint's meta, never hard-coded.
  const mask = await page.evaluate(() => window.__app.mask());
  const expectedMask = meta.mask && meta.mask.Stencil
    ? `stencil r${meta.mask.Stencil.radius} d${meta.mask.Stencil.dilation}`
    : 'full';
  check(mask === expectedMask, `mask expected '${expectedMask}', got '${mask}'`);
  console.log(`MASK: PASS (${mask})`);

  await page.evaluate(() => window.__app.step(25), { timeout: 120000 });

  // Receptive-field highlight: (2r+1)^2 token neighbourhood of the active
  // stencil, plus one more (undilated) token hop for the per-cell head's
  // own 3x3-token conv (`CellHead.token_mix` in training/jacobi2000_pt/model.py).
  const hl = await page.evaluate(() => window.__app.highlight(32, 32));
  const headReach = 1;
  const totalRadius = meta.mask.Stencil.radius + headReach;
  const expectedSpan = (2 * totalRadius + 1) * meta.patch;
  const xs = hl.cells.map((c) => c % meta.nx);
  const ys = hl.cells.map((c) => Math.floor(c / meta.nx));
  const xSpan = Math.max(...xs) - Math.min(...xs) + 1;
  const ySpan = Math.max(...ys) - Math.min(...ys) + 1;
  check(
    xSpan === expectedSpan && ySpan === expectedSpan,
    `highlight extent expected ${expectedSpan}x${expectedSpan} (radius ${meta.mask.Stencil.radius} + head reach ${headReach}), got ${xSpan}x${ySpan}`,
  );
  console.log(`HIGHLIGHT: PASS (${hl.cells.length} cells, radius ${hl.radius}, dilation ${hl.dilation}, span ${xSpan})`);

  await page.screenshot({ path: path.join(os.tmpdir(), 'jacobi-stencil.png') });

  await page.evaluate(() => window.__app.clearHighlight());
  await page.evaluate(() => window.__app.step(25), { timeout: 120000 });

  const ticks = await page.evaluate(() => window.__app.ticks());
  const errors = await page.evaluate(() => window.__app.errors());
  const ms = await page.evaluate(() => window.__app.msPerTick());
  const solverPeak = await page.evaluate(() => window.__app.peak());
  const modelPeak = await page.evaluate(() => window.__app.modelPeak());

  console.log(JSON.stringify({ ticks, errors, ms, solverPeak, modelPeak }, null, 2));

  // Health of the page and the engine: these are regressions if they break.
  check(ticks === 50, `expected 50 ticks, got ${ticks}`);
  check(Number.isFinite(errors[1]), 'no model error at 50 ticks');
  check(Number.isFinite(modelPeak), 'model field is not finite');
  check(ms > 0, 'no per-tick timing');
  console.log('HEALTH: PASS');

  const backend = await page.evaluate(() => window.__app.backendName());
  console.log(`BACKEND: ${backend}`);
  if (forceCpu) {
    check(backend === 'cpu', `SMOKE_FORCE_CPU=1 but the page is running on '${backend}'`);
  }

  // step/step10 buttons drive window.__app.step, same path as this test.
  await page.evaluate(() => window.__app.step(1));
  const ticksAfterStep1 = await page.evaluate(() => window.__app.ticks());
  check(ticksAfterStep1 === ticks + 1, `step(1) expected ${ticks + 1}, got ${ticksAfterStep1}`);
  await page.evaluate(() => window.__app.step(10));
  const ticksAfterStep10 = await page.evaluate(() => window.__app.ticks());
  check(ticksAfterStep10 === ticksAfterStep1 + 10, `step(10) expected ${ticksAfterStep1 + 10}, got ${ticksAfterStep10}`);
  console.log('STEP: PASS');

  // The gate is a result, not an invariant: reported either way.
  // "The drop still looks like a drop" = relief of the same order as the solver's.
  const beatsFrozen = errors[1] < errors[4];
  const looksLikeADrop = modelPeak > 0.3 * solverPeak && modelPeak < 3 * solverPeak;
  console.log(`GATE rollout@50 beats frozen field: ${beatsFrozen ? 'PASS' : 'FAIL'} (${errors[1].toFixed(4)} vs ${errors[4].toFixed(4)})`);
  console.log(`GATE drop still looks like a drop: ${looksLikeADrop ? 'PASS' : 'FAIL'} (model peak ${modelPeak.toExponential(2)}, solver ${solverPeak.toExponential(2)})`);

  // Regression: dropping while a step is mid-flight must not panic (RefCell
  // already borrowed) and the drop must still land. A drop resets the tick
  // counter to 0, so step(30) is guaranteed to cross the checkpoint at 10
  // and genuinely await mid-flight (not run tick-atomically to completion).
  await page.evaluate(() => window.__app.drop(5, 5));
  const ticksBeforeRace = await page.evaluate(() => window.__app.ticks());
  const peakBeforeRace = await page.evaluate(() => window.__app.peak());
  check(ticksBeforeRace === 0, `race setup: expected ticks 0 after drop, got ${ticksBeforeRace}`);
  const racePromise = page.evaluate(() => window.__app.step(30));
  await page.evaluate(() => {
    window.__app.drop(10, 10);
    window.__app.drop(10, 10);
    window.__app.drop(10, 10);
  });
  await racePromise;
  const raceErr = await page.evaluate(() => window.__app.error);
  check(!raceErr, `page error during race: ${raceErr}`);
  const ticksAfterRace = await page.evaluate(() => window.__app.ticks());
  const peakAfterRace = await page.evaluate(() => window.__app.peak());
  check(
    ticksAfterRace === ticksBeforeRace + 30,
    `race: expected tick counter to advance by 30, ${ticksBeforeRace} -> ${ticksAfterRace}`,
  );
  check(peakAfterRace !== peakBeforeRace, 'race: solver peak did not change after concurrent drops');
  console.log('RACE: PASS');
  // Task C7: rewind. Reset, drop at a known origin, step the forward model
  // 20 ticks, then rewind the reverse model 20 ticks and check the peak
  // lands back near the drop's origin. Skips (does not fail) if the page
  // has no reverse checkpoint (model/model-reverse.bin absent) — the demo
  // still works without one, per the no-op rewind() contract.
  await page.evaluate(() => window.__app.reset());
  await page.evaluate(() => window.__app.drop(20, 40));
  await page.evaluate(() => window.__app.step(20));
  const ticksBeforeRewind = await page.evaluate(() => window.__app.ticks());
  const tickAfterRewind = await page.evaluate(() => window.__app.rewind(20));
  const after = await page.evaluate(() => window.__app.peakLocation());
  if (tickAfterRewind === 0 && ticksBeforeRewind === 20) {
    const dist = Math.hypot(after[0] - 20, after[1] - 40);
    console.log(`REWIND: ${dist <= 4 ? 'PASS' : 'FAIL'} (peak at [${after}], origin [20,40], error ${dist.toFixed(2)} cells)`);
    check(dist <= 4, `rewind(20) peak expected within 4 cells of [20,40], got [${after}] (${dist.toFixed(2)} cells)`);
  } else {
    console.log(`REWIND: SKIPPED (no reverse checkpoint loaded — tick stayed at ${tickAfterRewind})`);
  }
} catch (e) {
  failure = e;
  console.error('FAIL:', e.message);
  console.error(logs.join('\n'));
} finally {
  await browser.close();
  server.close();
}
process.exit(failure ? 1 : 0);
