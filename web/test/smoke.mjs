// Headless Chromium check of the demo page. Drives window.__app, never DOM
// clicks. Never a personal browser: this is Playwright's bundled build.
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { serve } from './serve.mjs';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const port = 8099;
const server = await serve(root, port);

const browser = await chromium.launch({
  channel: 'chromium',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU'],
});
const page = await browser.newPage();
const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));

let failure = null;
try {
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__app && (window.__app.ready() || window.__app.error), null, { timeout: 60000 });
  const err = await page.evaluate(() => window.__app.error);
  if (err) throw new Error(`engine init failed: ${err}`);

  await page.evaluate(() => window.__app.drop(32, 32));

  // Horizon: centred drop on a 64x64 grid is 32 cells from every wall, so
  // horizon = 32 / c ticks, c = sqrt(g*depth)*dt/dx from the model's meta.
  const meta = JSON.parse(await (await fetch(`http://127.0.0.1:${port}/model/meta.json`)).text());
  const c = Math.sqrt(meta.g * meta.depth) * meta.dt / meta.dx;
  const horizon = await page.evaluate(() => window.__app.horizon());
  const expectedHorizon = 32 / c;
  if (Math.abs(horizon - expectedHorizon) > 1) {
    throw new Error(`horizon expected ~${expectedHorizon.toFixed(2)}, got ${horizon}`);
  }
  console.log(`HORIZON: PASS (${horizon.toFixed(2)} ticks)`);

  await page.evaluate(() => window.__app.step(50), { timeout: 120000 });

  const ticks = await page.evaluate(() => window.__app.ticks());
  const errors = await page.evaluate(() => window.__app.errors());
  const ms = await page.evaluate(() => window.__app.msPerTick());
  const solverPeak = await page.evaluate(() => window.__app.peak());
  const modelPeak = await page.evaluate(() => window.__app.modelPeak());

  console.log(JSON.stringify({ ticks, errors, ms, solverPeak, modelPeak }, null, 2));

  // Health of the page and the engine: these are regressions if they break.
  const check = (cond, msg) => { if (!cond) throw new Error(msg); };
  check(ticks === 50, `expected 50 ticks, got ${ticks}`);
  check(Number.isFinite(errors[1]), 'no model error at 50 ticks');
  check(Number.isFinite(modelPeak), 'model field is not finite');
  check(ms > 0, 'no per-tick timing');
  console.log('HEALTH: PASS');

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
} catch (e) {
  failure = e;
  console.error('FAIL:', e.message);
  console.error(logs.join('\n'));
} finally {
  await browser.close();
  server.close();
}
process.exit(failure ? 1 : 0);
