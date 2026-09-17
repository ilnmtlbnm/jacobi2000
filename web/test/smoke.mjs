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
  await page.evaluate(() => window.__app.step(50), { timeout: 120000 });

  const ticks = await page.evaluate(() => window.__app.ticks());
  const errors = await page.evaluate(() => window.__app.errors());
  const ms = await page.evaluate(() => window.__app.msPerTick());
  const solverPeak = await page.evaluate(() => window.__app.peak());
  const modelPeak = await page.evaluate(() => window.__app.modelPeak());

  console.log(JSON.stringify({ ticks, errors, ms, solverPeak, modelPeak }, null, 2));

  const check = (cond, msg) => { if (!cond) throw new Error(msg); };
  check(ticks === 50, `expected 50 ticks, got ${ticks}`);
  check(Number.isFinite(errors[1]), 'no model error at 50 ticks');
  check(errors[1] < errors[4], `model RMSE@50 ${errors[1]} not below frozen ${errors[4]}`);
  // "The drop still looks like a drop": the model keeps a comparable amount
  // of surface relief instead of decaying to flat or blowing up.
  check(modelPeak > 0.3 * solverPeak && modelPeak < 3 * solverPeak,
    `model peak ${modelPeak} vs solver ${solverPeak}`);
  console.log('PASS');
} catch (e) {
  failure = e;
  console.error('FAIL:', e.message);
  console.error(logs.join('\n'));
} finally {
  await browser.close();
  server.close();
}
process.exit(failure ? 1 : 0);
