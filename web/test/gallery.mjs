// Headless Chromium check of the model gallery: load every loadable
// checkpoint, run 30 ticks on the drop case, check finite fields and the
// rung6 corr@30 floor for model20/model-wide-2; then compare mode with two
// models + the solver. Drives window.__app, never DOM clicks.
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { serve } from './serve.mjs';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const port = 8098;
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
const check = (cond, msg) => { if (!cond) throw new Error(msg); };
const results = {};
try {
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__app && (window.__app.ready() || window.__app.error), null, { timeout: 60000 });
  const err = await page.evaluate(() => window.__app.error);
  if (err) throw new Error(`engine init failed: ${err}`);

  const gallery = await page.evaluate(() => window.__app.gallery());
  console.log(`GALLERY: ${gallery.length} entries: ${gallery.map((e) => e.name).join(', ')}`);

  for (const entry of gallery) {
    const label = entry.name;
    try {
      await page.evaluate((name) => window.__app.pickModel(name), label);
      await page.evaluate(() => window.__app.setCase('drop'));
      await page.evaluate(() => window.__app.drop(32, 32));
      await page.evaluate(() => window.__app.step(30), { timeout: 120000 });
      const peak = await page.evaluate(() => window.__app.modelPeak());
      const metrics = await page.evaluate(() => window.__app.compareMetrics());
      const corr30 = metrics[0];
      check(Number.isFinite(peak), `${label}: model field not finite`);
      check(Number.isFinite(corr30), `${label}: corr@30 not finite`);
      results[label] = corr30;
      console.log(`LOAD ${label}: PASS (peak ${peak.toExponential(2)}, corr@30 ${corr30.toFixed(4)})`);
    } catch (e) {
      results[label] = null;
      console.log(`LOAD ${label}: FAIL (${e.message})`);
      if (entry.loadable !== false) throw e; // only tolerate a failure the index already flags
    }
  }

  check(results['model20'] > 0.98, `model20 corr@30 expected > 0.98, got ${results['model20']}`);
  check(results['model-wide-2'] > 0.95, `model-wide-2 corr@30 expected > 0.95, got ${results['model-wide-2']}`);
  console.log('ACCEPTANCE: PASS (model20 > 0.98, model-wide-2 > 0.95)');

  // Compare mode: model20 vs model-wide-1, same drop case, solver as the
  // third panel — three canvases, all finite.
  await page.evaluate(() => window.__app.pickModel('model20'));
  await page.evaluate(() => window.__app.setCase('drop'));
  await page.evaluate(() => window.__app.enableCompare('model-wide-1'));
  // enable_compare re-applies the current case (a clean reset per model), so
  // the drop click has to come after it, same as a user would click post-toggle.
  await page.evaluate(() => window.__app.drop(32, 32));
  await page.evaluate(() => window.__app.step(20), { timeout: 120000 });
  const solverPeak = await page.evaluate(() => window.__app.peak());
  const primaryPeak = await page.evaluate(() => window.__app.modelPeak());
  const secondary = await page.evaluate(() => window.__app.secondaryField());
  const secondaryFinite = secondary.length > 0 && secondary.every((v) => Number.isFinite(v));
  const compareMetrics = await page.evaluate(() => window.__app.compareMetrics());
  check(Number.isFinite(solverPeak) && Number.isFinite(primaryPeak), 'compare: solver/primary field not finite');
  check(secondaryFinite, 'compare: secondary field not finite or empty');
  check(compareMetrics.every((v) => Number.isFinite(v)), `compare metrics not all finite: ${compareMetrics}`);
  // compareMetrics: [corrA, ampA, lagA, corrB, ampB, lagB, biasA, biasB].
  // model20 (A) is the projection-ON, four-bar-passing checkpoint (rung6);
  // its bias must be small. model-wide-1 (B) is reported, not gated.
  const biasA = compareMetrics[6];
  const biasB = compareMetrics[7];
  check(Math.abs(biasA) < 0.05, `model20 (A) |bias| expected < 0.05, got ${biasA}`);
  console.log(`COMPARE: PASS (solver peak ${solverPeak.toExponential(2)}, A corr ${compareMetrics[0].toFixed(3)} bias ${biasA.toFixed(3)}, B corr ${compareMetrics[3].toFixed(3)} bias ${biasB.toFixed(3)})`);

  const dir = '<tmp>';
  await page.screenshot({ path: `${dir}/jacobi-bias.png` });
  console.log(`SCREENSHOT: ${dir}/jacobi-bias.png`);

  await page.evaluate(() => window.__app.disableCompare());
} catch (e) {
  failure = e;
  console.error('FAIL:', e.message);
  console.error(logs.join('\n'));
} finally {
  console.log('RESULTS', JSON.stringify(results, null, 2));
  await browser.close();
  server.close();
}
process.exit(failure ? 1 : 0);
