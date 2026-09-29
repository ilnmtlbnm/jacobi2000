// One-off: screenshot model20 and model-pt-curriculum side by side for the
// PyTorch-training run doc (docs/runs/2026-09-29-pytorch-training.md).
// Not part of the gallery.mjs acceptance gate; ad hoc visual record only.
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { serve } from './serve.mjs';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const port = 8099;
const server = await serve(root, port);
const outDir = process.argv[2];

const browser = await chromium.launch({
  channel: 'chromium',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));

try {
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__app && (window.__app.ready() || window.__app.error), null, { timeout: 60000 });
  const err = await page.evaluate(() => window.__app.error);
  if (err) throw new Error(`engine init failed: ${err}`);

  for (const name of ['model20', 'model-pt-curriculum']) {
    await page.evaluate((n) => window.__app.pickModel(n), name);
    await page.evaluate(() => window.__app.setCase('drop'));
    await page.evaluate(() => window.__app.drop(32, 32));
    await page.evaluate(() => window.__app.step(30), { timeout: 120000 });
    const p = `${outDir}/${name}.png`;
    await page.screenshot({ path: p });
    console.log(`SCREENSHOT ${name}: ${p}`);
  }
} finally {
  await browser.close();
  server.close();
}
