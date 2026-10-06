// Headless check of the jacobi dev page: for each variant (query string),
// load the page, run it for a few seconds, print the status line and the
// worker's stats. The first variant is screenshotted twice, field view and
// error view (<shot>-error.png). A variant with `expect=error` passes when
// the page reports an error in its status line without a panic.
// Playwright's bundled Chromium only.
// Usage: PLAYWRIGHT=<path to playwright/index.mjs> node scripts/jacobi_headless.mjs \
//          http://127.0.0.1:<port>/www/index.html [shot.png] [seconds] [variant...]
const { chromium } = await import(process.env.PLAYWRIGHT || 'playwright');
const url = process.argv[2];
const shot = process.argv[3] || 'jacobi-dev-page.png';
const seconds = Number(process.argv[4] || 8);
const variants = process.argv.length > 5 ? process.argv.slice(5) : ['', 'path=webgpu', 'path=cpu', 'n=abc&expect=error', 'n=5000&expect=error'];
const browser = await chromium.launch({
  channel: 'chromium',
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-features=Vulkan'],
});
const results = [];
let ok = true;
for (const [k, v] of variants.entries()) {
  const page = await browser.newPage({ viewport: { width: 1400, height: 760 } });
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(v ? `${url}?${v}` : url, { waitUntil: 'load' });
  const expectError = v.includes('expect=error');
  try {
    await page.waitForFunction(() => window.jac && (window.jac.ready || window.jac.error), null, { timeout: 120000 });
    if (!expectError) {
      await page.evaluate(() => window.jac.start());
      await page.waitForTimeout(seconds * 1000);
      if (k === 0) {
        await page.screenshot({ path: shot });
        await page.evaluate(() => window.jac.setError(true));
        await page.waitForTimeout(1500);
        await page.screenshot({ path: shot.replace(/\.png$/, '-error.png') });
      }
      await page.evaluate(() => window.jac.pause());
      await page.waitForTimeout(500);
    }
  } catch (e) {
    errors.push(String(e));
  }
  const r = await page.evaluate(() => ({ build: window.jac.build, path: window.jac.path, frames: window.jac.frames, isolated: self.crossOriginIsolated, error: window.jac.error || null, last: window.jac.last }));
  r.requested = v || 'auto';
  r.status = await page.textContent('#status');
  r.errors = errors;
  const pass = expectError ? !!r.error && !errors.some((e) => e.includes('panicked') || e.includes('unreachable')) : !errors.length && !r.error && r.frames > 1 && r.last && r.last.pointUpdatesPerSec > 0;
  r.pass = pass;
  if (!pass) ok = false;
  results.push(r);
  await page.close();
}
console.log(JSON.stringify(results, null, 2));
await browser.close();
process.exit(ok ? 0 : 1);
