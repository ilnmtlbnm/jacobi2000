// Headless check of the fhp dev page: for each variant (query string), load
// the page, run it for a few seconds, and print the status line and the
// worker's stats; screenshot each variant (<shot>-<k>.png). A variant may
// carry secs=N (read here, ignored by the page) to run longer. A variant with
// `expect=error` passes when the page reports an error in its status line
// without a page error. Playwright's bundled Chromium only.
// Usage: PLAYWRIGHT=<path to playwright/index.mjs> node scripts/fhp_headless.mjs \
//          http://127.0.0.1:<port>/www/index.html [screenshot.png] [seconds] [variant...]
const { chromium } = await import(process.env.PLAYWRIGHT || 'playwright');
const url = process.argv[2];
const shot = process.argv[3] || 'fhp-dev-page.png';
const seconds = Number(process.argv[4] || 8);
const variants = process.argv.length > 5 ? process.argv.slice(5) : ['', 'path=webgpu', 'path=threads', 'path=single'];
const browser = await chromium.launch({
  channel: 'chromium',
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--enable-features=Vulkan', ...(process.env.CHROMIUM_ARGS ? process.env.CHROMIUM_ARGS.split(' ') : [])],
});
const results = [];
let ok = true;
let limits = null;
for (const [k, v] of variants.entries()) {
  const page = await browser.newPage({ viewport: { width: 1400, height: 600 } });
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(v ? `${url}?${v}` : url, { waitUntil: 'load' });
  if (!limits) {
    limits = await page.evaluate(async () => {
      const a = navigator.gpu && (await navigator.gpu.requestAdapter());
      return a && { maxBufferSize: a.limits.maxBufferSize, maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize, maxComputeWorkgroupsPerDimension: a.limits.maxComputeWorkgroupsPerDimension };
    });
  }
  const expectError = v.includes('expect=error');
  const secs = Number(new URLSearchParams(v).get('secs') || seconds);
  try {
    await page.waitForFunction(() => window.fhp && (window.fhp.ready || window.fhp.error), null, { timeout: 120000 });
    if (!expectError) {
      await page.evaluate(() => window.fhp.start());
      await page.waitForTimeout(secs * 1000);
      await page.evaluate(() => window.fhp.pause());
      await page.waitForTimeout(300);
    }
  } catch (e) {
    errors.push(String(e));
  }
  const r = await page.evaluate(() => ({ build: window.fhp.build, path: window.fhp.path, frames: window.fhp.frames, isolated: self.crossOriginIsolated, error: window.fhp.error || null, last: window.fhp.last, all: window.fhp.all }));
  r.requested = v || 'auto';
  r.status = await page.textContent('#status');
  r.errors = errors;
  r.shot = shot.replace(/\.png$/, `-${k}.png`);
  await page.screenshot({ path: r.shot, fullPage: true });
  const pass = expectError ? !!r.error && !errors.some((e) => e.includes('panicked') || e.includes('unreachable')) : !errors.length && !r.error && r.frames > 1 && r.last && r.last.siteUpdatesPerSec > 0;
  r.pass = pass;
  if (!pass) ok = false;
  results.push(r);
  await page.close();
}
console.log(JSON.stringify({ limits, results }, null, 2));
await browser.close();
process.exit(ok ? 0 : 1);
