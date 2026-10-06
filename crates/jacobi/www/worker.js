// Module worker: WebGPU if an adapter is granted, else the wasm CPU engines
// (one thread); ?path=webgpu|cpu forces one, and an unavailable forced path
// is an error. Each frame advances every panel by the same number of
// iterations, syncs it, and posts the images and one status line.
const ENGINE_BUILD = new URL(import.meta.url).searchParams.get('v');

let viewer = null;
let label = '';
let running = false;
let showError = false;
let iters = 16;
// Every call into the viewer goes through this chain, one at a time.
let busy = Promise.resolve();
function serial(fn) {
  busy = busy.catch(() => {}).then(fn);
  return busy;
}
const recent = [];

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function hasAdapter() {
  try {
    return !!(navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch {
    return false;
  }
}

function fmt(x) {
  return x.toExponential(2);
}

function draw(stats, status) {
  const images = [];
  for (let i = 0; i < viewer.panels; i++) images.push(viewer.frame(i, showError).buffer);
  post({ type: 'frame', images, size: viewer.imageSize, stats, status }, images);
}

function errors() {
  const out = [];
  for (let i = 0; i < viewer.panels; i++) out.push(viewer.maxError(i));
  return out;
}

function errorText(errs) {
  return errs.map((e, i) => `${viewer.panelName(i)} ${fmt(e)}`).join(', ');
}

async function init(search) {
  const q = new URLSearchParams(search);
  const n = Number(q.get('n') || 512);
  const problem = q.get('problem') || '';
  const mod = await import(`../pkg/jacobi.js?v=${ENGINE_BUILD}`);
  await mod.default(`../pkg/jacobi_bg.wasm?v=${ENGINE_BUILD}`);
  mod.jacobiInit();
  let path = q.get('path');
  if (!path) path = (await hasAdapter()) ? 'webgpu' : 'cpu';
  if (path === 'webgpu') {
    viewer = await mod.Viewer.createGpu(n, problem);
    label = viewer.adapter ? `WebGPU (${viewer.adapter})` : 'WebGPU';
  } else if (path === 'cpu') {
    viewer = mod.Viewer.createCpu(n, problem);
    label = 'wasm CPU, one thread';
  } else {
    throw new Error(`unknown path ${path}`);
  }
  if (viewer.problem !== 'quadratic') label += ` · problem ${viewer.problem}`;
  const names = [];
  for (let i = 0; i < viewer.panels; i++) names.push(viewer.panelName(i));
  post({ type: 'ready', path, names, size: viewer.imageSize, status: `${label} · n ${viewer.n} · ready, build ${ENGINE_BUILD}` });
  draw(null, null);
}

async function frame() {
  const n2 = viewer.n * viewer.n;
  // Rates time each panel's iterations alone (to GPU completion), not the
  // readback and drawing that follow.
  const per = [];
  let dt = 0;
  for (let i = 0; i < viewer.panels; i++) {
    const t = performance.now();
    viewer.advance(i, iters);
    await viewer.done();
    const d = performance.now() - t;
    dt += d;
    per.push((iters * n2) / (d / 1000));
  }
  for (let i = 0; i < viewer.panels; i++) await viewer.sync(i);
  recent.push([iters, dt]);
  if (recent.length > 10) recent.shift();
  // Frame pacing only: aim for about 50 ms of iterations per frame.
  if (dt < 30 && iters < 1 << 16) iters *= 2;
  else if (dt > 100 && iters > 1) iters = Math.max(1, iters >> 1);
  const total = recent.reduce((a, r) => a + r[0], 0) * viewer.panels * n2 / (recent.reduce((a, r) => a + r[1], 0) / 1000);
  const errs = errors();
  const it = viewer.iteration(0);
  const stats = { path: label, n: viewer.n, iteration: it, maxError: errs, pointUpdatesPerSec: total, perPanel: per, itersPerFrame: iters };
  const status = `iteration ${it} · max error ${errorText(errs)} · ${label} · n ${viewer.n} · ${fmt(total)} point-updates/s`;
  draw(stats, status);
}

async function loop() {
  while (running) {
    await serial(frame);
    await new Promise((r) => setTimeout(r, 0));
  }
}

function still(what) {
  const it = viewer.iteration(0);
  return `iteration ${it} · max error ${errorText(errors())} · ${label} · n ${viewer.n} · ${what}`;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') await init(m.search);
    else if (!viewer) return;
    else if (m.type === 'run') {
      const was = running;
      running = m.running;
      if (running && !was) loop().catch((err) => post({ type: 'error', status: `error: ${err.message || err}` }));
      if (!running) await serial(() => post({ type: 'paused', status: still('paused') }));
    } else if (m.type === 'round') {
      viewer.setRound(m.round);
    } else if (m.type === 'view') {
      showError = m.error;
      if (!running) await serial(() => draw(null, null));
    } else if (m.type === 'reset') {
      await serial(async () => {
        viewer.reset();
        recent.length = 0;
        for (let i = 0; i < viewer.panels; i++) await viewer.sync(i);
        draw(null, running ? null : still('reset'));
      });
    }
  } catch (err) {
    post({ type: 'error', status: `error: ${err.message || err}` });
  }
};
