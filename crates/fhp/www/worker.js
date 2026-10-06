// Module worker: picks the path by capability (WebGPU if an adapter is
// granted, else wasm threads if the page is cross-origin isolated with more
// than one hardware thread, else wasm single thread), loads that build, and
// loops advance -> sample -> frame while running. A forced path (?path=)
// that is unavailable is an error, not a fallback. With ?side=1 it holds
// FHP-III, ternary and signed and steps all three by the same count in
// every frame.
const ENGINE_BUILD = new URL(import.meta.url).searchParams.get('v');

let sims = [];
let label = '';
let running = false;
let stepsPerFrame = 10;

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

async function init(search) {
  const q = new URLSearchParams(search);
  const w = Number(q.get('w') || 4096);
  const h = Number(q.get('h') || 1024);
  const u = Number(q.get('u') || 0.2);
  const mode = q.get('mode') || 'randomize';
  const block = Number(q.get('block') || 16);
  // Signed only: the reference block, 0 for the default H/64.
  const refblock = Number(q.get('refblock') || 0);
  const models = q.get('side') === '1' ? ['fhp3', 'ternary', 'signed'] : [q.get('model') || 'fhp3'];
  const threadsOk = self.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined' && navigator.hardwareConcurrency > 1;
  let path = q.get('path');
  if (!path) path = (await hasAdapter()) ? 'webgpu' : threadsOk ? 'threads' : 'single';
  let create;
  if (path === 'threads') {
    if (!threadsOk) throw new Error(`wasm threads need crossOriginIsolated, SharedArrayBuffer and >1 hardware thread (got ${self.crossOriginIsolated}, ${typeof SharedArrayBuffer}, ${navigator.hardwareConcurrency})`);
    const mod = await import(`../pkg-mt/fhp.js?v=${ENGINE_BUILD}`);
    await mod.default(`../pkg-mt/fhp_bg.wasm?v=${ENGINE_BUILD}`);
    mod.fhpInit();
    await mod.initThreadPool(navigator.hardwareConcurrency);
    create = async (m) => mod.Sim.createCpu(m, w, h, u, mode, block, refblock);
  } else {
    const mod = await import(`../pkg/fhp.js?v=${ENGINE_BUILD}`);
    await mod.default(`../pkg/fhp_bg.wasm?v=${ENGINE_BUILD}`);
    mod.fhpInit();
    create = path === 'webgpu' ? (m) => mod.Sim.createGpu(m, w, h, u, mode, block, refblock) : async (m) => mod.Sim.createCpu(m, w, h, u, mode, block, refblock);
  }
  for (const m of models) sims.push({ sim: await create(m), recent: [], rate: 0, sps: 0 });
  label = path === 'webgpu' ? 'WebGPU' : path === 'threads' ? `wasm threads (${sims[0].sim.threads})` : 'wasm single';
  for (const [i, s] of sims.entries()) {
    s.sim.advance(0);
    await s.sim.sample();
    const rgba = s.sim.frame();
    post({ type: 'frame', index: i, rgba: rgba.buffer, width: s.sim.imageWidth, height: s.sim.imageHeight, stats: null }, [rgba.buffer]);
  }
  const status = sims.map((s) => `${head(s.sim)} · ready, build ${ENGINE_BUILD}`).join('\n');
  post({ type: 'ready', path, count: sims.length, status });
}

// The signed model's reference block (H/64 unless ?refblock=) is separate
// from the display block.
function blocks(sim) {
  return sim.refBlock ? `display block ${sim.block}, reference block ${sim.refBlock}` : `block ${sim.block}`;
}

function head(sim) {
  return `${sim.model} · ${label} · ${sim.width}x${sim.height} ${sim.mode} · ${blocks(sim)}`;
}

function reynolds(sim) {
  return `Re ${sim.reynolds.toFixed(0)} (measured g, ν)`;
}

async function loop() {
  while (running) {
    const n = stepsPerFrame;
    let total = 0;
    for (const [i, s] of sims.entries()) {
      const t0 = performance.now();
      s.sim.advance(n);
      await s.sim.sample();
      const rgba = s.sim.frame();
      const dt = performance.now() - t0;
      total += dt;
      s.recent.push([n, dt]);
      if (s.recent.length > 20) s.recent.shift();
      const steps = s.recent.reduce((a, r) => a + r[0], 0);
      const ms = s.recent.reduce((a, r) => a + r[1], 0);
      s.rate = (steps * s.sim.sites) / (ms / 1000);
      s.sps = (steps / ms) * 1000;
      const stats = { model: s.sim.model, path: label, width: s.sim.width, height: s.sim.height, block: s.sim.block, step: s.sim.step, siteUpdatesPerSec: s.rate, stepsPerSec: s.sps, reynolds: s.sim.reynolds, stepsPerFrame };
      post({ type: 'frame', index: i, rgba: rgba.buffer, width: s.sim.imageWidth, height: s.sim.imageHeight, stats }, [rgba.buffer]);
    }
    // Frame pacing only: aim for about 40 ms per frame.
    if (total < 25 && stepsPerFrame < 4000) stepsPerFrame *= 2;
    else if (total > 80 && stepsPerFrame > 1) stepsPerFrame = Math.max(1, stepsPerFrame >> 1);
    const status = sims.map((s) => `${s.sim.model} · ${reynolds(s.sim)} · ${label} · ${s.sim.width}x${s.sim.height} ${s.sim.mode} · ${blocks(s.sim)} · step ${s.sim.step} · ${(s.rate / 1e6).toFixed(0)} M site-updates/s · ${s.sps.toFixed(0)} steps/s`).join('\n');
    post({ type: 'status', status });
    await new Promise((r) => setTimeout(r, 0));
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') await init(m.search);
    else if (m.type === 'run') {
      const was = running;
      running = m.running;
      if (running && !was) loop().catch((err) => post({ type: 'error', status: `error: ${err.message || err}` }));
      if (!running && sims.length) post({ type: 'paused', status: sims.map((s) => `${head(s.sim)} · step ${s.sim.step} · paused`).join('\n') });
    }
  } catch (err) {
    post({ type: 'error', status: `error: ${err.message || err}` });
  }
};
