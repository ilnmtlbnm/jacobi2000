import init, * as engine from './pkg/web.js';

const $ = (id) => document.getElementById(id);
const ctx = {
  solver: $('c-solver').getContext('2d'),
  model: $('c-model').getContext('2d'),
  div: $('c-div').getContext('2d'),
};
let nx = 64, ny = 64, running = false, ready = false;
// Auto-stop at the pre-reflection horizon is armed per drop; a second click
// on "run" after it fires is the explicit choice to continue past it.
let horizonArmed = true;
const TICK_MS = 100; // 10 ticks/s

// Blue-white-red, symmetric about zero; scale is fixed so the two panels are
// directly comparable frame to frame.
function paint(context, field, scale) {
  const img = context.createImageData(nx, ny);
  for (let i = 0; i < nx * ny; i++) {
    const t = Math.max(-1, Math.min(1, field[i] / scale));
    const r = t > 0 ? 255 : Math.round(255 * (1 + t));
    const b = t < 0 ? 255 : Math.round(255 * (1 - t));
    const g = Math.round(255 * (1 - Math.abs(t)));
    img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
  }
  context.putImageData(img, 0, 0);
}

async function render() {
  const s = engine.solver_field();
  const md = await engine.model_and_divergence();
  const n = nx * ny;
  paint(ctx.solver, s, 0.05);
  paint(ctx.model, md.subarray(0, n), 0.05);
  paint(ctx.div, md.subarray(n, 2 * n), 0.05);

  let peak = 0;
  for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(s[i]));
  $('peak').textContent = peak.toFixed(4);
  $('ticks').textContent = engine.ticks();
  $('ms').textContent = engine.ms_per_tick().toFixed(1);
  const h = engine.horizon();
  $('horizon').textContent = Number.isFinite(h) ? Math.floor(h) : '—';
  const e = engine.rollout_errors();
  const fmt = (v) => (Number.isNaN(v) ? '—' : v.toFixed(4));
  $('e10').textContent = fmt(e[0]); $('e50').textContent = fmt(e[1]); $('e200').textContent = fmt(e[2]);
  $('p10').textContent = fmt(e[3]); $('p50').textContent = fmt(e[4]); $('p200').textContent = fmt(e[5]);
}

async function boot() {
  try {
    await init();
    const [weights, meta] = await Promise.all([
      fetch('./model/model.bin').then((r) => r.arrayBuffer()),
      fetch('./model/meta.json').then((r) => r.text()),
    ]);
    await engine.init(new Uint8Array(weights), meta);
    const g = engine.grid_size();
    nx = g[0]; ny = g[1];
    $('c-solver').width = nx; $('c-solver').height = ny;
    $('c-model').width = nx; $('c-model').height = ny;
    $('c-div').width = nx; $('c-div').height = ny;
    $('grid').textContent = `${nx}x${ny}`;
    $('mask').textContent = engine.mask_name();
    $('backend').textContent = 'burn wgpu / WebGPU';
    $('status').textContent = 'ready';
    ready = true;
    await render();
  } catch (err) {
    $('status').textContent = String(err).slice(0, 120);
    $('status').classList.add('bad');
    window.__app.error = String(err);
  }
}

// Serialized so `run`, `step` and `step10` never overlap engine.step() calls.
let stepChain = Promise.resolve();
function step(n) {
  stepChain = stepChain.then(async () => {
    await engine.step(n);
    await render();
  });
  return stepChain;
}

// Fixed 10 ticks/s while running; step/step10 stay instant (not tied to this).
async function loop() {
  while (running) {
    await step(1);
    if (horizonArmed && engine.ticks() >= engine.horizon()) {
      running = false;
      horizonArmed = false;
      $('b-run').textContent = 'run';
      $('status').textContent = `stopped at horizon: ${Math.floor(engine.horizon())} ticks (first wall reflection)`;
      break;
    }
    await new Promise((r) => setTimeout(r, TICK_MS));
  }
}

$('c-solver').addEventListener('click', async (ev) => {
  const r = ev.target.getBoundingClientRect();
  const x = ((ev.clientX - r.left) / r.width) * nx;
  const y = ((ev.clientY - r.top) / r.height) * ny;
  engine.drop_at(x, y, 0.05, 4.0);
  horizonArmed = true;
  await render();
});
$('b-drop').addEventListener('click', async () => { engine.drop_at(nx / 2, ny / 2, 0.05, 4.0); horizonArmed = true; await render(); });
$('b-reset').addEventListener('click', async () => { engine.reset(); horizonArmed = true; await render(); });
$('b-run').addEventListener('click', () => {
  running = !running;
  $('b-run').textContent = running ? 'pause' : 'run';
  if (running) $('status').textContent = 'run: 10 ticks/s';
  else if (ready) $('status').textContent = 'ready';
  if (running) loop();
});
$('b-step').addEventListener('click', () => { running = false; $('b-run').textContent = 'run'; step(1); });
$('b-step10').addEventListener('click', () => { running = false; $('b-run').textContent = 'run'; step(10); });

// Drive the page from the headless test through this, not through DOM clicks.
window.__app = {
  ready: () => ready,
  drop: (x, y) => engine.drop_at(x, y, 0.05, 4.0),
  step,
  ticks: () => engine.ticks(),
  errors: () => Array.from(engine.rollout_errors()),
  msPerTick: () => engine.ms_per_tick(),
  peak: () => { const s = engine.solver_field(); let p = 0; for (const v of s) p = Math.max(p, Math.abs(v)); return p; },
  modelPeak: async () => { const md = await engine.model_and_divergence(); let p = 0; for (let i = 0; i < nx * ny; i++) p = Math.max(p, Math.abs(md[i])); return p; },
  horizon: () => engine.horizon(),
  error: null,
};

boot();
