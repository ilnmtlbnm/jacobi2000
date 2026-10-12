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

// Parsed model/model meta.json, read once at boot; the source of truth for
// the active mask (never hard-coded).
let modelMeta = null;
// Current receptive-field highlight ({cells, radius, dilation, headReach}),
// or null. A second click on the model canvas clears it.
let highlightState = null;
// Peak location captured at tick 0 (right after a drop/reset), so a later
// "rewind 20" can report how close it lands to where the wave started.
let originPeak = null;

// Gallery state: web/model/index.json entries, and which are active.
let gallery = [];
let primaryName = null;
let compareName = null;
let compareOn = false;
let currentCase = 'drop';

function maskLabel(mask) {
  if (mask === 'Full') return 'full';
  if (mask && mask.Stencil) return `stencil r${mask.Stencil.radius} d${mask.Stencil.dilation}`;
  return String(mask);
}

// Receptive field of the cell at (x, y): the (2r+1)^2 token neighbourhood
// of the active stencil mask (radius r, dilation d), tokens mapped to cells
// (patch x patch each), then widened by one more (undilated) token hop for
// the per-cell head's own 3x3-token conv (`CellHead.token_mix` in
// training/jacobi2000_pt/model.py), which reads every attended token's immediate
// neighbours after attention.
function receptiveField(x, y) {
  const patch = modelMeta.patch;
  const tokensX = Math.ceil(nx / patch), tokensY = Math.ceil(ny / patch);
  const tx = Math.floor(x / patch), ty = Math.floor(y / patch);
  const mask = modelMeta.mask;
  const isFull = mask === 'Full';
  const radius = isFull ? null : mask.Stencil.radius;
  const dilation = isFull ? null : mask.Stencil.dilation;
  const headReach = 1;

  const attn = new Set();
  for (let ty2 = 0; ty2 < tokensY; ty2++) {
    for (let tx2 = 0; tx2 < tokensX; tx2++) {
      const dx = tx2 - tx, dy = ty2 - ty;
      const inAttn = isFull
        || (dx % dilation === 0 && dy % dilation === 0
          && Math.abs(dx / dilation) <= radius && Math.abs(dy / dilation) <= radius);
      if (inAttn) attn.add(ty2 * tokensX + tx2);
    }
  }
  // Head reach: widen every attended token by one more token in each
  // direction (dilation 1 — the conv is not dilated).
  const tokens = new Set();
  for (const idx of attn) {
    const tx2 = idx % tokensX, ty2 = Math.floor(idx / tokensX);
    for (let dy = -headReach; dy <= headReach; dy++) {
      for (let dx = -headReach; dx <= headReach; dx++) {
        const nx2 = tx2 + dx, ny2 = ty2 + dy;
        if (nx2 >= 0 && nx2 < tokensX && ny2 >= 0 && ny2 < tokensY) tokens.add(ny2 * tokensX + nx2);
      }
    }
  }
  const cells = [];
  for (const idx of tokens) {
    const tx2 = idx % tokensX, ty2 = Math.floor(idx / tokensX);
    for (let cy = ty2 * patch; cy < ty2 * patch + patch; cy++) {
      for (let cx = tx2 * patch; cx < tx2 * patch + patch; cx++) {
        cells.push(cy * nx + cx);
      }
    }
  }
  return { cells, radius, dilation, headReach };
}

function drawHighlightOverlay() {
  if (!highlightState || highlightState.cells.length === 0) return;
  let minX = nx, maxX = -1, minY = ny, maxY = -1;
  for (const c of highlightState.cells) {
    const x = c % nx, y = Math.floor(c / nx);
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  ctx.model.save();
  ctx.model.strokeStyle = '#ffcc00';
  ctx.model.lineWidth = 1;
  ctx.model.strokeRect(minX, minY, maxX - minX + 1, maxY - minY + 1);
  ctx.model.restore();
}

// Blue-white-red, symmetric about zero. `scale` is shared across every
// panel painted this frame (see `render`'s `colorScale`) so a difference
// between two models can never look like a difference in colour range.
let colorScale = 0.05;
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

function fmtMetric(label, v) {
  return `${label} ${Number.isFinite(v) ? v.toFixed(3) : '—'}`;
}

// bias is a signed constant offset (fraction of the solver field's spread),
// so it's shown with an explicit sign, e.g. "bias +0.12".
function fmtBias(v) {
  return `bias ${Number.isFinite(v) ? (v >= 0 ? '+' : '') + v.toFixed(2) : '—'}`;
}

function renderCompareMetrics() {
  const m = engine.compare_metrics(); // [corrA, ampA, lagA, corrB, ampB, lagB, biasA, biasB]
  $('metrics-a').textContent = [fmtMetric('corr', m[0]), fmtMetric('amp', m[1]), fmtMetric('lag', m[2]), fmtBias(m[6])].join('\n');
  $('metrics-b').textContent = compareOn
    ? [fmtMetric('corr', m[3]), fmtMetric('amp', m[4]), fmtMetric('lag', m[5]), fmtBias(m[7])].join('\n')
    : '';
}

// TODO(task 10, xr ensemble spread): add a fourth panel and call
//   const spread = await engine.spread_field(4);
// then paint(ctx.spread, spread, 0.01) alongside the calls below. Left as a
// TODO per the stay-true-for-longer queue: the JS/UI wiring (new canvas,
// STATUS row) is frontend-browser-audio's scope, not this task's.
async function render() {
  const s = engine.solver_field();
  const md = await engine.model_and_divergence();
  const n = nx * ny;

  let peak = 0;
  for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(s[i]));
  // Colour scale comes from the SOLVER panel only, and only every 10 ticks
  // (same cadence as compare_metrics), so amplitude-A-vs-amplitude-B never
  // shows up as a different colour range between panels.
  if (engine.ticks() % 10 === 0) colorScale = Math.max(0.05, peak);

  paint(ctx.solver, s, colorScale);
  paint(ctx.model, md.subarray(0, n), colorScale);
  if (compareOn) {
    const sec = await engine.secondary_field();
    paint(ctx.div, sec, colorScale);
  } else {
    paint(ctx.div, md.subarray(n, 2 * n), colorScale);
  }
  drawHighlightOverlay();
  renderCompareMetrics();

  if (engine.ticks() === 0) originPeak = await engine.peak_location();

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

function entryLabel(entry) {
  const params = entry.params >= 1e6 ? `${(entry.params / 1e6).toFixed(1)}M` : `${(entry.params / 1e3).toFixed(0)}k`;
  const corr = Number.isFinite(entry.corr50) ? entry.corr50.toFixed(3) : '—';
  return `${entry.name} · ${params} params · corr ${corr} at 50 ticks`;
}

function renderRecipe(entry) {
  $('model-name').textContent = entry.name;
  $('recipe').textContent = `${entry.dim}/${entry.layers}, ${entry.params.toLocaleString()} params — ${entry.desc}`;
}

async function fetchModel(dir) {
  const [weights, meta] = await Promise.all([
    fetch(`./model/${dir}/model.safetensors`).then((r) => r.arrayBuffer()),
    fetch(`./model/${dir}/config.json`).then((r) => r.text()),
  ]);
  return { weights: new Uint8Array(weights), meta };
}

function buildPicker(el, onPick, excludeName) {
  el.innerHTML = '';
  for (const entry of gallery) {
    if (entry.name === excludeName) continue;
    const b = document.createElement('button');
    b.textContent = entryLabel(entry);
    b.dataset.name = entry.name;
    b.addEventListener('click', () => onPick(entry));
    el.appendChild(b);
  }
}

function markActive(el, name) {
  for (const b of el.children) b.classList.toggle('active', b.dataset.name === name);
}

async function pickPrimary(entry) {
  running = false;
  $('b-run').textContent = 'run';
  $('status').textContent = `loading ${entry.name}...`;
  const { weights, meta } = await fetchModel(entry.dir);
  await engine.load_model(weights, meta, entry.name);
  primaryName = entry.name;
  modelMeta = JSON.parse(meta);
  highlightState = null;
  $('mask').textContent = maskLabel(modelMeta.mask);
  renderRecipe(entry);
  buildPicker($('model-picker'), pickPrimary, compareOn ? compareName : null);
  markActive($('model-picker'), primaryName);
  if (compareOn) buildPicker($('compare-picker'), pickCompare, primaryName);
  $('status').textContent = 'ready';
  await render();
}

async function pickCompare(entry) {
  $('status').textContent = `loading ${entry.name} (compare)...`;
  const { weights, meta } = await fetchModel(entry.dir);
  await engine.enable_compare(weights, meta, entry.name);
  compareName = entry.name;
  $('c-model-caption').textContent = `A: ${primaryName}`;
  $('c-div-caption').textContent = `B: ${compareName}`;
  buildPicker($('compare-picker'), pickCompare, primaryName);
  markActive($('compare-picker'), compareName);
  buildPicker($('model-picker'), pickPrimary, compareName);
  markActive($('model-picker'), primaryName);
  $('status').textContent = 'ready';
  await render();
}

$('t-compare').addEventListener('change', async (ev) => {
  compareOn = ev.target.checked;
  $('compare-picker').style.display = compareOn ? 'flex' : 'none';
  if (compareOn) {
    // Default the compare slot to the first gallery entry that isn't primary.
    const candidate = gallery.find((e) => e.name !== primaryName && e.loadable !== false);
    if (candidate) await pickCompare(candidate);
  } else {
    engine.disable_compare();
    compareName = null;
    $('c-model-caption').textContent = 'model';
    $('c-div-caption').textContent = 'divergence';
    $('compare-picker').innerHTML = '';
    buildPicker($('model-picker'), pickPrimary, null);
    markActive($('model-picker'), primaryName);
    await render();
  }
});

$('case-picker').addEventListener('click', async (ev) => {
  const btn = ev.target.closest('button[data-case]');
  if (!btn) return;
  currentCase = btn.dataset.case;
  for (const b of $('case-picker').children) b.classList.toggle('active', b === btn);
  running = false;
  $('b-run').textContent = 'run';
  horizonArmed = true;
  highlightState = null;
  engine.set_case(currentCase);
  await render();
});

$('t-mass').addEventListener('change', (ev) => engine.set_mass_projection(ev.target.checked));
$('t-energy').addEventListener('change', (ev) => engine.set_energy_projection(ev.target.checked));

async function boot() {
  try {
    await init();
    gallery = await fetch('./model/index.json').then((r) => r.json());
    const initial = gallery.find((e) => e.name === 'model20') ?? gallery[0];
    const { weights, meta } = await fetchModel(initial.dir);
    await engine.init(weights, meta, initial.name);
    primaryName = initial.name;
    modelMeta = JSON.parse(meta);
    renderRecipe(initial);
    buildPicker($('model-picker'), pickPrimary, null);
    markActive($('model-picker'), primaryName);
    // Task C7: reverse checkpoint is optional — a page whose model/<name>/
    // has no model-reverse.safetensors still boots and runs exactly as
    // before, `rewind` just no-ops (checked in engine.rewind itself).
    try {
      const [rw, rm] = await Promise.all([
        fetch(`./model/${initial.dir}/model-reverse.safetensors`).then((r) => { if (!r.ok) throw new Error('no reverse checkpoint'); return r.arrayBuffer(); }),
        fetch(`./model/${initial.dir}/config-reverse.json`).then((r) => r.text()),
      ]);
      await engine.init_reverse(new Uint8Array(rw), rm);
      console.log('STATUS: reverse checkpoint loaded, rewind() enabled');
    } catch (err) {
      console.log(`STATUS: no reverse checkpoint (${err}), rewind() is a no-op`);
    }
    const g = engine.grid_size();
    nx = g[0]; ny = g[1];
    $('c-solver').width = nx; $('c-solver').height = ny;
    $('c-model').width = nx; $('c-model').height = ny;
    $('c-div').width = nx; $('c-div').height = ny;
    $('grid').textContent = `${nx}x${ny}`;
    $('mask').textContent = maskLabel(modelMeta.mask);
    $('backend').textContent = engine.backend_name();
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
$('c-model').addEventListener('click', async (ev) => {
  const r = ev.target.getBoundingClientRect();
  const x = Math.floor(((ev.clientX - r.left) / r.width) * nx);
  const y = Math.floor(((ev.clientY - r.top) / r.height) * ny);
  if (highlightState) highlightState = null;
  else highlightState = receptiveField(x, y);
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
// Task C7: step the reverse model 20 ticks back from here and report how
// close the peak lands to where the wave started (captured at the last
// tick-0, i.e. the last drop/reset).
$('b-rewind20').addEventListener('click', async () => {
  running = false;
  $('b-run').textContent = 'run';
  await engine.rewind(20);
  await render();
  const after = await engine.peak_location();
  const n = originPeak ? Math.hypot(after[0] - originPeak[0], after[1] - originPeak[1]).toFixed(1) : '—';
  $('rewind-status').textContent = `rewind: 20 ticks back, peak error ${n} cells`;
});

// Drive the page from the headless test through this, not through DOM clicks.
window.__app = {
  ready: () => ready,
  drop: (x, y) => engine.drop_at(x, y, 0.05, 4.0),
  reset: () => engine.reset(),
  step,
  ticks: () => engine.ticks(),
  errors: () => Array.from(engine.rollout_errors()),
  msPerTick: () => engine.ms_per_tick(),
  peak: () => { const s = engine.solver_field(); let p = 0; for (const v of s) p = Math.max(p, Math.abs(v)); return p; },
  modelPeak: async () => { const md = await engine.model_and_divergence(); let p = 0; for (let i = 0; i < nx * ny; i++) p = Math.max(p, Math.abs(md[i])); return p; },
  horizon: () => engine.horizon(),
  mask: () => maskLabel(modelMeta.mask),
  highlight: (x, y) => { highlightState = receptiveField(x, y); drawHighlightOverlay(); return highlightState; },
  clearHighlight: () => { highlightState = null; },
  // Task C7: step the reverse model back n ticks from the current model
  // state. STATUS-line logged per call (console, matching the demo's
  // existing STATUS-panel-only convention — no new UI element).
  rewind: async (n) => {
    await engine.rewind(n);
    await render();
    console.log(`STATUS: rewind(${n}) -> tick ${engine.ticks()}`);
    return engine.ticks();
  },
  peakLocation: async () => Array.from(await engine.peak_location()),
  // Gallery API, driven by the headless test the same way the rest of
  // window.__app is: no DOM clicks.
  gallery: () => gallery,
  pickModel: async (name) => pickPrimary(gallery.find((e) => e.name === name)),
  enableCompare: async (name) => {
    compareOn = true;
    $('t-compare').checked = true;
    $('compare-picker').style.display = 'flex';
    await pickCompare(gallery.find((e) => e.name === name));
  },
  disableCompare: async () => {
    compareOn = false;
    $('t-compare').checked = false;
    engine.disable_compare();
    compareName = null;
    await render();
  },
  setCase: async (c) => { currentCase = c; engine.set_case(c); await render(); },
  setMassProjection: (on) => engine.set_mass_projection(on),
  setEnergyProjection: (on) => engine.set_energy_projection(on),
  compareMetrics: () => Array.from(engine.compare_metrics()),
  secondaryField: async () => Array.from(await engine.secondary_field()),
  modelName: () => engine.model_name(),
  backendName: () => engine.backend_name(),
  error: null,
};

boot();
