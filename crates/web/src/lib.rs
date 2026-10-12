//! Browser engine: solver on the left, learned operator(s) on the right, one
//! `window.__app` surface driven by the page (and by the headless test).
//!
//! The model runs on `engine`'s hand-written forward pass, either
//! `engine::cpu` (plain `Vec<f32>`, optionally `rayon`-threaded if this
//! wasm binary was built with the `threads` feature) or `engine::gpu`
//! (hand-written WGSL kernels on `lean_runtime`'s device/buffer/pipeline
//! helpers). State lives in plain `Vec<f32>` either way.
//!
//! Backend selection is capability-only (`lean_runtime::backend`'s
//! ladder): WebGPU if an adapter is granted, else whichever CPU path this
//! wasm binary was built with (rayon threads or one thread — a build-time
//! choice, not runtime). The choice is made once, in [`init`], by trying
//! [`engine::Model::to_gpu`] on the first checkpoint; every later
//! checkpoint (model switch, compare mode, the reverse model) reuses that
//! same decision via [`Backend::for_model`], so a page is never WebGPU for
//! one checkpoint and CPU for another.
//!
//! Gallery (task: model picker + case picker + compare mode): a checkpoint
//! is a [`LoadedModel`] — its own weights (a [`Backend`], which carries its
//! own grid/norm/mask config) and state, each stepped independently.
//! `App.primary` is always present; `App.secondary` exists only in compare
//! mode, stepped in lockstep with the same solver truth. Every model shares
//! one grid (64x64, patch 4) and one solver-conditioning vector
//! (`App.meta`, built from the primary's physics params) — all shipped
//! checkpoints share that grid and near-identical physics (dt~0.090,
//! nu=0.001), so this is a fair simplification, not an approximation that
//! changes which model wins.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use engine::metrics::{bias, correlation, decode_h, peak, radial_profile, xcorr_lag};
use engine::{project, Config, Model as EngineModel};
use solvers::{Params, ShallowWater, State};
use wasm_bindgen::prelude::*;

/// Either CPU (`engine::cpu`, via `Model::forward`/`step`) or WebGPU
/// (`engine::gpu`). See the module doc for how the choice is made. Always
/// stored behind an `Rc` (`LoadedModel::model`), so the size difference
/// between variants costs nothing in practice.
#[allow(clippy::large_enum_variant)]
enum Backend {
    Cpu(EngineModel),
    Gpu(engine::gpu::GpuModel),
}

impl Backend {
    /// Try WebGPU first (matching `use_gpu`'s earlier decision for this
    /// page); fall back to CPU if no adapter is granted this time either.
    async fn for_model(model: EngineModel, use_gpu: bool) -> Backend {
        if use_gpu {
            if let Ok(gpu) = model.to_gpu().await {
                return Backend::Gpu(gpu);
            }
        }
        Backend::Cpu(model)
    }

    fn cfg(&self) -> &Config {
        match self {
            Backend::Cpu(m) => &m.cfg,
            Backend::Gpu(m) => m.cfg(),
        }
    }

    async fn step(&self, state: &[f32], meta: &[f32]) -> Vec<f32> {
        match self {
            Backend::Cpu(m) => m.step(state, meta),
            Backend::Gpu(m) => m.step(state, meta).await,
        }
    }
}

/// Ticks at which the rollout error is recorded, matching the native CLI.
const CHECKPOINTS: [usize; 3] = [10, 50, 200];
/// Compare-mode metrics (corr, amplitude ratio, wavefront lag against the
/// solver) refresh every this many ticks.
const METRICS_EVERY: usize = 10;

/// Preset initial conditions for the case picker (CONCEPT sec. 6's
/// "generalization exhibit": trained on random ripples, shown a dam break,
/// a drop, a stacked/near-wall case it never saw at this amplitude).
#[derive(Debug, Clone, Copy, PartialEq)]
enum CaseKind {
    /// Flat pool; the user clicks to drop, as before the gallery task.
    Drop,
    /// Classic 1D dam break: a step in h across the domain, u = v = 0.
    DamBreak,
    /// Two drops at near-wall margins, high amplitude — the training run's
    /// own "extended" distribution (`--amplitude-max 0.3 --extra-drop-prob`),
    /// applied at case-select time instead of sampled.
    Extended,
}

impl CaseKind {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "drop" => Some(CaseKind::Drop),
            "dam_break" => Some(CaseKind::DamBreak),
            "extended" => Some(CaseKind::Extended),
            _ => None,
        }
    }
}

/// One checkpoint, stepped in its own state buffer. Shared grid/mask shape
/// with every other loaded checkpoint (see module doc).
struct LoadedModel {
    name: String,
    model: Rc<Backend>,
    x: Vec<f32>,
    /// Box mean of the normalized channel 0 at the last reset, in normalized
    /// units — the target `project::project_mass` restores.
    mass0: f32,
}

struct App {
    params: Params,
    meta: Vec<f32>,
    solver: ShallowWater,
    truth: State,
    /// Field frozen at the last reset: the persistence baseline.
    frozen: Vec<f32>,
    tick: usize,
    errors: Vec<f32>,
    persistence: Vec<f32>,
    last_step_ms: f64,
    /// Ticks until the wave from the nearest drop reaches the nearest wall
    /// (min over drops since the last reset). `INFINITY` with no drop yet.
    horizon: f32,
    /// The reverse (backward-stepping) checkpoint, loaded
    /// separately via `init_reverse`. `None` until then, so `rewind` is a
    /// no-op (logged by the caller) on a page that never loaded one.
    reverse_model: Option<Rc<Backend>>,
    /// Decided once in `init`, reused for every later checkpoint (model
    /// switch, compare mode, the reverse model) so the page never runs one
    /// checkpoint on WebGPU and another on CPU.
    use_gpu: bool,
    primary: LoadedModel,
    /// Compare mode's second checkpoint, stepped alongside `primary` from
    /// the same case. `None` outside compare mode.
    secondary: Option<LoadedModel>,
    case: CaseKind,
    mass_projection: bool,
    energy_projection: bool,
    /// `[corrA, ampA, lagA, corrB, ampB, lagB, biasA, biasB]` against the
    /// solver, refreshed every `METRICS_EVERY` ticks (NaN for B when compare
    /// mode is off, NaN for lag when the field has no variance yet). bias is
    /// appended rather than interleaved so existing A/B indices stay stable
    /// for callers.
    compare_metrics: [f32; 8],
}

/// Wave speed in cells/tick: `sqrt(g * depth) * dt / dx`.
fn wave_speed(p: &Params) -> f32 {
    (p.g * p.depth).sqrt() * p.dt / p.dx
}

/// Ticks for a drop at `(x, y)` (cell coordinates) to reach the nearest wall.
fn horizon_ticks(p: &Params, x: f32, y: f32) -> f32 {
    let dist = x.min(p.nx as f32 - x).min(y).min(p.ny as f32 - y);
    dist / wave_speed(p)
}

thread_local! {
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
    /// Set for the duration of `step`'s async body. The CPU engine never
    /// actually awaits mid-step any more, but this flag (and the pending
    /// queues below) are kept so `drop_at`/`reset`'s re-entrancy contract
    /// with `web/app.js` is unchanged.
    static STEPPING: Cell<bool> = const { Cell::new(false) };
    static PENDING_DROPS: RefCell<Vec<[f32; 4]>> = const { RefCell::new(Vec::new()) };
    static PENDING_RESET: Cell<bool> = const { Cell::new(false) };
}

fn encode_tensor(cfg_norm: &engine::Normalizer, s: &State, p: &Params) -> (Vec<f32>, Vec<f32>) {
    let mut packed = vec![0.0f32; 3 * p.cells()];
    cfg_norm.encode(s, &mut packed);
    (packed.clone(), packed)
}

async fn load_checkpoint(weights: Vec<u8>, config_json: &str, name: String, truth: &State, use_gpu: bool) -> LoadedModel {
    let raw = EngineModel::from_bytes(&weights, config_json);
    let (_, x) = encode_tensor(&raw.cfg.norm, truth, &raw.cfg.params());
    let mass0 = project::mass(&x, raw.cfg.ny * raw.cfg.nx);
    let model = Rc::new(Backend::for_model(raw, use_gpu).await);
    LoadedModel { name, model, x, mass0 }
}

/// Apply the current case's initial condition to `truth`/`frozen`/`primary`/
/// `secondary`, resetting tick/errors/horizon. Shared by `set_case`, model
/// switches and compare mode enable/disable so every entry point ends up in
/// the same clean state (the "switching model/case resets cleanly" rule).
fn apply_case(app: &mut App) {
    let p = &app.params;
    app.truth = State::flat(p);
    let mut origins: Vec<(f32, f32)> = Vec::new();
    match app.case {
        CaseKind::Drop => {}
        CaseKind::DamBreak => {
            for y in 0..p.ny {
                for x in 0..(p.nx / 2) {
                    app.truth.h[y * p.nx + x] = p.depth + 0.15;
                }
            }
        }
        CaseKind::Extended => {
            let (m1x, m1y) = (p.nx as f32 * 0.15, p.ny as f32 * 0.15);
            let (m2x, m2y) = (p.nx as f32 * 0.8, p.ny as f32 * 0.7);
            app.truth.add_drop(p, m1x, m1y, 0.28, 3.0);
            app.truth.add_drop(p, m2x, m2y, -0.22, 4.0);
            origins.push((m1x, m1y));
            origins.push((m2x, m2y));
        }
    }
    let n = p.cells();
    let (frozen, x) = encode_tensor(&app.primary.model.cfg().norm, &app.truth, p);
    app.frozen = frozen;
    app.primary.mass0 = project::mass(&x, n);
    app.primary.x = x;
    if let Some(sec) = app.secondary.as_mut() {
        let (_, x2) = encode_tensor(&sec.model.cfg().norm, &app.truth, p);
        sec.mass0 = project::mass(&x2, n);
        sec.x = x2;
    }
    app.tick = 0;
    app.errors = vec![f32::NAN; CHECKPOINTS.len()];
    app.persistence = vec![f32::NAN; CHECKPOINTS.len()];
    app.horizon = if origins.is_empty() {
        f32::INFINITY
    } else {
        origins.iter().map(|&(x, y)| horizon_ticks(p, x, y)).fold(f32::INFINITY, f32::min)
    };
    app.compare_metrics = [f32::NAN; 8];
}

#[wasm_bindgen]
pub async fn init(weights: Vec<u8>, config_json: String, name: String) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let raw = EngineModel::from_bytes(&weights, &config_json);
    let params = raw.cfg.params();
    let truth = State::flat(&params);
    let meta = engine::meta_of(&params).to_vec();
    let (_, x) = encode_tensor(&raw.cfg.norm, &truth, &params);
    let mass0 = project::mass(&x, params.cells());
    // Try WebGPU once, for this checkpoint; every later one reuses the
    // decision (module doc).
    let gpu = raw.to_gpu().await;
    let use_gpu = gpu.is_ok();
    let model = Rc::new(match gpu {
        Ok(g) => Backend::Gpu(g),
        Err(_) => Backend::Cpu(raw),
    });
    let primary = LoadedModel { name, model, x, mass0 };
    let (frozen, _) = encode_tensor(&primary.model.cfg().norm, &truth, &params);

    APP.with(|a| {
        *a.borrow_mut() = Some(App {
            params,
            meta,
            solver: ShallowWater::new(params),
            truth,
            frozen,
            tick: 0,
            errors: vec![f32::NAN; CHECKPOINTS.len()],
            persistence: vec![f32::NAN; CHECKPOINTS.len()],
            last_step_ms: 0.0,
            horizon: f32::INFINITY,
            reverse_model: None,
            use_gpu,
            primary,
            secondary: None,
            case: CaseKind::Drop,
            mass_projection: true,
            energy_projection: false,
            compare_metrics: [f32::NAN; 8],
        });
    });
    Ok(())
}

/// Gallery: swap the primary checkpoint for a different one from the model
/// picker. Resets the whole simulation to the current case (own docstring on
/// `apply_case`), same as switching case.
#[wasm_bindgen]
pub async fn load_model(weights: Vec<u8>, config_json: String, name: String) -> Result<(), JsValue> {
    let (truth, use_gpu) = APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        (app.truth.clone(), app.use_gpu)
    });
    let primary = load_checkpoint(weights, &config_json, name, &truth, use_gpu).await;
    APP.with(|a| {
        let mut a = a.borrow_mut();
        let app = a.as_mut().unwrap();
        app.primary = primary;
        app.reverse_model = None;
        apply_case(app);
    });
    Ok(())
}

/// Gallery: enable compare mode with a second checkpoint, stepped alongside
/// the primary from the same (re-applied) case.
#[wasm_bindgen]
pub async fn enable_compare(weights: Vec<u8>, config_json: String, name: String) -> Result<(), JsValue> {
    let (truth, use_gpu) = APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        (app.truth.clone(), app.use_gpu)
    });
    let secondary = load_checkpoint(weights, &config_json, name, &truth, use_gpu).await;
    APP.with(|a| {
        let mut a = a.borrow_mut();
        let app = a.as_mut().unwrap();
        app.secondary = Some(secondary);
        apply_case(app);
    });
    Ok(())
}

/// Gallery: leave compare mode, back to a single primary panel.
#[wasm_bindgen]
pub fn disable_compare() {
    APP.with(|a| {
        let mut a = a.borrow_mut();
        let app = a.as_mut().unwrap();
        app.secondary = None;
        apply_case(app);
    });
}

/// Gallery: switch the case picker. `case` is one of `drop`, `dam_break`,
/// `extended`; unrecognized strings are a no-op (JS validates against the
/// picker's own option list, so this only guards against a stale caller).
#[wasm_bindgen]
pub fn set_case(case: String) {
    let Some(kind) = CaseKind::parse(&case) else { return };
    APP.with(|a| {
        let mut a = a.borrow_mut();
        let app = a.as_mut().unwrap();
        app.case = kind;
        apply_case(app);
    });
}

/// Gallery: toggle the mass-conservation projection (equality constraint).
#[wasm_bindgen]
pub fn set_mass_projection(on: bool) {
    APP.with(|a| a.borrow_mut().as_mut().unwrap().mass_projection = on);
}

/// Gallery: toggle the energy-inequality projection. Uses each loaded
/// checkpoint's own `energy_r_max` (from its config.json; 1.0 — no energy
/// growth at all — for checkpoints that never set one, which is what those
/// checkpoints were actually scored with).
#[wasm_bindgen]
pub fn set_energy_projection(on: bool) {
    APP.with(|a| a.borrow_mut().as_mut().unwrap().energy_projection = on);
}

/// Load the reverse checkpoint next to the forward one. Same
/// architecture config as the forward model (its own `config.json`); only
/// the weights differ. Safe to call any time after `init`; `rewind` no-ops
/// until this has run.
#[wasm_bindgen]
pub async fn init_reverse(weights: Vec<u8>, config_json: String) -> Result<(), JsValue> {
    let raw = EngineModel::from_bytes(&weights, &config_json);
    let use_gpu = APP.with(|a| a.borrow().as_ref().unwrap().use_gpu);
    let model = Rc::new(Backend::for_model(raw, use_gpu).await);
    APP.with(|a| a.borrow_mut().as_mut().unwrap().reverse_model = Some(model));
    Ok(())
}

/// Step the reverse model back `n` ticks from the current model
/// state, mass-projected same as the forward `step`. No-op (returns
/// immediately) if `init_reverse` was never called. Does not touch the
/// solver/truth panel or the tick counter's forward meaning; `ticks()`
/// after a rewind reports `tick.saturating_sub(n)` so the demo's tick
/// readout stays consistent with "N ticks back from here".
#[wasm_bindgen]
pub async fn rewind(n: usize) {
    let has_reverse = APP.with(|a| a.borrow().as_ref().unwrap().reverse_model.is_some());
    if !has_reverse {
        return;
    }
    APP.with(|a| apply_pending(a.borrow_mut().as_mut().unwrap()));
    STEPPING.with(|s| s.set(true));
    for _ in 0..n {
        // Extract the (cheap, Rc) model handle and the state it needs
        // before awaiting: nothing below holds the APP borrow across an
        // await, so other exported fns (guarded by STEPPING where they'd
        // conflict) stay safe to call while a rewind step is in flight.
        let (rm, x, mass0, meta, n_cells) = APP.with(|a| {
            let a = a.borrow();
            let app = a.as_ref().unwrap();
            (app.reverse_model.clone().unwrap(), app.primary.x.clone(), app.primary.mass0, app.meta.clone(), app.params.cells())
        });
        let mut next = rm.step(&x, &meta).await;
        project::project_mass(&mut next, mass0, n_cells);
        APP.with(|a| {
            let mut a = a.borrow_mut();
            let app = a.as_mut().unwrap();
            app.primary.x = next;
            app.tick = app.tick.saturating_sub(1);
        });
    }
    STEPPING.with(|s| s.set(false));
}

/// `(x, y)` cell coordinates of the model's current `|h - depth|` peak,
/// physical units. Used by the rewind smoke test to check that stepping
/// back recovers a peak near a drop's known origin.
#[wasm_bindgen]
pub async fn peak_location() -> Vec<f32> {
    APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        let norm = &app.primary.model.cfg().norm;
        let n = app.params.cells();
        let (mut best_i, mut best_v) = (0usize, f32::NEG_INFINITY);
        for i in 0..n {
            let h = (app.primary.x[i] * norm.std[0] + norm.mean[0]).abs();
            if h > best_v {
                best_v = h;
                best_i = i;
            }
        }
        vec![(best_i % app.params.nx) as f32, (best_i / app.params.nx) as f32]
    })
}

fn apply_drop(app: &mut App, x: f32, y: f32, amp: f32, sigma: f32) {
    app.truth.add_drop(&app.params, x, y, amp, sigma);
    let n = app.params.cells();
    let (frozen, t) = encode_tensor(&app.primary.model.cfg().norm, &app.truth, &app.params);
    app.frozen = frozen;
    app.primary.mass0 = project::mass(&t, n);
    app.primary.x = t;
    if let Some(sec) = app.secondary.as_mut() {
        let (_, t2) = encode_tensor(&sec.model.cfg().norm, &app.truth, &app.params);
        sec.mass0 = project::mass(&t2, n);
        sec.x = t2;
    }
    app.tick = 0;
    app.errors = vec![f32::NAN; CHECKPOINTS.len()];
    app.persistence = vec![f32::NAN; CHECKPOINTS.len()];
    app.horizon = app.horizon.min(horizon_ticks(&app.params, x, y));
    app.compare_metrics = [f32::NAN; 8];
}

fn apply_reset(app: &mut App) {
    apply_case(app);
}

/// Drain drops and resets queued while a `step` was mid-flight. Called at
/// the start of `step`, and safe to call with no App yet (init not done).
fn apply_pending(app: &mut App) {
    if PENDING_RESET.with(|r| r.replace(false)) {
        apply_reset(app);
    }
    PENDING_DROPS.with(|p| {
        for d in p.borrow_mut().drain(..) {
            apply_drop(app, d[0], d[1], d[2], d[3]);
        }
    });
}

/// Add a drop to both panels and restart the comparison from here. Never
/// panics: if `step` is mid-flight, the drop is queued and applied at the
/// start of the next `step` call instead.
#[wasm_bindgen]
pub fn drop_at(x: f32, y: f32, amp: f32, sigma: f32) {
    if STEPPING.with(Cell::get) {
        PENDING_DROPS.with(|p| p.borrow_mut().push([x, y, amp, sigma]));
        return;
    }
    APP.with(|a| {
        if let Some(app) = a.borrow_mut().as_mut() {
            apply_pending(app);
            apply_drop(app, x, y, amp, sigma);
        }
    });
}

#[wasm_bindgen]
pub fn reset() {
    if STEPPING.with(Cell::get) {
        PENDING_RESET.with(|r| r.set(true));
        return;
    }
    APP.with(|a| {
        if let Some(app) = a.borrow_mut().as_mut() {
            apply_pending(app);
            apply_reset(app);
        }
    });
}

/// Step one loaded model forward once, applying whichever projections are
/// currently on (mass equality, energy inequality — CONCEPT sec. 4 step 2:
/// propose, then project). Takes the model handle and state by value/ref
/// (not a `LoadedModel`/`App` borrow) so the caller can hold nothing from
/// `APP` across the `.await` inside `model.step`.
async fn step_model(model: &Backend, x: &[f32], mass0: f32, meta: &[f32], mass_projection: bool, energy_projection: bool, params: &Params) -> Vec<f32> {
    let n = params.cells();
    let d = model.cfg().norm.denorm();
    let g = model.cfg().g;
    let prev_energy = project::energy(x, &d, g, n);
    let mut next = model.step(x, meta).await;
    if energy_projection {
        project::project_energy(&mut next, prev_energy, &d, g, model.cfg().energy_r_max, n);
    }
    if mass_projection {
        project::project_mass(&mut next, mass0, n);
    }
    next
}

/// Advance every panel by `n` ticks.
#[wasm_bindgen]
pub async fn step(n: usize) {
    APP.with(|a| apply_pending(a.borrow_mut().as_mut().unwrap()));
    STEPPING.with(|s| s.set(true));
    let t0 = js_sys::Date::now();
    for _ in 0..n {
        APP.with(|a| {
            let mut a = a.borrow_mut();
            let app = a.as_mut().unwrap();
            app.solver.step(&mut app.truth);
            app.tick += 1;
        });
        let (mass_on, energy_on, meta, params) = APP.with(|a| {
            let a = a.borrow();
            let app = a.as_ref().unwrap();
            (app.mass_projection, app.energy_projection, app.meta.clone(), app.params)
        });
        let (primary_model, primary_x, primary_mass0) = APP.with(|a| {
            let a = a.borrow();
            let app = a.as_ref().unwrap();
            (app.primary.model.clone(), app.primary.x.clone(), app.primary.mass0)
        });
        let next_primary = step_model(&primary_model, &primary_x, primary_mass0, &meta, mass_on, energy_on, &params).await;
        let secondary_handle = APP.with(|a| {
            let a = a.borrow();
            let app = a.as_ref().unwrap();
            app.secondary.as_ref().map(|sec| (sec.model.clone(), sec.x.clone(), sec.mass0))
        });
        let next_secondary = match secondary_handle {
            Some((m, x, m0)) => Some(step_model(&m, &x, m0, &meta, mass_on, energy_on, &params).await),
            None => None,
        };
        APP.with(|a| {
            let mut a = a.borrow_mut();
            let app = a.as_mut().unwrap();
            app.primary.x = next_primary;
            if let Some(next) = next_secondary {
                app.secondary.as_mut().unwrap().x = next;
            }
        });

        let (want_rollout, want_metrics) = APP.with(|a| {
            let a = a.borrow();
            let app = a.as_ref().unwrap();
            (CHECKPOINTS.iter().position(|&c| c == app.tick), app.tick % METRICS_EVERY == 0)
        });
        if let Some(i) = want_rollout {
            APP.with(|a| {
                let mut a = a.borrow_mut();
                let app = a.as_mut().unwrap();
                let pred = app.primary.x.clone();
                let mut target = vec![0.0f32; pred.len()];
                app.primary.model.cfg().norm.encode(&app.truth, &mut target);
                app.errors[i] = rmse(&pred, &target);
                app.persistence[i] = rmse(&app.frozen, &target);
            });
        }
        if want_metrics {
            refresh_compare_metrics();
        }
    }
    let dt = js_sys::Date::now() - t0;
    APP.with(|a| {
        a.borrow_mut().as_mut().unwrap().last_step_ms = dt / n.max(1) as f64;
    });
    STEPPING.with(|s| s.set(false));
}

/// Live localized metrics (corr, amplitude ratio, wavefront lag against the
/// solver) for compare mode, refreshed every `METRICS_EVERY` ticks. Origin
/// for the radial profile is the domain center.
fn refresh_compare_metrics() {
    APP.with(|a| {
        let mut a = a.borrow_mut();
        let app = a.as_mut().unwrap();
        let n = app.params.cells();
        let (nx, ny) = (app.params.nx, app.params.ny);
        let (cx, cy) = (nx as f32 / 2.0, ny as f32 / 2.0);
        let solver_h: Vec<f32> = app.truth.h.iter().map(|&h| h - app.params.depth).collect();
        let sr = radial_profile(&solver_h, nx, ny, cx, cy);

        let ph = decode_h(&app.primary.x, &app.primary.model.cfg().norm, n);
        let pr = radial_profile(&ph, nx, ny, cx, cy);
        let corr_a = correlation(&ph, &solver_h);
        let amp_a = peak(&ph) / peak(&solver_h).max(1e-12);
        let lag_a = xcorr_lag(&pr, &sr, 10);
        let bias_a = bias(&ph, &solver_h);

        let (corr_b, amp_b, lag_b, bias_b) = if let Some(sec) = &app.secondary {
            let sh = decode_h(&sec.x, &sec.model.cfg().norm, n);
            let sh_radial = radial_profile(&sh, nx, ny, cx, cy);
            (correlation(&sh, &solver_h), peak(&sh) / peak(&solver_h).max(1e-12), xcorr_lag(&sh_radial, &sr, 10), bias(&sh, &solver_h))
        } else {
            (f32::NAN, f32::NAN, f32::NAN, f32::NAN)
        };
        app.compare_metrics = [corr_a, amp_a, lag_a, corr_b, amp_b, lag_b, bias_a, bias_b];
    });
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    let s: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (x - y) as f64;
            d * d
        })
        .sum();
    (s / a.len() as f64).sqrt() as f32
}

/// Surface height anomaly from the solver, `[ny * nx]`.
#[wasm_bindgen]
pub fn solver_field() -> Vec<f32> {
    APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        app.truth.h.iter().map(|&h| h - app.params.depth).collect()
    })
}

/// Surface height anomaly from the primary model, and the model-minus-solver
/// map. Single-view mode's third panel; compare mode uses `secondary_field`
/// instead and drops the divergence map (three panels: A, B, solver).
#[wasm_bindgen]
pub async fn model_and_divergence() -> Vec<f32> {
    APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        let norm = &app.primary.model.cfg().norm;
        let n = app.params.cells();
        let mut out = Vec::with_capacity(2 * n);
        for value in app.primary.x.iter().take(n) {
            out.push(value * norm.std[0] + norm.mean[0]);
        }
        for i in 0..n {
            out.push(out[i] - (app.truth.h[i] - app.params.depth));
        }
        out
    })
}

/// Compare mode: the secondary model's surface height anomaly, `[ny * nx]`.
/// Empty if compare mode is off.
#[wasm_bindgen]
pub async fn secondary_field() -> Vec<f32> {
    APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        match &app.secondary {
            Some(sec) => decode_h(&sec.x, &sec.model.cfg().norm, app.params.cells()),
            None => Vec::new(),
        }
    })
}

/// `[corrA, ampA, lagA, corrB, ampB, lagB, biasA, biasB]`, NaN for the B
/// slots when compare mode is off. See `refresh_compare_metrics`.
#[wasm_bindgen]
pub fn compare_metrics() -> Vec<f32> {
    APP.with(|a| a.borrow().as_ref().unwrap().compare_metrics.to_vec())
}

/// Per-cell xr ensemble spread (CONCEPT sec. 4, task 10): not implemented
/// in this engine: `app.js` never calls it and the page's acceptance checks
/// do not cover it. Returns empty.
#[wasm_bindgen]
pub async fn spread_field(_k: usize) -> Vec<f32> {
    Vec::new()
}

#[wasm_bindgen]
pub fn ticks() -> usize {
    APP.with(|a| a.borrow().as_ref().unwrap().tick)
}

/// `[model@10, model@50, model@200, frozen@10, frozen@50, frozen@200]`,
/// NaN for checkpoints not reached yet.
#[wasm_bindgen]
pub fn rollout_errors() -> Vec<f32> {
    APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        let mut v = app.errors.clone();
        v.extend_from_slice(&app.persistence);
        v
    })
}

/// Ticks until the nearest drop reaches the nearest wall (min over drops
/// since the last reset), `Infinity` with no drop placed yet.
#[wasm_bindgen]
pub fn horizon() -> f64 {
    APP.with(|a| a.borrow().as_ref().unwrap().horizon as f64)
}

#[wasm_bindgen]
pub fn ms_per_tick() -> f64 {
    APP.with(|a| a.borrow().as_ref().unwrap().last_step_ms)
}

#[wasm_bindgen]
pub fn grid_size() -> Vec<usize> {
    APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        vec![app.params.nx, app.params.ny]
    })
}

/// The mask the loaded checkpoint runs with (read from its config.json).
#[wasm_bindgen]
pub fn mask_name() -> String {
    APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        format!("{:?}", app.primary.model.cfg().mask)
    })
}

/// Name of the currently loaded primary checkpoint (gallery STATUS row).
#[wasm_bindgen]
pub fn model_name() -> String {
    APP.with(|a| a.borrow().as_ref().unwrap().primary.name.clone())
}

/// `"webgpu"` or `"cpu"` — which backend `init`'s capability check landed
/// on (module doc). For the STATUS row and for headless/manual
/// verification that a page is actually running on the backend it claims.
#[wasm_bindgen]
pub fn backend_name() -> String {
    APP.with(|a| {
        let a = a.borrow();
        match &*a.as_ref().unwrap().primary.model {
            Backend::Cpu(_) => "cpu".to_string(),
            Backend::Gpu(_) => "webgpu".to_string(),
        }
    })
}
