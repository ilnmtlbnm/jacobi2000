//! Browser engine: solver on the left, learned operator(s) on the right, one
//! `window.__app` surface driven by the page (and by the headless test).
//!
//! WASM rules that are load-bearing here:
//! - never a synchronous GPU readback (`into_data()` deadlocks the tab);
//!   every read goes through `into_data_async().await`;
//! - the model state stays on the GPU between ticks, so N ticks cost one
//!   readback, not N.
//!
//! Gallery (task: model picker + case picker + compare mode): a checkpoint is
//! a [`LoadedModel`] — its own weights, normalizer and mask, each stepped in
//! its own tensor. `App.primary` is always present; `App.secondary` exists
//! only in compare mode, stepped in lockstep with the same solver truth.
//! Every model shares one grid (64x64, patch 4) and one solver-conditioning
//! tensor (`App.meta`, built from the primary's physics params) — all
//! shipped checkpoints share that grid and near-identical physics
//! (dt~0.090, nu=0.001), so this is a fair simplification, not an
//! approximation that changes which model wins.

use std::cell::{Cell, RefCell};

use burn::module::Module;
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use jacobi2000::{energy, project_energy, project_mass, Denorm, Jacobi2000, Jacobi2000Config, Mask};
use solvers::{Params, ShallowWater, State};
use train::checkpoint::Meta;
use train::data::{meta_of, Normalizer, META_DIM};
use train::metrics::{bias, correlation, decode_h, peak, radial_profile, xcorr_lag};
use wasm_bindgen::prelude::*;

type B = burn::backend::Wgpu;

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

/// One checkpoint, stepped in its own tensor. Shared grid/mask shape with
/// every other loaded checkpoint (see module doc).
struct LoadedModel {
    name: String,
    model: Jacobi2000<B>,
    norm: Normalizer,
    depth: f32,
    g: f32,
    energy_r_max: f32,
    mask_spec: Mask,
    mask: Option<Tensor<B, 2>>,
    x: Tensor<B, 4>,
    mass0: Tensor<B, 2>,
}

impl LoadedModel {
    fn denorm(&self) -> Denorm {
        Denorm { mean: self.norm.mean, std: self.norm.std, depth: self.depth }
    }
}

struct App {
    params: Params,
    device: <B as Backend>::Device,
    meta: Tensor<B, 2>,
    solver: ShallowWater,
    truth: State,
    /// Box total of h at the last reset: the constant the projection restores.
    /// Field frozen at the last reset: the persistence baseline.
    frozen: Vec<f32>,
    tick: usize,
    errors: Vec<f32>,
    persistence: Vec<f32>,
    last_step_ms: f64,
    /// Ticks until the wave from the nearest drop reaches the nearest wall
    /// (min over drops since the last reset). `INFINITY` with no drop yet.
    horizon: f32,
    /// Task C7: the reverse (backward-stepping) checkpoint, loaded
    /// separately via `init_reverse`. `None` until then, so `rewind` is a
    /// no-op (logged to STATUS by the caller) on a page that never loaded one.
    reverse_model: Option<Jacobi2000<B>>,
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
    /// Set for the duration of `step`'s async body (which awaits GPU
    /// readback partway through). While set, `drop_at`/`reset` must not
    /// touch `APP` — they queue instead — since `step` re-borrows it after
    /// every `.await` and a concurrent borrow would panic.
    static STEPPING: Cell<bool> = const { Cell::new(false) };
    /// Drops/resets that arrived while `STEPPING`; applied at the start of
    /// the next `step` call so `drop_at`/`reset` never have to panic, and so
    /// a drop never perturbs the tick count of the step already in flight.
    static PENDING_DROPS: RefCell<Vec<[f32; 4]>> = const { RefCell::new(Vec::new()) };
    static PENDING_RESET: Cell<bool> = const { Cell::new(false) };
}

fn encode_tensor(
    norm: &Normalizer,
    s: &State,
    p: &Params,
    device: &<B as Backend>::Device,
) -> (Vec<f32>, Tensor<B, 4>) {
    let mut packed = vec![0.0f32; 3 * p.cells()];
    norm.encode(s, &mut packed);
    let t = Tensor::<B, 1>::from_floats(packed.as_slice(), device).reshape([1, 3, p.ny, p.nx]);
    (packed, t)
}

fn load_checkpoint(
    weights: Vec<u8>,
    meta_json: &str,
    name: String,
    device: &<B as Backend>::Device,
    truth: &State,
    params: &Params,
) -> Result<LoadedModel, JsValue> {
    let meta: Meta = serde_json::from_str(meta_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let cfg = Jacobi2000Config::new()
        .with_nx(meta.nx)
        .with_ny(meta.ny)
        .with_patch(meta.patch)
        .with_dim(meta.dim)
        .with_layers(meta.layers)
        .with_heads(meta.heads)
        .with_ff_mult(meta.ff_mult);
    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(weights, device)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let model = cfg.init::<B>(device).load_record(record);
    let mask = meta.mask.to_tensor::<B>(&model.grid, device);
    let (_, x) = encode_tensor(&meta.norm, truth, params, device);
    let mass0 = jacobi2000::mass(&x);
    Ok(LoadedModel {
        name,
        model,
        norm: meta.norm,
        depth: meta.depth,
        g: meta.g,
        energy_r_max: meta.energy_r_max,
        mask_spec: meta.mask,
        mask,
        x,
        mass0,
    })
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
    let (frozen, x) = encode_tensor(&app.primary.norm, &app.truth, p, &app.device);
    app.frozen = frozen;
    app.primary.mass0 = jacobi2000::mass(&x);
    app.primary.x = x;
    if let Some(sec) = app.secondary.as_mut() {
        let (_, x2) = encode_tensor(&sec.norm, &app.truth, p, &app.device);
        sec.mass0 = jacobi2000::mass(&x2);
        sec.x = x2;
    }
    app.tick = 0;
    app.errors = vec![f32::NAN; CHECKPOINTS.len()];
    app.persistence = vec![f32::NAN; CHECKPOINTS.len()];
    app.horizon = if origins.is_empty() {
        f32::INFINITY
    } else {
        origins
            .iter()
            .map(|&(x, y)| horizon_ticks(p, x, y))
            .fold(f32::INFINITY, f32::min)
    };
    app.compare_metrics = [f32::NAN; 8];
}

#[wasm_bindgen]
pub async fn init(weights: Vec<u8>, meta_json: String, name: String) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let meta: Meta = serde_json::from_str(&meta_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let device = burn::backend::wgpu::WgpuDevice::default();
    burn::backend::wgpu::init_setup_async::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
        &device,
        Default::default(),
    )
    .await;

    let params = meta.params();
    let truth = State::flat(&params);
    let meta_t =
        Tensor::<B, 1>::from_floats(meta_of(&params).as_slice(), &device).reshape([1, META_DIM]);
    let primary = load_checkpoint(weights, &meta_json, name, &device, &truth, &params)?;
    let (frozen, _) = encode_tensor(&primary.norm, &truth, &params, &device);

    APP.with(|a| {
        *a.borrow_mut() = Some(App {
            params,
            device: device.clone(),
            meta: meta_t,
            solver: ShallowWater::new(params),
            truth,
            frozen,
            tick: 0,
            errors: vec![f32::NAN; CHECKPOINTS.len()],
            persistence: vec![f32::NAN; CHECKPOINTS.len()],
            last_step_ms: 0.0,
            horizon: f32::INFINITY,
            reverse_model: None,
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
pub async fn load_model(weights: Vec<u8>, meta_json: String, name: String) -> Result<(), JsValue> {
    let (device, truth, params) = APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        (app.device.clone(), app.truth.clone(), app.params)
    });
    let primary = load_checkpoint(weights, &meta_json, name, &device, &truth, &params)?;
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
pub async fn enable_compare(weights: Vec<u8>, meta_json: String, name: String) -> Result<(), JsValue> {
    let (device, truth, params) = APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        (app.device.clone(), app.truth.clone(), app.params)
    });
    let secondary = load_checkpoint(weights, &meta_json, name, &device, &truth, &params)?;
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
/// checkpoint's own `energy_r_max` (from its meta.json; 1.0 — no energy
/// growth at all — for checkpoints that never set one, which is what those
/// checkpoints were actually scored with).
#[wasm_bindgen]
pub fn set_energy_projection(on: bool) {
    APP.with(|a| a.borrow_mut().as_mut().unwrap().energy_projection = on);
}

/// Task C7: load the reverse checkpoint next to the forward one. Same
/// architecture config as the forward model (`Jacobi2000Config` from its own
/// `meta.json`); only the weights differ. Safe to call any time after
/// `init`; `rewind` no-ops until this has run.
#[wasm_bindgen]
pub async fn init_reverse(weights: Vec<u8>, meta_json: String) -> Result<(), JsValue> {
    let meta: Meta = serde_json::from_str(&meta_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let device = APP.with(|a| a.borrow().as_ref().unwrap().device.clone());
    let cfg = Jacobi2000Config::new()
        .with_nx(meta.nx)
        .with_ny(meta.ny)
        .with_patch(meta.patch)
        .with_dim(meta.dim)
        .with_layers(meta.layers)
        .with_heads(meta.heads)
        .with_ff_mult(meta.ff_mult);
    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(weights, &device)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let model = cfg.init::<B>(&device).load_record(record);
    APP.with(|a| a.borrow_mut().as_mut().unwrap().reverse_model = Some(model));
    Ok(())
}

/// Task C7: step the reverse model back `n` ticks from the current model
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
        let (x, mask, meta, mass0) = APP.with(|a| {
            let a = a.borrow();
            let app = a.as_ref().unwrap();
            (
                app.primary.x.clone(),
                app.primary.mask.clone(),
                app.meta.clone(),
                app.primary.mass0.clone(),
            )
        });
        let next = APP.with(|a| {
            let a = a.borrow();
            let app = a.as_ref().unwrap();
            let rm = app.reverse_model.as_ref().unwrap();
            project_mass(rm.step(x, meta, mask), mass0)
        });
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
    let (x, norm, params) = APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        (app.primary.x.clone(), app.primary.norm, app.params)
    });
    let data = x.into_data_async().await.unwrap();
    let v = data.as_slice::<f32>().unwrap();
    let n = params.cells();
    let (mut best_i, mut best_v) = (0usize, f32::NEG_INFINITY);
    for (i, &vi) in v.iter().enumerate().take(n) {
        let h = (vi * norm.std[0] + norm.mean[0]).abs();
        if h > best_v {
            best_v = h;
            best_i = i;
        }
    }
    vec![(best_i % params.nx) as f32, (best_i / params.nx) as f32]
}

fn apply_drop(app: &mut App, x: f32, y: f32, amp: f32, sigma: f32) {
    app.truth.add_drop(&app.params, x, y, amp, sigma);
    let (frozen, t) = encode_tensor(&app.primary.norm, &app.truth, &app.params, &app.device);
    app.frozen = frozen;
    app.primary.mass0 = jacobi2000::mass(&t);
    app.primary.x = t;
    if let Some(sec) = app.secondary.as_mut() {
        let (_, t2) = encode_tensor(&sec.norm, &app.truth, &app.params, &app.device);
        sec.mass0 = jacobi2000::mass(&t2);
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
            // Drain anything left over from a step/rewind call that finished
            // without a following step/rewind to flush it (e.g. drops queued
            // mid-flight, applied here instead of leaking into whichever
            // step/rewind call happens to run next).
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
/// propose, then project).
fn step_model(
    lm: &LoadedModel,
    meta: &Tensor<B, 2>,
    mass_projection: bool,
    energy_projection: bool,
) -> Tensor<B, 4> {
    let prev_energy = energy(&lm.x, &lm.denorm(), lm.g);
    let mut next = lm.model.step(lm.x.clone(), meta.clone(), lm.mask.clone());
    if energy_projection {
        next = project_energy(next, prev_energy, &lm.denorm(), lm.g, lm.energy_r_max);
    }
    if mass_projection {
        next = project_mass(next, lm.mass0.clone());
    }
    next
}

/// Advance every panel by `n` ticks. One GPU readback per checkpoint, one
/// per compare-metrics refresh, plus one at the end for rendering.
#[wasm_bindgen]
pub async fn step(n: usize) {
    APP.with(|a| apply_pending(a.borrow_mut().as_mut().unwrap()));
    STEPPING.with(|s| s.set(true));
    let t0 = js_sys::Date::now();
    for _ in 0..n {
        // Advance the solver (CPU) and every loaded model (GPU) by one tick.
        APP.with(|a| {
            let mut a = a.borrow_mut();
            let app = a.as_mut().unwrap();
            app.solver.step(&mut app.truth);
            app.tick += 1;
        });
        let (mass_on, energy_on, meta) = APP.with(|a| {
            let a = a.borrow();
            let app = a.as_ref().unwrap();
            (app.mass_projection, app.energy_projection, app.meta.clone())
        });
        let next_primary =
            APP.with(|a| step_model(&a.borrow().as_ref().unwrap().primary, &meta, mass_on, energy_on));
        let next_secondary = APP.with(|a| {
            a.borrow()
                .as_ref()
                .unwrap()
                .secondary
                .as_ref()
                .map(|sec| step_model(sec, &meta, mass_on, energy_on))
        });
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
            (
                CHECKPOINTS.iter().position(|&c| c == app.tick),
                app.tick % METRICS_EVERY == 0,
            )
        });
        if let Some(i) = want_rollout {
            let data = APP
                .with(|a| a.borrow().as_ref().unwrap().primary.x.clone())
                .into_data_async()
                .await
                .unwrap();
            let pred = data.as_slice::<f32>().unwrap().to_vec();
            APP.with(|a| {
                let mut a = a.borrow_mut();
                let app = a.as_mut().unwrap();
                let mut target = vec![0.0f32; pred.len()];
                app.primary.norm.encode(&app.truth, &mut target);
                app.errors[i] = rmse(&pred, &target);
                app.persistence[i] = rmse(&app.frozen, &target);
            });
        }
        if want_metrics {
            refresh_compare_metrics().await;
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
/// for the radial profile is the domain center — a fair generic choice
/// across cases (drop/dam-break/extended may not share a single source
/// point).
async fn refresh_compare_metrics() {
    let (n, nx, ny, cx, cy, solver_h, primary_x, primary_norm, secondary) = APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        let n = app.params.cells();
        let solver_h: Vec<f32> = app.truth.h.iter().map(|&h| h - app.params.depth).collect();
        (
            n,
            app.params.nx,
            app.params.ny,
            app.params.nx as f32 / 2.0,
            app.params.ny as f32 / 2.0,
            solver_h,
            app.primary.x.clone(),
            app.primary.norm,
            app.secondary.as_ref().map(|s| (s.x.clone(), s.norm)),
        )
    });
    let pdata = primary_x.into_data_async().await.unwrap();
    let ph = decode_h(pdata.as_slice::<f32>().unwrap(), &primary_norm, n);
    let sr = radial_profile(&solver_h, nx, ny, cx, cy);
    let pr = radial_profile(&ph, nx, ny, cx, cy);
    let corr_a = correlation(&ph, &solver_h);
    let amp_a = peak(&ph) / peak(&solver_h).max(1e-12);
    let lag_a = xcorr_lag(&pr, &sr, 10);

    let bias_a = bias(&ph, &solver_h);

    let (corr_b, amp_b, lag_b, bias_b) = if let Some((sx, snorm)) = secondary {
        let sdata = sx.into_data_async().await.unwrap();
        let sh = decode_h(sdata.as_slice::<f32>().unwrap(), &snorm, n);
        let sh_radial = radial_profile(&sh, nx, ny, cx, cy);
        (
            correlation(&sh, &solver_h),
            peak(&sh) / peak(&solver_h).max(1e-12),
            xcorr_lag(&sh_radial, &sr, 10),
            bias(&sh, &solver_h),
        )
    } else {
        (f32::NAN, f32::NAN, f32::NAN, f32::NAN)
    };
    APP.with(|a| {
        a.borrow_mut().as_mut().unwrap().compare_metrics =
            [corr_a, amp_a, lag_a, corr_b, amp_b, lag_b, bias_a, bias_b];
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
        app.truth
            .h
            .iter()
            .map(|&h| h - app.params.depth)
            .collect()
    })
}

/// Surface height anomaly from the primary model, and the model-minus-solver
/// map. Single-view mode's third panel; compare mode uses `secondary_field`
/// instead and drops the divergence map (three panels: A, B, solver).
#[wasm_bindgen]
pub async fn model_and_divergence() -> Vec<f32> {
    let (x, norm, params, truth) = APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        (app.primary.x.clone(), app.primary.norm, app.params, app.truth.h.clone())
    });
    let data = x.into_data_async().await.unwrap();
    let v = data.as_slice::<f32>().unwrap();
    let n = params.cells();
    let mut out = Vec::with_capacity(2 * n);
    for value in v.iter().take(n) {
        out.push(value * norm.std[0] + norm.mean[0]);
    }
    for i in 0..n {
        out.push(out[i] - (truth[i] - params.depth));
    }
    out
}

/// Compare mode: the secondary model's surface height anomaly, `[ny * nx]`.
/// Empty if compare mode is off.
#[wasm_bindgen]
pub async fn secondary_field() -> Vec<f32> {
    let Some((x, norm, n)) = APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        app.secondary.as_ref().map(|s| (s.x.clone(), s.norm, app.params.cells()))
    }) else {
        return Vec::new();
    };
    let data = x.into_data_async().await.unwrap();
    decode_h(data.as_slice::<f32>().unwrap(), &norm, n)
}

/// `[corrA, ampA, lagA, corrB, ampB, lagB, biasA, biasB]`, NaN for the B
/// slots when compare mode is off. See `refresh_compare_metrics`.
#[wasm_bindgen]
pub fn compare_metrics() -> Vec<f32> {
    APP.with(|a| a.borrow().as_ref().unwrap().compare_metrics.to_vec())
}

/// Per-cell xr ensemble spread of the surface height (task 10, CONCEPT sec.
/// 4), `[ny * nx]`, physical units, for `k` jittered patch-grid offsets on
/// the *current* model state (does not advance the tick). TODO(demo): wire
/// into the divergence panel as a fourth `paint()` call, e.g.
/// `paint(ctx.spread, await engine.spread_field(4), 0.01)` next to the
/// existing `model_and_divergence()` call in `web/app.js`'s `render()`.
#[wasm_bindgen]
pub async fn spread_field(k: usize) -> Vec<f32> {
    let (model, x, meta, mask, norm) = APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        (
            app.primary.model.clone(),
            app.primary.x.clone(),
            app.meta.clone(),
            app.primary.mask.clone(),
            app.primary.norm,
        )
    });
    let (_, spread) = train::xr_step(&model, x, meta, mask, k);
    let data = spread.into_data_async().await.unwrap();
    let v = data.as_slice::<f32>().unwrap();
    let n = APP.with(|a| a.borrow().as_ref().unwrap().params.cells());
    v[..n].iter().map(|&s| s * norm.std[0]).collect()
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

/// The mask the loaded checkpoint runs with (read from its meta.json).
#[wasm_bindgen]
pub fn mask_name() -> String {
    APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        format!("{:?}", app.primary.mask_spec)
    })
}

/// Name of the currently loaded primary checkpoint (gallery STATUS row).
#[wasm_bindgen]
pub fn model_name() -> String {
    APP.with(|a| a.borrow().as_ref().unwrap().primary.name.clone())
}
