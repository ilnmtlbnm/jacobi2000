//! Browser engine: solver on the left, learned operator on the right, one
//! `window.__app` surface driven by the page (and by the headless test).
//!
//! WASM rules that are load-bearing here:
//! - never a synchronous GPU readback (`into_data()` deadlocks the tab);
//!   every read goes through `into_data_async().await`;
//! - the model state stays on the GPU between ticks, so N ticks cost one
//!   readback, not N.

use std::cell::{Cell, RefCell};

use burn::module::Module;
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use jacobi2000::{project_mass, Jacobi2000, Jacobi2000Config, Mask};
use solvers::{Params, ShallowWater, State};
use train::checkpoint::Meta;
use train::data::{meta_of, Normalizer, META_DIM};
use wasm_bindgen::prelude::*;

type B = burn::backend::Wgpu;

/// Ticks at which the rollout error is recorded, matching the native CLI.
const CHECKPOINTS: [usize; 3] = [10, 50, 200];

struct App {
    params: Params,
    norm: Normalizer,
    model: Jacobi2000<B>,
    device: <B as Backend>::Device,
    meta: Tensor<B, 2>,
    solver: ShallowWater,
    truth: State,
    /// Model state, normalized, kept on the GPU between ticks.
    x: Tensor<B, 4>,
    /// Box total of h at the last reset: the constant the projection restores.
    mass0: Tensor<B, 2>,
    /// Field frozen at the last reset: the persistence baseline.
    frozen: Vec<f32>,
    tick: usize,
    errors: Vec<f32>,
    persistence: Vec<f32>,
    last_step_ms: f64,
    /// Ticks until the wave from the nearest drop reaches the nearest wall
    /// (min over drops since the last reset). `INFINITY` with no drop yet.
    horizon: f32,
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

#[wasm_bindgen]
pub async fn init(weights: Vec<u8>, meta_json: String) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let meta: Meta = serde_json::from_str(&meta_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let device = burn::backend::wgpu::WgpuDevice::default();
    burn::backend::wgpu::init_setup_async::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
        &device,
        Default::default(),
    )
    .await;

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

    let params = meta.params();
    let truth = State::flat(&params);
    let (frozen, x) = encode_tensor(&meta.norm, &truth, &params, &device);
    let meta_t =
        Tensor::<B, 1>::from_floats(meta_of(&params).as_slice(), &device).reshape([1, META_DIM]);

    APP.with(|a| {
        *a.borrow_mut() = Some(App {
            params,
            norm: meta.norm,
            model,
            device: device.clone(),
            meta: meta_t,
            solver: ShallowWater::new(params),
            truth,
            mass0: jacobi2000::mass(&x),
            x,
            frozen,
            tick: 0,
            errors: vec![f32::NAN; CHECKPOINTS.len()],
            persistence: vec![f32::NAN; CHECKPOINTS.len()],
            last_step_ms: 0.0,
            horizon: f32::INFINITY,
        });
    });
    Ok(())
}

fn apply_drop(app: &mut App, x: f32, y: f32, amp: f32, sigma: f32) {
    app.truth.add_drop(&app.params, x, y, amp, sigma);
    let (frozen, t) = encode_tensor(&app.norm, &app.truth, &app.params, &app.device);
    app.frozen = frozen;
    app.mass0 = jacobi2000::mass(&t);
    app.x = t;
    app.tick = 0;
    app.errors = vec![f32::NAN; CHECKPOINTS.len()];
    app.persistence = vec![f32::NAN; CHECKPOINTS.len()];
    app.horizon = app.horizon.min(horizon_ticks(&app.params, x, y));
}

fn apply_reset(app: &mut App) {
    app.truth = State::flat(&app.params);
    let (frozen, t) = encode_tensor(&app.norm, &app.truth, &app.params, &app.device);
    app.frozen = frozen;
    app.mass0 = jacobi2000::mass(&t);
    app.x = t;
    app.tick = 0;
    app.errors = vec![f32::NAN; CHECKPOINTS.len()];
    app.persistence = vec![f32::NAN; CHECKPOINTS.len()];
    app.horizon = f32::INFINITY;
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
            apply_reset(app);
        }
    });
}

/// Advance both panels by `n` ticks. One GPU readback per checkpoint, plus
/// one at the end for rendering.
#[wasm_bindgen]
pub async fn step(n: usize) {
    APP.with(|a| apply_pending(a.borrow_mut().as_mut().unwrap()));
    STEPPING.with(|s| s.set(true));
    let t0 = js_sys::Date::now();
    for _ in 0..n {
        // Advance the model on the GPU and the solver on the CPU.
        let (x, mask, meta, mass0) = APP.with(|a| {
            let mut a = a.borrow_mut();
            let app = a.as_mut().unwrap();
            app.solver.step(&mut app.truth);
            app.tick += 1;
            (
                app.x.clone(),
                None::<Tensor<B, 2>>,
                app.meta.clone(),
                app.mass0.clone(),
            )
        });
        let next = APP.with(|a| {
            let a = a.borrow();
            let app = a.as_ref().unwrap();
            project_mass(app.model.step(x, meta, mask), mass0)
        });
        APP.with(|a| a.borrow_mut().as_mut().unwrap().x = next);

        let want = APP.with(|a| {
            let a = a.borrow();
            let app = a.as_ref().unwrap();
            CHECKPOINTS.iter().position(|&c| c == app.tick)
        });
        if let Some(i) = want {
            let data = APP
                .with(|a| a.borrow().as_ref().unwrap().x.clone())
                .into_data_async()
                .await
                .unwrap();
            let pred = data.as_slice::<f32>().unwrap().to_vec();
            APP.with(|a| {
                let mut a = a.borrow_mut();
                let app = a.as_mut().unwrap();
                let mut target = vec![0.0f32; pred.len()];
                app.norm.encode(&app.truth, &mut target);
                app.errors[i] = rmse(&pred, &target);
                app.persistence[i] = rmse(&app.frozen, &target);
            });
        }
    }
    let dt = js_sys::Date::now() - t0;
    APP.with(|a| {
        a.borrow_mut().as_mut().unwrap().last_step_ms = dt / n.max(1) as f64;
    });
    STEPPING.with(|s| s.set(false));
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

/// Surface height anomaly from the model, and the model-minus-solver map.
#[wasm_bindgen]
pub async fn model_and_divergence() -> Vec<f32> {
    let (x, norm, params, truth) = APP.with(|a| {
        let a = a.borrow();
        let app = a.as_ref().unwrap();
        (app.x.clone(), app.norm, app.params, app.truth.h.clone())
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

/// The mask the demo runs with. Full tonight; stencil is the next step and
/// the engine already takes it as an argument.
#[wasm_bindgen]
pub fn mask_name() -> String {
    format!("{:?}", Mask::Full)
}
