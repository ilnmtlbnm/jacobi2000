//! Browser engine: solver on the left, learned operator on the right, one
//! `window.__app` surface driven by the page (and by the headless test).
//!
//! WASM rules that are load-bearing here:
//! - never a synchronous GPU readback (`into_data()` deadlocks the tab);
//!   every read goes through `into_data_async().await`;
//! - the model state stays on the GPU between ticks, so N ticks cost one
//!   readback, not N.

use std::cell::RefCell;

use burn::module::Module;
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use jacobi2000::{Jacobi2000, Jacobi2000Config, Mask};
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
    /// Field frozen at the last reset: the persistence baseline.
    frozen: Vec<f32>,
    tick: usize,
    errors: Vec<f32>,
    persistence: Vec<f32>,
    last_step_ms: f64,
}

thread_local! {
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
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
            x,
            frozen,
            tick: 0,
            errors: vec![f32::NAN; CHECKPOINTS.len()],
            persistence: vec![f32::NAN; CHECKPOINTS.len()],
            last_step_ms: 0.0,
        });
    });
    Ok(())
}

/// Add a drop to both panels and restart the comparison from here.
#[wasm_bindgen]
pub fn drop_at(x: f32, y: f32, amp: f32, sigma: f32) {
    APP.with(|a| {
        let mut a = a.borrow_mut();
        let app = a.as_mut().unwrap();
        app.truth.add_drop(&app.params, x, y, amp, sigma);
        let (frozen, t) = encode_tensor(&app.norm, &app.truth, &app.params, &app.device);
        app.frozen = frozen;
        app.x = t;
        app.tick = 0;
        app.errors = vec![f32::NAN; CHECKPOINTS.len()];
        app.persistence = vec![f32::NAN; CHECKPOINTS.len()];
    });
}

#[wasm_bindgen]
pub fn reset() {
    APP.with(|a| {
        let mut a = a.borrow_mut();
        let app = a.as_mut().unwrap();
        app.truth = State::flat(&app.params);
        let (frozen, t) = encode_tensor(&app.norm, &app.truth, &app.params, &app.device);
        app.frozen = frozen;
        app.x = t;
        app.tick = 0;
        app.errors = vec![f32::NAN; CHECKPOINTS.len()];
        app.persistence = vec![f32::NAN; CHECKPOINTS.len()];
    });
}

/// Advance both panels by `n` ticks. One GPU readback per checkpoint, plus
/// one at the end for rendering.
#[wasm_bindgen]
pub async fn step(n: usize) {
    let t0 = js_sys::Date::now();
    for _ in 0..n {
        // Advance the model on the GPU and the solver on the CPU.
        let (x, mask, meta) = APP.with(|a| {
            let mut a = a.borrow_mut();
            let app = a.as_mut().unwrap();
            app.solver.step(&mut app.truth);
            app.tick += 1;
            (app.x.clone(), None::<Tensor<B, 2>>, app.meta.clone())
        });
        let next = APP.with(|a| {
            let a = a.borrow();
            let app = a.as_ref().unwrap();
            app.model.step(x, meta, mask)
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
