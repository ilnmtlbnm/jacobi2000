//! Step mode with a rollout loss. Generic over the backend: the binary crate
//! picks wgpu (native training) or ndarray (CPU).

use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use jacobi2000::{Jacobi2000, Mask};

use crate::data::Generator;

#[derive(Debug, Clone, Copy)]
pub struct FitConfig {
    pub steps: usize,
    pub lr: f64,
    pub weight_decay: f32,
    pub warmup: usize,
    /// Ticks rolled out on the model's own outputs, once the curriculum is over.
    pub rollout: usize,
    /// Single-step only before this iteration (pushforward curriculum).
    pub rollout_after: usize,
    pub log_every: usize,
    pub mask: Mask,
}

impl Default for FitConfig {
    fn default() -> Self {
        FitConfig {
            steps: 2000,
            lr: 3e-4,
            weight_decay: 1e-4,
            warmup: 100,
            rollout: 4,
            rollout_after: 500,
            log_every: 100,
            mask: Mask::Full,
        }
    }
}

pub struct LogRow {
    pub step: usize,
    pub loss: f32,
    pub secs: f64,
}

pub fn fit<B: AutodiffBackend>(
    mut model: Jacobi2000<B>,
    gen: &mut Generator,
    cfg: &FitConfig,
    device: &B::Device,
    mut on_log: impl FnMut(&LogRow),
) -> Jacobi2000<B> {
    let grid = model.grid;
    let mask = cfg.mask.to_tensor::<B>(&grid, device);
    let mut opt = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay)
        .init();

    let t0 = std::time::Instant::now();
    let mut running = 0.0f32;
    let mut counted = 0usize;

    for step in 0..cfg.steps {
        let k = if step < cfg.rollout_after {
            1
        } else {
            cfg.rollout
        };
        let batch = gen.next_batch(k);
        let b = batch.batch;
        let dims = [b, 3, batch.ny, batch.nx];

        let to_field = |v: &[f32]| {
            Tensor::<B, 1>::from_floats(v, device).reshape(dims)
        };
        let meta = Tensor::<B, 1>::from_floats(batch.meta.as_slice(), device)
            .reshape([b, crate::data::META_DIM]);

        let mut x = to_field(&batch.states[0]);
        let mut loss = Tensor::<B, 1>::zeros([1], device);
        for step_k in 0..k {
            x = model.step(x, meta.clone(), mask.clone());
            let target = to_field(&batch.states[step_k + 1]);
            loss = loss + (x.clone() - target).powi_scalar(2).mean();
        }
        let loss = loss / k as f32;

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let lr = if step < cfg.warmup {
            cfg.lr * (step + 1) as f64 / cfg.warmup as f64
        } else {
            cfg.lr
        };
        model = opt.step(lr, model, grads);

        // Reading the loss forces a sync, so only do it on log rows.
        if (step + 1) % cfg.log_every == 0 || step == 0 {
            let l: f32 = loss.into_scalar().elem();
            running += l;
            counted += 1;
            on_log(&LogRow {
                step: step + 1,
                loss: l,
                secs: t0.elapsed().as_secs_f64(),
            });
        }
    }
    let _ = (running, counted);
    model
}

/// Drop the autodiff wrapper for inference.
pub fn to_inference<B: AutodiffBackend>(model: &Jacobi2000<B>) -> Jacobi2000<B::InnerBackend> {
    model.valid()
}
