//! Step mode with a rollout loss. Generic over the backend: the binary crate
//! picks wgpu (native training) or ndarray (CPU).

use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use jacobi2000::{energy, project_energy, project_mass, Denorm, Jacobi2000, Mask};

use crate::data::{Batch, Generator, Normalizer};
use crate::lagrangian::LagrangianGenerator;

/// Trait over the two batch sources (`Generator`, `LagrangianGenerator`) so
/// `fit` runs unchanged on either. `tick_params` is only read when
/// `FitConfig::project_energy` is on, which the CLI keeps off for
/// `--lagrangian` (see the field's doc there): the constants it carries
/// (`g`, `nu`, `dt`) matter only in that path.
pub trait BatchSource {
    fn next_batch(&mut self, rollout: usize) -> Batch;
    fn norm(&self) -> Normalizer;
    fn tick_params(&self) -> solvers::Params;
}

impl BatchSource for Generator {
    fn next_batch(&mut self, rollout: usize) -> Batch {
        Generator::next_batch(self, rollout)
    }
    fn norm(&self) -> Normalizer {
        self.norm
    }
    fn tick_params(&self) -> solvers::Params {
        self.config().params
    }
}

impl BatchSource for LagrangianGenerator {
    fn next_batch(&mut self, rollout: usize) -> Batch {
        LagrangianGenerator::next_batch(self, rollout)
    }
    fn norm(&self) -> Normalizer {
        self.norm
    }
    fn tick_params(&self) -> solvers::Params {
        let cfg = self.config();
        let sph = cfg.sph;
        solvers::Params {
            nx: sph.nx,
            ny: sph.ny,
            dx: cfg.grid_dx,
            dt: crate::lagrangian::SPH_STEPS_PER_TICK as f32 * sph.dt,
            g: sph.gravity,
            depth: 0.0,
            nu: sph.alpha,
        }
    }
}

/// Constant for the fallback energy-ratio formula below. Superseded as the
/// default by `r_max` measured directly from the solver's own trajectories
/// (`energy-stats`, docs/runs/2026-09-19-energy-stats.md): the formula
/// `1 - c*nu*dt` with `c = 1` turned out far tighter than the solver's real
/// per-tick dissipation and stripped kinetic energy every step
/// (docs/runs/2026-09-19-energy-smoke-2.md).
pub const ENERGY_C: f32 = 1.0;

/// Documented fallback for the per-tick energy ratio bound `r_max` when no
/// measured value is given (`train --r-max`): `1 - c*nu*dt` (CONCEPT sec. 4,
/// step 2), `nu == 0` collapsing it to 1 (no decay allowed).
pub fn fallback_r_max(nu: f32, dt: f32) -> f32 {
    1.0 - ENERGY_C * nu * dt
}

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
    /// Gaussian noise added to the input state at every rollout step, in
    /// normalized units. Run 2026-09-18-1 was stable single-step and blew up
    /// by 50 ticks: the model never saw its own error distribution. Noise
    /// injection makes it contractive on off-manifold states.
    pub noise: f32,
    /// Put every predicted state back on the constant-mass constraint before
    /// it becomes the next rollout input, so the model is trained on the
    /// states it will actually be run on (CONCEPT sec. 4: propose, project).
    pub project: bool,
    /// Also project onto the energy inequality (task 11), scaling (u, v)
    /// down when the predicted step's energy exceeds the viscosity-scaled
    /// bound. Off by default: a second hard constraint on top of mass.
    pub project_energy: bool,
    /// Per-tick energy ratio bound (`E_new <= E_prev * r_max`) used by the
    /// energy projection above, measured from the solver's own trajectories
    /// rather than assumed (see `fallback_r_max` for the coded-formula
    /// fallback). Unused when `project_energy` is off.
    pub r_max: f32,
    /// Weight of the wavenumber-banded error term (`spectral::band_mse`) added
    /// to plain MSE. 0 disables it.
    pub spectral: f32,
    pub log_every: usize,
    /// Final learning rate as a fraction of `lr`: cosine decay after warmup.
    pub lr_floor: f64,
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
            noise: 0.0,
            project: true,
            project_energy: false,
            r_max: 1.0,
            spectral: 0.0,
            log_every: 100,
            lr_floor: 0.05,
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
    gen: &mut dyn BatchSource,
    cfg: &FitConfig,
    device: &B::Device,
    mut on_log: impl FnMut(&LogRow),
) -> Jacobi2000<B> {
    let grid = model.grid;
    let mask = cfg.mask.to_tensor::<B>(&grid, device);
    let mut opt = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay)
        .init();
    let params = gen.tick_params();
    let norm = gen.norm();
    let denorm = Denorm {
        mean: norm.mean,
        std: norm.std,
        depth: norm.depth,
    };

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
        let mass0 = jacobi2000::mass(&x);
        let mut loss = Tensor::<B, 1>::zeros([1], device);
        for step_k in 0..k {
            if cfg.noise > 0.0 {
                x = x.clone()
                    + Tensor::<B, 4>::random(
                        dims,
                        burn::tensor::Distribution::Normal(0.0, cfg.noise as f64),
                        device,
                    );
            }
            let prev_energy = cfg
                .project_energy
                .then(|| energy(&x, &denorm, params.g));
            x = model.step(x, meta.clone(), mask.clone());
            if cfg.project {
                x = project_mass(x, mass0.clone());
            }
            if let Some(prev_energy) = prev_energy {
                x = project_energy(x, prev_energy, &denorm, params.g, cfg.r_max);
            }
            let target = to_field(&batch.states[step_k + 1]);
            let err = x.clone() - target;
            loss = loss + err.clone().powi_scalar(2).mean();
            if cfg.spectral > 0.0 {
                loss = loss + crate::spectral::band_mse(err) * cfg.spectral;
            }
        }
        let loss = loss / k as f32;

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let lr = if step < cfg.warmup {
            cfg.lr * (step + 1) as f64 / cfg.warmup as f64
        } else {
            let t = (step - cfg.warmup) as f64 / (cfg.steps - cfg.warmup).max(1) as f64;
            let cos = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
            cfg.lr * (cfg.lr_floor + (1.0 - cfg.lr_floor) * cos)
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use solvers::{Params, ShallowWater, State};

    type B = NdArray;

    /// Task A2: a bound measured (with margin) from the solver's own
    /// trajectory must never be exceeded by that same trajectory, i.e. the
    /// projection is the identity on solver states over the 100 ticks it was
    /// measured on. Guards the earlier, formula-based bound
    /// (`fallback_r_max`), which was tight enough to strip real kinetic
    /// energy from the solver's own states
    /// (docs/runs/2026-09-19-energy-smoke-2.md).
    #[test]
    fn measured_r_max_is_identity_on_the_solvers_own_trajectory() {
        let device = Default::default();
        let params = Params { nu: 1e-3, ..Params::default() };
        let denorm = Denorm { mean: [0.0; 3], std: [1.0; 3], depth: params.depth };
        let dims = [1, 3, params.ny, params.nx];

        let encode = |s: &State, device: &<B as Backend>::Device| {
            let mut v = Vec::with_capacity(3 * params.cells());
            v.extend_from_slice(&s.h);
            v.extend_from_slice(&s.u);
            v.extend_from_slice(&s.v);
            Tensor::<B, 1>::from_floats(v.as_slice(), device).reshape(dims)
        };

        let mut solver = ShallowWater::new(params);
        let mut state = State::flat(&params);
        state.add_drop(&params, 24.0, 40.0, 0.08, 4.0);

        let mut prev = Vec::new();
        let mut r_max = 0.0f32;
        for _ in 0..100 {
            let x0 = encode(&state, &device);
            let e0: f32 = energy(&x0, &denorm, params.g).into_scalar();
            solver.step(&mut state);
            let x1 = encode(&state, &device);
            let e1: f32 = energy(&x1, &denorm, params.g).into_scalar();
            r_max = r_max.max(e1 / e0);
            prev.push((e0, x1));
        }
        let r_max = r_max * (1.0 + 1e-4);

        for (e0, x1) in prev {
            let prev_energy = Tensor::<B, 2>::from_floats([[e0]], &device);
            let projected = project_energy(x1.clone(), prev_energy, &denorm, params.g, r_max);
            projected
                .into_data()
                .assert_approx_eq::<f32>(&x1.into_data(), burn::tensor::Tolerance::default());
        }
    }
}
