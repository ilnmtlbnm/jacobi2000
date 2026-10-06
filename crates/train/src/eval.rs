//! Rollout error against the solver, with the frozen-field baseline.
//!
//! All errors are RMSE in normalized field units over (h, u, v), so the three
//! channels are comparable and the numbers are dimensionless.

use burn::prelude::*;
use jacobi2000::{
    energy, excess_energy, project_energy, project_excess_energy, project_mass, Denorm, Jacobi2000,
    Mask,
};
use solvers::{Params, ShallowWater, State};

use crate::data::{meta_of, Normalizer, META_DIM};

#[derive(Debug, Clone)]
pub struct RolloutErrors {
    pub ticks: Vec<usize>,
    pub model: Vec<f32>,
    /// Frozen initial field, the baseline the model has to beat.
    pub persistence: Vec<f32>,
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

/// Roll `model` and the solver forward from the same initial state and
/// compare at each checkpoint in `ticks` (which must be sorted). With
/// `project`, each predicted state is put back on the constant-mass
/// constraint before it becomes the next input; with `project_energy`, also
/// onto the energy inequality (task 11).
#[allow(clippy::too_many_arguments)]
pub fn rollout_errors<B: Backend>(
    model: &Jacobi2000<B>,
    params: &Params,
    norm: &Normalizer,
    init: &State,
    ticks: &[usize],
    mask: Mask,
    project: bool,
    project_energy_flag: bool,
    r_max: f32,
    excess: Option<f32>,
    device: &B::Device,
) -> RolloutErrors {
    let n = params.cells();
    let grid = model.grid;
    let mask = mask.to_tensor::<B>(&grid, device);
    let meta = Tensor::<B, 1>::from_floats(meta_of(params).as_slice(), device)
        .reshape([1, META_DIM]);
    let denorm = Denorm {
        mean: norm.mean,
        std: norm.std,
        depth: norm.depth,
    };

    let dt_dx = params.dt / params.dx;
    let mut truth = init.clone();
    let mut solver = ShallowWater::new(*params);

    let mut packed = vec![0.0f32; 3 * n];
    norm.encode(init, &mut packed);
    let frozen = packed.clone();
    let mut x = Tensor::<B, 1>::from_floats(packed.as_slice(), device)
        .reshape([1, 3, params.ny, params.nx]);
    let mass0 = jacobi2000::mass(&x);

    let mut out = RolloutErrors {
        ticks: ticks.to_vec(),
        model: Vec::new(),
        persistence: Vec::new(),
    };

    let last = *ticks.last().unwrap();
    for t in 1..=last {
        solver.step(&mut truth);
        let prev_energy = project_energy_flag.then(|| energy(&x, &denorm, params.g));
        let prev_excess = excess.map(|_| excess_energy(&x, &denorm, params.g, dt_dx));
        x = model.step(x, meta.clone(), mask.clone());
        if project {
            x = project_mass(x, mass0.clone());
        }
        if let Some(prev_energy) = prev_energy {
            x = project_energy(x, prev_energy, &denorm, params.g, r_max);
        }
        if let (Some(r), Some(prev_excess)) = (excess, prev_excess) {
            x = project_excess_energy(x, prev_excess, &denorm, params.g, dt_dx, r);
        }
        if ticks.contains(&t) {
            let mut target = vec![0.0f32; 3 * n];
            norm.encode(&truth, &mut target);
            let pred = x.to_data();
            let pred = pred.as_slice::<f32>().unwrap();
            out.model.push(rmse(pred, &target));
            out.persistence.push(rmse(&frozen, &target));
        }
    }
    out
}
