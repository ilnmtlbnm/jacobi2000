//! xr ensemble at inference (CONCEPT sec. 4, step 3): "ensemble over
//! discretizations, not over models". Same weights, K jittered patch-grid
//! offsets: shift the field cyclically, run the model, shift the prediction
//! back, average. Per-cell std across members is uncertainty for free, no
//! ensemble of weights needed.

use burn::prelude::*;
use jacobi2000::Jacobi2000;

use crate::metrics::{score, Drop, Metrics};
use crate::data::{meta_of, Normalizer, META_DIM};
use jacobi2000::{project_mass, Mask};
use solvers::{Params, ShallowWater, State};

/// `k` offsets `(dx, dy) ∈ [0, patch)²`, spread roughly evenly over the
/// patch (an `m x m` sub-grid with `m = ceil(sqrt(k))`). `k <= 1` is just the
/// single-offset baseline.
pub fn offsets(k: usize, patch: usize) -> Vec<(usize, usize)> {
    if k <= 1 {
        return vec![(0, 0)];
    }
    let m = (k as f32).sqrt().ceil() as usize;
    let mut out = Vec::with_capacity(k);
    for j in 0..m {
        for i in 0..m {
            if out.len() == k {
                return out;
            }
            out.push(((i * patch) / m, (j * patch) / m));
        }
    }
    out
}

/// Cyclic shift along `dim` (2 = y, 3 = x) by `shift` cells, wrapping.
fn roll<B: Backend>(x: Tensor<B, 4>, dim: usize, shift: isize) -> Tensor<B, 4> {
    let n = x.dims()[dim];
    let shift = shift.rem_euclid(n as isize) as usize;
    if shift == 0 {
        return x;
    }
    Tensor::cat(
        vec![
            x.clone().narrow(dim, n - shift, shift),
            x.narrow(dim, 0, n - shift),
        ],
        dim,
    )
}

/// One ensemble member: shift `state` by `(dx, dy)`, run one model step,
/// shift the prediction back so every member lands on the same grid.
fn member_step<B: Backend>(
    model: &Jacobi2000<B>,
    state: Tensor<B, 4>,
    meta: Tensor<B, 2>,
    mask: Option<Tensor<B, 2>>,
    dx: usize,
    dy: usize,
) -> Tensor<B, 4> {
    let shifted = roll(roll(state, 3, dx as isize), 2, dy as isize);
    let out = model.step(shifted, meta, mask);
    roll(roll(out, 3, -(dx as isize)), 2, -(dy as isize))
}

/// Ensemble mean and per-cell spread (population std) over `k` jittered
/// patch-grid offsets, same weights. `k <= 1` degenerates to the plain
/// `model.step` (spread all zero).
pub fn xr_step<B: Backend>(
    model: &Jacobi2000<B>,
    state: Tensor<B, 4>,
    meta: Tensor<B, 2>,
    mask: Option<Tensor<B, 2>>,
    k: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let patch = model.grid.patch;
    let members: Vec<Tensor<B, 4>> = offsets(k, patch)
        .into_iter()
        .map(|(dx, dy)| member_step(model, state.clone(), meta.clone(), mask.clone(), dx, dy))
        .collect();
    let stack = Tensor::stack::<5>(members, 0);
    let mean: Tensor<B, 4> = stack.clone().mean_dim(0).squeeze_dim(0);
    let var: Tensor<B, 4> = (stack - mean.clone().unsqueeze_dim::<5>(0))
        .powi_scalar(2)
        .mean_dim(0)
        .squeeze_dim(0);
    (mean, var.sqrt())
}

/// Roll the xr ensemble and the single-offset baseline forward from the same
/// single-drop state, scoring both against the solver at each horizon. Also
/// returns the mean per-cell spread of the decoded surface height, in
/// physical units, at each horizon (`spread_field` in `crates/web` is the
/// same per-cell quantity, not yet averaged).
#[allow(clippy::too_many_arguments)]
pub fn score_rollout_xr<B: Backend>(
    model: &Jacobi2000<B>,
    params: &Params,
    norm: &Normalizer,
    init: &State,
    drop: &Drop,
    horizons: &[usize],
    mask: Mask,
    project: bool,
    k: usize,
    device: &B::Device,
) -> (Vec<Metrics>, Vec<Metrics>, Vec<f32>) {
    let n = params.cells();
    let grid = model.grid;
    let mask_t = mask.to_tensor::<B>(&grid, device);
    let meta =
        Tensor::<B, 1>::from_floats(meta_of(params).as_slice(), device).reshape([1, META_DIM]);

    let mut truth = init.clone();
    let mut solver = ShallowWater::new(*params);

    let mut packed = vec![0.0f32; 3 * n];
    norm.encode(init, &mut packed);
    let mut x_xr = Tensor::<B, 1>::from_floats(packed.as_slice(), device)
        .reshape([1, 3, params.ny, params.nx]);
    let mut x_base = x_xr.clone();
    let mass0 = jacobi2000::mass(&x_xr);

    let (mut xr_rows, mut base_rows) = (Vec::new(), Vec::new());
    let mut spreads = vec![0.0f32; horizons.len()];
    let last = *horizons.last().unwrap();
    for t in 1..=last {
        solver.step(&mut truth);

        let (mean, spread) = xr_step(model, x_xr, meta.clone(), mask_t.clone(), k);
        x_xr = if project {
            project_mass(mean, mass0.clone())
        } else {
            mean
        };
        x_base = model.step(x_base, meta.clone(), mask_t.clone());
        if project {
            x_base = project_mass(x_base, mass0.clone());
        }

        if let Some(i) = horizons.iter().position(|&h| h == t) {
            let mut target = vec![0.0f32; 3 * n];
            norm.encode(&truth, &mut target);

            let pred_xr = x_xr.to_data();
            let pred_xr = pred_xr.as_slice::<f32>().unwrap();
            let pred_base = x_base.to_data();
            let pred_base = pred_base.as_slice::<f32>().unwrap();
            xr_rows.push(score(pred_xr, &target, norm, params, drop.cx, drop.cy));
            base_rows.push(score(pred_base, &target, norm, params, drop.cx, drop.cy));

            let sp = spread.to_data();
            let sp = sp.as_slice::<f32>().unwrap();
            // Only channel 0 (h) carries a physical unit worth reporting;
            // the affine decode's scale is norm.std[0], mean does not matter
            // for a spread.
            let mean_spread: f64 = sp[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            spreads[i] = mean_spread as f32 * norm.std[0];
        }
    }
    (xr_rows, base_rows, spreads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jacobi2000::Jacobi2000Config;

    type B = burn::backend::NdArray;

    fn tiny_model(device: &<B as Backend>::Device) -> (Jacobi2000<B>, Jacobi2000Config) {
        let cfg = Jacobi2000Config::new()
            .with_nx(16)
            .with_ny(16)
            .with_dim(8)
            .with_layers(1)
            .with_heads(1)
            .with_ff_mult(1)
            .with_head_dim(4);
        (cfg.init::<B>(device), cfg)
    }

    fn nontrivial_state(cfg: &Jacobi2000Config, device: &<B as Backend>::Device) -> Tensor<B, 4> {
        let g = cfg.grid();
        let v: Vec<f32> = (0..g.channels * g.cells())
            .map(|i| {
                let (x, y) = ((i % g.nx) as f32, ((i / g.nx) % g.ny) as f32);
                0.1 * ((x * 0.4).sin() + (y * 0.3).cos())
            })
            .collect();
        Tensor::<B, 1>::from_floats(v.as_slice(), device).reshape([1, g.channels, g.ny, g.nx])
    }

    #[test]
    fn k_one_offset_zero_equals_plain_forward() {
        let device = Default::default();
        let (model, cfg) = tiny_model(&device);
        let state = nontrivial_state(&cfg, &device);
        let meta = Tensor::<B, 2>::zeros([1, cfg.meta_dim], &device);

        let plain = model.step(state.clone(), meta.clone(), None);
        let (mean, spread) = xr_step(&model, state, meta, None, 1);

        mean.into_data()
            .assert_approx_eq::<f32>(&plain.into_data(), burn::tensor::Tolerance::default());
        let s: f32 = spread.abs().max().into_scalar();
        assert_eq!(s, 0.0, "single-member spread must be exactly zero");
    }

    #[test]
    fn k_four_spread_is_nonnegative_and_not_constant() {
        let device = Default::default();
        let (model, cfg) = tiny_model(&device);
        let state = nontrivial_state(&cfg, &device);
        let meta = Tensor::<B, 2>::zeros([1, cfg.meta_dim], &device);

        assert_eq!(offsets(4, cfg.patch).len(), 4);

        let (_, spread) = xr_step(&model, state, meta, None, 4);
        let min: f32 = spread.clone().min().into_scalar();
        let max: f32 = spread.max().into_scalar();
        assert!(min >= 0.0, "spread went negative: {min}");
        assert!(
            max - min > 1e-8,
            "spread is constant across cells: min {min} max {max}"
        );
    }
}
