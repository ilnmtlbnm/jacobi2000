//! Rung-1 eval metrics (GOAL-2026-09-18 sec. 1): the numbers global RMSE
//! could not give. Eval only — none of this touches the training loss.
//!
//! Everything here is measured on a single-drop trajectory before the wave
//! first meets a wall, so the model is judged on the physics it was asked to
//! learn and not on the reflection pattern it never saw enough of.
//!
//! Correlation and the RMSEs are in normalized field units (h, u, v are
//! comparable there). Amplitude, the active-region mask and the wavefront
//! radius are computed on the decoded surface anomaly `h - depth`, which is
//! the field the drop actually lives in.

use burn::prelude::*;
use jacobi2000::{project_mass, Jacobi2000, Mask};
use rand::rngs::StdRng;
use rand::Rng;
use solvers::{Params, ShallowWater, State};

use crate::data::{meta_of, Normalizer, META_DIM};

pub const BANDS: usize = 3;

/// The drop a single-drop trajectory was built from. The centre is what the
/// wavefront radius is measured about and what sets the reflection horizon.
#[derive(Debug, Clone, Copy)]
pub struct Drop {
    pub cx: f32,
    pub cy: f32,
    pub amp: f32,
    pub sigma: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Metrics {
    /// Pearson correlation over (h, u, v).
    pub corr: f32,
    /// Pearson correlation over h alone.
    pub corr_h: f32,
    /// RMSE restricted to the cells where `|solver h anomaly| > 0.1 * peak`.
    pub active_rmse: f32,
    /// Model peak `|h|` over solver peak `|h|`. 1 is right, < 1 is the
    /// "average everywhere" failure.
    pub amplitude_ratio: f32,
    /// Model minus solver radius of the azimuthally-averaged `|h|` peak, in
    /// cells. Positive = the model's ring runs ahead.
    pub radius_error: f32,
    /// Lag (in cells) that best aligns the model's radial `|h|` profile to
    /// the solver's, by cross-correlation over ±8 cells with sub-cell
    /// parabolic refinement. Positive = the model's ring runs ahead. Unlike
    /// `radius_error` (a single argmax bin each side), this uses the whole
    /// profile shape and is not thrown by a bistable (two-peaked) profile.
    pub xcorr_lag: f32,
    /// RMSE of the h error per Laplacian-pyramid band, shortest wavelength
    /// first (roughly 2-4, 4-8, 8-16 cells).
    pub bands: [f32; BANDS],
    /// The number that lied: RMSE over the whole field.
    pub global_rmse: f32,
}

pub fn rmse(a: &[f32], b: &[f32]) -> f32 {
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

/// Pearson correlation. A constant field (the frozen baseline at the cells
/// that never moved, or a dead model) has no variance to correlate, and
/// scores 0 rather than a division by zero.
pub fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&x| x as f64).sum::<f64>() / n,
        b.iter().map(|&x| x as f64).sum::<f64>() / n,
    );
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    (sab / (saa * sbb).sqrt()) as f32
}

/// Cells where the solver field carries the drop: `|x| > 0.1 * peak`.
pub fn active_mask(truth: &[f32]) -> Vec<bool> {
    let peak = peak(truth);
    truth.iter().map(|x| x.abs() > 0.1 * peak).collect()
}

/// RMSE over the masked cells only. `pred` and `truth` may hold several
/// channels stacked on the same `mask.len()` grid.
pub fn masked_rmse(pred: &[f32], truth: &[f32], mask: &[bool]) -> f32 {
    let n = mask.len();
    let mut s = 0.0f64;
    let mut count = 0usize;
    for (i, (&p, &t)) in pred.iter().zip(truth).enumerate() {
        if mask[i % n] {
            let d = (p - t) as f64;
            s += d * d;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    (s / count as f64).sqrt() as f32
}

pub fn peak(f: &[f32]) -> f32 {
    f.iter().fold(0.0f32, |a, &x| a.max(x.abs()))
}

/// Azimuthally-averaged `|f|` profile about `(cx, cy)`, one value per
/// one-cell-wide radius bin. Bins are cut off at the nearest wall so every
/// annulus is fully sampled.
pub fn radial_profile(f: &[f32], nx: usize, ny: usize, cx: f32, cy: f32) -> Vec<f64> {
    let rmax = cx
        .min(cy)
        .min(nx as f32 - cx)
        .min(ny as f32 - cy)
        .floor()
        .max(1.0) as usize;
    let mut sum = vec![0.0f64; rmax + 1];
    let mut count = vec![0u32; rmax + 1];
    for y in 0..ny {
        for x in 0..nx {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let r = (dx * dx + dy * dy).sqrt();
            let bin = (r + 0.5) as usize;
            if bin <= rmax {
                sum[bin] += f[y * nx + x].abs() as f64;
                count[bin] += 1;
            }
        }
    }
    sum.iter()
        .zip(&count)
        .map(|(s, &c)| if c == 0 { 0.0 } else { s / c as f64 })
        .collect()
}

/// Radius (in cells) of a radial profile's peak, refined to sub-cell
/// accuracy by a parabola through the peak bin and its neighbours.
pub fn profile_peak(prof: &[f64]) -> f32 {
    let k = (0..prof.len()).fold(0, |b, i| if prof[i] > prof[b] { i } else { b });
    if k == 0 || k + 1 >= prof.len() {
        return k as f32;
    }
    let (l, c, r) = (prof[k - 1], prof[k], prof[k + 1]);
    let den = l - 2.0 * c + r;
    if den >= 0.0 {
        return k as f32;
    }
    (k as f64 + 0.5 * (l - r) / den) as f32
}

/// Radius (in cells) of the peak of the azimuthally-averaged `|f|` profile
/// about `(cx, cy)`. Bistable at 50 ticks when the profile carries two
/// comparable peaks (docs/runs/2026-09-18-radius-estimator.md): a tie
/// between bins on either side of the true wavefront flips this by one
/// argmax step. See `xcorr_lag` for a whole-profile alternative.
pub fn wavefront_radius(f: &[f32], nx: usize, ny: usize, cx: f32, cy: f32) -> f32 {
    profile_peak(&radial_profile(f, nx, ny, cx, cy))
}

/// Correlation of `model` against `solver` shifted by `lag` cells: position
/// `i` of `model` is compared to position `i - lag` of `solver`, over the
/// overlapping range. A profile shifted right by `k` (its features appear at
/// `k` cells greater radius than the reference) reports its best lag at
/// `+k`.
fn corr_at_lag(model: &[f64], solver: &[f64], lag: isize) -> f32 {
    let n = model.len() as isize;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for i in 0..n {
        let si = i - lag;
        if si >= 0 && si < n {
            a.push(model[i as usize] as f32);
            b.push(solver[si as usize] as f32);
        }
    }
    if a.len() < 3 {
        return f32::NEG_INFINITY;
    }
    correlation(&a, &b)
}

/// Lag (in cells) that maximizes the cross-correlation of the model's radial
/// profile against the solver's, searched over `±max_lag` cells and refined
/// to sub-cell accuracy by a parabola through the best lag and its
/// neighbours. Unlike a single argmax bin, this uses the whole profile shape
/// and survives a bistable (two-peaked) profile.
pub fn xcorr_lag(model: &[f64], solver: &[f64], max_lag: usize) -> f32 {
    let max_lag = max_lag as isize;
    let (mut best_lag, mut best_c) = (0isize, f32::NEG_INFINITY);
    for lag in -max_lag..=max_lag {
        let c = corr_at_lag(model, solver, lag);
        if c > best_c {
            best_c = c;
            best_lag = lag;
        }
    }
    if best_lag > -max_lag && best_lag < max_lag {
        let (l, c, r) = (
            corr_at_lag(model, solver, best_lag - 1),
            best_c,
            corr_at_lag(model, solver, best_lag + 1),
        );
        let den = l - 2.0 * c + r;
        if den < 0.0 {
            return best_lag as f32 + 0.5 * (l - r) / den;
        }
    }
    best_lag as f32
}

/// Separable `[1, 2, 1] / 4` smoothing with edge replication.
fn blur(f: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let at = |v: &[f32], x: isize, y: isize| {
        v[(y.clamp(0, ny as isize - 1) as usize) * nx + x.clamp(0, nx as isize - 1) as usize]
    };
    let mut tmp = vec![0.0f32; f.len()];
    for y in 0..ny as isize {
        for x in 0..nx as isize {
            tmp[y as usize * nx + x as usize] =
                0.25 * (at(f, x - 1, y) + 2.0 * at(f, x, y) + at(f, x + 1, y));
        }
    }
    let mut out = vec![0.0f32; f.len()];
    for y in 0..ny as isize {
        for x in 0..nx as isize {
            out[y as usize * nx + x as usize] =
                0.25 * (at(&tmp, x, y - 1) + 2.0 * at(&tmp, x, y) + at(&tmp, x, y + 1));
        }
    }
    out
}

/// One level down the Gaussian pyramid: smooth, then decimate by two.
fn downsample(f: &[f32], nx: usize, ny: usize) -> (Vec<f32>, usize, usize) {
    let (hx, hy) = (nx.div_ceil(2), ny.div_ceil(2));
    let g = blur(f, nx, ny);
    let mut out = vec![0.0f32; hx * hy];
    for y in 0..hy {
        for x in 0..hx {
            out[y * hx + x] = g[2 * y * nx + 2 * x];
        }
    }
    (out, hx, hy)
}

/// Back up a level: replicate, then smooth (bilinear interpolation).
fn upsample(f: &[f32], hx: usize, hy: usize, nx: usize, ny: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            out[y * nx + x] = f[(y / 2).min(hy - 1) * hx + (x / 2).min(hx - 1)];
        }
    }
    blur(&out, nx, ny)
}

/// RMSE of the error field per Laplacian-pyramid band, shortest wavelength
/// first. Each band holds what one pyramid level keeps and the next one
/// cannot represent, so a single wavenumber lands in the band that owns its
/// wavelength (roughly 2-4, 4-8, 8-16 cells). The decimation is what makes
/// the bands band-pass; the stride-1 blur used by the training term in
/// `spectral.rs` gives every band the same shape, which is fine for a loss
/// weighting and useless for reporting where the error sits.
pub fn band_errors(pred: &[f32], truth: &[f32], nx: usize, ny: usize) -> [f32; BANDS] {
    let mut cur: Vec<f32> = pred.iter().zip(truth).map(|(p, t)| p - t).collect();
    let (mut cx, mut cy) = (nx, ny);
    let mut out = [0.0f32; BANDS];
    for b in out.iter_mut() {
        let (lo, hx, hy) = downsample(&cur, cx, cy);
        let up = upsample(&lo, hx, hy, cx, cy);
        let zero = vec![0.0f32; cur.len()];
        let band: Vec<f32> = cur.iter().zip(&up).map(|(a, c)| a - c).collect();
        *b = rmse(&band, &zero);
        cur = lo;
        cx = hx;
        cy = hy;
    }
    out
}

/// Score one predicted state against the solver's, both packed normalized
/// `[3, ny, nx]`, for a drop centred at `(cx, cy)`.
/// Decode the packed `h` channel (the first `n` of `x`) back to physical
/// units.
pub fn decode_h(x: &[f32], norm: &Normalizer, n: usize) -> Vec<f32> {
    x[..n].iter().map(|v| v * norm.std[0] + norm.mean[0]).collect()
}

pub fn score(
    pred: &[f32],
    truth: &[f32],
    norm: &Normalizer,
    p: &Params,
    cx: f32,
    cy: f32,
) -> Metrics {
    let n = p.cells();
    let (ph, th) = (decode_h(pred, norm, n), decode_h(truth, norm, n));
    let mask = active_mask(&th);
    let (mprof, tprof) = (
        radial_profile(&ph, p.nx, p.ny, cx, cy),
        radial_profile(&th, p.nx, p.ny, cx, cy),
    );
    Metrics {
        corr: correlation(pred, truth),
        corr_h: correlation(&pred[..n], &truth[..n]),
        active_rmse: masked_rmse(pred, truth, &mask),
        amplitude_ratio: peak(&ph) / peak(&th),
        radius_error: profile_peak(&mprof) - profile_peak(&tprof),
        xcorr_lag: xcorr_lag(&mprof, &tprof, 8),
        bands: band_errors(&pred[..n], &truth[..n], p.nx, p.ny),
        global_rmse: rmse(pred, truth),
    }
}

/// A fresh single-drop initial condition: one drop, random position,
/// amplitude and width, no velocity field. The drop is placed away from the
/// walls; whether it is far enough for the horizon asked for is the caller's
/// check (`Params::horizon_ticks`).
pub fn sample_single_drop(p: &Params, rng: &mut StdRng) -> (State, Drop) {
    let drop = Drop {
        cx: rng.gen_range(8.0..(p.nx as f32 - 8.0)),
        cy: rng.gen_range(8.0..(p.ny as f32 - 8.0)),
        amp: rng.gen_range(0.03..0.07),
        sigma: rng.gen_range(3.0..5.5),
    };
    let mut s = State::flat(p);
    s.add_drop(p, drop.cx, drop.cy, drop.amp, drop.sigma);
    (s, drop)
}

/// Task 8b stress-set categories: single high-amplitude drops, stacked
/// drops at one spot, and drops in a corner (near two walls at once).
#[derive(Debug, Clone, Copy)]
pub enum StressKind {
    HighAmp,
    Stacked,
    Corner,
}

/// A stress-set initial condition. Unlike `sample_single_drop`, the caller
/// does not filter these by `Params::horizon_ticks`: high-amplitude and
/// stacked drops are placed away from the walls same as the standard set,
/// but corner drops are placed *in* a corner on purpose, so the
/// pre-reflection horizon is meaningless for them and every one is scored.
pub fn sample_stress_drop(kind: StressKind, p: &Params, rng: &mut StdRng) -> (State, Drop) {
    match kind {
        StressKind::HighAmp => {
            let drop = Drop {
                cx: rng.gen_range(8.0..(p.nx as f32 - 8.0)),
                cy: rng.gen_range(8.0..(p.ny as f32 - 8.0)),
                amp: rng.gen_range(0.15..0.3) * if rng.gen_bool(0.5) { -1.0 } else { 1.0 },
                sigma: rng.gen_range(3.0..5.5),
            };
            let mut s = State::flat(p);
            s.add_drop(p, drop.cx, drop.cy, drop.amp, drop.sigma);
            (s, drop)
        }
        StressKind::Stacked => {
            let cx = rng.gen_range(8.0..(p.nx as f32 - 8.0));
            let cy = rng.gen_range(8.0..(p.ny as f32 - 8.0));
            let n = rng.gen_range(2..=3);
            let mut s = State::flat(p);
            let (mut amp_total, mut sigma_last) = (0.0f32, 4.0f32);
            for _ in 0..n {
                let amp = rng.gen_range(0.05..0.1) * if rng.gen_bool(0.2) { -1.0 } else { 1.0 };
                let sigma = rng.gen_range(3.0..5.5);
                s.add_drop(p, cx, cy, amp, sigma);
                amp_total += amp;
                sigma_last = sigma;
            }
            (s, Drop { cx, cy, amp: amp_total, sigma: sigma_last })
        }
        StressKind::Corner => {
            let cx = if rng.gen_bool(0.5) {
                rng.gen_range(2.0..6.0)
            } else {
                p.nx as f32 - rng.gen_range(2.0..6.0)
            };
            let cy = if rng.gen_bool(0.5) {
                rng.gen_range(2.0..6.0)
            } else {
                p.ny as f32 - rng.gen_range(2.0..6.0)
            };
            let drop = Drop {
                cx,
                cy,
                amp: rng.gen_range(0.03..0.07) * if rng.gen_bool(0.25) { -1.0 } else { 1.0 },
                sigma: rng.gen_range(3.0..5.5),
            };
            let mut s = State::flat(p);
            s.add_drop(p, drop.cx, drop.cy, drop.amp, drop.sigma);
            (s, drop)
        }
    }
}

/// Roll the model and the solver forward from the same single-drop state and
/// score both the model and the frozen initial field at every horizon.
/// `horizons` must be sorted and must all be below the reflection horizon.
#[allow(clippy::too_many_arguments)]
pub fn score_rollout<B: Backend>(
    model: &Jacobi2000<B>,
    params: &Params,
    norm: &Normalizer,
    init: &State,
    drop: &Drop,
    horizons: &[usize],
    mask: Mask,
    project: bool,
    device: &B::Device,
) -> (Vec<Metrics>, Vec<Metrics>) {
    let n = params.cells();
    let grid = model.grid;
    let mask = mask.to_tensor::<B>(&grid, device);
    let meta =
        Tensor::<B, 1>::from_floats(meta_of(params).as_slice(), device).reshape([1, META_DIM]);

    let mut truth = init.clone();
    let mut solver = ShallowWater::new(*params);

    let mut packed = vec![0.0f32; 3 * n];
    norm.encode(init, &mut packed);
    let frozen = packed.clone();
    let mut x = Tensor::<B, 1>::from_floats(packed.as_slice(), device)
        .reshape([1, 3, params.ny, params.nx]);
    let mass0 = jacobi2000::mass(&x);

    let (mut model_rows, mut frozen_rows) = (Vec::new(), Vec::new());
    let last = *horizons.last().unwrap();
    for t in 1..=last {
        solver.step(&mut truth);
        x = model.step(x, meta.clone(), mask.clone());
        if project {
            x = project_mass(x, mass0.clone());
        }
        if horizons.contains(&t) {
            let mut target = vec![0.0f32; 3 * n];
            norm.encode(&truth, &mut target);
            let pred = x.to_data();
            let pred = pred.as_slice::<f32>().unwrap();
            model_rows.push(score(pred, &target, norm, params, drop.cx, drop.cy));
            frozen_rows.push(score(&frozen, &target, norm, params, drop.cx, drop.cy));
        }
    }
    (model_rows, frozen_rows)
}

/// Like `score_rollout`, but returns the raw azimuthally-averaged `|h|`
/// radial profiles (model, solver) at each horizon instead of scalar
/// metrics. For the docs/runs/2026-09-18-radius-estimator.md diagnostic:
/// printing the profile makes a bistable (two-peaked) argmax visible in a
/// way a single radius number cannot.
#[allow(clippy::too_many_arguments)]
pub fn rollout_profiles<B: Backend>(
    model: &Jacobi2000<B>,
    params: &Params,
    norm: &Normalizer,
    init: &State,
    drop: &Drop,
    horizons: &[usize],
    mask: Mask,
    project: bool,
    device: &B::Device,
) -> Vec<(usize, Vec<f64>, Vec<f64>)> {
    let n = params.cells();
    let grid = model.grid;
    let mask = mask.to_tensor::<B>(&grid, device);
    let meta =
        Tensor::<B, 1>::from_floats(meta_of(params).as_slice(), device).reshape([1, META_DIM]);

    let mut truth = init.clone();
    let mut solver = ShallowWater::new(*params);

    let mut packed = vec![0.0f32; 3 * n];
    norm.encode(init, &mut packed);
    let mut x = Tensor::<B, 1>::from_floats(packed.as_slice(), device)
        .reshape([1, 3, params.ny, params.nx]);
    let mass0 = jacobi2000::mass(&x);

    let mut out = Vec::new();
    let last = *horizons.last().unwrap();
    for t in 1..=last {
        solver.step(&mut truth);
        x = model.step(x, meta.clone(), mask.clone());
        if project {
            x = project_mass(x, mass0.clone());
        }
        if horizons.contains(&t) {
            let mut target = vec![0.0f32; 3 * n];
            norm.encode(&truth, &mut target);
            let pred = x.to_data();
            let pred = pred.as_slice::<f32>().unwrap();
            let (ph, th) = (decode_h(pred, norm, n), decode_h(&target, norm, n));
            let mprof = radial_profile(&ph, params.nx, params.ny, drop.cx, drop.cy);
            let tprof = radial_profile(&th, params.nx, params.ny, drop.cx, drop.cy);
            out.push((t, mprof, tprof));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(nx: usize, ny: usize, f: impl Fn(f32, f32) -> f32) -> Vec<f32> {
        (0..nx * ny)
            .map(|i| f((i % nx) as f32, (i / nx) as f32))
            .collect()
    }

    #[test]
    fn correlation_is_plus_one_minus_one_and_zero() {
        let f = field(32, 32, |x, y| (x * 0.3).sin() * (y * 0.2).cos());
        let neg: Vec<f32> = f.iter().map(|v| -v).collect();
        let flat = vec![0.7f32; f.len()];
        assert!((correlation(&f, &f) - 1.0).abs() < 1e-6);
        assert!((correlation(&f, &neg) + 1.0).abs() < 1e-6);
        assert!(correlation(&f, &flat).abs() < 1e-6);
    }

    #[test]
    fn amplitude_ratio_of_a_doubled_field_is_two() {
        let p = Params::default();
        let norm = Normalizer::identity(p.depth);
        let truth = field(p.nx, p.ny, |x, y| {
            0.05 * (-((x - 30.0).powi(2) + (y - 30.0).powi(2)) / 18.0).exp()
        });
        let pred: Vec<f32> = truth.iter().map(|v| 2.0 * v).collect();
        let pack = |h: &[f32]| {
            let mut v = vec![0.0f32; 3 * p.cells()];
            v[..p.cells()].copy_from_slice(h);
            v
        };
        let m = score(&pack(&pred), &pack(&truth), &norm, &p, 30.0, 30.0);
        assert!((m.amplitude_ratio - 2.0).abs() < 1e-5, "{}", m.amplitude_ratio);
    }

    #[test]
    fn active_region_mask_ignores_error_outside_the_drop() {
        // Noise everywhere the drop is not: global RMSE sees it, the
        // active-region RMSE must not. That difference is the whole point.
        let (nx, ny) = (64, 64);
        let truth = field(nx, ny, |x, y| {
            0.05 * (-((x - 20.0).powi(2) + (y - 40.0).powi(2)) / 18.0).exp()
        });
        let mask = active_mask(&truth);
        let pred: Vec<f32> = truth
            .iter()
            .zip(&mask)
            .map(|(&t, &m)| if m { t } else { t + 1e-3 })
            .collect();
        assert!(masked_rmse(&pred, &truth, &mask) < 1e-9);
        assert!(rmse(&pred, &truth) > 1e-4);
    }

    #[test]
    fn wavefront_radius_recovers_an_analytic_ring() {
        let (nx, ny) = (96, 96);
        let (cx, cy) = (48.0f32, 48.0f32);
        for r0 in [8.0f32, 16.0, 24.0] {
            let f = field(nx, ny, |x, y| {
                let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                (-(r - r0).powi(2) / 8.0).exp()
            });
            let got = wavefront_radius(&f, nx, ny, cx, cy);
            assert!((got - r0).abs() < 0.25, "r0 {r0} recovered as {got}");
        }
    }

    #[test]
    fn xcorr_lag_recovers_a_three_cell_shift() {
        let n = 64;
        let ring = |r0: f64| -> Vec<f64> {
            (0..n)
                .map(|i| (-(i as f64 - r0).powi(2) / 8.0).exp())
                .collect()
        };
        let solver = ring(20.0);
        let model = ring(23.0);
        let lag = xcorr_lag(&model, &solver, 8);
        assert!((lag - 3.0).abs() < 0.1, "lag {lag}");
    }

    #[test]
    fn a_single_wavenumber_lands_in_one_band() {
        // Bands cover wavelengths of roughly 2-4, 4-8 and 8-16 cells. A
        // pyramid's bands overlap at their edges, so the assertion is that
        // the band owning the wavelength carries the error, not that the
        // neighbours are exactly zero.
        let (nx, ny) = (64, 64);
        let zero = vec![0.0f32; nx * ny];
        for (band, lambda) in [(0usize, 3.0f32), (1, 8.0), (2, 16.0)] {
            let f = field(nx, ny, |x, _| {
                (2.0 * std::f32::consts::PI * x / lambda).sin()
            });
            let e = band_errors(&f, &zero, nx, ny);
            for (i, v) in e.iter().enumerate() {
                if i != band {
                    assert!(*v < e[band], "lambda {lambda}: band {i} = {v} vs reference {}", e[band]);
                }
            }
        }
    }
}
