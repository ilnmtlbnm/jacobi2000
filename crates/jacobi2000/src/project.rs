//! Hard constraints, applied after the model proposes (CONCEPT sec. 4, step 2:
//! propose, then project onto the admissible set).
//!
//! Mass is the first and cheapest of them: the solver conserves the box total
//! of `h` to 1.6e-8, so any drift in the model's total is pure error. The
//! projection that restores it is a single constant shift of `h`.
//!
//! Fields reach the model normalized, `x = (h - depth - mean) / std` per
//! channel. That map is affine with a per-channel constant, so shifting the
//! denormalized `h` by `c` is shifting normalized channel 0 by `c / std[0]`:
//! matching the normalized channel-0 box mean to the reference's is exactly
//! the denormalized mass constraint, with no need to denormalize first.

use burn::prelude::*;

/// The affine per-channel decode a normalized tensor was packed with
/// (`x = (physical - mean - depth) / std` for h, `(physical - mean) / std`
/// for u, v). Shape-for-shape the same as `train::data::Normalizer`, but
/// this crate sits below `train` in the dependency graph, so it is its own
/// tiny type: construct with
/// `Denorm { mean: norm.mean, std: norm.std, depth: norm.depth }`.
#[derive(Debug, Clone, Copy)]
pub struct Denorm {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub depth: f32,
}

/// Potential (`1/2 g h^2`) and kinetic (`1/2 h (u^2+v^2)`) energy per batch
/// member, summed over cells, physical units, `dx = 1` (a uniform cell-area
/// factor on both sides of the inequality this feeds, so it drops out).
fn potential_kinetic<B: Backend>(
    x: &Tensor<B, 4>,
    d: &Denorm,
    g: f32,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let [b, _, ny, nx] = x.dims();
    let n = ny * nx;
    let h = x.clone().slice([0..b, 0..1]) * d.std[0] + (d.mean[0] + d.depth);
    let u = x.clone().slice([0..b, 1..2]) * d.std[1] + d.mean[1];
    let v = x.clone().slice([0..b, 2..3]) * d.std[2] + d.mean[2];
    let potential = (h.clone().powi_scalar(2) * (0.5 * g))
        .reshape([b, n])
        .sum_dim(1)
        .reshape([b, 1]);
    let kinetic = (h * (u.powi_scalar(2) + v.powi_scalar(2)) * 0.5)
        .reshape([b, n])
        .sum_dim(1)
        .reshape([b, 1]);
    (potential, kinetic)
}

/// Total energy per batch member (`[batch, 1]`), physical units: `sum 1/2 g
/// h^2 + 1/2 h (u^2 + v^2)`.
pub fn energy<B: Backend>(x: &Tensor<B, 4>, d: &Denorm, g: f32) -> Tensor<B, 2> {
    let (potential, kinetic) = potential_kinetic(x, d, g);
    potential + kinetic
}

/// Project onto the energy inequality `E_new <= E_prev * r_max`, scaling the
/// velocity field `(u, v)` uniformly per batch member down to the bound when
/// it is exceeded. `r_max` is a per-tick total-energy ratio measured from the
/// solver's own trajectories on the training distribution (`energy-stats`,
/// docs/runs/2026-09-19-energy-stats.md), not assumed from a formula: the
/// earlier coded bound `1 - c*nu*dt` (see `train::fit::fallback_r_max`, kept
/// as a documented fallback) was far tighter than the solver's real per-tick
/// dissipation and stripped kinetic energy every step
/// (docs/runs/2026-09-19-energy-smoke-2.md). Potential energy (`h`) is left
/// untouched: `h` already carries its own hard constraint (`project_mass`),
/// and scaling it here would fight that projection. Never scales up:
/// `E_new <= bound` already is the identity (CONCEPT sec. 4, step 2: propose,
/// then project onto the admissible set; this is the inequality half,
/// `project_mass` is the equality half).
pub fn project_energy<B: Backend>(
    x: Tensor<B, 4>,
    prev_energy: Tensor<B, 2>,
    d: &Denorm,
    g: f32,
    r_max: f32,
) -> Tensor<B, 4> {
    let [b, _, _, _] = x.dims();
    let bound = prev_energy * r_max;
    let (potential, kinetic) = potential_kinetic(&x, d, g);
    // scale^2 * kinetic + potential <= bound. budget = 0 when potential
    // alone already exceeds the bound (kills all velocity); ratio >= 1 when
    // the state is already within budget, and the clamp below then makes
    // this the identity, so "never scale up" holds without a branch.
    //
    // scale is detached: the projection is a hard, post-hoc correction
    // applied to the model's proposal (CONCEPT sec. 4, "propose, then
    // project"), not a differentiable layer the model should learn to
    // anticipate through its gradient. `sqrt` has an unbounded derivative
    // as its argument -> 0, which `budget` reaches whenever a batch
    // member's potential energy alone already meets the bound; detaching
    // budget/kinetic before the sqrt keeps the backward pass finite there.
    // The scaled (u, v) below still carries gradient through the
    // un-scaled path, since `scale` is now just a constant multiplier.
    let budget = (bound.detach() - potential.detach()).clamp_min(0.0);
    let scale = (budget / kinetic.detach().clamp_min(1e-12))
        .sqrt()
        .clamp(0.0, 1.0);
    let scale = scale.reshape([b, 1, 1, 1]);

    let u = x.clone().slice([0..b, 1..2]) * d.std[1] + d.mean[1];
    let v = x.clone().slice([0..b, 2..3]) * d.std[2] + d.mean[2];
    let u = (u * scale.clone() - d.mean[1]) / d.std[1];
    let v = (v * scale - d.mean[2]) / d.std[2];
    x.slice_assign([0..b, 1..2], u).slice_assign([0..b, 2..3], v)
}

/// Box mean `hbar` of physical `h`, surface anomaly `eta = h - hbar`, and
/// physical `(u, v)`, per batch member. `eta` is taken in normalized units
/// and scaled by `std` only, never through `h = 1 + eta`, so it carries no
/// cancellation against the `depth` baseline.
fn anomaly_parts<B: Backend>(
    x: &Tensor<B, 4>,
    d: &Denorm,
) -> (Tensor<B, 2>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    let [b, _, _, _] = x.dims();
    let m = mass(x);
    let eta = (x.clone().slice([0..b, 0..1]) - m.clone().reshape([b, 1, 1, 1])) * d.std[0];
    let hbar = m * d.std[0] + (d.mean[0] + d.depth);
    let u = x.clone().slice([0..b, 1..2]) * d.std[1] + d.mean[1];
    let v = x.clone().slice([0..b, 2..3]) * d.std[2] + d.mean[2];
    (hbar, eta, u, v)
}

/// The three per-member sums the excess energy is built from, `[batch, 1]`
/// each: `a = sum 1/2 g eta^2 + 1/2 hbar |u|^2`, `q = sum 1/2 eta |u|^2`,
/// so that `excess = a + q` and scaling `(eta, u, v)` by `s` gives
/// `s^2 a + s^3 q`.
fn excess_terms<B: Backend>(x: &Tensor<B, 4>, d: &Denorm, g: f32) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let [b, _, ny, nx] = x.dims();
    let n = ny * nx;
    let (hbar, eta, u, v) = anomaly_parts(x, d);
    let sum = |t: Tensor<B, 4>| t.reshape([b, n]).sum_dim(1);
    let speed2 = u.powi_scalar(2) + v.powi_scalar(2);
    let potential = sum(eta.clone().powi_scalar(2)) * (0.5 * g);
    let s2 = sum(speed2.clone());
    let a = potential + hbar * s2 * 0.5;
    let q = sum(eta * speed2) * 0.5;
    (a, q)
}

/// Excess (available) energy per batch member, `[batch, 1]`, physical units:
/// `sum 1/2 g (h - hbar)^2 + 1/2 h (u^2 + v^2)`. With the box mean `hbar`
/// held fixed by [`project_mass`], total energy is this plus the constant
/// `1/2 g hbar^2 N`, so the two have the same per-tick increments; but this
/// one is the part that dissipates, about 1e-4 of the total on the training
/// distribution, where the total's own f32 rounding is the size of the
/// signal (docs/runs/2026-10-06-energy-inequality.md).
pub fn excess_energy<B: Backend>(x: &Tensor<B, 4>, d: &Denorm, g: f32) -> Tensor<B, 2> {
    let (a, q) = excess_terms(x, d, g);
    a + q
}

/// Project onto `excess_energy(x) <= prev_excess * r` (energy non-increasing
/// when `r <= 1`), keeping the box mass exactly. The fluctuation `(eta, u,
/// v)` is scaled by one `s` in `[0, 1]` per batch member, toward the rest
/// state `(hbar, 0, 0)`: `eta`'s mean stays 0, so mass is untouched, and the
/// state is always feasible at `s = 0`. With `excess(s) = s^2 a + s^3 q`,
/// `s = sqrt(bound / (a + max(q, 0)))` is feasible in closed form (for
/// `q >= 0`, `s^3 q <= s^2 q`; for `q < 0` the cubic term only helps) and
/// within `(1 - s) |q| / a` of the exact root, which is negligible since
/// `|q| / a ~ max|eta| / hbar`. Members already within the bound are
/// returned bit-for-bit. `s` is detached, as in [`project_energy`].
pub fn project_excess_energy<B: Backend>(
    x: Tensor<B, 4>,
    prev_excess: Tensor<B, 2>,
    d: &Denorm,
    g: f32,
    r: f32,
) -> Tensor<B, 4> {
    let [b, _, ny, nx] = x.dims();
    let bound = prev_excess.detach() * r;
    let (a, q) = excess_terms(&x, d, g);
    let (a, q) = (a.detach(), q.detach());
    let active = (a.clone() + q.clone()).greater(bound.clone());
    let s = (bound / (a + q.clamp_min(0.0)).clamp_min(1e-30))
        .clamp_min(0.0)
        .sqrt()
        .clamp(0.0, 1.0)
        .reshape([b, 1, 1, 1]);

    let m = mass(&x).reshape([b, 1, 1, 1]);
    let h = x.clone().slice([0..b, 0..1]);
    let h = (h - m.clone()) * s.clone() + m;
    let u = x.clone().slice([0..b, 1..2]) * d.std[1] + d.mean[1];
    let v = x.clone().slice([0..b, 2..3]) * d.std[2] + d.mean[2];
    let u = (u * s.clone() - d.mean[1]) / d.std[1];
    let v = (v * s - d.mean[2]) / d.std[2];
    let projected = x.clone().slice_assign([0..b, 0..1], h).slice_assign([0..b, 1..2], u).slice_assign([0..b, 2..3], v);
    let active = active.reshape([b, 1, 1, 1]).expand([b, 3, ny, nx]);
    x.mask_where(active, projected)
}

/// Box mean of channel 0 (the surface height), `[batch, 1]`.
pub fn mass<B: Backend>(x: &Tensor<B, 4>) -> Tensor<B, 2> {
    let [b, _, ny, nx] = x.dims();
    x.clone()
        .slice([0..b, 0..1])
        .reshape([b, ny * nx])
        .mean_dim(1)
}

/// Shift channel 0 by a constant per batch member so its box mean equals
/// `target` (which is what [`mass`] returned for the reference state).
pub fn project_mass<B: Backend>(x: Tensor<B, 4>, target: Tensor<B, 2>) -> Tensor<B, 4> {
    let [b, _, _, _] = x.dims();
    let delta = (target - mass(&x)).reshape([b, 1, 1, 1]);
    let h = x.clone().slice([0..b, 0..1]) + delta;
    x.slice_assign([0..b, 0..1], h)
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;

    #[test]
    fn projection_restores_the_total_and_leaves_u_and_v_alone() {
        let device = Default::default();
        let x = Tensor::<B, 1>::from_floats(
            (0..2 * 3 * 8 * 8)
                .map(|i| ((i % 17) as f32 - 8.0) / 9.0)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([2, 3, 8, 8]);
        let target = Tensor::<B, 2>::from_floats([[0.25f32], [-0.5]], &device);
        let y = project_mass(x.clone(), target.clone());

        let got: Vec<f32> = mass(&y).into_data().to_vec().unwrap();
        assert!((got[0] - 0.25).abs() < 1e-6, "{got:?}");
        assert!((got[1] + 0.5).abs() < 1e-6, "{got:?}");

        let uv = |t: Tensor<B, 4>| t.slice([0..2, 1..3]).into_data();
        uv(y.clone()).assert_approx_eq::<f32>(&uv(x.clone()), burn::tensor::Tolerance::default());

        // The shift is constant in space: differences within channel 0 hold.
        let d = |t: Tensor<B, 4>| {
            t.clone().slice([0..2, 0..1, 0..8, 1..8]) - t.slice([0..2, 0..1, 0..8, 0..7])
        };
        d(y).into_data()
            .assert_approx_eq::<f32>(&d(x).into_data(), burn::tensor::Tolerance::default());
    }

    fn flat_state(h: f32, u: f32, v: f32, device: &<B as Backend>::Device) -> Tensor<B, 4> {
        let n = 4 * 4;
        let mut v_all = vec![h; n];
        v_all.extend(vec![u; n]);
        v_all.extend(vec![v; n]);
        Tensor::<B, 1>::from_floats(v_all.as_slice(), device).reshape([1, 3, 4, 4])
    }

    fn identity_denorm() -> Denorm {
        Denorm {
            mean: [0.0; 3],
            std: [1.0; 3],
            depth: 0.0,
        }
    }

    #[test]
    fn energy_projection_scales_down_to_the_bound_and_leaves_mass_alone() {
        let device = Default::default();
        let d = identity_denorm();
        let g = 9.81;
        let x = flat_state(1.0, 2.0, 2.0, &device);
        let e0 = energy(&x, &d, g);
        // 78.48 potential + 64.0 kinetic = 142.48, well over a 100 bound.
        let bound = Tensor::<B, 2>::from_floats([[100.0f32]], &device);
        let m0 = mass(&x);

        let y = project_energy(x, bound.clone(), &d, g, 1.0);
        let e1: f32 = energy(&y, &d, g).into_scalar();
        assert!(e1 <= 100.0 + 1e-3, "energy {e1} exceeds the bound");
        assert!(e1 > 99.0, "projection over-corrected: {e1}");

        let m1: f32 = mass(&y).into_scalar();
        let m0: f32 = m0.into_scalar();
        assert!((m1 - m0).abs() < 1e-6, "mass changed: {m0} -> {m1}");
        let _ = e0;
    }

    #[test]
    fn energy_projection_is_identity_within_the_bound() {
        let device = Default::default();
        let d = identity_denorm();
        let g = 9.81;
        let x = flat_state(1.0, 2.0, 2.0, &device);
        // 142.48 actual energy, well inside a 200 bound: nothing to do.
        let bound = Tensor::<B, 2>::from_floats([[200.0f32]], &device);

        let y = project_energy(x.clone(), bound, &d, g, 1.0);
        y.into_data()
            .assert_approx_eq::<f32>(&x.into_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn energy_projection_gradient_is_finite_at_zero_budget() {
        type AB = burn::backend::Autodiff<B>;
        let device = Default::default();
        let d = identity_denorm();
        let g = 9.81;

        let n = 2 * 4 * 4;
        let mut v_all = vec![1.0f32; n];
        v_all.extend(vec![2.0f32; n]);
        v_all.extend(vec![2.0f32; n]);
        let x = Tensor::<AB, 1>::from_floats(v_all.as_slice(), &device)
            .reshape([2, 3, 4, 4])
            .require_grad();

        // Member 0's potential energy alone already meets a tiny bound, so
        // budget = 0 there; member 1 has a normal, non-tight bound.
        let bound = Tensor::<AB, 2>::from_floats([[0.0f32], [200.0]], &device);

        let y = project_energy(x.clone(), bound, &d, g, 1.0);
        let loss = y.powi_scalar(2).sum();
        let grads = loss.backward();

        let grad = x
            .grad(&grads)
            .expect("project_energy should be differentiable w.r.t. x");
        let grad: Vec<f32> = grad.into_data().to_vec::<f32>().unwrap();
        assert!(
            grad.iter().all(|v| v.is_finite()),
            "non-finite gradient at budget=0: {grad:?}"
        );
    }

    /// A `[b, 3, 8, 8]` state in normalized units, deterministic, with a
    /// different amplitude per member, decoded with model20's normalizer
    /// (`h` anomalies ~6e-3, speeds ~1e-2 around depth 1).
    fn wavy(b: usize, device: &<B as Backend>::Device) -> (Tensor<B, 4>, Denorm) {
        let n = 3 * 8 * 8;
        let vals: Vec<f32> = (0..b * n)
            .map(|i| {
                let amp = 1.0 + (i / n) as f32;
                amp * ((i as f32 * 0.37).sin() + 0.5 * (i as f32 * 1.91).cos())
            })
            .collect();
        let x = Tensor::<B, 1>::from_floats(vals.as_slice(), device).reshape([b, 3, 8, 8]);
        let d = Denorm {
            mean: [0.0008, 0.0004, 0.0005],
            std: [0.0063, 0.0138, 0.0135],
            depth: 1.0,
        };
        (x, d)
    }

    #[test]
    fn excess_energy_has_the_same_increments_as_total_energy_at_fixed_mass() {
        let device = Default::default();
        let (x, d) = wavy(1, &device);
        let g = 9.81;
        let x2 = project_mass(x.clone() * 0.5, mass(&x));
        let de: f32 = (energy(&x2, &d, g) - energy(&x, &d, g)).into_scalar();
        let dx: f32 = (excess_energy(&x2, &d, g) - excess_energy(&x, &d, g)).into_scalar();
        assert!((de - dx).abs() < 2e-3 * dx.abs(), "dE {de} vs dE' {dx}");
    }

    #[test]
    fn excess_projection_never_raises_excess_energy_and_keeps_mass() {
        let device = Default::default();
        let (x, d) = wavy(4, &device);
        let g = 9.81;
        let e: Vec<f32> = excess_energy(&x, &d, g).into_data().to_vec().unwrap();
        // Bounds at 0, 10 %, 90 % and 99.99 % of each member's own excess.
        let fr = [0.0f32, 0.1, 0.9, 0.9999];
        let prev: Vec<f32> = e.iter().zip(fr).map(|(e, f)| e * f).collect();
        let prev_t = Tensor::<B, 1>::from_floats(prev.as_slice(), &device).reshape([4, 1]);
        let m0: Vec<f32> = mass(&x).into_data().to_vec().unwrap();

        let y = project_excess_energy(x, prev_t, &d, g, 1.0);
        let e1: Vec<f32> = excess_energy(&y, &d, g).into_data().to_vec().unwrap();
        let m1: Vec<f32> = mass(&y).into_data().to_vec().unwrap();
        for i in 0..4 {
            assert!(e1[i] <= prev[i] * (1.0 + 1e-5) + 1e-12, "member {i}: {} > {}", e1[i], prev[i]);
            // Tight: the closed-form s gives back nearly the whole budget.
            assert!(e1[i] >= prev[i] * 0.99, "member {i} over-corrected: {} vs {}", e1[i], prev[i]);
            assert!((m1[i] - m0[i]).abs() < 1e-6, "member {i} mass {} -> {}", m0[i], m1[i]);
        }
    }

    #[test]
    fn excess_projection_is_the_identity_when_inactive() {
        let device = Default::default();
        let (x, d) = wavy(3, &device);
        let g = 9.81;
        let e: Vec<f32> = excess_energy(&x, &d, g).into_data().to_vec().unwrap();
        // Member 1 is over its bound, members 0 and 2 are inside theirs.
        let prev: Vec<f32> = vec![e[0] * 1.5, e[1] * 0.5, e[2]];
        let prev_t = Tensor::<B, 1>::from_floats(prev.as_slice(), &device).reshape([3, 1]);

        let y = project_excess_energy(x.clone(), prev_t, &d, g, 1.0);
        let member = |t: &Tensor<B, 4>, i: usize| -> Vec<f32> {
            t.clone().slice([i..i + 1]).into_data().to_vec().unwrap()
        };
        assert_eq!(member(&y, 0), member(&x, 0));
        assert_eq!(member(&y, 2), member(&x, 2));
        assert_ne!(member(&y, 1), member(&x, 1));
    }

    #[test]
    fn excess_projection_gradient_is_finite_at_zero_budget() {
        type AB = burn::backend::Autodiff<B>;
        let device = Default::default();
        let (x, d) = wavy(2, &device);
        let x = Tensor::<AB, 4>::from_data(x.into_data(), &device).require_grad();
        let prev = Tensor::<AB, 2>::from_floats([[0.0f32], [1e9]], &device);
        let y = project_excess_energy(x.clone(), prev, &d, 9.81, 1.0);
        let grads = y.powi_scalar(2).sum().backward();
        let grad: Vec<f32> = x.grad(&grads).unwrap().into_data().to_vec().unwrap();
        assert!(grad.iter().all(|v| v.is_finite()), "non-finite gradient at budget 0");
    }
}
