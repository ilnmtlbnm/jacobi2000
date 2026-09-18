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
}
