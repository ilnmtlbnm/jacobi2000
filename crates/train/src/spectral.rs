//! A wavenumber-weighted error term (CONCEPT sec. 5: "MSE lets the model damp
//! short wavelengths, invisible at 10 ticks, wrong at 100").
//!
//! The band split is a Laplacian pyramid, not a Fourier transform: a 3x3 box
//! blur is a low-pass, so `x - blur(x)` is the top octave of wavenumbers,
//! `blur(x) - blur(blur(x))` the next one down, and so on. Three levels cover
//! wavelengths of roughly 2-4, 4-8 and 8-16 cells. No FFT, one pooling op per
//! level, differentiable on the GPU.
//!
//! The error field of the predicted increment is the same field as the error
//! of the predicted state (both sides share the input), so this is applied to
//! the state error directly.

use burn::prelude::*;
use burn::tensor::module::avg_pool2d;

/// Weight per band, highest wavenumber first. Short wavelengths are the ones
/// plain MSE under-counts, so they carry the most weight.
pub const BAND_WEIGHTS: [f32; 3] = [1.0, 0.5, 0.25];

/// Sum over bands of `w_i * MSE(band_i(err))`, for `err` = `[batch, c, ny, nx]`.
pub fn band_mse<B: Backend>(err: Tensor<B, 4>) -> Tensor<B, 1> {
    let mut low = err;
    let mut loss = Tensor::<B, 1>::zeros([1], &low.device());
    for w in BAND_WEIGHTS {
        let next = avg_pool2d(low.clone(), [3, 3], [1, 1], [1, 1], false, false);
        let band = low - next.clone();
        loss = loss + band.powi_scalar(2).mean() * w;
        low = next;
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;

    fn checker<const N: usize>(period: usize, device: &<B as Backend>::Device) -> Tensor<B, 4> {
        let v: Vec<f32> = (0..N * N)
            .map(|i| {
                let (x, y) = (i % N, i / N);
                if (x / period + y / period).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        Tensor::<B, 1>::from_floats(v.as_slice(), device).reshape([1, 1, N, N])
    }

    #[test]
    fn short_wavelengths_cost_more_than_long_ones_at_equal_power() {
        // Two error fields with identical plain MSE (both are +-1 everywhere).
        // The term has to charge more for the one that wiggles fast, or it is
        // not doing the job MSE fails at.
        let device = Default::default();
        let fast: f32 = band_mse(checker::<32>(1, &device)).into_scalar();
        let slow: f32 = band_mse(checker::<32>(8, &device)).into_scalar();
        assert!(fast > 4.0 * slow, "fast {fast} vs slow {slow}");
    }
}
