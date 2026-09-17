//! Attention masks. The mask is the problem statement, so it is a value the
//! caller builds and hands to `forward`, never something the model decides.
//!
//! Masks are additive: 0 where attention is allowed, -inf where it is not,
//! shaped `[tokens, tokens]` and broadcast over batch and heads.

use burn::prelude::*;

use crate::grid::Grid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mask {
    /// Every token sees every token. Elliptic character: instantaneous
    /// global coupling.
    Full,
    /// Chebyshev neighbourhood of `radius` token-cells, stepping by
    /// `dilation`. Hyperbolic character: information moves one stencil per
    /// layer. `dilation > 1` is the coarse level of the multigrid hierarchy
    /// on the same grid with the same weights.
    Stencil { radius: usize, dilation: usize },
}

impl Mask {
    /// `None` means no mask tensor is needed at all (full attention).
    pub fn to_tensor<B: Backend>(&self, grid: &Grid, device: &B::Device) -> Option<Tensor<B, 2>> {
        let n = grid.tokens();
        match *self {
            Mask::Full => None,
            Mask::Stencil { radius, dilation } => {
                let mut data = vec![f32::NEG_INFINITY; n * n];
                for q in 0..n {
                    let (qx, qy) = grid.token_xy(q);
                    for k in 0..n {
                        let (kx, ky) = grid.token_xy(k);
                        let dx = kx as isize - qx as isize;
                        let dy = ky as isize - qy as isize;
                        let d = dilation as isize;
                        if dx % d == 0
                            && dy % d == 0
                            && (dx / d).unsigned_abs() <= radius
                            && (dy / d).unsigned_abs() <= radius
                        {
                            data[q * n + k] = 0.0;
                        }
                    }
                }
                Some(Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([n, n]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;

    #[test]
    fn stencil_allows_only_the_neighbourhood() {
        let g = Grid::default();
        let device = Default::default();
        let m = Mask::Stencil {
            radius: 1,
            dilation: 1,
        }
        .to_tensor::<B>(&g, &device)
        .unwrap();
        let d = m.into_data();
        let v = d.as_slice::<f32>().unwrap();
        let n = g.tokens();
        // Token (5,5) = 5 * 16 + 5.
        let q = 5 * 16 + 5;
        assert_eq!(v[q * n + q], 0.0);
        assert_eq!(v[q * n + (5 * 16 + 6)], 0.0);
        assert_eq!(v[q * n + (6 * 16 + 6)], 0.0);
        assert!(v[q * n + (5 * 16 + 7)].is_infinite());
        assert!(v[q * n + (9 * 16 + 5)].is_infinite());
        // Never mask a token from itself: a fully -inf row is NaN in softmax.
        for q in 0..n {
            assert_eq!(v[q * n + q], 0.0);
        }
    }

    #[test]
    fn dilation_skips_every_other_token() {
        let g = Grid::default();
        let device = Default::default();
        let m = Mask::Stencil {
            radius: 1,
            dilation: 2,
        }
        .to_tensor::<B>(&g, &device)
        .unwrap();
        let d = m.into_data();
        let v = d.as_slice::<f32>().unwrap();
        let n = g.tokens();
        let q = 5 * 16 + 5;
        assert_eq!(v[q * n + (5 * 16 + 7)], 0.0);
        assert!(v[q * n + (5 * 16 + 6)].is_infinite());
    }

    #[test]
    fn full_needs_no_tensor() {
        let device = Default::default();
        assert!(Mask::Full
            .to_tensor::<B>(&Grid::default(), &device)
            .is_none());
    }
}
