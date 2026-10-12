//! Attention mask. Rather
//! than materialize an additive `[tokens, tokens]` matrix, `allowed` is
//! queried directly in the attention inner loop (same Chebyshev-neighbourhood
//! predicate, same result, no `tokens^2` f32 allocation).

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Mask {
    Full,
    Stencil { radius: usize, dilation: usize },
}

impl Mask {
    /// Token-grid coordinates from a flat token index (row-major: `t = ty *
    /// tokens_x + tx`).
    #[inline]
    pub fn allowed(&self, tokens_x: usize, q: usize, k: usize) -> bool {
        match *self {
            Mask::Full => true,
            Mask::Stencil { radius, dilation } => {
                let (qx, qy) = (q % tokens_x, q / tokens_x);
                let (kx, ky) = (k % tokens_x, k / tokens_x);
                let dx = kx as isize - qx as isize;
                let dy = ky as isize - qy as isize;
                let d = dilation as isize;
                dx % d == 0 && dy % d == 0 && (dx / d).unsigned_abs() <= radius && (dy / d).unsigned_abs() <= radius
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stencil_matches_the_tensor_mask_truth_table() {
        // The mask truth table of `model.py`'s `stencil_mask`, on a 16x16 token grid.
        let m = Mask::Stencil { radius: 1, dilation: 1 };
        let tx = 16;
        let q = 5 * tx + 5;
        assert!(m.allowed(tx, q, q));
        assert!(m.allowed(tx, q, 5 * tx + 6));
        assert!(m.allowed(tx, q, 6 * tx + 6));
        assert!(!m.allowed(tx, q, 5 * tx + 7));
        assert!(!m.allowed(tx, q, 9 * tx + 5));
    }

    #[test]
    fn dilation_skips_every_other_token() {
        let m = Mask::Stencil { radius: 1, dilation: 2 };
        let tx = 16;
        let q = 5 * tx + 5;
        assert!(m.allowed(tx, q, 5 * tx + 7));
        assert!(!m.allowed(tx, q, 5 * tx + 6));
    }
}
