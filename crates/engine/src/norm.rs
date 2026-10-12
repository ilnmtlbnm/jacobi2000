//! Field normalizer.
//! State layout: `[channel, y, x]` flat, channel 0 = h, 1 = u, 2 = v, the
//! `[batch, channels, ny, nx]` layout of `training/jacobi2000_pt/model.py`.

use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Normalizer {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub depth: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Denorm {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub depth: f32,
}

impl Normalizer {
    pub fn denorm(&self) -> Denorm {
        Denorm { mean: self.mean, std: self.std, depth: self.depth }
    }

    /// Pack a solver state into a normalized `[3, ny, nx]` slice.
    pub fn encode(&self, s: &solvers::State, out: &mut [f32]) {
        let n = s.h.len();
        for i in 0..n {
            out[i] = (s.h[i] - self.depth - self.mean[0]) / self.std[0];
            out[n + i] = (s.u[i] - self.mean[1]) / self.std[1];
            out[2 * n + i] = (s.v[i] - self.mean[2]) / self.std[2];
        }
    }
}
