//! Field normalizer, mirroring `train::data::Normalizer` without Burn.
//! State layout: `[channel, y, x]` flat, channel 0 = h, 1 = u, 2 = v, same as
//! `jacobi2000::model::Jacobi2000`'s `[batch, channels, ny, nx]`.

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
}
