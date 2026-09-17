//! Checkpoints: weights as a Burn record plus the metadata the runtime needs
//! (grid, normalizer, solver parameters). Weights are never committed.

use serde::{Deserialize, Serialize};
use solvers::Params;

use crate::data::Normalizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub nx: usize,
    pub ny: usize,
    pub patch: usize,
    pub channels: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub ff_mult: usize,
    pub meta_dim: usize,
    pub norm: Normalizer,
    pub dx: f32,
    pub dt: f32,
    pub g: f32,
    pub depth: f32,
    pub nu: f32,
}

impl Meta {
    pub fn params(&self) -> Params {
        Params {
            nx: self.nx,
            ny: self.ny,
            dx: self.dx,
            dt: self.dt,
            g: self.g,
            depth: self.depth,
            nu: self.nu,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub meta: Meta,
    pub weights: Vec<u8>,
}
