//! `config.json`'s schema is the one `training/jacobi2000_pt/export.py`
//! writes (docs/pytorch-training.md); this reads the fields the engine needs and
//! infers `head_dim`/`ff_inner` from the checkpoint's own tensor shapes
//! rather than trusting fields that may not be present (`head_dim` isn't in
//! `config.json` at all).

use serde::Deserialize;

use crate::mask::Mask;
use crate::norm::Normalizer;
use crate::weights::{get, Weights};

#[derive(Debug, Clone, Deserialize)]
struct Raw {
    nx: usize,
    ny: usize,
    patch: usize,
    channels: usize,
    dim: usize,
    layers: usize,
    heads: usize,
    meta_dim: usize,
    norm: Normalizer,
    dx: f32,
    dt: f32,
    g: f32,
    depth: f32,
    nu: f32,
    #[serde(default = "default_r_max")]
    energy_r_max: f32,
    #[serde(default = "default_mask")]
    mask: Mask,
}

fn default_r_max() -> f32 {
    1.0
}

fn default_mask() -> Mask {
    Mask::Full
}

#[derive(Debug, Clone)]
pub struct Config {
    pub nx: usize,
    pub ny: usize,
    pub patch: usize,
    pub channels: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ff_inner: usize,
    pub meta_dim: usize,
    pub tokens_x: usize,
    pub tokens_y: usize,
    pub tokens: usize,
    pub patch_dim: usize,
    pub norm: Normalizer,
    pub dx: f32,
    pub dt: f32,
    pub g: f32,
    pub depth: f32,
    pub nu: f32,
    pub energy_r_max: f32,
    pub mask: Mask,
}

impl Config {
    pub fn from_json(json: &serde_json::Value, w: &Weights) -> Config {
        let r: Raw = serde_json::from_value(json.clone()).expect("config.json: missing/invalid field");
        let head_dim = get(w, "head.to_cells.weight").shape[0];
        let ff_inner = get(w, "blocks.0.ff1.weight").shape[1];
        let tokens_x = r.nx / r.patch;
        let tokens_y = r.ny / r.patch;
        Config {
            nx: r.nx,
            ny: r.ny,
            patch: r.patch,
            channels: r.channels,
            dim: r.dim,
            layers: r.layers,
            heads: r.heads,
            head_dim,
            ff_inner,
            meta_dim: r.meta_dim,
            tokens_x,
            tokens_y,
            tokens: tokens_x * tokens_y,
            patch_dim: r.channels * r.patch * r.patch,
            norm: r.norm,
            dx: r.dx,
            dt: r.dt,
            g: r.g,
            depth: r.depth,
            nu: r.nu,
            energy_r_max: r.energy_r_max,
            mask: r.mask,
        }
    }

    pub fn params(&self) -> solvers::Params {
        solvers::Params { nx: self.nx, ny: self.ny, dx: self.dx, dt: self.dt, g: self.g, depth: self.depth, nu: self.nu }
    }
}
