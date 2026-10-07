//! Jacobi2000 inference off Burn: reads the same `model.safetensors` +
//! `config.json` checkpoints the PyTorch export writes
//! (`docs/pytorch-training.md`), runs the forward pass over flat `f32`
//! buffers (`cpu.rs`), batch size 1 (the only size `crates/web` ever runs).
//!
//! Backend selection is `lean_runtime::backend`'s capability ladder
//! (WebGPU, then CPU threads, then one CPU thread) — never a runtime
//! benchmark. Only the CPU path (`threads` feature for the rayon variant)
//! is implemented so far; see `docs/runs/2026-10-07-inference-lean.md` for
//! what's left before WebGPU joins the ladder.

pub mod config;
pub mod cpu;
pub mod mask;
pub mod metrics;
pub mod norm;
pub mod project;
pub mod solver_meta;
pub mod weights;

use std::path::Path;

pub use config::Config;
pub use mask::Mask;
pub use norm::{Denorm, Normalizer};
pub use solver_meta::{meta_of, META_DIM};
pub use weights::Weights;

pub struct Model {
    pub cfg: Config,
    w: Weights,
}

impl Model {
    pub fn load(dir: &Path) -> Model {
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        let w = weights::read_safetensors(&dir.join("model.safetensors"));
        let cfg = Config::from_json(&json, &w);
        Model { cfg, w }
    }

    /// The predicted increment, same shape as `state` (`[channels, ny, nx]`).
    pub fn forward(&self, state: &[f32], meta: &[f32]) -> Vec<f32> {
        cpu::forward(&self.cfg, &self.w, state, meta)
    }

    /// One tick: `state + forward(state, meta)`.
    pub fn step(&self, state: &[f32], meta: &[f32]) -> Vec<f32> {
        cpu::step(&self.cfg, &self.w, state, meta)
    }
}
