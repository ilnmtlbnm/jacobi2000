//! Jacobi2000 inference: reads the `model.safetensors` +
//! `config.json` checkpoints the PyTorch export writes
//! (`docs/pytorch-training.md`), runs the forward pass over flat `f32`
//! buffers (`cpu.rs`), batch size 1 (the only size `crates/web` ever runs).
//!
//! Backend selection is `lean_runtime::backend`'s capability ladder
//! (WebGPU, then CPU threads, then one CPU thread) — never a runtime
//! benchmark. The CPU path (`threads` feature for the rayon variant) is in
//! `cpu.rs`, the WebGPU path (`wgpu` feature) in `gpu.rs`.

pub mod config;
pub mod cpu;
#[cfg(feature = "wgpu")]
pub mod gpu;
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
        let config_json = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let bytes = std::fs::read(dir.join("model.safetensors")).unwrap();
        Model::from_bytes(&bytes, &config_json)
    }

    /// Same as [`Self::load`], from in-memory bytes: the browser has no
    /// filesystem, so `crates/web` fetches `model.safetensors` and
    /// `config.json` and hands the bytes straight here.
    pub fn from_bytes(safetensors: &[u8], config_json: &str) -> Model {
        let json: serde_json::Value = serde_json::from_str(config_json).expect("config.json: invalid JSON");
        let w = weights::read_safetensors_bytes(safetensors);
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

    /// Build the WebGPU-backed twin of this model (same weights, same
    /// config), for the top of the capability ladder.
    #[cfg(feature = "wgpu")]
    pub async fn to_gpu(&self) -> Result<gpu::GpuModel, lean_runtime::Error> {
        gpu::GpuModel::new(self.cfg.clone(), &self.w).await
    }
}
