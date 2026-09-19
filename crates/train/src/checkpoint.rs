//! Checkpoints: weights as a Burn record plus the metadata the runtime needs
//! (grid, normalizer, solver parameters). Weights are never committed.

use std::path::Path;

use burn::module::Module;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use jacobi2000::{Jacobi2000, Jacobi2000Config, Mask};
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
    #[serde(default = "default_mask")]
    pub mask: Mask,
    /// Per-tick energy ratio bound (`E_new <= E_prev * r_max`) used by the
    /// energy projection this checkpoint was trained/evaluated with, measured
    /// from the solver's own trajectories (see
    /// `train::fit::fallback_r_max` and docs/runs/2026-09-19-energy-stats.md).
    /// Meaningless unless `--project-energy` is also on. Defaults to 1.0
    /// (no energy growth allowed) for checkpoints saved before this field
    /// existed, none of which used the energy projection anyway.
    #[serde(default = "default_r_max")]
    pub energy_r_max: f32,
}

fn default_mask() -> Mask {
    Mask::Full
}

fn default_r_max() -> f32 {
    1.0
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

pub fn save<B: Backend>(out: &Path, model: &Jacobi2000<B>, meta: &Meta) {
    std::fs::create_dir_all(out).unwrap();
    let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = rec.record(model.clone().into_record(), ()).unwrap();
    std::fs::write(out.join("model.bin"), bytes).unwrap();
    std::fs::write(
        out.join("meta.json"),
        serde_json::to_string_pretty(meta).unwrap(),
    )
    .unwrap();
}

pub fn load<B: Backend>(dir: &Path, device: &B::Device) -> (Jacobi2000<B>, Meta) {
    let meta: Meta =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let cfg = Jacobi2000Config::new()
        .with_nx(meta.nx)
        .with_ny(meta.ny)
        .with_patch(meta.patch)
        .with_dim(meta.dim)
        .with_layers(meta.layers)
        .with_heads(meta.heads)
        .with_ff_mult(meta.ff_mult);
    let bytes = std::fs::read(dir.join("model.bin")).unwrap();
    let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
    let record = rec.load(bytes, device).unwrap();
    (cfg.init::<B>(device).load_record(record), meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type B = NdArray;

    fn test_meta(cfg: &Jacobi2000Config) -> Meta {
        Meta {
            nx: cfg.nx,
            ny: cfg.ny,
            patch: cfg.patch,
            channels: cfg.channels,
            dim: cfg.dim,
            layers: cfg.layers,
            heads: cfg.heads,
            ff_mult: cfg.ff_mult,
            meta_dim: cfg.meta_dim,
            norm: Normalizer::identity(1.0),
            dx: 1.0,
            dt: 0.1,
            g: 9.81,
            depth: 1.0,
            nu: 1e-3,
            mask: Mask::Full,
            energy_r_max: 1.0,
        }
    }

    #[test]
    fn stencil_mask_round_trips() {
        let device = Default::default();
        let cfg = Jacobi2000Config::new().with_layers(1);
        let model = cfg.init::<B>(&device);
        let mut meta = test_meta(&cfg);
        meta.mask = Mask::Stencil {
            radius: 1,
            dilation: 1,
        };

        let dir = std::env::temp_dir().join(format!(
            "jacobi2000-checkpoint-mask-test-{}",
            std::process::id()
        ));
        save(&dir, &model, &meta);
        let (_, loaded_meta): (Jacobi2000<B>, Meta) = load(&dir, &device);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(
            loaded_meta.mask,
            Mask::Stencil {
                radius: 1,
                dilation: 1
            }
        );
    }

    #[test]
    fn meta_without_mask_field_defaults_to_full() {
        let cfg = Jacobi2000Config::new().with_layers(1);
        let mut json: serde_json::Value =
            serde_json::to_value(test_meta(&cfg)).unwrap();
        json.as_object_mut().unwrap().remove("mask");
        let meta: Meta = serde_json::from_value(json).unwrap();
        assert_eq!(meta.mask, Mask::Full);
    }

    #[test]
    fn load_reproduces_saved_weights() {
        let device = Default::default();
        let cfg = Jacobi2000Config::new().with_layers(1);
        let model = cfg.init::<B>(&device);
        let meta = test_meta(&cfg);

        let dir = std::env::temp_dir().join(format!(
            "jacobi2000-checkpoint-test-{}",
            std::process::id()
        ));
        save(&dir, &model, &meta);
        let (loaded, loaded_meta): (Jacobi2000<B>, Meta) = load(&dir, &device);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded_meta.dim, meta.dim);

        let g = cfg.grid();
        let x = Tensor::<B, 4>::zeros([1, cfg.channels, g.ny, g.nx], &device);
        let meta_t = Tensor::<B, 2>::zeros([1, cfg.meta_dim], &device);
        let out0 = model.forward(x.clone(), meta_t.clone(), None);
        let out1 = loaded.forward(x, meta_t, None);
        let diff: f32 = (out0 - out1).abs().max().into_scalar();
        assert!(diff < 1e-6, "loaded model diverges from original: {diff}");
    }
}
