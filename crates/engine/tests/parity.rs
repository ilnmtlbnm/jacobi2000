//! Parity gate: this crate's CPU forward pass against the Burn reference
//! (`jacobi2000::Jacobi2000::forward`, NdArray backend) on a random-init
//! model, exchanged through the same safetensors + config.json format the
//! PyTorch export writes (docs/pytorch-training.md). Not a trained
//! checkpoint — the gate is numerical agreement of the two
//! implementations, not model quality.

use std::path::Path;

use burn::prelude::*;
use jacobi2000::{Jacobi2000Config, Mask as BurnMask};

type B = burn::backend::NdArray;

fn write_checkpoint(dir: &Path, cfg: &Jacobi2000Config, mask: BurnMask) {
    let device = Default::default();
    let model = cfg.init::<B>(&device);
    let weights = model.to_weights();
    jacobi2000::write_safetensors(&dir.join("model.safetensors"), &weights);

    let mask_json = match mask {
        BurnMask::Full => serde_json::json!("Full"),
        BurnMask::Stencil { radius, dilation } => serde_json::json!({"Stencil": {"radius": radius, "dilation": dilation}}),
    };
    let config = serde_json::json!({
        "nx": cfg.nx, "ny": cfg.ny, "patch": cfg.patch, "channels": cfg.channels,
        "dim": cfg.dim, "layers": cfg.layers, "heads": cfg.heads, "ff_mult": cfg.ff_mult,
        "meta_dim": cfg.meta_dim,
        "norm": {"mean": [0.1, -0.05, 0.02], "std": [0.3, 0.2, 0.2], "depth": 1.0},
        "dx": 1.0, "dt": 0.09, "g": 9.81, "depth": 1.0, "nu": 1e-3,
        "mask": mask_json,
        "energy_r_max": 0.999,
    });
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
}

/// A deterministic, non-trivial input: two low-order sinusoids per channel,
/// matching the shape of `jacobi/cli`'s own `forward-probe` fixture.
fn probe_state(channels: usize, ny: usize, nx: usize) -> Vec<f32> {
    let mut state = vec![0.0f32; channels * ny * nx];
    for c in 0..channels {
        for y in 0..ny {
            for x in 0..nx {
                let v = ((c + 1) as f32 * 0.05)
                    * (std::f32::consts::TAU * (c + 1) as f32 * x as f32 / nx as f32).sin()
                    * (std::f32::consts::TAU * y as f32 / ny as f32).cos();
                state[(c * ny + y) * nx + x] = v;
            }
        }
    }
    state
}

fn max_abs_rel(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        let rel = d / x.abs().max(y.abs()).max(1e-6);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel)
}

fn run_case(name: &str, cfg: Jacobi2000Config, mask: BurnMask) {
    let dir = std::env::temp_dir().join(format!("jacobi2000-engine-parity-{name}-{}", std::process::id()));
    write_checkpoint(&dir, &cfg, mask);

    let device = Default::default();
    let (burn_model, _meta): (jacobi2000::Jacobi2000<B>, ()) = {
        let weights = jacobi2000::read_safetensors(&dir.join("model.safetensors"));
        (jacobi2000::Jacobi2000::from_weights(&cfg, &weights, &device), ())
    };
    let state = probe_state(cfg.channels, cfg.ny, cfg.nx);
    let meta_vec: Vec<f32> = (0..cfg.meta_dim).map(|i| 0.1 * (i as f32 + 1.0)).collect();

    let state_t = Tensor::<B, 1>::from_floats(state.as_slice(), &device).reshape([1, cfg.channels, cfg.ny, cfg.nx]);
    let meta_t = Tensor::<B, 1>::from_floats(meta_vec.as_slice(), &device).reshape([1, cfg.meta_dim]);
    let mask_t = mask.to_tensor::<B>(&burn_model.grid, &device);
    let burn_out: Vec<f32> = burn_model.forward(state_t, meta_t, mask_t).into_data().to_vec().unwrap();

    let engine_model = engine::Model::load(&dir);
    let engine_out = engine_model.forward(&state, &meta_vec);

    let (max_abs, max_rel) = max_abs_rel(&burn_out, &engine_out);
    println!("{name}: 1-step max_abs={max_abs:e} max_rel={max_rel:e}");
    assert!(max_abs < 1e-4, "{name}: max abs diff {max_abs} too large");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stencil_mask_one_step_matches_burn() {
    let cfg = Jacobi2000Config::new().with_layers(2);
    run_case("stencil", cfg, BurnMask::Stencil { radius: 1, dilation: 1 });
}

#[test]
fn full_mask_one_step_matches_burn() {
    let cfg = Jacobi2000Config::new().with_layers(2);
    run_case("full", cfg, BurnMask::Full);
}

/// GPU-vs-CPU parity on the same random-init checkpoint, same probe input.
/// Dispatches real GPU work — never run this on the M2 (Metal); it is a
/// box-only (Vulkan) gate. Skips cleanly (prints and returns) if no adapter
/// is granted, so it is still safe to leave in the default test binary.
#[cfg(feature = "wgpu")]
#[test]
fn gpu_matches_cpu_on_random_init() {
    let cfg = Jacobi2000Config::new().with_layers(2);
    let mask = BurnMask::Stencil { radius: 1, dilation: 1 };
    let dir = std::env::temp_dir().join(format!("jacobi2000-engine-parity-gpu-{}", std::process::id()));
    write_checkpoint(&dir, &cfg, mask);

    let cpu_model = engine::Model::load(&dir);
    let gpu_model = match pollster::block_on(cpu_model.to_gpu()) {
        Ok(m) => m,
        Err(e) => {
            println!("gpu_matches_cpu_on_random_init: skipped, no adapter ({e:?})");
            std::fs::remove_dir_all(&dir).ok();
            return;
        }
    };

    let state = probe_state(cfg.channels, cfg.ny, cfg.nx);
    let meta_vec: Vec<f32> = (0..cfg.meta_dim).map(|i| 0.1 * (i as f32 + 1.0)).collect();

    let cpu_out = cpu_model.forward(&state, &meta_vec);
    let gpu_out = pollster::block_on(gpu_model.forward(&state, &meta_vec));
    let (max_abs, max_rel) = max_abs_rel(&cpu_out, &gpu_out);
    println!("gpu vs cpu, 1-step: max_abs={max_abs:e} max_rel={max_rel:e}");
    assert!(max_abs < 1e-2, "GPU forward diverged from CPU: max abs diff {max_abs}");

    let mut cpu_state = state.clone();
    let mut gpu_state = state;
    let mut max_abs_50 = 0.0f32;
    for _ in 0..50 {
        cpu_state = cpu_model.step(&cpu_state, &meta_vec);
        gpu_state = pollster::block_on(gpu_model.step(&gpu_state, &meta_vec));
        let (a, _) = max_abs_rel(&cpu_state, &gpu_state);
        max_abs_50 = max_abs_50.max(a);
    }
    println!("gpu vs cpu, 50-step rollout: max_abs={max_abs_50:e}");
    assert!(max_abs_50 < 1.0, "GPU rollout diverged from CPU: max abs diff {max_abs_50}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn fifty_step_rollout_stays_close_to_burn() {
    let cfg = Jacobi2000Config::new().with_layers(2);
    let mask = BurnMask::Stencil { radius: 1, dilation: 1 };
    let dir = std::env::temp_dir().join(format!("jacobi2000-engine-parity-rollout-{}", std::process::id()));
    write_checkpoint(&dir, &cfg, mask);

    let device = Default::default();
    let weights = jacobi2000::read_safetensors(&dir.join("model.safetensors"));
    let burn_model = jacobi2000::Jacobi2000::from_weights(&cfg, &weights, &device);
    let mask_t = mask.to_tensor::<B>(&burn_model.grid, &device);
    let engine_model = engine::Model::load(&dir);

    let mut state = probe_state(cfg.channels, cfg.ny, cfg.nx);
    let meta_vec: Vec<f32> = (0..cfg.meta_dim).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let meta_t = Tensor::<B, 1>::from_floats(meta_vec.as_slice(), &device).reshape([1, cfg.meta_dim]);

    let mut burn_state_t = Tensor::<B, 1>::from_floats(state.as_slice(), &device).reshape([1, cfg.channels, cfg.ny, cfg.nx]);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut corr_num = 0.0f64;
    let (mut corr_a, mut corr_b) = (0.0f64, 0.0f64);
    for _ in 0..50 {
        burn_state_t = burn_model.step(burn_state_t, meta_t.clone(), mask_t.clone());
        state = engine_model.step(&state, &meta_vec);
        let burn_vec: Vec<f32> = burn_state_t.clone().into_data().to_vec().unwrap();
        let (a, r) = max_abs_rel(&burn_vec, &state);
        max_abs = max_abs.max(a);
        max_rel = max_rel.max(r);
        for (x, y) in burn_vec.iter().zip(state.iter()) {
            corr_num += (*x as f64) * (*y as f64);
            corr_a += (*x as f64) * (*x as f64);
            corr_b += (*y as f64) * (*y as f64);
        }
    }
    let corr = corr_num / (corr_a.sqrt() * corr_b.sqrt());
    println!("50-step rollout: max_abs={max_abs:e} max_rel={max_rel:e} corr_vs_burn={corr:.9}");
    assert!(max_abs < 5e-3, "50-step rollout drifted from Burn: max abs diff {max_abs}");
    assert!(corr > 0.999_99, "50-step rollout decorrelated from Burn: corr {corr}");

    std::fs::remove_dir_all(&dir).ok();
}
