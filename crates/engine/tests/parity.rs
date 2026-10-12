//! Parity gate: this crate's forward pass against reference outputs saved
//! from the PyTorch model (`training/jacobi2000_pt/model.py`) on a tiny
//! seeded checkpoint. `tests/fixtures/pt-tiny` holds the checkpoint
//! (`model.safetensors` + `config.json`, plus `config-full.json` for full
//! attention) and `reference.safetensors` (input, conditioning vector,
//! PyTorch's one-step outputs and its rollout after 1, 10 and 50 steps).
//! Regenerate with `python -m jacobi2000_pt.parity --out <dir>` from
//! `training/`. The gate is numerical agreement of the two implementations,
//! not model quality.

use std::path::PathBuf;

use engine::weights::{get, read_safetensors, Weights};
use engine::Model;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pt-tiny")
}

fn load_model(config: &str) -> Model {
    let dir = fixture_dir();
    let bytes = std::fs::read(dir.join("model.safetensors")).unwrap();
    let config_json = std::fs::read_to_string(dir.join(config)).unwrap();
    Model::from_bytes(&bytes, &config_json)
}

fn reference() -> Weights {
    read_safetensors(&fixture_dir().join("reference.safetensors"))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn stencil_mask_one_step_matches_pytorch() {
    let r = reference();
    let model = load_model("config.json");
    let out = model.forward(&get(&r, "state").data, &get(&r, "meta").data);
    let d = max_abs(&out, &get(&r, "forward_stencil").data);
    println!("stencil: 1-step max_abs={d:e}");
    assert!(d < 1e-5, "max abs diff {d} too large");
}

#[test]
fn full_mask_one_step_matches_pytorch() {
    let r = reference();
    let model = load_model("config-full.json");
    let out = model.forward(&get(&r, "state").data, &get(&r, "meta").data);
    let d = max_abs(&out, &get(&r, "forward_full").data);
    println!("full: 1-step max_abs={d:e}");
    assert!(d < 1e-5, "max abs diff {d} too large");
}

#[test]
fn fifty_step_rollout_stays_close_to_pytorch() {
    let r = reference();
    let model = load_model("config.json");
    let meta = &get(&r, "meta").data;
    let mut state = get(&r, "state").data.clone();
    for i in 1..=50 {
        state = model.step(&state, meta);
        if matches!(i, 1 | 10 | 50) {
            let d = max_abs(&state, &get(&r, &format!("rollout_{i}")).data);
            println!("rollout step {i}: max_abs={d:e}");
            assert!(d < 1e-4, "step {i}: max abs diff {d} from PyTorch");
        }
    }
}

/// GPU-vs-PyTorch parity on the same fixture. Skips cleanly (prints and
/// returns) if no adapter is granted, so it is safe in the default test
/// binary.
#[cfg(feature = "wgpu")]
#[test]
fn gpu_matches_pytorch() {
    let r = reference();
    let cpu_model = load_model("config.json");
    let gpu_model = match pollster::block_on(cpu_model.to_gpu()) {
        Ok(m) => m,
        Err(e) => {
            println!("gpu_matches_pytorch: skipped, no adapter ({e:?})");
            return;
        }
    };
    let meta = &get(&r, "meta").data;
    let state = get(&r, "state").data.clone();

    let out = pollster::block_on(gpu_model.forward(&state, meta));
    let d = max_abs(&out, &get(&r, "forward_stencil").data);
    println!("gpu vs pytorch, 1-step: max_abs={d:e}");
    assert!(d < 1e-4, "GPU forward diverged from PyTorch: max abs diff {d}");

    let mut s = state;
    for _ in 0..50 {
        s = pollster::block_on(gpu_model.step(&s, meta));
    }
    let d50 = max_abs(&s, &get(&r, "rollout_50").data);
    println!("gpu vs pytorch, 50-step rollout: max_abs={d50:e}");
    assert!(d50 < 1e-3, "GPU rollout diverged from PyTorch: max abs diff {d50}");
}

/// GPU parity for the other mask shapes the attention kernel enumerates
/// differently: full attention against PyTorch's saved output, and a dilated
/// radius-2 stencil that clips at the grid edge against the CPU forward pass.
#[cfg(feature = "wgpu")]
#[test]
fn gpu_matches_on_other_masks() {
    let r = reference();
    let meta = &get(&r, "meta").data;
    let state = &get(&r, "state").data;
    let dir = fixture_dir();
    let bytes = std::fs::read(dir.join("model.safetensors")).unwrap();

    let full = load_model("config-full.json");
    let Ok(gpu_full) = pollster::block_on(full.to_gpu()) else {
        println!("gpu_matches_on_other_masks: skipped, no adapter");
        return;
    };
    let out = pollster::block_on(gpu_full.forward(state, meta));
    let d = max_abs(&out, &get(&r, "forward_full").data);
    println!("gpu vs pytorch, 1-step, full: max_abs={d:e}");
    assert!(d < 1e-4, "full: GPU forward diverged from PyTorch: max abs diff {d}");

    let mut cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    cfg["mask"] = serde_json::json!({"Stencil": {"radius": 2, "dilation": 2}});
    let cpu = Model::from_bytes(&bytes, &cfg.to_string());
    let gpu = pollster::block_on(cpu.to_gpu()).unwrap();
    let d = max_abs(&cpu.forward(state, meta), &pollster::block_on(gpu.forward(state, meta)));
    println!("gpu vs cpu, 1-step, stencil r2 d2: max_abs={d:e}");
    assert!(d < 1e-3, "stencil r2 d2: GPU forward diverged from CPU: max abs diff {d}");
}
