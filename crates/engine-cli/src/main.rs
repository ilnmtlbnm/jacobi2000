//! `rollout`: run the `engine` crate from the command line.
//!
//! ```text
//! cargo run -p engine-cli --release --bin rollout -- \
//!     --model web/model/model20 --steps 50 [--backend auto|cpu|webgpu] [--case drop|dam_break] [--out field.npy]
//! ```
//!
//! `--model` is a checkpoint directory holding `model.safetensors` +
//! `config.json`, the format the PyTorch export writes
//! (`training/jacobi2000_pt/export.py`, docs/pytorch-training.md).
//!
//! Steps the reference solver (`solvers::ShallowWater`) alongside the
//! model from the same initial condition and prints the model's
//! correlation against it every step — the same quantity
//! `crates/web`'s compare-mode panel shows live, here as a one-shot CLI
//! number instead of a page.

use std::path::PathBuf;

mod backend;
use backend::pick_engine;

use engine::metrics::{correlation, decode_h};
use engine::{project, Model};
use solvers::{Params, ShallowWater, State};

struct Args {
    model: PathBuf,
    steps: usize,
    backend: String,
    case: String,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut model = None;
    let mut steps = 50usize;
    let mut backend = "auto".to_string();
    let mut case = "drop".to_string();
    let mut out = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = it.next().map(PathBuf::from),
            "--steps" => steps = it.next().and_then(|s| s.parse().ok()).unwrap_or(steps),
            "--backend" => backend = it.next().unwrap_or(backend),
            "--case" => case = it.next().unwrap_or(case),
            "--out" => out = it.next().map(PathBuf::from),
            other => {
                eprintln!("unrecognized argument `{other}`");
                std::process::exit(2);
            }
        }
    }
    let Some(model) = model else {
        eprintln!("usage: rollout --model <checkpoint-dir> [--steps N] [--backend auto|cpu|webgpu] [--case drop|dam_break] [--out field.npy]");
        std::process::exit(2);
    };
    Args { model, steps, backend, case, out }
}

/// The same initial condition `crates/web`'s case picker builds.
fn apply_case(case: &str, truth: &mut State, p: &Params) {
    match case {
        "drop" => truth.add_drop(p, p.nx as f32 / 2.0, p.ny as f32 / 2.0, 0.05, 4.0),
        "dam_break" => {
            for y in 0..p.ny {
                for x in 0..(p.nx / 2) {
                    truth.h[y * p.nx + x] = p.depth + 0.15;
                }
            }
        }
        other => {
            eprintln!("unrecognized --case `{other}` (want drop | dam_break)");
            std::process::exit(2);
        }
    }
}

/// Minimal NPY v1.0 writer (f32, 2-D, C order) — no dependency beyond std.
fn write_npy(path: &std::path::Path, data: &[f32], shape: (usize, usize)) -> std::io::Result<()> {
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}", shape.0, shape.1);
    let prefix_len = 10; // magic(6) + version(2) + header_len(2)
    let pad = (64 - (prefix_len + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');

    let mut bytes = Vec::with_capacity(prefix_len + header.len() + data.len() * 4);
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.extend_from_slice(&[1u8, 0u8]);
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

fn main() {
    let args = parse_args();
    let model = Model::load(&args.model);
    let cfg = model.cfg.clone();
    let params = cfg.params();
    let engine = pick_engine(model, &args.backend);
    println!("loaded {} ({}x{}, dim {}, {} layers, mask {:?}) on {}", args.model.display(), cfg.nx, cfg.ny, cfg.dim, cfg.layers, cfg.mask, engine.label());

    let mut truth = State::flat(&params);
    apply_case(&args.case, &mut truth, &params);
    let meta = engine::meta_of(&params).to_vec();
    let n = params.cells();
    let mut x = vec![0.0f32; 3 * n];
    cfg.norm.encode(&truth, &mut x);
    let mass0 = project::mass(&x, n);

    let mut solver = ShallowWater::new(params);
    let t0 = std::time::Instant::now();
    for step in 1..=args.steps {
        solver.step(&mut truth);
        let mut next = engine.step(&x, &meta);
        project::project_mass(&mut next, mass0, n);
        x = next;

        let solver_h: Vec<f32> = truth.h.iter().map(|&h| h - params.depth).collect();
        let model_h = decode_h(&x, &cfg.norm, n);
        let corr = correlation(&model_h, &solver_h);
        println!("step {step:4} corr={corr:.5}");
    }
    let elapsed = t0.elapsed();
    println!(
        "{} steps in {:.1} ms ({:.2} ms/step, {})",
        args.steps,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / args.steps.max(1) as f64,
        engine.label()
    );

    if let Some(out) = args.out {
        let model_h = decode_h(&x, &cfg.norm, n);
        write_npy(&out, &model_h, (params.ny, params.nx)).expect("writing .npy failed");
        println!("wrote {}", out.display());
    }
}
