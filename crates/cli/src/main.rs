//! Native entry point. The backend is chosen here by feature:
//! `--features wgpu` trains on Metal, the default `ndarray` build is the CPU
//! path used for tests and for running the model while the GPU is busy.

use std::path::{Path, PathBuf};

use burn::module::{AutodiffModule, Module};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use clap::{Parser, Subcommand};
use jacobi2000::{Jacobi2000Config, Mask};
use rand::{Rng, SeedableRng};
use solvers::{Params, State};
use train::{
    data::GeneratorConfig,
    eval::rollout_errors,
    fit::{fit, FitConfig},
    Generator, Meta,
};

#[cfg(feature = "wgpu")]
type Back = burn::backend::Wgpu;
#[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
type Back = burn::backend::NdArray;
type Auto = burn::backend::Autodiff<Back>;

#[cfg(feature = "wgpu")]
const BACKEND: &str = "wgpu (Metal)";
#[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
const BACKEND: &str = "ndarray (CPU)";

#[derive(Parser)]
#[command(about = "jacobi2000: shallow water, solver and learned operator")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Train the operator and write a checkpoint plus a run log.
    Train {
        #[arg(long, default_value_t = 3000)]
        steps: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
        /// Live trajectories in the pool = batch size.
        #[arg(long, default_value_t = 16)]
        batch: usize,
        #[arg(long, default_value_t = 4)]
        rollout: usize,
        #[arg(long, default_value_t = 1000)]
        rollout_after: usize,
        #[arg(long, default_value_t = 0.0)]
        noise: f32,
        #[arg(long, default_value_t = 1e-3)]
        nu: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 4)]
        layers: usize,
        #[arg(long, default_value_t = 128)]
        dim: usize,
        #[arg(long, default_value = "artifacts/model")]
        out: PathBuf,
        #[arg(long, default_value = "docs/runs")]
        log_dir: PathBuf,
        #[arg(long)]
        note: Option<String>,
    },
    /// Rollout error at 10/50/200 ticks against the solver, with the
    /// frozen-field baseline.
    Eval {
        #[arg(long, default_value = "artifacts/model")]
        model: PathBuf,
        #[arg(long, default_value_t = 4)]
        trajectories: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
}

fn main() {
    run(Cli::parse().cmd);
    // The wgpu runtime keeps a live client after the last tensor is dropped
    // and the process then sits at 0% CPU forever. Everything a run produces
    // (checkpoint, run log) is on disk by the time `run` returns, so leave.
    std::process::exit(0);
}

fn run(cmd: Cmd) {
    match cmd {
        Cmd::Train {
            steps,
            lr,
            batch,
            rollout,
            rollout_after,
            noise,
            nu,
            seed,
            layers,
            dim,
            out,
            log_dir,
            note,
        } => train_cmd(
            steps,
            lr,
            batch,
            rollout,
            rollout_after,
            noise,
            nu,
            seed,
            layers,
            dim,
            &out,
            &log_dir,
            note,
        ),
        Cmd::Eval {
            model,
            trajectories,
            seed,
        } => eval_cmd(&model, trajectories, seed),
    }
}

#[allow(clippy::too_many_arguments)]
fn train_cmd(
    steps: usize,
    lr: f64,
    batch: usize,
    rollout: usize,
    rollout_after: usize,
    noise: f32,
    nu: f32,
    seed: u64,
    layers: usize,
    dim: usize,
    out: &Path,
    log_dir: &Path,
    note: Option<String>,
) {
    let device = Default::default();
    let params = Params { nu, ..Params::default() };
    let gcfg = GeneratorConfig {
        params,
        pool: batch,
        seed,
        ..Default::default()
    };
    let mut gen = Generator::new(gcfg);
    gen.fit_normalizer(50);
    println!(
        "normalizer: mean {:?} std {:?}",
        gen.norm.mean, gen.norm.std
    );

    let mcfg = Jacobi2000Config::new()
        .with_nx(params.nx)
        .with_ny(params.ny)
        .with_dim(dim)
        .with_layers(layers);
    let model = mcfg.init::<Auto>(&device);
    println!("backend: {BACKEND}, params: {}", model.num_params());

    let fcfg = FitConfig {
        steps,
        lr,
        rollout,
        rollout_after,
        noise,
        mask: Mask::Full,
        ..Default::default()
    };
    let mut rows = Vec::new();
    let model = fit(model, &mut gen, &fcfg, &device, |row| {
        println!("step {:>6}  loss {:.6e}  {:.1}s", row.step, row.loss, row.secs);
        rows.push((row.step, row.loss, row.secs));
    });
    let train_secs = rows.last().map(|r| r.2).unwrap_or(0.0);

    let model = model.valid();
    let meta = Meta {
        nx: params.nx,
        ny: params.ny,
        patch: 4,
        channels: 3,
        dim,
        layers,
        heads: 4,
        ff_mult: 4,
        meta_dim: train::META_DIM,
        norm: gen.norm,
        dx: params.dx,
        dt: params.dt,
        g: params.g,
        depth: params.depth,
        nu: params.nu,
    };
    save(out, &model, &meta);
    println!("wrote {}", out.display());

    let errs = eval_model(&model, &meta, 4, seed + 1000, &device);
    print_errors(&errs);
    write_run_log(
        log_dir, steps, lr, batch, rollout, rollout_after, noise, nu, seed, layers, dim, &rows,
        train_secs, &errs, note, model.num_params(),
    );
}

fn save<B: burn::tensor::backend::Backend>(
    out: &Path,
    model: &jacobi2000::Jacobi2000<B>,
    meta: &Meta,
) {
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

fn load(dir: &Path, device: &<Back as burn::tensor::backend::Backend>::Device) -> (jacobi2000::Jacobi2000<Back>, Meta) {
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
    (cfg.init::<Back>(device).load_record(record), meta)
}

struct Errors {
    ticks: Vec<usize>,
    model: Vec<f32>,
    persistence: Vec<f32>,
}

fn eval_model(
    model: &jacobi2000::Jacobi2000<Back>,
    meta: &Meta,
    trajectories: usize,
    seed: u64,
    device: &<Back as burn::tensor::backend::Backend>::Device,
) -> Errors {
    let ticks = vec![10usize, 50, 200];
    let params = meta.params();
    let mut acc_m = vec![0.0f32; ticks.len()];
    let mut acc_p = vec![0.0f32; ticks.len()];
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for _ in 0..trajectories {
        let init = random_init(&params, &mut rng);
        let e = rollout_errors(
            model,
            &params,
            &meta.norm,
            &init,
            &ticks,
            Mask::Full,
            device,
        );
        for i in 0..ticks.len() {
            acc_m[i] += e.model[i];
            acc_p[i] += e.persistence[i];
        }
    }
    let n = trajectories as f32;
    Errors {
        ticks,
        model: acc_m.iter().map(|x| x / n).collect(),
        persistence: acc_p.iter().map(|x| x / n).collect(),
    }
}

fn random_init(p: &Params, rng: &mut rand::rngs::StdRng) -> State {
    let mut s = State::flat(p);
    let drops = rng.gen_range(1..=3);
    for _ in 0..drops {
        s.add_drop(
            p,
            rng.gen_range(8.0..(p.nx as f32 - 8.0)),
            rng.gen_range(8.0..(p.ny as f32 - 8.0)),
            rng.gen_range(0.03..0.07),
            rng.gen_range(3.0..5.5),
        );
    }
    s
}

fn print_errors(e: &Errors) {
    println!("\n  ticks |    model |  frozen field");
    for i in 0..e.ticks.len() {
        println!(
            "  {:>5} | {:.5} | {:.5}",
            e.ticks[i], e.model[i], e.persistence[i]
        );
    }
}

fn eval_cmd(dir: &Path, trajectories: usize, seed: u64) {
    let device = Default::default();
    let (model, meta) = load(dir, &device);
    println!("backend: {BACKEND}");
    let e = eval_model(&model, &meta, trajectories, seed, &device);
    print_errors(&e);
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
fn write_run_log(
    dir: &Path,
    steps: usize,
    lr: f64,
    batch: usize,
    rollout: usize,
    rollout_after: usize,
    noise: f32,
    nu: f32,
    seed: u64,
    layers: usize,
    dim: usize,
    rows: &[(usize, f32, f64)],
    train_secs: f64,
    e: &Errors,
    note: Option<String>,
    num_params: usize,
) {
    std::fs::create_dir_all(dir).unwrap();
    let date = std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let mut n = 1;
    while dir.join(format!("{date}-{n}.md")).exists() {
        n += 1;
    }
    let path = dir.join(format!("{date}-{n}.md"));
    let mut s = String::new();
    s.push_str(&format!("# Run {date}-{n}\n\n"));
    s.push_str("Machine: Apple M2, 16 GB. Timings **provisional** (the GPU may be shared).\n\n");
    s.push_str(&format!("- commit: `{}`\n", git_commit()));
    s.push_str(&format!("- backend: {BACKEND}\n"));
    s.push_str(&format!(
        "- command: `cargo run --release -p cli{} -- train --steps {steps} --lr {lr} --batch {batch} --rollout {rollout} --rollout-after {rollout_after} --noise {noise} --nu {nu} --seed {seed} --layers {layers} --dim {dim}`\n",
        if cfg!(feature = "wgpu") { " --features wgpu --no-default-features" } else { "" }
    ));
    s.push_str(&format!("- parameters: {num_params}\n"));
    s.push_str(&format!("- wall clock: {train_secs:.1} s\n"));
    if let Some(note) = note {
        s.push_str(&format!("- note: {note}\n"));
    }
    s.push_str("\n## Loss\n\n| step | loss | s |\n|---:|---:|---:|\n");
    for (step, loss, secs) in rows {
        s.push_str(&format!("| {step} | {loss:.6e} | {secs:.1} |\n"));
    }
    s.push_str("\n## Rollout error (RMSE, normalized units, 4 held-out trajectories)\n\n");
    s.push_str("| ticks | model | frozen field |\n|---:|---:|---:|\n");
    for i in 0..e.ticks.len() {
        s.push_str(&format!(
            "| {} | {:.5} | {:.5} |\n",
            e.ticks[i], e.model[i], e.persistence[i]
        ));
    }
    std::fs::write(&path, s).unwrap();
    println!("run log: {}", path.display());
}
