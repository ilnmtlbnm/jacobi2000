//! `eval-metrics`: score a checkpoint's rollouts against the reference solver
//! on fresh single-drop trajectories, at several horizons.
//!
//! ```text
//! cargo run -p engine-cli --release --bin eval-metrics -- \
//!     --model <checkpoint-dir> [--horizons 10,25,50] [--trajectories 16] [--seed 0] \
//!     [--project true|false] [--project-energy true|false] [--backend auto|cpu|webgpu]
//! ```
//!
//! Each trajectory is one drop, placed so that the wave has not yet met a
//! wall at the last horizon (or 50 ticks, whichever is later); trajectories
//! that cannot answer that are redrawn, so nothing is scored on reflections.
//! The model is stepped from the same initial state as the solver, with the
//! mass projection and, optionally, the energy projection applied after each
//! step. Reports, as the mean over trajectories at each horizon: Pearson
//! correlation over (h, u, v) and over h, amplitude ratio (model peak |h| over
//! solver peak |h|), wavefront lag in cells (cross-correlation of the radial
//! |h| profiles) and bias. With the energy projection on, also the largest
//! relative energy violation along the rollout against the bound
//! `energy_r_max` from `config.json`.

use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod backend;
use backend::pick_engine;

use engine::metrics::{bias, correlation, decode_h, peak, radial_profile, xcorr_lag};
use engine::{meta_of, project, Model};
use solvers::{ShallowWater, State};

struct Args {
    model: PathBuf,
    horizons: Vec<usize>,
    trajectories: usize,
    seed: u64,
    project: bool,
    project_energy: bool,
    backend: String,
}

fn parse_bool(s: &str) -> bool {
    match s {
        "true" => true,
        "false" => false,
        other => {
            eprintln!("expected true or false, got `{other}`");
            std::process::exit(2);
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args {
        model: PathBuf::new(),
        horizons: vec![10, 25, 50],
        trajectories: 16,
        seed: 0,
        project: true,
        project_energy: false,
        backend: "auto".to_string(),
    };
    let mut have_model = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| -> String {
            it.next().unwrap_or_else(|| {
                eprintln!("{name} needs a value");
                std::process::exit(2);
            })
        };
        match arg.as_str() {
            "--model" => {
                a.model = PathBuf::from(value("--model"));
                have_model = true;
            }
            "--horizons" => {
                a.horizons = value("--horizons").split(',').map(|s| s.trim().parse().expect("--horizons")).collect()
            }
            "--trajectories" => a.trajectories = value("--trajectories").parse().expect("--trajectories"),
            "--seed" => a.seed = value("--seed").parse().expect("--seed"),
            "--project" => a.project = parse_bool(&value("--project")),
            "--project-energy" => a.project_energy = parse_bool(&value("--project-energy")),
            "--backend" => a.backend = value("--backend"),
            other => {
                eprintln!("unrecognized argument `{other}`");
                std::process::exit(2);
            }
        }
    }
    if !have_model || a.horizons.is_empty() {
        eprintln!("usage: eval-metrics --model <checkpoint-dir> [--horizons 10,25,50] [--trajectories N] [--seed S] [--project true|false] [--project-energy true|false] [--backend auto|cpu|webgpu]");
        std::process::exit(2);
    }
    a.horizons.sort_unstable();
    a
}

#[derive(Clone, Copy, Default)]
struct Row {
    corr: f32,
    corr_h: f32,
    amplitude_ratio: f32,
    lag: f32,
    bias: f32,
}

fn main() {
    let args = parse_args();
    let model = Model::load(&args.model);
    let cfg = model.cfg.clone();
    let params = cfg.params();
    let engine = pick_engine(model, &args.backend);
    let n = params.cells();
    let meta = meta_of(&params).to_vec();
    let denorm = cfg.norm.denorm();
    let last = *args.horizons.last().unwrap();
    let need = last.max(50) as f32;

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut rows: Vec<Vec<Row>> = vec![Vec::new(); args.horizons.len()];
    let mut max_violation = 0.0f32;
    let (mut used, mut skipped) = (0usize, 0usize);
    while used < args.trajectories {
        let cx = rng.gen_range(8.0..(params.nx as f32 - 8.0));
        let cy = rng.gen_range(8.0..(params.ny as f32 - 8.0));
        let amp = rng.gen_range(0.03..0.07);
        let sigma = rng.gen_range(3.0..5.5);
        if params.horizon_ticks(cx, cy) < need {
            skipped += 1;
            continue;
        }
        let mut truth = State::flat(&params);
        truth.add_drop(&params, cx, cy, amp, sigma);
        let mut solver = ShallowWater::new(params);
        let mut x = vec![0.0f32; 3 * n];
        cfg.norm.encode(&truth, &mut x);
        let mass0 = project::mass(&x, n);
        let mut energy = project::energy(&x, &denorm, params.g, n);
        let mut target = vec![0.0f32; 3 * n];

        for t in 1..=last {
            solver.step(&mut truth);
            let prev_energy = energy;
            let mut next = engine.step(&x, &meta);
            if args.project {
                project::project_mass(&mut next, mass0, n);
            }
            if args.project_energy {
                project::project_energy(&mut next, prev_energy, &denorm, params.g, cfg.energy_r_max, n);
            }
            x = next;
            energy = project::energy(&x, &denorm, params.g, n);
            max_violation = max_violation.max(((energy - prev_energy * cfg.energy_r_max) / prev_energy).max(0.0));

            if let Some(i) = args.horizons.iter().position(|&h| h == t) {
                cfg.norm.encode(&truth, &mut target);
                let (ph, th) = (decode_h(&x, &cfg.norm, n), decode_h(&target, &cfg.norm, n));
                let (mprof, tprof) = (
                    radial_profile(&ph, params.nx, params.ny, cx, cy),
                    radial_profile(&th, params.nx, params.ny, cx, cy),
                );
                rows[i].push(Row {
                    corr: correlation(&x, &target),
                    corr_h: correlation(&x[..n], &target[..n]),
                    amplitude_ratio: peak(&ph) / peak(&th),
                    lag: xcorr_lag(&mprof, &tprof, 8),
                    bias: bias(&ph, &th),
                });
            }
        }
        used += 1;
    }

    println!(
        "backend: {}, mass projection: {}, energy projection: {} (r_max {}), mask: {:?}",
        engine.label(),
        args.project,
        args.project_energy,
        cfg.energy_r_max,
        cfg.mask
    );
    println!("{used} single-drop trajectories, {skipped} skipped (horizon < {need:.0} ticks)\n");
    println!("| ticks | corr (h,u,v) | corr (h) | amplitude ratio | wavefront lag (cells) | bias |");
    println!("|---:|---:|---:|---:|---:|---:|");
    for (t, r) in args.horizons.iter().zip(&rows) {
        let m = |f: fn(&Row) -> f32| r.iter().map(f).sum::<f32>() / r.len() as f32;
        println!(
            "| {t} | {:.4} | {:.4} | {:.4} | {:+.3} | {:+.3} |",
            m(|r| r.corr),
            m(|r| r.corr_h),
            m(|r| r.amplitude_ratio),
            m(|r| r.lag),
            m(|r| r.bias)
        );
    }
    if args.project_energy {
        println!("\nmax relative energy violation along the projected rollout: {max_violation:.3e} (bound: <= 1e-6)");
    }
}
