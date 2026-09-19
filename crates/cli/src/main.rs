//! Native entry point. The backend is chosen here by feature:
//! `--features wgpu` trains on Metal, the default `ndarray` build is the CPU
//! path used for tests and for running the model while the GPU is busy.

use std::path::{Path, PathBuf};

use burn::module::{AutodiffModule, Module};
use burn::tensor::Tensor;
use clap::{Parser, Subcommand};
use jacobi2000::{energy, Denorm, Jacobi2000Config, Mask};
use rand::{Rng, SeedableRng};
use solvers::{Params, State};
use train::{
    data::GeneratorConfig,
    eval::rollout_errors,
    fit::{fit, FitConfig},
    metrics::{
        energy_drift_rollout, rollout_profiles, sample_single_drop, sample_stress_drop,
        score_rollout, EnergyRow, Metrics, StressKind, BANDS,
    },
    Generator, Meta,
};
use train::checkpoint::{load, save};

#[cfg(feature = "wgpu")]
type Back = burn::backend::Wgpu;
#[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
type Back = burn::backend::NdArray;
type Auto = burn::backend::Autodiff<Back>;

#[cfg(feature = "wgpu")]
const BACKEND: &str = "wgpu (Metal)";
#[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
const BACKEND: &str = "ndarray (CPU)";

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum MaskKind {
    Full,
    Stencil,
}

impl MaskKind {
    fn build(self, radius: usize, dilation: usize) -> Mask {
        match self {
            MaskKind::Full => Mask::Full,
            MaskKind::Stencil => Mask::Stencil { radius, dilation },
        }
    }
}

fn mask_flags(m: Mask) -> String {
    match m {
        Mask::Full => "--mask full".to_string(),
        Mask::Stencil { radius, dilation } => {
            format!("--mask stencil --radius {radius} --dilation {dilation}")
        }
    }
}

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
        #[arg(long, default_value_t = 16)]
        batch: usize,
        /// Live trajectories held at once. 0 = same as the batch size.
        #[arg(long, default_value_t = 0)]
        pool: usize,
        /// Draw the pool from the rung-2 single-drop generator: one drop, no
        /// velocity field, most trajectories short enough to stay
        /// pre-reflection for their whole life.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        single_drop: bool,
        /// Share of single-drop trajectories drawn at the long life.
        #[arg(long, default_value_t = 0.2)]
        long_fraction: f32,
        /// Top of the single-drop amplitude range (log-uniform from 0.02).
        #[arg(long, default_value_t = 0.3)]
        amplitude_max: f32,
        /// Probability of an extra (stacked or elsewhere) drop at a random
        /// later tick, tried up to twice per single-drop trajectory.
        #[arg(long, default_value_t = 0.3)]
        extra_drop_prob: f32,
        /// Fraction of single-drop trajectories exempt from the
        /// pre-reflection margin, so near-wall/corner drops and their
        /// reflections are kept.
        #[arg(long, default_value_t = 0.3)]
        no_horizon_fraction: f32,
        /// Task 8c: fraction of single-drop pool births drawn from the
        /// extended distribution above (amplitude_max, extra_drop_prob,
        /// no_horizon_fraction). The rest draw from the original single-drop
        /// distribution (amplitude 0.02-0.08, no extra drops, margin-kept).
        #[arg(long, default_value_t = 0.0)]
        extended_fraction: f32,
        #[arg(long, default_value_t = 4)]
        rollout: usize,
        #[arg(long, default_value_t = 1000)]
        rollout_after: usize,
        #[arg(long, default_value_t = 0.0)]
        noise: f32,
        /// Cosine floor as a fraction of `lr`. 1.0 = constant learning rate.
        #[arg(long, default_value_t = 0.05)]
        lr_floor: f64,
        /// Project onto constant mass after every predicted step.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        project: bool,
        /// Also project onto the energy inequality (task 11): scale (u, v)
        /// down when a predicted step's energy exceeds the viscosity-scaled
        /// bound. Off by default, alongside the existing mass projection.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        project_energy: bool,
        /// Per-tick energy ratio bound (`E_new <= E_prev * r_max`) for the
        /// energy projection above. Defaults to the measured value from
        /// `energy-stats` (docs/runs/2026-09-19-energy-stats.md) when not
        /// given; falls back to the coded `1 - c*nu*dt` formula only if no
        /// measurement exists for this `nu`.
        #[arg(long)]
        r_max: Option<f32>,
        /// Weight of the wavenumber-banded error term. 0 = plain MSE.
        #[arg(long, default_value_t = 0.0)]
        spectral: f32,
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
        /// Warm-start weights from an existing checkpoint (same architecture
        /// config). Optimizer state and the LR schedule are fresh either way.
        #[arg(long)]
        init: Option<PathBuf>,
        #[arg(long, default_value = "full")]
        mask: MaskKind,
        #[arg(long, default_value_t = 1)]
        radius: usize,
        #[arg(long, default_value_t = 1)]
        dilation: usize,
        /// Task D9: train on SPH-splatted (h, u, v) fields (random dam
        /// breaks and drops) instead of the shallow-water generator.
        /// Overrides the shallow-water-only pool/single-drop flags above;
        /// the post-training solver-rollout eval and run-log write are
        /// skipped (score with `eval-metrics --lagrangian` instead).
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        lagrangian: bool,
        /// Task C5: train the backward stepper. The generator's time axis is
        /// flipped (mask and solver unchanged) so the model learns
        /// state(t+1) -> state(t) instead of state(t) -> state(t+1).
        /// Incompatible with `--lagrangian`.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        reverse: bool,
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
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        project: bool,
        /// Also project onto the energy inequality (task 11).
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        project_energy: bool,
    },
    /// Rung-1 metrics (GOAL-2026-09-18 sec. 1) on fresh single-drop
    /// trajectories, scored only before the first wall reflection.
    /// Generate a pool of fresh single-drop trajectories on the CPU and dump
    /// one record per trajectory.
    GenData {
        /// Trajectories to generate.
        #[arg(long, default_value_t = 2000)]
        pool: usize,
        /// Training batch size the pool will be drawn from. Does not change
        /// how many trajectories are generated.
        #[arg(long, default_value_t = 16)]
        batch: usize,
        #[arg(long, default_value_t = 0.2)]
        long_fraction: f32,
        /// Top of the single-drop amplitude range (log-uniform from 0.02).
        #[arg(long, default_value_t = 0.3)]
        amplitude_max: f32,
        /// Probability of an extra (stacked or elsewhere) drop at a random
        /// later tick, tried up to twice per trajectory.
        #[arg(long, default_value_t = 0.3)]
        extra_drop_prob: f32,
        /// Fraction of trajectories exempt from the pre-reflection margin,
        /// so near-wall/corner drops and their reflections are kept.
        #[arg(long, default_value_t = 0.3)]
        no_horizon_fraction: f32,
        #[arg(long, default_value_t = 1e-3)]
        nu: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        dump: Option<PathBuf>,
        /// Task D9: generate SPH dam-break/drop trajectories instead
        /// (random column widths/heights, drop positions/radii). Overrides
        /// the shallow-water-only flags above.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        lagrangian: bool,
    },
    EvalMetrics {
        #[arg(long, default_value = "artifacts/model")]
        model: PathBuf,
        /// Score the task 8b stress set instead: single high-amplitude,
        /// stacked and corner drops, 8 trajectories each.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        stress: bool,
        #[arg(long, default_value = "10,25,50", value_delimiter = ',')]
        horizons: Vec<usize>,
        #[arg(long, default_value_t = 16)]
        trajectories: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        project: bool,
        /// Also project onto the energy inequality (task 11).
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        project_energy: bool,
        /// xr ensemble at inference (task 10, CONCEPT sec. 4): K jittered
        /// patch-grid offsets, same weights, mean + per-cell spread. K <= 1
        /// disables the ensemble section of the report.
        #[arg(long, default_value_t = 1)]
        xr: usize,
        /// Task C5/C7: score a reverse-trained checkpoint's rewind rollout
        /// against the solver run backward (see `score_rollout_reverse`).
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        reverse: bool,
    },
    /// Native PGM picture sequence (task D9 step 3). Solver-truth frames
    /// only for now (no checkpoint to compare against yet).
    Pictures {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        lagrangian: bool,
        #[arg(long, default_value = "docs/pictures")]
        out_dir: PathBuf,
    },
    /// Measure the solver's own per-tick total-energy ratio `r_t =
    /// E_{t+1}/E_t` on the training distribution (single-drop 70/30
    /// mixture), to set the energy-projection bound `r_max` from data
    /// instead of assuming a formula (docs/runs/2026-09-19-energy-stats.md).
    EnergyStats {
        /// Live trajectories advanced in lockstep.
        #[arg(long, default_value_t = 32)]
        pool: usize,
        /// Ticks to advance the whole pool.
        #[arg(long, default_value_t = 100)]
        ticks: usize,
        #[arg(long, default_value_t = 0.2)]
        long_fraction: f32,
        #[arg(long, default_value_t = 0.3)]
        amplitude_max: f32,
        #[arg(long, default_value_t = 0.3)]
        extra_drop_prob: f32,
        #[arg(long, default_value_t = 0.3)]
        no_horizon_fraction: f32,
        /// 70/30 mixture: 30% of births drawn from the extended
        /// distribution above, matching run 17/18's training mixture.
        #[arg(long, default_value_t = 0.3)]
        extended_fraction: f32,
        #[arg(long, default_value_t = 1e-3)]
        nu: f32,
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
            pool,
            single_drop,
            long_fraction,
            amplitude_max,
            extra_drop_prob,
            no_horizon_fraction,
            extended_fraction,
            rollout,
            rollout_after,
            noise,
            lr_floor,
            project,
            project_energy,
            r_max,
            spectral,
            nu,
            seed,
            layers,
            dim,
            out,
            log_dir,
            note,
            init,
            mask,
            radius,
            dilation,
            lagrangian,
            reverse,
        } => train_cmd(TrainArgs {
            steps,
            lr,
            batch,
            pool,
            single_drop,
            long_fraction,
            amplitude_max,
            extra_drop_prob,
            no_horizon_fraction,
            extended_fraction,
            rollout,
            rollout_after,
            noise,
            lr_floor,
            project,
            project_energy,
            r_max,
            spectral,
            nu,
            seed,
            layers,
            dim,
            out,
            log_dir,
            note,
            init,
            mask: mask.build(radius, dilation),
            lagrangian,
            reverse,
        }),
        Cmd::Eval {
            model,
            trajectories,
            seed,
            project,
            project_energy,
        } => eval_cmd(&model, trajectories, seed, project, project_energy),
        Cmd::GenData {
            pool,
            batch,
            long_fraction,
            amplitude_max,
            extra_drop_prob,
            no_horizon_fraction,
            nu,
            seed,
            dump,
            lagrangian,
        } => {
            if lagrangian {
                gen_data_lagrangian_cmd(pool, batch, nu, seed, dump.as_deref())
            } else {
                gen_data_cmd(
                    pool, batch, long_fraction, amplitude_max, extra_drop_prob,
                    no_horizon_fraction, nu, seed, dump.as_deref(),
                )
            }
        }
        Cmd::EvalMetrics {
            model,
            stress,
            horizons,
            trajectories,
            seed,
            project,
            project_energy,
            xr,
            reverse,
        } => {
            if stress {
                eval_stress_cmd(&model, &horizons, project, project_energy)
            } else {
                eval_metrics_cmd(
                    &model, &horizons, trajectories, seed, project, project_energy, xr, reverse,
                )
            }
        }
        Cmd::Pictures { lagrangian, out_dir } => pictures_cmd(lagrangian, &out_dir),
        Cmd::EnergyStats {
            pool,
            ticks,
            long_fraction,
            amplitude_max,
            extra_drop_prob,
            no_horizon_fraction,
            extended_fraction,
            nu,
            seed,
        } => energy_stats_cmd(
            pool, ticks, long_fraction, amplitude_max, extra_drop_prob, no_horizon_fraction,
            extended_fraction, nu, seed,
        ),
    }
}

struct TrainArgs {
    steps: usize,
    lr: f64,
    batch: usize,
    pool: usize,
    single_drop: bool,
    long_fraction: f32,
    amplitude_max: f32,
    extra_drop_prob: f32,
    no_horizon_fraction: f32,
    extended_fraction: f32,
    rollout: usize,
    rollout_after: usize,
    noise: f32,
    lr_floor: f64,
    project: bool,
    project_energy: bool,
    r_max: Option<f32>,
    spectral: f32,
    nu: f32,
    seed: u64,
    layers: usize,
    dim: usize,
    out: PathBuf,
    log_dir: PathBuf,
    note: Option<String>,
    init: Option<PathBuf>,
    mask: Mask,
    lagrangian: bool,
    reverse: bool,
}

fn train_cmd(a: TrainArgs) {
    let TrainArgs {
        steps,
        lr,
        batch,
        pool,
        single_drop,
        long_fraction,
        amplitude_max,
        extra_drop_prob,
        no_horizon_fraction,
        extended_fraction,
        rollout,
        rollout_after,
        noise,
        lr_floor,
        project,
        project_energy,
        r_max,
        spectral,
        nu,
        seed,
        layers,
        dim,
        out,
        log_dir,
        note,
        init,
        mask,
        lagrangian,
        reverse,
    } = a;
    let (out, log_dir) = (out.as_path(), log_dir.as_path());
    let device = Default::default();
    let params = Params { nu, ..Params::default() };
    let r_max = r_max.unwrap_or_else(|| train::fit::fallback_r_max(params.nu, params.dt));
    let mut gen: Box<dyn train::fit::BatchSource> = if lagrangian {
        let lcfg = train::LagrangianConfig {
            pool: if pool == 0 { batch } else { pool },
            batch,
            seed,
            sph: solvers::sph::Params {
                alpha: nu,
                ..solvers::sph::Params::default()
            },
            ..Default::default()
        };
        let mut g = train::LagrangianGenerator::new(lcfg);
        g.fit_normalizer(50);
        println!("normalizer (lagrangian): mean {:?} std {:?}", g.norm.mean, g.norm.std);
        Box::new(g)
    } else {
        let gcfg = GeneratorConfig {
            params,
            pool: if pool == 0 { batch } else { pool },
            batch,
            single_drop,
            long_fraction,
            amplitude_max,
            extra_drop_prob,
            no_horizon_fraction,
            extended_fraction,
            seed,
            reverse,
            ..Default::default()
        };
        let mut g = Generator::new(gcfg);
        g.fit_normalizer(50);
        println!("normalizer: mean {:?} std {:?}", g.norm.mean, g.norm.std);
        Box::new(g)
    };

    let mcfg = Jacobi2000Config::new()
        .with_nx(params.nx)
        .with_ny(params.ny)
        .with_dim(dim)
        .with_layers(layers);
    let mut model = mcfg.init::<Auto>(&device);
    if let Some(init_dir) = &init {
        let (init_model, init_meta): (jacobi2000::Jacobi2000<Auto>, Meta) =
            load(init_dir, &device);
        assert_eq!(init_meta.nx, mcfg.nx, "--init checkpoint nx mismatch");
        assert_eq!(init_meta.ny, mcfg.ny, "--init checkpoint ny mismatch");
        assert_eq!(init_meta.patch, mcfg.patch, "--init checkpoint patch mismatch");
        assert_eq!(init_meta.channels, mcfg.channels, "--init checkpoint channels mismatch");
        assert_eq!(init_meta.dim, mcfg.dim, "--init checkpoint dim mismatch");
        assert_eq!(init_meta.layers, mcfg.layers, "--init checkpoint layers mismatch");
        assert_eq!(init_meta.heads, mcfg.heads, "--init checkpoint heads mismatch");
        assert_eq!(init_meta.ff_mult, mcfg.ff_mult, "--init checkpoint ff_mult mismatch");
        assert_eq!(init_meta.meta_dim, mcfg.meta_dim, "--init checkpoint meta_dim mismatch");
        model = init_model;
        println!("warm start: {}", init_dir.display());
    }
    println!(
        "backend: {BACKEND}, params: {}, mask: {}",
        model.num_params(),
        mask_flags(mask)
    );

    let fcfg = FitConfig {
        steps,
        lr,
        rollout,
        rollout_after,
        noise,
        lr_floor,
        project,
        project_energy,
        r_max,
        spectral,
        mask,
        ..Default::default()
    };
    let mut rows = Vec::new();
    let model = fit(model, gen.as_mut(), &fcfg, &device, |row| {
        println!("step {:>6}  loss {:.6e}  {:.1}s", row.step, row.loss, row.secs);
        rows.push((row.step, row.loss, row.secs));
    });
    let train_secs = rows.last().map(|r| r.2).unwrap_or(0.0);

    let model = model.valid();
    let tick_params = gen.tick_params();
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
        norm: gen.norm(),
        dx: tick_params.dx,
        dt: tick_params.dt,
        g: tick_params.g,
        depth: tick_params.depth,
        nu: tick_params.nu,
        mask,
        energy_r_max: r_max,
    };
    save(out, &model, &meta);
    println!("wrote {}", out.display());

    if lagrangian {
        println!(
            "lagrangian: skipping solver-rollout eval and run-log write (shallow-water-only); \
             score with `eval-metrics --lagrangian --model {}`",
            out.display()
        );
        return;
    }
    if reverse {
        println!(
            "reverse: skipping the forward-only solver-rollout eval and run-log write; \
             score with `eval-metrics --reverse --model {}`",
            out.display()
        );
        return;
    }

    let errs = eval_model(&model, &meta, 4, seed + 1000, project, project_energy, r_max, &device);
    print_errors(&errs);
    write_run_log(
        log_dir, steps, lr, batch, pool, single_drop, long_fraction, amplitude_max,
        extra_drop_prob, no_horizon_fraction, extended_fraction, rollout, rollout_after, noise,
        lr_floor, project, project_energy, r_max, spectral, nu, seed, layers, dim, &rows,
        train_secs, &errs, note, model.num_params(), init, mask,
    );
}

struct Errors {
    ticks: Vec<usize>,
    model: Vec<f32>,
    persistence: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn eval_model(
    model: &jacobi2000::Jacobi2000<Back>,
    meta: &Meta,
    trajectories: usize,
    seed: u64,
    project: bool,
    project_energy: bool,
    r_max: f32,
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
            meta.mask,
            project,
            project_energy,
            r_max,
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

fn eval_cmd(dir: &Path, trajectories: usize, seed: u64, project: bool, project_energy: bool) {
    let device = <Back as burn::tensor::backend::Backend>::Device::default();
    let (model, meta): (jacobi2000::Jacobi2000<Back>, Meta) = load(dir, &device);
    println!(
        "backend: {BACKEND}, mass projection: {project}, energy projection: {project_energy} (r_max {}), mask: {}",
        meta.energy_r_max,
        mask_flags(meta.mask)
    );
    let e = eval_model(
        &model, &meta, trajectories, seed, project, project_energy, meta.energy_r_max, &device,
    );
    print_errors(&e);
}

/// Mean of the per-trajectory metrics at one horizon.
fn mean_metrics(rows: &[Metrics]) -> Metrics {
    let n = rows.len() as f32;
    let mut bands = [0.0f32; BANDS];
    for (b, out) in bands.iter_mut().enumerate() {
        *out = rows.iter().map(|r| r.bands[b]).sum::<f32>() / n;
    }
    Metrics {
        corr: rows.iter().map(|r| r.corr).sum::<f32>() / n,
        corr_h: rows.iter().map(|r| r.corr_h).sum::<f32>() / n,
        active_rmse: rows.iter().map(|r| r.active_rmse).sum::<f32>() / n,
        amplitude_ratio: rows.iter().map(|r| r.amplitude_ratio).sum::<f32>() / n,
        radius_error: rows.iter().map(|r| r.radius_error).sum::<f32>() / n,
        xcorr_lag: rows.iter().map(|r| r.xcorr_lag).sum::<f32>() / n,
        bands,
        global_rmse: rows.iter().map(|r| r.global_rmse).sum::<f32>() / n,
    }
}

fn metrics_row(label: &str, ticks: usize, m: &Metrics) -> String {
    format!(
        "| {label} | {ticks} | {:.4} | {:.4} | {:.5} | {:.4} | {:+.3} | {:+.3} | {:.5} | {:.5} | {:.5} | {:.5} |",
        m.corr,
        m.corr_h,
        m.active_rmse,
        m.amplitude_ratio,
        m.radius_error,
        m.xcorr_lag,
        m.bands[0],
        m.bands[1],
        m.bands[2],
        m.global_rmse
    )
}

/// Mean, std, min, max and the count with `|x| > 2.0` over a per-trajectory
/// sample. Used for the radius-error and xcorr-lag diagnostics
/// (docs/runs/2026-09-18-radius-estimator.md, task 8a).
fn sample_stats(xs: &[f32]) -> (f32, f32, f32, f32, usize) {
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let min = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let over = xs.iter().filter(|x| x.abs() > 2.0).count();
    (mean, var.sqrt(), min, max, over)
}

#[allow(clippy::too_many_arguments)]
fn eval_metrics_cmd(
    dir: &Path,
    horizons: &[usize],
    trajectories: usize,
    seed: u64,
    project: bool,
    project_energy: bool,
    xr: usize,
    reverse: bool,
) {
    let device = <Back as burn::tensor::backend::Backend>::Device::default();
    let (model, meta): (jacobi2000::Jacobi2000<Back>, Meta) = load(dir, &device);
    let params = meta.params();
    let mut horizons = horizons.to_vec();
    horizons.sort_unstable();
    // Nothing is scored after the wave first touches a wall, and a trajectory
    // whose horizon is under 50 ticks cannot answer the 50-tick question at
    // all, so it is dropped rather than scored on reflections.
    let need = (*horizons.last().unwrap()).max(50) as f32;

    // Radial profiles for the first two trajectories, at 25 and 50 ticks
    // only (task 8a): the numeric table that shows whether the profile is
    // bistable, which the scalar radius error alone cannot.
    let profile_horizons: Vec<usize> = [25usize, 50]
        .into_iter()
        .filter(|h| horizons.contains(h))
        .collect();
    type Profile = (usize, Vec<f64>, Vec<f64>);
    let mut profiles: Vec<Vec<Profile>> = Vec::new();

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut model_rows: Vec<Vec<Metrics>> = vec![Vec::new(); horizons.len()];
    let mut frozen_rows: Vec<Vec<Metrics>> = vec![Vec::new(); horizons.len()];
    // Kept so the xr ensemble (task 10) is scored on the same held-out set
    // as the single-offset baseline above, not a fresh draw.
    let mut trajs: Vec<(State, train::metrics::Drop)> = Vec::new();
    let (mut used, mut skipped) = (0usize, 0usize);
    while used < trajectories {
        let (init, drop) = sample_single_drop(&params, &mut rng);
        if params.horizon_ticks(drop.cx, drop.cy) < need {
            skipped += 1;
            continue;
        }
        let (m, f) = if reverse {
            train::metrics::score_rollout_reverse(
                &model, &params, &meta.norm, &init, &drop, &horizons, meta.mask, project,
                project_energy, meta.energy_r_max, &device,
            )
        } else {
            score_rollout(
                &model, &params, &meta.norm, &init, &drop, &horizons, meta.mask, project,
                project_energy, meta.energy_r_max, &device,
            )
        };
        for i in 0..horizons.len() {
            model_rows[i].push(m[i]);
            frozen_rows[i].push(f[i]);
        }
        if profiles.len() < 2 && !profile_horizons.is_empty() {
            profiles.push(rollout_profiles(
                &model,
                &params,
                &meta.norm,
                &init,
                &drop,
                &profile_horizons,
                meta.mask,
                project,
                &device,
            ));
        }
        trajs.push((init, drop));
        used += 1;
    }

    println!(
        "backend: {BACKEND}, mass projection: {project}, mask: {}",
        mask_flags(meta.mask)
    );
    println!(
        "{used} single-drop trajectories, {skipped} skipped (horizon < {need:.0} ticks)\n"
    );
    println!("| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (i, t) in horizons.iter().enumerate() {
        println!("{}", metrics_row("model", *t, &mean_metrics(&model_rows[i])));
        println!(
            "{}",
            metrics_row("frozen field", *t, &mean_metrics(&frozen_rows[i]))
        );
    }

    println!("\nper-trajectory model radius err / xcorr lag (task 8a, docs/runs/2026-09-18-radius-estimator.md):\n");
    println!("| ticks | radius err mean | std | min | max | \\|err\\|>2 | xcorr lag mean | std |");
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (i, t) in horizons.iter().enumerate() {
        let radius: Vec<f32> = model_rows[i].iter().map(|m| m.radius_error).collect();
        let lag: Vec<f32> = model_rows[i].iter().map(|m| m.xcorr_lag).collect();
        let (rm, rs, rmin, rmax, rover) = sample_stats(&radius);
        let (lm, ls, ..) = sample_stats(&lag);
        println!(
            "| {t} | {rm:+.3} | {rs:.3} | {rmin:+.3} | {rmax:+.3} | {rover} | {lm:+.3} | {ls:.3} |"
        );
    }

    if reverse {
        println!(
            "\nreverse mode: skipping the forward-solver energy-drift, radial-profile and xr \
             sections below (they assume forward semantics); the corr/amp/lag table above is \
             the rewind score."
        );
        return;
    }

    let mut energy_rows: Vec<Vec<EnergyRow>> = vec![Vec::new(); horizons.len()];
    let mut max_violation = 0.0f32;
    for (init, _drop) in &trajs {
        let (rows, violation) = energy_drift_rollout(
            &model, &params, &meta.norm, init, &horizons, meta.mask, meta.energy_r_max, &device,
        );
        for (i, row) in rows.into_iter().enumerate() {
            energy_rows[i].push(row);
        }
        max_violation = max_violation.max(violation);
    }
    println!(
        "\nenergy drift (physical units), model rollout with energy projection ON vs OFF vs \
         the solver, r_max={:.6}, mean over {used} trajectories:\n",
        meta.energy_r_max
    );
    println!("| ticks | energy ON | energy OFF | energy solver |");
    println!("|---:|---:|---:|---:|");
    for (i, t) in horizons.iter().enumerate() {
        let n = energy_rows[i].len() as f32;
        let on: f32 = energy_rows[i].iter().map(|r| r.energy_on).sum::<f32>() / n;
        let off: f32 = energy_rows[i].iter().map(|r| r.energy_off).sum::<f32>() / n;
        let solver: f32 = energy_rows[i].iter().map(|r| r.energy_solver).sum::<f32>() / n;
        println!("| {t} | {on:.6} | {off:.6} | {solver:.6} |");
    }
    println!(
        "\nmax relative energy violation along the projected (ON) rollout: {max_violation:.3e} \
         (bound: <= 1e-6)"
    );

    for (traj_i, traj) in profiles.iter().enumerate() {
        for (t, mprof, tprof) in traj {
            println!("\nradial |h| profile, trajectory {traj_i}, {t} ticks (model vs solver):\n");
            println!("| r (cells) | model | solver |");
            println!("|---:|---:|---:|");
            for (r, (m, s)) in mprof.iter().zip(tprof).enumerate() {
                println!("| {r} | {m:.5} | {s:.5} |");
            }
        }
    }

    if xr > 1 {
        println!("\n## xr ensemble (task 10, CONCEPT sec. 4): K={xr} jittered patch-grid offsets, same weights\n");
        let mut xr_rows: Vec<Vec<Metrics>> = vec![Vec::new(); horizons.len()];
        let mut base_rows: Vec<Vec<Metrics>> = vec![Vec::new(); horizons.len()];
        let mut spread_sums = vec![0.0f64; horizons.len()];
        for (init, drop) in &trajs {
            let (xr_m, base_m, spread) = train::score_rollout_xr(
                &model, &params, &meta.norm, init, drop, &horizons, meta.mask, project, xr,
                &device,
            );
            for i in 0..horizons.len() {
                xr_rows[i].push(xr_m[i]);
                base_rows[i].push(base_m[i]);
                spread_sums[i] += spread[i] as f64;
            }
        }
        println!("| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) |");
        println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for (i, t) in horizons.iter().enumerate() {
            println!("{}", metrics_row("xr ensemble mean", *t, &mean_metrics(&xr_rows[i])));
            println!(
                "{}",
                metrics_row("single-offset baseline", *t, &mean_metrics(&base_rows[i]))
            );
        }
        println!("\n| ticks | mean spread (physical h units) |\n|---:|---:|");
        for (i, t) in horizons.iter().enumerate() {
            println!("| {t} | {:.6} |", spread_sums[i] / trajs.len() as f64);
        }
    }
}

/// Task 8b stress set: single high-amplitude, stacked and corner drops, 8
/// trajectories each, seed 0 fixed per category so the set is reproducible.
fn eval_stress_cmd(dir: &Path, horizons: &[usize], project: bool, project_energy: bool) {
    let device = <Back as burn::tensor::backend::Backend>::Device::default();
    let (model, meta): (jacobi2000::Jacobi2000<Back>, Meta) = load(dir, &device);
    let params = meta.params();
    let mut horizons = horizons.to_vec();
    horizons.sort_unstable();

    println!(
        "backend: {BACKEND}, mass projection: {project}, mask: {}, STRESS SET (task 8b)",
        mask_flags(meta.mask)
    );
    println!(
        "note: corner drops reflect off two walls almost immediately, so the \
         pre-reflection horizon filter used by the standard held-out set is \
         meaningless for them; every corner trajectory is scored at every \
         horizon regardless of reflection.\n"
    );

    for (label, kind) in [
        ("single high-amplitude (0.15-0.3)", StressKind::HighAmp),
        ("stacked (2-3 drops, same spot)", StressKind::Stacked),
        ("corner (within 6 cells of two walls)", StressKind::Corner),
    ] {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut model_rows: Vec<Vec<Metrics>> = vec![Vec::new(); horizons.len()];
        let mut frozen_rows: Vec<Vec<Metrics>> = vec![Vec::new(); horizons.len()];
        for _ in 0..8 {
            let (init, drop) = sample_stress_drop(kind, &params, &mut rng);
            let (m, f) = score_rollout(
                &model, &params, &meta.norm, &init, &drop, &horizons, meta.mask, project,
                project_energy, meta.energy_r_max, &device,
            );
            for i in 0..horizons.len() {
                model_rows[i].push(m[i]);
                frozen_rows[i].push(f[i]);
            }
        }
        println!("### {label} (8 trajectories)\n");
        println!("| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) |");
        println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for (i, t) in horizons.iter().enumerate() {
            println!("{}", metrics_row("model", *t, &mean_metrics(&model_rows[i])));
            println!(
                "{}",
                metrics_row("frozen field", *t, &mean_metrics(&frozen_rows[i]))
            );
        }
        println!();
    }
}

#[allow(clippy::too_many_arguments)]
fn gen_data_cmd(
    pool: usize,
    batch: usize,
    long_fraction: f32,
    amplitude_max: f32,
    extra_drop_prob: f32,
    no_horizon_fraction: f32,
    nu: f32,
    seed: u64,
    dump: Option<&Path>,
) {
    let cfg = GeneratorConfig {
        params: Params { nu, ..Params::default() },
        pool,
        batch,
        single_drop: true,
        long_fraction,
        amplitude_max,
        extra_drop_prob,
        no_horizon_fraction,
        seed,
        ..Default::default()
    };
    let recs = train::data::generate(&cfg, dump);
    let long = recs.iter().filter(|r| r.ticks > cfg.short_ticks.1).count();
    let high_amp = recs.iter().filter(|r| r.amp.abs() > 0.08).count();
    let near_wall = recs.iter().filter(|r| r.no_horizon).count();
    println!(
        "{} trajectories (batch {batch}), {long} long ({} ticks), {} short ({}-{} ticks), {high_amp} amplitude > 0.08, {near_wall} near-wall/corner",
        recs.len(),
        cfg.long_ticks,
        recs.len() - long,
        cfg.short_ticks.0,
        cfg.short_ticks.1
    );
    if let Some(dir) = dump {
        println!("dumped to {}", dir.display());
    }
}

fn gen_data_lagrangian_cmd(pool: usize, batch: usize, nu: f32, seed: u64, dump: Option<&Path>) {
    let cfg = train::LagrangianConfig {
        pool,
        batch,
        seed,
        sph: solvers::sph::Params {
            alpha: nu,
            ..solvers::sph::Params::default()
        },
        ..Default::default()
    };
    let recs = train::generate_lagrangian(&cfg, dump);
    let dam_break = recs.iter().filter(|r| r.dam_break).count();
    println!(
        "{} trajectories (batch {batch}), {dam_break} dam break, {} drop",
        recs.len(),
        recs.len() - dam_break
    );
    if let Some(dir) = dump {
        println!("dumped to {}", dir.display());
    }
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
    pool: usize,
    single_drop: bool,
    long_fraction: f32,
    amplitude_max: f32,
    extra_drop_prob: f32,
    no_horizon_fraction: f32,
    extended_fraction: f32,
    rollout: usize,
    rollout_after: usize,
    noise: f32,
    lr_floor: f64,
    project: bool,
    project_energy: bool,
    r_max: f32,
    spectral: f32,
    nu: f32,
    seed: u64,
    layers: usize,
    dim: usize,
    rows: &[(usize, f32, f64)],
    train_secs: f64,
    e: &Errors,
    note: Option<String>,
    num_params: usize,
    init: Option<PathBuf>,
    mask: Mask,
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
    s.push_str(&format!("- mask: {}\n", mask_flags(mask)));
    let init_flag = init
        .map(|p| format!(" --init {}", p.display()))
        .unwrap_or_default();
    s.push_str(&format!(
        "- command: `cargo run --release -p cli{} -- train --steps {steps} --lr {lr} --batch {batch} --pool {pool} --single-drop {single_drop} --long-fraction {long_fraction} --amplitude-max {amplitude_max} --extra-drop-prob {extra_drop_prob} --no-horizon-fraction {no_horizon_fraction} --extended-fraction {extended_fraction} --rollout {rollout} --rollout-after {rollout_after} --noise {noise} --lr-floor {lr_floor} --project {project} --project-energy {project_energy} --r-max {r_max} --spectral {spectral} --nu {nu} --seed {seed} --layers {layers} --dim {dim} {}{init_flag}`\n",
        if cfg!(feature = "wgpu") { " --features wgpu --no-default-features" } else { "" },
        mask_flags(mask)
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

/// Min, p50, p99, max of a sample. Nearest-rank percentile (no
/// interpolation): good enough for a diagnostic report, and this is the
/// only place that needs it.
fn distribution(xs: &mut [f32]) -> (f32, f32, f32, f32) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |p: f32| xs[((p * (xs.len() - 1) as f32).round() as usize).min(xs.len() - 1)];
    (xs[0], at(0.5), at(0.99), xs[xs.len() - 1])
}

/// Measure the solver's own per-tick total-energy ratio `r_t = E_{t+1}/E_t`
/// on the training distribution (task A1/A2 of
/// convergence/queues/constrained-wider-reversible.md): the coded bound
/// `1 - c*nu*dt` turned out far tighter than the solver's real dissipation
/// (docs/runs/2026-09-19-energy-smoke-2.md), so the energy-projection bound
/// is measured from data instead of assumed.
#[allow(clippy::too_many_arguments)]
fn energy_stats_cmd(
    pool: usize,
    ticks: usize,
    long_fraction: f32,
    amplitude_max: f32,
    extra_drop_prob: f32,
    no_horizon_fraction: f32,
    extended_fraction: f32,
    nu: f32,
    seed: u64,
) {
    let device = <Back as burn::tensor::backend::Backend>::Device::default();
    let params = Params { nu, ..Params::default() };
    let base = GeneratorConfig {
        params,
        pool,
        batch: pool,
        single_drop: true,
        long_fraction,
        amplitude_max,
        extra_drop_prob,
        no_horizon_fraction,
        extended_fraction,
        seed,
        ..Default::default()
    };

    let mut gen = Generator::new(base);
    let denorm = Denorm { mean: gen.norm.mean, std: gen.norm.std, depth: gen.norm.depth };
    let mut all_ratios = Vec::new();
    let mut all_energy = Vec::new();
    for _ in 0..ticks {
        let batch = gen.next_batch(1);
        let b = batch.batch;
        let dims = [b, 3, batch.ny, batch.nx];
        let x0 = Tensor::<Back, 1>::from_floats(batch.states[0].as_slice(), &device).reshape(dims);
        let x1 = Tensor::<Back, 1>::from_floats(batch.states[1].as_slice(), &device).reshape(dims);
        let e0: Vec<f32> = energy(&x0, &denorm, params.g).into_data().to_vec().unwrap();
        let e1: Vec<f32> = energy(&x1, &denorm, params.g).into_data().to_vec().unwrap();
        for i in 0..b {
            all_energy.push(e0[i]);
            if e0[i] > 1e-9 {
                all_ratios.push(e1[i] / e0[i]);
            }
        }
    }

    // The drop's initial transient: fresh (unstaggered) single trajectories,
    // ticks 1..5 from birth, each its own pool-of-1 Generator so its first
    // `next_batch` call really is the drop's first solver step (pool > 1
    // staggers births to decorrelate a training pool, which would mix in
    // already-aged states here).
    let transient_ticks = 5.min(ticks);
    let mut transient = vec![Vec::new(); transient_ticks];
    for i in 0..pool {
        let tcfg = GeneratorConfig {
            params,
            pool: 1,
            batch: 1,
            single_drop: true,
            long_fraction,
            amplitude_max,
            extra_drop_prob,
            no_horizon_fraction,
            extended_fraction,
            seed: seed.wrapping_add(1_000_000 + i as u64),
            ..Default::default()
        };
        let mut tg = Generator::new(tcfg);
        let td = Denorm { mean: tg.norm.mean, std: tg.norm.std, depth: tg.norm.depth };
        for row in transient.iter_mut().take(transient_ticks) {
            let batch = tg.next_batch(1);
            let dims = [1, 3, batch.ny, batch.nx];
            let x0 =
                Tensor::<Back, 1>::from_floats(batch.states[0].as_slice(), &device).reshape(dims);
            let x1 =
                Tensor::<Back, 1>::from_floats(batch.states[1].as_slice(), &device).reshape(dims);
            let e0: f32 = energy(&x0, &td, params.g).into_scalar();
            let e1: f32 = energy(&x1, &td, params.g).into_scalar();
            if e0 > 1e-9 {
                row.push(e1 / e0);
            }
        }
    }

    println!(
        "backend: {BACKEND}, pool {pool}, ticks {ticks}, nu {nu}, 70/30 mixture (extended_fraction {extended_fraction})\n"
    );
    let (rmin, rp50, rp99, rmax) = distribution(&mut all_ratios);
    println!("all-tick ratio r_t = E_{{t+1}}/E_t, {} samples:", all_ratios.len());
    println!("| min | p50 | p99 | max |\n|---:|---:|---:|---:|");
    println!("| {rmin:.6} | {rp50:.6} | {rp99:.6} | {rmax:.6} |\n");

    let (emin, ep50, ep99, emax) = distribution(&mut all_energy);
    println!("E (physical units), {} samples:", all_energy.len());
    println!("| min | p50 | p99 | max |\n|---:|---:|---:|---:|");
    println!("| {emin:.6} | {ep50:.6} | {ep99:.6} | {emax:.6} |\n");

    println!("initial transient (fresh drop, ticks 1..{transient_ticks}):");
    println!("| tick | min | p50 | p99 | max | n |\n|---:|---:|---:|---:|---:|---:|");
    for (k, row) in transient.iter_mut().enumerate() {
        let (min, p50, p99, max) = distribution(row);
        println!("| {} | {min:.6} | {p50:.6} | {p99:.6} | {max:.6} | {} |", k + 1, row.len());
    }

    let r_max = rmax * (1.0 + 1e-4);
    println!("\nsuggested --r-max (measured max * (1 + 1e-4)): {r_max:.6}");
}

/// Write an 8-bit grayscale PGM (binary P5), min-max normalized per frame.
/// Row `y=0` is written last (bottom of the image), matching the box
/// convention (`y=0` is the floor).
fn write_pgm(path: &Path, field: &[f32], nx: usize, ny: usize) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in field {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let range = (hi - lo).max(1e-6);
    let mut bytes = Vec::with_capacity(nx * ny);
    for y in (0..ny).rev() {
        for x in 0..nx {
            let v = field[y * nx + x];
            bytes.push((((v - lo) / range) * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    let mut out = format!("P5\n{nx} {ny}\n255\n").into_bytes();
    out.extend_from_slice(&bytes);
    std::fs::write(path, out).unwrap();
}

/// Task D9 step 3: SPH dam-break solver-truth PGM frames at t = 0, 10, 25
/// (model ticks, `train::lagrangian::SPH_STEPS_PER_TICK` SPH steps each).
fn pictures_cmd(lagrangian: bool, out_dir: &Path) {
    if !lagrangian {
        println!("pictures: only --lagrangian is implemented so far");
        return;
    }
    std::fs::create_dir_all(out_dir).unwrap();
    let sph_params = solvers::sph::Params::default();
    let mut sph = solvers::sph::Sph::new(sph_params, solvers::sph::Initial::DamBreak);
    let rho0 = sph_params.rest_density;
    let mut written = 0;
    for &tick in &[0usize, 10, 25] {
        while written < tick {
            sph.steps(train::lagrangian::SPH_STEPS_PER_TICK);
            written += 1;
        }
        let field = solvers::splat::splat(
            &sph, sph_params.nx, sph_params.ny, sph_params.dx, sph_params.h, rho0,
        );
        let path = out_dir.join(format!("sph-dambreak-solver-t{tick}.pgm"));
        write_pgm(&path, &field.h, sph_params.nx, sph_params.ny);
        println!("wrote {}", path.display());
    }
}
