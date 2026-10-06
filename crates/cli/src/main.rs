//! Native entry point. The backend is chosen here by feature:
//! `--features wgpu` trains on Metal, the default `ndarray` build is the CPU
//! path used for tests and for running the model while the GPU is busy.

use std::path::{Path, PathBuf};

use burn::module::{AutodiffModule, Module};
use burn::tensor::Tensor;
use clap::{Parser, Subcommand};
use jacobi2000::{energy, excess_energy, EnergyBand, read_safetensors, write_safetensors, Denorm, Jacobi2000Config, Mask};
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
        /// Excess-energy inequality: project every predicted step onto
        /// `E'_new <= E'_prev * r` (`jacobi2000::project_excess_energy`),
        /// after the mass projection. `r` from `energy-stats` (suggested
        /// `--project-excess`). Off when not given. Forward shallow water
        /// only (not with `--reverse` or `--lagrangian`).
        #[arg(long)]
        project_excess: Option<f32>,
        /// Lower edge of the excess-energy band, `E'_new >= E'_prev * r_min`
        /// (catches energy loss too). 0 = upper-only form. Needs
        /// `--project-excess`.
        #[arg(long, default_value_t = 0.0)]
        excess_r_min: f32,
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
        /// Excess-energy band upper ratio `r_max`. Defaults to the
        /// checkpoint's own (`excess_r` in meta.json, off if absent); `0`
        /// turns the projection off.
        #[arg(long)]
        project_excess: Option<f32>,
        /// Excess-energy band lower ratio `r_min`. Defaults to the
        /// checkpoint's own (`excess_r_min`, 0 if absent); 0 = upper only.
        #[arg(long)]
        excess_r_min: Option<f32>,
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
    /// Dump full single-drop trajectories (raw h, u, v per tick, not just
    /// the `Record` metadata `gen-data` writes) to `--out`, for the PyTorch
    /// training side (docs/pytorch-training.md). One-time call: the files
    /// this writes are the dataset, not regenerated per training run.
    DumpDataset {
        #[arg(long, default_value_t = 2000)]
        pool: usize,
        #[arg(long, default_value_t = 0.2)]
        long_fraction: f32,
        #[arg(long, default_value_t = 0.3)]
        amplitude_max: f32,
        #[arg(long, default_value_t = 0.3)]
        extra_drop_prob: f32,
        #[arg(long, default_value_t = 0.3)]
        no_horizon_fraction: f32,
        #[arg(long, default_value_t = 0.3)]
        extended_fraction: f32,
        #[arg(long, default_value_t = 1e-3)]
        nu: f32,
        #[arg(long, default_value_t = 20)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Export a Burn checkpoint's weights to safetensors + config.json, for a
    /// PyTorch-side parity check (docs/pytorch-training.md).
    ExportWeights {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Import safetensors + config.json (written by the PyTorch export, or
    /// by `export-weights`) into a Burn checkpoint the Rust CLI and `web/`
    /// can load unchanged.
    ImportWeights {
        #[arg(long)]
        safetensors: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Run one forward pass on a fixed, deterministic synthetic input and
    /// dump the raw output (`3*ny*nx` little-endian f32) to `--out`, for a
    /// cross-language parity check against the same input run through the
    /// PyTorch model (docs/pytorch-training.md).
    ForwardProbe {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
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
        /// Excess-energy band upper ratio `r_max`. Defaults to the
        /// checkpoint's own (`excess_r` in meta.json, off if absent); `0`
        /// turns the projection off.
        #[arg(long)]
        project_excess: Option<f32>,
        /// Excess-energy band lower ratio `r_min`. Defaults to the
        /// checkpoint's own (`excess_r_min`, 0 if absent); 0 = upper only.
        #[arg(long)]
        excess_r_min: Option<f32>,
        /// xr ensemble at inference (task 10, CONCEPT sec. 4): K jittered
        /// patch-grid offsets, same weights, mean + per-cell spread. K <= 1
        /// disables the ensemble section of the report.
        #[arg(long, default_value_t = 1)]
        xr: usize,
        /// Task C5/C7: score a reverse-trained checkpoint's rewind rollout
        /// against the solver run backward (see `score_rollout_reverse`).
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        reverse: bool,
        /// Task D9: score a `--lagrangian`-trained checkpoint against
        /// SPH-splatted truth (dam break + drop trajectories) instead of
        /// the shallow-water solver.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        lagrangian: bool,
    },
    /// Native PGM picture sequence (task D9 step 3). Solver-truth frames
    /// only for now (no checkpoint to compare against yet).
    Pictures {
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        lagrangian: bool,
        #[arg(long, default_value = "docs/pictures")]
        out_dir: PathBuf,
        /// Task D9: also roll a `--lagrangian`-trained checkpoint forward
        /// alongside the SPH solver and write its own frame sequence.
        #[arg(long)]
        model: Option<PathBuf>,
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
    /// Task C6: forward-then-backward round trip. Step `forward` N ticks,
    /// then `reverse` N ticks back, and score against the original state
    /// (see `metrics::round_trip`).
    RoundTrip {
        #[arg(long)]
        forward: PathBuf,
        #[arg(long)]
        reverse: PathBuf,
        #[arg(long, default_value = "10,25,50", value_delimiter = ',')]
        horizons: Vec<usize>,
        #[arg(long, default_value_t = 8)]
        trajectories: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 1e-3)]
        nu: f32,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        project: bool,
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
            project_excess,
            excess_r_min,
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
            project_excess: project_excess.map(|r_max| EnergyBand { r_min: excess_r_min, r_max }),
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
            project_excess,
            excess_r_min,
        } => eval_cmd(
            &model, trajectories, seed, project, project_energy, project_excess, excess_r_min,
        ),
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
        Cmd::DumpDataset {
            pool,
            long_fraction,
            amplitude_max,
            extra_drop_prob,
            no_horizon_fraction,
            extended_fraction,
            nu,
            seed,
            out,
        } => dump_dataset_cmd(
            pool, long_fraction, amplitude_max, extra_drop_prob,
            no_horizon_fraction, extended_fraction, nu, seed, &out,
        ),
        Cmd::ExportWeights { model, out } => export_weights_cmd(&model, &out),
        Cmd::ImportWeights { safetensors, config, out } => {
            import_weights_cmd(&safetensors, &config, &out)
        }
        Cmd::ForwardProbe { model, out } => forward_probe_cmd(&model, &out),
        Cmd::EvalMetrics {
            model,
            stress,
            horizons,
            trajectories,
            seed,
            project,
            project_energy,
            project_excess,
            excess_r_min,
            xr,
            reverse,
            lagrangian,
        } => {
            if lagrangian {
                eval_lagrangian_cmd(&model, seed)
            } else if stress {
                eval_stress_cmd(
                    &model, &horizons, project, project_energy, (project_excess, excess_r_min),
                )
            } else {
                eval_metrics_cmd(
                    &model, &horizons, trajectories, seed, project, project_energy,
                    (project_excess, excess_r_min), xr, reverse,
                )
            }
        }
        Cmd::Pictures { lagrangian, out_dir, model } => pictures_cmd(lagrangian, &out_dir, model.as_deref()),
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
        Cmd::RoundTrip { forward, reverse, horizons, trajectories, seed, nu, project } => {
            round_trip_cmd(&forward, &reverse, &horizons, trajectories, seed, nu, project)
        }
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
    project_excess: Option<EnergyBand>,
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
        project_excess,
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
    assert!(
        project_excess.is_none() || !(lagrangian || reverse),
        "--project-excess is for forward shallow water only (energy rises in reverse time)"
    );
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
        project_excess,
        spectral,
        mask,
        ..Default::default()
    };
    let mut rows = Vec::new();
    let model = fit(model, gen.as_mut(), &fcfg, &device, |row| {
        match row.excess_active {
            Some(a) => println!(
                "step {:>6}  loss {:.6e}  {:.1}s  excess active {:.3}",
                row.step, row.loss, row.secs, a
            ),
            None => println!("step {:>6}  loss {:.6e}  {:.1}s", row.step, row.loss, row.secs),
        }
        rows.push((row.step, row.loss, row.secs, row.excess_active));
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
        excess_r: project_excess.map(|b| b.r_max),
        excess_r_min: project_excess.map_or(0.0, |b| b.r_min),
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

    let errs = eval_model(
        &model, &meta, 4, seed + 1000, project, project_energy, r_max, project_excess, &device,
    );
    print_errors(&errs);
    write_run_log(
        log_dir, steps, lr, batch, pool, single_drop, long_fraction, amplitude_max,
        extra_drop_prob, no_horizon_fraction, extended_fraction, rollout, rollout_after, noise,
        lr_floor, project, project_energy, r_max, project_excess, spectral, nu, seed, layers, dim, &rows,
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
    excess: Option<EnergyBand>,
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
            excess,
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

/// `--project-excess` / `--excess-r-min` resolved against the checkpoint:
/// an absent flag takes the checkpoint's own value, `r_max <= 0` is off.
type ExcessFlags = (Option<f32>, Option<f32>);

fn resolve_excess(flags: ExcessFlags, meta: &Meta) -> Option<EnergyBand> {
    let r_max = match flags.0 {
        None => meta.excess_r?,
        Some(r) if r <= 0.0 => return None,
        Some(r) => r,
    };
    Some(EnergyBand { r_min: flags.1.unwrap_or(meta.excess_r_min), r_max })
}

fn eval_cmd(
    dir: &Path,
    trajectories: usize,
    seed: u64,
    project: bool,
    project_energy: bool,
    project_excess: Option<f32>,
    excess_r_min: Option<f32>,
) {
    let device = <Back as burn::tensor::backend::Backend>::Device::default();
    let (model, meta): (jacobi2000::Jacobi2000<Back>, Meta) = load(dir, &device);
    let excess = resolve_excess((project_excess, excess_r_min), &meta);
    println!(
        "backend: {BACKEND}, mass projection: {project}, energy projection: {project_energy} (r_max {}), excess projection: {excess:?}, mask: {}",
        meta.energy_r_max,
        mask_flags(meta.mask)
    );
    let e = eval_model(
        &model, &meta, trajectories, seed, project, project_energy, meta.energy_r_max, excess,
        &device,
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
        bias: rows.iter().map(|r| r.bias).sum::<f32>() / n,
    }
}

fn metrics_row(label: &str, ticks: usize, m: &Metrics) -> String {
    format!(
        "| {label} | {ticks} | {:.4} | {:.4} | {:.5} | {:.4} | {:+.3} | {:+.3} | {:.5} | {:.5} | {:.5} | {:.5} | {:+.3} |",
        m.corr,
        m.corr_h,
        m.active_rmse,
        m.amplitude_ratio,
        m.radius_error,
        m.xcorr_lag,
        m.bands[0],
        m.bands[1],
        m.bands[2],
        m.global_rmse,
        m.bias
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
    project_excess: ExcessFlags,
    xr: usize,
    reverse: bool,
) {
    let device = <Back as burn::tensor::backend::Backend>::Device::default();
    let (model, meta): (jacobi2000::Jacobi2000<Back>, Meta) = load(dir, &device);
    let excess = if reverse { None } else { resolve_excess(project_excess, &meta) };
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
                project_energy, meta.energy_r_max, excess, &device,
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
        "backend: {BACKEND}, mass projection: {project}, energy projection: {project_energy}, excess projection: {excess:?}, mask: {}",
        mask_flags(meta.mask)
    );
    println!(
        "{used} single-drop trajectories, {skipped} skipped (horizon < {need:.0} ticks)\n"
    );
    println!("| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) | bias |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
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

    if let Some(band) = excess {
        let mut rows: Vec<Vec<train::metrics::ExcessRow>> = vec![Vec::new(); horizons.len()];
        let (mut active, mut ticks) = (0usize, 0usize);
        for (init, _drop) in &trajs {
            let (r_rows, a, t) = train::metrics::excess_activity_rollout(
                &model, &params, &meta.norm, init, &horizons, meta.mask, band, &device,
            );
            for (i, row) in r_rows.into_iter().enumerate() {
                rows[i].push(row);
            }
            active += a;
            ticks += t;
        }
        println!(
            "\nexcess energy E'_t/E'_0, model rollout with the excess projection ON (band \
             [{}, {}]) vs \
             OFF vs the solver, mean over {used} trajectories:\n",
            band.r_min, band.r_max
        );
        println!("| ticks | ON | OFF | solver |\n|---:|---:|---:|---:|");
        for (i, t) in horizons.iter().enumerate() {
            let n = rows[i].len() as f32;
            let m = |f: fn(&train::metrics::ExcessRow) -> f32| rows[i].iter().map(f).sum::<f32>() / n;
            println!("| {t} | {:.6} | {:.6} | {:.6} |", m(|r| r.on), m(|r| r.off), m(|r| r.solver));
        }
        println!(
            "\nexcess constraint active (proposal outside the band before projection): {active} of \
             {ticks} ticks ({:.1} %)",
            100.0 * active as f32 / ticks as f32
        );
    }

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
        println!("| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) | bias |");
        println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
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
fn eval_stress_cmd(
    dir: &Path,
    horizons: &[usize],
    project: bool,
    project_energy: bool,
    project_excess: ExcessFlags,
) {
    let device = <Back as burn::tensor::backend::Backend>::Device::default();
    let (model, meta): (jacobi2000::Jacobi2000<Back>, Meta) = load(dir, &device);
    let excess = resolve_excess(project_excess, &meta);
    let params = meta.params();
    let mut horizons = horizons.to_vec();
    horizons.sort_unstable();

    println!(
        "backend: {BACKEND}, mass projection: {project}, excess projection: {excess:?}, mask: {}, STRESS SET (task 8b)",
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
                project_energy, meta.energy_r_max, excess, &device,
            );
            for i in 0..horizons.len() {
                model_rows[i].push(m[i]);
                frozen_rows[i].push(f[i]);
            }
        }
        println!("### {label} (8 trajectories)\n");
        println!("| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) | bias |");
        println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
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

#[allow(clippy::too_many_arguments)]
fn dump_dataset_cmd(
    pool: usize,
    long_fraction: f32,
    amplitude_max: f32,
    extra_drop_prob: f32,
    no_horizon_fraction: f32,
    extended_fraction: f32,
    nu: f32,
    seed: u64,
    out: &Path,
) {
    let cfg = GeneratorConfig {
        params: Params { nu, ..Params::default() },
        pool,
        single_drop: true,
        long_fraction,
        amplitude_max,
        extra_drop_prob,
        no_horizon_fraction,
        extended_fraction,
        seed,
        ..Default::default()
    };
    let headers = train::data::dump_trajectories(&cfg, out);
    let total_ticks: usize = headers.iter().map(|h| h.ticks).sum();
    println!(
        "{} trajectories, {} total ticks, written to {}",
        headers.len(),
        total_ticks,
        out.display()
    );
}

fn export_weights_cmd(model: &Path, out: &Path) {
    let device = Default::default();
    let (m, meta): (jacobi2000::Jacobi2000<Back>, train::Meta) = load(model, &device);
    let weights = m.to_weights();
    std::fs::create_dir_all(out).unwrap();
    write_safetensors(&out.join("model.safetensors"), &weights);
    std::fs::write(
        out.join("config.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
    println!(
        "exported {} tensors to {}",
        weights.len(),
        out.display()
    );
}

fn import_weights_cmd(safetensors: &Path, config: &Path, out: &Path) {
    let device = Default::default();
    let meta: train::Meta =
        serde_json::from_str(&std::fs::read_to_string(config).unwrap()).unwrap();
    let cfg = Jacobi2000Config::new()
        .with_nx(meta.nx)
        .with_ny(meta.ny)
        .with_patch(meta.patch)
        .with_dim(meta.dim)
        .with_layers(meta.layers)
        .with_heads(meta.heads)
        .with_ff_mult(meta.ff_mult);
    let weights = read_safetensors(safetensors);
    let model = jacobi2000::Jacobi2000::<Back>::from_weights(&cfg, &weights, &device);
    save(out, &model, &meta);
    println!("imported into {}", out.display());
}

/// A fixed, purely deterministic synthetic input (two low-order sinusoids per
/// channel) that a Python reader can reconstruct from `config.json` alone
/// (`nx`, `ny`) without calling into Rust at all.
fn probe_input(nx: usize, ny: usize) -> Vec<f32> {
    let mut state = vec![0.0f32; 3 * ny * nx];
    for c in 0..3 {
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

fn forward_probe_cmd(model: &Path, out: &Path) {
    let device = Default::default();
    let (m, meta): (jacobi2000::Jacobi2000<Back>, train::Meta) = load(model, &device);
    let (nx, ny) = (meta.nx, meta.ny);
    let state = probe_input(nx, ny);
    let params = meta.params();
    let meta_vec = train::data::meta_of(&params);

    let x = Tensor::<Back, 1>::from_floats(state.as_slice(), &device).reshape([1, 3, ny, nx]);
    let meta_t = Tensor::<Back, 1>::from_floats(meta_vec.as_slice(), &device)
        .reshape([1, train::META_DIM]);
    let mask_t = meta.mask.to_tensor::<Back>(&m.grid, &device);

    let out_t = m.forward(x, meta_t, mask_t);
    let data: Vec<f32> = out_t.into_data().to_vec().unwrap();
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(out, bytes).unwrap();
    println!("wrote {} floats to {}", data.len(), out.display());
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
    project_excess: Option<EnergyBand>,
    spectral: f32,
    nu: f32,
    seed: u64,
    layers: usize,
    dim: usize,
    rows: &[(usize, f32, f64, Option<f32>)],
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
    let init_flag = match project_excess {
        Some(b) => format!(" --project-excess {} --excess-r-min {}{init_flag}", b.r_max, b.r_min),
        None => init_flag,
    };
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
    if project_excess.is_some() {
        s.push_str("\n## Loss\n\n| step | loss | s | excess active |\n|---:|---:|---:|---:|\n");
    } else {
        s.push_str("\n## Loss\n\n| step | loss | s |\n|---:|---:|---:|\n");
    }
    for (step, loss, secs, active) in rows {
        match active {
            Some(a) => s.push_str(&format!("| {step} | {loss:.6e} | {secs:.1} | {a:.3} |\n")),
            None => s.push_str(&format!("| {step} | {loss:.6e} | {secs:.1} |\n")),
        }
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
/// on the training distribution: the coded bound
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
    let dt_dx = params.dt / params.dx;
    let mut excess_ratios = Vec::new();
    let mut plain_ratios = Vec::new();
    let mut all_excess = Vec::new();
    let mut injected = 0usize;
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
        // Excess energy, skipping windows where the generator injected a
        // drop mid-window (mass jumps; the solver did not produce that step).
        let x0e: Vec<f32> = excess_energy(&x0, &denorm, params.g, dt_dx).into_data().to_vec().unwrap();
        let x1e: Vec<f32> = excess_energy(&x1, &denorm, params.g, dt_dx).into_data().to_vec().unwrap();
        let p0: Vec<f32> = excess_energy(&x0, &denorm, params.g, 0.0).into_data().to_vec().unwrap();
        let p1: Vec<f32> = excess_energy(&x1, &denorm, params.g, 0.0).into_data().to_vec().unwrap();
        let m0: Vec<f32> = jacobi2000::mass(&x0).into_data().to_vec().unwrap();
        let m1: Vec<f32> = jacobi2000::mass(&x1).into_data().to_vec().unwrap();
        for i in 0..b {
            if (m1[i] - m0[i]).abs() > 1e-3 {
                injected += 1;
                continue;
            }
            all_excess.push(x0e[i]);
            if x0e[i] > 1e-12 {
                excess_ratios.push(x1e[i] / x0e[i]);
                plain_ratios.push(p1[i] / p0[i]);
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

    println!(
        "\nexcess energy E' = sum 1/2 g (h - hbar)^2 + 1/2 h |u|^2 - (dt/dx)(g/2) sum (h - hbar) div(h u), {} windows ({injected} skipped: drop injected mid-window):",
        all_excess.len()
    );
    let (xmin, xp50, xp99, xmax) = distribution(&mut all_excess);
    println!("| E' min | p50 | p99 | max |\n|---:|---:|---:|---:|");
    println!("| {xmin:.6} | {xp50:.6} | {xp99:.6} | {xmax:.6} |\n");
    println!("per-tick ratio E'_{{t+1}}/E'_t:");
    println!("| form | samples | above 1 | min | p01 | p50 | p99 | p999 | max |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
    let (mut r_excess, mut r_excess_min) = (0.0f32, 0.0f32);
    for (label, ratios) in [("plain (no cross term)", &plain_ratios), ("with cross term", &excess_ratios)] {
        let up = ratios.iter().filter(|r| **r > 1.0).count();
        let mut sorted = ratios.clone();
        let n_r = sorted.len();
        let (qmin, qp50, qp99, qmax) = distribution(&mut sorted);
        let at = |p: f32| sorted[((p * (n_r - 1) as f32).round() as usize).min(n_r - 1)];
        println!(
            "| {label} | {n_r} | {up} | {qmin:.7} | {:.7} | {qp50:.7} | {qp99:.7} | {:.7} | {qmax:.7} |",
            at(0.01),
            at(0.999)
        );
        r_excess = qmax;
        r_excess_min = qmin;
    }
    println!("\nsuggested --project-excess (measured max, cross-term form, + 1e-6): {:.7}", r_excess + 1e-6);
    println!("suggested --excess-r-min (measured min, cross-term form, - 1e-6): {:.7}", r_excess_min - 1e-6);

    // Running ratio E'_t / E'_0 from fresh drops (the eval sampler, plus
    // high-amplitude drops), 200 ticks each, both forms.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let norm = gen.norm;
    for (label, kind) in [
        ("eval single drop (amp 0.03-0.07)", None),
        ("high amplitude (0.15-0.3)", Some(train::metrics::StressKind::HighAmp)),
    ] {
        let mut rows: [Vec<Vec<f32>>; 2] = [vec![Vec::new(); 3], vec![Vec::new(); 3]];
        for _ in 0..pool {
            let (mut st, _) = match kind {
                None => sample_single_drop(&params, &mut rng),
                Some(k) => sample_stress_drop(k, &params, &mut rng),
            };
            let mut sw = solvers::ShallowWater::new(params);
            let mut packed = vec![0.0f32; 3 * params.cells()];
            let mut at = |st: &State| -> [f32; 2] {
                norm.encode(st, &mut packed);
                let x = Tensor::<Back, 1>::from_floats(packed.as_slice(), &device)
                    .reshape([1, 3, params.ny, params.nx]);
                [
                    excess_energy(&x, &denorm, params.g, 0.0).into_scalar(),
                    excess_energy(&x, &denorm, params.g, dt_dx).into_scalar(),
                ]
            };
            let e0 = at(&st);
            let mut m = [0.0f32; 2];
            for t in 1..=200 {
                sw.step(&mut st);
                let e = at(&st);
                for f in 0..2 {
                    let r = e[f] / e0[f];
                    m[f] = m[f].max(r);
                    if t == 50 {
                        rows[f][1].push(r);
                    }
                    if t == 200 {
                        rows[f][2].push(r);
                    }
                }
            }
            for f in 0..2 {
                rows[f][0].push(m[f]);
            }
        }
        println!("\n{label}, {pool} trajectories x 200 ticks:");
        println!("| form | quantity | min | p50 | max |\n|---|---|---:|---:|---:|");
        for (f, form) in ["plain", "cross term"].iter().enumerate() {
            for (q, name) in ["max_t E'_t/E'_0", "E'_50/E'_0", "E'_200/E'_0"].iter().enumerate() {
                let (a, b, _, d) = distribution(&mut rows[f][q]);
                println!("| {form} | {name} | {a:.6} | {b:.6} | {d:.6} |");
            }
        }
    }
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

/// Task C6: forward-then-backward round trip, at `nu` (which may differ
/// from either checkpoint's own training `nu` — the model sees it through
/// the meta channel), over `trajectories` fresh single-drop initial
/// conditions, reported at each horizon in `horizons`.
#[allow(clippy::too_many_arguments)]
fn round_trip_cmd(
    forward_dir: &Path,
    reverse_dir: &Path,
    horizons: &[usize],
    trajectories: usize,
    seed: u64,
    nu: f32,
    project: bool,
) {
    let device = <Back as burn::tensor::backend::Backend>::Device::default();
    let (fwd, fwd_meta): (jacobi2000::Jacobi2000<Back>, Meta) = load(forward_dir, &device);
    let (rev, _rev_meta): (jacobi2000::Jacobi2000<Back>, Meta) = load(reverse_dir, &device);
    let params = Params { nu, ..fwd_meta.params() };

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    println!("| horizon | corr | amplitude ratio | round-trip RMSE | solver reference RMSE (N-tick physical change) |");
    println!("|---:|---:|---:|---:|---:|");
    for &n in horizons {
        let (mut corr_sum, mut amp_sum, mut rt_sum, mut ref_sum) = (0.0, 0.0, 0.0, 0.0);
        for _ in 0..trajectories {
            let (init, drop) = sample_single_drop(&params, &mut rng);
            let (m, solver_ref) = train::metrics::round_trip(
                &fwd, &rev, &params, &fwd_meta.norm, &init, &drop, n, fwd_meta.mask, project,
                &device,
            );
            corr_sum += m.corr as f64;
            amp_sum += m.amplitude_ratio as f64;
            rt_sum += m.global_rmse as f64;
            ref_sum += solver_ref as f64;
        }
        let t = trajectories as f64;
        println!(
            "| {n} | {:.4} | {:.4} | {:.5} | {:.5} |",
            corr_sum / t, amp_sum / t, rt_sum / t, ref_sum / t
        );
    }
}

/// Task D9: score a `--lagrangian` checkpoint against SPH-splatted truth on
/// one dam-break and one drop trajectory, corr + amplitude ratio at 25
/// ticks (model ticks, `SPH_STEPS_PER_TICK` SPH steps each) — the rung-1
/// pair the goal doc asks for on the Lagrangian side.
fn eval_lagrangian_cmd(dir: &Path, _seed: u64) {
    let device = <Back as burn::tensor::backend::Backend>::Device::default();
    let (model, meta): (jacobi2000::Jacobi2000<Back>, Meta) = load(dir, &device);
    let sph_params = solvers::sph::Params::default();
    let rho0 = sph_params.rest_density;
    let n = sph_params.nx * sph_params.ny;
    let mask = meta.mask.to_tensor::<Back>(&model.grid, &device);

    println!("| trajectory | ticks | corr | amplitude ratio |");
    println!("|---|---:|---:|---:|");
    for (name, initial) in [
        ("dam break", solvers::sph::Initial::DamBreak),
        (
            "drop",
            solvers::sph::Initial::DropSized { cx_frac: 0.5, radius_frac: 0.1, layer_frac: 0.2 },
        ),
    ] {
        let mut sph = solvers::sph::Sph::new(sph_params, initial);
        let dproxy = train::lagrangian::depth_proxy(&sph, &sph_params);
        let field0 = solvers::splat::splat(&sph, sph_params.nx, sph_params.ny, sph_params.dx, sph_params.h, rho0);
        let state0 = solvers::State { h: field0.h, u: field0.u, v: field0.v };
        let mut norm = meta.norm;
        norm.depth = dproxy;
        let mut packed = vec![0.0f32; 3 * n];
        norm.encode(&state0, &mut packed);
        let mut x = Tensor::<Back, 1>::from_floats(packed.as_slice(), &device)
            .reshape([1, 3, sph_params.ny, sph_params.nx]);
        let meta_arr = [
            sph_params.dx.ln(),
            (train::lagrangian::SPH_STEPS_PER_TICK as f32 * sph_params.dt).ln(),
            sph_params.gravity.ln(),
            dproxy.ln(),
            (sph_params.alpha + 1e-6).ln(),
        ];
        let meta_t = Tensor::<Back, 1>::from_floats(meta_arr.as_slice(), &device)
            .reshape([1, train::META_DIM]);

        for _ in 0..25 {
            sph.steps(train::lagrangian::SPH_STEPS_PER_TICK);
            x = model.step(x, meta_t.clone(), mask.clone());
        }
        let truth = solvers::splat::splat(&sph, sph_params.nx, sph_params.ny, sph_params.dx, sph_params.h, rho0);
        let data = x.into_data();
        let pred = data.as_slice::<f32>().unwrap();
        let model_h = train::metrics::decode_h(pred, &norm, n);
        let corr = train::metrics::correlation(&model_h, &truth.h);
        let peak = |v: &[f32]| v.iter().fold(0.0f32, |a, &y| a.max(y.abs()));
        let amp_ratio = peak(&model_h) / peak(&truth.h);
        println!("| {name} | 25 | {corr:.4} | {amp_ratio:.4} |");
    }
}

/// Task D9 step 3: SPH dam-break solver-truth PGM frames at t = 0, 10, 25
/// (model ticks, `train::lagrangian::SPH_STEPS_PER_TICK` SPH steps each),
/// plus (with `--model`) the same ticks rolled out by a `--lagrangian`
/// checkpoint from the same initial splatted state, and the rung-1
/// correlation/amplitude-ratio at 25 ticks between them.
fn pictures_cmd(lagrangian: bool, out_dir: &Path, model_path: Option<&Path>) {
    if !lagrangian {
        println!("pictures: only --lagrangian is implemented so far");
        return;
    }
    std::fs::create_dir_all(out_dir).unwrap();
    let sph_params = solvers::sph::Params::default();
    let mut sph = solvers::sph::Sph::new(sph_params, solvers::sph::Initial::DamBreak);
    let rho0 = sph_params.rest_density;
    let n = sph_params.nx * sph_params.ny;
    let dproxy = train::lagrangian::depth_proxy(&sph, &sph_params);

    let device = <Back as burn::tensor::backend::Backend>::Device::default();
    let loaded: Option<(jacobi2000::Jacobi2000<Back>, Meta)> =
        model_path.map(|mp| load(mp, &device));
    let mut model_run = loaded.map(|(model, meta)| {
        let field0 = solvers::splat::splat(
            &sph, sph_params.nx, sph_params.ny, sph_params.dx, sph_params.h, rho0,
        );
        let state0 = solvers::State { h: field0.h, u: field0.u, v: field0.v };
        let mut norm = meta.norm;
        norm.depth = dproxy;
        let mut packed = vec![0.0f32; 3 * n];
        norm.encode(&state0, &mut packed);
        let mask = meta.mask.to_tensor::<Back>(&model.grid, &device);
        let meta_arr = [
            sph_params.dx.ln(),
            (train::lagrangian::SPH_STEPS_PER_TICK as f32 * sph_params.dt).ln(),
            sph_params.gravity.ln(),
            dproxy.ln(),
            (sph_params.alpha + 1e-6).ln(),
        ];
        let meta_t = Tensor::<Back, 1>::from_floats(meta_arr.as_slice(), &device)
            .reshape([1, train::META_DIM]);
        let x = Tensor::<Back, 1>::from_floats(packed.as_slice(), &device)
            .reshape([1, 3, sph_params.ny, sph_params.nx]);
        (model, norm, mask, meta_t, x)
    });

    let mut written = 0;
    let mut model_h_25 = None;
    let mut truth_h_25 = None;
    for &tick in &[0usize, 10, 25] {
        while written < tick {
            sph.steps(train::lagrangian::SPH_STEPS_PER_TICK);
            if let Some((model, _norm, mask, meta_t, x)) = model_run.as_mut() {
                // One model tick == SPH_STEPS_PER_TICK SPH steps (just
                // advanced above), matching the training generator.
                *x = model.step(x.clone(), meta_t.clone(), mask.clone());
            }
            written += 1;
        }
        let field = solvers::splat::splat(
            &sph, sph_params.nx, sph_params.ny, sph_params.dx, sph_params.h, rho0,
        );
        let path = out_dir.join(format!("sph-dambreak-solver-t{tick}.pgm"));
        write_pgm(&path, &field.h, sph_params.nx, sph_params.ny);
        println!("wrote {}", path.display());
        if tick == 25 {
            truth_h_25 = Some(field.h.clone());
        }

        if let Some((_, norm, _, _, x)) = model_run.as_ref() {
            let data = x.clone().into_data();
            let pred = data.as_slice::<f32>().unwrap();
            let h = train::metrics::decode_h(pred, norm, n);
            let path = out_dir.join(format!("sph-dambreak-model-t{tick}.pgm"));
            write_pgm(&path, &h, sph_params.nx, sph_params.ny);
            println!("wrote {}", path.display());
            if tick == 25 {
                model_h_25 = Some(h);
            }
        }
    }

    if let (Some(m), Some(t)) = (model_h_25, truth_h_25) {
        let corr = train::metrics::correlation(&m, &t);
        let peak = |v: &[f32]| v.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let amp_ratio = peak(&m) / peak(&t);
        println!("\nrung-1 @ 25 ticks, model vs SPH-splatted truth: corr {corr:.4}, amplitude ratio {amp_ratio:.4}");
    } else if model_run.is_none() {
        println!("model frames: skipped (--model not given)");
    }
}
