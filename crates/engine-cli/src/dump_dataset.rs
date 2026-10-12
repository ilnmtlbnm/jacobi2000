//! `dump-dataset`: write single-drop shallow-water trajectories from the
//! reference solver to disk, as the training set for the PyTorch side
//! (`training/jacobi2000_pt`).
//!
//! ```text
//! cargo run -p engine-cli --release --bin dump-dataset -- --pool 2000 --out data/ds
//! ```
//!
//! Output: one `traj-NNNNN.f32` per trajectory (raw little-endian
//! `ticks * 3 * nx * ny` floats, channel-major: all of `h`, then all of `u`,
//! then all of `v`, each tick-major within its channel), a matching
//! `traj-NNNNN.json` header, and a `manifest.json` listing every header.
//! The solver is the only place the physics is written down; Python reads
//! these flat files and never touches it.

use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use solvers::{Params, ShallowWater, State};

struct Config {
    params: Params,
    pool: usize,
    /// Life of a short trajectory, inclusive.
    short_ticks: (usize, usize),
    /// Life of the long minority.
    long_ticks: usize,
    /// Share of trajectories drawn at `long_ticks`.
    long_fraction: f32,
    amplitude: (f32, f32),
    sigma: (f32, f32),
    /// Top of the amplitude range of the extended distribution.
    amplitude_max: f32,
    /// Probability of an extra drop at a random later tick, tried up to
    /// twice per trajectory.
    extra_drop_prob: f32,
    /// Fraction of trajectories exempt from the pre-reflection margin, so
    /// near-wall and corner drops (and their reflections) are kept.
    no_horizon_fraction: f32,
    /// Fraction of births drawn from the extended distribution above; the
    /// rest use amplitude 0.02-0.08, no extra drops, margin respected.
    extended_fraction: f32,
    seed: u64,
}

/// A drop scheduled to land at a later tick of a trajectory.
struct PendingDrop {
    tick: usize,
    cx: f32,
    cy: f32,
    amp: f32,
    sigma: f32,
}

struct Birth {
    ticks: usize,
    cx: f32,
    cy: f32,
    amp: f32,
    sigma: f32,
}

#[derive(Serialize)]
struct Header {
    index: usize,
    /// Frame count, including tick 0 (the drop's birth).
    ticks: usize,
    nx: usize,
    ny: usize,
    dx: f32,
    dt: f32,
    g: f32,
    depth: f32,
    nu: f32,
    cx: f32,
    cy: f32,
    amp: f32,
    sigma: f32,
}

/// Log-uniform draw in `[lo, hi)`, `lo > 0`.
fn log_uniform(rng: &mut StdRng, lo: f32, hi: f32) -> f32 {
    rng.gen_range(lo.ln()..hi.ln()).exp()
}

fn apply_pending(state: &mut State, p: &Params, pending: &mut Vec<PendingDrop>, age: usize) {
    pending.retain(|d| {
        if d.tick == age {
            state.add_drop(p, d.cx, d.cy, d.amp, d.sigma);
            false
        } else {
            true
        }
    });
}

/// A single drop with a life drawn from the short range or the long
/// minority, plus an optional second and third drop at a random later tick.
/// Short, margin-respecting trajectories place the drop far enough from
/// every wall that the whole life is pre-reflection; long and
/// `no_horizon_fraction` ones do not, so reflections are in the data.
fn draw_single_drop(cfg: &Config, rng: &mut StdRng) -> (State, Birth, Vec<PendingDrop>) {
    let p = cfg.params;
    let extended = rng.gen_bool(cfg.extended_fraction as f64);
    let (amplitude_max, extra_drop_prob, no_horizon_fraction) = if extended {
        (cfg.amplitude_max, cfg.extra_drop_prob, cfg.no_horizon_fraction)
    } else {
        (0.08, 0.0, 0.0)
    };
    let long = rng.gen_bool(cfg.long_fraction as f64);
    let ticks = if long {
        cfg.long_ticks
    } else {
        rng.gen_range(cfg.short_ticks.0..=cfg.short_ticks.1)
    };
    let no_horizon = rng.gen_bool(no_horizon_fraction as f64);
    let margin = if long || no_horizon {
        8.0
    } else {
        ticks as f32 * p.cells_per_tick()
    };
    let cx = rng.gen_range(margin..(p.nx as f32 - margin));
    let cy = rng.gen_range(margin..(p.ny as f32 - margin));
    let amp = log_uniform(rng, cfg.amplitude.0, amplitude_max)
        * if rng.gen_bool(0.25) { -1.0 } else { 1.0 };
    let sigma = rng.gen_range(cfg.sigma.0..cfg.sigma.1);
    let mut s = State::flat(&p);
    s.add_drop(&p, cx, cy, amp, sigma);

    let mut pending = Vec::new();
    for _ in 0..2 {
        if ticks < 2 || !rng.gen_bool(extra_drop_prob as f64) {
            break;
        }
        let (dcx, dcy) = if rng.gen_bool(0.5) {
            (cx, cy)
        } else {
            (
                rng.gen_range(margin..(p.nx as f32 - margin)),
                rng.gen_range(margin..(p.ny as f32 - margin)),
            )
        };
        pending.push(PendingDrop {
            tick: rng.gen_range(1..ticks),
            cx: dcx,
            cy: dcy,
            amp: log_uniform(rng, cfg.amplitude.0, amplitude_max)
                * if rng.gen_bool(0.25) { -1.0 } else { 1.0 },
            sigma: rng.gen_range(cfg.sigma.0..cfg.sigma.1),
        });
    }
    (s, Birth { ticks, cx, cy, amp, sigma }, pending)
}

fn dump_trajectories(cfg: &Config, dir: &Path) -> Vec<Header> {
    let p = cfg.params;
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    std::fs::create_dir_all(dir).unwrap();
    let mut headers = Vec::with_capacity(cfg.pool);
    for index in 0..cfg.pool {
        let (mut state, birth, mut pending) = draw_single_drop(cfg, &mut rng);
        let mut solver = ShallowWater::new(p);
        let n = p.cells();
        let mut h_all = Vec::with_capacity((birth.ticks + 1) * n);
        let mut u_all = Vec::with_capacity((birth.ticks + 1) * n);
        let mut v_all = Vec::with_capacity((birth.ticks + 1) * n);
        h_all.extend_from_slice(&state.h);
        u_all.extend_from_slice(&state.u);
        v_all.extend_from_slice(&state.v);
        for t in 1..=birth.ticks {
            solver.step(&mut state);
            apply_pending(&mut state, &p, &mut pending, t);
            h_all.extend_from_slice(&state.h);
            u_all.extend_from_slice(&state.u);
            v_all.extend_from_slice(&state.v);
        }
        let mut bytes = Vec::with_capacity((h_all.len() + u_all.len() + v_all.len()) * 4);
        for v in h_all.iter().chain(u_all.iter()).chain(v_all.iter()) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join(format!("traj-{index:05}.f32")), &bytes).unwrap();
        let header = Header {
            index,
            ticks: birth.ticks + 1,
            nx: p.nx,
            ny: p.ny,
            dx: p.dx,
            dt: p.dt,
            g: p.g,
            depth: p.depth,
            nu: p.nu,
            cx: birth.cx,
            cy: birth.cy,
            amp: birth.amp,
            sigma: birth.sigma,
        };
        std::fs::write(
            dir.join(format!("traj-{index:05}.json")),
            serde_json::to_string(&header).unwrap(),
        )
        .unwrap();
        headers.push(header);
    }
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&headers).unwrap(),
    )
    .unwrap();
    headers
}

fn main() {
    let mut cfg = Config {
        params: Params { nu: 1e-3, ..Params::default() },
        pool: 2000,
        short_ticks: (60, 80),
        long_ticks: 400,
        long_fraction: 0.2,
        amplitude: (0.02, 0.08),
        sigma: (2.5, 6.0),
        amplitude_max: 0.3,
        extra_drop_prob: 0.3,
        no_horizon_fraction: 0.3,
        extended_fraction: 0.3,
        seed: 20,
    };
    let mut out: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| -> String {
            it.next().unwrap_or_else(|| {
                eprintln!("{name} needs a value");
                std::process::exit(2);
            })
        };
        match arg.as_str() {
            "--pool" => cfg.pool = value("--pool").parse().expect("--pool"),
            "--long-fraction" => cfg.long_fraction = value("--long-fraction").parse().expect("--long-fraction"),
            "--amplitude-max" => cfg.amplitude_max = value("--amplitude-max").parse().expect("--amplitude-max"),
            "--extra-drop-prob" => cfg.extra_drop_prob = value("--extra-drop-prob").parse().expect("--extra-drop-prob"),
            "--no-horizon-fraction" => {
                cfg.no_horizon_fraction = value("--no-horizon-fraction").parse().expect("--no-horizon-fraction")
            }
            "--extended-fraction" => {
                cfg.extended_fraction = value("--extended-fraction").parse().expect("--extended-fraction")
            }
            "--nu" => cfg.params.nu = value("--nu").parse().expect("--nu"),
            "--seed" => cfg.seed = value("--seed").parse().expect("--seed"),
            "--out" => out = Some(PathBuf::from(value("--out"))),
            other => {
                eprintln!("unrecognized argument `{other}`");
                std::process::exit(2);
            }
        }
    }
    let Some(out) = out else {
        eprintln!("usage: dump-dataset --out <dir> [--pool N] [--nu V] [--seed S] [--long-fraction F] [--amplitude-max A] [--extra-drop-prob P] [--no-horizon-fraction F] [--extended-fraction F]");
        std::process::exit(2);
    };
    let headers = dump_trajectories(&cfg, &out);
    let total_ticks: usize = headers.iter().map(|h| h.ticks).sum();
    println!("{} trajectories, {} total ticks, written to {}", headers.len(), total_ticks, out.display());
}
