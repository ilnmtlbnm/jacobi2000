//! Trajectories are generated on demand: no dataset on disk, the solver is
//! fast enough (~31k ticks/s at 64x64) that regenerating is cheaper than
//! reading. A pool of live trajectories is advanced in lockstep; each call
//! returns the window the pool is currently sitting on, so consecutive
//! batches are decorrelated in time without re-running the spin-up.

use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use solvers::{Params, ShallowWater, State};

/// `[ln dx, ln dt, ln g, ln depth, ln(nu + 1e-6)]`. Logs, because these span
/// decades and the model only ever needs their ratios.
pub const META_DIM: usize = 5;

pub fn meta_of(p: &Params) -> [f32; META_DIM] {
    [
        p.dx.ln(),
        p.dt.ln(),
        p.g.ln(),
        p.depth.ln(),
        (p.nu + 1e-6).ln(),
    ]
}

#[derive(Debug, Clone, Copy)]
pub struct GeneratorConfig {
    pub params: Params,
    /// Live trajectories held at once. Independent of `batch`: run 8 saw
    /// ~64 distinct trajectories in total because the pool was the batch.
    pub pool: usize,
    /// Trajectories drawn from the pool per training step.
    pub batch: usize,
    /// Rung 2: one drop, no velocity field, and a life short enough that the
    /// whole trajectory is pre-reflection (bar the long minority).
    pub single_drop: bool,
    /// Life of a short single-drop trajectory, inclusive.
    pub short_ticks: (usize, usize),
    /// Life of the long minority.
    pub long_ticks: usize,
    /// Share of single-drop trajectories drawn at `long_ticks`.
    pub long_fraction: f32,
    /// Ticks between a trajectory being born and being retired. Unused in
    /// `single_drop` mode, where the life is drawn per trajectory.
    pub trajectory_ticks: usize,
    /// Ticks discarded right after the initial condition is drawn.
    pub warmup_ticks: usize,
    pub drops: (usize, usize),
    pub amplitude: (f32, f32),
    pub sigma: (f32, f32),
    /// Amplitude of the random initial velocity field, in solver units.
    pub velocity: f32,
    /// Top of the single-drop amplitude range: drawn log-uniform between
    /// `amplitude.0` and this, so the old 0.02-0.08 uniform range is a
    /// subset of what can come out. Task 8b: repeated drops at one spot
    /// stack past 0.08, so the generator needs to be able to produce that
    /// amplitude directly too.
    pub amplitude_max: f32,
    /// Probability of an extra drop at a random later tick (same or a
    /// different position), tried up to twice per trajectory so a
    /// trajectory can carry up to three drops total. Task 8b: stacked
    /// drops at one spot were never seen during training.
    pub extra_drop_prob: f32,
    /// Fraction of single-drop trajectories exempt from the pre-reflection
    /// margin, so near-wall/corner drops (and their reflections) are kept
    /// instead of always being placed far enough from every wall to avoid
    /// them. Task 8b: reflections off a nearby wall were never seen.
    pub no_horizon_fraction: f32,
    /// Task 8c: fraction of single-drop pool slots that draw from the
    /// extended distribution above (`amplitude_max`, `extra_drop_prob`,
    /// `no_horizon_fraction`) on birth. The rest draw from the ORIGINAL
    /// single-drop distribution (amplitude 0.02-0.08, no extra drops,
    /// margin-respecting), so the pool is a mixture rather than a full
    /// replacement of the standard regime.
    pub extended_fraction: f32,
    pub seed: u64,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        GeneratorConfig {
            params: Params {
                nu: 1e-3,
                ..Params::default()
            },
            pool: 16,
            batch: 16,
            single_drop: false,
            short_ticks: (60, 80),
            long_ticks: 400,
            long_fraction: 0.2,
            trajectory_ticks: 400,
            warmup_ticks: 8,
            drops: (1, 4),
            amplitude: (0.02, 0.08),
            sigma: (2.5, 6.0),
            velocity: 0.02,
            amplitude_max: 0.3,
            extra_drop_prob: 0.3,
            no_horizon_fraction: 0.3,
            extended_fraction: 0.0,
            seed: 0,
        }
    }
}

/// One training window: `states[0]` is the input, `states[1..]` the targets,
/// each `[batch, 3, ny, nx]` flattened, normalized.
#[derive(Debug, Clone)]
pub struct Batch {
    pub states: Vec<Vec<f32>>,
    pub meta: Vec<f32>,
    pub batch: usize,
    pub nx: usize,
    pub ny: usize,
}

impl Batch {
    pub fn rollout(&self) -> usize {
        self.states.len() - 1
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Normalizer {
    /// Mean of `h - depth`, `u`, `v`.
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub depth: f32,
}

impl Normalizer {
    pub fn identity(depth: f32) -> Self {
        Normalizer {
            mean: [0.0; 3],
            std: [1.0; 3],
            depth,
        }
    }

    /// Pack a solver state into a normalized `[3, ny, nx]` slice.
    pub fn encode(&self, s: &State, out: &mut [f32]) {
        let n = s.h.len();
        for i in 0..n {
            out[i] = (s.h[i] - self.depth - self.mean[0]) / self.std[0];
            out[n + i] = (s.u[i] - self.mean[1]) / self.std[1];
            out[2 * n + i] = (s.v[i] - self.mean[2]) / self.std[2];
        }
    }

    pub fn decode(&self, x: &[f32], s: &mut State) {
        let n = s.h.len();
        for i in 0..n {
            s.h[i] = x[i] * self.std[0] + self.mean[0] + self.depth;
            s.u[i] = x[n + i] * self.std[1] + self.mean[1];
            s.v[i] = x[2 * n + i] * self.std[2] + self.mean[2];
        }
    }
}

/// A drop scheduled to land at `tick` (trajectory age, in ticks) rather than
/// at birth. Task 8b: "an optional second/third drop at a random later
/// tick" needs the trajectory to carry a schedule, since the pool steps one
/// tick at a time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PendingDrop {
    pub tick: usize,
    pub cx: f32,
    pub cy: f32,
    pub amp: f32,
    pub sigma: f32,
}

/// Apply (and remove) every pending drop scheduled for `age`.
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

struct Trajectory {
    solver: ShallowWater,
    state: State,
    age: usize,
    life: usize,
    pending: Vec<PendingDrop>,
}

pub struct Generator {
    cfg: GeneratorConfig,
    rng: StdRng,
    pool: Vec<Trajectory>,
    pub norm: Normalizer,
}

impl Generator {
    pub fn new(cfg: GeneratorConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut pool = Vec::with_capacity(cfg.pool);
        for k in 0..cfg.pool {
            let mut t = Trajectory {
                solver: ShallowWater::new(cfg.params),
                state: State::flat(&cfg.params),
                age: 0,
                life: cfg.trajectory_ticks,
                pending: Vec::new(),
            };
            init_trajectory(&cfg, &mut rng, &mut t);
            // Stagger the pool so members retire at different times.
            let skew = k * t.life / cfg.pool.max(1);
            t.solver.steps(&mut t.state, skew);
            t.age += skew;
            pool.push(t);
        }
        let norm = Normalizer::identity(cfg.params.depth);
        Generator {
            cfg,
            rng,
            pool,
            norm,
        }
    }

    pub fn config(&self) -> &GeneratorConfig {
        &self.cfg
    }

    /// Estimate per-field mean/std over `n` ticks of the live pool and adopt
    /// them. Call once before training; the model sees normalized fields.
    pub fn fit_normalizer(&mut self, ticks: usize) {
        let mut sum = [0.0f64; 3];
        let mut sq = [0.0f64; 3];
        let mut count = 0u64;
        for _ in 0..ticks {
            for t in self.pool.iter_mut() {
                let n = t.state.h.len();
                for i in 0..n {
                    let vals = [
                        (t.state.h[i] - self.cfg.params.depth) as f64,
                        t.state.u[i] as f64,
                        t.state.v[i] as f64,
                    ];
                    for c in 0..3 {
                        sum[c] += vals[c];
                        sq[c] += vals[c] * vals[c];
                    }
                }
                count += n as u64;
                t.solver.step(&mut t.state);
                t.age += 1;
                apply_pending(&mut t.state, &self.cfg.params, &mut t.pending, t.age);
            }
            for i in 0..self.pool.len() {
                self.retire(i);
            }
        }
        let mut mean = [0.0f32; 3];
        let mut std = [0.0f32; 3];
        for c in 0..3 {
            let m = sum[c] / count as f64;
            let var = (sq[c] / count as f64 - m * m).max(0.0);
            mean[c] = m as f32;
            std[c] = (var.sqrt() as f32).max(1e-8);
        }
        self.norm = Normalizer {
            mean,
            std,
            depth: self.cfg.params.depth,
        };
    }

    /// A window of `rollout + 1` consecutive states for `batch` trajectories
    /// drawn from the pool. Only the drawn ones are advanced, so a pool much
    /// larger than the batch costs nothing per step.
    pub fn next_batch(&mut self, rollout: usize) -> Batch {
        let p = self.cfg.params;
        let n = p.cells();
        let b = self.cfg.batch.min(self.pool.len());
        let picks: Vec<usize> =
            rand::seq::index::sample(&mut self.rng, self.pool.len(), b).into_vec();
        let mut states = vec![vec![0.0f32; b * 3 * n]; rollout + 1];
        let mut meta = vec![0.0f32; b * META_DIM];

        for (k, &pick) in picks.iter().enumerate() {
            let t = &mut self.pool[pick];
            let m = meta_of(&t.solver.params);
            meta[k * META_DIM..(k + 1) * META_DIM].copy_from_slice(&m);
            for (step, frame) in states.iter_mut().enumerate() {
                if step > 0 {
                    t.solver.step(&mut t.state);
                    t.age += 1;
                    apply_pending(&mut t.state, &p, &mut t.pending, t.age);
                }
                let off = k * 3 * n;
                self.norm.encode(&t.state, &mut frame[off..off + 3 * n]);
            }
        }
        for pick in picks {
            self.retire(pick);
        }
        Batch {
            states,
            meta,
            batch: b,
            nx: p.nx,
            ny: p.ny,
        }
    }

    fn retire(&mut self, i: usize) {
        if self.pool[i].age >= self.pool[i].life {
            init_trajectory(&self.cfg, &mut self.rng, &mut self.pool[i]);
        }
    }
}

/// One record of the rung-2 pool: the drop it was drawn from, how long it
/// lives, and how long it has before the wave meets a wall.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Record {
    pub index: usize,
    pub cx: f32,
    pub cy: f32,
    pub amp: f32,
    pub sigma: f32,
    pub ticks: usize,
    pub horizon: f32,
    /// Peak `|h - depth|` at the end of the trajectory: proof it was run.
    pub peak: f32,
    /// Extra drops (task 8b) scheduled at a random later tick, beyond the
    /// primary one above.
    pub extra_drops: usize,
    /// True if this trajectory was drawn exempt from the pre-reflection
    /// margin (task 8b `no_horizon_fraction`), so `horizon < ticks` is
    /// expected rather than a bug.
    pub no_horizon: bool,
    /// True if this trajectory was drawn from the extended distribution
    /// (task 8c `extended_fraction`) rather than the original single-drop
    /// distribution.
    pub extended: bool,
}

/// Log-uniform draw in `[lo, hi)`, `lo > 0`.
fn log_uniform(rng: &mut StdRng, lo: f32, hi: f32) -> f32 {
    rng.gen_range(lo.ln()..hi.ln()).exp()
}

/// A single drop with a life drawn from the short range, or from the long
/// minority, plus (task 8b) an optional second/third drop at a random later
/// tick and a configurable fraction of trajectories exempt from the
/// pre-reflection margin. Short, margin-respecting ones are placed far
/// enough from every wall that the whole life is pre-reflection; long ones
/// and the `no_horizon_fraction` exempt ones are not, which is the point of
/// them (reflections, near-wall/corner drops).
fn draw_single_drop(cfg: &GeneratorConfig, rng: &mut StdRng) -> (State, Record, Vec<PendingDrop>) {
    let p = cfg.params;
    // Task 8c: each birth is either drawn from the extended distribution
    // (as configured on the CLI) or from the original single-drop
    // distribution, so the pool is a mixture rather than a full replacement.
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

    // Up to two extra drops, each independently rolled, landing at a random
    // later tick either on top of the first drop (stacked) or elsewhere.
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

    (
        s,
        Record {
            index: 0,
            cx,
            cy,
            amp,
            sigma,
            ticks,
            horizon: p.horizon_ticks(cx, cy),
            peak: 0.0,
            extra_drops: pending.len(),
            no_horizon,
            extended,
        },
        pending,
    )
}

/// Generate `cfg.pool` fresh single-drop trajectories on the CPU, running
/// each one for its own life, and write one record per trajectory to `dump`.
/// This is the rung-2 pool: the count is the pool size, never the batch size.
pub fn generate(cfg: &GeneratorConfig, dump: Option<&Path>) -> Vec<Record> {
    let p = cfg.params;
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut solver = ShallowWater::new(p);
    if let Some(dir) = dump {
        std::fs::create_dir_all(dir).unwrap();
    }
    let mut out = Vec::with_capacity(cfg.pool);
    for index in 0..cfg.pool {
        let (mut state, mut rec, mut pending) = draw_single_drop(cfg, &mut rng);
        rec.index = index;
        for t in 1..=rec.ticks {
            solver.step(&mut state);
            apply_pending(&mut state, &p, &mut pending, t);
        }
        rec.peak = state
            .h
            .iter()
            .fold(0.0f32, |a, &x| a.max((x - p.depth).abs()));
        if let Some(dir) = dump {
            std::fs::write(
                dir.join(format!("traj-{index:05}.json")),
                serde_json::to_string(&rec).unwrap(),
            )
            .unwrap();
        }
        out.push(rec);
    }
    out
}

fn init_trajectory(cfg: &GeneratorConfig, rng: &mut StdRng, t: &mut Trajectory) {
    let p = cfg.params;
    if cfg.single_drop {
        let (state, rec, pending) = draw_single_drop(cfg, rng);
        t.state = state;
        t.solver = ShallowWater::new(p);
        t.solver.steps(&mut t.state, cfg.warmup_ticks);
        t.age = 0;
        t.life = rec.ticks;
        t.pending = pending;
        return;
    }
    t.state = State::flat(&p);
    let drops = rng.gen_range(cfg.drops.0..=cfg.drops.1);
    for _ in 0..drops {
        let cx = rng.gen_range(6.0..(p.nx as f32 - 6.0));
        let cy = rng.gen_range(6.0..(p.ny as f32 - 6.0));
        let amp = rng.gen_range(cfg.amplitude.0..cfg.amplitude.1)
            * if rng.gen_bool(0.25) { -1.0 } else { 1.0 };
        let sigma = rng.gen_range(cfg.sigma.0..cfg.sigma.1);
        t.state.add_drop(&p, cx, cy, amp, sigma);
    }
    if cfg.velocity > 0.0 {
        // Smooth random velocity: a few long-wavelength modes, zero at walls.
        for _ in 0..3 {
            let kx = rng.gen_range(1..4) as f32;
            let ky = rng.gen_range(1..4) as f32;
            let au = rng.gen_range(-cfg.velocity..cfg.velocity);
            let av = rng.gen_range(-cfg.velocity..cfg.velocity);
            for y in 0..p.ny {
                for x in 0..p.nx {
                    let fx = std::f32::consts::PI * kx * x as f32 / p.nx as f32;
                    let fy = std::f32::consts::PI * ky * y as f32 / p.ny as f32;
                    let i = y * p.nx + x;
                    t.state.u[i] += au * fx.sin() * fy.cos();
                    t.state.v[i] += av * fx.cos() * fy.sin();
                }
            }
        }
    }
    t.solver = ShallowWater::new(p);
    t.solver.steps(&mut t.state, cfg.warmup_ticks);
    t.age = 0;
    t.life = cfg.trajectory_ticks;
    t.pending = Vec::new();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizer_round_trips() {
        let p = Params::default();
        let mut s = State::flat(&p);
        s.add_drop(&p, 10.0, 12.0, 0.05, 3.0);
        s.u[7] = 0.01;
        let norm = Normalizer {
            mean: [1e-3, -2e-4, 5e-5],
            std: [0.01, 0.002, 0.003],
            depth: p.depth,
        };
        let mut packed = vec![0.0; 3 * p.cells()];
        norm.encode(&s, &mut packed);
        let mut back = State::flat(&p);
        norm.decode(&packed, &mut back);
        for i in 0..p.cells() {
            assert!((s.h[i] - back.h[i]).abs() < 1e-6);
            assert!((s.u[i] - back.u[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn batches_are_finite_and_advance() {
        let cfg = GeneratorConfig {
            pool: 4,
            ..Default::default()
        };
        let mut g = Generator::new(cfg);
        g.fit_normalizer(20);
        // A normalized field should have order-one spread.
        assert!(g.norm.std[0] > 0.0 && g.norm.std[0] < 1.0);
        let b = g.next_batch(4);
        assert_eq!(b.states.len(), 5);
        assert_eq!(b.states[0].len(), 4 * 3 * cfg.params.cells());
        assert_eq!(b.meta.len(), 4 * META_DIM);
        assert!(b.states.iter().all(|s| s.iter().all(|x| x.is_finite())));
        // Consecutive frames must differ: a stuck generator would train the
        // model on the identity.
        let d: f32 = b.states[0]
            .iter()
            .zip(&b.states[1])
            .map(|(a, c)| (a - c).abs())
            .sum();
        assert!(d > 0.0, "frames identical");
    }

    #[test]
    fn pool_size_is_independent_of_batch_size() {
        // Run 8's pool was its batch, so it saw ~64 distinct trajectories all
        // night. The generator must hold N and hand out B.
        let cfg = GeneratorConfig {
            pool: 11,
            batch: 4,
            single_drop: true,
            ..Default::default()
        };
        let dir = std::env::temp_dir().join(format!("jacobi2000-gen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let recs = generate(&cfg, Some(&dir));
        let files = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(recs.len(), 11);
        assert_eq!(files, 11);
        // Most trajectories are short, and a short one never reaches a wall.
        let short = recs.iter().filter(|r| r.ticks <= cfg.short_ticks.1).count();
        assert!(short * 2 > recs.len(), "{short} short of {}", recs.len());
        for r in &recs {
            assert!(r.peak.is_finite() && r.peak > 0.0, "{r:?}");
            if r.ticks <= cfg.short_ticks.1 && !r.no_horizon {
                assert!(r.horizon >= r.ticks as f32, "reflects before it ends: {r:?}");
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();

        let mut g = Generator::new(cfg);
        assert_eq!(g.pool.len(), 11);
        assert_eq!(g.next_batch(2).batch, 4);
    }

    #[test]
    fn extended_single_drop_emits_high_amplitude_and_near_wall_drops() {
        // Task 8b: I saw divergence on repeated drops (which stack
        // past the old 0.08 amplitude cap) and corner drops (which the old
        // generator never produced because it always kept the whole life
        // pre-reflection). Both must be reachable when every birth draws
        // from the extended distribution (task 8c: `extended_fraction`
        // gates this, defaulting to 0.0 for the original distribution).
        let cfg = GeneratorConfig {
            pool: 200,
            single_drop: true,
            extended_fraction: 1.0,
            seed: 7,
            ..Default::default()
        };
        let recs = generate(&cfg, None);
        assert!(
            recs.iter().any(|r| r.amp.abs() > 0.08),
            "no amplitude above the old 0.08 cap among {} trajectories",
            recs.len()
        );
        assert!(
            recs.iter().any(|r| r.no_horizon && r.horizon < r.ticks as f32),
            "no near-wall drop kept its reflection among {} trajectories",
            recs.len()
        );
        assert!(
            recs.iter().any(|r| r.extra_drops > 0),
            "no trajectory drew an extra (stacked) drop among {} trajectories",
            recs.len()
        );
    }

    #[test]
    fn extended_fraction_mixes_original_and_extended_distributions() {
        // Task 8c: the pool must be a mixture, not a full replacement, of
        // the extended distribution.
        let cfg = GeneratorConfig {
            pool: 200,
            single_drop: true,
            extended_fraction: 0.3,
            seed: 7,
            ..Default::default()
        };
        let recs = generate(&cfg, None);
        let extended = recs.iter().filter(|r| r.extended).count();
        assert_eq!(extended, 66, "seeded RNG count changed: {extended} of 200");
        assert!(
            recs.iter()
                .filter(|r| !r.extended)
                .all(|r| r.amp.abs() <= 0.08 && r.extra_drops == 0 && !r.no_horizon),
            "a non-extended birth used extended parameters"
        );
    }

    #[test]
    fn trajectories_stay_bounded_across_retirement() {
        let cfg = GeneratorConfig {
            pool: 4,
            trajectory_ticks: 30,
            ..Default::default()
        };
        let mut g = Generator::new(cfg);
        g.fit_normalizer(10);
        for _ in 0..40 {
            let b = g.next_batch(2);
            let max = b.states[2].iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            assert!(max.is_finite() && max < 50.0, "normalized field blew up: {max}");
        }
    }
}
