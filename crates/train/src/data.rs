//! Trajectories are generated on demand: no dataset on disk, the solver is
//! fast enough (~31k ticks/s at 64x64) that regenerating is cheaper than
//! reading. A pool of live trajectories is advanced in lockstep; each call
//! returns the window the pool is currently sitting on, so consecutive
//! batches are decorrelated in time without re-running the spin-up.

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
    pub pool: usize,
    /// Ticks between a trajectory being born and being retired.
    pub trajectory_ticks: usize,
    /// Ticks discarded right after the initial condition is drawn.
    pub warmup_ticks: usize,
    pub drops: (usize, usize),
    pub amplitude: (f32, f32),
    pub sigma: (f32, f32),
    /// Amplitude of the random initial velocity field, in solver units.
    pub velocity: f32,
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
            trajectory_ticks: 400,
            warmup_ticks: 8,
            drops: (1, 4),
            amplitude: (0.02, 0.08),
            sigma: (2.5, 6.0),
            velocity: 0.02,
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

struct Trajectory {
    solver: ShallowWater,
    state: State,
    age: usize,
    life: usize,
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
            };
            init_trajectory(&cfg, &mut rng, &mut t);
            // Stagger the pool so members retire at different times.
            let skew = k * cfg.trajectory_ticks / cfg.pool.max(1);
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
            }
            self.retire();
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

    /// A window of `rollout + 1` consecutive states for every pool member.
    pub fn next_batch(&mut self, rollout: usize) -> Batch {
        let p = self.cfg.params;
        let n = p.cells();
        let b = self.pool.len();
        let mut states = vec![vec![0.0f32; b * 3 * n]; rollout + 1];
        let mut meta = vec![0.0f32; b * META_DIM];

        for (k, t) in self.pool.iter_mut().enumerate() {
            let m = meta_of(&t.solver.params);
            meta[k * META_DIM..(k + 1) * META_DIM].copy_from_slice(&m);
            for (step, frame) in states.iter_mut().enumerate() {
                if step > 0 {
                    t.solver.step(&mut t.state);
                    t.age += 1;
                }
                let off = k * 3 * n;
                self.norm.encode(&t.state, &mut frame[off..off + 3 * n]);
            }
        }
        self.retire();
        Batch {
            states,
            meta,
            batch: b,
            nx: p.nx,
            ny: p.ny,
        }
    }

    fn retire(&mut self) {
        for t in self.pool.iter_mut() {
            if t.age >= t.life {
                init_trajectory(&self.cfg, &mut self.rng, t);
            }
        }
    }
}

fn init_trajectory(cfg: &GeneratorConfig, rng: &mut StdRng, t: &mut Trajectory) {
    let p = cfg.params;
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
