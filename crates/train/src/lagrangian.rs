//! SPH-splatted trajectories, `Batch`-compatible with `data::Generator`: same
//! `[3, ny, nx]` layout, `[h, u, v]` channels, same `Normalizer`. The grid
//! model trains on these unchanged.
//!
//! **Ticks.** A "model tick" here is `SPH_STEPS_PER_TICK` SPH steps followed
//! by one re-splat, chosen so the splatted fields move a comparable fraction
//! of a cell per tick as `shallow_water`'s default grid:
//!   - shallow water: `cells_per_tick = cfl_safety / sqrt(2) = 0.4/sqrt(2) ~=
//!     0.283` (its own default CFL choice; independent of g/depth).
//!   - SPH: characteristic flow speed for a dam break `c ~= sqrt(g * col_h)`,
//!     `col_h = 0.5 * domain_y` (the default column height) `= 32` on the
//!     default 64-cell domain, so `c ~= sqrt(9.81 * 32) ~= 17.7`. One SPH
//!     step (`dt` from its own CFL-0.2 choice, `~=1.857e-3`) then covers
//!     `c * dt / dx ~= 0.0329` cells.
//!   - `SPH_STEPS_PER_TICK = round(0.283 / 0.0329) = 9`, landing at `~=0.296`
//!     cells/tick, within a few percent of shallow water's own rate.
//!
//! Geometry (column width/height, drop position/radius) is drawn per
//! trajectory via `Initial::DamBreakSized`/`DropSized`, so the model sees a
//! spread of dam breaks and drops rather than one fixed shape.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use solvers::sph::{Initial, Params as SphParams, Sph};
use solvers::splat::splat;
use solvers::State;

use crate::data::{Batch, Normalizer};
use crate::META_DIM;

/// See the module doc for the derivation.
pub const SPH_STEPS_PER_TICK: usize = 9;

#[derive(Debug, Clone, Copy)]
pub struct LagrangianConfig {
    pub sph: SphParams,
    /// Splat grid spacing. The splat grid is always `sph.nx x sph.ny`, the
    /// same box the SPH solver uses.
    pub grid_dx: f32,
    /// Smoothing length used by the splat kernel (independent of the SPH
    /// solver's own `h`).
    pub splat_h: f32,
    pub pool: usize,
    pub batch: usize,
    /// Model ticks (each `SPH_STEPS_PER_TICK` SPH steps) a trajectory lives
    /// before it is retired and redrawn.
    pub life_ticks: (usize, usize),
    pub seed: u64,
}

impl Default for LagrangianConfig {
    fn default() -> Self {
        let sph = SphParams::default();
        LagrangianConfig {
            grid_dx: sph.dx,
            splat_h: sph.h,
            sph,
            pool: 16,
            batch: 16,
            life_ticks: (40, 80),
            seed: 0,
        }
    }
}

/// `[ln dx, ln dt_effective, ln g, ln depth_proxy, ln(nu + 1e-6)]`, matching
/// `data::meta_of`'s layout (`dt_effective = SPH_STEPS_PER_TICK * sph.dt`,
/// `depth_proxy` the rho0-based mean-column-height estimate, `nu` the
/// Monaghan artificial-viscosity `alpha`, 0 if XSPH-only).
fn meta_of(p: &SphParams, depth_proxy: f32) -> [f32; META_DIM] {
    let dt_effective = SPH_STEPS_PER_TICK as f32 * p.dt;
    [
        p.dx.ln(),
        dt_effective.ln(),
        p.gravity.ln(),
        depth_proxy.ln(),
        (p.alpha + 1e-6).ln(),
    ]
}

/// The flat-layer depth that would hold the same total mass across the full
/// domain width: `total_mass / (rho0 * domain_x)`, a rho0-based
/// column-height proxy standing in for shallow water's `depth`.
fn depth_proxy(sph: &Sph, p: &SphParams) -> f32 {
    sph.total_mass() as f32 / (p.rest_density * p.domain_x())
}

struct Trajectory {
    sph: Sph,
    age: usize,
    life: usize,
}

fn draw_trajectory(cfg: &LagrangianConfig, rng: &mut StdRng) -> Trajectory {
    let p = cfg.sph;
    let dam_break = rng.gen_bool(0.5);
    let initial = if dam_break {
        Initial::DamBreakSized {
            col_w_frac: rng.gen_range(0.15..0.35),
            col_h_frac: rng.gen_range(0.3..0.7),
        }
    } else {
        Initial::DropSized {
            cx_frac: rng.gen_range(0.3..0.7),
            radius_frac: rng.gen_range(0.06..0.14),
            layer_frac: rng.gen_range(0.1..0.3),
        }
    };
    let sph = Sph::new(p, initial);
    let life = rng.gen_range(cfg.life_ticks.0..=cfg.life_ticks.1);
    Trajectory { sph, age: 0, life }
}

pub struct LagrangianGenerator {
    cfg: LagrangianConfig,
    rng: StdRng,
    pool: Vec<Trajectory>,
    pub norm: Normalizer,
}

impl LagrangianGenerator {
    pub fn new(cfg: LagrangianConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut pool = Vec::with_capacity(cfg.pool);
        for k in 0..cfg.pool {
            let mut t = draw_trajectory(&cfg, &mut rng);
            // Stagger the pool so members retire at different times.
            let skew = k * t.life / cfg.pool.max(1);
            for _ in 0..skew {
                t.sph.steps(SPH_STEPS_PER_TICK);
            }
            t.age += skew;
            pool.push(t);
        }
        LagrangianGenerator {
            cfg,
            rng,
            pool,
            norm: Normalizer::identity(1.0),
        }
    }

    pub fn config(&self) -> &LagrangianConfig {
        &self.cfg
    }

    /// Estimate per-field mean/std over `n` model ticks of the live pool and
    /// adopt them, mirroring `data::Generator::fit_normalizer`.
    pub fn fit_normalizer(&mut self, ticks: usize) {
        let p = self.cfg.sph;
        let cells = p.nx * p.ny;
        let mut sum = [0.0f64; 3];
        let mut sq = [0.0f64; 3];
        let mut count = 0u64;
        for _ in 0..ticks {
            for t in self.pool.iter_mut() {
                let dproxy = depth_proxy(&t.sph, &p);
                let field = splat(&t.sph, p.nx, p.ny, self.cfg.grid_dx, self.cfg.splat_h, p.rest_density);
                for i in 0..cells {
                    let vals = [
                        (field.h[i] - dproxy) as f64,
                        field.u[i] as f64,
                        field.v[i] as f64,
                    ];
                    for c in 0..3 {
                        sum[c] += vals[c];
                        sq[c] += vals[c] * vals[c];
                    }
                }
                count += cells as u64;
                t.sph.steps(SPH_STEPS_PER_TICK);
                t.age += 1;
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
            // depth is folded into the per-trajectory dproxy subtraction
            // above, so the normalizer's own depth offset is 0.
            depth: 0.0,
        };
    }

    /// A window of `rollout + 1` consecutive splatted states for `batch`
    /// trajectories drawn from the pool, `Batch`-compatible with
    /// `data::Generator::next_batch`.
    pub fn next_batch(&mut self, rollout: usize) -> Batch {
        let p = self.cfg.sph;
        let n = p.nx * p.ny;
        let b = self.cfg.batch.min(self.pool.len());
        let picks: Vec<usize> =
            rand::seq::index::sample(&mut self.rng, self.pool.len(), b).into_vec();
        let mut states = vec![vec![0.0f32; b * 3 * n]; rollout + 1];
        let mut meta = vec![0.0f32; b * META_DIM];

        for (k, &pick) in picks.iter().enumerate() {
            let t = &mut self.pool[pick];
            let dproxy = depth_proxy(&t.sph, &p);
            let m = meta_of(&t.sph.params, dproxy);
            meta[k * META_DIM..(k + 1) * META_DIM].copy_from_slice(&m);
            for (step, frame) in states.iter_mut().enumerate() {
                if step > 0 {
                    t.sph.steps(SPH_STEPS_PER_TICK);
                    t.age += 1;
                }
                let field = splat(&t.sph, p.nx, p.ny, self.cfg.grid_dx, self.cfg.splat_h, p.rest_density);
                let state = State {
                    h: field.h,
                    u: field.u,
                    v: field.v,
                };
                let mut norm = self.norm;
                norm.depth = dproxy;
                let off = k * 3 * n;
                norm.encode(&state, &mut frame[off..off + 3 * n]);
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
            self.pool[i] = draw_trajectory(&self.cfg, &mut self.rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_are_finite_and_advance() {
        let cfg = LagrangianConfig {
            pool: 4,
            batch: 4,
            ..Default::default()
        };
        let mut g = LagrangianGenerator::new(cfg);
        g.fit_normalizer(3);
        let b = g.next_batch(3);
        assert_eq!(b.states.len(), 4);
        assert_eq!(b.states[0].len(), 4 * 3 * cfg.sph.nx * cfg.sph.ny);
        assert!(b.states.iter().all(|s| s.iter().all(|x| x.is_finite())));
        let d: f32 = b.states[0]
            .iter()
            .zip(&b.states[1])
            .map(|(a, c)| (a - c).abs())
            .sum();
        assert!(d > 0.0, "frames identical");
    }

    #[test]
    fn pool_size_is_independent_of_batch_size() {
        let cfg = LagrangianConfig {
            pool: 8,
            batch: 4,
            ..Default::default()
        };
        let mut g = LagrangianGenerator::new(cfg);
        assert_eq!(g.pool.len(), 8);
        g.fit_normalizer(2);
        let b = g.next_batch(2);
        assert_eq!(b.batch, 4);
        assert!(b.states.iter().all(|s| s.iter().all(|x| x.is_finite())));
    }
}
