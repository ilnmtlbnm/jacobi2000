//! 2D weakly-compressible SPH (Monaghan 1992-style).
//!
//! Particles carry position, velocity, mass, density, pressure. Density comes
//! from a kernel sum (Monaghan cubic spline, support radius `2h`); pressure
//! from the Tait equation of state (stiff enough that the fluid is only
//! weakly compressible); acceleration from the symmetric pressure-gradient
//! form (exact Newton's-third-law pairwise forces) plus Monaghan artificial
//! viscosity plus gravity. Positions are advanced with XSPH velocity
//! smoothing (Monaghan 1989) to keep particles from interpenetrating.
//!
//! Neighbor search is a uniform grid, cell size `2h` (the kernel support
//! radius), rebuilt every step: any interacting pair is in the same cell or
//! an adjacent one, so a 3x3 block is enough.
//!
//! Integration is kick-drift-kick (velocity Verlet, symplectic): the
//! acceleration computed at the end of a step is reused as the first half-kick
//! of the next, so there is exactly one force evaluation per step.
//!
//! Walls are the box `[0, nx*dx] x [0, ny*dx]` (same physical units as
//! `shallow_water`'s grid) enforced by position clamp + damped velocity
//! reflection — simpler than boundary particles and enough to keep a
//! dam-break column inside the box.

use std::collections::HashMap;
use std::f32::consts::PI;

const TAIT_GAMMA: f32 = 7.0;
const WALL_DAMPING: f32 = 0.3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub nx: usize,
    pub ny: usize,
    /// Cell size of the reference grid; the box is `nx*dx` by `ny*dx`.
    pub dx: f32,
    /// Smoothing length. Particle spacing at init is `h / 1.3`.
    pub h: f32,
    pub rest_density: f32,
    pub gravity: f32,
    /// Must be >= ~10x the expected max flow speed for "weakly compressible".
    pub sound_speed: f32,
    /// Monaghan artificial viscosity, linear term.
    pub alpha: f32,
    /// Monaghan artificial viscosity, quadratic term.
    pub beta: f32,
    /// XSPH velocity-smoothing coefficient (0 disables it).
    pub xsph_eps: f32,
    pub dt: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            nx: 64,
            ny: 64,
            dx: 1.0,
            h: 1.3,
            rest_density: 1000.0,
            gravity: 9.81,
            // ~10x sqrt(g * a plausible column height).
            sound_speed: 10.0 * (9.81_f32 * 20.0).sqrt(),
            alpha: 0.3,
            beta: 0.0,
            xsph_eps: 0.5,
            dt: 0.0,
        }
        .with_cfl_dt(0.2)
    }
}

impl Params {
    /// CFL limit dominated by the (fast) artificial sound speed, not flow
    /// speed, per the weakly-compressible assumption.
    pub fn cfl_dt(&self, safety: f32) -> f32 {
        safety * self.h / self.sound_speed
    }

    pub fn with_cfl_dt(mut self, safety: f32) -> Self {
        self.dt = self.cfl_dt(safety);
        self
    }

    pub fn domain_x(&self) -> f32 {
        self.nx as f32 * self.dx
    }

    pub fn domain_y(&self) -> f32 {
        self.ny as f32 * self.dx
    }

    fn spacing(&self) -> f32 {
        self.h / 1.3
    }
}

pub enum Initial {
    /// A rectangular column of fluid in the corner (x=0, y=0 is floor).
    DamBreak,
    /// A circular blob dropped above a resting layer at the bottom.
    Drop,
}

pub struct Sph {
    pub params: Params,
    x: Vec<f32>,
    y: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    mass: Vec<f32>,
    density: Vec<f32>,
    pressure: Vec<f32>,
    ax: Vec<f32>,
    ay: Vec<f32>,
}

pub(crate) fn cubic_spline_sigma(h: f32) -> f32 {
    10.0 / (7.0 * PI * h * h)
}

/// Monaghan cubic spline, `q = r / h`.
pub(crate) fn kernel(r: f32, h: f32, sigma: f32) -> f32 {
    let q = r / h;
    if q < 1.0 {
        sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
    } else if q < 2.0 {
        let t = 2.0 - q;
        sigma * 0.25 * t * t * t
    } else {
        0.0
    }
}

/// Gradient of the kernel w.r.t. the position of particle `i`, given the
/// separation `(rx, ry) = pos_i - pos_j` and its length `r`.
fn kernel_grad(rx: f32, ry: f32, r: f32, h: f32, sigma: f32) -> (f32, f32) {
    if r < 1e-12 {
        return (0.0, 0.0);
    }
    let q = r / h;
    let dwdq = if q < 1.0 {
        sigma * (-3.0 * q + 2.25 * q * q)
    } else if q < 2.0 {
        let t = 2.0 - q;
        sigma * (-0.75 * t * t)
    } else {
        0.0
    };
    let dwdr = dwdq / h;
    let factor = dwdr / r;
    (factor * rx, factor * ry)
}

type Grid = HashMap<(i32, i32), Vec<u32>>;

/// Offsets from a center, symmetric by construction: `k` and `-1-k` are
/// exact negatives of each other, so any two particles placed at
/// `center + offset` for a symmetric pair of offsets are exact mirror images
/// (no reliance on float accumulation lining up across a `while` loop).
fn centered_offsets(span: f32, spacing: f32) -> Vec<f32> {
    let count = (span / spacing).round() as i32;
    let half = count / 2;
    (-half..half).map(|k| (k as f32 + 0.5) * spacing).collect()
}

impl Sph {
    pub fn new(params: Params, initial: Initial) -> Self {
        let spacing = params.spacing();
        let m = params.rest_density * spacing * spacing;
        let mut x = Vec::new();
        let mut y = Vec::new();

        match initial {
            Initial::DamBreak => {
                let col_w = 0.25 * params.domain_x();
                let col_h = 0.5 * params.domain_y();
                let mut px = spacing * 0.5;
                while px < col_w {
                    let mut py = spacing * 0.5;
                    while py < col_h {
                        x.push(px);
                        y.push(py);
                        py += spacing;
                    }
                    px += spacing;
                }
            }
            Initial::Drop => {
                let layer_h = 0.2 * params.domain_y();
                let cx = 0.5 * params.domain_x();
                for dx in centered_offsets(params.domain_x(), spacing) {
                    let px = cx + dx;
                    let mut py = spacing * 0.5;
                    while py < layer_h {
                        x.push(px);
                        y.push(py);
                        py += spacing;
                    }
                }
                let blob_r = 0.1 * params.domain_x();
                let cy = layer_h + 3.0 * blob_r;
                let x_offsets = centered_offsets(2.0 * (blob_r + spacing), spacing);
                let y_offsets = centered_offsets(2.0 * (blob_r + spacing), spacing);
                for &dx in &x_offsets {
                    for &dy in &y_offsets {
                        if (dx * dx + dy * dy).sqrt() <= blob_r {
                            x.push(cx + dx);
                            y.push(cy + dy);
                        }
                    }
                }
            }
        }

        let n = x.len();
        let vx = vec![0.0; n];
        let vy = vec![0.0; n];
        let mass = vec![m; n];
        let density = vec![params.rest_density; n];
        let pressure = vec![0.0; n];
        let ax = vec![0.0; n];
        let ay = vec![0.0; n];

        let mut sph = Sph {
            params,
            x,
            y,
            vx,
            vy,
            mass,
            density,
            pressure,
            ax,
            ay,
        };
        let grid = sph.build_grid();
        sph.compute_density(&grid);
        sph.compute_pressure();
        let (ax, ay) = sph.compute_accel(&grid);
        sph.ax = ax;
        sph.ay = ay;
        sph
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn positions(&self) -> (&[f32], &[f32]) {
        (&self.x, &self.y)
    }

    pub fn velocities(&self) -> (&[f32], &[f32]) {
        (&self.vx, &self.vy)
    }

    pub fn densities(&self) -> &[f32] {
        &self.density
    }

    pub fn total_mass(&self) -> f64 {
        self.mass.iter().map(|&m| m as f64).sum()
    }

    pub fn masses(&self) -> &[f32] {
        &self.mass
    }

    fn cell_size(&self) -> f32 {
        2.0 * self.params.h
    }

    fn cell_of(&self, x: f32, y: f32) -> (i32, i32) {
        let c = self.cell_size();
        ((x / c).floor() as i32, (y / c).floor() as i32)
    }

    fn build_grid(&self) -> Grid {
        let mut grid: Grid = HashMap::new();
        for i in 0..self.len() {
            let cell = self.cell_of(self.x[i], self.y[i]);
            grid.entry(cell).or_default().push(i as u32);
        }
        grid
    }

    fn for_each_neighbor(&self, i: usize, grid: &Grid, mut f: impl FnMut(usize, f32, f32, f32)) {
        let ci = self.cell_of(self.x[i], self.y[i]);
        for dcx in -1..=1 {
            for dcy in -1..=1 {
                if let Some(list) = grid.get(&(ci.0 + dcx, ci.1 + dcy)) {
                    for &j in list {
                        let j = j as usize;
                        let rx = self.x[i] - self.x[j];
                        let ry = self.y[i] - self.y[j];
                        let r = (rx * rx + ry * ry).sqrt();
                        if r < 2.0 * self.params.h {
                            f(j, rx, ry, r);
                        }
                    }
                }
            }
        }
    }

    fn compute_density(&mut self, grid: &Grid) {
        let h = self.params.h;
        let sigma = cubic_spline_sigma(h);
        let n = self.len();
        let mut density = vec![0.0; n];
        for (i, rho) in density.iter_mut().enumerate() {
            self.for_each_neighbor(i, grid, |j, _rx, _ry, r| {
                *rho += self.mass[j] * kernel(r, h, sigma);
            });
        }
        self.density = density;
    }

    fn compute_pressure(&mut self) {
        let rho0 = self.params.rest_density;
        let c = self.params.sound_speed;
        let b = rho0 * c * c / TAIT_GAMMA;
        for i in 0..self.len() {
            let ratio = self.density[i] / rho0;
            let p = b * (ratio.powf(TAIT_GAMMA) - 1.0);
            self.pressure[i] = p.max(0.0);
        }
    }

    fn compute_accel(&self, grid: &Grid) -> (Vec<f32>, Vec<f32>) {
        let h = self.params.h;
        let sigma = cubic_spline_sigma(h);
        let c = self.params.sound_speed;
        let n = self.len();
        let mut ax = vec![0.0; n];
        let mut ay = vec![0.0; n];
        for i in 0..n {
            let pi_rho2 = self.pressure[i] / (self.density[i] * self.density[i]);
            let mut axi = 0.0;
            let mut ayi = 0.0;
            self.for_each_neighbor(i, grid, |j, rx, ry, r| {
                if j == i {
                    return;
                }
                let (gx, gy) = kernel_grad(rx, ry, r, h, sigma);
                let pj_rho2 = self.pressure[j] / (self.density[j] * self.density[j]);

                let vx_ij = self.vx[i] - self.vx[j];
                let vy_ij = self.vy[i] - self.vy[j];
                let vr = vx_ij * rx + vy_ij * ry;
                let mut pi_visc = 0.0;
                if vr < 0.0 {
                    let rho_bar = 0.5 * (self.density[i] + self.density[j]);
                    let mu = h * vr / (r * r + 0.01 * h * h);
                    pi_visc = (-self.params.alpha * c * mu + self.params.beta * mu * mu) / rho_bar;
                }

                let coeff = -self.mass[j] * (pi_rho2 + pj_rho2 + pi_visc);
                axi += coeff * gx;
                ayi += coeff * gy;
            });
            ax[i] = axi;
            ay[i] = ayi - self.params.gravity;
        }
        (ax, ay)
    }

    fn xsph_correction(&self, grid: &Grid) -> (Vec<f32>, Vec<f32>) {
        let h = self.params.h;
        let sigma = cubic_spline_sigma(h);
        let eps = self.params.xsph_eps;
        let n = self.len();
        let mut cx = vec![0.0; n];
        let mut cy = vec![0.0; n];
        if eps == 0.0 {
            return (cx, cy);
        }
        for i in 0..n {
            let mut cxi = 0.0;
            let mut cyi = 0.0;
            self.for_each_neighbor(i, grid, |j, _rx, _ry, r| {
                if j == i {
                    return;
                }
                let rho_bar = 0.5 * (self.density[i] + self.density[j]);
                let w = kernel(r, h, sigma);
                cxi += self.mass[j] / rho_bar * (self.vx[j] - self.vx[i]) * w;
                cyi += self.mass[j] / rho_bar * (self.vy[j] - self.vy[i]) * w;
            });
            cx[i] = eps * cxi;
            cy[i] = eps * cyi;
        }
        (cx, cy)
    }

    fn apply_walls(&mut self) {
        let lx = self.params.domain_x();
        let ly = self.params.domain_y();
        for i in 0..self.len() {
            if self.x[i] < 0.0 {
                self.x[i] = -self.x[i];
                self.vx[i] *= -WALL_DAMPING;
            } else if self.x[i] > lx {
                self.x[i] = 2.0 * lx - self.x[i];
                self.vx[i] *= -WALL_DAMPING;
            }
            if self.y[i] < 0.0 {
                self.y[i] = -self.y[i];
                self.vy[i] *= -WALL_DAMPING;
            } else if self.y[i] > ly {
                self.y[i] = 2.0 * ly - self.y[i];
                self.vy[i] *= -WALL_DAMPING;
            }
        }
    }

    pub fn step(&mut self) {
        let dt = self.params.dt;
        let n = self.len();

        let mut vx_half = vec![0.0; n];
        let mut vy_half = vec![0.0; n];
        for i in 0..n {
            vx_half[i] = self.vx[i] + 0.5 * dt * self.ax[i];
            vy_half[i] = self.vy[i] + 0.5 * dt * self.ay[i];
        }

        let grid0 = self.build_grid();
        let (cx, cy) = self.xsph_correction(&grid0);

        for i in 0..n {
            self.x[i] += dt * (vx_half[i] + cx[i]);
            self.y[i] += dt * (vy_half[i] + cy[i]);
        }
        self.vx.copy_from_slice(&vx_half);
        self.vy.copy_from_slice(&vy_half);
        self.apply_walls();

        let grid1 = self.build_grid();
        self.compute_density(&grid1);
        self.compute_pressure();
        let (ax, ay) = self.compute_accel(&grid1);

        for i in 0..n {
            self.vx[i] += 0.5 * dt * ax[i];
            self.vy[i] += 0.5 * dt * ay[i];
        }
        self.ax = ax;
        self.ay = ay;
    }

    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Test-only: build a system directly from particle positions/velocities,
    /// skipping `Initial::{DamBreak,Drop}` (both of which touch a wall by
    /// construction). Used by `splat`'s tests, which need interior
    /// configurations that stay clear of every wall.
    #[cfg(test)]
    pub(crate) fn from_particles(
        params: Params,
        x: Vec<f32>,
        y: Vec<f32>,
        vx: Vec<f32>,
        vy: Vec<f32>,
    ) -> Self {
        let spacing = params.spacing();
        let m = params.rest_density * spacing * spacing;
        let n = x.len();
        let mass = vec![m; n];
        let density = vec![params.rest_density; n];
        let pressure = vec![0.0; n];
        let ax = vec![0.0; n];
        let ay = vec![0.0; n];
        let mut sph = Sph {
            params,
            x,
            y,
            vx,
            vy,
            mass,
            density,
            pressure,
            ax,
            ay,
        };
        let grid = sph.build_grid();
        sph.compute_density(&grid);
        sph.compute_pressure();
        let (ax, ay) = sph.compute_accel(&grid);
        sph.ax = ax;
        sph.ay = ay;
        sph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params::default()
    }

    #[test]
    fn particle_count_and_mass_conserved() {
        let p = params();
        let mut sph = Sph::new(p, Initial::DamBreak);
        let n0 = sph.len();
        let m0 = sph.total_mass();
        sph.steps(200);
        assert_eq!(sph.len(), n0);
        assert!(
            ((sph.total_mass() - m0) / m0).abs() < 1e-9,
            "mass changed: {m0} -> {}",
            sph.total_mass()
        );
    }

    #[test]
    fn dam_break_stays_inside_box() {
        let p = params();
        let mut sph = Sph::new(p, Initial::DamBreak);
        sph.steps(2000);
        let (x, y) = sph.positions();
        let margin = p.h;
        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            assert!(xi.is_finite() && yi.is_finite(), "particle {i} is NaN/Inf");
            assert!(
                xi >= -margin && xi <= p.domain_x() + margin,
                "particle {i} escaped in x: {xi}"
            );
            assert!(
                yi >= -margin && yi <= p.domain_y() + margin,
                "particle {i} escaped in y: {yi}"
            );
        }
    }

    #[test]
    fn symmetric_drop_stays_symmetric() {
        let p = params();
        let sph0 = Sph::new(p, Initial::Drop);
        let (x0, y0) = sph0.positions();
        let x0 = x0.to_vec();
        let y0 = y0.to_vec();
        let n = sph0.len();
        let cx = 0.5 * p.domain_x();

        // Pair each particle with its mirror image about the domain's
        // vertical centerline, using the (mirror-symmetric) initial layout.
        let mut mirror = vec![usize::MAX; n];
        for i in 0..n {
            let target_x = 2.0 * cx - x0[i];
            let mut best = usize::MAX;
            let mut best_d = f32::INFINITY;
            for j in 0..n {
                let d = (x0[j] - target_x).abs() + (y0[j] - y0[i]).abs();
                if d < best_d {
                    best_d = d;
                    best = j;
                }
            }
            assert!(best_d < 1e-3, "no mirror partner found for particle {i}");
            mirror[i] = best;
        }

        let mut sph = sph0;
        sph.steps(50);
        let (x, y) = sph.positions();
        let (vx, vy) = sph.velocities();
        for i in 0..n {
            let j = mirror[i];
            assert!(
                (x[i] - (2.0 * cx - x[j])).abs() < 1e-2,
                "x not mirror-symmetric at {i}/{j}: {} vs {}",
                x[i],
                x[j]
            );
            assert!(
                (y[i] - y[j]).abs() < 1e-2,
                "y not symmetric at {i}/{j}: {} vs {}",
                y[i],
                y[j]
            );
            assert!(
                (vx[i] + vx[j]).abs() < 1e-2,
                "vx not antisymmetric at {i}/{j}"
            );
            assert!((vy[i] - vy[j]).abs() < 1e-2, "vy not symmetric at {i}/{j}");
        }
    }

    #[test]
    fn density_near_reference_at_rest() {
        let p = params();
        let mut sph = Sph::new(p, Initial::Drop);
        sph.steps(50);
        let (x, y) = sph.positions();
        let density = sph.densities();
        let rho0 = p.rest_density;
        // Only check particles away from walls / free surface, where the
        // kernel support isn't truncated and the SPH density estimate is
        // accurate.
        let mut checked = 0;
        for i in 0..sph.len() {
            if x[i] > 2.0 * p.h
                && x[i] < p.domain_x() - 2.0 * p.h
                && y[i] > 2.0 * p.h
                && y[i] < 0.15 * p.domain_y()
            {
                assert!(
                    (density[i] - rho0).abs() / rho0 < 0.1,
                    "density far from rest at {i}: {}",
                    density[i]
                );
                checked += 1;
            }
        }
        assert!(checked > 10, "not enough interior particles checked");
    }
}
