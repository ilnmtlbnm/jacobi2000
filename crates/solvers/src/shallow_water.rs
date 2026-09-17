//! 2D shallow water on a collocated grid with reflecting walls.
//!
//! Fields `h` (total depth), `u`, `v` all live at cell centres, row-major,
//! index `y * nx + x`. The scheme is forward-backward: continuity is advanced
//! first in flux form (so mass telescopes to zero flux at the walls and is
//! conserved to float rounding), then momentum is advanced with the *new*
//! surface, which is what makes the gravity-wave mode neutrally stable
//! instead of unconditionally growing.
//!
//! Walls are imposed through ghost-cell symmetry rather than special-cased
//! loops: `h` is even across a wall, the normal velocity is odd, the
//! tangential velocity is even. That keeps the update identical in every
//! cell, which is what makes a mirror-symmetric initial condition stay
//! mirror-symmetric.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub nx: usize,
    pub ny: usize,
    /// Cell size, same in x and y.
    pub dx: f32,
    pub dt: f32,
    /// Gravity.
    pub g: f32,
    /// Undisturbed depth.
    pub depth: f32,
    /// Kinematic viscosity on the momentum equations. 0 = inviscid.
    pub nu: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            nx: 64,
            ny: 64,
            dx: 1.0,
            dt: 0.0,
            g: 9.81,
            depth: 1.0,
            nu: 0.0,
        }
        .with_cfl_dt(0.4)
    }
}

impl Params {
    /// Largest stable dt for gravity waves on this grid, scaled by `safety`.
    /// `c = sqrt(g * depth)`, 2D CFL limit `dx / (c * sqrt(2))`.
    pub fn cfl_dt(&self, safety: f32) -> f32 {
        let c = (self.g * self.depth).sqrt();
        safety * self.dx / (c * std::f32::consts::SQRT_2)
    }

    pub fn with_cfl_dt(mut self, safety: f32) -> Self {
        self.dt = self.cfl_dt(safety);
        self
    }

    pub fn cells(&self) -> usize {
        self.nx * self.ny
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub h: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
}

impl State {
    pub fn flat(p: &Params) -> Self {
        State {
            h: vec![p.depth; p.cells()],
            u: vec![0.0; p.cells()],
            v: vec![0.0; p.cells()],
        }
    }

    pub fn mass(&self) -> f64 {
        self.h.iter().map(|&x| x as f64).sum()
    }

    /// Add a Gaussian bump of amplitude `amp` and radius `sigma` (in cells)
    /// centred on `(cx, cy)`, also in cells.
    pub fn add_drop(&mut self, p: &Params, cx: f32, cy: f32, amp: f32, sigma: f32) {
        let inv = 1.0 / (2.0 * sigma * sigma);
        for y in 0..p.ny {
            for x in 0..p.nx {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                self.h[y * p.nx + x] += amp * (-(dx * dx + dy * dy) * inv).exp();
            }
        }
    }
}

/// Ghost-cell index: reflect about the wall (cell -1 reads cell 0).
#[inline]
fn mirror(i: isize, n: usize) -> usize {
    if i < 0 {
        0
    } else if i as usize >= n {
        n - 1
    } else {
        i as usize
    }
}

pub struct ShallowWater {
    pub params: Params,
    fx: Vec<f32>,
    fy: Vec<f32>,
    h_new: Vec<f32>,
    u_new: Vec<f32>,
    v_new: Vec<f32>,
}

impl ShallowWater {
    pub fn new(params: Params) -> Self {
        let n = params.cells();
        ShallowWater {
            params,
            // Face fluxes: (nx + 1) x ny and nx x (ny + 1).
            fx: vec![0.0; (params.nx + 1) * params.ny],
            fy: vec![0.0; params.nx * (params.ny + 1)],
            h_new: vec![0.0; n],
            u_new: vec![0.0; n],
            v_new: vec![0.0; n],
        }
    }

    pub fn step(&mut self, s: &mut State) {
        let p = self.params;
        let (nx, ny) = (p.nx, p.ny);
        let idx = |x: usize, y: usize| y * nx + x;

        // Continuity, flux form. Wall faces carry zero flux, so the sum of
        // the divergences telescopes to zero and mass is conserved.
        for y in 0..ny {
            for x in 1..nx {
                let a = idx(x - 1, y);
                let b = idx(x, y);
                self.fx[y * (nx + 1) + x] = 0.5 * (s.h[a] + s.h[b]) * 0.5 * (s.u[a] + s.u[b]);
            }
            self.fx[y * (nx + 1)] = 0.0;
            self.fx[y * (nx + 1) + nx] = 0.0;
        }
        for x in 0..nx {
            for y in 1..ny {
                let a = idx(x, y - 1);
                let b = idx(x, y);
                self.fy[y * nx + x] = 0.5 * (s.h[a] + s.h[b]) * 0.5 * (s.v[a] + s.v[b]);
            }
            self.fy[x] = 0.0;
            self.fy[ny * nx + x] = 0.0;
        }

        let r = p.dt / p.dx;
        for y in 0..ny {
            for x in 0..nx {
                let dfx = self.fx[y * (nx + 1) + x + 1] - self.fx[y * (nx + 1) + x];
                let dfy = self.fy[(y + 1) * nx + x] - self.fy[y * nx + x];
                self.h_new[idx(x, y)] = s.h[idx(x, y)] - r * (dfx + dfy);
            }
        }

        // Momentum, centred differences on the updated surface.
        // Ghost rules: h even, u odd in x / even in y, v even in x / odd in y.
        let inv2dx = 1.0 / (2.0 * p.dx);
        let invdx2 = 1.0 / (p.dx * p.dx);
        for y in 0..ny {
            for x in 0..nx {
                let i = idx(x, y);
                let (xm, xp) = (x as isize - 1, x as isize + 1);
                let (ym, yp) = (y as isize - 1, y as isize + 1);

                let sxm = if x == 0 { -1.0 } else { 1.0 };
                let sxp = if x + 1 == nx { -1.0 } else { 1.0 };
                let sym = if y == 0 { -1.0 } else { 1.0 };
                let syp = if y + 1 == ny { -1.0 } else { 1.0 };

                let hl = self.h_new[idx(mirror(xm, nx), y)];
                let hr = self.h_new[idx(mirror(xp, nx), y)];
                let hd = self.h_new[idx(x, mirror(ym, ny))];
                let hu = self.h_new[idx(x, mirror(yp, ny))];

                // u: odd across x walls, even across y walls.
                let ul = sxm * s.u[idx(mirror(xm, nx), y)];
                let ur = sxp * s.u[idx(mirror(xp, nx), y)];
                let ud = s.u[idx(x, mirror(ym, ny))];
                let uu = s.u[idx(x, mirror(yp, ny))];
                // v: even across x walls, odd across y walls.
                let vl = s.v[idx(mirror(xm, nx), y)];
                let vr = s.v[idx(mirror(xp, nx), y)];
                let vd = sym * s.v[idx(x, mirror(ym, ny))];
                let vu = syp * s.v[idx(x, mirror(yp, ny))];

                let u0 = s.u[i];
                let v0 = s.v[i];

                let dudx = (ur - ul) * inv2dx;
                let dudy = (uu - ud) * inv2dx;
                let dvdx = (vr - vl) * inv2dx;
                let dvdy = (vu - vd) * inv2dx;
                let dhdx = (hr - hl) * inv2dx;
                let dhdy = (hu - hd) * inv2dx;

                let lap_u = (ul + ur + ud + uu - 4.0 * u0) * invdx2;
                let lap_v = (vl + vr + vd + vu - 4.0 * v0) * invdx2;

                self.u_new[i] =
                    u0 + p.dt * (-u0 * dudx - v0 * dudy - p.g * dhdx + p.nu * lap_u);
                self.v_new[i] =
                    v0 + p.dt * (-u0 * dvdx - v0 * dvdy - p.g * dhdy + p.nu * lap_v);
            }
        }

        s.h.copy_from_slice(&self.h_new);
        s.u.copy_from_slice(&self.u_new);
        s.v.copy_from_slice(&self.v_new);
    }

    pub fn steps(&mut self, s: &mut State, n: usize) {
        for _ in 0..n {
            self.step(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params {
            nx: 64,
            ny: 64,
            nu: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn flat_stays_flat() {
        let p = params();
        let mut sw = ShallowWater::new(p);
        let mut s = State::flat(&p);
        sw.steps(&mut s, 200);
        for i in 0..p.cells() {
            assert_eq!(s.h[i], p.depth);
            assert_eq!(s.u[i], 0.0);
            assert_eq!(s.v[i], 0.0);
        }
    }

    #[test]
    fn mass_is_conserved() {
        let p = params();
        let mut sw = ShallowWater::new(p);
        let mut s = State::flat(&p);
        s.add_drop(&p, 20.0, 30.0, 0.05, 4.0);
        let m0 = s.mass();
        sw.steps(&mut s, 500);
        let m1 = s.mass();
        // Relative drift is pure float rounding over 500 x 4096 f32 updates
        // (observed ~1.6e-8, an order below f32 eps).
        assert!(
            ((m1 - m0) / m0).abs() < 1e-7,
            "mass drifted: {m0} -> {m1}"
        );
    }

    #[test]
    fn symmetric_drop_stays_symmetric() {
        let p = params();
        let mut sw = ShallowWater::new(p);
        let mut s = State::flat(&p);
        // Centre of a 64-wide grid sits between cells 31 and 32.
        s.add_drop(&p, 31.5, 31.5, 0.05, 5.0);
        sw.steps(&mut s, 120);
        for y in 0..p.ny {
            for x in 0..p.nx {
                let i = y * p.nx + x;
                let mx = y * p.nx + (p.nx - 1 - x);
                let my = (p.ny - 1 - y) * p.nx + x;
                assert!((s.h[i] - s.h[mx]).abs() < 1e-6, "h not x-symmetric at {x},{y}");
                assert!((s.h[i] - s.h[my]).abs() < 1e-6, "h not y-symmetric at {x},{y}");
                assert!((s.u[i] + s.u[mx]).abs() < 1e-6, "u not x-antisymmetric");
                assert!((s.v[i] + s.v[my]).abs() < 1e-6, "v not y-antisymmetric");
            }
        }
    }

    #[test]
    fn drop_stays_bounded_over_a_long_run() {
        // Guards the forward-backward ordering: doing momentum first turns
        // this into an exponentially growing checkerboard.
        let p = Params { nu: 1e-3, ..params() };
        let mut sw = ShallowWater::new(p);
        let mut s = State::flat(&p);
        s.add_drop(&p, 24.0, 40.0, 0.08, 4.0);
        sw.steps(&mut s, 2000);
        let max = s.h.iter().fold(0.0f32, |a, &b| a.max((b - p.depth).abs()));
        assert!(max.is_finite() && max < 0.08, "amplitude grew to {max}");
    }
}
