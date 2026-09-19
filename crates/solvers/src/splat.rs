//! Particle fields -> grid fields, on the same `nx x ny` grid convention as
//! `shallow_water` (row-major `y*nx+x`; cell `(x,y)` sits at physical
//! position `(x*dx, y*dx)`, matching `shallow_water::State::add_drop`'s
//! cell-coordinate convention and `sph::Params`'s shared box).
//!
//! `h` is a column-height-like density proxy, `Σ m_j W(|pos-x_j|,h) / rho0`,
//! deliberately *not* Shepard-normalized: its area integral over the grid
//! reproduces `Σ m_j`, which is what makes total splatted mass track total
//! particle mass (the point of the field — it stands in for shallow water's
//! `h`, which is a depth/volume proxy, not an intensive quantity). `u`, `v`
//! are Shepard-normalized mass-weighted kernel averages — an intensive
//! quantity, so normalizing by the local weight sum is correct there — and
//! are left at zero wherever that weight sum is negligible (no particle
//! within kernel support of the cell).
//!
//! Same Monaghan cubic spline kernel as `sph`, support radius `2h`.

use crate::sph::{cubic_spline_sigma, kernel, Sph};

const WEIGHT_EPS: f32 = 1e-6;

pub struct SplatField {
    pub h: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
}

impl SplatField {
    /// `Σ h * rho0 * dx^2` over the grid: the mass the `h` field represents.
    pub fn total_mass(&self, dx: f32, rho0: f32) -> f64 {
        let cell_area = (dx * dx) as f64;
        self.h.iter().map(|&h| h as f64 * rho0 as f64 * cell_area).sum()
    }
}

/// Splat `sph`'s particles onto an `nx x ny` grid of spacing `dx`, using
/// smoothing length `h_smooth` (independent of the SPH solver's own `h`, so a
/// coarser/finer grid can use a different kernel width) and reference density
/// `rho0`.
pub fn splat(sph: &Sph, nx: usize, ny: usize, dx: f32, h_smooth: f32, rho0: f32) -> SplatField {
    let (px, py) = sph.positions();
    let (vx, vy) = sph.velocities();
    let mass = sph.masses();
    let sigma = cubic_spline_sigma(h_smooth);
    let support = 2.0 * h_smooth;

    let mut h_field = vec![0.0f32; nx * ny];
    let mut u_num = vec![0.0f32; nx * ny];
    let mut v_num = vec![0.0f32; nx * ny];
    let mut wsum = vec![0.0f32; nx * ny];

    let reach = (support / dx).ceil() as isize + 1;

    for j in 0..px.len() {
        let cx = (px[j] / dx).round() as isize;
        let cy = (py[j] / dx).round() as isize;
        let gx_lo = (cx - reach).max(0);
        let gx_hi = (cx + reach).min(nx as isize - 1);
        let gy_lo = (cy - reach).max(0);
        let gy_hi = (cy + reach).min(ny as isize - 1);
        for gy in gy_lo..=gy_hi {
            let celly = gy as f32 * dx;
            let ry0 = celly - py[j];
            for gx in gx_lo..=gx_hi {
                let cellx = gx as f32 * dx;
                let rx = cellx - px[j];
                let r = (rx * rx + ry0 * ry0).sqrt();
                if r >= support {
                    continue;
                }
                let w = kernel(r, h_smooth, sigma);
                let idx = gy as usize * nx + gx as usize;
                let mw = mass[j] * w;
                h_field[idx] += mw / rho0;
                u_num[idx] += mw * vx[j];
                v_num[idx] += mw * vy[j];
                wsum[idx] += mw;
            }
        }
    }

    let mut u_field = vec![0.0f32; nx * ny];
    let mut v_field = vec![0.0f32; nx * ny];
    for i in 0..nx * ny {
        if wsum[i] > WEIGHT_EPS {
            u_field[i] = u_num[i] / wsum[i];
            v_field[i] = v_num[i] / wsum[i];
        }
    }

    SplatField {
        h: h_field,
        u: u_field,
        v: v_field,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sph::Params;

    fn params() -> Params {
        Params {
            nx: 64,
            ny: 64,
            dx: 1.0,
            h: 1.3,
            ..Params::default()
        }
    }

    /// A block of particles on a regular grid, centred at `(cx, cy)`, far
    /// from every wall — the "interior configuration" the mass test needs
    /// (both `Initial::DamBreak` and `Initial::Drop` touch a wall).
    fn interior_block(p: &Params, cx: f32, cy: f32, half_w: f32, half_h: f32, spacing: f32, v: (f32, f32)) -> Sph {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut px = -half_w + spacing * 0.5;
        while px < half_w {
            let mut py = -half_h + spacing * 0.5;
            while py < half_h {
                x.push(cx + px);
                y.push(cy + py);
                py += spacing;
            }
            px += spacing;
        }
        let n = x.len();
        let vx = vec![v.0; n];
        let vy = vec![v.1; n];
        Sph::from_particles(*p, x, y, vx, vy)
    }

    #[test]
    fn total_mass_matches_particle_mass_for_interior_configuration() {
        let p = params();
        let spacing = p.h / 1.3;
        let sph = interior_block(&p, 32.0, 32.0, 10.0, 10.0, spacing, (0.0, 0.0));
        let field = splat(&sph, p.nx, p.ny, p.dx, p.h, p.rest_density);
        let m_particles = sph.total_mass();
        let m_field = field.total_mass(p.dx, p.rest_density);
        assert!(
            ((m_field - m_particles) / m_particles).abs() < 0.01,
            "splatted mass {m_field} vs particle mass {m_particles}"
        );
    }

    #[test]
    fn resting_flat_layer_splats_flat_away_from_walls_and_zero_above_surface() {
        let p = params();
        let spacing = p.h / 1.3;
        let layer_h = 0.2 * p.domain_y();
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut px = spacing * 0.5;
        while px < p.domain_x() {
            let mut py = spacing * 0.5;
            while py < layer_h {
                x.push(px);
                y.push(py);
                py += spacing;
            }
            px += spacing;
        }
        let n = x.len();
        let vx = vec![0.0; n];
        let vy = vec![0.0; n];
        let sph = Sph::from_particles(p, x, y, vx, vy);
        let field = splat(&sph, p.nx, p.ny, p.dx, p.h, p.rest_density);

        // Flat within 5% away from the side walls, well below the surface.
        let margin = 2.0 * p.h;
        let mut reference: Option<f32> = None;
        for gy in (margin.ceil() as usize)..((layer_h - margin).floor() as usize) {
            for gx in (margin.ceil() as usize)..(p.nx - margin.ceil() as usize) {
                let h = field.h[gy * p.nx + gx];
                match reference {
                    None => reference = Some(h),
                    Some(r) => {
                        assert!(
                            ((h - r) / r).abs() < 0.05,
                            "not flat at ({gx},{gy}): {h} vs reference {r}"
                        );
                    }
                }
            }
        }
        assert!(reference.is_some(), "no interior cells checked");

        // Zero well above the free surface.
        let above = (layer_h + 4.0 * p.h).ceil() as usize;
        for gy in above.min(p.ny - 1)..p.ny {
            for gx in 0..p.nx {
                let h = field.h[gy * p.nx + gx];
                assert!(h.abs() < 1e-6, "nonzero above surface at ({gx},{gy}): {h}");
            }
        }
    }

    #[test]
    fn moving_blob_mass_weighted_mean_matches_blob_velocity() {
        let p = params();
        let spacing = p.h / 1.3;
        let v = (2.0, -1.5);
        let sph = interior_block(&p, 32.0, 32.0, 8.0, 8.0, spacing, v);
        let field = splat(&sph, p.nx, p.ny, p.dx, p.h, p.rest_density);

        // Mass-weighted mean over the splatted field, weighting each cell by
        // its own h (proportional to the local kernel weight sum), which
        // recovers the particles' own mass-weighted mean velocity as long as
        // the kernel support isn't truncated (interior configuration).
        let mut wu = 0.0f64;
        let mut wv = 0.0f64;
        let mut wsum = 0.0f64;
        for i in 0..p.nx * p.ny {
            let w = field.h[i] as f64;
            wu += w * field.u[i] as f64;
            wv += w * field.v[i] as f64;
            wsum += w;
        }
        let mean_u = (wu / wsum) as f32;
        let mean_v = (wv / wsum) as f32;
        assert!(
            ((mean_u - v.0) / v.0).abs() < 0.01,
            "mean u {mean_u} vs blob {}",
            v.0
        );
        assert!(
            ((mean_v - v.1) / v.1).abs() < 0.01,
            "mean v {mean_v} vs blob {}",
            v.1
        );
    }
}
