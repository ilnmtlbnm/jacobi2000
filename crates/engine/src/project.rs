//! Hard constraints, mirroring `training/jacobi2000_pt/project.py`. Batch
//! size 1 only (the one `crates/web` ever runs); state is `[3, ny, nx]` flat
//! in normalized units, channel 0 = h, 1 = u, 2 = v.

use crate::norm::Denorm;

/// Potential (`1/2 g h^2`) and kinetic (`1/2 h (u^2+v^2)`) energy, summed
/// over cells, physical units.
fn potential_kinetic(x: &[f32], d: &Denorm, g: f32, n: usize) -> (f32, f32) {
    let (mut potential, mut kinetic) = (0.0f32, 0.0f32);
    for i in 0..n {
        let h = x[i] * d.std[0] + (d.mean[0] + d.depth);
        let u = x[n + i] * d.std[1] + d.mean[1];
        let v = x[2 * n + i] * d.std[2] + d.mean[2];
        potential += 0.5 * g * h * h;
        kinetic += 0.5 * h * (u * u + v * v);
    }
    (potential, kinetic)
}

/// Total energy, physical units.
pub fn energy(x: &[f32], d: &Denorm, g: f32, n: usize) -> f32 {
    let (p, k) = potential_kinetic(x, d, g, n);
    p + k
}

/// Box mean of the normalized channel 0.
pub fn mass(x: &[f32], n: usize) -> f32 {
    x[..n].iter().sum::<f32>() / n as f32
}

/// Shift channel 0 by a constant so its box mean equals `target`.
pub fn project_mass(x: &mut [f32], target: f32, n: usize) {
    let delta = target - mass(x, n);
    for v in &mut x[..n] {
        *v += delta;
    }
}

/// Scale `(u, v)` uniformly down to the energy bound `prev_energy * r_max`
/// when exceeded; never scales up. Same clamp and
/// `1e-12` kinetic floor as `project.py`.
pub fn project_energy(x: &mut [f32], prev_energy: f32, d: &Denorm, g: f32, r_max: f32, n: usize) {
    let bound = prev_energy * r_max;
    let (potential, kinetic) = potential_kinetic(x, d, g, n);
    let budget = (bound - potential).max(0.0);
    let scale = (budget / kinetic.max(1e-12)).sqrt().clamp(0.0, 1.0);
    for i in 0..n {
        let u = x[n + i] * d.std[1] + d.mean[1];
        let v = x[2 * n + i] * d.std[2] + d.mean[2];
        x[n + i] = (u * scale - d.mean[1]) / d.std[1];
        x[2 * n + i] = (v * scale - d.mean[2]) / d.std[2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_denorm() -> Denorm {
        Denorm { mean: [0.0; 3], std: [1.0; 3], depth: 0.0 }
    }

    #[test]
    fn mass_projection_restores_the_target_and_leaves_uv_alone() {
        let n = 4;
        let mut x = vec![0.1, 0.2, 0.3, 0.4, 5.0, 5.0, 5.0, 5.0, -1.0, -1.0, -1.0, -1.0];
        let u_before = x[n..2 * n].to_vec();
        project_mass(&mut x, 0.0, n);
        assert!((mass(&x, n) - 0.0).abs() < 1e-6);
        assert_eq!(&x[n..2 * n], u_before.as_slice());
    }

    #[test]
    fn energy_projection_scales_down_to_the_bound() {
        let d = identity_denorm();
        let g = 9.81;
        let n = 16;
        let mut x = vec![1.0f32; n]; // h = 1
        x.extend(vec![2.0f32; n]); // u = 2
        x.extend(vec![2.0f32; n]); // v = 2
        // potential = 16 * 0.5*9.81 = 78.48, kinetic = 16*0.5*1*(4+4) = 64,
        // total 142.48, well over a 100 bound.
        project_energy(&mut x, 1.0, &d, g, 100.0, n);
        let e = energy(&x, &d, g, n);
        assert!(e <= 100.0 + 1e-3, "energy {e} exceeds the bound");
        assert!(e > 99.0, "projection over-corrected: {e}");
        assert!((mass(&x, n) - 1.0).abs() < 1e-6, "mass changed");
    }

    #[test]
    fn energy_projection_is_identity_within_the_bound() {
        let d = identity_denorm();
        let g = 9.81;
        let n = 16;
        let mut x = vec![1.0f32; n];
        x.extend(vec![2.0f32; n]);
        x.extend(vec![2.0f32; n]);
        let before = x.clone();
        project_energy(&mut x, 2.0, &d, g, 100.0, n);
        assert_eq!(x, before);
    }
}
