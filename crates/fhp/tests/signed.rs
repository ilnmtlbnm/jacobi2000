//! Signed ternary FHP: equivalence with 3A on a global reference, exact
//! conservation of the represented mass and momentum, unbiased rounding,
//! streaming and bounce-back.

#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

use fhp::signed::{self, feq, feq_fixed, fixed_values, velocity, Params, Signed, OBST, ONE, ZERO};
use fhp::ternary::engine::{self, Ternary};
use fhp::ternary::{self, Rule, NCH};
use fhp::{lowbias32, NDIR};

fn params(rule: Rule, block: usize, every: u32) -> Params {
    Params { seed: 7, rule, block, every, n0: 3.0, boundary: None }
}

fn random_cells(w: usize, h: usize, obstacle: bool, seed: u32) -> Vec<u16> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let s = (lowbias32(seed ^ lowbias32(i as u32)) % 2187) as u16;
            if obstacle && (40..48).contains(&x) && (8..16).contains(&y) {
                OBST | s
            } else {
                s
            }
        })
        .collect()
}

/// A smooth reference: a shear flow and a density bump, by sampling.
fn smooth(w: usize, h: usize, p: Params, obstacle: bool, seed: u32) -> Signed {
    let ky = 2.0 * std::f64::consts::PI / (h as f64 * 3f64.sqrt() / 2.0);
    let kx = 2.0 * std::f64::consts::PI / w as f64;
    let rho0 = 7.0 * p.n0;
    Signed::sample(
        w,
        h,
        p,
        |x, y| obstacle && (40..48).contains(&x) && (8..16).contains(&y),
        |x, y| (rho0 * (1.0 + 0.05 * (kx * x).cos()), 0.08 * (ky * y).sin(), 0.04 * (kx * x).sin()),
        seed,
    )
}

#[test]
fn reference_equilibrium_has_exact_moments() {
    for &(r, ux, uy) in &[(7.0, 0.0, 0.0), (21.0, 0.2, -0.1), (3.5, -0.13, 0.27)] {
        let f = feq(r, ux, uy);
        let (mut m, mut jx, mut jy, mut pxx, mut pxy, mut pyy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..NCH {
            let c = velocity(i);
            m += f[i];
            jx += c.0 * f[i];
            jy += c.1 * f[i];
            pxx += c.0 * c.0 * f[i];
            pxy += c.0 * c.1 * f[i];
            pyy += c.1 * c.1 * f[i];
        }
        let p = 3.0 / 7.0 * r;
        for (a, b) in [(m, r), (jx, r * ux), (jy, r * uy), (pxx, p + r * ux * ux), (pxy, r * ux * uy), (pyy, p + r * uy * uy)] {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }
}

/// The fixed-point equilibrium has the float one's moments to a few units
/// of 2^-12 (one truncation per channel, u to 2^-16).
#[test]
fn fixed_equilibrium_matches_the_float_one() {
    let q = ONE as f64;
    for &(r, ux, uy) in &[(28.0, 0.0, 0.0), (21.0, 0.2, -0.1), (3.5, -0.13, 0.27), (40.0, 0.35, 0.05), (12.0, -0.3, -0.2)] {
        let [fr, fjx, fjy] = fixed_values(r, ux, uy);
        let fx = feq_fixed(fr, fjx, fjy);
        let ff = feq(r, ux, uy);
        for i in 0..NCH {
            assert!((fx[i] as f64 / q - ff[i]).abs() < 3.0 / q, "({r}, {ux}, {uy}) channel {i}: {} vs {}", fx[i] as f64 / q, ff[i]);
        }
        let m: i32 = fx.iter().sum();
        let jx: i32 = (0..NDIR).map(|c| fhp::CX2[c] * fx[c]).sum();
        let jy: i32 = (0..NDIR).map(|c| fhp::CY2[c] * fx[c]).sum();
        assert!((m - fr).abs() <= 7 && (jx - fjx).abs() <= 12 && (jy - fjy).abs() <= 8, "({r}, {ux}, {uy}): {m} {jx} {jy} vs {fr} {fjx} {fjy}");
    }
    // At rest with integer n0 the reference is exact.
    assert_eq!(feq_fixed(28 * ONE, 0, 0), [4 * ONE; NCH]);
}

#[test]
fn global_fixed_reference_is_ternary_bit_for_bit() {
    let (w, h) = (128, 32);
    for rule in [Rule::Optimal, Rule::Chiral, Rule::Other] {
        let cells = random_cells(w, h, true, 3);
        let mut t = Ternary::new(w, h, cells.clone(), engine::Params { seed: 7, rule, boundary: None });
        let mut s = Signed::new(w, h, cells, params(rule, 16, 0));
        for step in 0..60 {
            t.step();
            s.step();
            assert_eq!(t.cells, s.cells, "rule {rule:?} step {step}");
        }
        assert_eq!(s.kicks, 0);
        assert_eq!(s.saturated, 0);
    }
}

#[test]
fn streaming_moves_deviations_and_obstacles_bounce_back() {
    // Uniform reference: each site's mass and momentum after one step are
    // those streamed in (collisions conserve both per site); obstacle sites
    // hold the reversed streamed state.
    let (w, h) = (128, 32);
    let cells = random_cells(w, h, true, 5);
    let mut s = Signed::new(w, h, cells.clone(), params(Rule::Optimal, 8, 0));
    let before: Vec<[i8; NCH]> = (0..w * h).map(|i| s.deltas(i)).collect();
    s.step();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let mut inn = [0i8; NCH];
            inn[6] = before[i][6];
            for d in 0..NDIR {
                let (sx, sy) = fhp::source(x, y, d, w, h);
                inn[d] = before[sy * w + sx][d];
            }
            let after = s.deltas(i);
            if cells[i] & OBST != 0 {
                assert_ne!(s.cells[i] & OBST, 0);
                let mut rev = [0i8; NCH];
                rev[6] = inn[6];
                for d in 0..NDIR {
                    rev[(d + 3) % 6] = inn[d];
                }
                assert_eq!(after, rev, "obstacle site {x},{y}");
            } else {
                let m = |v: &[i8; NCH]| v.iter().map(|&a| a as i32).sum::<i32>();
                let p = |v: &[i8; NCH]| {
                    (0..NDIR).fold((0, 0), |a, d| (a.0 + v[d] as i32 * fhp::CX2[d], a.1 + v[d] as i32 * fhp::CY2[d]))
                };
                assert_eq!(m(&after), m(&inn));
                assert_eq!(p(&after), p(&inn));
            }
        }
    }
}

#[test]
fn represented_mass_and_momentum_are_conserved_exactly() {
    let (w, h) = (128, 64);
    for (block, every) in [(1, 1), (4, 1), (8, 4), (8, 0), (16, 3)] {
        let mut s = smooth(w, h, params(Rule::Optimal, block, every), false, 11);
        let t0 = s.totals_fixed();
        for _ in 0..40 {
            s.step();
            assert_eq!(s.totals_fixed(), t0, "B {block} N {every}");
        }
        assert!(s.kicks > 0);
    }
    // With an obstacle: mass exact; obstacle sites keep the rest reference.
    let mut s = smooth(w, h, params(Rule::Optimal, 8, 2), true, 13);
    let m0 = s.totals_fixed().0;
    let rest = feq_fixed(s.params.rho0(), 0, 0);
    for _ in 0..40 {
        s.step();
        assert_eq!(s.totals_fixed().0, m0);
    }
    for i in 0..w * h {
        if s.cells[i] & OBST != 0 {
            assert_eq!(s.phi[i], rest);
        }
    }
}

#[test]
fn requantization_conserves_every_block_exactly() {
    let (w, h) = (128, 64);
    let mut s = smooth(w, h, params(Rule::Optimal, 8, 0), true, 17);
    s.run(30);
    let moments = |c: &[i32; NCH]| {
        let m: i64 = c.iter().map(|&v| v as i64).sum();
        let jx: i64 = (0..NDIR).map(|d| fhp::CX2[d] as i64 * c[d] as i64).sum();
        let jy: i64 = (0..NDIR).map(|d| fhp::CY2[d] as i64 * c[d] as i64).sum();
        (m, jx, jy)
    };
    let before = s.block_content();
    s.reestimate(false);
    let after = s.block_content();
    for k in 0..before.len() {
        assert_eq!(moments(&before[k]), moments(&after[k]), "block {k}");
    }
}

/// Mean over seeds of per-site represented values against the exact ones,
/// as z-scores with the variance of the stochastic rounding (p (1 - p) per
/// rounding of fractional part p): streaming (site mass and momentum, which
/// collisions keep) and re-quantization (every channel) are unbiased.
#[test]
fn rounding_is_unbiased() {
    let (w, h) = (64, 32);
    let seeds = 400u32;
    let base = smooth(w, h, params(Rule::Optimal, 4, 0), false, 19);
    // Zero deviations, so no rounding can saturate.
    let cells0 = vec![ZERO; w * h];
    let phi = base.phi.clone();
    let q = ONE as f64;
    let fphi: Vec<[f64; NCH]> = phi.iter().map(|p| p.map(|v| v as f64 / q)).collect();
    let pv = |r: f64| {
        let f = r - r.floor();
        f * (1.0 - f)
    };
    // Exact streamed site moments and their rounding variances.
    let mut exact = vec![[0.0f64; 3]; w * h];
    let mut var = vec![[0.0f64; 3]; w * h];
    for i in 0..w * h {
        let (x, y) = (i % w, i / w);
        exact[i][0] = fphi[i][6];
        for d in 0..NDIR {
            let (sx, sy) = fhp::source(x, y, d, w, h);
            let f = fphi[sy * w + sx][d];
            let c = velocity(d);
            let v = pv(f - fphi[i][d]);
            exact[i][0] += f;
            exact[i][1] += c.0 * f;
            exact[i][2] += c.1 * f;
            var[i][0] += v;
            var[i][1] += c.0 * c.0 * v;
            var[i][2] += c.1 * c.1 * v;
        }
    }
    let mut acc = vec![[0.0f64; 3]; w * h];
    for seed in 0..seeds {
        let mut s = Signed::new(w, h, cells0.clone(), Params { seed, ..base.params });
        s.phi = phi.clone();
        s.step();
        assert_eq!(s.saturated, 0);
        for i in 0..w * h {
            let d = s.deltas(i);
            for ch in 0..NCH {
                let f = s.phi[i][ch] as f64 / q + d[ch] as f64;
                let c = velocity(ch);
                acc[i][0] += f;
                acc[i][1] += c.0 * f;
                acc[i][2] += c.1 * f;
            }
        }
    }
    let n = seeds as f64;
    let check = |name: &str, acc: &[f64], exact: &[f64], var: &[f64]| {
        let (mut zmax, mut chi2, mut cnt) = (0.0f64, 0.0, 0);
        let mut pool = (0.0f64, 0.0f64);
        for k in 0..acc.len() {
            let mean = acc[k] / n;
            if var[k] < 1e-12 {
                assert!((mean - exact[k]).abs() < 1e-9, "{name}: deterministic value off");
                continue;
            }
            if var[k] * n < 5.0 {
                // Rare kicks are Poisson, not Gaussian: pool them.
                pool.0 += (mean - exact[k]) * n;
                pool.1 += var[k] * n;
                continue;
            }
            let z = (mean - exact[k]) / (var[k] / n).sqrt();
            zmax = zmax.max(z.abs());
            chi2 += z * z;
            cnt += 1;
        }
        let chi = chi2 / cnt as f64;
        let zp = pool.0 / pool.1.max(1e-12).sqrt();
        assert!(cnt > 1000 && (chi - 1.0).abs() < 0.1 && zmax < 6.0 && zp.abs() < 5.0, "{name}: {cnt} values, chi2/n {chi}, max |z| {zmax}, pooled rare z {zp}");
    };
    let flat = |v: &[[f64; 3]]| v.iter().flatten().copied().collect::<Vec<f64>>();
    check("streaming", &flat(&acc), &flat(&exact), &flat(&var));

    // Re-quantization: the new reference is the block-mean interpolation,
    // which differs from the sampled one, so every channel gets rounded.
    let mut s0 = Signed::new(w, h, cells0.clone(), Params { seed: 0, ..base.params });
    s0.phi = phi.clone();
    s0.nodes = base.nodes.clone();
    s0.reestimate(false);
    let new_phi = s0.phi.clone();
    let (mut acc, mut ex, mut vr) = (vec![0.0; w * h * NCH], vec![0.0; w * h * NCH], vec![0.0; w * h * NCH]);
    for i in 0..w * h {
        for ch in 0..NCH {
            ex[i * NCH + ch] = fphi[i][ch];
            vr[i * NCH + ch] = pv((phi[i][ch] - new_phi[i][ch]) as f64 / q);
        }
    }
    for seed in 0..seeds {
        let mut s = Signed::new(w, h, cells0.clone(), Params { seed, ..base.params });
        s.phi = phi.clone();
        s.step = 5;
        s.reestimate(false);
        assert_eq!(s.saturated, 0);
        assert_eq!(s.phi, new_phi);
        for i in 0..w * h {
            let d = s.deltas(i);
            for ch in 0..NCH {
                acc[i * NCH + ch] += s.phi[i][ch] as f64 / q + d[ch] as f64;
            }
        }
    }
    check("re-quantization", &acc, &ex, &vr);
}

#[test]
fn half_filled_deviations_relax_to_the_uniform_ternary_measure() {
    // A uniform reference with deviations all at one state of zero mass and
    // momentum relaxes to P(delta = -1, 0, 1) = 1/3 each (3A at density 1).
    let (w, h) = (256, 128);
    let s0 = ternary::encode(&[2, 0, 2, 0, 2, 0, 1]);
    assert_eq!(ternary::mass(s0), 7);
    assert_eq!(ternary::momentum(s0), (0, 0));
    let mut s = Signed::new(w, h, vec![s0; w * h], params(Rule::Optimal, 8, 0));
    s.run(300);
    let mut hist = [[0u64; 3]; NCH];
    for i in 0..w * h {
        let d = s.deltas(i);
        for ch in 0..NCH {
            hist[ch][(d[ch] + 1) as usize] += 1;
        }
    }
    for ch in 0..NCH {
        for k in 0..3 {
            let p = hist[ch][k] as f64 / (w * h) as f64;
            assert!((p - 1.0 / 3.0).abs() < 0.01, "channel {ch}: {p}");
        }
    }
    let _ = signed::REFRESH;
}


#[test]
fn paired_division_is_div16() {
    let mut x = 0x1234_5678u32;
    let mut next = || {
        x = fhp::lowbias32(x);
        x
    };
    let edges = [0, 1, -1, 2, -2, 255, 256, -256, (1 << 15) - 1, 1 << 15, (1 << 22) + 3, (1 << 23) - 1, 1 << 23, -(1 << 23), i32::MAX, i32::MIN + 1];
    let mut cases: Vec<(i32, i32, i32)> = Vec::new();
    for &a in &edges {
        for &d in &edges {
            if d > 0 {
                cases.push((a, -a, d));
            }
        }
    }
    for _ in 0..2_000_000 {
        let den = (next() >> (next() % 31)) as i32 & i32::MAX;
        let shift = next() % 31;
        let (a, b) = ((next() as i32) >> shift, (next() as i32) >> shift);
        cases.push((a, b, den.max(1)));
    }
    for (a, b, d) in cases {
        assert_eq!(signed::div16_pair(a, b, d), (signed::div16(a, d), signed::div16(b, d)), "{a} {b} / {d}");
    }
}
