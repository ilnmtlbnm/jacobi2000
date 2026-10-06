//! Ternary FHP: collision table, streaming, bounce-back, equilibrium and
//! theory checks.

#![allow(clippy::needless_range_loop)]

use fhp::ternary::engine::{self, Ternary, OBST};
use fhp::ternary::*;
use fhp::{lowbias32, neighbor, NDIR};

#[test]
fn encoding_round_trips() {
    for s in 0..NSTATE as u16 {
        assert_eq!(encode(&counts(s)), s);
        assert_eq!(bounce(bounce(s)), s);
        assert_eq!(mirror(mirror(s)), s);
        assert_eq!(rotate(s, 6), s);
    }
}

#[test]
fn collisions_conserve_mass_and_momentum_exhaustively() {
    let t = Table::new();
    for rule in [Rule::Other, Rule::Uniform] {
        for s in 0..NSTATE as u16 {
            let q = t.class(s).len();
            // Every r lands on a class member; sweep r so every output is hit.
            let mut hit = vec![false; q];
            for k in 0..(64 * q as u64) {
                let r = ((k << 32) / (64 * q as u64)) as u32;
                let o = t.collide(s, r, rule);
                assert_eq!(mass(o), mass(s));
                assert_eq!(momentum(o), momentum(s));
                hit[t.pos[o as usize] as usize] = true;
                if rule == Rule::Other && q > 1 {
                    assert_ne!(o, s);
                }
            }
            let reach = hit.iter().filter(|&&h| h).count();
            assert_eq!(
                reach,
                if rule == Rule::Other && q > 1 {
                    q - 1
                } else {
                    q
                },
                "state {s}"
            );
        }
    }
    for s in 0..NSTATE as u16 {
        for k in 0..12u64 {
            let o = t.collide(s, ((k << 32) / 12 + 1) as u32, Rule::Optimal);
            assert_eq!(o, t.opt[s as usize][k as usize]);
            assert_eq!(mass(o), mass(s));
            assert_eq!(momentum(o), momentum(s));
        }
        for c in 0..2u32 {
            let o = t.collide(s, c << 31, Rule::Chiral);
            assert_eq!(o, t.chiral[s as usize][c as usize]);
            assert_eq!(mass(o), mass(s));
            assert_eq!(momentum(o), momentum(s));
        }
    }
}

#[test]
fn optimal_rules_are_doubly_stochastic_and_symmetric_under_the_lattice() {
    let t = Table::new();
    // Chiral: both choices are permutations, the first commutes with rotation.
    for c in 0..2 {
        let mut seen = vec![false; NSTATE];
        for s in 0..NSTATE {
            let o = t.chiral[s][c] as usize;
            assert!(!seen[o]);
            seen[o] = true;
        }
    }
    for s in 0..NSTATE as u16 {
        assert_eq!(
            t.chiral[rotate(s, 1) as usize][0],
            rotate(t.chiral[s as usize][0], 1)
        );
    }
    // Each of the 12 images is a permutation of the states.
    for k in 0..12 {
        let mut seen = vec![false; NSTATE];
        for s in 0..NSTATE {
            let o = t.opt[s][k] as usize;
            assert!(!seen[o]);
            seen[o] = true;
        }
    }
    // Transitions commute with rotations and the mirror.
    for rule in [Rule::Optimal, Rule::Chiral] {
        for s in 0..NSTATE as u16 {
            let mut a = t.transitions(s, rule);
            let mut b: Vec<(u16, f64)> = t.transitions(rotate(s, 1), rule);
            a = a.into_iter().map(|(o, p)| (rotate(o, 1), p)).collect();
            a.sort_by_key(|x| x.0);
            b.sort_by_key(|x| x.0);
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.0, y.0);
                assert!((x.1 - y.1).abs() < 1e-12);
            }
            let mut a: Vec<(u16, f64)> = t
                .transitions(s, rule)
                .into_iter()
                .map(|(o, p)| (mirror(o), p))
                .collect();
            let mut b = t.transitions(mirror(s), rule);
            a.sort_by_key(|x| x.0);
            b.sort_by_key(|x| x.0);
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.0, y.0);
                assert!((x.1 - y.1).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn transitions_satisfy_semi_detailed_balance() {
    let t = Table::new();
    for rule in [Rule::Other, Rule::Uniform, Rule::Optimal, Rule::Chiral] {
        let mut a = vec![0.0f64; NSTATE * NSTATE];
        for s in 0..NSTATE {
            let tr = t.transitions(s as u16, rule);
            let total: f64 = tr.iter().map(|x| x.1).sum();
            assert!((total - 1.0).abs() < 1e-12);
            for (o, p) in tr {
                a[s * NSTATE + o as usize] += p;
            }
        }
        for s in 0..NSTATE {
            // Semi-detailed balance: every column sums to one.
            let col: f64 = (0..NSTATE).map(|r| a[r * NSTATE + s]).sum();
            assert!((col - 1.0).abs() < 1e-12);
            // Detailed balance for the two uniform rules.
            if matches!(rule, Rule::Other | Rule::Uniform) {
                for o in 0..NSTATE {
                    assert_eq!(a[s * NSTATE + o], a[o * NSTATE + s]);
                }
            }
        }
    }
}

#[test]
fn classes_are_rotation_and_mirror_invariant() {
    let t = Table::new();
    let sizes = t.class_sizes();
    assert_eq!(sizes.iter().sum::<usize>(), NSTATE);
    for s in 0..NSTATE as u16 {
        for k in 0..6 {
            let g: Vec<u16> = t.class(s).iter().map(|&m| rotate(m, k)).collect();
            let mut h = t.class(rotate(s, k)).to_vec();
            let mut g2 = g.clone();
            g2.sort();
            h.sort();
            assert_eq!(g2, h);
        }
        let mut g: Vec<u16> = t.class(s).iter().map(|&m| mirror(m)).collect();
        let mut h = t.class(mirror(s)).to_vec();
        g.sort();
        h.sort();
        assert_eq!(g, h);
    }
}

fn random_cells(w: usize, h: usize, seed: u32) -> Vec<u16> {
    (0..w * h)
        .map(|i| (lowbias32(seed ^ lowbias32(i as u32)) % NSTATE as u32) as u16)
        .collect()
}

#[test]
fn streaming_is_a_permutation_of_the_counts() {
    let (w, h) = (64, 16);
    let cells = random_cells(w, h, 7);
    let t = Ternary::new(
        w,
        h,
        cells.clone(),
        engine::Params {
            seed: 1,
            rule: Rule::Other,
            boundary: None,
        },
    );
    let out = t.stream_only();
    for y in 0..h {
        for x in 0..w {
            let n = counts(cells[y * w + x]);
            for d in 0..NDIR {
                let (nx, ny) = neighbor(x, y, d, w, h);
                assert_eq!(counts(out[ny * w + nx])[d], n[d]);
            }
            assert_eq!(counts(out[y * w + x])[6], n[6]);
        }
    }
}

#[test]
fn steps_conserve_mass_and_momentum_and_obstacles_bounce_back() {
    let (w, h) = (64, 16);
    let mut cells = random_cells(w, h, 9);
    // A block of obstacle sites: they hold bounced counts.
    for y in 6..10 {
        for x in 20..24 {
            cells[y * w + x] = OBST;
        }
    }
    let mut t = Ternary::new(
        w,
        h,
        cells,
        engine::Params {
            seed: 3,
            rule: Rule::Other,
            boundary: None,
        },
    );
    let m0 = t.mass();
    for _ in 0..50 {
        let pre = t.stream_only();
        let p0 = t.momentum();
        t.step();
        assert_eq!(t.mass(), m0);
        // Momentum changes only by what the obstacles reverse.
        let mut dp = (0i64, 0i64);
        for i in 0..w * h {
            if pre[i] & OBST != 0 {
                let s = pre[i] & !OBST;
                assert_eq!(t.cells[i], OBST | bounce(s));
                let p = momentum(s);
                dp.0 -= 2 * p.0 as i64;
                dp.1 -= 2 * p.1 as i64;
            }
        }
        let p1 = t.momentum();
        assert_eq!((p1.0 - p0.0, p1.1 - p0.1), dp);
    }
    // Without obstacles momentum is exact.
    let mut t = Ternary::new(
        w,
        h,
        random_cells(w, h, 11),
        engine::Params {
            seed: 3,
            rule: Rule::Uniform,
            boundary: None,
        },
    );
    let p0 = t.momentum();
    t.run(20);
    assert_eq!(t.momentum(), p0);
}

#[test]
fn equilibrium_has_the_target_mass_and_momentum() {
    for m in [1, 2] {
        for (d, ux, uy) in [
            (0.28, 0.0, 0.0),
            (0.28, 0.2, 0.0),
            (0.6, 0.1, 0.05),
            (0.15, 0.0, 0.2),
        ] {
            if m == 1 && d > 0.5 {
                continue;
            }
            let n = equilibrium(m, m, d, ux, uy);
            let mass: f64 = n.iter().sum();
            let jx: f64 = (0..7).map(|i| velocity(i).0 * n[i]).sum();
            let jy: f64 = (0..7).map(|i| velocity(i).1 * n[i]).sum();
            assert!((mass - 7.0 * d).abs() < 1e-10);
            assert!((jx - 7.0 * d * ux).abs() < 1e-10, "{m} {d} {ux}: {jx}");
            assert!((jy - 7.0 * d * uy).abs() < 1e-10);
        }
    }
    // Capacity 1 along x is FHP's Fermi-Dirac equilibrium.
    let a = equilibrium(1, 1, 0.28, 0.2, 0.0);
    let b = fhp::boundary::equilibrium(0.28, 0.2);
    for i in 0..7 {
        assert!((a[i] - b[i]).abs() < 1e-10);
    }
}

#[test]
fn sampler_matches_the_equilibrium() {
    let (d, u) = (0.5, 0.2);
    let mean = equilibrium(2, 2, d, u, 0.0);
    let n = 400_000u32;
    let mut hist = [[0u64; 3]; 7];
    for site in 0..n {
        let c = counts(engine::sample_equilibrium(&mean, 5, site));
        for i in 0..7 {
            hist[i][c[i] as usize] += 1;
        }
    }
    for i in 0..7 {
        let p = ternary_probs(mean[i]);
        let probs = [p[0], p[1], 1.0 - p[0] - p[1]];
        for k in 0..3 {
            let f = hist[i][k] as f64 / n as f64;
            let sigma = (probs[k] * (1.0 - probs[k]) / n as f64).sqrt();
            assert!(
                (f - probs[k]).abs() < 5.0 * sigma,
                "channel {i} count {k}: {f} vs {}",
                probs[k]
            );
        }
        let m = (hist[i][1] + 2 * hist[i][2]) as f64 / n as f64;
        assert!((m - mean[i]).abs() < 0.01);
    }
}

/// Started far from equilibrium (every site holds the same state), the
/// collisions drive the channel counts to the product measure z^k at the
/// lattice's density: the equilibrium the rule was chosen for.
#[test]
fn collisions_relax_to_the_derived_equilibrium() {
    for rule in [Rule::Other, Rule::Uniform, Rule::Optimal, Rule::Chiral] {
        let (w, h) = (256, 128);
        // n = (2, 0, 1, 0, 0, 2, 1): mass 6, nonzero momentum.
        let s = encode(&[2, 0, 1, 0, 0, 2, 1]);
        let p = momentum(s);
        let mut t = Ternary::new(
            w,
            h,
            vec![s; w * h],
            engine::Params {
                seed: 17,
                rule,
                boundary: None,
            },
        );
        t.run(300);
        let mut hist = [[0u64; 3]; 7];
        for _ in 0..10 {
            t.run(10);
            for &c in &t.cells {
                let n = counts(c);
                for i in 0..7 {
                    hist[i][n[i] as usize] += 1;
                }
            }
        }
        let d = 6.0 / 7.0;
        let ux = p.0 as f64 * 0.5 / 6.0;
        let uy = p.1 as f64 * 3f64.sqrt() / 2.0 / 6.0;
        let mean = equilibrium(2, 2, d, ux, uy);
        let total = (w * h * 10) as f64;
        for i in 0..7 {
            let pr = ternary_probs(mean[i]);
            let probs = [pr[0], pr[1], 1.0 - pr[0] - pr[1]];
            for k in 0..3 {
                let f = hist[i][k] as f64 / total;
                assert!(
                    (f - probs[k]).abs() < 0.01,
                    "{rule:?} channel {i} count {k}: {f:.4} vs {:.4}",
                    probs[k]
                );
            }
        }
    }
}

#[test]
fn theory_reduces_to_fhp3() {
    for d in [0.1, 0.2, 0.28, 0.35, 0.45] {
        let g = g_theory(1, 1, d);
        assert!(
            (g - 7.0 / 12.0 * (1.0 - 2.0 * d) / (1.0 - d)).abs() < 1e-9,
            "g at {d}"
        );
        let q = d * (1.0 - d);
        let nu = 1.0 / (28.0 * q * (1.0 - 8.0 * q / 7.0)) - 0.125;
        assert!(
            (nu_fhp3(d) - nu).abs() < 1e-9,
            "nu at {d}: {} vs {nu}",
            nu_fhp3(d)
        );
    }
}

#[test]
fn generated_circuit_matches_the_table() {
    use fhp::ternary::collide_gen::{collide, ENCODING};
    use fhp::ternary::synth::{OBST_IN, CHIR_IN};
    let t = Table::new();
    let n = 1usize << 16;
    for i0 in (0..n).step_by(64) {
        let x: [u64; 16] = std::array::from_fn(|k| (0..64).fold(0u64, |a, l| a | ((((i0 + l) >> k) & 1) as u64) << l));
        let o = collide(x);
        for l in 0..64 {
            let v = i0 + l;
            let Some(s) = ENCODING.state((v & 0x3fff) as u16) else { continue };
            let want = if (v >> OBST_IN) & 1 == 1 { bounce(s) } else { t.chiral[s as usize][(v >> CHIR_IN) & 1] };
            let got = (0..14).fold(0u16, |a, k| a | (((o[k] >> l) & 1) as u16) << k);
            assert_eq!(got, ENCODING.planes(want), "input {v:#06x}");
        }
    }
}

#[test]
fn bitsliced_matches_the_reference_bit_for_bit() {
    use fhp::ternary::bitsliced::Bitsliced;
    for (w, h, open) in [(128, 16, false), (192, 32, true)] {
        let mut cells = random_cells(w, h, 21);
        for y in 5..11 {
            for x in 40..42 {
                cells[y * w + x] = OBST | (cells[y * w + x] % 50);
            }
        }
        let boundary = if open { Some(engine::Boundary::new(0.6, 0.2)) } else { None };
        let p = engine::Params { seed: 4, rule: Rule::Chiral, boundary };
        let mut r = Ternary::new(w, h, cells.clone(), p);
        let mut b = Bitsliced::from_cells(w, h, &cells, p);
        assert_eq!(b.to_cells(), cells);
        for step in 0..70 {
            r.step();
            b.step();
            assert_eq!(b.to_cells(), r.cells, "{w}x{h} step {step}");
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_matches_the_reference_bit_for_bit() {
    use fhp::ternary::gpu::{Gpu, Layout};
    for layout in [Layout::Bitsliced, Layout::Table] {
    let (w, h) = (128, 32);
    let mut cells = random_cells(w, h, 33);
    for y in 10..20 {
        for x in 60..62 {
            cells[y * w + x] = OBST;
        }
    }
    let mut r = Ternary::new(w, h, cells.clone(), engine::Params { seed: 8, rule: Rule::Chiral, boundary: None });
    let Ok(mut g) = pollster::block_on(Gpu::with_layout(w, h, &cells, 8, layout)) else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    for n in [1usize, 7, 130] {
        r.run(n);
        g.run(n);
        assert_eq!(pollster::block_on(g.read_cells()), r.cells, "{layout:?} after {} steps", r.step);
    }
    }
}
