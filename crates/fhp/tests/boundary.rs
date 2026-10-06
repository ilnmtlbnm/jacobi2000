//! Open boundaries and walls: the equilibrium sampler, mass balance, and
//! bit-identity of the reference and bit-sliced engines.

#![allow(clippy::needless_range_loop)]

use fhp::bitsliced::Bitsliced;
use fhp::boundary::{self, bern, equilibrium, Boundary, Outlet, QONE};
use fhp::reference::Reference;
use fhp::{Obstacle, Params, Setup, CX2};

#[test]
fn equilibrium_has_the_target_mass_and_momentum() {
    for (d, u) in [(0.28, 0.0), (0.28, 0.2), (0.2, 0.35), (0.4, 0.1)] {
        let n = equilibrium(d, u);
        let m: f64 = n.iter().sum();
        let j: f64 = (0..6).map(|i| CX2[i] as f64 / 2.0 * n[i]).sum();
        assert!((m - 7.0 * d).abs() < 1e-12, "{d} {u}: mass {m}");
        assert!((j - 7.0 * d * u).abs() < 1e-12, "{d} {u}: momentum {j}");
        // Upstream-moving directions are less occupied than downstream ones.
        if u > 0.0 {
            assert!(n[0] > n[1] && n[1] > n[2] && n[2] > n[3]);
        }
    }
}

#[test]
fn sampler_gives_the_equilibrium_occupations() {
    let (d, u) = (0.28, 0.2);
    let b = Boundary::new(d, u, 1, 2, Outlet::Sponge);
    let n = equilibrium(d, u);
    let seeds = boundary::stream_seeds(12345);
    let samples = 400 * 1000 * 32;
    let mut count = [0u64; 7];
    let mut jx = 0i64;
    for step in 0..400u32 {
        let keys = boundary::step_keys(&seeds, step);
        for idx in 0..1000u32 {
            let planes: [u32; 7] = std::array::from_fn(|p| bern(&keys, p, idx, b.eq[p]));
            for p in 0..7 {
                count[p] += planes[p].count_ones() as u64;
            }
            for p in 0..6 {
                jx += CX2[p] as i64 * planes[p].count_ones() as i64;
            }
        }
    }
    for p in 0..7 {
        let mean = count[p] as f64 / samples as f64;
        let sigma = (n[p] * (1.0 - n[p]) / samples as f64).sqrt();
        // 5 sigma plus the 12-bit quantization of the threshold.
        assert!((mean - n[p]).abs() < 5.0 * sigma + 0.5 / QONE as f64, "plane {p}: {mean} vs {}", n[p]);
    }
    let j = jx as f64 / 2.0 / samples as f64;
    assert!((j - 7.0 * d * u).abs() < 0.005, "momentum per site {j} vs {}", 7.0 * d * u);
    // The replacement mask follows its threshold too.
    let mut m = 0u64;
    for idx in 0..100000u32 {
        m += bern(&boundary::step_keys(&seeds, 3), 7, idx, 1000).count_ones() as u64;
    }
    let mean = m as f64 / 3.2e6;
    assert!((mean - 1000.0 / 4096.0).abs() < 0.002, "mask {mean}");
}

#[test]
fn sponge_ramps_to_one() {
    let b = Boundary::new(0.28, 0.2, 1, 8, Outlet::Sponge);
    let q: Vec<u32> = (0..128).map(|c| b.q(c, 128)).collect();
    assert_eq!(q[0], QONE);
    assert!(q[1..120].iter().all(|&x| x == 0));
    assert!(q[120..].windows(2).all(|w| w[0] < w[1]));
    assert_eq!(q[127], QONE);
}

#[test]
fn mass_in_balances_mass_out() {
    for kind in [Outlet::Sponge, Outlet::Randomize] {
        mass_balance(Setup::street_with(512, 64, 0.2, kind));
    }
}

fn mass_balance(s: Setup) {
    let mut r = Reference::new(s.w, s.h, s.initial_state(), s.params);
    let m0 = r.mass() as i64;
    r.run(500);
    let (i0, o0) = (r.flux_in, r.flux_out);
    let n = 8000;
    r.run(n);
    let (inn, out) = (r.flux_in - i0, r.flux_out - o0);
    let m1 = r.mass() as i64;
    eprintln!("flux in {inn}, out {out}, imposed {:.0}, mass {m0} -> {m1}", 7.0 * s.density * s.ux * s.h as f64 * n as f64);
    // The inflow is the imposed one, 7 d U per row per step...
    let want = 7.0 * s.density * s.ux * s.h as f64 * n as f64;
    assert!((inn as f64 - want).abs() < 0.1 * want, "in {inn}, imposed {want}");
    // ...and it leaves through the sponge: interior mass is bounded.
    assert!(((inn - out) as f64).abs() < 0.02 * inn as f64, "in {inn} out {out}");
    assert!(((m1 - m0) as f64).abs() < 0.01 * m0 as f64, "mass {m0} -> {m1}");
}

#[test]
fn walls_conserve_mass_and_x_momentum() {
    let s = Setup { w: 128, h: 32, density: 0.3, ux: 0.2, obstacle: Obstacle::None, params: Params { seed: 4, walls: true, ..Default::default() } };
    let mut r = Reference::new(s.w, s.h, s.initial_state(), s.params);
    let (m0, p0) = (r.mass(), r.momentum());
    r.run(100);
    assert_eq!(r.mass(), m0);
    assert_eq!(r.momentum().0, p0.0);
    // No particle crosses the wall: with particles only in the upper half
    // and walls on, the lower rows next to the periodic seam stay empty.
    let (w, h) = (64, 16);
    let mut cells = vec![0u8; w * h];
    for x in 0..w {
        cells[(h - 2) * w + x] = 0b0000110; // NE + NW moving up into row H-1
    }
    let mut r = Reference::new(w, h, cells, Params { seed: 4, walls: true, ..Default::default() });
    r.run(1);
    r.run(1);
    for x in 0..w {
        assert_eq!(r.cells[x], 0, "row 0 reached through the wall");
    }
}

fn check(s: Setup, steps: usize) {
    let cells = s.initial_state();
    let mut r = Reference::new(s.w, s.h, cells.clone(), s.params);
    let mut b = Bitsliced::from_cells(s.w, s.h, &cells, s.params);
    for k in 0..steps {
        r.step();
        b.step();
        if k % 25 == 24 || k + 1 == steps {
            assert_eq!(b.to_cells(), r.cells, "{}x{} diverged at step {}", s.w, s.h, k + 1);
        }
    }
}

#[test]
fn bit_identical_to_reference_with_open_boundaries_and_walls() {
    for kind in [Outlet::Sponge, Outlet::Randomize, Outlet::Delete] {
        check(Setup::street_with(256, 32, 0.2, kind), 150);
        check(Setup::street_with(512, 64, 0.3, kind), 150);
    }
    let mut s = Setup::street(1024, 128, 0.2);
    s.params.walls = true;
    s.params.force_threshold = Params::with_force_probability(0, 1e-3).force_threshold;
    check(s, 120);
    let mut s = Setup::street(192, 50, 0.1);
    s.params.walls = true;
    check(s, 120);
}
