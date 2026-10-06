//! Stage a: the per-site reference engine.

#![allow(clippy::needless_range_loop)]

use fhp::reference::Reference;
use fhp::rules::{self, bounce, mass, momentum, rotate};
use fhp::{neighbor, source, Obstacle, Params, Setup, OBST};

#[test]
fn collisions_conserve_mass_and_momentum_exhaustively() {
    let t = rules::table();
    for c in 0..2 {
        for s in 0..256usize {
            let o = t[c][s];
            assert_eq!(mass(o), mass(s as u8), "mass, state {s:08b} chirality {c}");
            assert_eq!(o & OBST, s as u8 & OBST);
            if s < 128 {
                assert_eq!(momentum(o), momentum(s as u8), "momentum, state {s:08b} chirality {c}");
            }
        }
    }
}

#[test]
fn collisions_are_permutations_and_fhp3() {
    let t = rules::table();
    for c in 0..2 {
        let mut seen = [false; 256];
        for s in 0..256 {
            assert!(!seen[t[c][s] as usize], "chirality {c} is not a permutation");
            seen[t[c][s] as usize] = true;
        }
    }
    // Chirality 1 undoes chirality 0 (the 3-cycles run backwards).
    for s in 0..128 {
        assert_eq!(t[1][t[0][s] as usize], s as u8);
    }
    // d'Humieres and Lallemand 1987: "76 configurations give active collisions".
    assert_eq!(rules::active_configurations(&t), 76);
    // Saturation: every state with an alternative in its group changes,
    // under both chiralities, and the two chiralities reach different states
    // in groups of three.
    for g in rules::groups() {
        for &s in &g {
            if g.len() > 1 {
                assert_ne!(t[0][s as usize], s);
                assert_ne!(t[1][s as usize], s);
            }
            if g.len() == 3 {
                assert_ne!(t[0][s as usize], t[1][s as usize]);
            }
        }
    }
    // Rotation equivariance, and reflection swaps the chirality.
    let reflect = |s: u8| {
        // Mirror about the x axis: direction i -> -i mod 6.
        let mut o = s & 0xc0;
        for i in 0..6 {
            if s & (1 << i) != 0 {
                o |= 1 << ((6 - i) % 6);
            }
        }
        o
    };
    for s in 0..128u8 {
        for k in 0..6 {
            assert_eq!(t[0][rotate(s, k) as usize], rotate(t[0][s as usize], k));
        }
        assert_eq!(t[1][reflect(s) as usize], reflect(t[0][s as usize]), "state {s:07b}");
    }
}

#[test]
fn obstacle_bounces_back() {
    let t = rules::table();
    for s in 128..256usize {
        assert_eq!(t[0][s], bounce(s as u8));
        assert_eq!(t[1][s], bounce(s as u8));
    }
    // A single particle sent east into a wall comes back west to where it
    // started, two steps after it left: site x0 -> wall -> x0.
    let (w, h) = (64, 4);
    let mut cells = vec![0u8; w * h];
    cells[w + 10] = OBST;
    cells[w + 9] = 1 << 0;
    let mut r = Reference::new(w, h, cells, Params { seed: 3, force_threshold: 0, ..Default::default() });
    r.step();
    assert_eq!(r.cells[w + 10], OBST | (1 << 3), "reversed at the wall");
    r.step();
    assert_eq!(r.cells[w + 9], 1 << 3, "back where it came from, moving west");
    // Diagonals: NE into a wall comes back SW, for both row parities.
    for (y, dir) in [(0usize, 1usize), (1, 1), (0, 2), (1, 5), (2, 4)] {
        let (w, h) = (64, 6);
        let mut cells = vec![0u8; w * h];
        let (wx, wy) = neighbor(20, y, dir, w, h);
        cells[wy * w + wx] = OBST;
        cells[y * w + 20] = 1 << dir;
        let mut r = Reference::new(w, h, cells, Params { seed: 3, force_threshold: 0, ..Default::default() });
        r.run(2);
        assert_eq!(r.cells[y * w + 20], 1 << ((dir + 3) % 6), "row {y} dir {dir}");
    }
}

#[test]
fn streaming_is_a_permutation() {
    for (w, h) in [(64, 2), (64, 8), (128, 30), (192, 50)] {
        let mut hit = vec![false; w * h * 6];
        for y in 0..h {
            for x in 0..w {
                for d in 0..6 {
                    let (nx, ny) = neighbor(x, y, d, w, h);
                    let k = (ny * w + nx) * 6 + d;
                    assert!(!hit[k], "two particles land on ({nx},{ny}) dir {d}");
                    hit[k] = true;
                    assert_eq!(source(nx, ny, d, w, h), (x, y));
                }
            }
        }
        // Random states: streaming alone (no collision) preserves the count
        // per direction and is undone by streaming the reversed lattice.
        let s = Setup { w, h, density: 0.3, ux: 0.2, obstacle: Obstacle::None, params: Params { seed: 7, force_threshold: 0, ..Default::default() } };
        let cells = s.initial_state();
        let r = Reference::new(w, h, cells.clone(), s.params);
        let streamed = r.stream_only();
        for d in 0..7 {
            let n0 = cells.iter().filter(|&&c| c & (1 << d) != 0).count();
            let n1 = streamed.iter().filter(|&&c| c & (1 << d) != 0).count();
            assert_eq!(n0, n1);
        }
        let rev: Vec<u8> = streamed.iter().map(|&c| bounce(c)).collect();
        let back = Reference::new(w, h, rev, s.params).stream_only();
        let back: Vec<u8> = back.iter().map(|&c| bounce(c)).collect();
        assert_eq!(back, cells);
    }
}

#[test]
fn steps_conserve_mass_and_forcing_adds_x_momentum() {
    let mut s = Setup::plate(256, 64);
    s.params = Params::with_force_probability(11, 0.01);
    let mut r = Reference::new(s.w, s.h, s.initial_state(), s.params);
    let m0 = r.mass();
    r.run(50);
    assert_eq!(r.mass(), m0);

    // Forcing alone: each flip turns W into E, +4 in x, nothing in y.
    let s = Setup { w: 128, h: 32, density: 0.3, ux: 0.0, obstacle: Obstacle::None, params: Params::with_force_probability(9, 0.01) };
    let mut r = Reference::new(s.w, s.h, s.initial_state(), s.params);
    let (m0, p0) = (r.mass(), r.momentum());
    r.run(20);
    let p1 = r.momentum();
    assert_eq!(r.mass(), m0);
    assert_eq!(p1.1, p0.1);
    assert!(p1.0 > p0.0 && (p1.0 - p0.0) % 4 == 0, "{p0:?} -> {p1:?}");

    // Without forcing or obstacles momentum is conserved exactly.
    let s = Setup { w: 128, h: 32, density: 0.3, ux: 0.2, obstacle: Obstacle::None, params: Params { seed: 5, force_threshold: 0, ..Default::default() } };
    let mut r = Reference::new(s.w, s.h, s.initial_state(), s.params);
    let (m0, p0) = (r.mass(), r.momentum());
    r.run(40);
    assert_eq!((r.mass(), r.momentum()), (m0, p0));
}
