//! Stage b: the bit-sliced CPU engine against the reference.

#![allow(clippy::needless_range_loop)]

use fhp::bitsliced::Bitsliced;
use fhp::reference::Reference;
use fhp::{rules, synth, Obstacle, Params, Setup};

#[test]
fn generated_circuit_matches_the_table_exhaustively() {
    let t = rules::table();
    for x in 0..512u64 {
        let ins: [u64; 9] = std::array::from_fn(|k| if (x >> k) & 1 == 1 { u64::MAX } else { 0 });
        let o = fhp::collide_gen::collide(ins);
        let want = t[(x >> 8) as usize][(x & 255) as usize];
        for k in 0..7 {
            assert_eq!(o[k], if (want >> k) & 1 == 1 { u64::MAX } else { 0 }, "input {x:09b} output {k}");
        }
    }
}

#[test]
fn generated_files_are_up_to_date() {
    // Re-synthesize with the recorded order and form; the committed files
    // must be what `fhp gen` writes.
    let src = include_str!("../src/collide_gen.rs");
    let line = src.lines().nth(1).unwrap();
    let order: Vec<usize> = line[line.find('[').unwrap() + 1..line.find(']').unwrap()].split(", ").map(|s| s.parse().unwrap()).collect();
    let flip = line.contains("flip form");
    let c = synth::synthesize(&rules::table(), order.try_into().unwrap(), flip);
    assert_eq!(c.to_rust(), src);
    assert_eq!(c.to_wgsl(), include_str!("../src/shaders/collide_gen.wgsl"));
    assert_eq!(c.gate_count(), fhp::collide_gen::GATES);
}

fn check(s: Setup, steps: usize) {
    let cells = s.initial_state();
    let mut r = Reference::new(s.w, s.h, cells.clone(), s.params);
    let mut b = Bitsliced::from_cells(s.w, s.h, &cells, s.params);
    assert_eq!(b.to_cells(), cells);
    for k in 0..steps {
        r.step();
        b.step();
        if k % 10 == 9 || k + 1 == steps {
            assert_eq!(b.to_cells(), r.cells, "{}x{} diverged at step {}", s.w, s.h, k + 1);
        }
    }
}

#[test]
fn bit_identical_to_reference() {
    // Several sizes, with and without obstacles and forcing.
    check(Setup { w: 64, h: 2, density: 0.3, ux: 0.1, obstacle: Obstacle::None, params: Params { seed: 1, force_threshold: 0, ..Default::default() } }, 60);
    check(Setup { w: 128, h: 30, density: 0.4, ux: 0.2, obstacle: Obstacle::Cylinder { x: 40.0, y: 15.0, d: 9.0 }, params: Params::with_force_probability(2, 0.02) }, 100);
    check(Setup { w: 192, h: 50, ..Setup::plate(192, 50) }, 100);
    let mut s = Setup::plate(512, 128);
    s.params = Params::with_force_probability(42, 0.001);
    check(s, 200);
}
