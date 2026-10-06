//! Stage c: the GPU engine against the CPU engines. Needs a GPU adapter.

use fhp::bitsliced::Bitsliced;
use fhp::gpu::Gpu;
use fhp::reference::Reference;
use fhp::boundary::Outlet;
use fhp::{Obstacle, Params, Setup};

fn check(s: Setup, steps: usize, with_reference: bool) {
    let cells = s.initial_state();
    let mut gpu = pollster::block_on(Gpu::new(s.w, s.h, &cells, s.params)).expect("GPU adapter");
    assert_eq!(pollster::block_on(gpu.read_cells()), cells);
    let mut b = Bitsliced::from_cells(s.w, s.h, &cells, s.params);
    let mut r = with_reference.then(|| Reference::new(s.w, s.h, cells.clone(), s.params));
    // Uneven chunks cross the batch boundary of the uniform slots.
    let mut done = 0;
    for chunk in [1, 7, 128, 129, 300] {
        let n = chunk.min(steps - done);
        gpu.run(n);
        b.run(n);
        if let Some(r) = r.as_mut() {
            r.run(n);
        }
        done += n;
        let g = pollster::block_on(gpu.read_cells());
        assert_eq!(g, b.to_cells(), "{}x{}: GPU and bit-sliced CPU differ after {done} steps", s.w, s.h);
        if let Some(r) = r.as_ref() {
            assert_eq!(g, r.cells, "{}x{}: GPU and reference differ after {done} steps", s.w, s.h);
        }
        if done == steps {
            break;
        }
    }
    assert_eq!(done, steps);
    // The coarse-graining kernel agrees with the CPU one.
    for (bw, bh) in [(32, 2), (16, 16), (8, 2)] {
        if s.h.is_multiple_of(bh) {
            assert_eq!(pollster::block_on(gpu.block_sums(bw, bh)), b.block_sums(bw, bh));
        }
    }
}

#[test]
fn gpu_accumulated_sums_match_cpu() {
    let s = Setup::street_periodic(512, 128);
    let cells = s.initial_state();
    let mut gpu = pollster::block_on(Gpu::new(s.w, s.h, &cells, s.params)).expect("GPU adapter");
    let mut b = Bitsliced::from_cells(s.w, s.h, &cells, s.params);
    let mut want = b.block_sums(16, 8);
    gpu.accumulate_sums(16, 8);
    for n in [3, 5, 8] {
        gpu.run(n);
        gpu.accumulate_sums(16, 8);
        b.run(n);
        want.add(&b.block_sums(16, 8));
    }
    assert_eq!(pollster::block_on(gpu.take_sums()), want);
    gpu.accumulate_sums(16, 8);
    assert_eq!(pollster::block_on(gpu.take_sums()), b.block_sums(16, 8), "take_sums clears");
}

#[test]
fn gpu_bit_identical_to_cpu() {
    check(Setup { w: 64, h: 2, density: 0.3, ux: 0.1, obstacle: Obstacle::None, params: Params { seed: 1, force_threshold: 0, ..Default::default() } }, 100, true);
    check(Setup { w: 128, h: 30, density: 0.4, ux: 0.2, obstacle: Obstacle::Cylinder { x: 40.0, y: 15.0, d: 9.0 }, params: Params::with_force_probability(2, 0.02) }, 300, true);
    let mut s = Setup::plate(512, 128);
    s.params = Params::with_force_probability(42, 0.001);
    check(s, 565, true);
    let mut s = Setup::plate(2048, 512);
    s.params = Params::with_force_probability(7, 1e-4);
    check(s, 400, false);
}

#[test]
fn gpu_bit_identical_with_open_boundaries_and_walls() {
    for kind in [Outlet::Sponge, Outlet::Randomize, Outlet::Delete] {
        check(Setup::street_with(256, 32, 0.2, kind), 300, true);
        check(Setup::street_with(2048, 512, 0.2, kind), 300, false);
    }
    let mut s = Setup::street(1024, 128, 0.2);
    s.params.walls = true;
    s.params.force_threshold = Params::with_force_probability(0, 1e-3).force_threshold;
    check(s, 565, true);
    let mut s = Setup::street(192, 50, 0.1);
    s.params.walls = true;
    check(s, 300, true);
    check(Setup::street(4096, 1024, 0.2), 565, false);
}
