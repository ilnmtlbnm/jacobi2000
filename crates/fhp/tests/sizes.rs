//! Requested sizes from the page: rounding, the too-big error, and the path
//! the wasm CPU constructor takes (street_for, from_setup, steps, blocks).

use fhp::bitsliced::Bitsliced;
use fhp::{round_size, street_for, SizeLimits, BLOCK, WASM_CPU_MAX_BYTES};

#[test]
fn rounding_meets_every_constraint() {
    for (w, h) in [(30000, 15000), (1000, 999), (65, 3), (1, 1), (0, 0), (4096, 1024), (64, 16)] {
        let (rw, rh) = round_size(w, h, BLOCK);
        assert!(rw >= w && rh >= h && rw - w.max(1) < 64 && rh - h.max(1) < 16, "{w}x{h} -> {rw}x{rh}");
        assert!(rw % 64 == 0 && rw % BLOCK == 0 && rw >= 64, "{w}x{h} -> {rw}x{rh}");
        assert!(rh % 2 == 0 && rh % BLOCK == 0 && rh >= 16, "{w}x{h} -> {rw}x{rh}");
        fhp::check_shape(rw, rh).unwrap();
    }
    assert_eq!(round_size(30000, 15000, BLOCK), (30016, 15008));
    assert_eq!(round_size(4096, 1024, BLOCK), (4096, 1024));
}

#[test]
fn awkward_sizes_run_on_the_cpu_path() {
    let lim = SizeLimits::cpu(WASM_CPU_MAX_BYTES);
    for (w, h) in [(1000, 999), (65, 3), (300, 47)] {
        let s = street_for(w, h, 0.2, "randomize", BLOCK, &lim).unwrap();
        assert_eq!((s.w, s.h), round_size(w, h, BLOCK));
        let mut b = Bitsliced::from_setup(&s);
        b.run(20);
        let sums = b.block_sums(BLOCK, BLOCK);
        assert_eq!(sums.mass.len(), (s.w / BLOCK) * (s.h / BLOCK));
    }
    // 30000x15000 rounds to 30016x15008 and fits the 2 GiB CPU bound
    // (0.84 GB); only checked, not allocated, here.
    let s = street_for(30000, 15000, 0.2, "randomize", BLOCK, &lim).unwrap();
    assert_eq!((s.w, s.h), (30016, 15008));
}

#[test]
fn too_big_is_an_error_with_the_largest_size_that_fits() {
    let lim = SizeLimits::cpu(WASM_CPU_MAX_BYTES);
    let e = street_for(60000, 30000, 0.2, "randomize", BLOCK, &lim).unwrap_err();
    assert!(e.contains("too large") && e.contains("Largest at this aspect ratio"), "{e}");
    let largest = e.rsplit(": ").next().unwrap();
    let (w, h) = largest.split_once('x').unwrap();
    let (w, h): (usize, usize) = (w.parse().unwrap(), h.parse().unwrap());
    assert!(w > 0 && h > 0 && w % 64 == 0 && h % 16 == 0);
    assert!(street_for(w, h, 0.2, "randomize", BLOCK, &lim).is_ok(), "{w}x{h} should fit");
    // GPU-style limits: the state buffer and the rows.
    let gpu = SizeLimits { max_state_buffer: 1 << 30, max_total: u64::MAX, max_rows: 65535 };
    assert!(gpu.check(30016, 15008).is_ok());
    assert!(gpu.check(4096, 70000).unwrap_err().contains("rows 70000"));
    assert!(street_for(512, 64, 0.0, "randomize", BLOCK, &lim).is_err());
    assert!(street_for(512, 64, f64::NAN, "randomize", BLOCK, &lim).is_err());
    assert!(street_for(512, 64, 0.2, "bogus", BLOCK, &lim).is_err());
    for mode in ["periodic", "sponge", "randomize", "delete"] {
        assert!(street_for(512, 64, 0.2, mode, BLOCK, &lim).is_ok());
    }
}
