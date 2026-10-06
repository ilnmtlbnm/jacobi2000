use jacobi::circuit::Circuit;

/// Gate counts from docs/runs/2026-10-05-jacobi-gates.md (exp-jacobi),
/// ripple3, section 2.
#[test]
fn gate_counts_match_the_run_doc() {
    for (b, trunc, round) in [(4, 54, 62), (8, 114, 130), (12, 174, 198), (16, 234, 266)] {
        assert_eq!(Circuit::jacobi(b, false).gate_count(), trunc, "b={b} truncating");
        assert_eq!(Circuit::jacobi(b, true).gate_count(), round, "b={b} rounding");
    }
    // Per kind (AND, OR, XOR, NOT), truncating, from the same table.
    assert_eq!(Circuit::jacobi(4, false).counts(), [23, 10, 21, 0]);
    assert_eq!(Circuit::jacobi(16, false).counts(), [95, 46, 93, 0]);
}

/// Word-level evaluation against integer arithmetic: every input at b=4,
/// pseudo-random inputs at 8, 12, 16. Each lane of a word is its own case.
#[test]
fn circuit_is_integer_jacobi() {
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for b in [4usize, 8, 12, 16] {
        for round in [false, true] {
            let c = Circuit::jacobi(b, round);
            let cases: usize = if b == 4 { 1 << 16 } else { 1 << 18 };
            for chunk in 0..cases / 64 {
                let vals: Vec<[u32; 4]> = (0..64)
                    .map(|l| {
                        if b == 4 {
                            let x = (chunk * 64 + l) as u32;
                            [x & 15, (x >> 4) & 15, (x >> 8) & 15, (x >> 12) & 15]
                        } else {
                            let r = rnd();
                            let m = (1u64 << b) - 1;
                            [(r & m) as u32, ((r >> 16) & m) as u32, ((r >> 32) & m) as u32, ((r >> 48) & m) as u32]
                        }
                    })
                    .collect();
                let mut inputs = vec![0u64; 4 * b];
                for (l, v) in vals.iter().enumerate() {
                    for op in 0..4 {
                        for i in 0..b {
                            inputs[op * b + i] |= (((v[op] >> i) & 1) as u64) << l;
                        }
                    }
                }
                let out = c.eval(&inputs);
                for (l, v) in vals.iter().enumerate() {
                    let want = (v.iter().sum::<u32>() + if round { 2 } else { 0 }) >> 2;
                    let got: u32 = (0..b).map(|i| (((out[i] >> l) & 1) as u32) << i).sum();
                    assert_eq!(got, want, "b={b} round={round} {v:?}");
                }
            }
        }
    }
}
