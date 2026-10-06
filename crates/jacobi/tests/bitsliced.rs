use jacobi::bitsliced::Bitsliced;
use jacobi::reference::int_run;
use jacobi::{qmax, Problem};

fn random_interior(p: &Problem, bits: usize, seed: u64) -> Vec<u32> {
    let mut s = seed | 1;
    let mut q = p.initial_q(bits);
    let n = p.n;
    for (k, v) in q.iter_mut().enumerate() {
        if !p.is_boundary(k / n, k % n) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *v = (s as u32) & qmax(bits);
        }
    }
    q
}

/// The bit-sliced engine equals integer Jacobi at every point after every
/// block of iterations, for every width and both roundings, from the
/// problem's start and from a random interior.
#[test]
fn bitsliced_matches_integer_jacobi() {
    for n in [64usize, 128, 192] {
        let p = Problem::new(n).unwrap();
        for bits in jacobi::BITS {
            for round in [false, true] {
                for start in [p.initial_q(bits), random_interior(&p, bits, 7 + bits as u64)] {
                    let mut e = Bitsliced::new(n, bits, &start);
                    e.round = round;
                    let mut q = start.clone();
                    for _ in 0..6 {
                        e.run(50);
                        int_run(n, &mut q, 50, round);
                        assert_eq!(e.read_q(), q, "n={n} b={bits} round={round} iter={}", e.iter);
                    }
                }
            }
        }
    }
}

/// Switching the rounding mid-run stays exact.
#[test]
fn rounding_switch_mid_run() {
    let p = Problem::new(64).unwrap();
    let mut e = Bitsliced::from_problem(&p, 16);
    let mut q = p.initial_q(16);
    e.run(100);
    int_run(64, &mut q, 100, false);
    e.round = true;
    e.run(100);
    int_run(64, &mut q, 100, true);
    assert_eq!(e.read_q(), q);
}
