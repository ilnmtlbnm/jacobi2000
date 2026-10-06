//! The three approaches: gate circuits against integer arithmetic, bit-sliced
//! engines against the per-site references, and particle conservation.

use jacobi::circuit::Circuit;
use jacobi::{gas, qmax, stochastic, ternary, Problem};

fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// Evaluate `c` on 64 lanes, lane l's input k being bit k of `lanes[l]`
/// (inputs given as a list of (input index, value bit) per lane via `bit`).
fn eval_lanes(c: &Circuit, bit: impl Fn(usize, usize) -> bool) -> Vec<u64> {
    let mut inputs = vec![0u64; c.inputs()];
    for (k, w) in inputs.iter_mut().enumerate() {
        for l in 0..64 {
            if bit(l, k) {
                *w |= 1 << l;
            }
        }
    }
    c.eval(&inputs)
}

fn lane(out: &[u64], range: std::ops::Range<usize>, l: usize) -> u32 {
    range.enumerate().map(|(b, o)| (((out[o] >> l) & 1) as u32) << b).sum()
}

/// The ternary circuit equals the integer update t(D, s), u + s t, and
/// D ^ (D >> 31), on random fields within the engine's range and random
/// scales (powers of two and others), including D at the dead-zone edges.
#[test]
fn ternary_circuit_is_integer_arithmetic() {
    let c = ternary::circuit();
    let counts = c.phase_counts();
    eprintln!("ternary circuit: {} gates, per phase {:?} ({:?})", c.gate_count(), counts, ternary::PHASE_NAMES);
    let mut rnd = xorshift(0x1234_5678);
    for case in 0..4000 {
        let s: i32 = if case % 2 == 0 { 1 << (rnd() % 27) } else { (rnd() % (1 << 26)) as i32 + 1 };
        let vals: Vec<[i32; 5]> = (0..64)
            .map(|l| {
                let r = |rnd: &mut dyn FnMut() -> u64| (rnd() % (1 << 29)) as i32 - (1 << 27);
                let mut v = [r(&mut rnd), r(&mut rnd), r(&mut rnd), r(&mut rnd), r(&mut rnd)];
                // Put D exactly at or next to +-2s for some lanes.
                if l % 4 == 1 {
                    let sum = v[0].wrapping_add(v[1]).wrapping_add(v[2]);
                    let target = [2 * s, 2 * s - 1, -2 * s, -2 * s + 1, 0][(rnd() % 5) as usize];
                    // D = sum + v3 - 4 v4 = target.
                    v[3] = target.wrapping_add(v[4].wrapping_mul(4)).wrapping_sub(sum);
                }
                v
            })
            .collect();
        let k = ternary::constants(s);
        let out = eval_lanes(&c, |l, i| {
            let (op, j) = (i / ternary::W, i % ternary::W);
            let v = if op < 5 { vals[l][op] as u32 } else { k[op - 5] };
            (v >> j) & 1 == 1
        });
        for (l, v) in vals.iter().enumerate() {
            let d = v[0].wrapping_add(v[1]).wrapping_add(v[2]).wrapping_add(v[3]).wrapping_sub(v[4].wrapping_mul(4));
            let t = ternary::trit(d, s, false);
            let want = v[4].wrapping_add(t.wrapping_mul(s)) as u32;
            assert_eq!(lane(&out, 0..32, l), want, "s={s} v={v:?} d={d}");
            assert_eq!(lane(&out, 32..63, l), ((d ^ (d >> 31)) as u32) & 0x7fff_ffff, "s={s} d={d}");
        }
    }
}

fn random_ternary_field(p: &Problem, seed: u64) -> Vec<i32> {
    let mut rnd = xorshift(seed);
    let mut u = ternary::initial(p);
    let n = p.n;
    for (k, v) in u.iter_mut().enumerate() {
        if !p.is_boundary(k / n, k % n) {
            *v = (rnd() % (ternary::Q as u64)) as i32;
        }
    }
    u
}

/// The bit-sliced ternary engine equals the integer engine (red-black,
/// adaptive schedule, non-strict dead zone) after every block of sweeps.
#[test]
fn ternary_bitsliced_matches_integer() {
    for n in [64usize, 128] {
        let p = Problem::new(n).unwrap();
        for start in [ternary::initial(&p), random_ternary_field(&p, 99 + n as u64)] {
            let mut b = ternary::Bitsliced::new(n, &start);
            let mut r = ternary::IntEngine::from_field(n, start.clone(), ternary::Schedule::Adaptive);
            r.red_black = true;
            for _ in 0..6 {
                b.run(40);
                r.run(40);
                assert_eq!(b.read(), r.cur, "n={n} iter={}", b.iter);
                assert_eq!(b.last_or, r.last_or);
            }
        }
    }
}

/// The stochastic-rounding circuit equals (sum + r) >> 2 for the same r.
#[test]
fn stochastic_circuit_is_integer_stochastic_rounding() {
    let mut rnd = xorshift(0xfeed);
    for b in [4usize, 8, 12, 16] {
        let c = Circuit::jacobi_stochastic(b);
        eprintln!("stochastic rounding b={b}: {} gates", c.gate_count());
        let cases = 1 << 18;
        for chunk in 0..cases / 64 {
            let vals: Vec<[u32; 5]> = (0..64)
                .map(|l| {
                    if b == 4 {
                        let x = (chunk * 64 + l) as u32;
                        [x & 15, (x >> 4) & 15, (x >> 8) & 15, (x >> 12) & 15, (x >> 16) & 3]
                    } else {
                        let r = rnd();
                        let m = qmax(b) as u64;
                        [(r & m) as u32, ((r >> 16) & m) as u32, ((r >> 32) & m) as u32, ((r >> 48) & m) as u32, (rnd() & 3) as u32]
                    }
                })
                .collect();
            let out = eval_lanes(&c, |l, i| if i < 4 * b { (vals[l][i / b] >> (i % b)) & 1 == 1 } else { (vals[l][4] >> (i - 4 * b)) & 1 == 1 });
            for (l, v) in vals.iter().enumerate() {
                let want = (v[0] + v[1] + v[2] + v[3] + v[4]) >> 2;
                assert_eq!(lane(&out, 0..b, l), want, "b={b} {v:?}");
            }
        }
    }
}

/// The bit-sliced engine with stochastic rounding equals integer
/// stochastic rounding drawing the same random bits.
#[test]
fn stochastic_bitsliced_matches_integer() {
    for n in [64usize, 128, 192] {
        let p = Problem::new(n).unwrap();
        for bits in [4usize, 8, 16] {
            let mut e = jacobi::bitsliced::Bitsliced::from_problem(&p, bits);
            e.stochastic = true;
            e.seed = 7;
            let mut q = p.initial_q(bits);
            for _ in 0..5 {
                let first = e.iter as u32;
                e.run(37);
                stochastic::int_run(n, &mut q, 37, 7, first);
                assert_eq!(e.read_q(), q, "n={n} b={bits} iter={}", e.iter);
            }
        }
    }
}

/// The collision circuit equals annihilate-then-rotate on every input.
#[test]
fn gas_circuit_is_the_collision() {
    for ann in [false, true] {
        let c = gas::circuit(ann);
        eprintln!("gas collision, annihilation={ann}: {} gates", c.gate_count());
        for base in (0..1024usize).step_by(64) {
            let out = eval_lanes(&c, |l, i| ((base + l) >> i) & 1 == 1);
            for l in 0..64 {
                let x = base + l;
                let s = (x & 0xff) as u8;
                let r = (x >> 8) as u32;
                let a = if ann { gas::annihilate(s) } else { s };
                assert_eq!(lane(&out, 0..8, l), gas::rotate(a, r) as u32, "ann={ann} s={s:08b} r={r}");
            }
        }
    }
}

/// The bit-sliced gas equals the per-site reference, states and
/// accumulators, with and without annihilation.
#[test]
fn gas_bitsliced_matches_reference() {
    for n in [64usize, 128] {
        let p = Problem::new(n).unwrap();
        for ann in [false, true] {
            let mut b = gas::Bitsliced::from_problem(&p, 3, ann);
            let mut r = gas::RefEngine::new(&p, 3, ann);
            for _ in 0..5 {
                b.run(23);
                r.run(23);
                assert_eq!(b.read(), r.cur, "n={n} ann={ann} iter={}", b.iter);
                assert_eq!(b.acc, r.acc, "n={n} ann={ann} iter={}", b.iter);
            }
        }
    }
}

/// Particle bookkeeping of one step: without annihilation every particle
/// that arrives at an interior site is still there after the collision, so
/// the interior count after the step equals the number of particles that
/// were in channels pointing at interior sites; with annihilation the same
/// holds for the charge. Checked on the bit-sliced engine for 200 steps.
#[test]
fn gas_conserves_particles_in_the_interior() {
    let n = 64;
    let p = Problem::new(n).unwrap();
    let interior = |i: usize, j: usize| i > 0 && j > 0 && i < n - 1 && j < n - 1;
    for ann in [false, true] {
        let mut e = gas::Bitsliced::from_problem(&p, 11, ann);
        for _ in 0..200 {
            let before = e.read();
            let (mut inflow_count, mut inflow_charge) = (0i64, 0i64);
            for i in 0..n {
                for j in 0..n {
                    let s = before[i * n + j];
                    for (c, (di, dj)) in [(0i64, 1i64), (1, 0), (0, -1), (-1, 0)].iter().enumerate() {
                        let (ti, tj) = (i as i64 + di, j as i64 + dj);
                        if ti < 0 || tj < 0 || ti >= n as i64 || tj >= n as i64 || !interior(ti as usize, tj as usize) {
                            continue;
                        }
                        let (pp, mm) = ((s >> c) & 1, (s >> (4 + c)) & 1);
                        inflow_count += (pp + mm) as i64;
                        inflow_charge += pp as i64 - mm as i64;
                    }
                }
            }
            e.run(1);
            let after = e.read();
            let (mut count, mut charge) = (0i64, 0i64);
            for i in 1..n - 1 {
                for j in 1..n - 1 {
                    let s = after[i * n + j];
                    count += s.count_ones() as i64;
                    charge += gas::charge(s) as i64;
                }
            }
            assert_eq!(charge, inflow_charge, "ann={ann} step {}", e.iter);
            if !ann {
                assert_eq!(count, inflow_count, "step {}", e.iter);
            } else {
                assert!(count <= inflow_count && (inflow_count - count) % 2 == 0);
            }
        }
    }
}
