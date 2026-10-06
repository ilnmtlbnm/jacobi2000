//! The two control panels (f32 red-black, int32 red-black) and the sinh
//! problem: circuit against integer arithmetic, bit-sliced and GPU engines
//! against the references, boundary values.

use jacobi::gpu::{GpuCtx, GpuPanel};
use jacobi::reference::{f32_rb_run, F32Engine};
use jacobi::{int32, ternary, Case, Problem};

fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn random_interior(p: &Problem, seed: u64) -> Vec<i32> {
    let mut rnd = xorshift(seed);
    let n = p.n;
    let mut u = ternary::initial(p);
    for (k, v) in u.iter_mut().enumerate() {
        if !p.is_boundary(k / n, k % n) {
            *v = (rnd() % (ternary::Q as u64 + 1)) as i32;
        }
    }
    u
}

/// The int32 circuit equals (E + W + N + S + 2) >> 2 in wrapping i32
/// arithmetic, on random words (any 32-bit value, so the wrap and the sign
/// are exercised) and on the engine's range [0, 2^28].
#[test]
fn int32_circuit_is_integer_arithmetic() {
    let c = int32::circuit();
    eprintln!("int32 circuit: {} gates, per phase {:?} ({:?}), AND/OR/XOR/NOT {:?}", c.gate_count(), c.phase_counts(), int32::PHASE_NAMES, c.counts());
    let mut rnd = xorshift(0xabcdef);
    for case in 0..4000 {
        let vals: Vec<[i32; 4]> = (0..64)
            .map(|_| {
                let mut r = || if case % 2 == 0 { rnd() as i32 } else { (rnd() % (ternary::Q as u64 + 1)) as i32 };
                [r(), r(), r(), r()]
            })
            .collect();
        let mut inputs = vec![0u64; c.inputs()];
        for (l, v) in vals.iter().enumerate() {
            for k in 0..4 {
                for j in 0..32 {
                    if (v[k] >> j) & 1 == 1 {
                        inputs[ternary::input(k, j)] |= 1 << l;
                    }
                }
            }
        }
        let out = c.eval(&inputs);
        for (l, v) in vals.iter().enumerate() {
            let got = (0..32).map(|j| (((out[j] >> l) & 1) as u32) << j).sum::<u32>() as i32;
            let want = v[0].wrapping_add(v[1]).wrapping_add(v[2].wrapping_add(v[3])).wrapping_add(2) >> 2;
            assert_eq!(got, want, "case {case} lane {l} {v:?}");
        }
    }
}

#[test]
fn int32_bitsliced_matches_integer() {
    for n in [64usize, 128] {
        for case in [Case::Quadratic, Case::Sinh] {
            let p = Problem::with_case(n, case).unwrap();
            for start in [ternary::initial(&p), random_interior(&p, 7 + n as u64)] {
                let mut r = int32::IntEngine::new(n, start.clone());
                let mut b = int32::Bitsliced::new(n, &start);
                for _ in 0..6 {
                    r.run(40);
                    b.run(40);
                    assert_eq!(b.read(), r.cur, "n={n} {case:?} iter={}", r.iter);
                }
            }
        }
    }
}

/// int32 red-black on the GPU against the bit-sliced CPU engine and the
/// integer reference, bit for bit.
#[test]
fn int32_gpu_matches_cpu() {
    let ctx = pollster::block_on(GpuCtx::new()).expect("GPU adapter");
    for n in [64usize, 128, 192] {
        for case in [Case::Quadratic, Case::Sinh] {
            let p = Problem::with_case(n, case).unwrap();
            for start in [ternary::initial(&p), random_interior(&p, 11 + n as u64)] {
                let mut g = GpuPanel::int32_red_black(&ctx, n, &start).unwrap();
                let mut b = int32::Bitsliced::new(n, &start);
                let mut r = int32::IntEngine::new(n, start.clone());
                for _ in 0..4 {
                    g.run(&ctx, 150);
                    b.run(150);
                    r.run(150);
                    let gu: Vec<i32> = pollster::block_on(g.read_planes(&ctx, ternary::W)).into_iter().map(|x| x as i32).collect();
                    assert_eq!(gu, b.read(), "int32 GPU vs CPU n={n} {case:?} iter={}", b.iter);
                    assert_eq!(gu, r.cur, "int32 GPU vs reference n={n} {case:?} iter={}", r.iter);
                }
            }
        }
    }
}

/// The f32 red-black reference is Gauss-Seidel on the checkerboard (an
/// in-place loop over colour 0 then colour 1), and the CPU engine and the
/// GPU engine match it bit for bit.
#[test]
fn f32_red_black_matches_reference() {
    let ctx = pollster::block_on(GpuCtx::new()).expect("GPU adapter");
    for n in [64usize, 128, 192] {
        for case in [Case::Quadratic, Case::Sinh] {
            let p = Problem::with_case(n, case).unwrap();
            let mut r = p.initial_f32();
            let mut gs = p.initial_f32();
            let mut e = F32Engine::new(n, p.initial_f32());
            e.red_black = true;
            let mut g = GpuPanel::f32_red_black(&ctx, &p).unwrap();
            for _ in 0..3 {
                f32_rb_run(n, &mut r, 100);
                for _ in 0..100 {
                    for c in 0..2 {
                        for i in 1..n - 1 {
                            for j in 1..n - 1 {
                                if (i + j) % 2 == c {
                                    let k = i * n + j;
                                    gs[k] = (((gs[k + 1] + gs[k - 1]) + gs[k - n]) + gs[k + n]) * 0.25;
                                }
                            }
                        }
                    }
                }
                e.run(100);
                g.run(&ctx, 100);
                let gf = pollster::block_on(g.read_f32(&ctx));
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&gs), bits(&r), "in-place Gauss-Seidel n={n} {case:?}");
                assert_eq!(bits(&e.cur), bits(&r), "CPU engine n={n} {case:?}");
                let same = gf.iter().zip(&r).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
                eprintln!("n={n} {case:?} f32 red-black GPU: {same}/{} bit-identical", r.len());
                assert_eq!(bits(&gf), bits(&r), "GPU n={n} {case:?}");
            }
        }
    }
}

/// u = sin(pi x) sinh(pi y) / sinh(pi), v = (u + 1) / 2: v = 1/2 on x = +-1,
/// (1 + sin(pi x)) / 2 on y = 1 and (1 - sin(pi x)) / 2 on y = -1; the
/// engines start from those values on the boundary and zero inside.
#[test]
fn sinh_boundary_values() {
    use std::f64::consts::PI;
    for n in [64usize, 128, 512] {
        let p = Problem::with_case(n, Case::Sinh).unwrap();
        assert_eq!(p.coord(0), -1.0);
        assert_eq!(p.coord(n - 1), 1.0);
        let f = p.initial_f32();
        let u = ternary::initial(&p);
        for i in 0..n {
            for j in 0..n {
                let k = i * n + j;
                let (x, y) = (p.coord(i), p.coord(j));
                let want = if i == 0 || i == n - 1 {
                    Some(0.5)
                } else if j == n - 1 {
                    Some((1.0 + (PI * x).sin()) / 2.0)
                } else if j == 0 {
                    Some((1.0 - (PI * x).sin()) / 2.0)
                } else {
                    None
                };
                match want {
                    Some(v) => {
                        assert!((p.analytic(i, j) - v).abs() < 1e-15, "n={n} ({x}, {y}): {} vs {v}", p.analytic(i, j));
                        assert_eq!(f[k], p.analytic(i, j) as f32);
                        assert_eq!(u[k], (p.analytic(i, j) * ternary::Q as f64).round_ties_even() as i32);
                        assert!((0.0..=1.0).contains(&p.analytic(i, j)));
                    }
                    None => {
                        assert_eq!(f[k], 0.0);
                        assert_eq!(u[k], 0);
                    }
                }
            }
        }
        // Harmonic: the continuous Laplacian vanishes; the 5-point one is
        // O(h^2) but not zero (the discrete solution differs from u).
        let h = 2.0 / (n - 1) as f64;
        let (i, j) = (n / 3, 3 * n / 4);
        let lap = (p.analytic(i + 1, j) + p.analytic(i - 1, j) + p.analytic(i, j + 1) + p.analytic(i, j - 1) - 4.0 * p.analytic(i, j)) / (h * h);
        assert!(lap.abs() > 1e-6 && lap.abs() < 10.0 * h * h * PI.powi(4), "n={n} discrete Laplacian {lap:e}");
    }
    // The default problem is unchanged.
    let q = Problem::new(64).unwrap();
    assert_eq!(q.case, Case::Quadratic);
    assert_eq!(q.analytic(0, 0), 0.5);
    assert_eq!(q.analytic(0, 31), (1.0 - q.coord(31) * q.coord(31) + 1.0) / 2.0);
}

/// The per-block scale with one block covering the grid is the global
/// adaptive schedule of the integer engine.
#[test]
fn one_block_is_the_global_scale() {
    let p = Problem::with_case(64, Case::Sinh).unwrap();
    let mut e = ternary::IntEngine::new(&p, ternary::Schedule::Adaptive);
    e.red_black = true;
    e.run(500);
    let mut u = ternary::initial(&p);
    let mut other = u.clone();
    let mut ors = [0u32; 1];
    for k in 0..500 {
        ternary::sweep_blocks(64, &mut u, &mut other, 64, &mut ors, k == 0);
    }
    assert_eq!(u, e.cur);
}
