//! GPU engine against the CPU engines: bit-identical for the circuit, and
//! for f32 (same summation order). Needs an adapter.

use jacobi::bitsliced::Bitsliced;
use jacobi::gpu::{GpuCtx, GpuPanel};
use jacobi::reference::f32_run;
use jacobi::{qmax, Problem};

#[test]
fn gpu_matches_cpu() {
    let ctx = pollster::block_on(GpuCtx::new()).expect("GPU adapter");
    eprintln!("adapter: {}", ctx.adapter);
    for n in [64usize, 128, 192] {
        let p = Problem::new(n).unwrap();
        for bits in jacobi::BITS {
            // Random interior so every bit-plane carries data from the start.
            let mut s = 0x2545f4914f6cdd1du64 ^ n as u64 ^ (bits as u64) << 20;
            let mut start = p.initial_q(bits);
            for (k, v) in start.iter_mut().enumerate() {
                if !p.is_boundary(k / n, k % n) {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    *v = (s as u32) & qmax(bits);
                }
            }
            for round in [false, true] {
                let mut g = GpuPanel::from_q(&ctx, n, bits, &start).unwrap();
                let mut c = Bitsliced::new(n, bits, &start);
                g.round = round;
                c.round = round;
                for _ in 0..4 {
                    g.run(&ctx, 75);
                    c.run(75);
                    assert_eq!(pollster::block_on(g.read_q(&ctx)), c.read_q(), "n={n} b={bits} round={round} iter={}", c.iter);
                }
            }
        }
        let mut g = GpuPanel::f32(&ctx, &p).unwrap();
        let mut f = p.initial_f32();
        g.run(&ctx, 300);
        f32_run(n, &mut f, 300);
        let gf = pollster::block_on(g.read_f32(&ctx));
        let same = gf.iter().zip(&f).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        let maxdiff = gf.iter().zip(&f).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("n={n} f32: {same}/{} bit-identical, max diff {maxdiff:e}", f.len());
        assert_eq!(same, f.len());
    }
}

/// The three approaches on the GPU against their bit-sliced CPU engines,
/// same seeds, bit for bit (gas: states and accumulators).
#[test]
fn approaches_gpu_match_cpu() {
    use jacobi::{gas, ternary};
    let ctx = pollster::block_on(GpuCtx::new()).expect("GPU adapter");
    for n in [64usize, 128, 192] {
        let p = Problem::new(n).unwrap();
        for bits in [4usize, 8, 16] {
            let mut g = GpuPanel::stochastic(&ctx, n, bits, &p.initial_q(bits), 5).unwrap();
            let mut c = Bitsliced::from_problem(&p, bits);
            c.stochastic = true;
            c.seed = 5;
            for _ in 0..3 {
                g.run(&ctx, 300);
                c.run(300);
                assert_eq!(pollster::block_on(g.read_q(&ctx)), c.read_q(), "stochastic n={n} b={bits} iter={}", c.iter);
            }
        }
        // Ternary from the problem's start and from a random interior.
        let mut s = 0x51u64 ^ n as u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut random = ternary::initial(&p);
        for (k, v) in random.iter_mut().enumerate() {
            if !p.is_boundary(k / n, k % n) {
                *v = (rnd() % ternary::Q as u64) as i32;
            }
        }
        for start in [ternary::initial(&p), random] {
            let mut g = GpuPanel::ternary(&ctx, n, &start).unwrap();
            let mut c = ternary::Bitsliced::new(n, &start);
            for _ in 0..4 {
                g.run(&ctx, 150);
                c.run(150);
                let gu: Vec<i32> = pollster::block_on(g.read_planes(&ctx, ternary::W)).into_iter().map(|x| x as i32).collect();
                assert_eq!(gu, c.read(), "ternary n={n} iter={}", c.iter);
            }
        }
        for ann in [false, true] {
            let mut g = GpuPanel::gas(&ctx, &p, &gas::initial(n), 9, ann).unwrap();
            let mut c = gas::Bitsliced::from_problem(&p, 9, ann);
            for _ in 0..3 {
                g.run(&ctx, 200);
                c.run(200);
                let gs: Vec<u8> = pollster::block_on(g.read_planes(&ctx, 8)).into_iter().map(|x| x as u8).collect();
                assert_eq!(gs, c.read(), "gas n={n} ann={ann} iter={}", c.iter);
                let ga: Vec<i32> = pollster::block_on(g.read_aux(&ctx)).into_iter().map(|x| x as i32).collect();
                assert_eq!(ga, c.acc, "gas acc n={n} ann={ann} iter={}", c.iter);
            }
        }
    }
}
