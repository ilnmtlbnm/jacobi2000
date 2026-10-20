//! The signed GPU engine against the CPU engine: the whole state (cells,
//! reference, block values, reservoirs) bit for bit over several hundred
//! steps, periodic and with the plate and the open boundary, both GPU rules;
//! the page's block sums; exact conservation in the integer units; and a
//! longer street at the stable setting (B = 16, N = 4) without divergence.

#![cfg(feature = "gpu")]

use fhp::signed::gpu::Gpu;
use fhp::signed::{Boundary, Params, Signed, OBST};
use fhp::ternary::{Rule, NCH};
use fhp::{CX2, CY2, NDIR};

fn gpu(s: &Signed) -> Option<Gpu> {
    match pollster::block_on(Gpu::from_cpu(s)) {
        Ok(g) => Some(g),
        Err(e) if e.contains("adapter") => {
            eprintln!("no GPU adapter; skipped");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

fn plate(w: usize, h: usize) -> impl Fn(usize, usize) -> bool {
    let (px, py, rows) = (w / 6, h / 2 + 1, h / 8);
    move |x, y| (px..px + 2).contains(&x) && (py - rows / 2..py - rows / 2 + rows).contains(&y)
}

fn smooth(w: usize, h: usize, p: Params, obstacle: impl Fn(usize, usize) -> bool) -> Signed {
    let ky = 2.0 * std::f64::consts::PI / (h as f64 * 3f64.sqrt() / 2.0);
    let kx = 2.0 * std::f64::consts::PI / w as f64;
    let rho0 = 7.0 * p.n0;
    Signed::sample(w, h, p, obstacle, |x, y| (rho0 * (1.0 + 0.05 * (kx * x).cos()), 0.15 + 0.08 * (ky * y).sin(), 0.04 * (kx * x).sin()), 3)
}

fn same(s: &Signed, g: &Gpu, what: &str) {
    let st = pollster::block_on(g.read_state());
    assert_eq!(g.step, s.step, "{what}");
    let bad = (0..s.w * s.h).find(|&i| st.cells[i] != s.cells[i]);
    assert!(bad.is_none(), "{what} step {}: cells differ first at site {:?} ({} vs {})", s.step, bad, bad.map_or(0, |i| st.cells[i]), bad.map_or(0, |i| s.cells[i]));
    let bad = (0..s.w * s.h).find(|&i| st.phi[i] != s.phi[i]);
    assert!(bad.is_none(), "{what} step {}: phi differs first at site {:?}", s.step, bad.map(|i| (i, st.phi[i], s.phi[i])));
    assert_eq!(st.nodes, s.nodes, "{what} step {}: block values", s.step);
    assert_eq!(st.res, s.res, "{what} step {}: reservoirs", s.step);
}

fn compare(mut s: Signed, steps: usize, every: usize, what: &str) {
    let Some(mut g) = gpu(&s) else { return };
    same(&s, &g, what);
    for _ in 0..steps / every {
        s.run(every);
        g.run(every);
        same(&s, &g, what);
    }
    assert!(s.kicks > 0, "{what}: no kicks");
}

fn params(rule: Rule, block: usize, every: u32, boundary: Option<Boundary>) -> Params {
    Params { seed: 11, rule, block, every, n0: 4.0, boundary }
}

#[test]
fn gpu_equals_cpu_periodic() {
    compare(smooth(128, 64, params(Rule::Optimal, 8, 4, None), |_, _| false), 300, 50, "128x64 B8 N4 optimal");
    compare(smooth(192, 64, params(Rule::Chiral, 4, 1, None), |_, _| false), 200, 40, "192x64 B4 N1 chiral");
    compare(smooth(64, 32, params(Rule::Optimal, 2, 1, None), |_, _| false), 120, 40, "64x32 B2 N1 optimal");
    // Five block rows: the second tile has one.
    compare(smooth(128, 40, params(Rule::Optimal, 8, 1, None), |_, _| false), 120, 40, "128x40 B8 N1 optimal");
}

#[test]
fn gpu_equals_cpu_with_plate_and_open_boundary() {
    for (w, h, rule) in [(256, 128, Rule::Optimal), (384, 64, Rule::Chiral)] {
        let bd = Some(Boundary { rho: 28.0, u: 0.2, inlet: 32, outlet: 64 });
        let s = Signed::sample(w, h, params(rule, 16, 4, bd), plate(w, h), |_, _| (28.0, 0.2, 0.0), 1);
        assert!(s.cells.iter().any(|c| c & OBST != 0));
        compare(s, 400, 100, &format!("{w}x{h} street {rule:?}"));
    }
}

#[test]
fn gpu_block_sums_equal_the_cpu_page_sums() {
    let (w, h) = (256, 128);
    let bd = Some(Boundary { rho: 28.0, u: 0.2, inlet: 32, outlet: 64 });
    let mut s = Signed::sample(w, h, params(Rule::Optimal, 16, 4, bd), plate(w, h), |_, _| (28.0, 0.2, 0.0), 1);
    let Some(mut g) = gpu(&s) else { return };
    for b in [8, 16, 32] {
        s.run(37);
        g.run(37);
        let mut cpu = s.page_sums(b, b);
        g.accumulate_sums(b, b);
        s.run(5);
        g.run(5);
        cpu.add(&s.page_sums(b, b));
        g.accumulate_sums(b, b);
        assert_eq!(pollster::block_on(g.take_sums()), cpu, "block {b}");
    }
}

/// Mass and momentum of a read-back state in integer units, exact.
fn totals(cells: &[u16], phi: &[[i32; NCH]], res: &[[i32; NCH]]) -> (i64, i64, i64) {
    let mut t = [0i64; NCH];
    for (c, p) in cells.iter().zip(phi) {
        let n = fhp::ternary::counts(c & fhp::signed::STATE);
        for ch in 0..NCH {
            t[ch] += p[ch] as i64 + (n[ch] as i64 - 1) * fhp::signed::ONE as i64;
        }
    }
    for r in res {
        for ch in 0..NCH {
            t[ch] += r[ch] as i64;
        }
    }
    (t.iter().sum(), (0..NDIR).map(|c| CX2[c] as i64 * t[c]).sum(), (0..NDIR).map(|c| CY2[c] as i64 * t[c]).sum())
}

#[test]
fn gpu_conserves_mass_and_momentum_exactly() {
    let s = smooth(256, 128, params(Rule::Optimal, 16, 4, None), |_, _| false);
    let Some(mut g) = gpu(&s) else { return };
    let t0 = s.totals_fixed();
    for _ in 0..6 {
        g.run(101);
        let st = pollster::block_on(g.read_state());
        assert_eq!(totals(&st.cells, &st.phi, &st.res), t0, "step {}", g.step);
    }
}

/// The street at the stable setting on 1024 x 256: 6000 steps, the mean
/// speed stays near the inflow and every block value finite and bounded.
#[test]
fn gpu_street_at_b16_n4_does_not_diverge() {
    let (w, h) = (1024, 256);
    let bd = Some(Boundary { rho: 28.0, u: 0.2, inlet: 32, outlet: 64 });
    let s = Signed::sample(w, h, params(Rule::Optimal, 16, 4, bd), plate(w, h), |_, _| (28.0, 0.2, 0.0), 1);
    let Some(mut g) = gpu(&s) else { return };
    for _ in 0..6 {
        g.run(1000);
        let st = pollster::block_on(g.read_state());
        let q = fhp::signed::ONE as f64;
        let (mut m, mut jx) = (0.0, 0.0);
        for v in &st.nodes {
            let r = v[0] as f64 / q;
            let u = (v[1] as f64 / 2.0).hypot(v[2] as f64 * 3f64.sqrt() / 2.0) / v[0] as f64;
            assert!(r > 14.0 && r < 42.0 && u < 0.6, "step {}: block rho {r}, u {u}", g.step);
            m += r;
            jx += v[1] as f64 / 2.0 / q;
        }
        let ux = jx / m;
        assert!(ux > 0.15 && ux < 0.25, "step {}: mean u_x {ux}", g.step);
    }
}
