#![allow(clippy::too_many_arguments)]
//! Measurements shared by FHP-III and ternary FHP: shear-wave decay
//! (viscosity), advection of a transverse wave by a uniform flow (Galilean
//! factor), and block-averaged velocity noise at rest.

use super::engine::{self, Ternary};
use super::{equilibrium, Rule, NCH};
use crate::bitsliced::Bitsliced;
use crate::{lowbias32, rules, Params};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Model {
    Fhp3,
    Ternary(Rule),
}

impl Model {
    pub fn name(&self) -> String {
        match self {
            Model::Ternary(Rule::Mix(k)) => format!("ternary-mix{k}"),
            _ => self.base_name().to_string(),
        }
    }
    fn base_name(&self) -> &'static str {
        match self {
            Model::Fhp3 => "fhp3",
            Model::Ternary(Rule::Other) => "ternary",
            Model::Ternary(Rule::Uniform) => "ternary-uniform",
            Model::Ternary(Rule::Optimal) => "ternary-optimal",
            Model::Ternary(Rule::Chiral) => "ternary-chiral",
            Model::Ternary(Rule::Mix(_)) => "ternary-mix",
        }
    }
    pub fn capacity(&self) -> u32 {
        match self {
            Model::Fhp3 => 1,
            Model::Ternary(_) => 2,
        }
    }
}

/// Per-site momentum (integer units 2 px, 2 py / sqrt 3) and mass; obstacle
/// sites have mass 0.
pub struct Fields {
    pub jx: Vec<i32>,
    pub jy: Vec<i32>,
    pub mass: Vec<i32>,
}

pub enum Lattice {
    Fhp3(Box<Bitsliced>),
    Ternary(Box<Ternary>),
}

/// Position of site (x, y) on the hexagonal lattice (odd rows shifted half a site).
pub fn position(x: usize, y: usize) -> (f64, f64) {
    (
        x as f64 + 0.5 * (y & 1) as f64,
        y as f64 * 3f64.sqrt() / 2.0,
    )
}

impl Lattice {
    /// A periodic lattice sampled from the local equilibrium at density `d`
    /// per channel and velocity `u(x, y)` (positions from `position`).
    pub fn new(
        model: Model,
        w: usize,
        h: usize,
        d: f64,
        seed: u32,
        u: impl Fn(f64, f64) -> (f64, f64),
    ) -> Self {
        let base = lowbias32(seed ^ 0x6a09_e667);
        let rnd = |site: usize, k: usize| {
            lowbias32(base ^ lowbias32((site as u32).wrapping_mul(8).wrapping_add(k as u32)))
        };
        let mut cache: Option<((f64, f64), [f64; NCH])> = None;
        let mut mean = |v: (f64, f64)| -> [f64; NCH] {
            if let Some((k, m)) = cache {
                if k == v {
                    return m;
                }
            }
            let m = equilibrium(model.capacity(), model.capacity(), d, v.0, v.1);
            cache = Some((v, m));
            m
        };
        match model {
            Model::Fhp3 => {
                let mut cells = vec![0u8; w * h];
                for y in 0..h {
                    for x in 0..w {
                        let (px, py) = position(x, y);
                        let m = mean(u(px, py));
                        let i = y * w + x;
                        cells[i] = (0..NCH).fold(0u8, |a, k| {
                            a | (((rnd(i, k) as f64 / 4294967296.0) < m[k]) as u8) << k
                        });
                    }
                }
                Lattice::Fhp3(Box::new(Bitsliced::from_cells(
                    w,
                    h,
                    &cells,
                    Params {
                        seed,
                        ..Default::default()
                    },
                )))
            }
            Model::Ternary(rule) => {
                let mut cells = vec![0u16; w * h];
                for y in 0..h {
                    for x in 0..w {
                        let (px, py) = position(x, y);
                        let c = engine::cdf(&mean(u(px, py)));
                        let i = y * w + x;
                        cells[i] = engine::sample(&c, |k| rnd(i, k));
                    }
                }
                Lattice::Ternary(Box::new(Ternary::new(
                    w,
                    h,
                    cells,
                    engine::Params {
                        seed,
                        rule,
                        boundary: None,
                    },
                )))
            }
        }
    }

    pub fn wh(&self) -> (usize, usize) {
        match self {
            Lattice::Fhp3(b) => (b.w, b.h),
            Lattice::Ternary(t) => (t.w, t.h),
        }
    }

    pub fn run(&mut self, n: usize) {
        match self {
            Lattice::Fhp3(b) => b.run(n),
            Lattice::Ternary(t) => t.run(n),
        }
    }

    pub fn fields(&self) -> Fields {
        match self {
            Lattice::Fhp3(b) => {
                let cells = b.to_cells();
                let p: Vec<(i32, i32)> = cells.iter().map(|&s| rules::momentum(s)).collect();
                Fields {
                    jx: p.iter().map(|q| q.0).collect(),
                    jy: p.iter().map(|q| q.1).collect(),
                    mass: cells
                        .iter()
                        .map(|&s| {
                            if s & crate::OBST != 0 {
                                0
                            } else {
                                rules::mass(s) as i32
                            }
                        })
                        .collect(),
                }
            }
            Lattice::Ternary(t) => {
                let st = |c: u16| c & !engine::OBST;
                Fields {
                    jx: t.cells.iter().map(|&c| super::momentum(st(c)).0).collect(),
                    jy: t.cells.iter().map(|&c| super::momentum(st(c)).1).collect(),
                    mass: t
                        .cells
                        .iter()
                        .map(|&c| {
                            if c & engine::OBST != 0 {
                                0
                            } else {
                                super::mass(c) as i32
                            }
                        })
                        .collect(),
                }
            }
        }
    }

    pub fn block_sums(&self, bw: usize, bh: usize) -> crate::coarse::BlockSums {
        match self {
            Lattice::Fhp3(b) => b.block_sums(bw, bh),
            Lattice::Ternary(t) => t.block_sums(bw, bh),
        }
    }
}

/// Least-squares line through (x, y): slope, intercept, slope standard error.
pub fn fit_line(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let slope = sxy / sxx;
    let icpt = my - slope * mx;
    let res: f64 = x
        .iter()
        .zip(y)
        .map(|(a, b)| (b - icpt - slope * a).powi(2))
        .sum();
    let se = if n > 2.0 {
        (res / (n - 2.0) / sxx).sqrt()
    } else {
        f64::NAN
    };
    (slope, icpt, se)
}

/// Complex projection sum_sites v(site) exp(-i k . r) / (sum of mass), in
/// lattice velocity units (v in integer momentum units: x halves, y sqrt3/2).
fn project(f: &Fields, w: usize, h: usize, comp_y: bool, kx: f64, ky: f64) -> (f64, f64) {
    let (mut re, mut im, mut m) = (0.0, 0.0, 0.0);
    let scale = if comp_y { 3f64.sqrt() / 2.0 } else { 0.5 };
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (px, py) = position(x, y);
            let ph = kx * px + ky * py;
            let v = if comp_y { f.jy[i] } else { f.jx[i] } as f64 * scale;
            re += v * ph.cos();
            im -= v * ph.sin();
            m += f.mass[i] as f64;
        }
    }
    let n = m / (w * h) as f64;
    // Amplitude of u for u = A sin(k.r + phi): |sum| = A N n / 2.
    (
        2.0 * re / (n * (w * h) as f64),
        2.0 * im / (n * (w * h) as f64),
    )
}

pub struct Series {
    pub t: Vec<f64>,
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

impl Series {
    pub fn amp(&self) -> Vec<f64> {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(a, b)| (a * a + b * b).sqrt())
            .collect()
    }
}

/// Shear wave u_x = u0 sin(k y), k = 2 pi m / (h sqrt3/2): the projected
/// amplitude every `every` steps for `steps` steps.
pub fn shear_series(
    model: Model,
    w: usize,
    h: usize,
    d: f64,
    m: usize,
    u0: f64,
    seed: u32,
    steps: usize,
    every: usize,
) -> Series {
    let ky = 2.0 * std::f64::consts::PI * m as f64 / (h as f64 * 3f64.sqrt() / 2.0);
    let mut l = Lattice::new(model, w, h, d, seed, |_, y| (u0 * (ky * y).sin(), 0.0));
    let mut s = Series {
        t: vec![],
        re: vec![],
        im: vec![],
    };
    let mut t = 0;
    loop {
        let (re, im) = project(&l.fields(), w, h, false, 0.0, ky);
        s.t.push(t as f64);
        s.re.push(re);
        s.im.push(im);
        if t >= steps {
            break;
        }
        l.run(every);
        t += every;
    }
    s
}

/// Fit the decay of a shear series: nu from ln amplitude, over the samples
/// from `t0` while the amplitude stays above `floor` times the first one.
pub fn shear_nu(s: &Series, k: f64, t0: f64, floor: f64) -> (f64, f64) {
    // The component along the initial phase (no noise bias, unlike |A|).
    let a0 = s.amp()[0];
    let (c, sn) = (s.re[0] / a0, s.im[0] / a0);
    let a: Vec<f64> =
        s.re.iter()
            .zip(&s.im)
            .map(|(r, i)| r * c + i * sn)
            .collect();
    let (mut x, mut y) = (vec![], vec![]);
    for i in 0..a.len() {
        if s.t[i] < t0 {
            continue;
        }
        if a[i] < floor * a0 {
            break;
        }
        x.push(s.t[i]);
        y.push(a[i].ln());
    }
    let (slope, _, se) = fit_line(&x, &y);
    (-slope / (k * k), se / (k * k))
}

/// Transverse wave u_y = a0 sin(k x) in a uniform flow u_x = U, k = 2 pi m / w:
/// phase every `every` steps. Returns the series and the measured mean u_x.
pub fn advect_series(
    model: Model,
    w: usize,
    h: usize,
    d: f64,
    u: f64,
    m: usize,
    a0: f64,
    seed: u32,
    steps: usize,
    every: usize,
) -> (Series, f64) {
    let kx = 2.0 * std::f64::consts::PI * m as f64 / w as f64;
    let mut l = Lattice::new(model, w, h, d, seed, |x, _| (u, a0 * (kx * x).sin()));
    let f = l.fields();
    let ux = f.jx.iter().map(|&v| v as f64 * 0.5).sum::<f64>()
        / f.mass.iter().map(|&v| v as f64).sum::<f64>();
    let mut s = Series {
        t: vec![],
        re: vec![],
        im: vec![],
    };
    let mut t = 0;
    loop {
        let (re, im) = project(&l.fields(), w, h, true, kx, 0.0);
        s.t.push(t as f64);
        s.re.push(re);
        s.im.push(im);
        if t >= steps {
            break;
        }
        l.run(every);
        t += every;
    }
    (s, ux)
}

/// Phase velocity of an advect series: the unwrapped phase of
/// sum v exp(-i k x) decreases at k c for a wave moving at speed c.
pub fn phase_speed(s: &Series, k: f64) -> (f64, f64) {
    let mut ph = Vec::with_capacity(s.t.len());
    let mut prev = 0.0;
    let mut off = 0.0;
    for i in 0..s.t.len() {
        let p = s.im[i].atan2(s.re[i]);
        if i > 0 {
            let d = p - prev;
            if d > std::f64::consts::PI {
                off -= 2.0 * std::f64::consts::PI;
            } else if d < -std::f64::consts::PI {
                off += 2.0 * std::f64::consts::PI;
            }
        }
        prev = p;
        ph.push(p + off);
    }
    let (slope, _, se) = fit_line(&s.t, &ph);
    (-slope / k, se / k)
}

/// Standard deviation of block-averaged u_x and u_y at rest, per block size
/// b (b x b sites), over `snaps` snapshots `gap` steps apart after `warm`
/// steps. Returns (b, std ux, std uy) per block size.
pub fn rest_noise(
    model: Model,
    w: usize,
    h: usize,
    d: f64,
    seed: u32,
    blocks: &[usize],
    warm: usize,
    snaps: usize,
    gap: usize,
) -> Vec<(usize, f64, f64)> {
    let mut l = Lattice::new(model, w, h, d, seed, |_, _| (0.0, 0.0));
    l.run(warm);
    let mut acc = vec![(0.0f64, 0.0f64, 0usize); blocks.len()];
    for _ in 0..snaps {
        let f = l.fields();
        for (bi, &b) in blocks.iter().enumerate() {
            let (nx, ny) = (w / b, h / b);
            let mut sums = vec![(0i64, 0i64, 0i64); nx * ny];
            for y in 0..ny * b {
                for x in 0..nx * b {
                    let i = y * w + x;
                    let s = &mut sums[(y / b) * nx + x / b];
                    s.0 += f.jx[i] as i64;
                    s.1 += f.jy[i] as i64;
                    s.2 += f.mass[i] as i64;
                }
            }
            for s in &sums {
                let m = s.2.max(1) as f64;
                let ux = s.0 as f64 * 0.5 / m;
                let uy = s.1 as f64 * 3f64.sqrt() / 2.0 / m;
                acc[bi].0 += ux * ux;
                acc[bi].1 += uy * uy;
                acc[bi].2 += 1;
            }
        }
        l.run(gap);
    }
    blocks
        .iter()
        .zip(acc)
        .map(|(&b, a)| (b, (a.0 / a.2 as f64).sqrt(), (a.1 / a.2 as f64).sqrt()))
        .collect()
}

/// Theory for rest_noise: independent sites, std = sqrt(3 var / N) / (7 d).
pub fn noise_theory(model: Model, d: f64, b: usize) -> f64 {
    let z = super::fugacity(model.capacity(), model.capacity(), d);
    let (_, var, _) = super::channel_moments(model.capacity(), z);
    (3.0 * var / (b * b) as f64).sqrt() / (7.0 * d)
}
