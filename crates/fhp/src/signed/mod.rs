//! Signed ternary FHP: each channel holds a deviation delta_i in {-1, 0, +1}
//! around a reference equilibrium F, and the represented occupation is
//! f_i = F_i + s delta_i (units with s = 1 here).
//!
//! - delta is stored as 3A's base-3 code of n = delta + 1 (`crate::ternary`),
//!   so the (sum delta, sum c delta) classes are 3A's (mass, momentum)
//!   classes and the collision is 3A's table.
//! - The reference is per block of B x B sites: a density and momentum per
//!   block, interpolated bilinearly to the sites, through the polynomial
//!   equilibrium `feq_fixed` (exact momentum flux rho u u, c_s^2 = 3/7).
//! - Streaming moves delta and adds the reference mismatch
//!   phi(x - c) - phi(x) by stochastic rounding; the rounding residual and
//!   saturation overflow go to a per-block, per-channel reservoir (parked,
//!   not streaming), so every channel's total is conserved exactly.
//! - Every N steps the block references are re-estimated from block sums
//!   and delta re-quantized; the reservoir takes the exact remainder.
//!
//! Everything but the deviations is fixed point: i32 in units of 2^-12
//! particles (`ONE`), with the same integer operations as the WGSL engine
//! (`gpu`), so the two agree bit for bit and conservation is exact in those
//! units. With N = 0 (never re-estimated) and a uniform reference this is
//! 3A, state for state (a test checks it bit for bit).


pub mod gpu;
pub mod measure;

use crate::ternary::{counts, Rule, Table, NCH, NSTATE, POW3};
use crate::{lowbias32, source, step_key, CX2, CY2, NDIR};

/// Diagnostics: keep the reference at rest (density only), and skip the
/// re-quantization kicks (the reservoir takes everything).
pub static RHOONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static NOKICK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Diagnostic, GPU engine only: obstacle sites with `WALL` set reflect
/// specularly (free-slip walls) and hold the inflow reference, and the
/// reference is not interpolated across the y wrap.
pub static WALLS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub const WALL: u16 = 1 << 14;
/// Diagnostic: the column geometry of the first engine (the reference
/// interpolated at the column index, so odd rows' half-site offset is
/// ignored and the block values sit at the blocks' column centres).
pub static COLUMNS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub const OBST: u16 = 1 << 15;
pub const STATE: u16 = OBST - 1;
/// The all-zero deviation (n = 1 in every channel).
pub const ZERO: u16 = 1093;

/// Fraction bits of the fixed-point reference, reservoir and sums.
pub const FRAC: u32 = 12;
pub const ONE: i32 = 1 << FRAC;

pub const STREAM_COLLIDE: u32 = 0x7e01;
pub const STREAM_ROUND: u32 = 0x7e11;
pub const STREAM_REQUANT: u32 = 0x7e12;
pub const STREAM_REFRESH: u32 = 0x7e13;
/// Open-boundary refresh period, as 3A and FHP-III.
pub const REFRESH: u32 = 32;
/// Display block sums are shifted down by this many bits per sample, so a
/// frame's sum of 32 x 32 blocks fits an i32.
pub const SUM_SHIFT: u32 = 4;

pub fn velocity(i: usize) -> (f64, f64) {
    crate::ternary::velocity(i)
}

/// Polynomial equilibrium on the seven FHP channels with exact momentum
/// flux: sum F = rho, sum c F = rho u, sum c c F = (3/7) rho I + rho u u.
pub fn feq(rho: f64, ux: f64, uy: f64) -> [f64; NCH] {
    let u2 = ux * ux + uy * uy;
    std::array::from_fn(|i| {
        if i == 6 {
            rho * (1.0 / 7.0 - u2)
        } else {
            let c = velocity(i);
            let cu = c.0 * ux + c.1 * uy;
            rho * (1.0 / 7.0 + cu / 3.0 + (2.0 / 3.0) * cu * cu - u2 / 6.0)
        }
    })
}

/// trunc(num 2^16 / den) in two 8-bit digits, so no product leaves i32
/// for |num|, den < 2^23 (as `div16` in the shader).
#[inline(always)]
pub fn div16(num: i32, den: i32) -> i32 {
    let a = num.wrapping_mul(256);
    let q1 = a.wrapping_div(den);
    let r1 = a.wrapping_sub(q1.wrapping_mul(den));
    let q2 = r1.wrapping_mul(256).wrapping_div(den);
    q1.wrapping_mul(256).wrapping_add(q2)
}

/// sign(a b) (|a| |b| >> s), the low 32 bits (as `mul_shr` in the shader).
#[inline(always)]
pub fn mul_shr(a: i32, b: i32, s: u32) -> i32 {
    let p = ((a.unsigned_abs() as u64 * b.unsigned_abs() as u64) >> s) as u32 as i32;
    if (a < 0) != (b < 0) {
        p.wrapping_neg()
    } else {
        p
    }
}

/// `feq` in fixed point from the site values r = rho ONE, jx = 2 j_x ONE,
/// jy = (2 / sqrt 3) j_y ONE (the integer momentum units of `CX2`, `CY2`):
/// u = j / rho to 2^-16, then 168 F_i = 24 rho + 56 c.j + 112 rho (c.u)^2
/// - 28 rho u^2 and 28 F_0 = 4 rho - 28 rho u^2, truncated.
#[inline(always)]
pub fn feq_fixed(r: i32, jx: i32, jy: i32) -> [i32; NCH] {
    let r = r.max(1);
    let ux = div16(jx, r);
    let uy = div16(jy, r);
    // 4 rho u^2 (ONE units).
    let p = mul_shr(jx, ux, 16).wrapping_add(3i32.wrapping_mul(mul_shr(jy, uy, 16)));
    std::array::from_fn(|i| {
        if i == 6 {
            (4i32.wrapping_mul(r).wrapping_sub(7i32.wrapping_mul(p))).wrapping_div(28)
        } else {
            // 4 c.j and 4 c.u 2^16; a = rho (c.u)^2.
            let d = (CX2[i] * jx).wrapping_add(3 * CY2[i] * jy);
            let cu = (CX2[i] * ux).wrapping_add(3 * CY2[i] * uy);
            let a = mul_shr(d, cu, 20);
            24i32
                .wrapping_mul(r)
                .wrapping_add(14i32.wrapping_mul(d))
                .wrapping_add(112i32.wrapping_mul(a))
                .wrapping_sub(7i32.wrapping_mul(p))
                .wrapping_div(168)
        }
    })
}

/// Stochastic rounding of x (ONE units) with a uniform word: floor(x/ONE +
/// U) with U the word's top 12 bits, unbiased for x on the 2^-12 grid.
#[inline(always)]
pub fn sround(x: i32, r: u32) -> i32 {
    x.wrapping_add((r >> (32 - FRAC)) as i32) >> FRAC
}

/// floor(r m / 2^32) for m < 2^15, as `mulhi` in the shader.
#[inline(always)]
pub fn mulhi(r: u32, m: u32) -> u32 {
    ((r >> 16) * m + (((r & 0xffff) * m) >> 16)) >> 16
}

/// Fixed-point block values (r, jx, jy) of a density and velocity.
pub fn fixed_values(rho: f64, ux: f64, uy: f64) -> [i32; 3] {
    let q = ONE as f64;
    [(rho * q).round() as i32, (2.0 * rho * ux * q).round() as i32, (2.0 / 3f64.sqrt() * rho * uy * q).round() as i32]
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Boundary {
    pub rho: f64,
    pub u: f64,
    pub inlet: usize,
    pub outlet: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub seed: u32,
    pub rule: Rule,
    /// Reference block side B, a power of two (divides W and H).
    pub block: usize,
    /// Re-estimate every N steps (0: never).
    pub every: u32,
    /// Reference occupation per channel at rest, in units of s.
    pub n0: f64,
    pub boundary: Option<Boundary>,
}

impl Params {
    pub fn rho0(&self) -> i32 {
        fixed_values(7.0 * self.n0, 0.0, 0.0)[0]
    }
}

/// Float block sums of the represented momentum and mass, and fluid sites.
#[derive(Clone, Debug, Default)]
pub struct BlockSums {
    pub jx: Vec<f64>,
    pub jy: Vec<f64>,
    pub mass: Vec<f64>,
    pub fluid: Vec<u32>,
}

impl BlockSums {
    pub fn add(&mut self, o: &BlockSums) {
        for (a, b) in [(&mut self.jx, &o.jx), (&mut self.jy, &o.jy), (&mut self.mass, &o.mass)] {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        for (x, y) in self.fluid.iter_mut().zip(&o.fluid) {
            *x += y;
        }
    }
}

/// The per-site reference (r, jx, jy) at (x, y) from block values `nodes`
/// (`nbx` x `nby` blocks of side `b`): bilinear between block centroids,
/// periodic, in integers (floored after each axis).
///
/// Positions are the sites' true ones (`measure::position`): odd rows sit
/// half a site to the right. A block's centroid is then a quarter site right
/// of its column centre, and in quarter sites the site's offset from block
/// 0's centroid is fx = 4x + 2 (y & 1) + 1 - 2B, weights out of 4B. Rows are
/// evenly spaced, so y is in half rows from the centre row, out of 2B.
#[inline(always)]
pub fn interp(nodes: &[[i32; 3]], nbx: usize, nby: usize, b: usize, x: usize, y: usize) -> [i32; 3] {
    let columns = COLUMNS.load(std::sync::atomic::Ordering::Relaxed);
    let bi = b as i32;
    let sy = 2 * bi;
    let ly = sy.trailing_zeros();
    let (sx, lx, fx) = if columns { (sy, ly, 2 * x as i32 + 1 - bi) } else { (4 * bi, ly + 1, 4 * x as i32 + 2 * (y & 1) as i32 + 1 - 2 * bi) };
    let fy = 2 * y as i32 + 1 - bi;
    let (bx0, by0) = (fx >> lx, fy >> ly);
    let (tx, ty) = (fx - (bx0 << lx), fy - (by0 << ly));
    let wrap = |v: i32, n: usize| v.rem_euclid(n as i32) as usize;
    let (x0, x1) = (wrap(bx0, nbx), wrap(bx0 + 1, nbx));
    let (y0, y1) = (wrap(by0, nby), wrap(by0 + 1, nby));
    std::array::from_fn(|c| {
        let row = |by: usize| ((sx - tx).wrapping_mul(nodes[by * nbx + x0][c]).wrapping_add(tx.wrapping_mul(nodes[by * nbx + x1][c]))) >> lx;
        ((sy - ty).wrapping_mul(row(y0)).wrapping_add(ty.wrapping_mul(row(y1)))) >> ly
    })
}

/// The initial deviation codes and block values for the field rho(x, y),
/// u(x, y) (positions of `measure::position`): block values from block
/// averages of the target, then per site and channel a deviation from the
/// half-filled ternary distribution whose mean is the target minus the
/// reference. Obstacle sites get OBST (state 0).
pub fn initial(
    w: usize,
    h: usize,
    params: &Params,
    obstacle: impl Fn(usize, usize) -> bool,
    field: impl Fn(f64, f64) -> (f64, f64, f64),
    seed: u32,
) -> (Vec<u16>, Vec<[i32; 3]>) {
    let b = params.block;
    let (nbx, nby) = (w / b, h / b);
    let block_of = |x: usize, y: usize| (y / b) * nbx + x / b;
    let mut acc = vec![(0.0f64, 0.0f64, 0.0f64, 0u32); nbx * nby];
    for y in 0..h {
        for x in 0..w {
            if obstacle(x, y) {
                continue;
            }
            let (px, py) = measure::position(x, y);
            let (r, ux, uy) = field(px, py);
            let a = &mut acc[block_of(x, y)];
            a.0 += r;
            a.1 += r * ux;
            a.2 += r * uy;
            a.3 += 1;
        }
    }
    let rest = fixed_values(7.0 * params.n0, 0.0, 0.0);
    let nodes: Vec<[i32; 3]> = acc
        .iter()
        .map(|a| if a.3 > 0 { let n = a.3 as f64; fixed_values(a.0 / n, a.1 / a.0, a.2 / a.0) } else { rest })
        .collect();
    let base = lowbias32(seed ^ 0x6a09_e667);
    let q = ONE as f64;
    let mut cells = vec![OBST; w * h];
    for y in 0..h {
        for x in 0..w {
            if obstacle(x, y) {
                continue;
            }
            let i = y * w + x;
            let (px, py) = measure::position(x, y);
            let (r, ux, uy) = field(px, py);
            let f = feq(r, ux, uy);
            let [nr, nx, ny] = interp(&nodes, nbx, nby, b, x, y);
            let phi = feq_fixed(nr, nx, ny);
            let mut code = 0u16;
            for c in 0..NCH {
                let mean = (f[c] - phi[c] as f64 / q).clamp(-1.0, 1.0);
                let p = crate::ternary::ternary_probs(1.0 + mean);
                let u = lowbias32(base ^ lowbias32((i as u32).wrapping_mul(8).wrapping_add(c as u32))) as f64 * (1.0 / 4294967296.0);
                let n = (u >= p[0]) as u16 + (u >= p[0] + p[1]) as u16;
                code += n * POW3[c];
            }
            cells[i] = code;
        }
    }
    (cells, nodes)
}

pub struct Signed {
    pub w: usize,
    pub h: usize,
    /// Base-3 code of delta + 1 per site, OBST for obstacles.
    pub cells: Vec<u16>,
    next: Vec<u16>,
    src: Vec<[u32; NDIR]>,
    /// Reference F per site and channel (ONE units), a function of `nodes`
    /// cached between re-estimates.
    pub phi: Vec<[i32; NCH]>,
    pub nbx: usize,
    pub nby: usize,
    /// Block values (r, jx, jy) the reference interpolates (ONE units).
    pub nodes: Vec<[i32; 3]>,
    /// Parked content per block and channel (ONE units).
    pub res: Vec<[i32; NCH]>,
    /// Fluid sites per block.
    pub fluid: Vec<u32>,
    delta: Vec<[i8; NCH]>,
    pub table: Table,
    pub params: Params,
    pub step: u32,
    /// Clipped units (saturation) and nonzero rounding kicks, cumulative.
    pub saturated: u64,
    pub kicks: u64,
}

impl Signed {
    /// A lattice of deviation codes on a uniform reference at rest,
    /// rho0 = 7 n0 everywhere.
    pub fn new(w: usize, h: usize, cells: Vec<u16>, params: Params) -> Self {
        let b = params.block;
        crate::check_shape(w, h).expect("lattice shape");
        assert!(b.is_power_of_two() && w.is_multiple_of(b) && h.is_multiple_of(b), "block must be a power of two dividing the lattice");
        assert_eq!(cells.len(), w * h);
        let mut src = vec![[0u32; NDIR]; w * h];
        for y in 0..h {
            for x in 0..w {
                for d in 0..NDIR {
                    let (sx, sy) = source(x, y, d, w, h);
                    src[y * w + x][d] = (sy * w + sx) as u32;
                }
            }
        }
        let (nbx, nby) = (w / b, h / b);
        let mut fluid = vec![0u32; nbx * nby];
        for (i, c) in cells.iter().enumerate() {
            if c & OBST == 0 {
                fluid[(i / w / b) * nbx + (i % w) / b] += 1;
            }
        }
        let delta = (0..NSTATE as u16).map(|s| counts(s).map(|n| n as i8 - 1)).collect();
        let rest = [params.rho0(), 0, 0];
        Signed {
            w,
            h,
            next: vec![0; w * h],
            cells,
            src,
            phi: vec![feq_fixed(rest[0], 0, 0); w * h],
            nbx,
            nby,
            nodes: vec![rest; nbx * nby],
            res: vec![[0; NCH]; nbx * nby],
            fluid,
            delta,
            table: Table::new(),
            params,
            step: 0,
            saturated: 0,
            kicks: 0,
        }
    }

    /// A lattice sampled around the field rho(x, y), u(x, y) (`initial`).
    pub fn sample(
        w: usize,
        h: usize,
        params: Params,
        obstacle: impl Fn(usize, usize) -> bool,
        field: impl Fn(f64, f64) -> (f64, f64, f64),
        seed: u32,
    ) -> Self {
        let (cells, nodes) = initial(w, h, &params, obstacle, field, seed);
        Self::from_state(w, h, cells, nodes, params)
    }

    /// A lattice with these cells and block values, empty reservoirs.
    pub fn from_state(w: usize, h: usize, cells: Vec<u16>, nodes: Vec<[i32; 3]>, params: Params) -> Self {
        let mut s = Signed::new(w, h, cells, params);
        s.nodes = nodes;
        s.phi = s.interpolated();
        s
    }

    #[inline(always)]
    pub fn block_of(&self, i: usize) -> usize {
        let lb = self.params.block.trailing_zeros();
        ((i / self.w) >> lb) * self.nbx + ((i % self.w) >> lb)
    }

    /// Per block: the reservoir spread over its fluid sites, truncated (zero
    /// for blocks without fluid).
    fn shares(&self) -> Vec<[i32; NCH]> {
        self.res.iter().zip(&self.fluid).map(|(r, &n)| if n == 0 { [0; NCH] } else { r.map(|v| v.wrapping_div(n as i32)) }).collect()
    }

    pub fn rho0(&self) -> f64 {
        7.0 * self.params.n0
    }

    /// Deviations of site i (zero for obstacle sites' state bits too).
    pub fn deltas(&self, i: usize) -> [i8; NCH] {
        self.delta[(self.cells[i] & STATE) as usize]
    }

    /// Per-site reference from the block values; obstacle sites get the
    /// rest reference.
    pub fn interpolated(&self) -> Vec<[i32; NCH]> {
        let rest = feq_fixed(self.params.rho0(), 0, 0);
        let mut out = vec![rest; self.w * self.h];
        let rows: Vec<(usize, &mut [[i32; NCH]])> = out.chunks_mut(self.w).enumerate().collect();
        let work = |(y, row): (usize, &mut [[i32; NCH]])| {
            for (x, o) in row.iter_mut().enumerate() {
                if self.cells[y * self.w + x] & OBST != 0 {
                    continue;
                }
                let [r, jx, jy] = interp(&self.nodes, self.nbx, self.nby, self.params.block, x, y);
                *o = feq_fixed(r, jx, jy);
            }
        };
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            rows.into_par_iter().for_each(work);
        }
        #[cfg(not(feature = "threads"))]
        rows.into_iter().for_each(work);
        out
    }

    /// Stream (with the reference correction) and collide one row; returns
    /// the row's reservoir additions per block column, saturations, kicks.
    fn row(&self, y: usize, out: &mut [u16]) -> (Vec<[i32; NCH]>, u64, u64) {
        let lb = self.params.block.trailing_zeros();
        let mut add = vec![[0i32; NCH]; self.nbx];
        let (mut sat, mut kicks) = (0u64, 0u64);
        let kc = step_key(self.params.seed, self.step, STREAM_COLLIDE);
        let ks = step_key(self.params.seed, self.step, STREAM_ROUND);
        let rule = self.params.rule;
        let w32 = (self.w / 32) as u32;
        let mut chir = 0u32;
        for (x, o) in out.iter_mut().enumerate() {
            let i = y * self.w + x;
            let own = self.cells[i];
            let mut s = (self.delta[(own & STATE) as usize][6] + 1) as u16 * POW3[6];
            let src = &self.src[i];
            let a = &mut add[x >> lb];
            for d in 0..NDIR {
                let j = src[d] as usize;
                let dl = self.delta[(self.cells[j] & STATE) as usize][d] as i32;
                let r = self.phi[j][d].wrapping_sub(self.phi[i][d]);
                let v = if r == 0 {
                    dl
                } else {
                    let k = sround(r, lowbias32(ks ^ (i as u32).wrapping_mul(8).wrapping_add(d as u32).wrapping_mul(0x9e37_79b9)));
                    kicks += (k != 0) as u64;
                    a[d] = a[d].wrapping_add(r.wrapping_sub(k.wrapping_mul(ONE)));
                    dl + k
                };
                let c = v.clamp(-1, 1);
                if c != v {
                    a[d] = a[d].wrapping_add((v - c).wrapping_mul(ONE));
                    sat += (v - c).unsigned_abs() as u64;
                }
                s += (c + 1) as u16 * POW3[d];
            }
            let rnd = if rule == Rule::Chiral {
                if x % 32 == 0 {
                    chir = crate::chunk_rand(kc, y as u32 * w32 + (x / 32) as u32);
                }
                ((chir >> (x % 32)) & 1) << 31
            } else {
                lowbias32(kc ^ (i as u32).wrapping_mul(0x9e37_79b9))
            };
            *o = if own & OBST != 0 { OBST | crate::ternary::bounce(s) } else { self.table.collide(s, rnd, rule) };
        }
        (add, sat, kicks)
    }

    /// Whether step `t` ends with a re-estimate, and whether it is an
    /// open-boundary refresh.
    pub fn reestimates(params: &Params, t: u32) -> (bool, bool) {
        let refresh = params.boundary.is_some() && t.is_multiple_of(REFRESH);
        let ev = params.every;
        (refresh || (ev > 0 && (t + 1).is_multiple_of(ev)), refresh)
    }

    pub fn step(&mut self) {
        let mut next = std::mem::take(&mut self.next);
        let w = self.w;
        #[cfg(feature = "threads")]
        let rows: Vec<_> = {
            use rayon::prelude::*;
            next.par_chunks_mut(w).enumerate().map(|(y, out)| self.row(y, out)).collect()
        };
        #[cfg(not(feature = "threads"))]
        let rows: Vec<_> = next.chunks_mut(w).enumerate().map(|(y, out)| self.row(y, out)).collect();
        let b = self.params.block;
        for (y, (add, sat, kicks)) in rows.into_iter().enumerate() {
            let k0 = (y / b) * self.nbx;
            for (bx, a) in add.iter().enumerate() {
                for c in 0..NCH {
                    self.res[k0 + bx][c] = self.res[k0 + bx][c].wrapping_add(a[c]);
                }
            }
            self.saturated += sat;
            self.kicks += kicks;
        }
        self.next = std::mem::replace(&mut self.cells, next);
        let (re, refresh) = Self::reestimates(&self.params, self.step);
        if re {
            self.reestimate(refresh);
        }
        self.step += 1;
    }

    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Per block and channel: sum over fluid sites of phi + delta, plus the
    /// reservoir (ONE units).
    pub fn block_content(&self) -> Vec<[i32; NCH]> {
        let mut c = self.fluid_sums();
        for (a, r) in c.iter_mut().zip(&self.res) {
            for ch in 0..NCH {
                a[ch] = a[ch].wrapping_add(r[ch]);
            }
        }
        c
    }

    /// Per block and channel: sum over fluid sites of phi + delta.
    fn fluid_sums(&self) -> Vec<[i32; NCH]> {
        let b = self.params.block;
        let lb = b.trailing_zeros();
        let work = |by: usize| {
            let mut v = vec![[0i32; NCH]; self.nbx];
            for y in by * b..(by + 1) * b {
                for x in 0..self.w {
                    let i = y * self.w + x;
                    if self.cells[i] & OBST != 0 {
                        continue;
                    }
                    let d = self.deltas(i);
                    let a = &mut v[x >> lb];
                    for ch in 0..NCH {
                        a[ch] = a[ch].wrapping_add(self.phi[i][ch].wrapping_add(d[ch] as i32 * ONE));
                    }
                }
            }
            v
        };
        #[cfg(feature = "threads")]
        let rows: Vec<Vec<[i32; NCH]>> = {
            use rayon::prelude::*;
            (0..self.nby).into_par_iter().map(work).collect()
        };
        #[cfg(not(feature = "threads"))]
        let rows: Vec<Vec<[i32; NCH]>> = (0..self.nby).map(work).collect();
        rows.concat()
    }

    /// Site i's represented f per channel in ONE units, plus `share`, its
    /// block's reservoir spread over the block's fluid sites (`shares`).
    #[inline(always)]
    fn site_fixed(&self, i: usize, share: &[i32; NCH]) -> [i32; NCH] {
        let d = self.deltas(i);
        std::array::from_fn(|ch| self.phi[i][ch].wrapping_add(d[ch] as i32 * ONE).wrapping_add(share[ch]))
    }

    /// Block sums of the represented mass and momentum over `bw` x `bh`
    /// blocks (the reservoir spread over its reference block's fluid sites).
    pub fn block_sums(&self, bw: usize, bh: usize) -> BlockSums {
        let (nx, ny) = (self.w / bw, self.h / bh);
        let q = ONE as f64;
        let work = |by: usize| {
            let mut v = vec![(0.0f64, 0.0f64, 0.0f64, 0u32); nx];
            for y in by * bh..(by + 1) * bh {
                for x in 0..self.w {
                    let i = y * self.w + x;
                    if self.cells[i] & OBST != 0 {
                        continue;
                    }
                    let k = self.block_of(i);
                    let d = self.deltas(i);
                    let nf = self.fluid[k] as f64;
                    let a = &mut v[x / bw];
                    for ch in 0..NCH {
                        let f = (self.phi[i][ch] as f64 + self.res[k][ch] as f64 / nf) / q + d[ch] as f64;
                        let c = velocity(ch);
                        a.0 += c.0 * f;
                        a.1 += c.1 * f;
                        a.2 += f;
                    }
                    a.3 += 1;
                }
            }
            v
        };
        #[cfg(feature = "threads")]
        let rows: Vec<_> = {
            use rayon::prelude::*;
            (0..ny).into_par_iter().map(work).collect()
        };
        #[cfg(not(feature = "threads"))]
        let rows: Vec<_> = (0..ny).map(work).collect();
        let all = rows.concat();
        BlockSums {
            jx: all.iter().map(|a| a.0).collect(),
            jy: all.iter().map(|a| a.1).collect(),
            mass: all.iter().map(|a| a.2).collect(),
            fluid: all.iter().map(|a| a.3).collect(),
        }
    }

    /// Integer block sums for the page (`coarse::BlockSums`, momentum in
    /// units 2 px, 2 py / sqrt 3), each block's sum in ONE units shifted
    /// down by `SUM_SHIFT`; the GPU engine's `coarse` gives the same.
    pub fn page_sums(&self, bw: usize, bh: usize) -> crate::coarse::BlockSums {
        let (nx, ny) = (self.w / bw, self.h / bh);
        let mut s = crate::coarse::BlockSums { nx, ny, bw, bh, jx: vec![0; nx * ny], jy: vec![0; nx * ny], mass: vec![0; nx * ny], fluid: vec![0; nx * ny] };
        let shares = self.shares();
        let lb = self.params.block.trailing_zeros();
        for by in 0..ny {
            for bx in 0..nx {
                let (mut m, mut jx, mut jy, mut n) = (0i32, 0i32, 0i32, 0i32);
                for y in by * bh..(by + 1) * bh {
                    for x in bx * bw..(bx + 1) * bw {
                        let i = y * self.w + x;
                        if self.cells[i] & OBST != 0 {
                            continue;
                        }
                        let f = self.site_fixed(i, &shares[(y >> lb) * self.nbx + (x >> lb)]);
                        for ch in 0..NCH {
                            m = m.wrapping_add(f[ch]);
                            if ch < NDIR {
                                jx = jx.wrapping_add(CX2[ch].wrapping_mul(f[ch]));
                                jy = jy.wrapping_add(CY2[ch].wrapping_mul(f[ch]));
                            }
                        }
                        n += 1;
                    }
                }
                let b = by * nx + bx;
                (s.jx[b], s.jy[b], s.mass[b], s.fluid[b]) = (jx >> SUM_SHIFT, jy >> SUM_SHIFT, m >> SUM_SHIFT, n);
            }
        }
        s
    }

    fn in_zone(&self, x: usize) -> bool {
        match self.params.boundary {
            Some(bd) => x < bd.inlet || x >= self.w - bd.outlet,
            None => false,
        }
    }

    /// New block values from block sums, new per-site reference,
    /// re-quantized deviations, and the exact remainder in the reservoir.
    pub fn reestimate(&mut self, refresh: bool) {
        let b = self.params.block;
        let content = self.block_content();
        let inflow = self.params.boundary.map(|bd| fixed_values(bd.rho, bd.u, 0.0));
        let rhoonly = RHOONLY.load(std::sync::atomic::Ordering::Relaxed);
        for k in 0..self.nbx * self.nby {
            if self.fluid[k] == 0 {
                continue;
            }
            let n = self.fluid[k] as i32;
            let (mut m, mut jx, mut jy) = (0i32, 0i32, 0i32);
            for ch in 0..NCH {
                m = m.wrapping_add(content[k][ch]);
                if ch < NDIR {
                    jx = jx.wrapping_add(CX2[ch].wrapping_mul(content[k][ch]));
                    jy = jy.wrapping_add(CY2[ch].wrapping_mul(content[k][ch]));
                }
            }
            self.nodes[k] = [m.wrapping_div(n), jx.wrapping_div(n), jy.wrapping_div(n)];
            if rhoonly {
                self.nodes[k][1] = 0;
                self.nodes[k][2] = 0;
            }
            if refresh && self.in_zone((k % self.nbx) * b) {
                self.nodes[k] = inflow.unwrap();
            }
        }
        let new = self.interpolated();
        let kq = step_key(self.params.seed, self.step, STREAM_REQUANT);
        let kr = step_key(self.params.seed, self.step, STREAM_REFRESH);
        let nokick = NOKICK.load(std::sync::atomic::Ordering::Relaxed);
        let shares = self.shares();
        let lb = b.trailing_zeros();
        let mut cells = std::mem::take(&mut self.cells);
        let w = self.w;
        let this = &*self;
        let work = |(y, row): (usize, &mut [u16])| {
            let (mut sat, mut kicks) = (0u64, 0u64);
            for (x, cell) in row.iter_mut().enumerate() {
                if *cell & OBST != 0 {
                    continue;
                }
                let i = y * w + x;
                let mut code = 0u16;
                if refresh && this.in_zone(x) {
                    for ch in 0..NCH {
                        let r = lowbias32(kr ^ (i as u32).wrapping_mul(8).wrapping_add(ch as u32).wrapping_mul(0x9e37_79b9));
                        code += mulhi(r, 3) as u16 * POW3[ch];
                    }
                } else {
                    let d = this.delta[*cell as usize];
                    let share = &shares[(y >> lb) * this.nbx + (x >> lb)];
                    for ch in 0..NCH {
                        // The parked content is released into the field.
                        let dphi = this.phi[i][ch].wrapping_add(share[ch]).wrapping_sub(new[i][ch]);
                        let k = if dphi == 0 || nokick {
                            0
                        } else {
                            sround(dphi, lowbias32(kq ^ (i as u32).wrapping_mul(8).wrapping_add(ch as u32).wrapping_mul(0x9e37_79b9)))
                        };
                        kicks += (k != 0) as u64;
                        let v = d[ch] as i32 + k;
                        let c = v.clamp(-1, 1);
                        sat += (v - c).unsigned_abs() as u64;
                        code += (c + 1) as u16 * POW3[ch];
                    }
                }
                *cell = code;
            }
            (sat, kicks)
        };
        #[cfg(feature = "threads")]
        let counts: Vec<(u64, u64)> = {
            use rayon::prelude::*;
            cells.par_chunks_mut(w).enumerate().map(work).collect()
        };
        #[cfg(not(feature = "threads"))]
        let counts: Vec<(u64, u64)> = cells.chunks_mut(w).enumerate().map(work).collect();
        self.cells = cells;
        for (s, k) in counts {
            self.saturated += s;
            self.kicks += k;
        }
        self.phi = new;
        // The reservoir takes each block's exact remainder.
        let after = self.fluid_sums();
        for k in 0..self.nbx * self.nby {
            if self.fluid[k] == 0 {
                continue;
            }
            let zone = refresh && self.in_zone((k % self.nbx) * b);
            for ch in 0..NCH {
                self.res[k][ch] = if zone { 0 } else { content[k][ch].wrapping_sub(after[k][ch]) };
            }
        }
    }

    /// Total represented content per channel (all sites and reservoirs),
    /// exact, in ONE units.
    pub fn channel_totals_fixed(&self) -> [i64; NCH] {
        let mut t = [0i64; NCH];
        for i in 0..self.w * self.h {
            let d = self.deltas(i);
            for ch in 0..NCH {
                t[ch] += self.phi[i][ch] as i64 + d[ch] as i64 * ONE as i64;
            }
        }
        for r in &self.res {
            for ch in 0..NCH {
                t[ch] += r[ch] as i64;
            }
        }
        t
    }

    /// Total mass and momentum in ONE units and the integer momentum units
    /// (2 px, 2 py / sqrt 3), exact.
    pub fn totals_fixed(&self) -> (i64, i64, i64) {
        let t = self.channel_totals_fixed();
        let m = t.iter().sum();
        let jx = (0..NDIR).map(|c| CX2[c] as i64 * t[c]).sum();
        let jy = (0..NDIR).map(|c| CY2[c] as i64 * t[c]).sum();
        (m, jx, jy)
    }

    /// Total represented content per channel (all sites and reservoirs).
    pub fn channel_totals(&self) -> [f64; NCH] {
        self.channel_totals_fixed().map(|v| v as f64 / ONE as f64)
    }

    /// Total represented mass and momentum.
    pub fn totals(&self) -> (f64, f64, f64) {
        let t = self.channel_totals();
        let (mut m, mut jx, mut jy) = (0.0, 0.0, 0.0);
        for ch in 0..NCH {
            let c = velocity(ch);
            m += t[ch];
            jx += c.0 * t[ch];
            jy += c.1 * t[ch];
        }
        (m, jx, jy)
    }

    /// Per-site represented mass and momentum (physical velocity units),
    /// the reservoir spread over its block's fluid sites; obstacle sites 0.
    pub fn fields(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.w * self.h;
        let q = ONE as f64;
        let (mut m, mut jx, mut jy) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            if self.cells[i] & OBST != 0 {
                continue;
            }
            let k = self.block_of(i);
            let d = self.deltas(i);
            let nf = self.fluid[k] as f64;
            for ch in 0..NCH {
                let f = (self.phi[i][ch] as f64 + self.res[k][ch] as f64 / nf) / q + d[ch] as f64;
                let c = velocity(ch);
                m[i] += f;
                jx[i] += c.0 * f;
                jy[i] += c.1 * f;
            }
        }
        (m, jx, jy)
    }
}
