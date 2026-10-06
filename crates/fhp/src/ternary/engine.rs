//! Per-site ternary engine: one u16 per site (the base-3 state, bit 15 for
//! an obstacle), the same lattice, streaming and step order as the FHP-III
//! engines: state <- boundary(collide(stream(state))).

use super::{bounce, counts, ternary_probs, Rule, Table, NCH, NSTATE, POW3};
use crate::{lowbias32, source, step_key, NDIR};

pub const OBST: u16 = 1 << 15;
const STATE: u16 = OBST - 1;

/// Open boundary in x, as the FHP-III `randomize` outlet: every `EVERY`
/// steps the first 32 and the last 64 columns get fresh equilibrium samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Boundary {
    /// Per channel, thresholds out of 2^32 for n < 1 and n < 2.
    pub cdf: [[u32; 2]; NCH],
    pub inlet: usize,
    pub outlet: usize,
}

impl Boundary {
    pub fn new(d: f64, u: f64) -> Self {
        Boundary {
            cdf: cdf(&super::equilibrium(2, 2, d, u, 0.0)),
            inlet: 32,
            outlet: 64,
        }
    }
}

pub fn cdf(mean: &[f64; NCH]) -> [[u32; 2]; NCH] {
    std::array::from_fn(|i| {
        let p = ternary_probs(mean[i]);
        let t = |x: f64| (x * 4294967296.0).round().clamp(0.0, 4294967295.0) as u32;
        [t(p[0]), t(p[0] + p[1])]
    })
}

/// Sample a state from per-channel thresholds with seven random words.
#[inline]
pub fn sample(cdf: &[[u32; 2]; NCH], r: impl Fn(usize) -> u32) -> u16 {
    let mut s = 0u16;
    for i in 0..NCH {
        let x = r(i);
        let n = (x >= cdf[i][0]) as u16 + (x >= cdf[i][1]) as u16;
        s += n * POW3[i];
    }
    s
}

pub const EVERY: u32 = 32;

/// The fresh state of boundary site `i` on a refresh with key `kb`.
#[inline]
pub fn boundary_state(b: &Boundary, kb: u32, i: u32) -> u16 {
    sample(&b.cdf, |c| lowbias32(kb ^ lowbias32(i.wrapping_mul(8).wrapping_add(c as u32))))
}
pub const STREAM_COLLIDE: u32 = 0x7e01;
pub const STREAM_BOUNDARY: u32 = 0x7e02;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub seed: u32,
    pub rule: Rule,
    pub boundary: Option<Boundary>,
}

pub struct Ternary {
    pub w: usize,
    pub h: usize,
    pub cells: Vec<u16>,
    next: Vec<u16>,
    src: Vec<[u32; NDIR]>,
    /// digit[i][s] = n_i(s) 3^i, for streaming.
    digit: Vec<[u16; NCH]>,
    bounce: Vec<u16>,
    pub table: Table,
    pub params: Params,
    pub step: u32,
}

impl Ternary {
    pub fn new(w: usize, h: usize, cells: Vec<u16>, params: Params) -> Self {
        crate::check_shape(w, h).expect("lattice shape");
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
        let digit = (0..NSTATE as u16).map(|s| {
            let n = counts(s);
            std::array::from_fn(|i| n[i] as u16 * POW3[i])
        });
        let bounce = (0..NSTATE as u16).map(bounce).collect();
        Ternary {
            w,
            h,
            next: vec![0; w * h],
            cells,
            src,
            digit: digit.collect(),
            bounce,
            table: Table::new(),
            params,
            step: 0,
        }
    }

    /// The streamed (pre-collision) state of site i.
    #[inline(always)]
    fn gather(&self, i: usize) -> u16 {
        let c = self.cells[i];
        let mut s = self.digit[(c & STATE) as usize][6];
        let src = &self.src[i];
        for d in 0..NDIR {
            s += self.digit[(self.cells[src[d] as usize] & STATE) as usize][d];
        }
        s | (c & OBST)
    }

    pub fn stream_only(&self) -> Vec<u16> {
        (0..self.w * self.h).map(|i| self.gather(i)).collect()
    }

    fn row(&self, y: usize, out: &mut [u16]) {
        let kc = step_key(self.params.seed, self.step, STREAM_COLLIDE);
        let rule = self.params.rule;
        let w32 = (self.w / 32) as u32;
        let mut chir = 0u32;
        for (x, o) in out.iter_mut().enumerate() {
            let i = y * self.w + x;
            let s = self.gather(i);
            // Rule Chiral takes one bit per site from the 32-site chunk's
            // word, as FHP-III, so the bit-sliced engines can match it.
            let r = if rule == Rule::Chiral {
                if x % 32 == 0 {
                    chir = crate::chunk_rand(kc, y as u32 * w32 + (x / 32) as u32);
                }
                ((chir >> (x % 32)) & 1) << 31
            } else {
                lowbias32(kc ^ (i as u32).wrapping_mul(0x9e37_79b9))
            };
            *o = if s & OBST != 0 {
                OBST | self.bounce[(s & STATE) as usize]
            } else {
                self.table.collide(s, r, rule)
            };
        }
        if let (Some(b), true) = (self.params.boundary, self.step.is_multiple_of(EVERY)) {
            let kb = step_key(self.params.seed, self.step, STREAM_BOUNDARY);
            for x in (0..b.inlet).chain(self.w - b.outlet..self.w) {
                if out[x] & OBST != 0 {
                    continue;
                }
                out[x] = boundary_state(&b, kb, (y * self.w + x) as u32);
            }
        }
    }

    pub fn step(&mut self) {
        let mut next = std::mem::take(&mut self.next);
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            next.par_chunks_mut(self.w)
                .enumerate()
                .for_each(|(y, out)| self.row(y, out));
        }
        #[cfg(not(feature = "threads"))]
        for (y, out) in next.chunks_mut(self.w).enumerate() {
            self.row(y, out);
        }
        self.next = std::mem::replace(&mut self.cells, next);
        self.step += 1;
    }

    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    pub fn mass(&self) -> u64 {
        self.cells
            .iter()
            .map(|&c| super::mass(c & STATE) as u64)
            .sum()
    }

    /// Total momentum in integer units (2 px, 2 py / sqrt 3), obstacles included.
    pub fn momentum(&self) -> (i64, i64) {
        self.cells.iter().fold((0, 0), |a, &c| {
            let p = super::momentum(c & STATE);
            (a.0 + p.0 as i64, a.1 + p.1 as i64)
        })
    }

    /// Block sums of momentum, mass and fluid sites, as `coarse::block_sums`.
    pub fn block_sums(&self, bw: usize, bh: usize) -> crate::coarse::BlockSums {
        let (nx, ny) = (self.w / bw, self.h / bh);
        let mut s = crate::coarse::BlockSums {
            nx,
            ny,
            bw,
            bh,
            jx: vec![0; nx * ny],
            jy: vec![0; nx * ny],
            mass: vec![0; nx * ny],
            fluid: vec![0; nx * ny],
        };
        let moments: Vec<(i32, i32, i32)> = (0..NSTATE as u16)
            .map(|c| {
                let p = super::momentum(c);
                (p.0, p.1, super::mass(c) as i32)
            })
            .collect();
        for y in 0..self.h {
            for x in 0..self.w {
                let c = self.cells[y * self.w + x];
                let b = (y / bh) * nx + x / bw;
                if c & OBST != 0 {
                    continue;
                }
                let m = moments[c as usize];
                s.jx[b] += m.0;
                s.jy[b] += m.1;
                s.mass[b] += m.2;
                s.fluid[b] += 1;
            }
        }
        s
    }
}

/// A state sampled from the equilibrium of capacity 2 at density `d` and
/// velocity (ux, uy), from a counter-based stream.
pub fn sample_equilibrium(mean: &[f64; NCH], seed: u32, site: u32) -> u16 {
    let c = cdf(mean);
    let base = lowbias32(seed ^ 0x2545_f491);
    sample(&c, |k| {
        lowbias32(base ^ lowbias32(site.wrapping_mul(8).wrapping_add(k as u32)))
    })
}
