//! Random-walk lattice gas for Laplace (the diffusion automaton of Chopard
//! and Droz, "Cellular Automata Modeling of Physical Systems", 1998, ch. 4,
//! with charges). Each site has 4 channels, one per lattice direction
//! (0: +j east, 1: +i south, 2: -j west, 3: -i north); a channel is empty,
//! holds a particle (+1) or an antiparticle (-1). One step:
//!
//! 1. stream: the particle in channel d moves to the neighbour in direction d;
//! 2. at interior sites, optionally annihilate head-on pairs: channels 0 and
//!    2 (and 1 and 3) holding opposite charges are both emptied;
//! 3. at interior sites, rotate all 4 channels by r in {0, 1, 2, 3}, uniform
//!    per site per step (channel c moves to channel c + r mod 4), so every
//!    particle's next hop is to a uniformly random neighbour;
//! 4. boundary sites are reservoirs: whatever arrived is absorbed, and every
//!    channel is re-seeded, + with probability v_b and - otherwise, so the
//!    mean charge per channel is u_b = 2 v_b - 1.
//!
//! The expected charge obeys the Jacobi recursion (each particle's hop is
//! uniform over the 4 neighbours, and the rotation is drawn independently of
//! the state), so its steady state is the harmonic function with boundary
//! values u_b (Kakutani's random-walk representation). The estimate is the
//! charge averaged over a time window and over 8 x 8 blocks.
//!
//! Site state: bits 0..3 the + channels, bits 4..7 the - channels.

use crate::rng::{chunk_rand, site_bit, step_key};

pub const STREAM_ROT0: u32 = 20;
pub const STREAM_ROT1: u32 = 21;
pub const STREAM_BOUNDARY: u32 = 22;
/// Accumulators sum the charge over 8 consecutive columns of a row.
pub const GROUP: usize = 8;
/// Estimates are 8 x 8 block means.
pub const BLOCK: usize = 8;

pub fn keys(seed: u32, step: u32) -> [u32; 3] {
    [step_key(seed, step, STREAM_ROT0), step_key(seed, step, STREAM_ROT1), step_key(seed, step, STREAM_BOUNDARY)]
}

/// P(+) of a boundary channel as a 32-bit threshold: + when the hash is below.
pub fn threshold(v: f64) -> u32 {
    (v.clamp(0.0, 1.0) * 4294967296.0).round().min(4294967295.0) as u32
}

/// Thresholds of the boundary sites, laid out as top row, bottom row, left
/// column, right column (n each; corners appear twice, the row entry is used).
pub fn thresholds(p: &crate::Problem) -> Vec<u32> {
    let n = p.n;
    let mut t = Vec::with_capacity(4 * n);
    t.extend((0..n).map(|j| threshold(p.analytic(0, j))));
    t.extend((0..n).map(|j| threshold(p.analytic(n - 1, j))));
    t.extend((0..n).map(|i| threshold(p.analytic(i, 0))));
    t.extend((0..n).map(|i| threshold(p.analytic(i, n - 1))));
    t
}

pub fn boundary_threshold(th: &[u32], n: usize, i: usize, j: usize) -> u32 {
    if i == 0 {
        th[j]
    } else if i == n - 1 {
        th[n + j]
    } else if j == 0 {
        th[2 * n + i]
    } else {
        th[3 * n + i]
    }
}

/// The re-seeded state of boundary site (i, j).
#[inline(always)]
pub fn reseed(key: u32, th: u32, n: usize, i: usize, j: usize) -> u8 {
    let site = (i * n + j) as u32;
    let mut s = 0u8;
    for c in 0..4 {
        let h = chunk_rand(key, site.wrapping_mul(4).wrapping_add(c));
        s |= if h < th { 1 << c } else { 16 << c };
    }
    s
}

/// Head-on annihilation of one arrived site state.
#[inline(always)]
pub fn annihilate(s: u8) -> u8 {
    let (p, m) = (s & 15, s >> 4);
    // Opposite channel of each channel: rotate the 4 bits by 2.
    let po = ((p << 2) | (p >> 2)) & 15;
    let kill = (p & ((m << 2) | (m >> 2))) | (m & po);
    let kill = kill & 15;
    (p & !kill) | ((m & !kill) << 4)
}

#[inline(always)]
pub fn rotate(s: u8, r: u32) -> u8 {
    let rot = |x: u8| ((x << r) | (x >> (4 - r))) & 15;
    if r == 0 {
        s
    } else {
        rot(s & 15) | rot(s >> 4) << 4
    }
}

pub fn charge(s: u8) -> i32 {
    (s & 15).count_ones() as i32 - (s >> 4).count_ones() as i32
}

/// Initial state: boundary empty, every interior channel holding a - (v = 0,
/// like the other panels' zero interior).
pub fn initial(n: usize) -> Vec<u8> {
    (0..n * n).map(|k| if k / n == 0 || k / n == n - 1 || k % n == 0 || k % n == n - 1 { 0 } else { 0xf0 }).collect()
}

/// One step; adds the interior charges to `acc` (one entry per 8 columns of
/// a row, `acc[i * n / 8 + j / 8]`).
#[allow(clippy::too_many_arguments)]
pub fn step(n: usize, src: &[u8], dst: &mut [u8], th: &[u32], acc: &mut [i32], seed: u32, step: u32, annihilation: bool) {
    let k = keys(seed, step);
    for i in 0..n {
        for j in 0..n {
            let c = i * n + j;
            if i == 0 || j == 0 || i == n - 1 || j == n - 1 {
                dst[c] = reseed(k[2], boundary_threshold(th, n, i, j), n, i, j);
                continue;
            }
            let arrived = (src[c - 1] & 0x11) | (src[c - n] & 0x22) | (src[c + 1] & 0x44) | (src[c + n] & 0x88);
            let a = if annihilation { annihilate(arrived) } else { arrived };
            let r = site_bit(k[0], n, i, j) | site_bit(k[1], n, i, j) << 1;
            let s = rotate(a, r);
            dst[c] = s;
            acc[i * (n / GROUP) + j / GROUP] = acc[i * (n / GROUP) + j / GROUP].wrapping_add(charge(s));
        }
    }
}

/// Windowed block estimate from cumulative accumulators: keeps snapshots
/// (step, acc) at geometric spacing and differences the current one with
/// the latest snapshot at or before half the current step.
pub struct Estimator {
    pub n: usize,
    snaps: Vec<(u64, Vec<i32>)>,
}

impl Estimator {
    pub fn new(n: usize) -> Self {
        Estimator { n, snaps: vec![(0, vec![0; n * n / GROUP])] }
    }

    pub fn reset(&mut self) {
        *self = Estimator::new(self.n);
    }

    /// Record a reading; returns v per 8 x 8 block (row-major, n/8 a side)
    /// over the window, or None before the first step.
    pub fn update(&mut self, step: u64, acc: &[i32]) -> Option<(Vec<f64>, u64)> {
        let half = step / 2;
        let base = self.snaps.iter().rev().find(|s| s.0 <= half).unwrap_or(&self.snaps[0]);
        let (t0, a0) = (base.0, &base.1);
        let out = if step > t0 { Some((block_estimate(self.n, acc, a0, step - t0), step - t0)) } else { None };
        if step as f64 >= 1.1 * self.snaps.last().unwrap().0 as f64 && step > self.snaps.last().unwrap().0 {
            self.snaps.push((step, acc.to_vec()));
        }
        out
    }
}

/// v per block from accumulator differences over `steps` steps; interior
/// sites only (boundary rows and columns are not accumulated).
pub fn block_estimate(n: usize, acc: &[i32], base: &[i32], steps: u64) -> Vec<f64> {
    let nb = n / BLOCK;
    let gpr = n / GROUP;
    let mut out = vec![0.0; nb * nb];
    for bi in 0..nb {
        for bj in 0..nb {
            let mut q = 0i64;
            let mut sites = 0i64;
            for i in bi * BLOCK..(bi + 1) * BLOCK {
                if i == 0 || i == n - 1 {
                    continue;
                }
                let g = i * gpr + bj;
                q += acc[g].wrapping_sub(base[g]) as i64;
                let lo = (bj * BLOCK).max(1);
                let hi = ((bj + 1) * BLOCK).min(n - 1);
                sites += (hi - lo) as i64;
            }
            let u = q as f64 / (4.0 * sites as f64 * steps as f64);
            out[bi * nb + bj] = (u + 1.0) / 2.0;
        }
    }
    out
}

/// Max over blocks of |block estimate - block mean of the analytic v| (both
/// over the block's interior sites).
pub fn block_error(p: &crate::Problem, est: &[f64]) -> f64 {
    let n = p.n;
    let nb = n / BLOCK;
    let mut m = 0f64;
    for bi in 0..nb {
        for bj in 0..nb {
            let (mut s, mut c) = (0.0, 0.0);
            for i in (bi * BLOCK).max(1)..((bi + 1) * BLOCK).min(n - 1) {
                for j in (bj * BLOCK).max(1)..((bj + 1) * BLOCK).min(n - 1) {
                    s += p.analytic(i, j);
                    c += 1.0;
                }
            }
            m = m.max((est[bi * nb + bj] - s / c).abs());
        }
    }
    m
}

/// Per-site reference engine.
pub struct RefEngine {
    pub n: usize,
    pub cur: Vec<u8>,
    other: Vec<u8>,
    pub th: Vec<u32>,
    pub acc: Vec<i32>,
    pub seed: u32,
    pub annihilation: bool,
    pub iter: u64,
}

impl RefEngine {
    pub fn new(p: &crate::Problem, seed: u32, annihilation: bool) -> Self {
        let cur = initial(p.n);
        RefEngine { n: p.n, other: cur.clone(), cur, th: thresholds(p), acc: vec![0; p.n * p.n / GROUP], seed, annihilation, iter: 0 }
    }

    pub fn run(&mut self, iters: usize) {
        for _ in 0..iters {
            step(self.n, &self.cur, &mut self.other, &self.th, &mut self.acc, self.seed, self.iter as u32, self.annihilation);
            std::mem::swap(&mut self.cur, &mut self.other);
            self.iter += 1;
        }
    }
}

/// Inputs: arrived + channels 0..4, - channels 4..8, rotation bits 8 (low)
/// and 9. Outputs: + channels 0..4, - channels 4..8 after annihilation (if
/// on) and rotation.
pub fn circuit(annihilation: bool) -> crate::circuit::Circuit {
    use crate::circuit::{Builder, Circuit, Sig};
    let mut g = Builder::default();
    let mut p: Vec<Sig> = (0..4).map(Sig::In).collect();
    let mut m: Vec<Sig> = (4..8).map(Sig::In).collect();
    let (r0, r1) = (Sig::In(8), Sig::In(9));
    if annihilation {
        for c in 0..2 {
            let a = g.and(p[c], m[c + 2]);
            let b = g.and(m[c], p[c + 2]);
            let kill = g.or(a, b);
            let keep = g.not(kill);
            for x in [c, c + 2] {
                p[x] = g.and(p[x], keep);
                m[x] = g.and(m[x], keep);
            }
        }
    }
    g.phase = 1;
    let mut outputs = Vec::new();
    for a in [p, m] {
        // Out channel d = in channel d - r: lo[d] picks d or d - 1 by r0,
        // then r1 picks lo[d] or lo[d + 2] (= in d - 2 or d - 3).
        let lo: Vec<Sig> = (0..4).map(|d| g.mux(r0, a[d], a[(d + 3) % 4])).collect();
        for d in 0..4 {
            let o = g.mux(r1, lo[d], lo[(d + 2) % 4]);
            outputs.push(o);
        }
    }
    Circuit::prune(1, false, 10, g, outputs)
}

/// Bit-sliced CPU engine: 8 bit-planes of u64 words (plane c the + channel
/// c, plane 4 + c the - channel c), the collision as gates, boundary sites
/// re-seeded per site, charges accumulated per 8 columns with popcounts.
pub struct Bitsliced {
    pub n: usize,
    pub wn: usize,
    pub iter: u64,
    pub seed: u32,
    pub th: Vec<u32>,
    pub acc: Vec<i32>,
    cur: Vec<u64>,
    other: Vec<u64>,
    prog: crate::bitsliced::Program,
    scratch: Vec<u64>,
}

impl Bitsliced {
    pub fn new(n: usize, state: &[u8], th: Vec<u32>, seed: u32, annihilation: bool) -> Self {
        let q: Vec<u32> = state.iter().map(|&x| x as u32).collect();
        let cur = crate::bitsliced::pack(n, 8, &q);
        let prog = crate::bitsliced::Program::new(&circuit(annihilation));
        let scratch = vec![0u64; prog.slots() * crate::bitsliced::CHUNK];
        Bitsliced { n, wn: n / 64, iter: 0, seed, th, acc: vec![0; n * n / GROUP], other: cur.clone(), cur, prog, scratch }
    }

    pub fn from_problem(p: &crate::Problem, seed: u32, annihilation: bool) -> Self {
        Self::new(p.n, &initial(p.n), thresholds(p), seed, annihilation)
    }

    pub fn run(&mut self, iters: usize) {
        for _ in 0..iters {
            self.step();
            self.iter += 1;
        }
    }

    fn reseed_lane(&self, dst: &mut [u64], key: u32, i: usize, j: usize) {
        let (n, wn) = (self.n, self.wn);
        let plane = n * wn;
        let s = reseed(key, boundary_threshold(&self.th, n, i, j), n, i, j);
        let w = i * wn + j / 64;
        let bit = 1u64 << (j % 64);
        for b in 0..8 {
            dst[b * plane + w] = (dst[b * plane + w] & !bit) | if (s >> b) & 1 == 1 { bit } else { 0 };
        }
    }

    fn step(&mut self) {
        use crate::bitsliced::CHUNK;
        let (n, wn) = (self.n, self.wn);
        let plane = n * wn;
        let k = keys(self.seed, self.iter as u32);
        let mut dst = std::mem::take(&mut self.other);
        let src = &self.cur;
        let prog = &self.prog;
        let (start, end) = (wn, (n - 1) * wn);
        let mut i0 = start;
        while i0 < end {
            let len = CHUNK.min(end - i0);
            for c in 0..2 {
                // Planes of channel c (+) and 4 + c (-).
                for b in [c, 4 + c] {
                    let sp = &src[b * plane..][..plane];
                    let sp2 = &src[(b + 2) * plane..][..plane];
                    for l in 0..len {
                        let i = i0 + l;
                        // Channel 0 arrives from the west, 2 from the east; 1 from the row above, 3 from below.
                        if c == 0 {
                            self.scratch[b * CHUNK + l] = (sp[i] << 1) | (sp[i - 1] >> 63);
                            self.scratch[(b + 2) * CHUNK + l] = (sp2[i] >> 1) | (sp2[i + 1] << 63);
                        } else {
                            self.scratch[b * CHUNK + l] = sp[i - wn];
                            self.scratch[(b + 2) * CHUNK + l] = sp2[i + wn];
                        }
                    }
                }
            }
            for l in 0..len {
                let i = i0 + l;
                self.scratch[8 * CHUNK + l] = crate::rng::word_rand(k[0], n, i / wn, i % wn);
                self.scratch[9 * CHUNK + l] = crate::rng::word_rand(k[1], n, i / wn, i % wn);
            }
            prog.run(&mut self.scratch);
            for l in 0..len {
                let i = i0 + l;
                let (r, col) = (i / wn, i % wn);
                let mut interior = u64::MAX;
                if col == 0 {
                    interior &= !1;
                }
                if col == wn - 1 {
                    interior &= !(1 << 63);
                }
                let mut pw = [0u64; 4];
                let mut mw = [0u64; 4];
                for c in 0..4 {
                    pw[c] = self.scratch[prog.outputs[c] * CHUNK + l];
                    mw[c] = self.scratch[prog.outputs[4 + c] * CHUNK + l];
                    dst[c * plane + i] = pw[c];
                    dst[(4 + c) * plane + i] = mw[c];
                }
                for g in 0..8 {
                    let mask = interior & (0xffu64 << (8 * g));
                    let q: i32 = (0..4).map(|c| (pw[c] & mask).count_ones() as i32 - (mw[c] & mask).count_ones() as i32).sum();
                    let a = &mut self.acc[r * (n / GROUP) + 8 * col + g];
                    *a = a.wrapping_add(q);
                }
            }
            i0 += len;
        }
        for i in 0..n {
            if i == 0 || i == n - 1 {
                for j in 0..n {
                    self.reseed_lane(&mut dst, k[2], i, j);
                }
            } else {
                self.reseed_lane(&mut dst, k[2], i, 0);
                self.reseed_lane(&mut dst, k[2], i, n - 1);
            }
        }
        self.other = std::mem::replace(&mut self.cur, dst);
    }

    /// Site states (bits 0..3 +, 4..7 -), row-major.
    pub fn read(&self) -> Vec<u8> {
        crate::bitsliced::unpack(self.n, 8, &self.cur).into_iter().map(|x| x as u8).collect()
    }

    pub fn words(&self) -> &[u64] {
        &self.cur
    }
}
