//! Open boundaries in x: an inlet and an outlet sponge that replace sites
//! with fresh samples of the FHP equilibrium, plus optional free-slip walls
//! in y.
//!
//! After the collision (and forcing), on every EVERY-th step, each 32-site
//! chunk column c has a replacement probability q(c) in units of 1/4096:
//! 1 over the inlet chunks; over the outlet chunks at the right end, 1 for
//! `Randomize` and `Delete` and a quadratic ramp from 0 to 1 for `Sponge`;
//! 0 elsewhere. `Delete` empties the replaced outlet sites. A replaced fluid site gets an independent sample of the
//! Fermi-Dirac equilibrium at the inflow density and speed (d'Humieres and
//! Lallemand 1987, eq. 2.2). Every Bernoulli draw is a bit-serial comparison
//! of 12 random words, one per random stream, so the per-site bits are the
//! same in every engine.

use crate::{chunk_rand, lowbias32, CX2};

/// Probabilities are thresholds out of 2^QBITS.
pub const QBITS: u32 = 12;
pub const QONE: u32 = 1 << QBITS;
/// The boundary chunks are refreshed every EVERY steps (the steps with
/// step % EVERY == 0), all rows at once; on the GPU that is one small extra
/// dispatch per EVERY steps, and the main step kernel has no boundary code.
/// The inlet chunk is 32 columns wide, so what streams in from the wrap
/// between refreshes (at most EVERY - 1 columns) never reaches the interior.
pub const EVERY: u32 = 32;
/// The sponge ramps quadratically up to 1 per refresh (1/32 per step on
/// average); its last chunk is replaced on every refresh, and what streams
/// from it to the inlet between refreshes stays inside the inlet chunk. A
/// fluid parcel spends about 32 / U steps in a chunk, so the relaxation
/// builds up over the sponge instead of at its first column.
pub const SPONGE_MAX: u32 = QONE;
/// Bernoulli streams: one per plane (7) and one for the replacement mask.
pub const NBERN: usize = 8;
pub const NSEEDS: usize = NBERN * QBITS as usize;
/// First random-stream id of the boundary (0 is chirality, 1 forcing).
pub const STREAM_BOUNDARY: u32 = 2;

/// What the outlet chunks do on a refresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outlet {
    /// Replace sites with fresh equilibrium samples with a probability that
    /// ramps quadratically from 0 to 1 across the outlet.
    Sponge = 0,
    /// Replace every outlet site with a fresh equilibrium sample.
    Randomize = 1,
    /// Empty every outlet site (a low-density sink).
    Delete = 2,
}

impl Outlet {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sponge" => Some(Outlet::Sponge),
            "randomize" => Some(Outlet::Randomize),
            "delete" => Some(Outlet::Delete),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Outlet::Sponge => "sponge",
            Outlet::Randomize => "randomize",
            Outlet::Delete => "delete",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Boundary {
    /// Occupation thresholds of the six moving directions and the rest particle.
    pub eq: [u32; 7],
    /// Inlet width in 32-site chunks.
    pub inlet: u32,
    /// Outlet width in 32-site chunks, ending at the right edge.
    pub outlet: u32,
    pub kind: Outlet,
}

impl Boundary {
    pub fn new(density: f64, u: f64, inlet_chunks: u32, outlet_chunks: u32, kind: Outlet) -> Self {
        let n = equilibrium(density, u);
        Boundary { eq: n.map(|p| (p * QONE as f64).round() as u32), inlet: inlet_chunks, outlet: outlet_chunks, kind }
    }

    /// Whether the boundary is refreshed at `step` (all rows at once).
    #[inline]
    pub fn active(step: u32) -> bool {
        step.is_multiple_of(EVERY)
    }

    /// Replacement threshold of chunk column `c` of `w32`, on a refresh.
    pub fn q(&self, c: u32, w32: u32) -> u32 {
        if c < self.inlet {
            return QONE;
        }
        let start = w32 - self.outlet;
        if c + 1 == w32 || (c >= start && self.kind != Outlet::Sponge) {
            return QONE;
        }
        if c >= start {
            let k = (c - start + 1) as u64;
            let s = self.outlet as u64;
            return ((SPONGE_MAX as u64 * k * k) / (s * s)) as u32;
        }
        0
    }
}

/// Fermi-Dirac equilibrium N_i = 1 / (1 + exp(h + a c_ix)) for the moving
/// directions and 1 / (1 + exp(h)) for the rest particle (d'Humieres and
/// Lallemand 1987, eq. 2.2 with momentum along x), with h and a solved so
/// that the mass is 7 d and the x momentum 7 d u.
pub fn equilibrium(d: f64, u: f64) -> [f64; 7] {
    let cx: [f64; 7] = std::array::from_fn(|i| if i < 6 { CX2[i] as f64 / 2.0 } else { 0.0 });
    let (mut h, mut a) = (((1.0 - d) / d).ln(), 0.0);
    let n_of = |h: f64, a: f64| -> [f64; 7] { std::array::from_fn(|i| 1.0 / (1.0 + (h + a * cx[i]).exp())) };
    for _ in 0..100 {
        let n = n_of(h, a);
        let f1: f64 = n.iter().sum::<f64>() - 7.0 * d;
        let f2: f64 = (0..7).map(|i| cx[i] * n[i]).sum::<f64>() - 7.0 * d * u;
        // dN/dh = -N(1-N), dN/da = -c N(1-N)
        let (mut j11, mut j12, mut j22) = (0.0, 0.0, 0.0);
        for i in 0..7 {
            let v = n[i] * (1.0 - n[i]);
            j11 -= v;
            j12 -= cx[i] * v;
            j22 -= cx[i] * cx[i] * v;
        }
        let det = j11 * j22 - j12 * j12;
        h -= (f1 * j22 - f2 * j12) / det;
        a -= (j11 * f2 - j12 * f1) / det;
    }
    n_of(h, a)
}

/// Per-stream seeds of the boundary's 96 random streams, constant over steps.
pub fn stream_seeds(seed: u32) -> [u32; NSEEDS] {
    std::array::from_fn(|k| crate::stream_seed(seed, STREAM_BOUNDARY + k as u32))
}

/// The boundary's 96 per-step keys (one per random stream), computed once
/// per step: key = lowbias32(stream seed ^ step), as `step_key`.
pub fn step_keys(seeds: &[u32; NSEEDS], step: u32) -> [u32; NSEEDS] {
    seeds.map(|s| lowbias32(s ^ step))
}

/// 32 Bernoulli(t / 4096) bits of chunk `idx` from Bernoulli stream `s`,
/// given this step's keys.
#[inline]
pub fn bern(keys: &[u32; NSEEDS], s: usize, idx: u32, t: u32) -> u32 {
    if t >= QONE {
        return u32::MAX;
    }
    if t == 0 {
        return 0;
    }
    let (mut lt, mut eq) = (0u32, u32::MAX);
    for b in 0..QBITS {
        let r = chunk_rand(keys[s * QBITS as usize + b as usize], idx);
        if (t >> (QBITS - 1 - b)) & 1 == 1 {
            lt |= eq & !r;
            eq &= r;
        } else {
            eq &= !r;
        }
        if eq == 0 {
            break;
        }
    }
    lt
}

/// The replacement of chunk `c` (index `idx`) at `step`: the mask of sites
/// to replace (before excluding obstacles) and the seven fresh planes.
#[inline]
pub fn replacement(b: &Boundary, keys: &[u32; NSEEDS], idx: u32, c: u32, w32: u32) -> Option<(u32, [u32; 7])> {
    let q = b.q(c, w32);
    if q == 0 {
        return None;
    }
    let mask = bern(keys, 7, idx, q);
    if mask == 0 {
        return None;
    }
    if b.kind == Outlet::Delete && c >= b.inlet {
        return Some((mask, [0; 7]));
    }
    Some((mask, std::array::from_fn(|p| bern(keys, p, idx, b.eq[p]))))
}

/// Free-slip wall (rows 0 and H-1 when walls are on): mirror the moving
/// bits about the x axis, NE <-> SE, NW <-> SW.
pub fn mirror(s: u8) -> u8 {
    let bit = |k: u8| (s >> k) & 1;
    (s & 0b1100_1001) | bit(5) << 1 | bit(4) << 2 | bit(2) << 4 | bit(1) << 5
}
