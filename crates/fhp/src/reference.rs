//! Per-site reference engine: one byte per site, a 2 x 256 lookup table.

use crate::boundary::{self, mirror};
use crate::{chunk_rand, force_mask, rules, source, step_key, Params, E, NDIR, OBST, REST, STREAM_CHIRALITY, STREAM_FORCING, W};

pub struct Reference {
    pub w: usize,
    pub h: usize,
    pub cells: Vec<u8>,
    next: Vec<u8>,
    /// For each site and direction, the index of the source site.
    src: Vec<[u32; NDIR]>,
    table: [[u8; 256]; 2],
    pub params: Params,
    pub step: u32,
    seeds: [u32; boundary::NSEEDS],
    /// With open boundaries: particles that crossed the inlet's edge into the
    /// interior, and the outlet's first column out of it (net, all steps).
    pub flux_in: i64,
    pub flux_out: i64,
}

impl Reference {
    pub fn new(w: usize, h: usize, cells: Vec<u8>, params: Params) -> Self {
        assert!(w.is_multiple_of(64) && h.is_multiple_of(2));
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
        let seeds = boundary::stream_seeds(params.seed);
        Reference { w, h, next: vec![0; w * h], cells, src, table: rules::table(), params, step: 0, seeds, flux_in: 0, flux_out: 0 }
    }

    pub fn stream_only(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.w * self.h];
        for (i, o) in out.iter_mut().enumerate() {
            let mut s = self.cells[i] & (REST | OBST);
            for d in 0..NDIR {
                s |= self.cells[self.src[i][d] as usize] & (1 << d);
            }
            *o = s;
        }
        out
    }

    pub fn step(&mut self) {
        let w32 = (self.w / 32) as u32;
        let kc = step_key(self.params.seed, self.step, STREAM_CHIRALITY);
        let kf = step_key(self.params.seed, self.step, STREAM_FORCING);
        let thr = self.params.force_threshold;
        // Columns whose left edge the fluxes are counted across.
        let edges = self.params.boundary.map(|b| [(b.inlet * 32) as usize, self.w - (b.outlet * 32) as usize]);
        let keys = boundary::step_keys(&self.seeds, self.step);
        for y in 0..self.h {
            let wall = self.params.walls && (y == 0 || y == self.h - 1);
            for chunk in 0..self.w / 32 {
                let idx = y as u32 * w32 + chunk as u32;
                let chir = chunk_rand(kc, idx);
                let force = if thr > 0 && !wall { force_mask(kf, idx, thr) } else { 0 };
                for b in 0..32 {
                    let i = y * self.w + chunk * 32 + b;
                    let mut s = self.cells[i] & (REST | OBST);
                    for d in 0..NDIR {
                        s |= self.cells[self.src[i][d] as usize] & (1 << d);
                    }
                    let x = chunk * 32 + b;
                    if let Some(edges) = edges.filter(|e| e.iter().any(|&e| x == e || x + 1 == e)) {
                        for d in 0..NDIR {
                            let sx = self.src[i][d] as usize % self.w;
                            if s & (1 << d) == 0 {
                                continue;
                            }
                            for (k, &e) in edges.iter().enumerate() {
                                let sign = if sx + 1 == e && x == e { 1 } else if sx == e && x + 1 == e { -1 } else { 0 };
                                if k == 0 {
                                    self.flux_in += sign;
                                } else {
                                    self.flux_out += sign;
                                }
                            }
                        }
                    }
                    let c = ((chir >> b) & 1) as usize;
                    let mut post = if wall && s & OBST == 0 { mirror(s) } else { self.table[c][s as usize] };
                    let fw = (1u8 << W) | (1u8 << E);
                    if (force >> b) & 1 == 1 && post & OBST == 0 && post & fw == 1 << W {
                        post ^= fw;
                    }
                    self.next[i] = post;
                }
                if let (Some(bd), false, true) = (self.params.boundary.as_ref(), wall, boundary::Boundary::active(self.step)) {
                    if let Some((mask, eq)) = boundary::replacement(bd, &keys, idx, chunk as u32, w32) {
                        for b in 0..32 {
                            let i = y * self.w + chunk * 32 + b;
                            if (mask >> b) & 1 == 1 && self.next[i] & OBST == 0 {
                                self.next[i] = (0..7).fold(0u8, |a, p| a | ((((eq[p] >> b) & 1) as u8) << p));
                            }
                        }
                    }
                }
            }
        }
        std::mem::swap(&mut self.cells, &mut self.next);
        self.step += 1;
    }

    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Particles in the lattice (moving + rest).
    pub fn mass(&self) -> u64 {
        self.cells.iter().map(|&s| rules::mass(s) as u64).sum()
    }

    /// Total momentum in integer units (2 px, 2 py / sqrt 3), obstacle sites included.
    pub fn momentum(&self) -> (i64, i64) {
        self.cells.iter().fold((0, 0), |a, &s| {
            let p = rules::momentum(s);
            (a.0 + p.0 as i64, a.1 + p.1 as i64)
        })
    }
}
