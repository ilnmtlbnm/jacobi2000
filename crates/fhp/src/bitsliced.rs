//! Bit-sliced CPU engine: one bit-plane per u64 word (64 sites), the
//! collision as the generated boolean circuit, streaming as shifts.
//!
//! Layout: `data[(y * 7 + p) * wn + i]`, plane p of row y, word i, site
//! x = 64 i + bit. Rows are independent within a step, so they run on a
//! rayon pool when the `threads` feature is on.

use crate::boundary::{self, Boundary, NSEEDS};
use crate::collide_gen::collide;
use crate::{chunk_rand, force_mask, step_key, Params, DX, DY, OBST, STREAM_CHIRALITY, STREAM_FORCING};

pub const PLANES: usize = 7;
/// Words per inner block; the circuit runs over a block at a time.
const CH: usize = 8;

pub struct Bitsliced {
    pub w: usize,
    pub h: usize,
    pub wn: usize,
    pub data: Vec<u64>,
    next: Vec<u64>,
    pub obst: Vec<u64>,
    pub params: Params,
    pub step: u32,
    seeds: [u32; NSEEDS],
}

impl Bitsliced {
    pub fn from_cells(w: usize, h: usize, cells: &[u8], params: Params) -> Self {
        Self::from_rows(w, h, params, |f| {
            for y in 0..h {
                f(y, &cells[y * w..][..w]);
            }
        })
    }

    /// Build from the setup's initial state, one row at a time.
    pub fn from_setup(s: &crate::Setup) -> Self {
        Self::from_rows(s.w, s.h, s.params, |f| s.initial_rows(f))
    }

    fn from_rows(w: usize, h: usize, params: Params, feed: impl FnOnce(&mut dyn FnMut(usize, &[u8]))) -> Self {
        crate::check_shape(w, h).expect("lattice shape");
        let wn = w / 64;
        let mut data = vec![0u64; h * PLANES * wn];
        let mut obst = vec![0u64; h * wn];
        feed(&mut |y, row| {
            for (x, &s) in row.iter().enumerate() {
                let (i, b) = (x / 64, x % 64);
                for p in 0..PLANES {
                    if s & (1 << p) != 0 {
                        data[(y * PLANES + p) * wn + i] |= 1 << b;
                    }
                }
                if s & OBST != 0 {
                    obst[y * wn + i] |= 1 << b;
                }
            }
        });
        let next = vec![0u64; data.len()];
        Bitsliced { w, h, wn, next, data, obst, params, step: 0, seeds: boundary::stream_seeds(params.seed) }
    }

    pub fn to_cells(&self) -> Vec<u8> {
        let mut cells = vec![0u8; self.w * self.h];
        for y in 0..self.h {
            for x in 0..self.w {
                let (i, b) = (x / 64, x % 64);
                let mut s = 0u8;
                for p in 0..PLANES {
                    s |= (((self.data[(y * PLANES + p) * self.wn + i] >> b) & 1) as u8) << p;
                }
                s |= (((self.obst[y * self.wn + i] >> b) & 1) as u8) << 7;
                cells[y * self.w + x] = s;
            }
        }
        cells
    }

    /// The seven planes as separate vectors, `wn` words per row.
    pub fn planes(&self) -> [Vec<u64>; 7] {
        std::array::from_fn(|p| {
            let mut v = Vec::with_capacity(self.h * self.wn);
            for y in 0..self.h {
                v.extend_from_slice(&self.data[(y * PLANES + p) * self.wn..][..self.wn]);
            }
            v
        })
    }

    pub fn block_sums(&self, bw: usize, bh: usize) -> crate::coarse::BlockSums {
        crate::coarse::block_sums(self.w, self.h, &self.planes(), &self.obst, bw, bh)
    }

    pub fn step(&mut self) {
        let ctx = Ctx {
            h: self.h,
            wn: self.wn,
            data: &self.data,
            obst: &self.obst,
            kc: step_key(self.params.seed, self.step, STREAM_CHIRALITY),
            kf: step_key(self.params.seed, self.step, STREAM_FORCING),
            thr: self.params.force_threshold,
            walls: self.params.walls,
            boundary: self.params.boundary,
            keys: boundary::step_keys(&self.seeds, self.step),
            step: self.step,
        };
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            self.next.par_chunks_mut(PLANES * self.wn).enumerate().for_each(|(y, out)| ctx.row(y, out));
        }
        #[cfg(not(feature = "threads"))]
        for (y, out) in self.next.chunks_mut(PLANES * self.wn).enumerate() {
            ctx.row(y, out);
        }
        std::mem::swap(&mut self.data, &mut self.next);
        self.step += 1;
    }

    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
}

struct Ctx<'a> {
    h: usize,
    wn: usize,
    data: &'a [u64],
    obst: &'a [u64],
    kc: u32,
    kf: u32,
    thr: u32,
    walls: bool,
    boundary: Option<Boundary>,
    keys: [u32; NSEEDS],
    step: u32,
}

impl Ctx<'_> {
    #[inline(always)]
    fn plane(&self, y: usize, p: usize) -> &[u64] {
        &self.data[(y * PLANES + p) * self.wn..][..self.wn]
    }

    fn row(&self, y: usize, out: &mut [u64]) {
        let (h, wn) = (self.h, self.wn);
        // Source row and column shift of each moving plane.
        let src: [(&[u64], i32); 6] = std::array::from_fn(|d| {
            let sy = (y as i64 - DY[d] as i64).rem_euclid(h as i64) as usize;
            (self.plane(sy, d), DX[sy & 1][d])
        });
        let rest = self.plane(y, 6);
        let obst = &self.obst[y * wn..][..wn];
        let w32 = (2 * wn) as u32;
        let wall = self.walls && (y == 0 || y == h - 1);
        let mut i0 = 0;
        while i0 < wn {
            let n = CH.min(wn - i0);
            let mut x = [[0u64; CH]; 9];
            for (d, &(s, dx)) in src.iter().enumerate() {
                for j in 0..n {
                    let i = i0 + j;
                    x[d][j] = match dx {
                        1 => (s[i] << 1) | (s[(i + wn - 1) % wn] >> 63),
                        -1 => (s[i] >> 1) | (s[(i + 1) % wn] << 63),
                        _ => s[i],
                    };
                }
            }
            let mut force = [0u64; CH];
            for j in 0..n {
                let i = i0 + j;
                x[6][j] = rest[i];
                x[7][j] = obst[i];
                let idx = y as u32 * w32 + 2 * i as u32;
                x[8][j] = chunk_rand(self.kc, idx) as u64 | (chunk_rand(self.kc, idx + 1) as u64) << 32;
                if self.thr > 0 && !wall {
                    force[j] = force_mask(self.kf, idx, self.thr) as u64 | (force_mask(self.kf, idx + 1, self.thr) as u64) << 32;
                }
            }
            let mut o = [[0u64; CH]; 7];
            for j in 0..CH {
                let mut r = collide([x[0][j], x[1][j], x[2][j], x[3][j], x[4][j], x[5][j], x[6][j], x[7][j], x[8][j]]);
                if wall {
                    // Free-slip: fluid sites mirror NE <-> SE, NW <-> SW.
                    let ob = x[7][j];
                    let m = [x[0][j], x[5][j], x[4][j], x[3][j], x[2][j], x[1][j], x[6][j]];
                    for k in 0..7 {
                        r[k] = (r[k] & ob) | (m[k] & !ob);
                    }
                }
                let f = force[j] & !x[7][j] & r[3] & !r[0];
                o[0][j] = r[0] ^ f;
                o[1][j] = r[1];
                o[2][j] = r[2];
                o[3][j] = r[3] ^ f;
                o[4][j] = r[4];
                o[5][j] = r[5];
                o[6][j] = r[6];
            }
            if let (Some(bd), false, true) = (self.boundary.as_ref(), wall, Boundary::active(self.step)) {
                for j in 0..n {
                    let i = i0 + j;
                    for half in 0..2 {
                        let c = 2 * i as u32 + half;
                        let idx = y as u32 * w32 + c;
                        if let Some((mask, eq)) = boundary::replacement(bd, &self.keys, idx, c, w32) {
                            let m = ((mask as u64) << (32 * half)) & !obst[i];
                            for p in 0..PLANES {
                                o[p][j] = (o[p][j] & !m) | (((eq[p] as u64) << (32 * half)) & m);
                            }
                        }
                    }
                }
            }
            for p in 0..PLANES {
                out[p * wn + i0..][..n].copy_from_slice(&o[p][..n]);
            }
            i0 += n;
        }
    }
}
