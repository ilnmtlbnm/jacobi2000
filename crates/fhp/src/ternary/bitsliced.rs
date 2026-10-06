//! Bit-sliced ternary engine: two bit-planes per channel (14 planes) in u64
//! words, the collision as the generated circuit (rule Chiral), streaming
//! as shifts; bit-identical to `engine::Ternary` with rule Chiral.
//!
//! Layout: `data[(y * 14 + p) * wn + i]`, plane p = 2 c + b of channel c,
//! bit b of the count's code (`collide_gen::ENCODING`).

use super::collide_gen::{collide, ENCODING};
use super::Rule;
use super::engine::{self, boundary_state, Boundary, EVERY, OBST, STREAM_BOUNDARY, STREAM_COLLIDE};
use crate::{chunk_rand, step_key, DX, DY};

pub const PLANES: usize = 14;

pub struct Bitsliced {
    pub w: usize,
    pub h: usize,
    pub wn: usize,
    pub data: Vec<u64>,
    next: Vec<u64>,
    pub obst: Vec<u64>,
    pub seed: u32,
    pub boundary: Option<Boundary>,
    pub step: u32,
}

impl Bitsliced {
    pub fn from_cells(w: usize, h: usize, cells: &[u16], params: engine::Params) -> Self {
        crate::check_shape(w, h).expect("lattice shape");
        assert_eq!(params.rule, Rule::Chiral, "the circuit is rule Chiral");
        let wn = w / 64;
        let mut data = vec![0u64; h * PLANES * wn];
        let mut obst = vec![0u64; h * wn];
        for y in 0..h {
            for x in 0..w {
                let c = cells[y * w + x];
                let (i, b) = (x / 64, x % 64);
                let p = ENCODING.planes(c & !OBST);
                for k in 0..PLANES {
                    if (p >> k) & 1 == 1 {
                        data[(y * PLANES + k) * wn + i] |= 1 << b;
                    }
                }
                if c & OBST != 0 {
                    obst[y * wn + i] |= 1 << b;
                }
            }
        }
        Bitsliced { w, h, wn, next: vec![0; data.len()], data, obst, seed: params.seed, boundary: params.boundary, step: 0 }
    }

    pub fn to_cells(&self) -> Vec<u16> {
        let mut cells = vec![0u16; self.w * self.h];
        for y in 0..self.h {
            for x in 0..self.w {
                let (i, b) = (x / 64, x % 64);
                let p = (0..PLANES).fold(0u16, |a, k| a | ((((self.data[(y * PLANES + k) * self.wn + i] >> b) & 1) as u16) << k));
                let s = ENCODING.state(p).expect("planes hold a count");
                cells[y * self.w + x] = s | if (self.obst[y * self.wn + i] >> b) & 1 == 1 { OBST } else { 0 };
            }
        }
        cells
    }

    fn row(&self, y: usize, out: &mut [u64]) {
        let (h, wn) = (self.h, self.wn);
        let plane = |sy: usize, p: usize| &self.data[(sy * PLANES + p) * wn..][..wn];
        let src: [(usize, i32); 6] = std::array::from_fn(|d| {
            let sy = (y as i64 - DY[d] as i64).rem_euclid(h as i64) as usize;
            (sy, DX[sy & 1][d])
        });
        let kc = step_key(self.seed, self.step, STREAM_COLLIDE);
        let w32 = (2 * wn) as u32;
        for i in 0..wn {
            let mut x = [0u64; 16];
            for (d, &(sy, dx)) in src.iter().enumerate() {
                for b in 0..2 {
                    let s = plane(sy, 2 * d + b);
                    x[2 * d + b] = match dx {
                        1 => (s[i] << 1) | (s[(i + wn - 1) % wn] >> 63),
                        -1 => (s[i] >> 1) | (s[(i + 1) % wn] << 63),
                        _ => s[i],
                    };
                }
            }
            x[12] = plane(y, 12)[i];
            x[13] = plane(y, 13)[i];
            x[14] = self.obst[y * wn + i];
            let idx = y as u32 * w32 + 2 * i as u32;
            x[15] = chunk_rand(kc, idx) as u64 | (chunk_rand(kc, idx + 1) as u64) << 32;
            let o = collide(x);
            for p in 0..PLANES {
                out[p * wn + i] = o[p];
            }
        }
        if let (Some(b), true) = (self.boundary, self.step.is_multiple_of(EVERY)) {
            let kb = step_key(self.seed, self.step, STREAM_BOUNDARY);
            for x in (0..b.inlet).chain(self.w - b.outlet..self.w) {
                let (i, bit) = (x / 64, x % 64);
                if (self.obst[y * wn + i] >> bit) & 1 == 1 {
                    continue;
                }
                let p = ENCODING.planes(boundary_state(&b, kb, (y * self.w + x) as u32));
                for k in 0..PLANES {
                    let w = &mut out[k * wn + i];
                    *w = (*w & !(1 << bit)) | (((p >> k) & 1) as u64) << bit;
                }
            }
        }
    }

    pub fn step(&mut self) {
        let mut next = std::mem::take(&mut self.next);
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            next.par_chunks_mut(PLANES * self.wn).enumerate().for_each(|(y, out)| self.row(y, out));
        }
        #[cfg(not(feature = "threads"))]
        for (y, out) in next.chunks_mut(PLANES * self.wn).enumerate() {
            self.row(y, out);
        }
        self.next = std::mem::replace(&mut self.data, next);
        self.step += 1;
    }

    pub fn run(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
}
