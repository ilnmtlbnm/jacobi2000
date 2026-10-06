//! Plain integer red-black Jacobi in the ternary engine's fixed point: u is
//! i32 two's complement, v = u / 2^28 (the same start as `ternary::initial`),
//! and each point of the active colour gets
//!
//!   u = (uE + uW + uN + uS + 2) >> 2        (round to nearest, arithmetic shift)
//!
//! with every add wrapping at 32 bits, as the ternary accumulator does. The
//! values stay in [0, 2^28], so nothing wraps in practice.

use crate::ternary::{colour_mask64, input, IN_E, IN_N, IN_S, IN_W, W};

/// The update at interior point k.
#[inline(always)]
pub fn update(n: usize, u: &[i32], k: usize) -> i32 {
    u[k + 1].wrapping_add(u[k - 1]).wrapping_add(u[k - n].wrapping_add(u[k + n])).wrapping_add(2) >> 2
}

/// Half of a red-black sweep: points with (i + j) % 2 == colour updated
/// from `src`, the others copied.
pub fn half(n: usize, src: &[i32], dst: &mut [i32], colour: usize) {
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            let k = i * n + j;
            dst[k] = if (i + j) % 2 == colour { update(n, src, k) } else { src[k] };
        }
    }
}

/// Plain integer engine (the reference of the bit-sliced and GPU engines).
pub struct IntEngine {
    pub n: usize,
    pub cur: Vec<i32>,
    other: Vec<i32>,
    pub iter: u64,
}

impl IntEngine {
    pub fn new(n: usize, u: Vec<i32>) -> Self {
        IntEngine { n, other: u.clone(), cur: u, iter: 0 }
    }

    pub fn run(&mut self, iters: usize) {
        for _ in 0..iters {
            for c in 0..2 {
                half(self.n, &self.cur, &mut self.other, c);
                std::mem::swap(&mut self.cur, &mut self.other);
            }
            self.iter += 1;
        }
    }
}

pub const PHASE_NAMES: [&str; 2] = ["adder tree E + W + N + S", "round (+ 2)"];

/// The update as gates: three 32-bit ripple-carry adds ((E + W) + (N + S)),
/// a fourth adding the constant 2, then the arithmetic shift by 2 (bits 2..31
/// of the sum, then bit 31 twice). Inputs E, W, N, S as in the ternary
/// circuit (`input(k, j)`); outputs the new u, LSB first.
pub fn circuit() -> crate::circuit::Circuit {
    use crate::circuit::{Builder, Circuit, Sig};
    let mut g = Builder::default();
    let op = |k: usize| -> Vec<Sig> { (0..W).map(|j| Sig::In(input(k, j))).collect() };
    let s1 = g.add(&op(IN_E), &op(IN_W), W);
    let s2 = g.add(&op(IN_N), &op(IN_S), W);
    let sum = g.add(&s1, &s2, W);
    g.phase = 1;
    let two: Vec<Sig> = (0..W).map(|j| if j == 1 { Sig::One } else { Sig::Zero }).collect();
    let sum = g.add(&sum, &two, W);
    let mut outputs = sum[2..].to_vec();
    outputs.extend([sum[W - 1], sum[W - 1]]);
    Circuit::prune(W, true, 4 * W, g, outputs)
}

/// Bit-sliced CPU engine: red-black sweeps, 32 bit-planes of u64 words.
pub struct Bitsliced {
    pub n: usize,
    pub wn: usize,
    pub iter: u64,
    cur: Vec<u64>,
    other: Vec<u64>,
    prog: crate::bitsliced::Program,
    scratch: Vec<u64>,
}

impl Bitsliced {
    pub fn new(n: usize, u: &[i32]) -> Self {
        let q: Vec<u32> = u.iter().map(|&x| x as u32).collect();
        let cur = crate::bitsliced::pack(n, W, &q);
        let prog = crate::bitsliced::Program::new(&circuit());
        let scratch = vec![0u64; prog.slots() * crate::bitsliced::CHUNK];
        Bitsliced { n, wn: n / 64, iter: 0, other: cur.clone(), cur, prog, scratch }
    }

    pub fn from_problem(p: &crate::Problem) -> Self {
        Self::new(p.n, &crate::ternary::initial(p))
    }

    pub fn run(&mut self, iters: usize) {
        for _ in 0..iters {
            self.half(0);
            self.half(1);
            self.iter += 1;
        }
    }

    fn half(&mut self, colour: usize) {
        use crate::bitsliced::CHUNK;
        let (n, wn) = (self.n, self.wn);
        let plane = n * wn;
        let (src, dst) = (&self.cur, &mut self.other);
        let (start, end) = (wn, (n - 1) * wn);
        let mut i0 = start;
        while i0 < end {
            let len = CHUNK.min(end - i0);
            for j in 0..W {
                let sp = &src[j * plane..][..plane];
                for l in 0..len {
                    let i = i0 + l;
                    let c = sp[i];
                    self.scratch[input(IN_E, j) * CHUNK + l] = (c >> 1) | (sp[i + 1] << 63);
                    self.scratch[input(IN_W, j) * CHUNK + l] = (c << 1) | (sp[i - 1] >> 63);
                    self.scratch[input(IN_N, j) * CHUNK + l] = sp[i - wn];
                    self.scratch[input(IN_S, j) * CHUNK + l] = sp[i + wn];
                }
            }
            self.prog.run(&mut self.scratch);
            for l in 0..len {
                let i = i0 + l;
                let (r, col) = (i / wn, i % wn);
                let mut active = colour_mask64(r, colour);
                if col == 0 {
                    active &= !1;
                }
                if col == wn - 1 {
                    active &= !(1 << 63);
                }
                for j in 0..W {
                    let new = self.scratch[self.prog.outputs[j] * CHUNK + l];
                    dst[j * plane + i] = (new & active) | (src[j * plane + i] & !active);
                }
            }
            i0 += len;
        }
        std::mem::swap(&mut self.cur, &mut self.other);
    }

    pub fn read(&self) -> Vec<i32> {
        crate::bitsliced::unpack(self.n, W, &self.cur).into_iter().map(|x| x as i32).collect()
    }
}
