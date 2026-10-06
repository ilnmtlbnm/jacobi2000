//! Bit-sliced CPU engine: b bit-planes of u64 words, plane k at
//! `k * n * wn`, word `row * wn + col`, lane `l` of a word is column
//! `64 * col + l`. The circuit is run over chunks of `CHUNK` consecutive
//! words, one gate at a time across the chunk. Read as little-endian u32
//! pairs, the same buffer is the GPU engine's layout.

use crate::circuit::{Circuit, Op, Sig};

pub(crate) const CHUNK: usize = 64;

/// A circuit with its signals numbered as slots of a scratch buffer: inputs,
/// then the two constants, then the gates.
pub(crate) struct Program {
    pub inputs: usize,
    ops: Vec<(Op, usize, usize)>,
    pub outputs: Vec<usize>,
}

impl Program {
    pub fn new(c: &Circuit) -> Self {
        let inputs = c.inputs();
        let slot = |s: Sig| match s {
            Sig::In(k) => k,
            Sig::Zero => inputs,
            Sig::One => inputs + 1,
            Sig::Gate(i) => inputs + 2 + i,
        };
        Program { inputs, ops: c.gates.iter().map(|g| (g.op, slot(g.a), slot(g.b))).collect(), outputs: c.outputs.iter().map(|&s| slot(s)).collect() }
    }

    pub fn slots(&self) -> usize {
        self.inputs + 2 + self.ops.len()
    }

    /// Evaluate every gate across the chunk; inputs already in their slots.
    pub fn run(&self, buf: &mut [u64]) {
        buf[self.inputs * CHUNK..][..CHUNK].fill(0);
        buf[(self.inputs + 1) * CHUNK..][..CHUNK].fill(u64::MAX);
        for (g, &(op, a, b)) in self.ops.iter().enumerate() {
            let (lo, hi) = buf.split_at_mut((self.inputs + 2 + g) * CHUNK);
            let out = &mut hi[..CHUNK];
            let x = &lo[a * CHUNK..][..CHUNK];
            let y = &lo[b * CHUNK..][..CHUNK];
            match op {
                Op::And => out.iter_mut().zip(x.iter().zip(y)).for_each(|(o, (x, y))| *o = x & y),
                Op::Or => out.iter_mut().zip(x.iter().zip(y)).for_each(|(o, (x, y))| *o = x | y),
                Op::Xor => out.iter_mut().zip(x.iter().zip(y)).for_each(|(o, (x, y))| *o = x ^ y),
                Op::Not => out.iter_mut().zip(x).for_each(|(o, x)| *o = !x),
            }
        }
    }
}

pub struct Bitsliced {
    pub n: usize,
    pub bits: usize,
    pub wn: usize,
    pub round: bool,
    /// Stochastic rounding (overrides `round`), random streams from `seed`;
    /// step t of the run uses step index `iter + t`.
    pub stochastic: bool,
    pub seed: u32,
    pub iter: u64,
    cur: Vec<u64>,
    other: Vec<u64>,
    /// Truncating, round-to-nearest and stochastic programs.
    programs: [Program; 3],
    scratch: Vec<u64>,
}

/// Pack a row-major n x n field of b-bit values into bit-planes.
pub fn pack(n: usize, bits: usize, q: &[u32]) -> Vec<u64> {
    let wn = n / 64;
    let mut w = vec![0u64; bits * n * wn];
    for (k, &v) in q.iter().enumerate() {
        let (r, c) = (k / n, k % n);
        for p in 0..bits {
            w[(p * n + r) * wn + c / 64] |= (((v >> p) & 1) as u64) << (c % 64);
        }
    }
    w
}

pub fn unpack(n: usize, bits: usize, w: &[u64]) -> Vec<u32> {
    let wn = n / 64;
    let mut q = vec![0u32; n * n];
    for (k, v) in q.iter_mut().enumerate() {
        let (r, c) = (k / n, k % n);
        for p in 0..bits {
            *v |= (((w[(p * n + r) * wn + c / 64] >> (c % 64)) & 1) as u32) << p;
        }
    }
    q
}

impl Bitsliced {
    pub fn new(n: usize, bits: usize, q: &[u32]) -> Self {
        assert!(n >= 64 && n.is_multiple_of(64) && q.len() == n * n);
        let cur = pack(n, bits, q);
        let programs = [Program::new(&Circuit::jacobi(bits, false)), Program::new(&Circuit::jacobi(bits, true)), Program::new(&Circuit::jacobi_stochastic(bits))];
        let scratch = vec![0u64; programs.iter().map(|p| p.slots()).max().unwrap() * CHUNK];
        Bitsliced { n, bits, wn: n / 64, round: false, stochastic: false, seed: 1, iter: 0, other: cur.clone(), cur, programs, scratch }
    }

    pub fn from_problem(p: &crate::Problem, bits: usize) -> Self {
        Self::new(p.n, bits, &p.initial_q(bits))
    }

    pub fn run(&mut self, iters: usize) {
        for _ in 0..iters {
            self.step();
            self.iter += 1;
        }
    }

    fn step(&mut self) {
        let (n, wn, b) = (self.n, self.wn, self.bits);
        let plane = n * wn;
        let prog = &self.programs[if self.stochastic { 2 } else { self.round as usize }];
        let keys = crate::stochastic::keys(self.seed, self.iter as u32);
        let (src, dst) = (&self.cur, &mut self.other);
        // Interior rows only; rows 0 and n-1 never change and both buffers hold them.
        let (start, end) = (wn, (n - 1) * wn);
        let mut i0 = start;
        while i0 < end {
            let len = CHUNK.min(end - i0);
            for k in 0..b {
                let s = &src[k * plane..][..plane];
                for j in 0..len {
                    let i = i0 + j;
                    let c = s[i];
                    self.scratch[k * CHUNK + j] = (c >> 1) | (s[i + 1] << 63);
                    self.scratch[(b + k) * CHUNK + j] = (c << 1) | (s[i - 1] >> 63);
                    self.scratch[(2 * b + k) * CHUNK + j] = s[i - wn];
                    self.scratch[(3 * b + k) * CHUNK + j] = s[i + wn];
                }
            }
            if self.stochastic {
                for j in 0..len {
                    let i = i0 + j;
                    for (r, &key) in keys.iter().enumerate() {
                        self.scratch[(4 * b + r) * CHUNK + j] = crate::rng::word_rand(key, n, i / wn, i % wn);
                    }
                }
            }
            prog.run(&mut self.scratch);
            for (k, &o) in prog.outputs.iter().enumerate() {
                dst[k * plane + i0..][..len].copy_from_slice(&self.scratch[o * CHUNK..][..len]);
            }
            i0 += len;
        }
        // Boundary columns: lane 0 of the first word and lane 63 of the last.
        for k in 0..b {
            for r in 1..n - 1 {
                let (a, z) = (k * plane + r * wn, k * plane + r * wn + wn - 1);
                dst[a] = (dst[a] & !1) | (src[a] & 1);
                dst[z] = (dst[z] & !(1 << 63)) | (src[z] & (1 << 63));
            }
        }
        std::mem::swap(&mut self.cur, &mut self.other);
    }

    pub fn words(&self) -> &[u64] {
        &self.cur
    }

    pub fn read_q(&self) -> Vec<u32> {
        unpack(self.n, self.bits, &self.cur)
    }
}
