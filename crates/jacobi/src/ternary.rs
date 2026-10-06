//! Ternary corrections with a scale schedule. The field is a 32-bit two's
//! complement integer per point, v = u / 2^28. Each sweep k, with one scale
//! s_k for the whole grid,
//!
//!   D_i = uE + uW + uN + uS - 4 u_i          (= 4 Delta_i, exact)
//!   t_i = +1 if D_i >= 2 s_k, -1 if D_i <= -2 s_k, else 0   (|Delta_i| < s_k / 2)
//!   u_i += s_k t_i
//!
//! All arithmetic wraps at 32 bits; the values stay far inside the range, so
//! nothing wraps in practice.

pub const SHIFT: u32 = 28;
pub const Q: i32 = 1 << SHIFT;

pub fn initial(p: &crate::Problem) -> Vec<i32> {
    let n = p.n;
    (0..n * n).map(|k| if p.is_boundary(k / n, k % n) { (p.analytic(k / n, k % n) * Q as f64).round_ties_even() as i32 } else { 0 }).collect()
}

/// D at interior point k.
#[inline(always)]
pub fn residual(n: usize, u: &[i32], k: usize) -> i32 {
    u[k + 1].wrapping_add(u[k - 1]).wrapping_add(u[k - n]).wrapping_add(u[k + n]).wrapping_sub(u[k].wrapping_mul(4))
}

#[inline(always)]
pub fn trit(d: i32, s: i32, strict: bool) -> i32 {
    let two_s = s.wrapping_mul(2);
    let e = strict as i32;
    if d.wrapping_sub(two_s).wrapping_sub(e) >= 0 {
        1
    } else if d.wrapping_add(two_s).wrapping_sub(1 - e) < 0 {
        -1
    } else {
        0
    }
}

/// One sweep at scale s. Returns the OR over the interior of D ^ (D >> 31)
/// (|D| in one's complement), whose top set bit the adaptive schedule reads.
pub fn step(n: usize, src: &[i32], dst: &mut [i32], s: i32, strict: bool, colour: Option<usize>) -> (u32, u64) {
    let mut or = 0u32;
    let mut moved = 0;
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            let k = i * n + j;
            if colour.is_some_and(|c| (i + j) % 2 != c) {
                dst[k] = src[k];
                continue;
            }
            let d = residual(n, src, k);
            or |= (d ^ (d >> 31)) as u32;
            let t = trit(d, s, strict);
            moved += (t != 0) as u64;
            dst[k] = src[k].wrapping_add(t.wrapping_mul(s));
        }
    }
    (or, moved)
}

/// Red-black sweep with one scale per `b` x `b` block: each block's scale
/// is `adaptive_scale` of that block's own OR of |D| over the previous sweep
/// (`ors`, updated in place; S0 on the first sweep). No global reduction.
pub fn sweep_blocks(n: usize, cur: &mut Vec<i32>, other: &mut Vec<i32>, b: usize, ors: &mut [u32], first: bool) {
    let nb = n.div_ceil(b);
    let scales: Vec<i32> = ors.iter().map(|&o| if first { S0 } else { adaptive_scale(o) }).collect();
    ors.fill(0);
    for c in 0..2 {
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                let k = i * n + j;
                if (i + j) % 2 != c {
                    other[k] = cur[k];
                    continue;
                }
                let blk = (i / b) * nb + j / b;
                let d = residual(n, cur, k);
                ors[blk] |= (d ^ (d >> 31)) as u32;
                let s = scales[blk];
                other[k] = cur[k].wrapping_add(trit(d, s, false).wrapping_mul(s));
            }
        }
        std::mem::swap(cur, other);
    }
}

/// The scale of each sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Schedule {
    /// s = max(1, s0 / factor^floor((k mod cycle) / m)), integer division
    /// repeated; `cycle = 0` means no restart.
    Geometric { s0: i32, factor: i32, m: u32, cycle: u32 },
    /// s_k = 2^max(0, msb(OR_{k-1}) - 2) with OR_{k-1} the OR of |D| of the
    /// previous sweep (`step`'s return), so max |Delta| is in [s, 2 s);
    /// s_0 = 2^(SHIFT - 2).
    Adaptive,
    /// Start at s0, divide by `factor` after a sweep in which no point moved.
    Stall { s0: i32, factor: i32 },
}

impl Schedule {
    /// Scale of sweep k given the previous sweep's OR of |D|.
    pub fn scale(&self, k: u64, prev_or: u32, prev_s: i32, prev_moved: u64) -> i32 {
        match *self {
            Schedule::Geometric { s0, factor, m, cycle } => {
                let kk = if cycle > 0 { k % cycle as u64 } else { k };
                let mut s = s0;
                for _ in 0..(kk / m as u64).min(64) {
                    s /= factor;
                }
                s.max(1)
            }
            Schedule::Adaptive => {
                if k == 0 {
                    1 << (SHIFT - 2)
                } else {
                    adaptive_scale(prev_or)
                }
            }
            Schedule::Stall { s0, factor } => {
                if k == 0 {
                    s0
                } else if prev_moved == 0 {
                    (prev_s / factor).max(1)
                } else {
                    prev_s
                }
            }
        }
    }
}

pub fn adaptive_scale(or: u32) -> i32 {
    let msb = 31 - or.max(1).leading_zeros() as i32;
    1 << (msb - 2).max(0)
}

/// Plain integer engine with a schedule.
pub struct IntEngine {
    pub n: usize,
    pub cur: Vec<i32>,
    other: Vec<i32>,
    pub iter: u64,
    pub schedule: Schedule,
    pub last_or: u32,
    pub last_scale: i32,
    pub last_moved: u64,
    pub strict: bool,
    pub red_black: bool,
}

impl IntEngine {
    pub fn new(p: &crate::Problem, schedule: Schedule) -> Self {
        Self::from_field(p.n, initial(p), schedule)
    }

    pub fn from_field(n: usize, u: Vec<i32>, schedule: Schedule) -> Self {
        IntEngine { n, other: u.clone(), cur: u, iter: 0, schedule, last_or: 0, last_scale: 0, last_moved: 0, strict: false, red_black: false }
    }

    pub fn run(&mut self, iters: usize) {
        for _ in 0..iters {
            let s = self.schedule.scale(self.iter, self.last_or, self.last_scale, self.last_moved);
            if self.red_black {
                let (o1, m1) = step(self.n, &self.cur, &mut self.other, s, self.strict, Some(0));
                std::mem::swap(&mut self.cur, &mut self.other);
                let (o2, m2) = step(self.n, &self.cur, &mut self.other, s, self.strict, Some(1));
                self.last_or = o1 | o2;
                self.last_moved = m1 + m2;
            } else {
                (self.last_or, self.last_moved) = step(self.n, &self.cur, &mut self.other, s, self.strict, None);
            }
            self.last_scale = s;
            std::mem::swap(&mut self.cur, &mut self.other);
            self.iter += 1;
        }
    }

    pub fn max_error(&self, p: &crate::Problem) -> f64 {
        p.max_error(|k| self.cur[k] as f64 / Q as f64)
    }
}

/// Word width of the accumulator.
pub const W: usize = 32;
/// Input index of operand `k` bit `j`: E, W, N, S, C (the point itself), then
/// the per-sweep constants A = -2s, B = 2s - 1, s and -s (uniform planes).
pub fn input(k: usize, j: usize) -> usize {
    k * W + j
}
pub const IN_E: usize = 0;
pub const IN_W: usize = 1;
pub const IN_N: usize = 2;
pub const IN_S: usize = 3;
pub const IN_C: usize = 4;
pub const IN_A: usize = 5;
pub const IN_B: usize = 6;
pub const IN_SP: usize = 7;
pub const IN_SN: usize = 8;
pub const INPUTS: usize = 9 * W;

/// Phases of the circuit's gates.
pub const PHASE_STENCIL: u8 = 0;
pub const PHASE_DECISION: u8 = 1;
pub const PHASE_ACCUMULATE: u8 = 2;
pub const PHASE_RESIDUAL: u8 = 3;
pub const PHASE_NAMES: [&str; 4] = ["stencil residual D", "ternary decision t", "accumulate u + s t", "residual check |D|"];

/// The per-sweep constant planes: A = -2s, B = 2s - 1, s, -s.
pub fn constants(s: i32) -> [u32; 4] {
    [s.wrapping_mul(-2) as u32, s.wrapping_mul(2).wrapping_sub(1) as u32, s as u32, s.wrapping_neg() as u32]
}

/// The update as gates (non-strict dead zone). Outputs 0..32: the new u;
/// outputs 32..63: D ^ (D >> 31), bits 0..31, for the residual check.
pub fn circuit() -> crate::circuit::Circuit {
    use crate::circuit::{Builder, Circuit, Sig};
    let mut g = Builder::default();
    let op = |k: usize| -> Vec<Sig> { (0..W).map(|j| Sig::In(input(k, j))).collect() };
    g.phase = PHASE_STENCIL;
    let s1 = g.add(&op(IN_E), &op(IN_W), W);
    let s2 = g.add(&op(IN_N), &op(IN_S), W);
    let sum = g.add(&s1, &s2, W);
    // D = sum - 4 C = sum + ~(C << 2) + 1.
    let c = op(IN_C);
    let not4c: Vec<Sig> = (0..W).map(|j| if j < 2 { Sig::One } else { g.not(c[j - 2]) }).collect();
    let d = g.add_carry(&sum, &not4c, W, Sig::One);
    g.phase = PHASE_DECISION;
    // t = +1 iff D - 2s >= 0; t = -1 iff D + 2s - 1 < 0: sign bits of two sums.
    let da = g.add(&d, &op(IN_A), W);
    let db = g.add(&d, &op(IN_B), W);
    let tp = g.not(da[W - 1]);
    let tn = db[W - 1];
    g.phase = PHASE_ACCUMULATE;
    let (sp, sn) = (op(IN_SP), op(IN_SN));
    let addend: Vec<Sig> = (0..W)
        .map(|j| {
            let a = g.and(tp, sp[j]);
            let b = g.and(tn, sn[j]);
            g.or(a, b)
        })
        .collect();
    let mut outputs = g.add(&c, &addend, W);
    g.phase = PHASE_RESIDUAL;
    for &dj in d.iter().take(W - 1) {
        let x = g.xor(dj, d[W - 1]);
        outputs.push(x);
    }
    Circuit::prune(W, false, INPUTS, g, outputs)
}

/// First sweep's scale.
pub const S0: i32 = 1 << (SHIFT - 2);

/// Lanes of colour `c` in row `r` of a 64-bit word: sites with (r + j) % 2 == c.
pub fn colour_mask64(r: usize, c: usize) -> u64 {
    if (r + c).is_multiple_of(2) {
        0x5555_5555_5555_5555
    } else {
        0xaaaa_aaaa_aaaa_aaaa
    }
}

/// Bit-sliced CPU engine: red-black ternary sweeps with the adaptive
/// (residual check) schedule, 32 bit-planes of u64 words.
pub struct Bitsliced {
    pub n: usize,
    pub wn: usize,
    pub iter: u64,
    pub last_or: u32,
    pub last_scale: i32,
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
        Bitsliced { n, wn: n / 64, iter: 0, last_or: 0, last_scale: 0, other: cur.clone(), cur, prog, scratch }
    }

    pub fn from_problem(p: &crate::Problem) -> Self {
        Self::new(p.n, &initial(p))
    }

    pub fn run(&mut self, iters: usize) {
        for _ in 0..iters {
            let s = Schedule::Adaptive.scale(self.iter, self.last_or, self.last_scale, 0);
            let a = self.half(s, 0);
            let b = self.half(s, 1);
            self.last_or = a | b;
            self.last_scale = s;
            self.iter += 1;
        }
    }

    /// Update the points of one colour; returns the OR of |D| over them.
    fn half(&mut self, s: i32, colour: usize) -> u32 {
        use crate::bitsliced::CHUNK;
        let (n, wn) = (self.n, self.wn);
        let plane = n * wn;
        let (src, dst) = (&self.cur, &mut self.other);
        let prog = &self.prog;
        let consts = constants(s);
        for (k, &cv) in consts.iter().enumerate() {
            for j in 0..W {
                let v = if (cv >> j) & 1 == 1 { u64::MAX } else { 0 };
                self.scratch[input(IN_A + k, j) * CHUNK..][..CHUNK].fill(v);
            }
        }
        let mut or = 0u32;
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
                    self.scratch[input(IN_C, j) * CHUNK + l] = c;
                }
            }
            prog.run(&mut self.scratch);
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
                    let new = self.scratch[prog.outputs[j] * CHUNK + l];
                    dst[j * plane + i] = (new & active) | (src[j * plane + i] & !active);
                }
                for j in 0..W - 1 {
                    if self.scratch[prog.outputs[W + j] * CHUNK + l] & active != 0 {
                        or |= 1 << j;
                    }
                }
            }
            i0 += len;
        }
        std::mem::swap(&mut self.cur, &mut self.other);
        or
    }

    pub fn words(&self) -> &[u64] {
        &self.cur
    }

    pub fn read(&self) -> Vec<i32> {
        crate::bitsliced::unpack(self.n, W, &self.cur).into_iter().map(|x| x as i32).collect()
    }
}
