//! The b-bit Jacobi update `(uE + uW + uN + uS) >> 2` (or `(... + 2) >> 2`,
//! round to nearest) as a list of AND/OR/XOR/NOT gates over bit-planes.
//!
//! Mirrors experiments/jacobi/jacobi_gates.py (`ripple3`) and the
//! constructors of pytorch/bitslice_ops.py on the exp-jacobi branch: the same
//! constant folding in the gate constructors, two-half-adder ripple-carry
//! adders, widths b+1, b+1, b+2, and the shift by 2 as dropping the two low
//! bit-planes. Only gates reachable from the outputs are kept, which is what
//! `count_ops` counts there.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    And,
    Or,
    Xor,
    Not,
}

/// A signal: a constant, input bit `k` (operand `k / b`, bit `k % b`, the
/// operands in the order E, W, N, S), or the output of gate `i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sig {
    Zero,
    One,
    In(usize),
    Gate(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Gate {
    pub op: Op,
    pub a: Sig,
    /// Unused for `Not`.
    pub b: Sig,
}

/// Gate constructors with constant folding. `phase` labels the gates
/// pushed from now on (for per-part gate counts).
#[derive(Default)]
pub struct Builder {
    pub gates: Vec<Gate>,
    pub phases: Vec<u8>,
    pub phase: u8,
}

impl Builder {
    fn push(&mut self, op: Op, a: Sig, b: Sig) -> Sig {
        self.gates.push(Gate { op, a, b });
        self.phases.push(self.phase);
        Sig::Gate(self.gates.len() - 1)
    }

    pub fn not(&mut self, a: Sig) -> Sig {
        match a {
            Sig::Zero => Sig::One,
            Sig::One => Sig::Zero,
            Sig::Gate(i) if self.gates[i].op == Op::Not => self.gates[i].a,
            _ => self.push(Op::Not, a, a),
        }
    }

    pub fn and(&mut self, a: Sig, b: Sig) -> Sig {
        match (a, b) {
            (Sig::Zero, _) => Sig::Zero,
            (Sig::One, _) => b,
            (_, Sig::Zero | Sig::One) => self.and(b, a),
            _ => self.push(Op::And, a, b),
        }
    }

    pub fn or(&mut self, a: Sig, b: Sig) -> Sig {
        match (a, b) {
            (Sig::Zero, _) => b,
            (Sig::One, _) => Sig::One,
            (_, Sig::Zero | Sig::One) => self.or(b, a),
            _ => self.push(Op::Or, a, b),
        }
    }

    pub fn xor(&mut self, a: Sig, b: Sig) -> Sig {
        match (a, b) {
            (Sig::Zero, _) => b,
            (Sig::One, _) => self.not(b),
            (_, Sig::Zero | Sig::One) => self.xor(b, a),
            _ => self.push(Op::Xor, a, b),
        }
    }

    pub fn half_adder(&mut self, a: Sig, b: Sig) -> (Sig, Sig) {
        (self.xor(a, b), self.and(a, b))
    }

    /// Ripple-carry add, operands zero-extended and the result truncated to
    /// `width` bits (bitslice_ops.bv_add).
    pub fn add(&mut self, a: &[Sig], b: &[Sig], width: usize) -> Vec<Sig> {
        self.add_carry(a, b, width, Sig::Zero)
    }

    /// `add` with a carry in.
    pub fn add_carry(&mut self, a: &[Sig], b: &[Sig], width: usize, carry_in: Sig) -> Vec<Sig> {
        let bit = |v: &[Sig], i: usize| v.get(i).copied().unwrap_or(Sig::Zero);
        let mut out = Vec::with_capacity(width);
        let mut carry = carry_in;
        for i in 0..width {
            let (s1, c1) = self.half_adder(bit(a, i), bit(b, i));
            let (s2, c2) = self.half_adder(s1, carry);
            out.push(s2);
            carry = self.or(c1, c2);
        }
        out
    }

    /// `s ? y : x` as x ^ ((x ^ y) & s).
    pub fn mux(&mut self, s: Sig, x: Sig, y: Sig) -> Sig {
        let d = self.xor(x, y);
        let d = self.and(d, s);
        self.xor(x, d)
    }
}

/// The update circuit for one bit width and rounding mode. `gates` is in
/// topological order and holds only gates the outputs depend on.
#[derive(Clone, Debug)]
pub struct Circuit {
    pub bits: usize,
    pub round: bool,
    pub n_inputs: usize,
    pub gates: Vec<Gate>,
    /// Phase label of each gate (see `Builder::phase`).
    pub phases: Vec<u8>,
    /// Output bit-planes, LSB first.
    pub outputs: Vec<Sig>,
}

impl Circuit {
    /// `ripple3`: (E + W), (N + S), then their sum, plus 2 when rounding,
    /// then the shift by 2.
    pub fn jacobi(bits: usize, round: bool) -> Circuit {
        assert!((1..=30).contains(&bits));
        let mut g = Builder::default();
        let operand = |k: usize| -> Vec<Sig> { (0..bits).map(|i| Sig::In(k * bits + i)).collect() };
        let (e, w, n, s) = (operand(0), operand(1), operand(2), operand(3));
        let s1 = g.add(&e, &w, bits + 1);
        let s2 = g.add(&n, &s, bits + 1);
        let mut total = g.add(&s1, &s2, bits + 2);
        if round {
            // At most 4 (2^b - 1) + 2 < 2^(b+2): no width growth.
            let two: Vec<Sig> = (0..bits + 2).map(|i| if i == 1 { Sig::One } else { Sig::Zero }).collect();
            total = g.add(&total, &two, bits + 2);
        }
        let outputs = total[2..2 + bits].to_vec();
        Self::prune(bits, round, 4 * bits, g, outputs)
    }

    /// Stochastic rounding: `(E + W + N + S + r) >> 2` with r the 2-bit
    /// number on inputs 4b (low) and 4b + 1.
    pub fn jacobi_stochastic(bits: usize) -> Circuit {
        assert!((1..=30).contains(&bits));
        let mut g = Builder::default();
        let operand = |k: usize| -> Vec<Sig> { (0..bits).map(|i| Sig::In(k * bits + i)).collect() };
        let (e, w, n, s) = (operand(0), operand(1), operand(2), operand(3));
        let s1 = g.add(&e, &w, bits + 1);
        let s2 = g.add(&n, &s, bits + 1);
        let total = g.add(&s1, &s2, bits + 2);
        // At most 4 (2^b - 1) + 3 < 2^(b+2).
        let total = g.add(&total, &[Sig::In(4 * bits), Sig::In(4 * bits + 1)], bits + 2);
        let outputs = total[2..2 + bits].to_vec();
        Self::prune(bits, false, 4 * bits + 2, g, outputs)
    }

    /// Keep the gates reachable from the outputs, renumbered in order.
    pub fn prune(bits: usize, round: bool, n_inputs: usize, b: Builder, outputs: Vec<Sig>) -> Circuit {
        let (all, all_phases) = (b.gates, b.phases);
        let mut live = vec![false; all.len()];
        let mut stack: Vec<usize> = outputs.iter().filter_map(|s| if let Sig::Gate(i) = s { Some(*i) } else { None }).collect();
        while let Some(i) = stack.pop() {
            if live[i] {
                continue;
            }
            live[i] = true;
            let g = all[i];
            let args: &[Sig] = if g.op == Op::Not { &[g.a] } else { &[g.a, g.b] };
            for a in args {
                if let Sig::Gate(j) = a {
                    stack.push(*j);
                }
            }
        }
        let mut new_index = vec![usize::MAX; all.len()];
        let mut gates = Vec::new();
        let mut phases = Vec::new();
        let remap = |s: Sig, new_index: &[usize]| match s {
            Sig::Gate(i) => Sig::Gate(new_index[i]),
            s => s,
        };
        for (i, g) in all.iter().enumerate() {
            if live[i] {
                new_index[i] = gates.len();
                gates.push(Gate { op: g.op, a: remap(g.a, &new_index), b: remap(g.b, &new_index) });
                phases.push(all_phases[i]);
            }
        }
        let outputs = outputs.iter().map(|&s| remap(s, &new_index)).collect();
        Circuit { bits, round, n_inputs, gates, phases, outputs }
    }

    pub fn inputs(&self) -> usize {
        self.n_inputs
    }

    /// Gate count per phase label, labels 0..=max.
    pub fn phase_counts(&self) -> Vec<usize> {
        let m = self.phases.iter().copied().max().unwrap_or(0) as usize;
        let mut c = vec![0; m + 1];
        for &p in &self.phases {
            c[p as usize] += 1;
        }
        c
    }

    /// One WGSL `let` per gate (`g{i}`), inputs named by `input(k)`.
    pub fn wgsl_gates(&self, input: impl Fn(usize) -> String) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        for (i, g) in self.gates.iter().enumerate() {
            let e = match g.op {
                Op::And => format!("{} & {}", self.wgsl_sig(g.a, &input), self.wgsl_sig(g.b, &input)),
                Op::Or => format!("{} | {}", self.wgsl_sig(g.a, &input), self.wgsl_sig(g.b, &input)),
                Op::Xor => format!("{} ^ {}", self.wgsl_sig(g.a, &input), self.wgsl_sig(g.b, &input)),
                Op::Not => format!("~{}", self.wgsl_sig(g.a, &input)),
            };
            let _ = writeln!(s, "    let g{i} = {e};");
        }
        s
    }

    pub fn wgsl_sig(&self, sig: Sig, input: &impl Fn(usize) -> String) -> String {
        match sig {
            Sig::Zero => "0u".to_string(),
            Sig::One => "0xffffffffu".to_string(),
            Sig::In(k) => input(k),
            Sig::Gate(i) => format!("g{i}"),
        }
    }

    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// Gates per kind: AND, OR, XOR, NOT.
    pub fn counts(&self) -> [usize; 4] {
        let mut c = [0; 4];
        for g in &self.gates {
            c[g.op as usize] += 1;
        }
        c
    }

    /// Evaluate on whole words (64 lanes each); `inputs[k]` is input bit `k`.
    pub fn eval(&self, inputs: &[u64]) -> Vec<u64> {
        assert_eq!(inputs.len(), self.inputs());
        let mut v = vec![0u64; self.gates.len()];
        let get = |s: Sig, v: &[u64]| match s {
            Sig::Zero => 0,
            Sig::One => u64::MAX,
            Sig::In(k) => inputs[k],
            Sig::Gate(i) => v[i],
        };
        for (i, g) in self.gates.iter().enumerate() {
            let (a, b) = (get(g.a, &v), get(g.b, &v));
            v[i] = match g.op {
                Op::And => a & b,
                Op::Or => a | b,
                Op::Xor => a ^ b,
                Op::Not => !a,
            };
        }
        self.outputs.iter().map(|&s| get(s, &v)).collect()
    }

    /// WGSL compute kernel: one thread per 32-point word of an interior row,
    /// b bit-planes in one buffer (plane k at `k * params.plane`), the two
    /// edge columns and the first and last rows held fixed (Dirichlet).
    pub fn to_wgsl(&self) -> String {
        use std::fmt::Write;
        let b = self.bits;
        let mut s = String::new();
        let _ = writeln!(s, "// b = {b}, {}: {} gates. Generated by jacobi::circuit::Circuit::to_wgsl.", if self.round { "round to nearest" } else { "truncating" }, self.gate_count());
        s.push_str(WGSL_HEAD);
        for k in 0..b {
            let _ = writeln!(s, "    let c{k} = src[{k}u * p.plane + i];");
            let _ = writeln!(s, "    let x{} = (c{k} >> 1u) | (src[{k}u * p.plane + i + 1u] << 31u);", k);
            let _ = writeln!(s, "    let x{} = (c{k} << 1u) | (src[{k}u * p.plane + i - 1u] >> 31u);", b + k);
            let _ = writeln!(s, "    let x{} = src[{k}u * p.plane + i - p.wpr];", 2 * b + k);
            let _ = writeln!(s, "    let x{} = src[{k}u * p.plane + i + p.wpr];", 3 * b + k);
        }
        let name = |sig: Sig| match sig {
            Sig::Zero => "0u".to_string(),
            Sig::One => "0xffffffffu".to_string(),
            Sig::In(k) => format!("x{k}"),
            Sig::Gate(i) => format!("g{i}"),
        };
        for (i, g) in self.gates.iter().enumerate() {
            let e = match g.op {
                Op::And => format!("{} & {}", name(g.a), name(g.b)),
                Op::Or => format!("{} | {}", name(g.a), name(g.b)),
                Op::Xor => format!("{} ^ {}", name(g.a), name(g.b)),
                Op::Not => format!("~{}", name(g.a)),
            };
            let _ = writeln!(s, "    let g{i} = {e};");
        }
        for (k, &o) in self.outputs.iter().enumerate() {
            let _ = writeln!(s, "    dst[{k}u * p.plane + i] = ({} & ~keep) | (c{k} & keep);", name(o));
        }
        s.push_str("}\n");
        s
    }
}

const WGSL_HEAD: &str = "
struct Params {
    wpr: u32,
    h: u32,
    plane: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= p.wpr || row == 0u || row + 1u >= p.h) { return; }
    let i = row * p.wpr + col;
    // Lanes on the left and right boundary columns keep their value.
    let keep = select(0u, 1u, col == 0u) | select(0u, 0x80000000u, col + 1u == p.wpr);
";

/// The f32 reference kernel, same boundary handling and summation order as
/// `reference::f32_step`.
pub const F32_WGSL: &str = "
struct Params {
    wpr: u32,
    h: u32,
    plane: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = p.h;
    let x = gid.x;
    let y = gid.y;
    if (x == 0u || y == 0u || x + 1u >= n || y + 1u >= n) { return; }
    let i = y * n + x;
    dst[i] = (((src[i + 1u] + src[i - 1u]) + src[i - n]) + src[i + n]) * 0.25;
}
";
