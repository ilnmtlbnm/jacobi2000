//! Ternary FHP: the FHP lattice with 0, 1 or 2 particles per channel (six
//! moving directions and the rest channel), the smallest integer lattice gas
//! on the FHP lattice (Boghosian, Yepez, Alexander, Margolus, "Integer
//! lattice gases", PRE 55, 4137 (1997), arXiv:comp-gas/9602001, with a
//! maximum of 2 instead of their 2^L - 1).
//!
//! A fluid state is s = sum_i n_i 3^i, n_i in {0, 1, 2}, i = 0..6 (E, NE, NW,
//! W, SW, SE, rest): 3^7 = 2187 states. The collision maps s to another
//! member of its (mass, momentum) class, each with probability 1/(q - 1)
//! (`Rule::Other`, the FHP-III choice of d'Humieres and Lallemand 1987
//! applied to whole classes) or to any member including itself with
//! probability 1/q (`Rule::Uniform`, the rule Boghosian et al. used for
//! their L = 2 runs). Both transition matrices are symmetric, so detailed
//! balance holds, hence semi-detailed balance, and the equilibrium is the
//! product measure P(n_i = k) ~ z_i^k with z_i = exp(mu + a . c_i)
//! (Boghosian et al., section IV, with the per-channel sum running to 2).

pub mod engine;
pub mod bitsliced;
pub mod collide_gen;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod measure;
pub mod synth;

use crate::{rules, CX2, CY2, NDIR};
use std::collections::HashMap;

pub const NCH: usize = 7;
pub const NSTATE: usize = 2187;
pub const POW3: [u16; NCH] = [1, 3, 9, 27, 81, 243, 729];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// Each other member of the class with probability 1/(q - 1).
    Other,
    /// Each member of the class, itself included, with probability 1/q.
    Uniform,
    /// Lowest viscosity: within each class the permutation that minimizes
    /// sum_s Q(s) . Q(pi(s)) (Q the traceless stress, two components), found
    /// by the assignment problem, averaged over the 12 symmetries of the
    /// lattice (each image with probability 1/12). Doubly stochastic, so
    /// semi-detailed balance holds.
    Optimal,
    /// `Optimal` with probability k/8, else `Other` (k in 1..8).
    Mix(u8),
    /// The bit-sliceable form of `Optimal`: a minimum-cost permutation pi
    /// that commutes with the 60-degree rotation (constrained on the
    /// zero-momentum classes, carried by rotation to the other classes),
    /// and its mirror image M pi M, chosen by one random chirality bit (the
    /// top bit of r). As FHP-III, every probability is exact with one bit.
    Chiral,
}

impl Rule {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "other" => Some(Rule::Other),
            "uniform" => Some(Rule::Uniform),
            "optimal" => Some(Rule::Optimal),
            "chiral" => Some(Rule::Chiral),
            _ => s
                .strip_prefix("mix")
                .and_then(|k| k.parse().ok())
                .filter(|k| (1..8).contains(k))
                .map(Rule::Mix),
        }
    }
}

pub fn counts(s: u16) -> [u8; NCH] {
    let mut s = s;
    std::array::from_fn(|_| {
        let d = (s % 3) as u8;
        s /= 3;
        d
    })
}

pub fn encode(n: &[u8; NCH]) -> u16 {
    (0..NCH).map(|i| n[i] as u16 * POW3[i]).sum()
}

pub fn mass(s: u16) -> u32 {
    counts(s).iter().map(|&c| c as u32).sum()
}

/// Momentum in integer units (2 px, 2 py / sqrt 3), as `rules::momentum`.
pub fn momentum(s: u16) -> (i32, i32) {
    let n = counts(s);
    (0..NDIR).fold((0, 0), |p, i| {
        (p.0 + n[i] as i32 * CX2[i], p.1 + n[i] as i32 * CY2[i])
    })
}

/// Rotate the moving counts by 60 degrees `k` times.
pub fn rotate(s: u16, k: usize) -> u16 {
    let n = counts(s);
    let mut o = [0u8; NCH];
    for i in 0..NDIR {
        o[(i + k) % NDIR] = n[i];
    }
    o[6] = n[6];
    encode(&o)
}

/// Reverse every moving count (bounce-back).
pub fn bounce(s: u16) -> u16 {
    rotate(s, 3)
}

/// Mirror about the x axis: NE <-> SE, NW <-> SW.
pub fn mirror(s: u16) -> u16 {
    let n = counts(s);
    encode(&[n[0], n[5], n[4], n[3], n[2], n[1], n[6]])
}

/// The collision classes and, per state, its class and position in it.
pub struct Table {
    pub members: Vec<u16>,
    pub start: Vec<u32>,
    pub len: Vec<u16>,
    pub class_of: Vec<u16>,
    pub pos: Vec<u16>,
    /// Rule::Optimal: the 12 symmetry images of the optimal permutation.
    pub opt: Vec<[u16; 12]>,
    /// Rule::Chiral: pi(s) and M pi(M s).
    pub chiral: Vec<[u16; 2]>,
}

impl Table {
    pub fn new() -> Self {
        let mut map: HashMap<(u32, i32, i32), Vec<u16>> = HashMap::new();
        let mut order = Vec::new();
        for s in 0..NSTATE as u16 {
            let (px, py) = momentum(s);
            let k = (mass(s), px, py);
            map.entry(k).or_insert_with(|| {
                order.push(k);
                Vec::new()
            });
            map.get_mut(&k).unwrap().push(s);
        }
        let (mut members, mut start, mut len) = (Vec::new(), Vec::new(), Vec::new());
        let (mut class_of, mut pos) = (vec![0u16; NSTATE], vec![0u16; NSTATE]);
        for (c, k) in order.iter().enumerate() {
            let g = &map[k];
            start.push(members.len() as u32);
            len.push(g.len() as u16);
            for (j, &s) in g.iter().enumerate() {
                class_of[s as usize] = c as u16;
                pos[s as usize] = j as u16;
                members.push(s);
            }
        }
        let mut t = Table {
            members,
            start,
            len,
            class_of,
            pos,
            opt: vec![],
            chiral: vec![],
        };
        t.opt = t.optimal();
        t.chiral = t.chiral_table();
        t
    }

    /// Rule::Chiral's table: a rotation-equivariant minimum-cost permutation
    /// and its mirror image.
    fn chiral_table(&self) -> Vec<[u16; 2]> {
        let mut pi = vec![u16::MAX; NSTATE];
        let dot = |a: (f64, f64), b: (f64, f64)| a.0 * b.0 + a.1 * b.1;
        for c in 0..self.len.len() {
            let g = &self.members[self.start[c] as usize..][..self.len[c] as usize];
            if pi[g[0] as usize] != u16::MAX {
                continue;
            }
            if momentum(g[0]) != (0, 0) {
                // Solve on this class, carry to its five rotations.
                let cost: Vec<Vec<f64>> = g
                    .iter()
                    .map(|&a| {
                        g.iter()
                            .map(|&b| dot(stress(a), stress(b)) + if a == b { 1e-6 } else { 0.0 })
                            .collect()
                    })
                    .collect();
                let asg = hungarian(&cost);
                for k in 0..6 {
                    for i in 0..g.len() {
                        pi[rotate(g[i], k) as usize] = rotate(g[asg[i]], k);
                    }
                }
                continue;
            }
            // Zero momentum: rotation orbits, matched orbit to orbit within each size.
            let mut orbits: Vec<(u16, usize)> = vec![];
            for &s in g {
                let o: Vec<u16> = (0..6).map(|k| rotate(s, k)).collect();
                let rep = *o.iter().min().unwrap();
                let size = (1..=6).find(|&m| rotate(s, m) == s).unwrap();
                if !orbits.iter().any(|x| x.0 == rep) {
                    orbits.push((rep, size));
                }
            }
            for size in [1, 2, 3, 6] {
                let os: Vec<u16> = orbits.iter().filter(|o| o.1 == size).map(|o| o.0).collect();
                if os.is_empty() {
                    continue;
                }
                // cost[a][b] = min over j of size * Q(a) . Q(R^j b); j recorded.
                let best = |a: u16, b: u16| -> (f64, usize) {
                    (0..size)
                        .map(|j| {
                            (
                                size as f64 * dot(stress(a), stress(rotate(b, j)))
                                    + if a == b && j == 0 { 1e-6 } else { 0.0 },
                                j,
                            )
                        })
                        .fold((f64::INFINITY, 0), |m, x| if x.0 < m.0 { x } else { m })
                };
                let cost: Vec<Vec<f64>> = os
                    .iter()
                    .map(|&a| os.iter().map(|&b| best(a, b).0).collect())
                    .collect();
                let asg = hungarian(&cost);
                for (ia, &a) in os.iter().enumerate() {
                    let b = os[asg[ia]];
                    let j = best(a, b).1;
                    for i in 0..size {
                        pi[rotate(a, i) as usize] = rotate(b, i + j);
                    }
                }
            }
        }
        (0..NSTATE as u16)
            .map(|s| [pi[s as usize], mirror(pi[mirror(s) as usize])])
            .collect()
    }

    /// The 12 symmetry images of the minimum-viscosity permutation.
    fn optimal(&self) -> Vec<[u16; 12]> {
        let q2 = stress;
        let mut pi = vec![0u16; NSTATE];
        for c in 0..self.len.len() {
            let g = &self.members[self.start[c] as usize..][..self.len[c] as usize];
            let n = g.len();
            // Cost Q(s).Q(s'), plus a small penalty on fixed points so that
            // ties prefer moving.
            let cost: Vec<Vec<f64>> = g
                .iter()
                .map(|&a| {
                    let qa = q2(a);
                    g.iter()
                        .map(|&b| {
                            let qb = q2(b);
                            qa.0 * qb.0 + qa.1 * qb.1 + if a == b { 1e-6 } else { 0.0 }
                        })
                        .collect()
                })
                .collect();
            let asg = hungarian(&cost);
            for i in 0..n {
                pi[g[i] as usize] = g[asg[i]];
            }
        }
        // Images g pi g^-1 under the 6 rotations and 6 mirrored rotations.
        let sym = |k: usize, s: u16| -> u16 {
            if k < 6 {
                rotate(s, k)
            } else {
                rotate(mirror(s), k - 6)
            }
        };
        let inv = |k: usize, s: u16| -> u16 {
            if k < 6 {
                rotate(s, 6 - k)
            } else {
                mirror(rotate(s, 12 - k))
            }
        };
        (0..NSTATE as u16)
            .map(|s| std::array::from_fn(|k| sym(k, pi[inv(k, s) as usize])))
            .collect()
    }

    pub fn class(&self, s: u16) -> &[u16] {
        let c = self.class_of[s as usize] as usize;
        &self.members[self.start[c] as usize..][..self.len[c] as usize]
    }

    /// Post-collision state of fluid state `s` for a uniform random word `r`.
    #[inline(always)]
    pub fn collide(&self, s: u16, r: u32, rule: Rule) -> u16 {
        let c = self.class_of[s as usize] as usize;
        let q = self.len[c] as u64;
        let k = match rule {
            Rule::Optimal => return self.opt[s as usize][((r as u64 * 12) >> 32) as usize],
            Rule::Chiral => return self.chiral[s as usize][(r >> 31) as usize],
            Rule::Mix(m) => {
                let r2 = crate::lowbias32(r ^ 0x5851_f42d);
                return self.collide(
                    s,
                    r,
                    if (r2 >> 29) < m as u32 {
                        Rule::Optimal
                    } else {
                        Rule::Other
                    },
                );
            }
            Rule::Uniform => ((r as u64 * q) >> 32) as u32,
            Rule::Other => {
                if q == 1 {
                    return s;
                }
                let k = ((r as u64 * (q - 1)) >> 32) as u32;
                k + (k >= self.pos[s as usize] as u32) as u32
            }
        };
        self.members[self.start[c] as usize + k as usize]
    }

    /// Transition probabilities out of `s`.
    pub fn transitions(&self, s: u16, rule: Rule) -> Vec<(u16, f64)> {
        let g = self.class(s);
        let q = g.len();
        match rule {
            Rule::Mix(m) => {
                let p = m as f64 / 8.0;
                let mut v: Vec<(u16, f64)> = self
                    .transitions(s, Rule::Optimal)
                    .into_iter()
                    .map(|(o, a)| (o, a * p))
                    .collect();
                for (o, a) in self.transitions(s, Rule::Other) {
                    match v.iter_mut().find(|x| x.0 == o) {
                        Some(x) => x.1 += a * (1.0 - p),
                        None => v.push((o, a * (1.0 - p))),
                    }
                }
                v
            }
            Rule::Chiral => {
                let [a, b] = self.chiral[s as usize];
                if a == b {
                    vec![(a, 1.0)]
                } else {
                    vec![(a, 0.5), (b, 0.5)]
                }
            }
            Rule::Optimal => {
                let mut v: Vec<(u16, f64)> = vec![];
                for &o in &self.opt[s as usize] {
                    match v.iter_mut().find(|x| x.0 == o) {
                        Some(x) => x.1 += 1.0 / 12.0,
                        None => v.push((o, 1.0 / 12.0)),
                    }
                }
                v
            }
            Rule::Uniform => g.iter().map(|&o| (o, 1.0 / q as f64)).collect(),
            Rule::Other if q == 1 => vec![(s, 1.0)],
            Rule::Other => g
                .iter()
                .filter(|&&o| o != s)
                .map(|&o| (o, 1.0 / (q - 1) as f64))
                .collect(),
        }
    }

    pub fn class_sizes(&self) -> Vec<usize> {
        self.len.iter().map(|&l| l as usize).collect()
    }
}

impl Default for Table {
    fn default() -> Self {
        Self::new()
    }
}

/// The traceless momentum-flux (stress) of a state, two components
/// (sum n_i (c_ix^2 - c_iy^2), sum n_i 2 c_ix c_iy).
pub fn stress(s: u16) -> (f64, f64) {
    let n = counts(s);
    (0..NDIR).fold((0.0, 0.0), |a, i| {
        let c = velocity(i);
        (
            a.0 + n[i] as f64 * (c.0 * c.0 - c.1 * c.1),
            a.1 + n[i] as f64 * 2.0 * c.0 * c.1,
        )
    })
}

/// Minimum-cost assignment (Hungarian algorithm, O(n^3)): row i -> column a[i].
pub fn hungarian(c: &[Vec<f64>]) -> Vec<usize> {
    let n = c.len();
    let inf = f64::INFINITY;
    let (mut u, mut v) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    let (mut p, mut way) = (vec![0usize; n + 1], vec![0usize; n + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let (mut delta, mut j1) = (inf, 0);
            for j in 1..=n {
                if !used[j] {
                    let cur = c[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut a = vec![0; n];
    for j in 1..=n {
        a[p[j] - 1] = j - 1;
    }
    a
}

/// Lattice velocity of channel i (rest = 0).
pub fn velocity(i: usize) -> (f64, f64) {
    if i < NDIR {
        (CX2[i] as f64 / 2.0, CY2[i] as f64 * 3f64.sqrt() / 2.0)
    } else {
        (0.0, 0.0)
    }
}

/// Moments of one channel of capacity `m` at fugacity z: mean, variance,
/// third cumulant.
pub fn channel_moments(m: u32, z: f64) -> (f64, f64, f64) {
    let (mut zs, mut m1, mut m2, mut m3) = (0.0, 0.0, 0.0, 0.0);
    for k in 0..=m {
        let w = z.powi(k as i32);
        let k = k as f64;
        zs += w;
        m1 += k * w;
        m2 += k * k * w;
        m3 += k * k * k * w;
    }
    let (m1, m2, m3) = (m1 / zs, m2 / zs, m3 / zs);
    (m1, m2 - m1 * m1, m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1)
}

/// Equilibrium mean occupations of the seven channels, capacity `m` for the
/// moving channels and `mr` for the rest channel (m = mr = 1 is FHP's
/// Fermi-Dirac distribution, 2 is ternary), at density `d` per channel
/// (mass 7 d) and velocity (ux, uy): P(n_i = k) ~ z_i^k with
/// z_i = exp(mu + a . c_i), mu and a solved by Newton's method.
pub fn equilibrium(m: u32, mr: u32, d: f64, ux: f64, uy: f64) -> [f64; NCH] {
    let cap = |i: usize| if i < NDIR { m } else { mr };
    let c: [(f64, f64); NCH] = std::array::from_fn(velocity);
    let mut x = [((d / (m as f64 - d).max(1e-9)).max(1e-9)).ln(), 0.0, 0.0];
    let eval = |x: &[f64; 3]| -> [(f64, f64); NCH] {
        std::array::from_fn(|i| {
            let (f, v, _) = channel_moments(cap(i), (x[0] + x[1] * c[i].0 + x[2] * c[i].1).exp());
            (f, v)
        })
    };
    for _ in 0..200 {
        let n = eval(&x);
        let mut f = [-7.0 * d, -7.0 * d * ux, -7.0 * d * uy];
        let mut j = [[0.0; 3]; 3];
        for i in 0..NCH {
            let e = [1.0, c[i].0, c[i].1];
            for a in 0..3 {
                f[a] += e[a] * n[i].0;
                for b in 0..3 {
                    j[a][b] += e[a] * e[b] * n[i].1;
                }
            }
        }
        let dx = solve3(j, f);
        let mut step = 1.0;
        // Damp large steps (far from the solution at high density).
        let big = dx.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if big > 2.0 {
            step = 2.0 / big;
        }
        for a in 0..3 {
            x[a] -= step * dx[a];
        }
        if big < 1e-14 {
            break;
        }
    }
    eval(&x).map(|p| p.0)
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    std::array::from_fn(|k| {
        let mut m = a;
        for r in 0..3 {
            m[r][k] = b[r];
        }
        det(m) / d
    })
}

/// Per-channel probabilities P(n_i = 0), P(n_i = 1) of a channel of
/// capacity 2 with mean `f`.
pub fn ternary_probs(f: f64) -> [f64; 2] {
    // Mean f = (z + 2 z^2) / (1 + z + z^2): solve for z.
    let z = if f <= 0.0 {
        0.0
    } else if (f - 1.0).abs() < 1e-15 {
        1.0
    } else {
        // (2 - f) z^2 + (1 - f) z - f = 0
        let (a, b, c) = (2.0 - f, 1.0 - f, -f);
        (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a)
    };
    let zs = 1.0 + z + z * z;
    [1.0 / zs, z / zs]
}

/// Zero-velocity fugacity at density `d` per channel, capacities m and mr.
pub fn fugacity(m: u32, mr: u32, d: f64) -> f64 {
    let (mut lo, mut hi) = (-60.0f64, 60.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let z = mid.exp();
        let rho = 6.0 * channel_moments(m, z).0 + channel_moments(mr, z).0;
        if rho < 7.0 * d {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).exp()
}

/// Galilean factor g(rho) at density `d` per channel (Boghosian et al.
/// eq. VI.17 with per-channel capacities): g = (A4 / A2^2) rho k3 / var^2
/// with the moving channels' variance and third cumulant, A2 = 3, A4 = 3/4.
pub fn g_theory(m: u32, mr: u32, d: f64) -> f64 {
    let z = fugacity(m, mr, d);
    let (_, v, k3) = channel_moments(m, z);
    (0.75 / 9.0) * 7.0 * d * k3 / (v * v)
}

/// Shear viscosity in the Boltzmann (molecular chaos) approximation at
/// density `d` per channel, for a collision given as transitions out of
/// every state (Boghosian et al., section VIII; for FHP it is Henon's
/// formula). `occ(s)` gives the seven channel counts of state s; capacities
/// are `m` for every channel. nu = (A4 / A2) (lambda - 1/2), with -1/lambda
/// the eigenvalue of the linearized collision operator on the shear mode
/// Q_i = c_ix c_iy.
pub fn nu_theory(
    m: u32,
    d: f64,
    nstates: usize,
    occ: impl Fn(usize) -> [u8; NCH],
    trans: impl Fn(usize) -> Vec<(usize, f64)>,
) -> f64 {
    let z = fugacity(m, m, d);
    let (f, var, _) = channel_moments(m, z);
    let zs: f64 = (0..=m).map(|k| z.powi(k as i32)).sum();
    let q: [f64; NCH] = std::array::from_fn(|i| {
        let c = velocity(i);
        c.0 * c.1
    });
    // Q . J . Q = sum_s P0(s) sum_s' A(s -> s') (Q.(s' - s)) (Q.(s - f)) / var
    let (mut qjq, mut qq) = (0.0, 0.0);
    for i in 0..NCH {
        qq += q[i] * q[i];
    }
    for s in 0..nstates {
        let n = occ(s);
        let p0 = n.iter().map(|&k| z.powi(k as i32) / zs).product::<f64>();
        if p0 == 0.0 {
            continue;
        }
        let qs: f64 = (0..NCH).map(|i| q[i] * (n[i] as f64 - f)).sum();
        let qn: f64 = (0..NCH).map(|i| q[i] * n[i] as f64).sum();
        for (o, a) in trans(s) {
            let no = occ(o);
            let qo: f64 = (0..NCH).map(|i| q[i] * no[i] as f64).sum();
            qjq += p0 * a * (qo - qn) * qs / var;
        }
    }
    let mu = qjq / qq;
    let lambda = -1.0 / mu;
    0.25 * (lambda - 0.5)
}

/// nu_theory for the ternary model with a rule.
pub fn nu_ternary(d: f64, rule: Rule) -> f64 {
    let t = Table::new();
    nu_theory(
        2,
        d,
        NSTATE,
        |s| counts(s as u16),
        |s| {
            t.transitions(s as u16, rule)
                .into_iter()
                .map(|(o, p)| (o as usize, p))
                .collect()
        },
    )
}

/// nu_theory for FHP-III (two chiralities, each with probability 1/2).
pub fn nu_fhp3(d: f64) -> f64 {
    let t = rules::table();
    nu_theory(
        1,
        d,
        128,
        |s| std::array::from_fn(|i| ((s >> i) & 1) as u8),
        |s| vec![(t[0][s] as usize, 0.5), (t[1][s] as usize, 0.5)],
    )
}
