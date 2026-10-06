//! 2D Laplace on an N x N grid by Jacobi iteration, with the update done by
//! a b-bit gate circuit over bit-planes (b = 4, 8, 12, 16), next to an f32
//! reference. Engines: plain integer and f32 references, a bit-sliced CPU
//! engine on u64 words, and a wgpu engine whose WGSL is emitted from the same
//! circuit. Problem as in experiments/jacobi/accuracy.py (exp-jacobi):
//! u = x^2 - y^2 on [-1, 1]^2, values mapped to v = (u + 1) / 2 in [0, 1],
//! Dirichlet boundary from the analytic function, interior starting at zero.
//! A second problem, u = sin(pi x) sinh(pi y) / sinh(pi), is harmonic too but
//! not reproduced exactly by the 5-point stencil.

pub mod bitsliced;
pub mod circuit;
pub mod gas;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod int32;
pub mod kernels;
pub mod reference;
pub mod render;
pub mod rng;
pub mod stochastic;
pub mod ternary;
#[cfg(feature = "web")]
pub mod web;

/// Bit widths of the circuit panels.
pub const BITS: [usize; 4] = [4, 8, 12, 16];
/// Grid sizes are multiples of this (one u64 word per 64 points of a row).
pub const GRID_STEP: usize = 64;
pub const MAX_N: usize = 4096;

/// The grid size the engines use for a requested one: rounded up to a
/// multiple of 64, at least 64, at most 4096.
pub fn grid_size(requested: f64) -> Result<usize, String> {
    if !requested.is_finite() || requested < 1.0 {
        return Err(format!("grid size must be a positive number, got {requested}"));
    }
    if requested > MAX_N as f64 {
        return Err(format!("grid size {requested} is above the maximum of {MAX_N}"));
    }
    let n = (requested.ceil() as usize).div_ceil(GRID_STEP) * GRID_STEP;
    Ok(n)
}

pub fn qmax(bits: usize) -> u32 {
    ((1u64 << bits) - 1) as u32
}

/// The harmonic function u of the problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Case {
    /// u = x^2 - y^2: the 5-point stencil is exact, the discrete solution is u.
    Quadratic,
    /// u = sin(pi x) sinh(pi y) / sinh(pi): the discrete solution differs
    /// from u by the discretization error.
    Sinh,
}

impl Case {
    /// "quadratic" (also "", the default) or "sinh".
    pub fn parse(s: &str) -> Result<Case, String> {
        match s {
            "" | "quadratic" => Ok(Case::Quadratic),
            "sinh" => Ok(Case::Sinh),
            _ => Err(format!("unknown problem {s:?} (quadratic or sinh)")),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Case::Quadratic => "quadratic",
            Case::Sinh => "sinh",
        }
    }

    pub fn u(&self, x: f64, y: f64) -> f64 {
        match self {
            Case::Quadratic => x * x - y * y,
            Case::Sinh => (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sinh() / std::f64::consts::PI.sinh(),
        }
    }
}

/// The test problem on an n x n grid. Index `(i, j)`, stored at `i * n + j`,
/// is the point (x_i, y_j) with x and y on numpy's `linspace(-1, 1, n)`.
#[derive(Clone, Copy, Debug)]
pub struct Problem {
    pub n: usize,
    pub case: Case,
}

impl Problem {
    pub fn new(n: usize) -> Result<Problem, String> {
        Self::with_case(n, Case::Quadratic)
    }

    pub fn with_case(n: usize, case: Case) -> Result<Problem, String> {
        if n < GRID_STEP || !n.is_multiple_of(GRID_STEP) || n > MAX_N {
            return Err(format!("grid size {n} is not a multiple of {GRID_STEP} in [{GRID_STEP}, {MAX_N}]"));
        }
        Ok(Problem { n, case })
    }

    /// numpy.linspace(-1, 1, n)[i].
    pub fn coord(&self, i: usize) -> f64 {
        if i + 1 == self.n {
            1.0
        } else {
            i as f64 * (2.0 / (self.n - 1) as f64) + -1.0
        }
    }

    /// v = (u + 1) / 2.
    pub fn analytic(&self, i: usize, j: usize) -> f64 {
        (self.case.u(self.coord(i), self.coord(j)) + 1.0) / 2.0
    }

    pub fn is_boundary(&self, i: usize, j: usize) -> bool {
        i == 0 || j == 0 || i + 1 == self.n || j + 1 == self.n
    }

    /// Quantized start: boundary round(v * qmax) (ties to even, as numpy),
    /// interior zero.
    pub fn initial_q(&self, bits: usize) -> Vec<u32> {
        let q = qmax(bits) as f64;
        let n = self.n;
        (0..n * n).map(|k| if self.is_boundary(k / n, k % n) { (self.analytic(k / n, k % n) * q).round_ties_even() as u32 } else { 0 }).collect()
    }

    pub fn initial_f32(&self) -> Vec<f32> {
        let n = self.n;
        (0..n * n).map(|k| if self.is_boundary(k / n, k % n) { self.analytic(k / n, k % n) as f32 } else { 0.0 }).collect()
    }

    /// Max |v - analytic| over interior points, `v = value(k)`.
    pub fn max_error(&self, value: impl Fn(usize) -> f64) -> f64 {
        let n = self.n;
        let mut m = 0f64;
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                m = m.max((value(i * n + j) - self.analytic(i, j)).abs());
            }
        }
        m
    }

    pub fn max_error_q(&self, q: &[u32], bits: usize) -> f64 {
        let s = qmax(bits) as f64;
        self.max_error(|k| q[k] as f64 / s)
    }

    pub fn max_error_f32(&self, f: &[f32]) -> f64 {
        self.max_error(|k| f[k] as f64)
    }
}
