//! Plain references: the integer Jacobi step the circuit must match bit for
//! bit, and the f32 step. Row-major n x n, boundary rows and columns fixed.

/// `(E + W + N + S) >> 2`, or `(E + W + N + S + 2) >> 2` when `round`.
pub fn int_step(n: usize, src: &[u32], dst: &mut [u32], round: bool) {
    let add = if round { 2 } else { 0 };
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            let k = i * n + j;
            dst[k] = (src[k + 1] + src[k - 1] + src[k - n] + src[k + n] + add) >> 2;
        }
    }
}

pub fn int_run(n: usize, q: &mut Vec<u32>, iters: usize, round: bool) {
    let mut other = q.clone();
    for _ in 0..iters {
        int_step(n, q, &mut other, round);
        std::mem::swap(q, &mut other);
    }
}

/// `(((E + W) + N) + S) * 0.25`, the order the WGSL kernel uses.
pub fn f32_step(n: usize, src: &[f32], dst: &mut [f32]) {
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            let k = i * n + j;
            dst[k] = (((src[k + 1] + src[k - 1]) + src[k - n]) + src[k + n]) * 0.25;
        }
    }
}

/// Half of a red-black sweep: points with (i + j) % 2 == colour get
/// `f32_step`'s update from `src`, the others are copied.
pub fn f32_half(n: usize, src: &[f32], dst: &mut [f32], colour: usize) {
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            let k = i * n + j;
            dst[k] = if (i + j) % 2 == colour { (((src[k + 1] + src[k - 1]) + src[k - n]) + src[k + n]) * 0.25 } else { src[k] };
        }
    }
}

/// Red-black sweeps (Gauss-Seidel on the checkerboard): colour 0, then 1.
pub fn f32_rb_run(n: usize, f: &mut Vec<f32>, iters: usize) {
    let mut other = f.clone();
    for _ in 0..iters {
        for c in 0..2 {
            f32_half(n, f, &mut other, c);
            std::mem::swap(f, &mut other);
        }
    }
}

pub fn f32_run(n: usize, f: &mut Vec<f32>, iters: usize) {
    let mut other = f.clone();
    for _ in 0..iters {
        f32_step(n, f, &mut other);
        std::mem::swap(f, &mut other);
    }
}

/// f32 engine with its own ping-pong buffers, for the page and the bench.
pub struct F32Engine {
    pub n: usize,
    pub cur: Vec<f32>,
    other: Vec<f32>,
    pub iter: u64,
    /// Red-black sweeps instead of Jacobi.
    pub red_black: bool,
}

impl F32Engine {
    pub fn new(n: usize, f: Vec<f32>) -> Self {
        F32Engine { n, other: f.clone(), cur: f, iter: 0, red_black: false }
    }

    pub fn run(&mut self, iters: usize) {
        for _ in 0..iters {
            if self.red_black {
                for c in 0..2 {
                    f32_half(self.n, &self.cur, &mut self.other, c);
                    std::mem::swap(&mut self.cur, &mut self.other);
                }
            } else {
                f32_step(self.n, &self.cur, &mut self.other);
                std::mem::swap(&mut self.cur, &mut self.other);
            }
        }
        self.iter += iters as u64;
    }
}

/// The discrete solution of the problem in f64: red-black SOR with the
/// optimal factor 2 / (1 + sin(pi / (n - 1))), until the largest correction
/// of a sweep is below 1e-13 (at most 200000 sweeps).
pub fn discrete_solution(p: &crate::Problem) -> Vec<f64> {
    let n = p.n;
    let mut u: Vec<f64> = (0..n * n).map(|k| if p.is_boundary(k / n, k % n) { p.analytic(k / n, k % n) } else { 0.0 }).collect();
    let omega = 2.0 / (1.0 + (std::f64::consts::PI / (n - 1) as f64).sin());
    for _ in 0..200_000 {
        let mut change = 0f64;
        for c in 0..2 {
            for i in 1..n - 1 {
                for j in 1..n - 1 {
                    if (i + j) % 2 != c {
                        continue;
                    }
                    let k = i * n + j;
                    let d = (u[k + 1] + u[k - 1] + u[k - n] + u[k + n]) * 0.25 - u[k];
                    u[k] += omega * d;
                    change = change.max(d.abs());
                }
            }
        }
        if change < 1e-13 {
            break;
        }
    }
    u
}

/// Max |a - b| over interior points.
pub fn max_diff_interior(n: usize, a: &[f64], b: &[f64]) -> f64 {
    let mut m = 0f64;
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            m = m.max((a[i * n + j] - b[i * n + j]).abs());
        }
    }
    m
}
