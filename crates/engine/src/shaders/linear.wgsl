// Tiled shared-memory matmul with 2x2 register blocking: one 16x16
// workgroup (256 invocations, the browser cap) computes a 32x32 output
// block, each thread owning the 2x2 sub-tile (m, m+16) x (n, n+16) held in
// registers. Adapted from the project's own t0-fast linear_tiled.wgsl,
// which reads PyTorch's [out, in] weights; here `w` is the
// layout `[in, out]`, row-major (y = x @ W + b), so the
// weight tile is loaded k-major with n contiguous (coalesced global reads)
// and read k-major with tx contiguous (conflict-free shared reads).
// out[m,n] = dot(x[m,:], w[:,n]) + b[n], then by `mode`: 0 store, 1 store
// GELU of it, 2 add it to the existing out[m,n] (residual connection, so
// the caller binds `out` to the residual stream).
const TM: u32 = 32u;
const TN: u32 = 32u;
const TK: u32 = 16u;
const THREADS: u32 = 256u;

struct Dims { m: u32, k: u32, n: u32, mode: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

var<workgroup> a_tile: array<array<f32, TK>, TM>;
var<workgroup> b_tile: array<array<f32, TN>, TK>;

// Exact erf-based GELU; `erf` has no WGSL builtin, so this is the standard
// Abramowitz-Stegun rational approximation (7.1.26, max error 1.5e-7),
// checked against `cpu.rs`'s `libm::erff` in the GPU-vs-CPU parity tests.
fn erf_approx(v: f32) -> f32 {
    let s = sign(v);
    let a = abs(v);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}

fn gelu(v: f32) -> f32 {
    return 0.5 * v * (1.0 + erf_approx(v * 0.7071067811865476));
}

fn store(idx: u32, acc: f32, bias: f32) {
    let v = acc + bias;
    if (dims.mode == 1u) {
        out[idx] = gelu(v);
    } else if (dims.mode == 2u) {
        out[idx] = out[idx] + v;
    } else {
        out[idx] = v;
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let m0 = wg.y * TM;
    let n0 = wg.x * TN;

    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    for (var kb: u32 = 0u; kb < dims.k; kb = kb + TK) {
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let mm = idx / TK;
            let kk = idx % TK;
            let m = m0 + mm;
            let ka = kb + kk;
            let valid = m < dims.m && ka < dims.k;
            a_tile[mm][kk] = select(0.0, x[select(0u, m * dims.k + ka, valid)], valid);
        }
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let kk = idx / TN;
            let nn = idx % TN;
            let n = n0 + nn;
            let bk = kb + kk;
            let valid = n < dims.n && bk < dims.k;
            b_tile[kk][nn] = select(0.0, w[select(0u, bk * dims.n + n, valid)], valid);
        }
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < TK; kk = kk + 1u) {
            let a0 = a_tile[ty][kk];
            let a1 = a_tile[ty + 16u][kk];
            let b0 = b_tile[kk][tx];
            let b1 = b_tile[kk][tx + 16u];
            acc00 = acc00 + a0 * b0;
            acc01 = acc01 + a0 * b1;
            acc10 = acc10 + a1 * b0;
            acc11 = acc11 + a1 * b1;
        }
        workgroupBarrier();
    }

    let mr0 = m0 + ty;
    let mr1 = m0 + ty + 16u;
    let nc0 = n0 + tx;
    let nc1 = n0 + tx + 16u;
    if (mr0 < dims.m && nc0 < dims.n) { store(mr0 * dims.n + nc0, acc00, b[nc0]); }
    if (mr0 < dims.m && nc1 < dims.n) { store(mr0 * dims.n + nc1, acc01, b[nc1]); }
    if (mr1 < dims.m && nc0 < dims.n) { store(mr1 * dims.n + nc0, acc10, b[nc0]); }
    if (mr1 < dims.m && nc1 < dims.n) { store(mr1 * dims.n + nc1, acc11, b[nc1]); }
}
