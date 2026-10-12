// Zero-padded 2D cross-correlation (PyTorch `conv2d` convention, no
// kernel flip) as an implicit GEMM: out[co, p] = sum_kk w[co, kk] *
// patch[kk, p], with kk = (ci, ky, kx) flattened exactly as the weight
// tensor is laid out (`weight: [cout,cin,k,k]`, so a weight row is already
// contiguous in kk) and patch[kk, p] gathered from `input: [cin,h,w]` with
// zero padding `pad = k/2` (odd kernels only). Same 32x32 tile, 16x16
// workgroup, 2x2 register blocking as linear.wgsl (itself from the
// project's t0-fast linear_tiled.wgsl). `bias: [cout]`, `out: [cout,h,w]`.
// `gelu` = 1 applies GELU to the result before the store.
// Dispatch: (ceil(h*w / 32), ceil(cout / 32)).
const TM: u32 = 32u;
const TN: u32 = 32u;
const TK: u32 = 16u;
const THREADS: u32 = 256u;

struct Dims { cin: u32, h: u32, w: u32, cout: u32, k: u32, len: u32, gelu: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
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
    out[idx] = select(v, gelu(v), dims.gelu == 1u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let plane = dims.h * dims.w;
    let kk2 = dims.k * dims.k;
    let ktot = dims.cin * kk2;
    let pad = i32(dims.k / 2u);
    let co0 = wg.y * TM;
    let p0 = wg.x * TN;

    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    for (var kb: u32 = 0u; kb < ktot; kb = kb + TK) {
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let mm = idx / TK;
            let kk = idx % TK;
            let co = co0 + mm;
            let ka = kb + kk;
            let valid = co < dims.cout && ka < ktot;
            a_tile[mm][kk] = select(0.0, weight[select(0u, co * ktot + ka, valid)], valid);
        }
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let idx = tid + i * THREADS;
            let kk = idx / TN;
            let pp = idx % TN;
            let p = p0 + pp;
            let ka = kb + kk;
            var v: f32 = 0.0;
            if (p < plane && ka < ktot) {
                let ci = ka / kk2;
                let r = ka % kk2;
                let iy = i32(p / dims.w) + i32(r / dims.k) - pad;
                let ix = i32(p % dims.w) + i32(r % dims.k) - pad;
                if (iy >= 0 && iy < i32(dims.h) && ix >= 0 && ix < i32(dims.w)) {
                    v = input[ci * plane + u32(iy) * dims.w + u32(ix)];
                }
            }
            b_tile[kk][pp] = v;
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

    let cr0 = co0 + ty;
    let cr1 = co0 + ty + 16u;
    let pc0 = p0 + tx;
    let pc1 = p0 + tx + 16u;
    if (cr0 < dims.cout && pc0 < plane) { store(cr0 * plane + pc0, acc00, bias[cr0]); }
    if (cr0 < dims.cout && pc1 < plane) { store(cr0 * plane + pc1, acc01, bias[cr0]); }
    if (cr1 < dims.cout && pc0 < plane) { store(cr1 * plane + pc0, acc10, bias[cr1]); }
    if (cr1 < dims.cout && pc1 < plane) { store(cr1 * plane + pc1, acc11, bias[cr1]); }
}
