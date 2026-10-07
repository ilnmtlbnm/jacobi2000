// Exact erf-based GELU, elementwise, same 1-D dispatch shape as this
// project's add_inplace/silu_mul kernels (lean, t0-fast). `erf` has no WGSL
// builtin, so it's the standard Abramowitz-Stegun rational approximation
// (7.1.26, max error 1.5e-7, matched against `cpu.rs`'s `libm::erff` in
// the GPU-vs-CPU parity test rather than assumed).
struct Dims { len: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> dims: Dims;

fn erf_approx(v: f32) -> f32 {
    let s = sign(v);
    let a = abs(v);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.len) { return; }
    let v = x[i];
    x[i] = 0.5 * v * (1.0 + erf_approx(v * 0.7071067811865476));
}
