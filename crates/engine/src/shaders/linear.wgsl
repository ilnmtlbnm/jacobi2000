// Naive one-thread-per-output-element matmul, the same dispatch shape as
// this project's other kernels (llm-web's lean crate, t0-fast): one
// workgroup per output tile, no shared-memory tiling. Right for this
// model's sizes (at most 256x512) and for a browser's 256-invocation cap;
// a tiled native variant is not worth the complexity at this scale.
//
// `w` is burn's own `nn::Linear` layout, `[in, out]` row-major (y = x @ W +
// b), NOT the `[out, in]` GGUF/PyTorch-`nn.Linear` layout lean's own
// linear.wgsl reads — the index arithmetic below reflects that.
// out[m,n] = dot(x[m,:], w[:,n]) + b[n].
struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.y;
    let n = gid.x;
    if (m >= dims.m || n >= dims.n) {
        return;
    }
    var acc: f32 = 0.0;
    let x_base = m * dims.k;
    for (var k: u32 = 0u; k < dims.k; k = k + 1u) {
        acc = acc + x[x_base + k] * w[k * dims.n + n];
    }
    out[m * dims.n + n] = acc + b[n];
}
