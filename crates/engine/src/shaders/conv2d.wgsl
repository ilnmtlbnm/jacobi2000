// Zero-padded 2D cross-correlation (PyTorch/Burn `conv2d` convention, no
// kernel flip), one invocation per output element — the naive kernel shape
// this project uses for small per-cell ops; CellHead's images are at most
// 128x64x64, well within a single naive dispatch. `input: [cin,h,w]`,
// `weight: [cout,cin,k,k]`, `bias: [cout]`, `pad = k/2` (odd kernels only),
// `out: [cout,h,w]`. `MAX_K` bounds the (unused beyond k=3) const loop
// trip count; this model only ever calls with k=1 or k=3.
struct Dims { cin: u32, h: u32, w: u32, cout: u32, k: u32, len: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dims.len) { return; }
    let plane = dims.h * dims.w;
    let co = idx / plane;
    let rem = idx % plane;
    let y = i32(rem / dims.w);
    let x = i32(rem % dims.w);
    let pad = i32(dims.k / 2u);

    var acc: f32 = bias[co];
    let wbase = co * dims.cin * dims.k * dims.k;
    for (var ci: u32 = 0u; ci < dims.cin; ci = ci + 1u) {
        let wci = wbase + ci * dims.k * dims.k;
        for (var ky: u32 = 0u; ky < dims.k; ky = ky + 1u) {
            let iy = y + i32(ky) - pad;
            if (iy < 0 || iy >= i32(dims.h)) { continue; }
            for (var kx: u32 = 0u; kx < dims.k; kx = kx + 1u) {
                let ix = x + i32(kx) - pad;
                if (ix < 0 || ix >= i32(dims.w)) { continue; }
                acc = acc + input[ci * plane + u32(iy) * dims.w + u32(ix)] * weight[wci + ky * dims.k + kx];
            }
        }
    }
    out[idx] = acc;
}
