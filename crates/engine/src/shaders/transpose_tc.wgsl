// [tokens, dim] -> [dim, tokens], matching CellHead's
// `x.reshape([b, ty, tx, d]).permute([0, 3, 1, 2])` (the `[dim, ty*tx]`
// result is addressed as `[dim, ty, tx]` by every kernel downstream, same
// as `cpu.rs`'s explicit transpose loop).
struct Dims { tokens: u32, dim: u32, len: u32, _p0: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dims.len) { return; }
    let t = idx / dims.dim;
    let c = idx % dims.dim;
    out[c * dims.tokens + t] = x[idx];
}
