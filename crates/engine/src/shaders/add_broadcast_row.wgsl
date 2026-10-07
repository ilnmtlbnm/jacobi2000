// x[r, :] += row[:] for every row r, matching Jacobi2000::forward's
// `x + m.reshape([batch, 1, dim])` broadcast add (the meta conditioning
// vector added to every token).
struct Dims { rows: u32, dim: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> row: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let len = dims.rows * dims.dim;
    if (i >= len) { return; }
    x[i] = x[i] + row[i % dims.dim];
}
