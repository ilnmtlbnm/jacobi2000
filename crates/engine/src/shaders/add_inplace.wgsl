// a[i] += b[i], length `len`. Generic elementwise op, a generic
// elementwise kernel with no model content.
struct Dims { len: u32, _p0: u32, _p1: u32, _p2: u32 };

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.len) { return; }
    a[i] = a[i] + b[i];
}
