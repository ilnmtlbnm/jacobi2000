// Packs 0/1 cells (one u32 each) into bytes, four per u32, for readback.
struct Dims { w: u32, h: u32, n: u32, flags: u32 }
@group(0) @binding(0) var<storage, read> cells: array<u32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let j = (wid.y * nwg.x + wid.x) * 64u + li;
    if (j * 4u >= dims.n) { return; }
    var v = 0u;
    for (var k = 0u; k < 4u; k++) {
        let c = j * 4u + k;
        if (c < dims.n) { v |= (cells[c] & 255u) << (8u * k); }
    }
    packed[j] = v;
}
