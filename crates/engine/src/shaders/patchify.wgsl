// state[c, ny, nx] -> patches[token, patch_dim], token = ty*tokens_x+tx,
// feature d = c*patch*patch + py*patch + px — the same layout
// `jacobi2000::model::Jacobi2000::patchify`'s reshape/permute produces, and
// `cpu.rs::forward`'s patchify loop mirrors directly.
// `patch` is a WGSL reserved keyword (caught by wgpu's shader validation,
// not by any Rust-side check, since this struct's field names are never
// read back from the Dims uniform — only its byte layout matters there);
// the field is `patch_size` here for exactly that reason.
struct Dims { tokens_x: u32, patch_size: u32, channels: u32, nx: u32, ny: u32, patch_dim: u32, len: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read_write> patches: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dims.len) { return; }
    let t = idx / dims.patch_dim;
    let d = idx % dims.patch_dim;
    let ty = t / dims.tokens_x;
    let tx = t % dims.tokens_x;
    let pp = dims.patch_size * dims.patch_size;
    let c = d / pp;
    let rem = d % pp;
    let py = rem / dims.patch_size;
    let px = rem % dims.patch_size;
    let y = ty * dims.patch_size + py;
    let x = tx * dims.patch_size + px;
    patches[idx] = state[c * dims.ny * dims.nx + y * dims.nx + x];
}
