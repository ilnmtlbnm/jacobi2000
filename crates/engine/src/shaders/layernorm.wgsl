// One workgroup per row, 256-thread shared-memory tree reduction — the same
// design as this project's rmsnorm kernel (lean, llm-wasm), extended to a
// full layernorm (subtract the mean too, not just normalize by the second
// moment) since the model uses LayerNorm (biased variance, `eps` added
// before the square root), not RMSNorm.
// out[r,:] = (x[r,:] - mean(x[r,:])) / sqrt(var(x[r,:]) + eps) * gamma + beta.
struct Dims { rows: u32, dim: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

const WG_SIZE: u32 = 256u;
var<workgroup> partial: array<f32, WG_SIZE>;
var<workgroup> row_mean: f32;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = wg_id.x;
    let valid = row < dims.rows;
    let tid = local_id.x;
    let row_base = row * dims.dim;

    var acc: f32 = 0.0;
    if (valid) {
        var i: u32 = tid;
        loop {
            if (i >= dims.dim) { break; }
            acc += x[row_base + i];
            i = i + WG_SIZE;
        }
    }
    partial[tid] = acc;
    workgroupBarrier();
    var stride: u32 = WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        row_mean = partial[0] / f32(dims.dim);
    }
    workgroupBarrier();
    let mean = row_mean;

    var vacc: f32 = 0.0;
    if (valid) {
        var i: u32 = tid;
        loop {
            if (i >= dims.dim) { break; }
            let d = x[row_base + i] - mean;
            vacc += d * d;
            i = i + WG_SIZE;
        }
    }
    partial[tid] = vacc;
    workgroupBarrier();
    stride = WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (valid) {
        let inv_std = 1.0 / sqrt(partial[0] / f32(dims.dim) + dims.eps);
        var j: u32 = tid;
        loop {
            if (j >= dims.dim) { break; }
            out[row_base + j] = (x[row_base + j] - mean) * inv_std * gamma[j] + beta[j];
            j = j + WG_SIZE;
        }
    }
}
