// Bidirectional multi-head attention with an optional Chebyshev-stencil
// mask. One workgroup of 64 threads per (head, query) pair: the threads
// stride over the keys for the scores, the max and the sum are
// shared-memory tree reductions (the rmsnorm/layernorm pattern of this
// project's kernels), and then the threads stride over the head_dim
// outputs. Non-causal, so no online softmax is needed (unlike lean's
// attn_prefill.wgsl): every token attends to a small fixed neighbourhood
// or to everything, and the whole score row fits in workgroup memory.
// `q`/`k`/`v`: [tokens, dim], heads occupy contiguous `dim` ranges
// (`[head*hd, (head+1)*hd)`), matching `cpu.rs::attention`.
// `MAX_TOKENS` bounds the workgroup `scores` array (WGSL needs a const
// length); 1024 covers this model's 256-token (64x64, patch 4) grid with
// headroom for a finer patch.
const MAX_TOKENS: u32 = 1024u;

struct Dims {
    tokens: u32, dim: u32, heads: u32, head_dim: u32,
    tokens_x: u32, mask_kind: u32, radius: u32, dilation: u32,
    scale: f32, _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

fn allowed(qi: u32, ki: u32) -> bool {
    if (dims.mask_kind == 0u) { return true; }
    let qx = i32(qi % dims.tokens_x);
    let qy = i32(qi / dims.tokens_x);
    let kx = i32(ki % dims.tokens_x);
    let ky = i32(ki / dims.tokens_x);
    let d = i32(dims.dilation);
    let dx = kx - qx;
    let dy = ky - qy;
    if (dx % d != 0 || dy % d != 0) { return false; }
    let r = i32(dims.radius);
    return abs(dx / d) <= r && abs(dy / d) <= r;
}

const WG: u32 = 64u;
var<workgroup> scores: array<f32, MAX_TOKENS>;
var<workgroup> red: array<f32, WG>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let head = wg.x;
    let qi = wg.y;
    let hd = dims.head_dim;
    let q_base = qi * dims.dim + head * hd;

    var local_max: f32 = -1e30;
    for (var ki: u32 = tid; ki < dims.tokens; ki = ki + WG) {
        var s: f32 = -1e30;
        if (allowed(qi, ki)) {
            let k_base = ki * dims.dim + head * hd;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                dot = dot + q[q_base + d] * k[k_base + d];
            }
            s = dot * dims.scale;
        }
        scores[ki] = s;
        local_max = max(local_max, s);
    }
    red[tid] = local_max;
    workgroupBarrier();
    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) { red[tid] = max(red[tid], red[tid + stride]); }
        workgroupBarrier();
    }
    let max_s = red[0];
    workgroupBarrier();

    var local_sum: f32 = 0.0;
    for (var ki: u32 = tid; ki < dims.tokens; ki = ki + WG) {
        let e = exp(scores[ki] - max_s);
        scores[ki] = e;
        local_sum = local_sum + e;
    }
    red[tid] = local_sum;
    workgroupBarrier();
    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) { red[tid] = red[tid] + red[tid + stride]; }
        workgroupBarrier();
    }
    let denom = red[0];

    for (var d: u32 = tid; d < hd; d = d + WG) {
        var acc: f32 = 0.0;
        for (var ki: u32 = 0u; ki < dims.tokens; ki = ki + 1u) {
            let p = scores[ki];
            if (p != 0.0) {
                acc = acc + p * v[ki * dims.dim + head * hd + d];
            }
        }
        out[q_base + d] = acc / denom;
    }
}
