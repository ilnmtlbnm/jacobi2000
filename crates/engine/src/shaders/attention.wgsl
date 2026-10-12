// Bidirectional multi-head attention with an optional Chebyshev-stencil
// mask. One workgroup of 64 threads per (head, query) pair. Only the
// candidate keys are visited: all `tokens` keys for the full mask, or the
// (2r+1)^2 stencil offsets (dilated, clipped to the token grid) for the
// stencil mask, so the work per query is the neighbourhood, not the grid.
// Threads stride over the candidates for the scores, the max and the sum
// are shared-memory tree reductions (the layernorm pattern of this
// project's kernels), then threads stride over the head_dim outputs.
// Non-causal, so no online softmax is needed (unlike lean's
// attn_prefill.wgsl): the whole score row fits in workgroup memory.
// `q`/`k`/`v`: [tokens, dim], heads occupy contiguous `dim` ranges
// (`[head*hd, (head+1)*hd)`), matching `cpu.rs::attention`.
// `MAX_CAND` bounds the workgroup arrays (WGSL needs a const length); 1024
// covers this model's 256-token (64x64, patch 4) grid with headroom for a
// finer patch.
const MAX_CAND: u32 = 1024u;

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

const WG: u32 = 64u;
var<workgroup> scores: array<f32, MAX_CAND>;
var<workgroup> keys: array<u32, MAX_CAND>;
var<workgroup> red: array<f32, WG>;

// Key index of candidate `c` for query `qi`, or `dims.tokens` when the
// candidate falls outside the grid.
fn candidate(qi: u32, c: u32) -> u32 {
    if (dims.mask_kind == 0u) { return c; }
    let side = 2u * dims.radius + 1u;
    let tokens_y = dims.tokens / dims.tokens_x;
    let kx = i32(qi % dims.tokens_x) + (i32(c % side) - i32(dims.radius)) * i32(dims.dilation);
    let ky = i32(qi / dims.tokens_x) + (i32(c / side) - i32(dims.radius)) * i32(dims.dilation);
    if (kx < 0 || ky < 0 || kx >= i32(dims.tokens_x) || ky >= i32(tokens_y)) { return dims.tokens; }
    return u32(ky) * dims.tokens_x + u32(kx);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let head = wg.x;
    let qi = wg.y;
    let hd = dims.head_dim;
    let q_base = qi * dims.dim + head * hd;
    let side = 2u * dims.radius + 1u;
    let n_cand = select(side * side, dims.tokens, dims.mask_kind == 0u);

    var local_max: f32 = -1e30;
    for (var c: u32 = tid; c < n_cand; c = c + WG) {
        let ki = candidate(qi, c);
        var s: f32 = -1e30;
        if (ki < dims.tokens) {
            let k_base = ki * dims.dim + head * hd;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                dot = dot + q[q_base + d] * k[k_base + d];
            }
            s = dot * dims.scale;
        }
        keys[c] = ki;
        scores[c] = s;
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
    for (var c: u32 = tid; c < n_cand; c = c + WG) {
        let e = select(0.0, exp(scores[c] - max_s), keys[c] < dims.tokens);
        scores[c] = e;
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
        for (var c: u32 = 0u; c < n_cand; c = c + 1u) {
            let p = scores[c];
            if (p != 0.0) {
                acc = acc + p * v[keys[c] * dims.dim + head * hd + d];
            }
        }
        out[q_base + d] = acc / denom;
    }
}
