// Bidirectional multi-head attention with an optional Chebyshev-stencil
// mask, one invocation per (head, query) pair — the naive "one workgroup
// per output row" shape this project uses elsewhere for small,
// non-causal attention (distinct from attn_prefill.wgsl's causal online
// softmax, which this model has no use for: every token attends to a
// small, fixed, non-causal neighbourhood or to everything).
// `q`/`k`/`v`: [tokens, dim], heads occupy contiguous `dim` ranges
// (`[head*hd, (head+1)*hd)`), matching `cpu.rs::attention`.
// `MAX_TOKENS` bounds the private `scores` array (WGSL needs a const
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

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
    let head = wg.x;
    let qi = wg.y;
    if (head >= dims.heads || qi >= dims.tokens) { return; }
    let hd = dims.head_dim;
    let q_base = qi * dims.dim + head * hd;

    var scores: array<f32, MAX_TOKENS>;
    var max_s: f32 = -1e30;
    for (var ki: u32 = 0u; ki < dims.tokens; ki = ki + 1u) {
        if (allowed(qi, ki)) {
            let k_base = ki * dims.dim + head * hd;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                dot = dot + q[q_base + d] * k[k_base + d];
            }
            let s = dot * dims.scale;
            scores[ki] = s;
            max_s = max(max_s, s);
        } else {
            scores[ki] = -1e30;
        }
    }

    var denom: f32 = 0.0;
    for (var ki: u32 = 0u; ki < dims.tokens; ki = ki + 1u) {
        let e = exp(scores[ki] - max_s);
        scores[ki] = e;
        denom = denom + e;
    }

    let out_base = qi * dims.dim + head * hd;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        var acc: f32 = 0.0;
        for (var ki: u32 = 0u; ki < dims.tokens; ki = ki + 1u) {
            if (scores[ki] != 0.0) {
                acc = acc + scores[ki] * v[ki * dims.dim + head * hd + d];
            }
        }
        out[out_base + d] = acc / denom;
    }
}
