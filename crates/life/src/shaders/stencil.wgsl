// Stencil model, one thread per cell. Pre-LN block: attention over the
// cell's 9 taps on the torus (NW, N, NE, W, E, SW, S, SE, then the cell
// itself), then the feed-forward, then the per-cell head. A tap's key and
// value depend only on that cell's state, 0 or 1, so they are read from the
// two-row table at `O_KV` (computed once per model, `cpu::stencil_kv`), not
// recomputed per cell.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = (wid.y * nwg.x + wid.x) * 64u + li;
    if (i >= dims.n) { return; }
    let nb = neighbors(i);
    var x: array<vec4<f32>, D4>;
    let cv = select(0.0, 1.0, cells[i] != 0u);
    for (var e = 0u; e < D4; e++) { x[e] = wt[O_EMBED_WEIGHT + e] * cv + wt[O_EMBED_BIAS + e]; }
    var hh = x;
    layer_norm(&hh, O_BLOCKS_0_NORM1_WEIGHT, O_BLOCKS_0_NORM1_BIAS);
    var q = lin_D4_D4(&hh, O_BLOCKS_0_Q_WEIGHT, O_BLOCKS_0_Q_BIAS);
    var slot: array<u32, 9>;
    for (var t = 0u; t < 9u; t++) { slot[t] = O_KV + select(0u, 1u, cells[nb[t]] != 0u) * 2u * D4; }
    var att: array<vec4<f32>, D4>;
    for (var hd = 0u; hd < H; hd++) {
        let o0 = hd * DK4;
        var sc: array<f32, 9>;
        for (var t = 0u; t < 9u; t++) {
            var s = vec4<f32>(0.0);
            for (var e = 0u; e < DK4; e++) { s += q[o0 + e] * wt[slot[t] + o0 + e]; }
            sc[t] = (s.x + s.y + s.z + s.w) / SCALE;
        }
        softmax9(&sc);
        for (var e = 0u; e < DK4; e++) {
            var a = vec4<f32>(0.0);
            for (var t = 0u; t < 9u; t++) { a += sc[t] * wt[slot[t] + D4 + o0 + e]; }
            att[o0 + e] = a;
        }
    }
    let pr = lin_D4_D4(&att, O_BLOCKS_0_PROJ_WEIGHT, O_BLOCKS_0_PROJ_BIAS);
    for (var e = 0u; e < D4; e++) { x[e] += pr[e]; }
    hh = x;
    layer_norm(&hh, O_BLOCKS_0_NORM2_WEIGHT, O_BLOCKS_0_NORM2_BIAS);
    var ff = lin_D4_DFF4(&hh, O_BLOCKS_0_FF1_WEIGHT, O_BLOCKS_0_FF1_BIAS);
    for (var f = 0u; f < DFF4; f++) { ff[f] = gelu(ff[f]); }
    let z = lin_DFF4_D4(&ff, O_BLOCKS_0_FF2_WEIGHT, O_BLOCKS_0_FF2_BIAS);
    for (var e = 0u; e < D4; e++) { x[e] += z[e]; }
    let l = lin_D4_1(&x, O_HEAD_WEIGHT, O_HEAD_BIAS);
    if ((dims.flags & WRITE_LOGITS) != 0u) { logits[i] = l[0].x; }
    next[i] = select(0u, 1u, l[0].x > 0.0);
}
