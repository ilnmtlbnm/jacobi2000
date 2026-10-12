// BERT of Life, one thread per cell. One post-LN encoder layer, no mask; the
// head reads token 8 (the cell itself), so only token 8's query, residual,
// norms and feed-forward are computed: it attends over all 9 keys/values and
// the other tokens' later stages are never needed. Tokens: the 8 neighbours
// NW..SE (ids 0/1 into a 2-row table) then the cell, plus learned positions.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = (wid.y * nwg.x + wid.x) * 64u + li;
    if (i >= dims.n) { return; }
    let nb = neighbors(i);
    var k: array<f32, KV>;
    var v: array<f32, KV>;
    var x8: array<f32, D>;
    for (var t = 0u; t < 9u; t++) {
        let tv = select(0u, 1u, cells[nb[t]] != 0u);
        var xi: array<f32, D>;
        for (var e = 0u; e < D; e++) { xi[e] = wt[O_TOK_EMB_WEIGHT + tv * D + e] + wt[O_POS_EMB_WEIGHT + t * D + e]; }
        if (t == 8u) { x8 = xi; }
        for (var o = 0u; o < D; o++) {
            var sk = wt[O_MHA_KEY_BIAS + o];
            var sv = wt[O_MHA_VALUE_BIAS + o];
            for (var e = 0u; e < D; e++) {
                sk += wt[O_MHA_KEY_WEIGHT + o * D + e] * xi[e];
                sv += wt[O_MHA_VALUE_WEIGHT + o * D + e] * xi[e];
            }
            k[t * D + o] = sk;
            v[t * D + o] = sv;
        }
    }
    var q: array<f32, D>;
    for (var o = 0u; o < D; o++) {
        var acc = wt[O_MHA_QUERY_BIAS + o];
        for (var e = 0u; e < D; e++) { acc += wt[O_MHA_QUERY_WEIGHT + o * D + e] * x8[e]; }
        q[o] = acc;
    }
    var att: array<f32, D>;
    for (var hd = 0u; hd < H; hd++) {
        let o0 = hd * DK;
        var sc: array<f32, 9>;
        for (var t = 0u; t < 9u; t++) {
            var s = 0.0;
            for (var e = 0u; e < DK; e++) { s += q[o0 + e] * k[t * D + o0 + e]; }
            sc[t] = s / SCALE;
        }
        softmax9(&sc);
        for (var e = 0u; e < DK; e++) {
            var a = 0.0;
            for (var t = 0u; t < 9u; t++) { a += sc[t] * v[t * D + o0 + e]; }
            att[o0 + e] = a;
        }
    }
    var y: array<f32, D>;
    for (var o = 0u; o < D; o++) {
        var acc = wt[O_MHA_OUTPUT_BIAS + o];
        for (var e = 0u; e < D; e++) { acc += wt[O_MHA_OUTPUT_WEIGHT + o * D + e] * att[e]; }
        y[o] = acc + x8[o];
    }
    layer_norm(&y, O_NORM_1_WEIGHT, O_NORM_1_BIAS);
    var z: array<f32, D>;
    for (var o = 0u; o < D; o++) { z[o] = wt[O_PWFF_LINEAR_OUTER_BIAS + o]; }
    // Feed-forward unit by unit: gelu(inner_f(y)) feeds outer's column f.
    for (var f = 0u; f < DFF; f++) {
        var acc = wt[O_PWFF_LINEAR_INNER_BIAS + f];
        for (var e = 0u; e < D; e++) { acc += wt[O_PWFF_LINEAR_INNER_WEIGHT + f * D + e] * y[e]; }
        let g = gelu(acc);
        for (var o = 0u; o < D; o++) { z[o] += wt[O_PWFF_LINEAR_OUTER_WEIGHT + o * DFF + f] * g; }
    }
    for (var o = 0u; o < D; o++) { z[o] += y[o]; }
    layer_norm(&z, O_NORM_2_WEIGHT, O_NORM_2_BIAS);
    var l: array<f32, 2>;
    for (var o = 0u; o < 2u; o++) {
        var acc = wt[O_HEAD_BIAS + o];
        for (var e = 0u; e < D; e++) { acc += wt[O_HEAD_WEIGHT + o * D + e] * z[e]; }
        l[o] = acc;
    }
    finish(i, l[0], l[1]);
}
