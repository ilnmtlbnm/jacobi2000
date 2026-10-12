// BERT of Life, one thread per cell. One post-LN encoder layer, no mask; the
// head reads token 8 (the cell itself), so only token 8's residual, norms and
// feed-forward are computed: its query attends over all 9 keys/values and the
// other tokens' later stages are never needed. Tokens: the 8 neighbours
// NW..SE (ids 0/1 into a 2-row table) then the cell, plus learned positions.
// A token is an id 0/1 at one of 9 positions, so every key and value (18
// rows, table `O_KV`) and the centre query (2 rows, `O_Q`) come from tables
// computed once per model (`cpu::bert_tables`).
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = (wid.y * nwg.x + wid.x) * 64u + li;
    if (i >= dims.n) { return; }
    let nb = neighbors(i);
    var row: array<u32, 9>;
    for (var t = 0u; t < 9u; t++) { row[t] = O_KV + (t * 2u + select(0u, 1u, cells[nb[t]] != 0u)) * 2u * D4; }
    let id8 = select(0u, 1u, cells[i] != 0u);
    var x8: array<vec4<f32>, D4>;
    for (var e = 0u; e < D4; e++) { x8[e] = wt[O_TOK_EMB_WEIGHT + id8 * D4 + e] + wt[O_POS_EMB_WEIGHT + 8u * D4 + e]; }
    var att: array<vec4<f32>, D4>;
    for (var hd = 0u; hd < H; hd++) {
        let o0 = hd * DK4;
        var sc: array<f32, 9>;
        for (var t = 0u; t < 9u; t++) {
            var s = vec4<f32>(0.0);
            for (var e = 0u; e < DK4; e++) { s += wt[O_Q + id8 * D4 + o0 + e] * wt[row[t] + o0 + e]; }
            sc[t] = (s.x + s.y + s.z + s.w) / SCALE;
        }
        softmax9(&sc);
        for (var e = 0u; e < DK4; e++) {
            var a = vec4<f32>(0.0);
            for (var t = 0u; t < 9u; t++) { a += sc[t] * wt[row[t] + D4 + o0 + e]; }
            att[o0 + e] = a;
        }
    }
    var y = lin_D4_D4(&att, O_MHA_OUTPUT_WEIGHT, O_MHA_OUTPUT_BIAS);
    for (var e = 0u; e < D4; e++) { y[e] += x8[e]; }
    layer_norm(&y, O_NORM_1_WEIGHT, O_NORM_1_BIAS);
    var ff = lin_D4_DFF4(&y, O_PWFF_LINEAR_INNER_WEIGHT, O_PWFF_LINEAR_INNER_BIAS);
    for (var f = 0u; f < DFF4; f++) { ff[f] = gelu(ff[f]); }
    var z = lin_DFF4_D4(&ff, O_PWFF_LINEAR_OUTER_WEIGHT, O_PWFF_LINEAR_OUTER_BIAS);
    for (var e = 0u; e < D4; e++) { z[e] += y[e]; }
    layer_norm(&z, O_NORM_2_WEIGHT, O_NORM_2_BIAS);
    let l = lin_D4_1(&z, O_HEAD_WEIGHT, O_HEAD_BIAS);
    finish(i, l[0].x, l[0].y);
}
