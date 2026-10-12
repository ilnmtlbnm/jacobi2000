// Stencil model, one 8x8 tile of cells per workgroup (64 threads). Pre-LN
// block: attention over each cell's 9 taps on the torus, then the feed-
// forward, then the per-cell head. Keys and values depend only on a cell, so
// the tile computes them once for itself and its one-cell halo (10x10 cells,
// wrapped around the board edges) into workgroup memory, and every cell
// reads its 9 taps from there. Taps are in the order NW, N, NE, W, E, SW, S,
// SE, then the cell itself.
const KVN: u32 = 200u * D;
var<workgroup> skv: array<f32, KVN>;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.y * 8u + lid.x;
    let x0 = i32(wid.x * 8u);
    let y0 = i32(wid.y * 8u);
    let w = i32(dims.w);
    let h = i32(dims.h);
    for (var s = tid; s < 100u; s += 64u) {
        let gx = ((x0 + i32(s % 10u) - 1) % w + w) % w;
        let gy = ((y0 + i32(s / 10u) - 1) % h + h) % h;
        let cv = select(0.0, 1.0, cells[u32(gy * w + gx)] != 0u);
        var hh: array<f32, D>;
        for (var e = 0u; e < D; e++) { hh[e] = wt[O_EMBED_WEIGHT + e] * cv + wt[O_EMBED_BIAS + e]; }
        layer_norm(&hh, O_BLOCKS_0_NORM1_WEIGHT, O_BLOCKS_0_NORM1_BIAS);
        for (var o = 0u; o < D; o++) {
            var sk = wt[O_BLOCKS_0_K_BIAS + o];
            var sv = wt[O_BLOCKS_0_V_BIAS + o];
            for (var e = 0u; e < D; e++) {
                sk += wt[O_BLOCKS_0_K_WEIGHT + o * D + e] * hh[e];
                sv += wt[O_BLOCKS_0_V_WEIGHT + o * D + e] * hh[e];
            }
            skv[s * 2u * D + o] = sk;
            skv[s * 2u * D + D + o] = sv;
        }
    }
    workgroupBarrier();

    let px = x0 + i32(lid.x);
    let py = y0 + i32(lid.y);
    if (px >= w || py >= h) { return; }
    let i = u32(py * w + px);

    var x: array<f32, D>;
    var hh: array<f32, D>;
    let cv = select(0.0, 1.0, cells[i] != 0u);
    for (var e = 0u; e < D; e++) {
        x[e] = wt[O_EMBED_WEIGHT + e] * cv + wt[O_EMBED_BIAS + e];
        hh[e] = x[e];
    }
    layer_norm(&hh, O_BLOCKS_0_NORM1_WEIGHT, O_BLOCKS_0_NORM1_BIAS);
    var q: array<f32, D>;
    for (var o = 0u; o < D; o++) {
        var acc = wt[O_BLOCKS_0_Q_BIAS + o];
        for (var e = 0u; e < D; e++) { acc += wt[O_BLOCKS_0_Q_WEIGHT + o * D + e] * hh[e]; }
        q[o] = acc;
    }
    var slot: array<u32, 9>;
    for (var t = 0u; t < 9u; t++) {
        let u = select(select(t, t + 1u, t >= 4u), 4u, t == 8u);
        slot[t] = ((lid.y + u / 3u) * 10u + lid.x + u % 3u) * 2u * D;
    }
    var att: array<f32, D>;
    for (var hd = 0u; hd < H; hd++) {
        let o0 = hd * DK;
        var sc: array<f32, 9>;
        for (var t = 0u; t < 9u; t++) {
            var s = 0.0;
            for (var e = 0u; e < DK; e++) { s += q[o0 + e] * skv[slot[t] + o0 + e]; }
            sc[t] = s / SCALE;
        }
        softmax9(&sc);
        for (var e = 0u; e < DK; e++) {
            var a = 0.0;
            for (var t = 0u; t < 9u; t++) { a += sc[t] * skv[slot[t] + D + o0 + e]; }
            att[o0 + e] = a;
        }
    }
    for (var o = 0u; o < D; o++) {
        var acc = wt[O_BLOCKS_0_PROJ_BIAS + o];
        for (var e = 0u; e < D; e++) { acc += wt[O_BLOCKS_0_PROJ_WEIGHT + o * D + e] * att[e]; }
        x[o] += acc;
    }
    for (var e = 0u; e < D; e++) { hh[e] = x[e]; }
    layer_norm(&hh, O_BLOCKS_0_NORM2_WEIGHT, O_BLOCKS_0_NORM2_BIAS);
    var z: array<f32, D>;
    for (var o = 0u; o < D; o++) { z[o] = wt[O_BLOCKS_0_FF2_BIAS + o]; }
    for (var f = 0u; f < DFF; f++) {
        var acc = wt[O_BLOCKS_0_FF1_BIAS + f];
        for (var e = 0u; e < D; e++) { acc += wt[O_BLOCKS_0_FF1_WEIGHT + f * D + e] * hh[e]; }
        let g = gelu(acc);
        for (var o = 0u; o < D; o++) { z[o] += wt[O_BLOCKS_0_FF2_WEIGHT + o * DFF + f] * g; }
    }
    var lg = wt[O_HEAD_BIAS];
    for (var e = 0u; e < D; e++) { lg += wt[O_HEAD_WEIGHT + e] * (x[e] + z[e]); }
    if ((dims.flags & WRITE_LOGITS) != 0u) { logits[i] = lg; }
    next[i] = select(0u, 1u, lg > 0.0);
}
