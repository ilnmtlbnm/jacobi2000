// 9 -> HID -> HID -> 2 with ReLU, one thread per cell. Input order: the 8
// neighbours NW..SE, then the cell itself.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = (wid.y * nwg.x + wid.x) * 64u + li;
    if (i >= dims.n) { return; }
    let nb = neighbors(i);
    var x: array<f32, 9>;
    for (var t = 0u; t < 9u; t++) { x[t] = select(0.0, 1.0, cells[nb[t]] != 0u); }
    var a: array<f32, HID>;
    var b: array<f32, HID>;
    for (var o = 0u; o < HID; o++) {
        var acc = wt[O_FC1_BIAS + o];
        for (var j = 0u; j < 9u; j++) { acc += wt[O_FC1_WEIGHT + o * 9u + j] * x[j]; }
        a[o] = max(acc, 0.0);
    }
    for (var o = 0u; o < HID; o++) {
        var acc = wt[O_FC2_BIAS + o];
        for (var j = 0u; j < HID; j++) { acc += wt[O_FC2_WEIGHT + o * HID + j] * a[j]; }
        b[o] = max(acc, 0.0);
    }
    var l: array<f32, 2>;
    for (var o = 0u; o < 2u; o++) {
        var acc = wt[O_FC3_BIAS + o];
        for (var j = 0u; j < HID; j++) { acc += wt[O_FC3_WEIGHT + o * HID + j] * b[j]; }
        l[o] = acc;
    }
    finish(i, l[0], l[1]);
}
