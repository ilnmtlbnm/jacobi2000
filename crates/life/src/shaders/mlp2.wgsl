// 9 -> HID -> HID -> 2 with ReLU, one thread per cell. Input order: the 8
// neighbours NW..SE, then the cell itself.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = (wid.y * nwg.x + wid.x) * 64u + li;
    if (i >= dims.n) { return; }
    let nb = neighbors(i);
    var a: array<vec4<f32>, HID4>;
    for (var n = 0u; n < HID4; n++) { a[n] = wt[O_FC1_BIAS + n]; }
    for (var j = 0u; j < 9u; j++) {
        let xj = select(0.0, 1.0, cells[nb[j]] != 0u);
        for (var n = 0u; n < HID4; n++) { a[n] += wt[O_FC1_WEIGHT + j * HID4 + n] * xj; }
    }
    for (var n = 0u; n < HID4; n++) { a[n] = max(a[n], vec4<f32>(0.0)); }
    var b = lin_HID4_HID4(&a, O_FC2_WEIGHT, O_FC2_BIAS);
    for (var n = 0u; n < HID4; n++) { b[n] = max(b[n], vec4<f32>(0.0)); }
    let l = lin_HID4_1(&b, O_FC3_WEIGHT, O_FC3_BIAS);
    finish(i, l[0].x, l[0].y);
}
