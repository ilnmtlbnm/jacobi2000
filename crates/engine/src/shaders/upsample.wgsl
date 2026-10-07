// Linear upsample by `p` along one axis (0 = y, 1 = x), replicating the
// last row/column so the far edge has a neighbour to interpolate towards:
// cell `i` of token `t` mixes `t` and `t+1` with weight `(i + 0.5) / p`,
// matching `jacobi2000::model::upsample` / `cpu.rs::upsample_y`/`upsample_x`.
// One invocation per output element, same naive-kernel shape as the rest
// of this model's ops. `input: [c,h,w]`; axis 0 -> `out: [c,h*p,w]`; axis 1
// -> `out: [c,h,w*p]`.
struct Dims { c: u32, h: u32, w: u32, p: u32, axis: u32, len: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= dims.len) { return; }

    if (dims.axis == 0u) {
        let out_h = dims.h * dims.p;
        let plane = out_h * dims.w;
        let c = idx / plane;
        let rem = idx % plane;
        let yo = rem / dims.w;
        let x = rem % dims.w;
        let yt = yo / dims.p;
        let frac = yo % dims.p;
        let wgt = (f32(frac) + 0.5) / f32(dims.p);
        let next = min(yt + 1u, dims.h - 1u);
        let base = c * dims.h * dims.w;
        let a = input[base + yt * dims.w + x];
        let b = input[base + next * dims.w + x];
        out[idx] = a * (1.0 - wgt) + b * wgt;
    } else {
        let out_w = dims.w * dims.p;
        let plane = dims.h * out_w;
        let c = idx / plane;
        let rem = idx % plane;
        let y = rem / out_w;
        let xo = rem % out_w;
        let xt = xo / dims.p;
        let frac = xo % dims.p;
        let wgt = (f32(frac) + 0.5) / f32(dims.p);
        let next = min(xt + 1u, dims.w - 1u);
        let base = c * dims.h * dims.w + y * dims.w;
        let a = input[base + xt];
        let b = input[base + next];
        out[idx] = a * (1.0 - wgt) + b * wgt;
    }
}
