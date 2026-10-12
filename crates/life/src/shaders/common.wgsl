// Shared by the three model kernels. gpu.rs prepends the constants (D, H,
// DK, DFF, HID, EPS, SCALE and one `O_<TENSOR>` offset per weight tensor
// into `wt`, the packed weight vector).
struct Dims { w: u32, h: u32, n: u32, flags: u32 }

@group(0) @binding(0) var<storage, read> wt: array<f32>;
@group(0) @binding(1) var<storage, read> cells: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<storage, read_write> next: array<u32>;
@group(0) @binding(4) var<uniform> dims: Dims;

const WRITE_LOGITS: u32 = 1u;

// Exact erf-based GELU; `erf` has no WGSL builtin, so this is the
// Abramowitz-Stegun 7.1.26 rational approximation (absolute error 1.5e-7).
fn erf_approx(v: f32) -> f32 {
    let s = sign(v);
    let a = abs(v);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}

fn gelu(v: f32) -> f32 {
    return 0.5 * v * (1.0 + erf_approx(v * 0.7071067811865476));
}

// Cell indices of the 8 neighbours NW, N, NE, W, E, SW, S, SE on the torus,
// then the cell itself.
fn neighbors(i: u32) -> array<u32, 9> {
    let w = dims.w;
    let h = dims.h;
    let x = i % w;
    let y = i / w;
    let xm = (x + w - 1u) % w;
    let xp = (x + 1u) % w;
    let ym = (y + h - 1u) % h;
    let yp = (y + 1u) % h;
    return array<u32, 9>(ym * w + xm, ym * w + x, ym * w + xp, y * w + xm, y * w + xp, yp * w + xm, yp * w + x, yp * w + xp, i);
}

fn layer_norm(v: ptr<function, array<f32, D>>, g: u32, b: u32) {
    var mean = 0.0;
    for (var j = 0u; j < D; j++) { mean += (*v)[j]; }
    mean = mean / f32(D);
    var acc = 0.0;
    for (var j = 0u; j < D; j++) {
        let d = (*v)[j] - mean;
        acc += d * d;
    }
    let inv = 1.0 / sqrt(acc / f32(D) + EPS);
    for (var j = 0u; j < D; j++) {
        (*v)[j] = ((*v)[j] - mean) * inv * wt[g + j] + wt[b + j];
    }
}

fn softmax9(s: ptr<function, array<f32, 9>>) {
    var m = (*s)[0];
    for (var t = 1u; t < 9u; t++) { m = max(m, (*s)[t]); }
    var sum = 0.0;
    for (var t = 0u; t < 9u; t++) {
        let e = exp((*s)[t] - m);
        (*s)[t] = e;
        sum += e;
    }
    for (var t = 0u; t < 9u; t++) { (*s)[t] = (*s)[t] / sum; }
}

fn finish(i: u32, l0: f32, l1: f32) {
    if ((dims.flags & WRITE_LOGITS) != 0u) {
        logits[2u * i] = l0;
        logits[2u * i + 1u] = l1;
    }
    next[i] = select(0u, 1u, l1 > l0);
}
