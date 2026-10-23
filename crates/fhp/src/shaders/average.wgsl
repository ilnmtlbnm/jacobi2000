// The page's running time average of block velocity, folded on the GPU
// after each sample of the block sums (`coarse` output: jx, jy, mass,
// fluid per block). Integer only, as `coarse::Average` on the CPU, so the
// two agree bit for bit: per block the velocity u = j / (2 m) to 2^-16
// (`div16`, after a common right shift that keeps every operand below
// 2^22), then q += floor((u - q) a / 2^16), the first sample replacing q.
// q per block: ux, uy (units of j), the sample's fluid sites, unused.

struct AvgParams {
    n: u32,
    first: u32,
    a: i32,
    _p: u32,
}

@group(0) @binding(0) var<uniform> A: AvgParams;
@group(0) @binding(1) var<storage, read> s: array<i32>;
@group(0) @binding(2) var<storage, read_write> q: array<i32>;

fn div16(num: i32, den: i32) -> i32 {
    let a = num * 256;
    let q1 = a / den;
    let r1 = a - q1 * den;
    let q2 = (r1 * 256) / den;
    return q1 * 256 + q2;
}

// floor(d a / 2^16) for 0 <= a < 2^13.
fn mul_a(d: i32, a: i32) -> i32 {
    return (d >> 16u) * a + (((d & 0xffff) * a) >> 16u);
}

fn bits(v: i32) -> u32 {
    return 32u - countLeadingZeros(u32(v));
}

@compute @workgroup_size(64)
fn fold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= A.n) {
        return;
    }
    let o = 4u * b;
    var jx = s[o];
    var jy = s[o + 1u];
    var m2 = 2 * max(s[o + 2u], 1);
    let big = bits(max(m2, max(abs(jx), abs(jy))));
    let k = select(0u, big - 22u, big > 22u);
    jx = jx >> k;
    jy = jy >> k;
    m2 = max(m2 >> k, 1);
    let ux = div16(jx, m2);
    let uy = div16(jy, m2);
    if (A.first != 0u) {
        q[o] = ux;
        q[o + 1u] = uy;
    } else {
        q[o] = q[o] + mul_a(ux - q[o], A.a);
        q[o + 1u] = q[o + 1u] + mul_a(uy - q[o + 1u], A.a);
    }
    q[o + 2u] = s[o + 3u];
}
