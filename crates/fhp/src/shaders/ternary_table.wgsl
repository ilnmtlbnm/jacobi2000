// One ternary FHP step with one u32 per site (the base-3 state, bit 15 the
// obstacle) and the collision as a lookup in the rule-Chiral table, one
// invocation per site. Same random bits as the CPU engines (one chirality
// bit per site from its 32-site chunk's word). On a boundary step (`on`),
// fluid sites in the inlet and outlet columns get a fresh equilibrium sample
// instead, as `engine::boundary_state`.

struct Params {
    w: u32,
    h: u32,
    kc: u32,
    kb: u32,
    on: u32,
    inlet: u32,
    outlet_start: u32,
    _p: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
// [0, 2187): chirality 0 | chirality 1 << 16; [2187, 4374): bounce-back;
// [4374, 4388): the boundary's thresholds for n >= 1 and n >= 2 per channel.
@group(0) @binding(3) var<storage, read> T: array<u32>;

fn lowbias32(v: u32) -> u32 {
    var x = v;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// Count of channel d (power 3^d) in the state at (x, y).
fn count(x: u32, y: u32, p: u32) -> u32 {
    return ((src[y * P.w + x] & 0x7fffu) / p) % 3u;
}

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= P.w || y >= P.h) {
        return;
    }
    let ym = select(y - 1u, P.h - 1u, y == 0u);
    let yp = select(y + 1u, 0u, y + 1u == P.h);
    let xm = select(x - 1u, P.w - 1u, x == 0u);
    let xp = select(x + 1u, 0u, x + 1u == P.w);
    // Source columns of the diagonal directions from the source row's parity
    // (lib.rs DX): odd rows are shifted half a site to the right.
    let om = (ym & 1u) == 1u;
    let op = (yp & 1u) == 1u;
    let own = src[y * P.w + x];
    var s = ((own & 0x7fffu) / 729u) % 3u * 729u;
    s += count(xm, y, 1u);
    s += count(select(x, xm, om), ym, 3u) * 3u;
    s += count(select(xp, x, om), ym, 9u) * 9u;
    s += count(xp, y, 27u) * 27u;
    s += count(select(xp, x, op), yp, 81u) * 81u;
    s += count(select(x, xm, op), yp, 243u) * 243u;
    var o: u32;
    if (P.on != 0u && (x < P.inlet || x >= P.outlet_start) && (own & 0x8000u) == 0u) {
        let i = y * P.w + x;
        o = 0u;
        var p = 1u;
        for (var c = 0u; c < 7u; c++) {
            let r = lowbias32(P.kb ^ lowbias32(i * 8u + c));
            o += (u32(r >= T[4374u + 2u * c]) + u32(r >= T[4375u + 2u * c])) * p;
            p *= 3u;
        }
    } else if ((own & 0x8000u) != 0u) {
        o = T[2187u + s] | 0x8000u;
    } else {
        let c = (lowbias32(P.kc ^ ((y * (P.w / 32u) + x / 32u) * 0x9e3779b9u)) >> (x % 32u)) & 1u;
        o = (T[s] >> (16u * c)) & 0xffffu;
    }
    dst[y * P.w + x] = o;
}

// Block sums for the coarse-grained field, added to `sums` (cleared by the
// host): jx, jy (integer units 2 px, 2 py / sqrt 3), mass and fluid sites
// per block, as `engine::Ternary::block_sums`. One invocation per block.
struct CoarseParams {
    w: u32,
    h: u32,
    bw: u32,
    bh: u32,
    nx: u32,
    ny: u32,
    _a: u32,
    _b: u32,
}

@group(0) @binding(0) var<uniform> C: CoarseParams;
@group(0) @binding(1) var<storage, read> cstate: array<u32>;
@group(0) @binding(2) var<storage, read_write> sums: array<i32>;

@compute @workgroup_size(64)
fn coarse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    if (bx >= C.nx || by >= C.ny) {
        return;
    }
    var jx = 0i;
    var jy = 0i;
    var mass = 0i;
    var fluid = 0i;
    for (var y = by * C.bh; y < (by + 1u) * C.bh; y++) {
        for (var x = bx * C.bw; x < (bx + 1u) * C.bw; x++) {
            var s = cstate[y * C.w + x];
            if ((s & 0x8000u) != 0u) {
                continue;
            }
            var n: array<i32, 7>;
            for (var c = 0u; c < 7u; c++) {
                n[c] = i32(s % 3u);
                s /= 3u;
            }
            jx += 2i * n[0] + n[1] - n[2] - 2i * n[3] - n[4] + n[5];
            jy += n[1] + n[2] - n[4] - n[5];
            mass += n[0] + n[1] + n[2] + n[3] + n[4] + n[5] + n[6];
            fluid += 1i;
        }
    }
    let b = (by * C.nx + bx) * 4u;
    sums[b] += jx;
    sums[b + 1u] += jy;
    sums[b + 2u] += mass;
    sums[b + 3u] += fluid;
}
