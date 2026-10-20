// Signed ternary FHP (`signed::Signed`) on the GPU, bit-identical to the CPU
// engine. State: one u32 per site (the base-3 code of delta + 1, bit 15 the
// obstacle, bit 14 the diagnostic wall), the reference phi as 7 i32 per site
// (channel-major, units of 2^-12 particles), and 32 words per reference
// block. Every integer operation is the CPU's (`feq_fixed`, `div16`,
// `mul_shr`, `interp`, `sround`), so the two agree bit for bit.
//
// Work is split in tiles of 64 columns x TH rows (TH = max(B, 4)), one
// workgroup of 64 x 4 invocations per tile, looping over the tile's rows four
// at a time. A tile holds whole reference blocks, so every per-block sum is
// a workgroup reduction written by that one workgroup (no atomics).
//
// Entry points:
// - step: stream with the reference correction, collide, and add the
//   rounding residuals and clipped overflow to the receiving sites' block
//   reservoirs. On a re-estimating step the same workgroup then has its
//   blocks' content (the block's sum of phi, kept since the last
//   re-estimate, + delta + reservoir) and writes the new block values to the
//   other node slot, the old reservoir's share per fluid site, and the
//   reservoir set to the content. No separate content pass.
// - requant: new reference and re-quantized deviations, each block's new
//   sum of phi, and its reservoir minus the new content, which leaves the
//   exact remainder. With INIT it only computes phi (all sites) and the
//   blocks' sums of phi from the current block values.
// - phi_out (tests) and coarse (the page's block sums) read the state.

struct Params {
    w: u32,
    h: u32,
    b: u32,
    // log2(2B)
    lb: u32,
    nbx: u32,
    nby: u32,
    rule: u32,
    flags: u32,
    kc: u32,
    ks: u32,
    kq: u32,
    kr: u32,
    inlet: u32,
    outlet_start: u32,
    r_in: i32,
    jx_in: i32,
    rho0: i32,
    // node slot of the current block values (0 or 1)
    cur: u32,
    // tile height, max(B, 4)
    th: u32,
    _a: u32,
}

// flags
const REFRESH: u32 = 1u;
const NOKICK: u32 = 2u;
const BOUNDARY: u32 = 4u;
// Diagnostic (`signed::WALLS`): obstacle sites with bit 14 reflect
// specularly (free slip) and hold the inflow reference; the reference is
// not interpolated across the y wrap.
const WALLS: u32 = 8u;
const WALL: u32 = 0x4000u;
// Diagnostic (`signed::COLUMNS`): the first engine's column geometry (the
// reference interpolated at the column index).
const COLUMNS: u32 = 16u;
// This step ends with a re-estimate.
const REEST: u32 = 32u;
// requant: only the blocks' sums of phi.
const INIT: u32 = 64u;

const ONE: i32 = 4096;
const OBST: u32 = 0x8000u;

// Per block, 32 words.
const RES: u32 = 0u;     // reservoir, 7 channels
const FLUID: u32 = 7u;   // fluid sites
const REL: u32 = 8u;     // during a re-estimate: old reservoir per fluid site, 7
const SPHI: u32 = 16u;   // sum of phi over the fluid sites, 7
const NODE: u32 = 24u;   // block values (r, jx, jy, -) in slot 0, then slot 1

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
// [0, 2187): rule chiral, chirality 0 | chirality 1 << 16; [2187, 4374):
// bounce-back; [4374, 4374 + 6 * 2187): rule optimal, the 12 images of each
// state, two per word.
@group(0) @binding(3) var<storage, read> T: array<u32>;
@group(0) @binding(4) var<storage, read_write> blk: array<i32>;
// phi per site, channel-major (7 planes).
@group(0) @binding(5) var<storage, read_write> phi: array<i32>;

fn load_phi(i: u32) -> Ch {
    let n = P.w * P.h;
    return Ch(vec4<i32>(phi[i], phi[n + i], phi[2u * n + i], phi[3u * n + i]), vec4<i32>(phi[4u * n + i], phi[5u * n + i], phi[6u * n + i], 0));
}

fn store_phi(i: u32, f: Ch) {
    let n = P.w * P.h;
    phi[i] = f.a.x;
    phi[n + i] = f.a.y;
    phi[2u * n + i] = f.a.z;
    phi[3u * n + i] = f.a.w;
    phi[4u * n + i] = f.b.x;
    phi[5u * n + i] = f.b.y;
    phi[6u * n + i] = f.b.z;
}

// Seven per-channel values as two vectors (channels 0-3, 4-6), so no
// channel is indexed by a variable.
struct Ch {
    a: vec4<i32>,
    b: vec4<i32>,
}

const G: u32 = 0x9e3779b9u;

fn lowbias32(v: u32) -> u32 {
    var x = v;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn mulhi(r: u32, m: u32) -> u32 {
    return ((r >> 16u) * m + (((r & 0xffffu) * m) >> 16u)) >> 16u;
}

fn div16(num: i32, den: i32) -> i32 {
    let a = num * 256;
    let q1 = a / den;
    let r1 = a - q1 * den;
    let q2 = (r1 * 256) / den;
    return q1 * 256 + q2;
}

fn mul_shr(a: i32, b: i32, s: u32) -> i32 {
    let ua = u32(abs(a));
    let ub = u32(abs(b));
    let al = ua & 0xffffu;
    let ah = ua >> 16u;
    let bl = ub & 0xffffu;
    let bh = ub >> 16u;
    let lh = al * bh;
    let hl = ah * bl;
    var lo = al * bl;
    var hi = ah * bh + (lh >> 16u) + (hl >> 16u);
    let t1 = lo + (lh << 16u);
    hi += u32(t1 < lo);
    lo = t1;
    let t2 = lo + (hl << 16u);
    hi += u32(t2 < lo);
    lo = t2;
    let p = i32((lo >> s) | (hi << (32u - s)));
    return select(p, -p, (a < 0) != (b < 0));
}

// `signed::feq_fixed`. Channel i + 3 has d and c.u of channel i negated,
// so the same rho (c.u)^2 term.
fn feq(r0: i32, jx: i32, jy: i32) -> Ch {
    let r = max(r0, 1);
    let ux = div16(jx, r);
    let uy = div16(jy, r);
    let p = mul_shr(jx, ux, 16u) + 3 * mul_shr(jy, uy, 16u);
    let d0 = 2 * jx;
    let d1 = jx + 3 * jy;
    let d2 = -jx + 3 * jy;
    let e0 = 24 * r - 7 * p + 112 * mul_shr(d0, 2 * ux, 20u);
    let e1 = 24 * r - 7 * p + 112 * mul_shr(d1, ux + 3 * uy, 20u);
    let e2 = 24 * r - 7 * p + 112 * mul_shr(d2, -ux + 3 * uy, 20u);
    return Ch(
        vec4<i32>(e0 + 14 * d0, e1 + 14 * d1, e2 + 14 * d2, e0 - 14 * d0) / 168,
        vec4<i32>((e1 - 14 * d1) / 168, (e2 - 14 * d2) / 168, (4 * r - 7 * p) / 28, 0),
    );
}

// The deviations of a cell, -1, 0 or 1 per channel.
fn deltas(cell: u32) -> Ch {
    var c = cell & 0x3fffu;
    let n0 = c % 3u;
    c /= 3u;
    let n1 = c % 3u;
    c /= 3u;
    let n2 = c % 3u;
    c /= 3u;
    let n3 = c % 3u;
    c /= 3u;
    let n4 = c % 3u;
    c /= 3u;
    let n5 = c % 3u;
    let n6 = c / 3u;
    return Ch(vec4<i32>(vec4<u32>(n0, n1, n2, n3)) - 1, vec4<i32>(vec4<u32>(n4, n5, n6, 1u)) - 1);
}

fn node(k: u32, slot: u32) -> vec3<i32> {
    let o = k * 32u + NODE + slot * 4u;
    return vec3<i32>(blk[o], blk[o + 1u], blk[o + 2u]);
}

// Bilinear block values at (x, y), as `signed::interp`: x in quarter sites
// at the true positions (odd rows half a site right) from the blocks'
// centroids, weights out of 4B; y in half rows, out of 2B.
fn interp(x: u32, y: u32, slot: u32) -> vec3<i32> {
    let b = i32(P.b);
    let sy = 2 * b;
    let columns = (P.flags & COLUMNS) != 0u;
    let sx = select(4 * b, sy, columns);
    let lx = select(P.lb + 1u, P.lb, columns);
    let fx = select(4 * i32(x) + 2 * i32(y & 1u) + 1 - 2 * b, 2 * i32(x) + 1 - b, columns);
    let fy = 2 * i32(y) + 1 - b;
    let bx0 = fx >> lx;
    let by0 = fy >> P.lb;
    let tx = fx - (bx0 << lx);
    let ty = fy - (by0 << P.lb);
    let nx = i32(P.nbx);
    let ny = i32(P.nby);
    let x0 = u32((bx0 + nx) % nx);
    let x1 = u32((bx0 + 1 + nx) % nx);
    var y0 = u32((by0 + ny) % ny);
    var y1 = u32((by0 + 1 + ny) % ny);
    if ((P.flags & WALLS) != 0u) {
        y0 = u32(clamp(by0, 0, ny - 1));
        y1 = u32(clamp(by0 + 1, 0, ny - 1));
    }
    let v00 = node(y0 * P.nbx + x0, slot);
    let v01 = node(y0 * P.nbx + x1, slot);
    let v10 = node(y1 * P.nbx + x0, slot);
    let v11 = node(y1 * P.nbx + x1, slot);
    let r0 = ((sx - tx) * v00 + tx * v01) >> vec3<u32>(lx);
    let r1 = ((sx - tx) * v10 + tx * v11) >> vec3<u32>(lx);
    return ((sy - ty) * r0 + ty * r1) >> vec3<u32>(P.lb);
}

// The reference at site (x, y) holding `cell`, from node slot `slot`:
// obstacles hold the rest reference (walls the inflow one).
fn site_phi(cell: u32, x: u32, y: u32, slot: u32) -> Ch {
    if ((cell & OBST) != 0u) {
        if ((cell & WALL) != 0u) {
            return feq(P.r_in, P.jx_in, 0);
        }
        return feq(P.rho0, 0, 0);
    }
    let v = interp(x, y, slot);
    return feq(v.x, v.y, v.z);
}

// Specular reflection at a horizontal wall: channels 1 <-> 5, 2 <-> 4.
fn mirror(s: u32) -> u32 {
    let d1 = (s / 3u) % 3u;
    let d2 = (s / 9u) % 3u;
    let d4 = (s / 81u) % 3u;
    let d5 = (s / 243u) % 3u;
    return s - d1 * 3u - d2 * 9u - d4 * 81u - d5 * 243u + d5 * 3u + d4 * 9u + d2 * 81u + d1 * 243u;
}

fn sround(x: i32, r: u32) -> i32 {
    return (x + i32(r >> 20u)) >> 12u;
}

fn in_zone(x: u32) -> bool {
    return (P.flags & BOUNDARY) != 0u && (x < P.inlet || x >= P.outlet_start);
}

// ---- Tiles and per-block reductions ----

const TW: u32 = 64u;

// Reduction scratch, 7 words per invocation, then the partial sums, then
// per (tile block, channel) two results.
var<workgroup> scratch: array<i32, 1792>;
var<workgroup> part: array<i32, 448>;
var<workgroup> bsum0: array<i32, 448>;
var<workgroup> bsum1: array<i32, 448>;

// Blocks per tile and the global block of tile block t.
fn tile_blocks() -> u32 {
    return (TW / P.b) * (4u / min(P.b, 4u));
}

fn tile_block(wid: vec2<u32>, t: u32) -> u32 {
    let nbt = TW / P.b;
    let kx = wid.x * nbt + t % nbt;
    let ky = (wid.y * P.th) / P.b + t / nbt;
    return ky * P.nbx + kx;
}

fn put7(li: u32, v: Ch) {
    let o = li * 7u;
    scratch[o] = v.a.x;
    scratch[o + 1u] = v.a.y;
    scratch[o + 2u] = v.a.z;
    scratch[o + 3u] = v.a.w;
    scratch[o + 4u] = v.b.x;
    scratch[o + 5u] = v.b.y;
    scratch[o + 6u] = v.b.z;
}

// The per-block totals of v into `bsum0` (a = false) or `bsum1`, in two
// levels: tile block t / 7 holds the B x min(B, 4) invocations
// (ly / min(B, 4)) * (64 / B) + lx / B; its channel t % 7 is summed in
// parts of (at most) 8 values, then the parts are summed. Integer sums do
// not depend on the order.
fn reduce(li: u32, v: Ch, into1: bool) {
    workgroupBarrier();
    put7(li, v);
    workgroupBarrier();
    let lbb = P.lb - 1u;
    let rows = min(P.b, 4u);
    let lrows = min(lbb, 2u);
    let chunk = min(P.b * rows, 8u);
    let lchunk = min(lbb + lrows, 3u);
    let lparts = lbb + lrows - lchunk;
    let lnbt = 6u - lbb;
    let n = tile_blocks() * 7u;
    for (var t = li; t < n << lparts; t += 256u) {
        let o = t >> lparts;
        let b = o / 7u;
        let c = o % 7u;
        let k0 = (t & ((1u << lparts) - 1u)) << lchunk;
        let bx = b & ((1u << lnbt) - 1u);
        let by = b >> lnbt;
        var s = 0;
        for (var j = 0u; j < chunk; j++) {
            let k = k0 + j;
            let lx = (bx << lbb) + (k & (P.b - 1u));
            let ly = (by << lrows) + (k >> lbb);
            s += scratch[(ly * TW + lx) * 7u + c];
        }
        part[t] = s;
    }
    workgroupBarrier();
    for (var t = li; t < n; t += 256u) {
        var s = 0;
        for (var q = 0u; q < 1u << lparts; q++) {
            s += part[(t << lparts) + q];
        }
        if (into1) {
            bsum1[t] = s;
        } else {
            bsum0[t] = s;
        }
    }
    workgroupBarrier();
}

// Channel d of site i streamed from a neighbour whose deviation is dl, with
// reference mismatch r: the post-streaming count (0, 1, 2) and the
// reservoir addition.
fn chan(d: u32, dl: i32, r: i32, i: u32) -> vec2<i32> {
    var v = dl;
    var a = 0;
    if (r != 0) {
        let k = sround(r, lowbias32(P.ks ^ ((i * 8u + d) * G)));
        a = r - k * ONE;
        v = dl + k;
    }
    let c = clamp(v, -1, 1);
    return vec2<i32>(c + 1, a + (v - c) * ONE);
}

@compute @workgroup_size(64, 4)
fn step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let x0 = wid.x * TW;
    let y0 = wid.y * P.th;
    let x = x0 + lid.x;
    let reest = (P.flags & REEST) != 0u;
    var res = Ch(vec4<i32>(0), vec4<i32>(0));
    var dsum = Ch(vec4<i32>(0), vec4<i32>(0));
    for (var c0 = 0u; c0 < P.th; c0 += 4u) {
        let y = y0 + c0 + lid.y;
        let i = y * P.w + x;
        let n = P.w * P.h;
        let ym = select(y - 1u, P.h - 1u, y == 0u);
        let yp = select(y + 1u, 0u, y + 1u == P.h);
        let xm = select(x - 1u, P.w - 1u, x == 0u);
        let xp = select(x + 1u, 0u, x + 1u == P.w);
        let om = (ym & 1u) == 1u;
        let op = (yp & 1u) == 1u;
        let j0 = y * P.w + xm;
        let j1 = ym * P.w + select(x, xm, om);
        let j2 = ym * P.w + select(xp, x, om);
        let j3 = y * P.w + xp;
        let j4 = yp * P.w + select(xp, x, op);
        let j5 = yp * P.w + select(x, xm, op);
        let own = src[i];
        let q0 = chan(0u, i32((src[j0] & 0x3fffu) % 3u) - 1, phi[j0] - phi[i], i);
        let q1 = chan(1u, i32(((src[j1] & 0x3fffu) / 3u) % 3u) - 1, phi[n + j1] - phi[n + i], i);
        let q2 = chan(2u, i32(((src[j2] & 0x3fffu) / 9u) % 3u) - 1, phi[2u * n + j2] - phi[2u * n + i], i);
        let q3 = chan(3u, i32(((src[j3] & 0x3fffu) / 27u) % 3u) - 1, phi[3u * n + j3] - phi[3u * n + i], i);
        let q4 = chan(4u, i32(((src[j4] & 0x3fffu) / 81u) % 3u) - 1, phi[4u * n + j4] - phi[4u * n + i], i);
        let q5 = chan(5u, i32(((src[j5] & 0x3fffu) / 243u) % 3u) - 1, phi[5u * n + j5] - phi[5u * n + i], i);
        res.a += vec4<i32>(q0.y, q1.y, q2.y, q3.y);
        res.b += vec4<i32>(q4.y, q5.y, 0, 0);
        let s = (((own & 0x3fffu) / 729u) % 3u) * 729u + u32(q0.x) + u32(q1.x) * 3u + u32(q2.x) * 9u + u32(q3.x) * 27u + u32(q4.x) * 81u + u32(q5.x) * 243u;
        var nc: u32;
        if ((own & WALL) != 0u) {
            nc = mirror(s) | OBST | WALL;
        } else if ((own & OBST) != 0u) {
            nc = T[2187u + s] | OBST;
        } else if (P.rule == 1u) {
            let c = (lowbias32(P.kc ^ ((y * (P.w / 32u) + x / 32u) * G)) >> (x % 32u)) & 1u;
            nc = (T[s] >> (16u * c)) & 0xffffu;
        } else {
            let k = mulhi(lowbias32(P.kc ^ (i * G)), 12u);
            nc = (T[4374u + s * 6u + k / 2u] >> (16u * (k & 1u))) & 0xffffu;
        }
        dst[i] = nc;
        if (reest && (nc & OBST) == 0u) {
            let d = deltas(nc);
            dsum.a += d.a;
            dsum.b += d.b;
        }
    }
    reduce(li, res, false);
    let n = tile_blocks() * 7u;
    if (!reest) {
        for (var t = li; t < n; t += 256u) {
            let o = tile_block(wid.xy, t / 7u) * 32u + RES + t % 7u;
            blk[o] += bsum0[t];
        }
        return;
    }
    reduce(li, dsum, true);
    // Per (block, channel): the content (sum of phi + delta + reservoir
    // after this step), the old reservoir's share, the reservoir set to the
    // content (0 in a refreshed zone; unchanged without fluid sites).
    for (var t = li; t < n; t += 256u) {
        let k = tile_block(wid.xy, t / 7u);
        let c = t % 7u;
        let o = k * 32u;
        let nf = blk[o + FLUID];
        let r = blk[o + RES + c] + bsum0[t];
        let content = blk[o + SPHI + c] + bsum1[t] * ONE + r;
        bsum1[t] = content;
        if (nf == 0) {
            blk[o + RES + c] = r;
        } else {
            let zone = (P.flags & REFRESH) != 0u && in_zone((k % P.nbx) * P.b);
            blk[o + REL + c] = r / nf;
            blk[o + RES + c] = select(content, 0, zone);
        }
    }
    workgroupBarrier();
    // Per block: the new values into the other node slot.
    for (var t = li; t < tile_blocks(); t += 256u) {
        let k = tile_block(wid.xy, t);
        let o = k * 32u;
        let nf = blk[o + FLUID];
        var v = node(k, P.cur);
        if (nf != 0) {
            let s = t * 7u;
            let m = bsum1[s] + bsum1[s + 1u] + bsum1[s + 2u] + bsum1[s + 3u] + bsum1[s + 4u] + bsum1[s + 5u] + bsum1[s + 6u];
            let jx = 2 * bsum1[s] + bsum1[s + 1u] - bsum1[s + 2u] - 2 * bsum1[s + 3u] - bsum1[s + 4u] + bsum1[s + 5u];
            let jy = bsum1[s + 1u] + bsum1[s + 2u] - bsum1[s + 4u] - bsum1[s + 5u];
            v = vec3<i32>(m / nf, jx / nf, jy / nf);
            if ((P.flags & REFRESH) != 0u && in_zone((k % P.nbx) * P.b)) {
                v = vec3<i32>(P.r_in, P.jx_in, 0);
            }
        }
        let p = o + NODE + (1u - P.cur) * 4u;
        blk[p] = v.x;
        blk[p + 1u] = v.y;
        blk[p + 2u] = v.z;
    }
}

fn kick(c: u32, d: i32, dphi: i32, i: u32) -> u32 {
    var k = 0;
    if (dphi != 0 && (P.flags & NOKICK) == 0u) {
        k = sround(dphi, lowbias32(P.kq ^ ((i * 8u + c) * G)));
    }
    return u32(clamp(d + k, -1, 1) + 1);
}

fn fresh(c: u32, i: u32) -> u32 {
    return mulhi(lowbias32(P.kr ^ ((i * 8u + c) * G)), 3u);
}

// The re-estimate's second half, after a `step` with REEST: node slot cur
// holds the old block values, 1 - cur the new ones.
@compute @workgroup_size(64, 4)
fn requant(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let x = wid.x * TW + lid.x;
    let init = (P.flags & INIT) != 0u;
    let slot = select(1u - P.cur, P.cur, init);
    var sphi = Ch(vec4<i32>(0), vec4<i32>(0));
    var dsum = Ch(vec4<i32>(0), vec4<i32>(0));
    for (var c0 = 0u; c0 < P.th; c0 += 4u) {
        let y = wid.y * P.th + c0 + lid.y;
        let i = y * P.w + x;
        let cell = dst[i];
        if (init) {
            let f = site_phi(cell, x, y, slot);
            store_phi(i, f);
            if ((cell & OBST) == 0u) {
                sphi.a += f.a;
                sphi.b += f.b;
            }
            continue;
        }
        if ((cell & OBST) != 0u) {
            continue;
        }
        let nv = interp(x, y, slot);
        let f = feq(nv.x, nv.y, nv.z);
        sphi.a += f.a;
        sphi.b += f.b;
        var code = 0u;
        if ((P.flags & REFRESH) != 0u && in_zone(x)) {
            code = fresh(0u, i) + fresh(1u, i) * 3u + fresh(2u, i) * 9u + fresh(3u, i) * 27u + fresh(4u, i) * 81u + fresh(5u, i) * 243u + fresh(6u, i) * 729u;
        } else {
            let old = load_phi(i);
            let o = ((y / P.b) * P.nbx + x / P.b) * 32u + REL;
            let rel = Ch(vec4<i32>(blk[o], blk[o + 1u], blk[o + 2u], blk[o + 3u]), vec4<i32>(blk[o + 4u], blk[o + 5u], blk[o + 6u], 0));
            let da = old.a + rel.a - f.a;
            let db = old.b + rel.b - f.b;
            let d = deltas(cell);
            code = kick(0u, d.a.x, da.x, i) + kick(1u, d.a.y, da.y, i) * 3u + kick(2u, d.a.z, da.z, i) * 9u + kick(3u, d.a.w, da.w, i) * 27u
                + kick(4u, d.b.x, db.x, i) * 81u + kick(5u, d.b.y, db.y, i) * 243u + kick(6u, d.b.z, db.z, i) * 729u;
        }
        dst[i] = code;
        store_phi(i, f);
        let nd = deltas(code);
        dsum.a += nd.a;
        dsum.b += nd.b;
    }
    reduce(li, sphi, false);
    if (!init) {
        reduce(li, dsum, true);
    }
    let n = tile_blocks() * 7u;
    for (var t = li; t < n; t += 256u) {
        let k = tile_block(wid.xy, t / 7u);
        let o = k * 32u;
        if (blk[o + FLUID] == 0) {
            continue;
        }
        let c = t % 7u;
        blk[o + SPHI + c] = bsum0[t];
        if (!init && !((P.flags & REFRESH) != 0u && in_zone((k % P.nbx) * P.b))) {
            blk[o + RES + c] -= bsum0[t] + bsum1[t] * ONE;
        }
    }
}

// ---- Read-out ----

@group(0) @binding(7) var<storage, read_write> out: array<i32>;

// phi per site, channel-major (7 planes), for the tests.
@compute @workgroup_size(64)
fn phi_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let n = P.w * P.h;
    let i = y * P.w + x;
    let f = load_phi(i);
    out[i] = f.a.x;
    out[n + i] = f.a.y;
    out[2u * n + i] = f.a.z;
    out[3u * n + i] = f.a.w;
    out[4u * n + i] = f.b.x;
    out[5u * n + i] = f.b.y;
    out[6u * n + i] = f.b.z;
}

// Block sums for the coarse-grained field, added to `out` (cleared by the
// host): jx, jy (units 2 px, 2 py / sqrt 3), mass, each shifted down by
// SUM_SHIFT, and fluid sites, as `Signed::page_sums`. One invocation per
// display block.
struct CoarseParams {
    bw: u32,
    bh: u32,
    nx: u32,
    ny: u32,
}

@group(0) @binding(6) var<uniform> C: CoarseParams;

@compute @workgroup_size(64)
fn coarse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    if (bx >= C.nx || by >= C.ny) {
        return;
    }
    var jx = 0;
    var jy = 0;
    var mass = 0;
    var fluid = 0;
    for (var y = by * C.bh; y < (by + 1u) * C.bh; y++) {
        for (var x = bx * C.bw; x < (bx + 1u) * C.bw; x++) {
            let i = y * P.w + x;
            let cell = src[i];
            if ((cell & OBST) != 0u) {
                continue;
            }
            let o = ((y / P.b) * P.nbx + x / P.b) * 32u;
            let nf = blk[o + FLUID];
            let f = load_phi(i);
            let d = deltas(cell);
            let fa = f.a + d.a * ONE + vec4<i32>(blk[o], blk[o + 1u], blk[o + 2u], blk[o + 3u]) / nf;
            let fb = f.b + d.b * ONE + vec4<i32>(blk[o + 4u] / nf, blk[o + 5u] / nf, blk[o + 6u] / nf, 0);
            mass += fa.x + fa.y + fa.z + fa.w + fb.x + fb.y + fb.z;
            jx += 2 * fa.x + fa.y - fa.z - 2 * fa.w - fb.x + fb.y;
            jy += fa.y + fa.z - fb.x - fb.y;
            fluid += 1;
        }
    }
    let o = (by * C.nx + bx) * 4u;
    out[o] += jx >> 4u;
    out[o + 1u] += jy >> 4u;
    out[o + 2u] += mass >> 4u;
    out[o + 3u] += fluid;
}
