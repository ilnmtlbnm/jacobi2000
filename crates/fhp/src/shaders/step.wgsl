// One FHP step on bit-planes, one invocation per 32-site word:
// stream (pull from the neighbours' post-collision planes), collide with the
// generated circuit (collide_gen.wgsl, prepended), force, store.
// Layout: state[(y * 7 + p) * wn + i], obst[y * wn + i]; same as the CPU
// engine with u32 words.

struct Params {
    wn: u32,
    h: u32,
    kc: u32,
    kf: u32,
    thr: u32,
    step: u32,
    _p1: u32,
    _p2: u32,
}

// This step's keys of the boundary's 8 x 12 random streams (boundary.rs
// step_keys), three vec4 per Bernoulli stream; read by `boundary` only.
struct Keys {
    k: array<vec4<u32>, 24>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> obst: array<u32>;
// Boundary constants (boundary.rs): [0..7) equilibrium thresholds, 7 inlet
// chunks, 8 outlet chunks, 9 open boundary on, 10 walls on, 11 outlet kind
// (0 sponge, 1 randomize, 2 delete).
@group(0) @binding(4) var<storage, read> B: array<u32>;
@group(0) @binding(5) var<uniform> KEYS: Keys;

fn lowbias32(v: u32) -> u32 {
    var x = v;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn chunk_rand(key: u32, idx: u32) -> u32 {
    return lowbias32(key ^ (idx * 0x9e3779b9u));
}

// One bit of the bit-serial comparison R < t (boundary.rs bern).
fn bstep(lt: ptr<function, u32>, eq: ptr<function, u32>, key: u32, idx: u32, tb: u32) {
    if (*eq == 0u) {
        return;
    }
    let r = chunk_rand(key, idx);
    if (tb == 1u) {
        *lt |= *eq & ~r;
        *eq &= r;
    } else {
        *eq &= ~r;
    }
}

// 32 Bernoulli(t / 4096) bits of chunk idx from Bernoulli stream s (a
// constant at every call, so the keys are read with constant indices).
fn bern(s: u32, idx: u32, t: u32) -> u32 {
    if (t >= 4096u) {
        return 0xffffffffu;
    }
    if (t == 0u) {
        return 0u;
    }
    let k0 = KEYS.k[s * 3u];
    let k1 = KEYS.k[s * 3u + 1u];
    let k2 = KEYS.k[s * 3u + 2u];
    var lt = 0u;
    var eq = 0xffffffffu;
    bstep(&lt, &eq, k0.x, idx, (t >> 11u) & 1u);
    bstep(&lt, &eq, k0.y, idx, (t >> 10u) & 1u);
    bstep(&lt, &eq, k0.z, idx, (t >> 9u) & 1u);
    bstep(&lt, &eq, k0.w, idx, (t >> 8u) & 1u);
    bstep(&lt, &eq, k1.x, idx, (t >> 7u) & 1u);
    bstep(&lt, &eq, k1.y, idx, (t >> 6u) & 1u);
    bstep(&lt, &eq, k1.z, idx, (t >> 5u) & 1u);
    bstep(&lt, &eq, k1.w, idx, (t >> 4u) & 1u);
    bstep(&lt, &eq, k2.x, idx, (t >> 3u) & 1u);
    bstep(&lt, &eq, k2.y, idx, (t >> 2u) & 1u);
    bstep(&lt, &eq, k2.z, idx, (t >> 1u) & 1u);
    bstep(&lt, &eq, k2.w, idx, t & 1u);
    return lt;
}

// Replacement threshold of chunk column c (boundary.rs Boundary::q):
// inlet 1; outlet 1 (randomize, delete) or a quadratic ramp to 1 (sponge).
fn qthr(c: u32) -> u32 {
    if (c < B[7]) {
        return 4096u;
    }
    let start = P.wn - B[8];
    if (c + 1u == P.wn || (c >= start && B[11] != 0u)) {
        return 4096u;
    }
    if (c >= start) {
        let k = c - start + 1u;
        return (4096u * k * k) / (B[8] * B[8]);
    }
    return 0u;
}

// Word i of plane p in row sy, shifted so bit b holds site (32 i + b - dx).
fn fetch(p: u32, sy: u32, i: u32, dx: i32) -> u32 {
    let base = (sy * 7u + p) * P.wn;
    let s = src[base + i];
    if (dx == 1) {
        let prev = src[base + select(i - 1u, P.wn - 1u, i == 0u)];
        return (s << 1u) | (prev >> 31u);
    }
    if (dx == -1) {
        let next = src[base + select(i + 1u, 0u, i + 1u == P.wn)];
        return (s >> 1u) | (next << 31u);
    }
    return s;
}

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let y = gid.y;
    if (i >= P.wn || y >= P.h) {
        return;
    }
    let ym = select(y - 1u, P.h - 1u, y == 0u);
    let yp = select(y + 1u, 0u, y + 1u == P.h);
    // Row parity of the source rows decides the diagonal shifts.
    let om = i32(ym & 1u);
    let op = i32(yp & 1u);
    let a0 = fetch(0u, y, i, 1);
    let a1 = fetch(1u, ym, i, om);
    let a2 = fetch(2u, ym, i, om - 1);
    let a3 = fetch(3u, y, i, -1);
    let a4 = fetch(4u, yp, i, op - 1);
    let a5 = fetch(5u, yp, i, op);
    let r = src[(y * 7u + 6u) * P.wn + i];
    let ob = obst[y * P.wn + i];
    let idx = y * P.wn + i;
    let c = chunk_rand(P.kc, idx);
    var o = collide(a0, a1, a2, a3, a4, a5, r, ob, c);
    var wall = false;
    // The host drops the lines from WALLS{ to }WALLS when walls are off, so
    // the step kernel is then the periodic one, with no boundary code.
    // WALLS{
    wall = B[10] == 1u && (y == 0u || y == P.h - 1u);
    if (wall) {
        // Free-slip: fluid sites mirror NE <-> SE, NW <-> SW.
        let m = array<u32, 7>(a0, a5, a4, a3, a2, a1, r);
        for (var k = 0u; k < 7u; k++) {
            o[k] = (o[k] & ob) | (m[k] & ~ob);
        }
    }
    // }WALLS
    if (P.thr > 0u && !wall) {
        let hf = chunk_rand(P.kf, idx);
        var f = 0u;
        if ((hf >> 5u) < P.thr) {
            f = 1u << (hf & 31u);
        }
        f = f & ~ob & o[3] & ~o[0];
        o[0] ^= f;
        o[3] ^= f;
    }
    let base = y * 7u * P.wn + i;
    for (var p = 0u; p < 7u; p++) {
        dst[base + p * P.wn] = o[p];
    }
}

// Open boundary refresh, its own dispatch after `step` on every 32nd step
// (boundary.rs EVERY): one invocation per row and boundary chunk (the inlet
// chunks, then the outlet chunks), in place on the step's output.
@compute @workgroup_size(64)
fn boundary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_in = B[7];
    let n_out = B[8];
    let y = gid.y;
    if (gid.x >= n_in + n_out || y >= P.h) {
        return;
    }
    if (B[10] == 1u && (y == 0u || y == P.h - 1u)) {
        return;
    }
    var i = gid.x;
    if (i >= n_in) {
        i = P.wn - n_out + (i - n_in);
        if (i < n_in) {
            return;
        }
    }
    let q = qthr(i);
    if (q == 0u) {
        return;
    }
    let idx = y * P.wn + i;
    let mask = bern(7u, idx, q) & ~obst[idx];
    if (mask == 0u) {
        return;
    }
    let base = y * 7u * P.wn + i;
    let del = B[11] == 2u && i >= n_in;
    for (var p = 0u; p < 7u; p++) {
        let a = base + p * P.wn;
        var fresh = 0u;
        if (!del) {
            fresh = bern(p, idx, B[p]);
        }
        dst[a] = (dst[a] & ~mask) | (fresh & mask);
    }
}

// Block sums for the coarse-grained field, added to `sums` (cleared by the
// host): one invocation per block of bw (dividing 32) x bh sites; jx, jy in
// units (2 px, 2 py / sqrt 3).
struct CoarseParams {
    wn: u32,
    h: u32,
    bw: u32,
    bh: u32,
    nx: u32,
    ny: u32,
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<uniform> C: CoarseParams;
@group(0) @binding(1) var<storage, read> cstate: array<u32>;
@group(0) @binding(2) var<storage, read> cobst: array<u32>;
@group(0) @binding(3) var<storage, read_write> sums: array<i32>;

@compute @workgroup_size(64)
fn coarse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    if (bx >= C.nx || by >= C.ny) {
        return;
    }
    let x0 = bx * C.bw;
    let i = x0 / 32u;
    var m = 0xffffffffu;
    if (C.bw < 32u) {
        m = ((1u << C.bw) - 1u) << (x0 % 32u);
    }
    var jx = 0i;
    var jy = 0i;
    var mass = 0i;
    var fluid = 0i;
    for (var y = by * C.bh; y < (by + 1u) * C.bh; y++) {
        var n: array<i32, 7>;
        for (var p = 0u; p < 7u; p++) {
            n[p] = i32(countOneBits(cstate[(y * 7u + p) * C.wn + i] & m));
            mass += n[p];
        }
        jx += 2 * n[0] + n[1] - n[2] - 2 * n[3] - n[4] + n[5];
        jy += n[1] + n[2] - n[4] - n[5];
        fluid += i32(C.bw) - i32(countOneBits(cobst[y * C.wn + i] & m));
    }
    let b = (by * C.nx + bx) * 4u;
    sums[b] += jx;
    sums[b + 1u] += jy;
    sums[b + 2u] += mass;
    sums[b + 3u] += fluid;
}
