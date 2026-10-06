// One ternary FHP step on 14 bit-planes (two per channel), one invocation
// per 32-site word: stream (pull from the neighbours' post-collision
// planes), collide with the generated circuit (ternary_collide_gen.wgsl,
// prepended; rule Chiral), store. Periodic in x and y.
// Layout: state[(y * 14 + p) * wn + i], obst[y * wn + i].

struct Params {
    wn: u32,
    h: u32,
    kc: u32,
    _p: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> obst: array<u32>;

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

// Word i of plane p in row sy, shifted so bit b holds site (32 i + b - dx).
fn fetch(p: u32, sy: u32, i: u32, dx: i32) -> u32 {
    let base = (sy * 14u + p) * P.wn;
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
    let om = i32(ym & 1u);
    let op = i32(yp & 1u);
    let x = array<u32, 16>(
        fetch(0u, y, i, 1),
        fetch(1u, y, i, 1),
        fetch(2u, ym, i, om),
        fetch(3u, ym, i, om),
        fetch(4u, ym, i, om - 1),
        fetch(5u, ym, i, om - 1),
        fetch(6u, y, i, -1),
        fetch(7u, y, i, -1),
        fetch(8u, yp, i, op - 1),
        fetch(9u, yp, i, op - 1),
        fetch(10u, yp, i, op),
        fetch(11u, yp, i, op),
        src[(y * 14u + 12u) * P.wn + i],
        src[(y * 14u + 13u) * P.wn + i],
        obst[y * P.wn + i],
        chunk_rand(P.kc, y * P.wn + i),
    );
    let o = collide(x);
    let base = y * 14u * P.wn + i;
    dst[base + 0u * P.wn] = o[0];
    dst[base + 1u * P.wn] = o[1];
    dst[base + 2u * P.wn] = o[2];
    dst[base + 3u * P.wn] = o[3];
    dst[base + 4u * P.wn] = o[4];
    dst[base + 5u * P.wn] = o[5];
    dst[base + 6u * P.wn] = o[6];
    dst[base + 7u * P.wn] = o[7];
    dst[base + 8u * P.wn] = o[8];
    dst[base + 9u * P.wn] = o[9];
    dst[base + 10u * P.wn] = o[10];
    dst[base + 11u * P.wn] = o[11];
    dst[base + 12u * P.wn] = o[12];
    dst[base + 13u * P.wn] = o[13];
}
