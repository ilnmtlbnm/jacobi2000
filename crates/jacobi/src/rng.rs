//! Deterministic random streams, the same as the fhp crate's: every random
//! bit is a function of (seed, stream, step, row, 32-site chunk), so the
//! per-site references, the bit-sliced CPU engines (u64 words = two chunks)
//! and the GPU engines (u32 words = one chunk) draw the same bits.

/// lowbias32 (Chris Wellons' hash-prospector), the same in WGSL.
#[inline(always)]
pub fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Seed of a random stream, constant over steps.
pub fn stream_seed(seed: u32, stream: u32) -> u32 {
    lowbias32(seed ^ lowbias32(stream.wrapping_add(0x9e37_79b9)))
}

/// Per-step key of a random stream.
pub fn step_key(seed: u32, step: u32, stream: u32) -> u32 {
    lowbias32(stream_seed(seed, stream) ^ step)
}

/// 32 random bits for the 32-site chunk `idx = row * (n / 32) + col / 32`.
#[inline(always)]
pub fn chunk_rand(key: u32, idx: u32) -> u32 {
    lowbias32(key ^ idx.wrapping_mul(0x9e37_79b9))
}

/// The random bit of site (i, j) on an n-wide grid.
#[inline(always)]
pub fn site_bit(key: u32, n: usize, i: usize, j: usize) -> u32 {
    (chunk_rand(key, (i * (n / 32) + j / 32) as u32) >> (j % 32)) & 1
}

/// A 64-site u64 word of random bits: chunks `2 c` (low half) and `2 c + 1`.
#[inline(always)]
pub fn word_rand(key: u32, n: usize, row: usize, c64: usize) -> u64 {
    let idx = (row * (n / 32) + 2 * c64) as u32;
    chunk_rand(key, idx) as u64 | (chunk_rand(key, idx + 1) as u64) << 32
}

pub const WGSL: &str = "
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
";
