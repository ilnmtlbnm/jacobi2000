//! Stochastic rounding for the b-bit circuit: `(E + W + N + S + r) >> 2`
//! with r in {0, 1, 2, 3} uniform per site per step, two random bit-planes
//! from `rng::chunk_rand`, so E[update] = (E + W + N + S) / 4 exactly.

use crate::rng::{site_bit, step_key};

pub const STREAM_R0: u32 = 10;
pub const STREAM_R1: u32 = 11;

pub fn keys(seed: u32, step: u32) -> [u32; 2] {
    [step_key(seed, step, STREAM_R0), step_key(seed, step, STREAM_R1)]
}

/// The random offset r of site (i, j) under the step's keys.
#[inline(always)]
pub fn offset(keys: [u32; 2], n: usize, i: usize, j: usize) -> u32 {
    site_bit(keys[0], n, i, j) | site_bit(keys[1], n, i, j) << 1
}

pub fn int_step(n: usize, src: &[u32], dst: &mut [u32], seed: u32, step: u32) {
    let k = keys(seed, step);
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            let c = i * n + j;
            dst[c] = (src[c + 1] + src[c - 1] + src[c - n] + src[c + n] + offset(k, n, i, j)) >> 2;
        }
    }
}

/// `iters` steps starting at step index `first`.
pub fn int_run(n: usize, q: &mut Vec<u32>, iters: usize, seed: u32, first: u32) {
    let mut other = q.clone();
    for t in 0..iters as u32 {
        int_step(n, q, &mut other, seed, first + t);
        std::mem::swap(q, &mut other);
    }
}
