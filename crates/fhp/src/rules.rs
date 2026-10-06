//! The FHP-III collision table.
//!
//! Model III of d'Humieres and Lallemand (Complex Systems 1, 599 (1987),
//! section 2.2): states are grouped by mass and momentum; a state in a group
//! of q > 1 configurations goes to each of the others with probability
//! 1/(q-1), except the two zero-momentum groups of five (three and four
//! particles), which split into a group of three (rest + head-on pair, or two
//! head-on pairs) and a group of two (the symmetric triples, with or without
//! a rest particle) that do not exchange. Every group is then of size 1, 2 or
//! 3, so one random chirality bit realizes the probabilities exactly: size 2
//! swaps, size 3 steps one way round a 3-cycle (chirality 0) or the other
//! (chirality 1). The cycles are chosen rotation-equivariant.
//!
//! Obstacle sites (bit 7) bounce back: every moving bit is reversed.

use crate::{CX2, CY2, NDIR};
use std::collections::HashMap;

pub fn mass(s: u8) -> u32 {
    (s & 0x7f).count_ones()
}

pub fn momentum(s: u8) -> (i32, i32) {
    let mut p = (0, 0);
    for i in 0..NDIR {
        if s & (1 << i) != 0 {
            p.0 += CX2[i];
            p.1 += CY2[i];
        }
    }
    p
}

/// Rotate the moving bits by 60 degrees `k` times.
pub fn rotate(s: u8, k: usize) -> u8 {
    let m = s & 0x3f;
    let k = k % 6;
    let r = ((m << k) | (m >> (6 - k))) & 0x3f;
    (s & 0xc0) | r
}

/// Reverse every moving bit (bounce-back).
pub fn bounce(s: u8) -> u8 {
    rotate(s, 3)
}

fn is_triple(s: u8) -> bool {
    matches!(s & 0x3f, 0b010101 | 0b101010)
}

/// The group key of a seven-bit fluid state.
fn group(s: u8) -> (u32, i32, i32, bool) {
    let (px, py) = momentum(s);
    let m = mass(s);
    let split = px == 0 && py == 0 && (m == 3 || m == 4);
    (m, px, py, split && is_triple(s))
}

/// Groups of the 128 fluid states, members ascending.
pub fn groups() -> Vec<Vec<u8>> {
    let mut map: HashMap<(u32, i32, i32, bool), Vec<u8>> = HashMap::new();
    let mut order = Vec::new();
    for s in 0u8..128 {
        let k = group(s);
        map.entry(k).or_insert_with(|| {
            order.push(k);
            Vec::new()
        });
        map.get_mut(&k).unwrap().push(s);
    }
    order.into_iter().map(|k| map.remove(&k).unwrap()).collect()
}

/// `table[c][s]`: post-collision state of the eight-bit state `s` with
/// chirality bit `c`.
pub fn table() -> [[u8; 256]; 2] {
    let mut t = [[0u8; 256]; 2];
    let mut done = [false; 128];
    for s in 0u8..128 {
        if done[s as usize] {
            continue;
        }
        let g = groups().into_iter().find(|g| g.contains(&s)).unwrap();
        match g.len() {
            1 => {
                t[0][s as usize] = s;
                t[1][s as usize] = s;
                done[s as usize] = true;
            }
            2 => {
                t[0][g[0] as usize] = g[1];
                t[1][g[0] as usize] = g[1];
                t[0][g[1] as usize] = g[0];
                t[1][g[1] as usize] = g[0];
                done[g[0] as usize] = true;
                done[g[1] as usize] = true;
            }
            3 => {
                let p = momentum(g[0]);
                if p == (0, 0) {
                    // Rotation maps the group onto itself, cyclically.
                    for &m in &g {
                        assert!(g.contains(&rotate(m, 1)));
                        t[0][m as usize] = rotate(m, 1);
                        t[1][m as usize] = rotate(m, 5);
                        done[m as usize] = true;
                    }
                } else {
                    // Fix the cycle on this group, carry it to its five rotations.
                    for k in 0..6 {
                        let c: Vec<u8> = g.iter().map(|&m| rotate(m, k)).collect();
                        for j in 0..3 {
                            t[0][c[j] as usize] = c[(j + 1) % 3];
                            t[1][c[j] as usize] = c[(j + 2) % 3];
                            done[c[j] as usize] = true;
                        }
                    }
                }
            }
            q => panic!("group of size {q}: FHP-III groups are of size 1, 2 or 3"),
        }
    }
    for s in 128usize..256 {
        t[0][s] = bounce(s as u8);
        t[1][s] = bounce(s as u8);
    }
    t
}

/// Number of seven-bit configurations a collision changes (76 in the paper).
pub fn active_configurations(t: &[[u8; 256]; 2]) -> usize {
    (0..128).filter(|&s| t[0][s] != s as u8 || t[1][s] != s as u8).count()
}
