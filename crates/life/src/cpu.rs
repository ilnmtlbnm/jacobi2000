//! CPU forward passes over a whole board: one logit row per cell, no
//! intermediate tensor larger than a row band. The arithmetic order (bias
//! first, then the products in index order; softmax subtracting the max;
//! `1/sqrt(var + eps)` LayerNorm) is the order `src/shaders/*.wgsl` uses.
//!
//! BERT of Life keeps only what the read-out needs: the encoder is one
//! post-LN layer and the head reads the centre token, so the centre token's
//! query attends over all 9 keys/values and the other 8 tokens' later
//! stages are never computed. The result is identical to running the full
//! encoder and slicing token 8.

#[cfg(feature = "threads")]
use rayon::prelude::*;

use crate::model::{Config, Kind, Params};

const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// The 8 neighbours of cell `(x, y)` on the torus, in the order NW, N, NE,
/// W, E, SW, S, SE.
pub fn taps(x: usize, y: usize, w: usize, h: usize) -> [usize; 8] {
    let xm = (x + w - 1) % w;
    let xp = (x + 1) % w;
    let ym = (y + h - 1) % h;
    let yp = (y + 1) % h;
    [ym * w + xm, ym * w + x, ym * w + xp, y * w + xm, y * w + xp, yp * w + xm, yp * w + x, yp * w + xp]
}

fn linear(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    let n_in = x.len();
    for (o, out_o) in out.iter_mut().enumerate() {
        let row = &w[o * n_in..(o + 1) * n_in];
        let mut acc = b[o];
        for j in 0..n_in {
            acc += row[j] * x[j];
        }
        *out_o = acc;
    }
}

fn layer_norm(x: &mut [f32], g: &[f32], b: &[f32], eps: f32) {
    let n = x.len() as f32;
    let mut mean = 0.0;
    for v in x.iter() {
        mean += *v;
    }
    mean /= n;
    let mut var = 0.0;
    for v in x.iter() {
        let d = *v - mean;
        var += d * d;
    }
    let inv = 1.0 / (var / n + eps).sqrt();
    for (j, v) in x.iter_mut().enumerate() {
        *v = (*v - mean) * inv * g[j] + b[j];
    }
}

fn gelu(v: f32) -> f32 {
    0.5 * v * (1.0 + libm::erff(v * INV_SQRT2))
}

/// `scores` -> softmax weights in place.
fn softmax(s: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for v in s.iter() {
        m = m.max(*v);
    }
    let mut sum = 0.0;
    for v in s.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    for v in s.iter_mut() {
        *v /= sum;
    }
}

/// Runs `f(first_cell, out_chunk)` over `out` in chunks of `chunk` cells
/// (`per_cell` floats each), in parallel under the `threads` feature.
fn for_chunks(out: &mut [f32], per_cell: usize, chunk: usize, f: impl Fn(usize, &mut [f32]) + Sync + Send) {
    #[cfg(feature = "threads")]
    out.par_chunks_mut(chunk * per_cell).enumerate().for_each(|(c, o)| f(c * chunk, o));
    #[cfg(not(feature = "threads"))]
    out.chunks_mut(chunk * per_cell).enumerate().for_each(|(c, o)| f(c * chunk, o));
}

/// Logits for every cell: `[n * 2]` (`[dead, alive]`) for `bert` and `mlp2`,
/// `[n]` for `stencil`. A cell counts as alive if its byte is nonzero.
pub fn logits(cfg: &Config, p: &Params, cells: &[u8], w: usize, h: usize) -> Vec<f32> {
    assert_eq!(cells.len(), w * h, "board is {w}x{h} but has {} cells", cells.len());
    let mut out = vec![0.0f32; w * h * cfg.kind.outputs()];
    if out.is_empty() {
        return out;
    }
    match cfg.kind {
        Kind::Mlp2 => mlp2(cfg, p, cells, w, h, &mut out),
        Kind::Bert => bert(cfg, p, cells, w, h, &mut out),
        Kind::Stencil => stencil(cfg, p, cells, w, h, &mut out),
    }
    out
}

fn mlp2(cfg: &Config, p: &Params, cells: &[u8], w: usize, h: usize, out: &mut [f32]) {
    let hid = cfg.hidden;
    let (w1, b1, w2, b2, w3, b3) = (p.get("fc1.weight"), p.get("fc1.bias"), p.get("fc2.weight"), p.get("fc2.bias"), p.get("fc3.weight"), p.get("fc3.bias"));
    for_chunks(out, 2, 256, |first, o| {
        let (mut a, mut b) = (vec![0.0f32; hid], vec![0.0f32; hid]);
        for (k, o2) in o.chunks_mut(2).enumerate() {
            let i = first + k;
            let nb = taps(i % w, i / w, w, h);
            let mut x = [0.0f32; 9];
            for (t, j) in nb.iter().enumerate() {
                x[t] = (cells[*j] != 0) as u8 as f32;
            }
            x[8] = (cells[i] != 0) as u8 as f32;
            linear(w1, b1, &x, &mut a);
            a.iter_mut().for_each(|v| *v = v.max(0.0));
            linear(w2, b2, &a, &mut b);
            b.iter_mut().for_each(|v| *v = v.max(0.0));
            linear(w3, b3, &b, o2);
        }
    });
}

/// BERT of Life's first-layer projections for every token the encoder can
/// see. A token is an id 0 or 1 at one of 9 positions, so its key and value
/// (`[position * 2 + id][k (d), v (d)]`, 18 rows) and the centre token's
/// query (`[id][d]`, 2 rows) are computed here once per model, with the
/// same arithmetic a per-cell pass would use.
pub fn bert_tables(cfg: &Config, p: &Params) -> (Vec<f32>, Vec<f32>) {
    let d = cfg.d_model;
    let (tok, pos) = (p.get("tok_emb.weight"), p.get("pos_emb.weight"));
    let (wq, bq, wk, bk, wv, bv) = (p.get("mha.query.weight"), p.get("mha.query.bias"), p.get("mha.key.weight"), p.get("mha.key.bias"), p.get("mha.value.weight"), p.get("mha.value.bias"));
    let mut kv = vec![0.0f32; 18 * 2 * d];
    let mut q = vec![0.0f32; 2 * d];
    let mut xi = vec![0.0f32; d];
    for t in 0..9 {
        for id in 0..2 {
            for e in 0..d {
                xi[e] = tok[id * d + e] + pos[t * d + e];
            }
            let row = &mut kv[(t * 2 + id) * 2 * d..(t * 2 + id + 1) * 2 * d];
            linear(wk, bk, &xi, &mut row[..d]);
            linear(wv, bv, &xi, &mut row[d..]);
            if t == 8 {
                linear(wq, bq, &xi, &mut q[id * d..(id + 1) * d]);
            }
        }
    }
    (kv, q)
}

fn bert(cfg: &Config, p: &Params, cells: &[u8], w: usize, h: usize, out: &mut [f32]) {
    let (d, heads, dff, eps) = (cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.eps);
    let dk = d / heads;
    let (tok, pos) = (p.get("tok_emb.weight"), p.get("pos_emb.weight"));
    let (wo, bo) = (p.get("mha.output.weight"), p.get("mha.output.bias"));
    let (g1, be1, g2, be2) = (p.get("norm_1.weight"), p.get("norm_1.bias"), p.get("norm_2.weight"), p.get("norm_2.bias"));
    let (wi, bi, wf, bf, wh, bh) =
        (p.get("pwff.linear_inner.weight"), p.get("pwff.linear_inner.bias"), p.get("pwff.linear_outer.weight"), p.get("pwff.linear_outer.bias"), p.get("head.weight"), p.get("head.bias"));
    let scale = (dk as f32).sqrt();
    let (kv, qt) = bert_tables(cfg, p);
    for_chunks(out, 2, 64, |first, o| {
        let (mut x8, mut att, mut y) = (vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d]);
        let (mut ff, mut z) = (vec![0.0f32; dff], vec![0.0f32; d]);
        let mut sc = [0.0f32; 9];
        for (c, o2) in o.chunks_mut(2).enumerate() {
            let i = first + c;
            let nb = taps(i % w, i / w, w, h);
            let mut row = [0usize; 9];
            for (t, r) in row.iter_mut().enumerate() {
                *r = (t * 2 + (cells[if t < 8 { nb[t] } else { i }] != 0) as usize) * 2 * d;
            }
            let id8 = (cells[i] != 0) as usize;
            for e in 0..d {
                x8[e] = tok[id8 * d + e] + pos[8 * d + e];
            }
            let q = &qt[id8 * d..(id8 + 1) * d];
            for hd in 0..heads {
                let o0 = hd * dk;
                for (t, st) in sc.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for e in 0..dk {
                        s += q[o0 + e] * kv[row[t] + o0 + e];
                    }
                    *st = s / scale;
                }
                softmax(&mut sc);
                for e in 0..dk {
                    let mut a = 0.0;
                    for t in 0..9 {
                        a += sc[t] * kv[row[t] + d + o0 + e];
                    }
                    att[o0 + e] = a;
                }
            }
            linear(wo, bo, &att, &mut y);
            for e in 0..d {
                y[e] += x8[e];
            }
            layer_norm(&mut y, g1, be1, eps);
            linear(wi, bi, &y, &mut ff);
            ff.iter_mut().for_each(|t| *t = gelu(*t));
            linear(wf, bf, &ff, &mut z);
            for e in 0..d {
                z[e] += y[e];
            }
            layer_norm(&mut z, g2, be2, eps);
            linear(wh, bh, &z, o2);
        }
    });
}

/// Keys and values of the stencil block for a cell in state 0 and in state
/// 1, `[state][k (d), v (d)]`. A tap's key and value depend only on that
/// cell's state, so they are computed here once per model, with the same
/// arithmetic a per-cell pass would use, and every cell reads its 9 taps
/// from this table.
pub fn stencil_kv(cfg: &Config, p: &Params) -> Vec<f32> {
    let d = cfg.d_model;
    let (we, be) = (p.get("embed.weight"), p.get("embed.bias"));
    let (g1, be1) = (p.get("blocks.0.norm1.weight"), p.get("blocks.0.norm1.bias"));
    let (wk, bk, wv, bv) = (p.get("blocks.0.k.weight"), p.get("blocks.0.k.bias"), p.get("blocks.0.v.weight"), p.get("blocks.0.v.bias"));
    let mut out = vec![0.0f32; 2 * 2 * d];
    let mut hh = vec![0.0f32; d];
    for state in 0..2 {
        for e in 0..d {
            hh[e] = we[e] * state as f32 + be[e];
        }
        layer_norm(&mut hh, g1, be1, cfg.eps);
        let row = &mut out[state * 2 * d..(state + 1) * 2 * d];
        linear(wk, bk, &hh, &mut row[..d]);
        linear(wv, bv, &hh, &mut row[d..]);
    }
    out
}

fn stencil(cfg: &Config, p: &Params, cells: &[u8], w: usize, h: usize, out: &mut [f32]) {
    let (d, heads, dff, eps) = (cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.eps);
    let dk = d / heads;
    let (we, be) = (p.get("embed.weight"), p.get("embed.bias"));
    let (g1, be1, g2, be2) = (p.get("blocks.0.norm1.weight"), p.get("blocks.0.norm1.bias"), p.get("blocks.0.norm2.weight"), p.get("blocks.0.norm2.bias"));
    let (wq, bq, wp, bp) = (p.get("blocks.0.q.weight"), p.get("blocks.0.q.bias"), p.get("blocks.0.proj.weight"), p.get("blocks.0.proj.bias"));
    let (w1, b1, w2, b2, wh, bh) = (p.get("blocks.0.ff1.weight"), p.get("blocks.0.ff1.bias"), p.get("blocks.0.ff2.weight"), p.get("blocks.0.ff2.bias"), p.get("head.weight"), p.get("head.bias"));
    let scale = (dk as f32).sqrt();
    let kv = stencil_kv(cfg, p);
    for_chunks(out, 1, 64, |first, o| {
        let (mut x, mut hh, mut q, mut att, mut pr) = (vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d]);
        let (mut ff, mut f2) = (vec![0.0f32; dff], vec![0.0f32; d]);
        let mut sc = [0.0f32; 9];
        for (c, o1) in o.iter_mut().enumerate() {
            let i = first + c;
            let nb = taps(i % w, i / w, w, h);
            let mut state = [0usize; 9];
            for t in 0..9 {
                state[t] = (cells[if t < 8 { nb[t] } else { i }] != 0) as usize;
            }
            let cv = state[8] as f32;
            for e in 0..d {
                x[e] = we[e] * cv + be[e];
            }
            hh.copy_from_slice(&x);
            layer_norm(&mut hh, g1, be1, eps);
            linear(wq, bq, &hh, &mut q);
            for hd in 0..heads {
                let o0 = hd * dk;
                for (t, st) in sc.iter_mut().enumerate() {
                    let b = state[t] * 2 * d;
                    let mut s = 0.0;
                    for e in 0..dk {
                        s += q[o0 + e] * kv[b + o0 + e];
                    }
                    *st = s / scale;
                }
                softmax(&mut sc);
                for e in 0..dk {
                    let mut a = 0.0;
                    for t in 0..9 {
                        a += sc[t] * kv[state[t] * 2 * d + d + o0 + e];
                    }
                    att[o0 + e] = a;
                }
            }
            linear(wp, bp, &att, &mut pr);
            for e in 0..d {
                x[e] += pr[e];
            }
            hh.copy_from_slice(&x);
            layer_norm(&mut hh, g2, be2, eps);
            linear(w1, b1, &hh, &mut ff);
            ff.iter_mut().for_each(|t| *t = gelu(*t));
            linear(w2, b2, &ff, &mut f2);
            for e in 0..d {
                x[e] += f2[e];
            }
            let mut lg = [0.0f32; 1];
            linear(wh, bh, &x, &mut lg);
            *o1 = lg[0];
        }
    });
}
