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

fn bert(cfg: &Config, p: &Params, cells: &[u8], w: usize, h: usize, out: &mut [f32]) {
    let (d, heads, dff, eps) = (cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.eps);
    let dk = d / heads;
    let (tok, pos) = (p.get("tok_emb.weight"), p.get("pos_emb.weight"));
    let (wq, bq, wk, bk, wv, bv, wo, bo) =
        (p.get("mha.query.weight"), p.get("mha.query.bias"), p.get("mha.key.weight"), p.get("mha.key.bias"), p.get("mha.value.weight"), p.get("mha.value.bias"), p.get("mha.output.weight"), p.get("mha.output.bias"));
    let (g1, be1, g2, be2) = (p.get("norm_1.weight"), p.get("norm_1.bias"), p.get("norm_2.weight"), p.get("norm_2.bias"));
    let (wi, bi, wf, bf, wh, bh) =
        (p.get("pwff.linear_inner.weight"), p.get("pwff.linear_inner.bias"), p.get("pwff.linear_outer.weight"), p.get("pwff.linear_outer.bias"), p.get("head.weight"), p.get("head.bias"));
    let scale = (dk as f32).sqrt();
    for_chunks(out, 2, 64, |first, o| {
        let mut k = vec![0.0f32; 9 * d];
        let mut v = vec![0.0f32; 9 * d];
        let (mut xi, mut x8, mut q, mut att, mut y) = (vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d]);
        let (mut ff, mut z) = (vec![0.0f32; dff], vec![0.0f32; d]);
        let mut sc = [0.0f32; 9];
        for (c, o2) in o.chunks_mut(2).enumerate() {
            let i = first + c;
            let nb = taps(i % w, i / w, w, h);
            for t in 0..9 {
                let j = if t < 8 { nb[t] } else { i };
                let tv = (cells[j] != 0) as usize;
                for e in 0..d {
                    xi[e] = tok[tv * d + e] + pos[t * d + e];
                }
                linear(wk, bk, &xi, &mut k[t * d..(t + 1) * d]);
                linear(wv, bv, &xi, &mut v[t * d..(t + 1) * d]);
                if t == 8 {
                    x8.copy_from_slice(&xi);
                }
            }
            linear(wq, bq, &x8, &mut q);
            for hd in 0..heads {
                let o0 = hd * dk;
                for t in 0..9 {
                    let mut s = 0.0;
                    for e in 0..dk {
                        s += q[o0 + e] * k[t * d + o0 + e];
                    }
                    sc[t] = s / scale;
                }
                softmax(&mut sc);
                for e in 0..dk {
                    let mut a = 0.0;
                    for t in 0..9 {
                        a += sc[t] * v[t * d + o0 + e];
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

fn stencil(cfg: &Config, p: &Params, cells: &[u8], w: usize, h: usize, out: &mut [f32]) {
    const BAND: usize = 8;
    let (d, heads, dff, eps) = (cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.eps);
    let dk = d / heads;
    let (we, be) = (p.get("embed.weight"), p.get("embed.bias"));
    let (g1, be1, g2, be2) = (p.get("blocks.0.norm1.weight"), p.get("blocks.0.norm1.bias"), p.get("blocks.0.norm2.weight"), p.get("blocks.0.norm2.bias"));
    let (wq, bq, wk, bk, wv, bv, wp, bp) = (
        p.get("blocks.0.q.weight"),
        p.get("blocks.0.q.bias"),
        p.get("blocks.0.k.weight"),
        p.get("blocks.0.k.bias"),
        p.get("blocks.0.v.weight"),
        p.get("blocks.0.v.bias"),
        p.get("blocks.0.proj.weight"),
        p.get("blocks.0.proj.bias"),
    );
    let (w1, b1, w2, b2, wh, bh) = (p.get("blocks.0.ff1.weight"), p.get("blocks.0.ff1.bias"), p.get("blocks.0.ff2.weight"), p.get("blocks.0.ff2.bias"), p.get("head.weight"), p.get("head.bias"));
    let scale = (dk as f32).sqrt();
    let embed = move |c: u8, x: &mut [f32]| {
        let cv = (c != 0) as u8 as f32;
        for e in 0..d {
            x[e] = we[e] * cv + be[e];
        }
    };
    // Row bands of BAND rows: k/v are computed once per cell of the band and
    // of the row above and below it, then every cell reads its 9 taps out of
    // that buffer. The buffer's row r is board row (y0 + r - 1) mod h.
    for_chunks(out, 1, BAND * w, |first, o| {
        let y0 = first / w;
        let rows = o.len() / w;
        let mut kv = vec![0.0f32; (rows + 2) * w * 2 * d];
        let (mut x, mut hh) = (vec![0.0f32; d], vec![0.0f32; d]);
        for r in 0..rows + 2 {
            let gy = (y0 + r + h - 1) % h;
            for gx in 0..w {
                embed(cells[gy * w + gx], &mut x);
                hh.copy_from_slice(&x);
                layer_norm(&mut hh, g1, be1, eps);
                let base = (r * w + gx) * 2 * d;
                linear(wk, bk, &hh, &mut kv[base..base + d]);
                linear(wv, bv, &hh, &mut kv[base + d..base + 2 * d]);
            }
        }
        let (mut q, mut att, mut pr, mut ff, mut f2) = (vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; dff], vec![0.0f32; d]);
        let mut sc = [0.0f32; 9];
        for ly in 0..rows {
            let y = y0 + ly;
            for gx in 0..w {
                embed(cells[y * w + gx], &mut x);
                hh.copy_from_slice(&x);
                layer_norm(&mut hh, g1, be1, eps);
                linear(wq, bq, &hh, &mut q);
                // taps NW, N, NE, W, E, SW, S, SE, self as (row offset, col offset)
                const OFF: [(isize, isize); 9] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (0, 0)];
                let tap = |t: usize| -> usize {
                    let (dy, dx) = OFF[t];
                    let r = (ly as isize + 1 + dy) as usize;
                    let c = (gx as isize + dx + w as isize) as usize % w;
                    (r * w + c) * 2 * d
                };
                for hd in 0..heads {
                    let o0 = hd * dk;
                    for (t, st) in sc.iter_mut().enumerate() {
                        let b = tap(t);
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
                            a += sc[t] * kv[tap(t) + d + o0 + e];
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
                o[ly * w + gx] = lg[0];
            }
        }
    });
}
