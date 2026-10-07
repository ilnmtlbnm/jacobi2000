//! The forward pass, hand-written over flat `f32` buffers: no tensor
//! framework, no per-op dispatch overhead, one function per layer type,
//! mirroring `jacobi2000::model::Jacobi2000::forward` operation for
//! operation (same patchify order, same head-split convention, same conv
//! padding, same upsample weights) so the two stay numerically identical.
//!
//! Row layout for a `[rows, cols]` matrix is always `row * cols + col`
//! (the mathematical convention `Linear` already stores its weight in:
//! `y = x @ W + b`, `W: [in, out]`). The grid layout for a `[c, h, w]` image
//! is `c * h * w + y * w + x`.
//!
//! Single-threaded by default; the `threads` feature parallelizes the hot
//! loops (linear, layernorm, attention per query row, conv2d per output row)
//! over rayon, picked at the call site by `lean_runtime`'s capability
//! ladder — never autotuned, never per-device.

use crate::config::Config;
use crate::mask::Mask;
use crate::weights::{get, Weights};

#[cfg(feature = "threads")]
use rayon::prelude::*;

/// `y = x @ w + b`. `x: [rows, din]`, `w: [din, dout]`, `b: [dout]`, `out:
/// [rows, dout]`.
fn linear(x: &[f32], rows: usize, din: usize, w: &[f32], b: &[f32], dout: usize, out: &mut [f32]) {
    let row = |r: usize, out_row: &mut [f32]| {
        out_row.copy_from_slice(&b[..dout]);
        let xr = &x[r * din..(r + 1) * din];
        for (i, &xi) in xr.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let wr = &w[i * dout..(i + 1) * dout];
            for o in 0..dout {
                out_row[o] += xi * wr[o];
            }
        }
    };
    #[cfg(feature = "threads")]
    out.par_chunks_mut(dout).enumerate().for_each(|(r, out_row)| row(r, out_row));
    #[cfg(not(feature = "threads"))]
    out.chunks_mut(dout).enumerate().for_each(|(r, out_row)| row(r, out_row));
    let _ = rows;
}

/// Burn's `LayerNorm::forward`: biased variance, `eps` added before the
/// square root, `gamma`/`beta` both `[dim]`.
fn layernorm(x: &[f32], rows: usize, dim: usize, gamma: &[f32], beta: &[f32], out: &mut [f32]) {
    const EPS: f32 = 1e-5;
    let row = |r: usize, out_row: &mut [f32]| {
        let xr = &x[r * dim..(r + 1) * dim];
        let mean = xr.iter().sum::<f32>() / dim as f32;
        let var = xr.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv_std = 1.0 / (var + EPS).sqrt();
        for d in 0..dim {
            out_row[d] = (xr[d] - mean) * inv_std * gamma[d] + beta[d];
        }
    };
    #[cfg(feature = "threads")]
    out.par_chunks_mut(dim).enumerate().for_each(|(r, out_row)| row(r, out_row));
    #[cfg(not(feature = "threads"))]
    out.chunks_mut(dim).enumerate().for_each(|(r, out_row)| row(r, out_row));
    let _ = rows;
}

/// Exact erf-based GELU (`0.5 x (1 + erf(x / sqrt(2)))`), matching Burn's
/// `activation::gelu` (not the tanh approximation).
fn gelu_inplace(x: &mut [f32]) {
    const INV_SQRT2: f32 = std::f32::consts::SQRT_2.recip();
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + libm::erff(*v * INV_SQRT2));
    }
}

fn add_inplace(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// `state + forward(state, meta)`: one tick. Public so callers never need to
/// re-derive the residual convention.
pub fn step(cfg: &Config, w: &Weights, state: &[f32], meta: &[f32]) -> Vec<f32> {
    let mut out = forward(cfg, w, state, meta);
    add_inplace(&mut out, state);
    out
}

/// Multi-head attention with an additive mask applied as a boolean
/// predicate (equivalent to Burn's `-inf` additive mask before softmax,
/// without materializing a `[tokens, tokens]` matrix). `h: [tokens, dim]`,
/// `out: [tokens, dim]`, heads occupy contiguous `dim` ranges
/// (`[head*hd, (head+1)*hd)`), matching
/// `y.reshape([b,t,heads,hd]).permute([0,2,1,3])`.
#[allow(clippy::too_many_arguments)]
fn attention(
    h: &[f32],
    tokens: usize,
    dim: usize,
    heads: usize,
    tokens_x: usize,
    mask: &Mask,
    wq: &[f32],
    bq: &[f32],
    wk: &[f32],
    bk: &[f32],
    wv: &[f32],
    bv: &[f32],
    out: &mut [f32],
) {
    let hd = dim / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut q = vec![0.0f32; tokens * dim];
    let mut k = vec![0.0f32; tokens * dim];
    let mut v = vec![0.0f32; tokens * dim];
    linear(h, tokens, dim, wq, bq, dim, &mut q);
    linear(h, tokens, dim, wk, bk, dim, &mut k);
    linear(h, tokens, dim, wv, bv, dim, &mut v);

    let row = |qi: usize, out_row: &mut [f32]| {
        let mut scores = vec![0.0f32; tokens];
        for head in 0..heads {
            let qh = &q[qi * dim + head * hd..qi * dim + (head + 1) * hd];
            let mut max_s = f32::NEG_INFINITY;
            let mut any = false;
            for (ki, s) in scores.iter_mut().enumerate() {
                if mask.allowed(tokens_x, qi, ki) {
                    let kh = &k[ki * dim + head * hd..ki * dim + (head + 1) * hd];
                    let dot: f32 = qh.iter().zip(kh).map(|(a, b)| a * b).sum();
                    *s = dot * scale;
                    if *s > max_s {
                        max_s = *s;
                    }
                    any = true;
                } else {
                    *s = f32::NEG_INFINITY;
                }
            }
            debug_assert!(any, "a token must always attend to at least itself");
            let mut denom = 0.0f32;
            for s in scores.iter_mut() {
                *s = if s.is_finite() { (*s - max_s).exp() } else { 0.0 };
                denom += *s;
            }
            let out_h = &mut out_row[head * hd..(head + 1) * hd];
            out_h.fill(0.0);
            for (ki, &w) in scores.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let vh = &v[ki * dim + head * hd..ki * dim + (head + 1) * hd];
                let wn = w / denom;
                for (o, &vi) in out_h.iter_mut().zip(vh) {
                    *o += wn * vi;
                }
            }
        }
    };
    #[cfg(feature = "threads")]
    out.par_chunks_mut(dim).enumerate().for_each(|(qi, out_row)| row(qi, out_row));
    #[cfg(not(feature = "threads"))]
    out.chunks_mut(dim).enumerate().for_each(|(qi, out_row)| row(qi, out_row));
}

/// Zero-padded 2D cross-correlation (PyTorch/Burn `conv2d` convention, no
/// kernel flip). `input: [cin, h, w]`, `weight: [cout, cin, k, k]`, `bias:
/// [cout]`, `pad = k / 2` (odd kernels only, matching
/// `PaddingConfig2d::Explicit(k/2, k/2)`), `out: [cout, h, w]`.
#[allow(clippy::too_many_arguments)]
fn conv2d(input: &[f32], cin: usize, h: usize, w: usize, weight: &[f32], bias: &[f32], cout: usize, k: usize, out: &mut [f32]) {
    let pad = k / 2;
    let plane = h * w;
    let at = |c: usize, y: isize, x: isize| -> f32 {
        if y < 0 || x < 0 || y as usize >= h || x as usize >= w {
            0.0
        } else {
            input[c * plane + y as usize * w + x as usize]
        }
    };
    debug_assert_eq!(out.len(), cout * plane);
    let row = |co: usize, out_plane: &mut [f32]| {
        let wbase = co * cin * k * k;
        for y in 0..h {
            for x in 0..w {
                let mut acc = bias[co];
                for ci in 0..cin {
                    let wci = wbase + ci * k * k;
                    for ky in 0..k {
                        for kx in 0..k {
                            let iy = y as isize + ky as isize - pad as isize;
                            let ix = x as isize + kx as isize - pad as isize;
                            acc += at(ci, iy, ix) * weight[wci + ky * k + kx];
                        }
                    }
                }
                out_plane[y * w + x] = acc;
            }
        }
    };
    #[cfg(feature = "threads")]
    out.par_chunks_mut(plane).enumerate().for_each(|(co, p)| row(co, p));
    #[cfg(not(feature = "threads"))]
    out.chunks_mut(plane).enumerate().for_each(|(co, p)| row(co, p));
}

/// Linear upsample along the `y` axis by `p`, replicating the last row so
/// the far edge has a neighbour to interpolate towards: cell `i` of token
/// `t` mixes `t` and `t+1` with weight `(i + 0.5) / p`, matching
/// `jacobi2000::model::upsample`. `input: [c, h, w]`, `out: [c, h*p, w]`.
fn upsample_y(input: &[f32], c: usize, h: usize, w: usize, p: usize, out: &mut [f32]) {
    for ci in 0..c {
        for yt in 0..h {
            let next = (yt + 1).min(h - 1);
            let row_a = &input[ci * h * w + yt * w..ci * h * w + (yt + 1) * w];
            let row_b = &input[ci * h * w + next * w..ci * h * w + (next + 1) * w];
            for i in 0..p {
                let wgt = (i as f32 + 0.5) / p as f32;
                let y = yt * p + i;
                let out_row = &mut out[ci * h * p * w + y * w..ci * h * p * w + (y + 1) * w];
                for x in 0..w {
                    out_row[x] = row_a[x] * (1.0 - wgt) + row_b[x] * wgt;
                }
            }
        }
    }
}

/// Same idea along the `x` axis. `input: [c, h, w]`, `out: [c, h, w*p]`.
fn upsample_x(input: &[f32], c: usize, h: usize, w: usize, p: usize, out: &mut [f32]) {
    for ci in 0..c {
        for y in 0..h {
            let in_row = &input[ci * h * w + y * w..ci * h * w + (y + 1) * w];
            let out_row = &mut out[ci * h * w * p + y * w * p..ci * h * w * p + (y + 1) * w * p];
            for xt in 0..w {
                let next = (xt + 1).min(w - 1);
                let (a, b) = (in_row[xt], in_row[next]);
                for i in 0..p {
                    let wgt = (i as f32 + 0.5) / p as f32;
                    out_row[xt * p + i] = a * (1.0 - wgt) + b * wgt;
                }
            }
        }
    }
}

/// `state: [channels, ny, nx]` normalized, `meta: [meta_dim]`. Returns the
/// predicted increment, same shape as `state`.
pub fn forward(cfg: &Config, w: &Weights, state: &[f32], meta: &[f32]) -> Vec<f32> {
    let (tx, ty, tdim, dim) = (cfg.tokens_x, cfg.tokens_y, cfg.patch_dim, cfg.dim);
    let tokens = cfg.tokens;

    // Patchify: token t = ty*tokens_x+tx, feature d = c*patch*patch+py*patch+px.
    let mut patches = vec![0.0f32; tokens * tdim];
    for tyi in 0..ty {
        for txi in 0..tx {
            let t = tyi * tx + txi;
            for c in 0..cfg.channels {
                for py in 0..cfg.patch {
                    for px in 0..cfg.patch {
                        let d = c * cfg.patch * cfg.patch + py * cfg.patch + px;
                        let y = tyi * cfg.patch + py;
                        let xx = txi * cfg.patch + px;
                        patches[t * tdim + d] = state[c * cfg.ny * cfg.nx + y * cfg.nx + xx];
                    }
                }
            }
        }
    }

    let mut x = vec![0.0f32; tokens * dim];
    linear(&patches, tokens, tdim, get(w, "embed.weight").data.as_slice(), get(w, "embed.bias").as_1(), dim, &mut x);
    let pos = get(w, "pos");
    add_inplace(&mut x, &pos.data);

    let mut m = vec![0.0f32; dim];
    linear(meta, 1, cfg.meta_dim, get(w, "meta.weight").data.as_slice(), get(w, "meta.bias").as_1(), dim, &mut m);
    for t in 0..tokens {
        add_inplace(&mut x[t * dim..(t + 1) * dim], &m);
    }

    let mut h = vec![0.0f32; tokens * dim];
    let mut attn_out = vec![0.0f32; tokens * dim];
    let mut proj_out = vec![0.0f32; tokens * dim];
    let mut ff1_out = vec![0.0f32; tokens * cfg.ff_inner];
    let mut ff2_out = vec![0.0f32; tokens * dim];
    for i in 0..cfg.layers {
        let p = |s: &str| format!("blocks.{i}.{s}");
        layernorm(&x, tokens, dim, get(w, &p("norm1.gamma")).as_1(), get(w, &p("norm1.beta")).as_1(), &mut h);
        attention(
            &h,
            tokens,
            dim,
            cfg.heads,
            tx,
            &cfg.mask,
            get(w, &p("q.weight")).data.as_slice(),
            get(w, &p("q.bias")).as_1(),
            get(w, &p("k.weight")).data.as_slice(),
            get(w, &p("k.bias")).as_1(),
            get(w, &p("v.weight")).data.as_slice(),
            get(w, &p("v.bias")).as_1(),
            &mut attn_out,
        );
        linear(&attn_out, tokens, dim, get(w, &p("proj.weight")).data.as_slice(), get(w, &p("proj.bias")).as_1(), dim, &mut proj_out);
        add_inplace(&mut x, &proj_out);

        layernorm(&x, tokens, dim, get(w, &p("norm2.gamma")).as_1(), get(w, &p("norm2.beta")).as_1(), &mut h);
        linear(&h, tokens, dim, get(w, &p("ff1.weight")).data.as_slice(), get(w, &p("ff1.bias")).as_1(), cfg.ff_inner, &mut ff1_out);
        gelu_inplace(&mut ff1_out);
        linear(&ff1_out, tokens, cfg.ff_inner, get(w, &p("ff2.weight")).data.as_slice(), get(w, &p("ff2.bias")).as_1(), dim, &mut ff2_out);
        add_inplace(&mut x, &ff2_out);
    }

    let mut xn = vec![0.0f32; tokens * dim];
    layernorm(&x, tokens, dim, get(w, "norm.gamma").as_1(), get(w, "norm.beta").as_1(), &mut xn);

    // CellHead. t0[dim, tokens_y, tokens_x] from xn[tokens, dim].
    let mut t0 = vec![0.0f32; dim * tokens];
    for t in 0..tokens {
        for c in 0..dim {
            t0[c * tokens + t] = xn[t * dim + c];
        }
    }
    let mut mix = vec![0.0f32; dim * tokens];
    conv2d(&t0, dim, ty, tx, get(w, "head.token_mix.weight").data.as_slice(), get(w, "head.token_mix.bias").as_1(), dim, 3, &mut mix);
    gelu_inplace(&mut mix);
    add_inplace(&mut t0, &mix); // residual: t1 = t0 + gelu(conv(t0))

    let head_dim = cfg.head_dim;
    let mut z0 = vec![0.0f32; head_dim * tokens];
    conv2d(&t0, dim, ty, tx, get(w, "head.to_cells.weight").data.as_slice(), get(w, "head.to_cells.bias").as_1(), head_dim, 1, &mut z0);

    let mut z1 = vec![0.0f32; head_dim * cfg.ny * tx];
    upsample_y(&z0, head_dim, ty, tx, cfg.patch, &mut z1);
    let mut z2 = vec![0.0f32; head_dim * cfg.ny * cfg.nx];
    upsample_x(&z1, head_dim, cfg.ny, tx, cfg.patch, &mut z2);

    let cells = cfg.ny * cfg.nx;
    let mut zc = vec![0.0f32; (head_dim + cfg.channels) * cells];
    zc[..head_dim * cells].copy_from_slice(&z2);
    zc[head_dim * cells..].copy_from_slice(state);

    let mut c1 = vec![0.0f32; head_dim * cells];
    conv2d(&zc, head_dim + cfg.channels, cfg.ny, cfg.nx, get(w, "head.conv1.weight").data.as_slice(), get(w, "head.conv1.bias").as_1(), head_dim, 3, &mut c1);
    gelu_inplace(&mut c1);

    let mut increment = vec![0.0f32; cfg.channels * cells];
    conv2d(&c1, head_dim, cfg.ny, cfg.nx, get(w, "head.conv2.weight").data.as_slice(), get(w, "head.conv2.bias").as_1(), cfg.channels, 3, &mut increment);

    increment
}
