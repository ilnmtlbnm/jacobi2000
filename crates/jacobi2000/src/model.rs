//! Patch transformer. Floats in, floats out: no tokens, no vocab. The rule
//! parameters (dx, dt, g, depth, viscosity) enter as a conditioning vector
//! added to every token, not as text.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::grid::Grid;
use crate::weights::{get, put, Weights};

#[derive(Config, Debug)]
pub struct Jacobi2000Config {
    #[config(default = 64)]
    pub nx: usize,
    #[config(default = 64)]
    pub ny: usize,
    #[config(default = 4)]
    pub patch: usize,
    #[config(default = 3)]
    pub channels: usize,
    #[config(default = 0)]
    pub halo: usize,
    #[config(default = 128)]
    pub dim: usize,
    #[config(default = 4)]
    pub layers: usize,
    #[config(default = 4)]
    pub heads: usize,
    #[config(default = 4)]
    pub ff_mult: usize,
    #[config(default = 5)]
    pub meta_dim: usize,
    /// Width of the cell-resolution output head (see [`CellHead`]).
    #[config(default = 32)]
    pub head_dim: usize,
}

impl Jacobi2000Config {
    pub fn grid(&self) -> Grid {
        Grid {
            nx: self.nx,
            ny: self.ny,
            channels: self.channels,
            patch: self.patch,
            halo: self.halo,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Jacobi2000<B> {
        let g = self.grid();
        Jacobi2000 {
            embed: LinearConfig::new(g.patch_dim(), self.dim).init(device),
            meta: LinearConfig::new(self.meta_dim, self.dim).init(device),
            pos: Param::from_tensor(Tensor::zeros([1, g.tokens(), self.dim], device)),
            blocks: (0..self.layers)
                .map(|_| Block::new(self.dim, self.heads, self.ff_mult, device))
                .collect(),
            norm: LayerNormConfig::new(self.dim).init(device),
            head: CellHead::new(self.dim, self.head_dim, g, device),
            grid: g,
            dim: self.dim,
            heads: self.heads,
        }
    }
}

#[derive(Module, Debug)]
pub struct Jacobi2000<B: Backend> {
    embed: Linear<B>,
    meta: Linear<B>,
    /// Learned absolute position per token. Relative/RoPE positions are the
    /// next dial along (CONCEPT sec. 3); absolute is enough for a fixed grid.
    pos: Param<Tensor<B, 3>>,
    blocks: Vec<Block<B>>,
    norm: LayerNorm<B>,
    head: CellHead<B>,
    pub grid: Grid,
    dim: usize,
    heads: usize,
}

impl<B: Backend> Jacobi2000<B> {
    /// `state`: `[batch, channels, ny, nx]`, normalized.
    /// `meta`: `[batch, meta_dim]`.
    /// `mask`: additive `[tokens, tokens]`, or `None` for full attention.
    ///
    /// Returns the predicted *increment* over one tick, same shape as
    /// `state`. Predicting the increment rather than the state keeps the
    /// identity part of the map out of the weights.
    pub fn forward(
        &self,
        state: Tensor<B, 4>,
        meta: Tensor<B, 2>,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 4> {
        let [batch, _, _, _] = state.dims();
        let tokens = self.grid.tokens();

        let x = self.patchify(state.clone());
        let mut x = self.embed.forward(x) + self.pos.val();
        let m = self.meta.forward(meta).reshape([batch, 1, self.dim]);
        x = x + m;

        let mask = mask.map(|m| m.reshape([1, 1, tokens, tokens]));
        for b in self.blocks.iter() {
            x = b.forward(x, mask.clone(), self.heads);
        }
        self.head.forward(self.norm.forward(x), state, self.grid)
    }

    /// One tick: state + predicted increment.
    pub fn step(
        &self,
        state: Tensor<B, 4>,
        meta: Tensor<B, 2>,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 4> {
        state.clone() + self.forward(state, meta, mask)
    }

    fn patchify(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, c, _, _] = x.dims();
        let g = self.grid;
        let (p, tx, ty) = (g.patch, g.tokens_x(), g.tokens_y());
        x.reshape([b, c, ty, p, tx, p])
            .permute([0, 2, 4, 1, 3, 5])
            .reshape([b, g.tokens(), g.patch_dim()])
    }

    /// Build a model of `cfg`'s architecture, then overwrite every learnable
    /// tensor from `weights` (burn's own field names, dotted for nesting:
    /// `embed.weight`, `embed.bias`, `meta.weight`, `meta.bias`, `pos`,
    /// `blocks.<i>.{norm1,norm2}.{gamma,beta}`,
    /// `blocks.<i>.{q,k,v,proj,ff1,ff2}.{weight,bias}`, `norm.{gamma,beta}`,
    /// `head.{token_mix,to_cells,conv1,conv2}.{weight,bias}`). This is the
    /// import half of the weight exchange with the PyTorch training side
    /// (`docs/pytorch-training.md`); the export half is [`Self::to_weights`].
    pub fn from_weights(cfg: &Jacobi2000Config, weights: &Weights, device: &B::Device) -> Self {
        let mut m = cfg.init::<B>(device);

        m.embed.weight = Param::from_tensor(get::<2, B>(weights, "embed.weight", device));
        m.embed.bias = Some(Param::from_tensor(get::<1, B>(weights, "embed.bias", device)));
        m.meta.weight = Param::from_tensor(get::<2, B>(weights, "meta.weight", device));
        m.meta.bias = Some(Param::from_tensor(get::<1, B>(weights, "meta.bias", device)));
        m.pos = Param::from_tensor(get::<3, B>(weights, "pos", device));

        for (i, b) in m.blocks.iter_mut().enumerate() {
            let p = |s: &str| format!("blocks.{i}.{s}");
            b.norm1.gamma = Param::from_tensor(get::<1, B>(weights, &p("norm1.gamma"), device));
            b.norm1.beta = Some(Param::from_tensor(get::<1, B>(weights, &p("norm1.beta"), device)));
            b.q.weight = Param::from_tensor(get::<2, B>(weights, &p("q.weight"), device));
            b.q.bias = Some(Param::from_tensor(get::<1, B>(weights, &p("q.bias"), device)));
            b.k.weight = Param::from_tensor(get::<2, B>(weights, &p("k.weight"), device));
            b.k.bias = Some(Param::from_tensor(get::<1, B>(weights, &p("k.bias"), device)));
            b.v.weight = Param::from_tensor(get::<2, B>(weights, &p("v.weight"), device));
            b.v.bias = Some(Param::from_tensor(get::<1, B>(weights, &p("v.bias"), device)));
            b.proj.weight = Param::from_tensor(get::<2, B>(weights, &p("proj.weight"), device));
            b.proj.bias = Some(Param::from_tensor(get::<1, B>(weights, &p("proj.bias"), device)));
            b.norm2.gamma = Param::from_tensor(get::<1, B>(weights, &p("norm2.gamma"), device));
            b.norm2.beta = Some(Param::from_tensor(get::<1, B>(weights, &p("norm2.beta"), device)));
            b.ff1.weight = Param::from_tensor(get::<2, B>(weights, &p("ff1.weight"), device));
            b.ff1.bias = Some(Param::from_tensor(get::<1, B>(weights, &p("ff1.bias"), device)));
            b.ff2.weight = Param::from_tensor(get::<2, B>(weights, &p("ff2.weight"), device));
            b.ff2.bias = Some(Param::from_tensor(get::<1, B>(weights, &p("ff2.bias"), device)));
        }

        m.norm.gamma = Param::from_tensor(get::<1, B>(weights, "norm.gamma", device));
        m.norm.beta = Some(Param::from_tensor(get::<1, B>(weights, "norm.beta", device)));

        m.head.token_mix.weight = Param::from_tensor(get::<4, B>(weights, "head.token_mix.weight", device));
        m.head.token_mix.bias = Some(Param::from_tensor(get::<1, B>(weights, "head.token_mix.bias", device)));
        m.head.to_cells.weight = Param::from_tensor(get::<4, B>(weights, "head.to_cells.weight", device));
        m.head.to_cells.bias = Some(Param::from_tensor(get::<1, B>(weights, "head.to_cells.bias", device)));
        m.head.conv1.weight = Param::from_tensor(get::<4, B>(weights, "head.conv1.weight", device));
        m.head.conv1.bias = Some(Param::from_tensor(get::<1, B>(weights, "head.conv1.bias", device)));
        m.head.conv2.weight = Param::from_tensor(get::<4, B>(weights, "head.conv2.weight", device));
        m.head.conv2.bias = Some(Param::from_tensor(get::<1, B>(weights, "head.conv2.bias", device)));

        m
    }

    /// The reverse of [`Self::from_weights`]: every learnable tensor this
    /// model holds, under the same names. Used to export a Burn-trained
    /// checkpoint (this crate's own training path, or a checkpoint saved by
    /// `crates/train`) into the exchange format for a PyTorch-side parity
    /// check.
    pub fn to_weights(&self) -> Weights {
        let mut w = Weights::new();
        put(&mut w, "embed.weight", self.embed.weight.val());
        put(&mut w, "embed.bias", self.embed.bias.as_ref().unwrap().val());
        put(&mut w, "meta.weight", self.meta.weight.val());
        put(&mut w, "meta.bias", self.meta.bias.as_ref().unwrap().val());
        put(&mut w, "pos", self.pos.val());

        for (i, b) in self.blocks.iter().enumerate() {
            let p = |s: &str| format!("blocks.{i}.{s}");
            put(&mut w, &p("norm1.gamma"), b.norm1.gamma.val());
            put(&mut w, &p("norm1.beta"), b.norm1.beta.as_ref().unwrap().val());
            put(&mut w, &p("q.weight"), b.q.weight.val());
            put(&mut w, &p("q.bias"), b.q.bias.as_ref().unwrap().val());
            put(&mut w, &p("k.weight"), b.k.weight.val());
            put(&mut w, &p("k.bias"), b.k.bias.as_ref().unwrap().val());
            put(&mut w, &p("v.weight"), b.v.weight.val());
            put(&mut w, &p("v.bias"), b.v.bias.as_ref().unwrap().val());
            put(&mut w, &p("proj.weight"), b.proj.weight.val());
            put(&mut w, &p("proj.bias"), b.proj.bias.as_ref().unwrap().val());
            put(&mut w, &p("norm2.gamma"), b.norm2.gamma.val());
            put(&mut w, &p("norm2.beta"), b.norm2.beta.as_ref().unwrap().val());
            put(&mut w, &p("ff1.weight"), b.ff1.weight.val());
            put(&mut w, &p("ff1.bias"), b.ff1.bias.as_ref().unwrap().val());
            put(&mut w, &p("ff2.weight"), b.ff2.weight.val());
            put(&mut w, &p("ff2.bias"), b.ff2.bias.as_ref().unwrap().val());
        }

        put(&mut w, "norm.gamma", self.norm.gamma.val());
        put(&mut w, "norm.beta", self.norm.beta.as_ref().unwrap().val());

        put(&mut w, "head.token_mix.weight", self.head.token_mix.weight.val());
        put(&mut w, "head.token_mix.bias", self.head.token_mix.bias.as_ref().unwrap().val());
        put(&mut w, "head.to_cells.weight", self.head.to_cells.weight.val());
        put(&mut w, "head.to_cells.bias", self.head.to_cells.bias.as_ref().unwrap().val());
        put(&mut w, "head.conv1.weight", self.head.conv1.weight.val());
        put(&mut w, "head.conv1.bias", self.head.conv1.bias.as_ref().unwrap().val());
        put(&mut w, "head.conv2.weight", self.head.conv2.weight.val());
        put(&mut w, "head.conv2.bias", self.head.conv2.bias.as_ref().unwrap().val());

        w
    }
}

/// Per-cell output head.
///
/// A plain `dim -> patch_dim` projection paints each 4x4 patch from 48
/// independent rows, so the increment it writes is discontinuous at every
/// patch boundary and the stencil mask of CONCEPT sec. 3 would only ever be
/// per patch. This head decodes to cells instead, in three moves:
///
/// 1. a 3x3 convolution over the *token* grid, so a patch is decoded knowing
///    its eight neighbours (one token of extra reach, the same reach one
///    stencil layer has);
/// 2. a linear upsample from token centres to cell centres, which is
///    continuous across patch boundaries by construction (cell `i` of token
///    `t` mixes `t` and `t+1` with weight `(i + 0.5) / patch`);
/// 3. two 3x3 convolutions at cell resolution that also read the input
///    field, so the sub-patch structure of the increment is a learned
///    stencil on the state rather than 16 unrelated projections.
#[derive(Module, Debug)]
struct CellHead<B: Backend> {
    token_mix: Conv2d<B>,
    to_cells: Conv2d<B>,
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    act: Gelu,
}

impl<B: Backend> CellHead<B> {
    fn new(dim: usize, head_dim: usize, grid: Grid, device: &B::Device) -> Self {
        let conv = |cin: usize, cout: usize, k: usize| {
            Conv2dConfig::new([cin, cout], [k, k])
                .with_padding(PaddingConfig2d::Explicit(k / 2, k / 2))
                .init(device)
        };
        CellHead {
            token_mix: conv(dim, dim, 3),
            to_cells: conv(dim, head_dim, 1),
            conv1: conv(head_dim + grid.channels, head_dim, 3),
            conv2: conv(head_dim, grid.channels, 3),
            act: Gelu::new(),
        }
    }

    /// `x`: `[batch, tokens, dim]`, `state`: `[batch, channels, ny, nx]`.
    fn forward(&self, x: Tensor<B, 3>, state: Tensor<B, 4>, grid: Grid) -> Tensor<B, 4> {
        let [b, _, d] = x.dims();
        let t = x
            .reshape([b, grid.tokens_y(), grid.tokens_x(), d])
            .permute([0, 3, 1, 2]);
        let t = t.clone() + self.act.forward(self.token_mix.forward(t));
        let z = self.to_cells.forward(t);
        let z = upsample(z, 2, grid.patch);
        let z = upsample(z, 3, grid.patch);
        let z = Tensor::cat(vec![z, state], 1);
        self.conv2
            .forward(self.act.forward(self.conv1.forward(z)))
    }
}

/// Linear upsample of `dim` (2 = y, 3 = x) by `p`, replicating the last
/// token so the far edge has a neighbour to interpolate towards.
fn upsample<B: Backend>(x: Tensor<B, 4>, dim: usize, p: usize) -> Tensor<B, 4> {
    let dims = x.dims();
    let n = dims[dim];
    let next = Tensor::cat(
        vec![
            x.clone().narrow(dim, 1, n - 1),
            x.clone().narrow(dim, n - 1, 1),
        ],
        dim,
    );
    let mut out = dims;
    out[dim] = n * p;
    let repeat = |y: Tensor<B, 4>| {
        y.unsqueeze_dim::<5>(dim + 1)
            .repeat_dim(dim + 1, p)
            .reshape(out)
    };
    let mut wshape = [1usize; 4];
    wshape[dim] = n * p;
    let w: Vec<f32> = (0..n * p)
        .map(|i| ((i % p) as f32 + 0.5) / p as f32)
        .collect();
    let w = Tensor::<B, 1>::from_floats(w.as_slice(), &x.device()).reshape(wshape);
    repeat(x) * (-w.clone() + 1.0) + repeat(next) * w
}

#[derive(Module, Debug)]
struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    proj: Linear<B>,
    norm2: LayerNorm<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
    act: Gelu,
}

impl<B: Backend> Block<B> {
    fn new(dim: usize, _heads: usize, ff_mult: usize, device: &B::Device) -> Self {
        Block {
            norm1: LayerNormConfig::new(dim).init(device),
            q: LinearConfig::new(dim, dim).init(device),
            k: LinearConfig::new(dim, dim).init(device),
            v: LinearConfig::new(dim, dim).init(device),
            proj: LinearConfig::new(dim, dim).init(device),
            norm2: LayerNormConfig::new(dim).init(device),
            ff1: LinearConfig::new(dim, dim * ff_mult).init(device),
            ff2: LinearConfig::new(dim * ff_mult, dim).init(device),
            act: Gelu::new(),
        }
    }

    fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>, heads: usize) -> Tensor<B, 3> {
        let h = self.norm1.forward(x.clone());
        let a = self.attention(h, mask, heads);
        let x = x + self.proj.forward(a);
        let h = self.norm2.forward(x.clone());
        x + self.ff2.forward(self.act.forward(self.ff1.forward(h)))
    }

    fn attention(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>, heads: usize) -> Tensor<B, 3> {
        let [b, t, d] = x.dims();
        let hd = d / heads;
        let split = |y: Tensor<B, 3>| y.reshape([b, t, heads, hd]).permute([0, 2, 1, 3]);
        let q = split(self.q.forward(x.clone()));
        let k = split(self.k.forward(x.clone()));
        let v = split(self.v.forward(x));

        let scores = q.matmul(k.permute([0, 1, 3, 2])) / (hd as f32).sqrt();
        let scores = match mask {
            Some(m) => scores + m,
            None => scores,
        };
        let w = softmax(scores, 3);
        w.matmul(v).permute([0, 2, 1, 3]).reshape([b, t, d])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::Mask;

    type B = burn::backend::NdArray;

    fn inputs(batch: usize, cfg: &Jacobi2000Config, device: &<B as Backend>::Device) -> (Tensor<B, 4>, Tensor<B, 2>) {
        let g = cfg.grid();
        let state = Tensor::<B, 1>::from_floats(
            (0..batch * g.channels * g.cells())
                .map(|i| ((i % 37) as f32 - 18.0) / 40.0)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        )
        .reshape([batch, g.channels, g.ny, g.nx]);
        let meta = Tensor::<B, 2>::zeros([batch, cfg.meta_dim], device);
        (state, meta)
    }

    #[test]
    fn forward_keeps_the_field_shape() {
        let device = Default::default();
        let cfg = Jacobi2000Config::new();
        let model = cfg.init::<B>(&device);
        let (state, meta) = inputs(2, &cfg, &device);
        let out = model.forward(state, meta, None);
        assert_eq!(out.dims(), [2, 3, 64, 64]);
    }

    /// The inverse of `patchify`, which the model no longer needs now that
    /// the head decodes to cells: kept here so the token layout stays pinned.
    fn unpatchify(x: Tensor<B, 3>, g: Grid) -> Tensor<B, 4> {
        let [b, _, _] = x.dims();
        let (p, tx, ty) = (g.patch, g.tokens_x(), g.tokens_y());
        x.reshape([b, ty, tx, g.channels, p, p])
            .permute([0, 3, 1, 4, 2, 5])
            .reshape([b, g.channels, g.ny, g.nx])
    }

    #[test]
    fn patchify_round_trips() {
        let device = Default::default();
        let cfg = Jacobi2000Config::new();
        let model = cfg.init::<B>(&device);
        let (state, _) = inputs(2, &cfg, &device);
        let back = unpatchify(model.patchify(state.clone()), cfg.grid());
        back.into_data()
            .assert_approx_eq::<f32>(&state.into_data(), burn::tensor::Tolerance::default());
    }

    /// A smooth input must come out smooth: the defect the per-cell head
    /// replaces is a `dim -> patch_dim` projection, which paints each 4x4
    /// patch from 16 unrelated rows and so writes white noise at cell scale.
    /// The two assertions are not equally strong. The boundary-vs-interior
    /// ratio is the block-structure statement, but a random `dim ->
    /// patch_dim` head passes it too (its cell-to-cell steps are large
    /// *everywhere*); the absolute smoothness bound is what that head fails.
    #[test]
    fn per_cell_head_has_no_4x4_block_structure() {
        let device = Default::default();
        let cfg = Jacobi2000Config::new();
        let model = cfg.init::<B>(&device);
        let g = cfg.grid();
        // A smooth field: two long-wavelength modes, nothing near cell scale.
        let field: Vec<f32> = (0..g.channels * g.cells())
            .map(|i| {
                let (x, y) = ((i % g.nx) as f32, ((i / g.nx) % g.ny) as f32);
                let c = (i / g.cells()) as f32;
                let k = std::f32::consts::TAU / 32.0;
                0.5 * ((k * x + c).sin() + (k * y - c).cos())
            })
            .collect();
        let state = Tensor::<B, 1>::from_floats(field.as_slice(), &device)
            .reshape([1, g.channels, g.ny, g.nx]);
        let meta = Tensor::<B, 2>::zeros([1, cfg.meta_dim], &device);
        let out = model.forward(state, meta, None);

        let d = out.clone().into_data();
        let v = d.as_slice::<f32>().unwrap();
        let at = |c: usize, y: usize, x: usize| v[(c * g.ny + y) * g.nx + x];
        let (mut bsum, mut bn, mut isum, mut inn) = (0.0f64, 0usize, 0.0f64, 0usize);
        for c in 0..g.channels {
            for y in 0..g.ny {
                for x in 1..g.nx {
                    let step = (at(c, y, x) - at(c, y, x - 1)).abs() as f64;
                    if x % g.patch == 0 {
                        bsum += step;
                        bn += 1;
                    } else {
                        isum += step;
                        inn += 1;
                    }
                }
            }
        }
        let (boundary, interior) = (bsum / bn as f64, isum / inn as f64);
        assert!(
            boundary < 1.25 * interior,
            "patch boundaries are hard edges: {boundary:.5} across vs {interior:.5} inside"
        );
        let std = {
            let n = v.len() as f64;
            let m = v.iter().map(|&x| x as f64).sum::<f64>() / n;
            (v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / n).sqrt()
        };
        assert!(
            interior < 0.5 * std,
            "output is not smooth at cell scale: step {interior:.5} vs std {std:.5}"
        );
    }

    #[test]
    fn stencil_mask_stops_far_information() {
        // One layer, radius-1 stencil: perturbing a patch 5 tokens away must
        // not move the output of token (0,0). This is the plumbing test; if
        // the mask were dropped on the way to softmax, it would fail.
        let device = Default::default();
        let cfg = Jacobi2000Config::new().with_layers(1);
        let model = cfg.init::<B>(&device);
        let g = cfg.grid();
        let mask = Mask::Stencil {
            radius: 1,
            dilation: 1,
        }
        .to_tensor::<B>(&g, &device);
        let (state, meta) = inputs(1, &cfg, &device);

        let out0 = model.forward(state.clone(), meta.clone(), mask.clone());
        // Patch at token (5,5) covers cells x,y in [20,24).
        let bump = Tensor::<B, 4>::ones([1, 3, 4, 4], &device) * 3.0;
        let perturbed = state.slice_assign([0..1, 0..3, 20..24, 20..24], bump);
        let out1 = model.forward(perturbed, meta, mask);

        let near = out1.clone().slice([0..1, 0..3, 0..4, 0..4])
            - out0.clone().slice([0..1, 0..3, 0..4, 0..4]);
        let far = out1.slice([0..1, 0..3, 16..28, 16..28])
            - out0.slice([0..1, 0..3, 16..28, 16..28]);
        let near_max: f32 = near.abs().max().into_scalar();
        let far_max: f32 = far.abs().max().into_scalar();
        assert!(near_max < 1e-6, "far patch leaked into token (0,0): {near_max}");
        assert!(far_max > 1e-4, "perturbation had no local effect either: {far_max}");
    }

    #[test]
    fn full_attention_does_propagate_far() {
        // The counterpart: with no mask the same perturbation must reach
        // token (0,0), otherwise the stencil test above proves nothing.
        let device = Default::default();
        let cfg = Jacobi2000Config::new().with_layers(1);
        let model = cfg.init::<B>(&device);
        let (state, meta) = inputs(1, &cfg, &device);
        let out0 = model.forward(state.clone(), meta.clone(), None);
        let bump = Tensor::<B, 4>::ones([1, 3, 4, 4], &device) * 3.0;
        let perturbed = state.slice_assign([0..1, 0..3, 20..24, 20..24], bump);
        let out1 = model.forward(perturbed, meta, None);
        let d = (out1.slice([0..1, 0..3, 0..4, 0..4]) - out0.slice([0..1, 0..3, 0..4, 0..4]))
            .abs()
            .max()
            .into_scalar();
        let d: f32 = d;
        assert!(d > 1e-7, "full attention did not propagate: {d}");
    }
}
