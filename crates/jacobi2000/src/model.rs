//! Patch transformer. Floats in, floats out: no tokens, no vocab. The rule
//! parameters (dx, dt, g, depth, viscosity) enter as a conditioning vector
//! added to every token, not as text.

use burn::nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::grid::Grid;

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
            head: LinearConfig::new(self.dim, g.patch_dim()).init(device),
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
    head: Linear<B>,
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

        let x = self.patchify(state);
        let mut x = self.embed.forward(x) + self.pos.val();
        let m = self.meta.forward(meta).reshape([batch, 1, self.dim]);
        x = x + m;

        let mask = mask.map(|m| m.reshape([1, 1, tokens, tokens]));
        for b in self.blocks.iter() {
            x = b.forward(x, mask.clone(), self.heads);
        }
        let x = self.head.forward(self.norm.forward(x));
        self.unpatchify(x)
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

    fn unpatchify(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, _, _] = x.dims();
        let g = self.grid;
        let (p, tx, ty) = (g.patch, g.tokens_x(), g.tokens_y());
        x.reshape([b, ty, tx, g.channels, p, p])
            .permute([0, 3, 1, 4, 2, 5])
            .reshape([b, g.channels, g.ny, g.nx])
    }
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

    #[test]
    fn patchify_round_trips() {
        let device = Default::default();
        let cfg = Jacobi2000Config::new();
        let model = cfg.init::<B>(&device);
        let (state, _) = inputs(2, &cfg, &device);
        let back = model.unpatchify(model.patchify(state.clone()));
        back.into_data()
            .assert_approx_eq::<f32>(&state.into_data(), burn::tensor::Tolerance::default());
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
