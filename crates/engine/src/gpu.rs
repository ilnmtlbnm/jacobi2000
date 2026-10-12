//! WebGPU forward pass: the same operation sequence as `cpu.rs`, each op a
//! small hand-written WGSL kernel (`src/shaders/*.wgsl`), built on
//! `lean_runtime`'s device/buffer/pipeline helpers. One command encoder per
//! `forward()` call, one submit, one `into()`-free async readback at the
//! end (`lean_runtime::buffers::read`) — never a sync `.into_data()`-style
//! wait, which deadlocks a browser.
//!
//! All weight buffers and every intermediate scratch buffer are allocated
//! once, in [`GpuModel::new`]; `forward()` only `queue.write_buffer`s the
//! new state/meta and replays a fixed list of precomputed bind groups.
//! Scratch buffers are reused across layers (the residual stream `x`, the
//! layernorm scratch `h`, `q`/`k`/`v`, etc. are each one buffer, overwritten
//! every layer) — correct because wgpu serializes storage-buffer
//! read-after-write hazards between compute passes recorded in the same
//! encoder. Only the weight-bearing bind groups (one per layer per
//! weight-touching op) differ layer to layer; attention shares one bind
//! group across every layer. GELU and the two residual adds are fused into
//! the epilogue of the matmul/conv that produces their input.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use lean_runtime::{buffers, gpu as rtgpu, pipeline};

use crate::config::Config;
use crate::mask::Mask;
use crate::weights::{get, Weights};

const WG_LINEAR: &str = include_str!("shaders/linear.wgsl");
const WG_LAYERNORM: &str = include_str!("shaders/layernorm.wgsl");
const WG_ADD_INPLACE: &str = include_str!("shaders/add_inplace.wgsl");
const WG_ADD_BROADCAST: &str = include_str!("shaders/add_broadcast_row.wgsl");
const WG_PATCHIFY: &str = include_str!("shaders/patchify.wgsl");
const WG_ATTENTION: &str = include_str!("shaders/attention.wgsl");
const WG_CONV2D: &str = include_str!("shaders/conv2d.wgsl");
const WG_UPSAMPLE: &str = include_str!("shaders/upsample.wgsl");
const WG_TRANSPOSE: &str = include_str!("shaders/transpose_tc.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearDims { m: u32, k: u32, n: u32, mode: u32 }
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayernormDims { rows: u32, dim: u32, eps: f32, _pad: u32 }
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LenDims { len: u32, _p0: u32, _p1: u32, _p2: u32 }
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BroadcastDims { rows: u32, dim: u32, _p0: u32, _p1: u32 }
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PatchifyDims { tokens_x: u32, patch: u32, channels: u32, nx: u32, ny: u32, patch_dim: u32, len: u32, _pad: u32 }
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttentionDims { tokens: u32, dim: u32, heads: u32, head_dim: u32, tokens_x: u32, mask_kind: u32, radius: u32, dilation: u32, scale: f32, _p0: u32, _p1: u32, _p2: u32 }
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Conv2dDims { cin: u32, h: u32, w: u32, cout: u32, k: u32, len: u32, gelu: u32, _p1: u32 }
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpsampleDims { c: u32, h: u32, w: u32, p: u32, axis: u32, len: u32, _p0: u32, _p1: u32 }
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransposeDims { tokens: u32, dim: u32, len: u32, _p0: u32 }

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// A storage buffer sized for `n` f32s, read_write, usable as both a
/// dispatch's input and output across different ops.
fn scratch(device: &wgpu::Device, label: &str, n: usize) -> wgpu::Buffer {
    buffers::empty(device, label, (n.max(1) * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST)
}

fn uniform<T: Pod>(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, data: &T) -> wgpu::Buffer {
    buffers::uniform(device, queue, label, data)
}

pub struct GpuModel {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cfg: Config,

    pl_linear: wgpu::ComputePipeline,
    pl_layernorm: wgpu::ComputePipeline,
    pl_add_inplace: wgpu::ComputePipeline,
    pl_add_broadcast: wgpu::ComputePipeline,
    pl_patchify: wgpu::ComputePipeline,
    pl_attention: wgpu::ComputePipeline,
    pl_conv2d: wgpu::ComputePipeline,
    pl_upsample: wgpu::ComputePipeline,
    pl_transpose: wgpu::ComputePipeline,

    state_buf: wgpu::Buffer,
    meta_buf: wgpu::Buffer,
    increment_buf: wgpu::Buffer,
    readback_len: u64,

    // Per-layer bind groups for the weight-bearing ops, in forward order.
    norm1: Vec<wgpu::BindGroup>,
    qkv: Vec<wgpu::BindGroup>,
    proj: Vec<wgpu::BindGroup>,
    norm2: Vec<wgpu::BindGroup>,
    ff1: Vec<wgpu::BindGroup>,
    ff2: Vec<wgpu::BindGroup>,

    // Shape-only ops, shared across every layer (same scratch buffers).
    attention_bg: wgpu::BindGroup,

    // The rest of the pipeline, each used once per forward().
    patchify_bg: wgpu::BindGroup,
    embed_bg: wgpu::BindGroup,
    add_pos_bg: wgpu::BindGroup,
    meta_linear_bg: wgpu::BindGroup,
    add_meta_bg: wgpu::BindGroup,
    final_norm_bg: wgpu::BindGroup,
    transpose_bg: wgpu::BindGroup,
    token_mix_bg: wgpu::BindGroup,
    t0_resid_bg: wgpu::BindGroup,
    to_cells_bg: wgpu::BindGroup,
    upsample_y_bg: wgpu::BindGroup,
    upsample_x_bg: wgpu::BindGroup,
    head_conv1_bg: wgpu::BindGroup,
    head_conv2_bg: wgpu::BindGroup,

    zc_z2_bytes: u64,
    zc_state_bytes: u64,
    z2_buf: wgpu::Buffer,
    zc_buf: wgpu::Buffer,
}

#[allow(clippy::too_many_arguments)]
fn linear_bg(device: &wgpu::Device, queue: &wgpu::Queue, pl: &wgpu::ComputePipeline, x: &wgpu::Buffer, w: &wgpu::Buffer, b: &wgpu::Buffer, out: &wgpu::Buffer, m: u32, k: u32, n: u32, mode: u32) -> wgpu::BindGroup {
    let dims = uniform(device, queue, "linear_dims", &LinearDims { m, k, n, mode });
    pipeline::bind_group(device, pl, 0, &[
        wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
    ])
}

#[allow(clippy::too_many_arguments)]
fn layernorm_bg(device: &wgpu::Device, queue: &wgpu::Queue, pl: &wgpu::ComputePipeline, x: &wgpu::Buffer, gamma: &wgpu::Buffer, beta: &wgpu::Buffer, out: &wgpu::Buffer, rows: u32, dim: u32) -> wgpu::BindGroup {
    let dims = uniform(device, queue, "ln_dims", &LayernormDims { rows, dim, eps: 1e-5, _pad: 0 });
    pipeline::bind_group(device, pl, 0, &[
        wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: gamma.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: beta.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
    ])
}

fn add_inplace_bg(device: &wgpu::Device, queue: &wgpu::Queue, pl: &wgpu::ComputePipeline, a: &wgpu::Buffer, b: &wgpu::Buffer, len: u32) -> wgpu::BindGroup {
    let dims = uniform(device, queue, "add_dims", &LenDims { len, _p0: 0, _p1: 0, _p2: 0 });
    pipeline::bind_group(device, pl, 0, &[
        wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
    ])
}

fn add_broadcast_bg(device: &wgpu::Device, queue: &wgpu::Queue, pl: &wgpu::ComputePipeline, x: &wgpu::Buffer, row: &wgpu::Buffer, rows: u32, dim: u32) -> wgpu::BindGroup {
    let dims = uniform(device, queue, "bcast_dims", &BroadcastDims { rows, dim, _p0: 0, _p1: 0 });
    pipeline::bind_group(device, pl, 0, &[
        wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: row.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
    ])
}

#[allow(clippy::too_many_arguments)]
fn conv2d_bg(device: &wgpu::Device, queue: &wgpu::Queue, pl: &wgpu::ComputePipeline, input: &wgpu::Buffer, weight: &wgpu::Buffer, bias: &wgpu::Buffer, out: &wgpu::Buffer, cin: u32, h: u32, w: u32, cout: u32, k: u32, gelu: bool) -> wgpu::BindGroup {
    let dims = uniform(device, queue, "conv_dims", &Conv2dDims { cin, h, w, cout, k, len: cout * h * w, gelu: gelu as u32, _p1: 0 });
    pipeline::bind_group(device, pl, 0, &[
        wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: bias.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: dims.as_entire_binding() },
    ])
}

#[allow(clippy::too_many_arguments)]
fn upsample_bg(device: &wgpu::Device, queue: &wgpu::Queue, pl: &wgpu::ComputePipeline, input: &wgpu::Buffer, out: &wgpu::Buffer, c: u32, h: u32, w: u32, p: u32, axis: u32) -> wgpu::BindGroup {
    let (out_h, out_w) = if axis == 0 { (h * p, w) } else { (h, w * p) };
    let dims = uniform(device, queue, "ups_dims", &UpsampleDims { c, h, w, p, axis, len: c * out_h * out_w, _p0: 0, _p1: 0 });
    pipeline::bind_group(device, pl, 0, &[
        wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
    ])
}

impl GpuModel {
    pub async fn new(cfg: Config, w: &Weights) -> Result<GpuModel, lean_runtime::Error> {
        let adapter = rtgpu::request_adapter().await?;
        let (device, queue) = rtgpu::request_device(&adapter, "jacobi2000-engine", wgpu::Features::empty()).await?;
        // The linear kernel runs 16x16 workgroups and the layernorm 256 invocations.
        lean_runtime::limits::require_compute(&device.limits(), 256, 16384)?;
        Ok(Self::from_device(device, queue, cfg, w))
    }

    fn from_device(device: wgpu::Device, queue: wgpu::Queue, cfg: Config, w: &Weights) -> GpuModel {
        let pl = |src: &str, entry: &str| pipeline::compute_pipeline(&device, entry, src, entry, &[]);
        let pl_linear = pl(WG_LINEAR, "main");
        let pl_layernorm = pl(WG_LAYERNORM, "main");
        let pl_add_inplace = pl(WG_ADD_INPLACE, "main");
        let pl_add_broadcast = pl(WG_ADD_BROADCAST, "main");
        let pl_patchify = pl(WG_PATCHIFY, "main");
        let pl_attention = pl(WG_ATTENTION, "main");
        let pl_conv2d = pl(WG_CONV2D, "main");
        let pl_upsample = pl(WG_UPSAMPLE, "main");
        let pl_transpose = pl(WG_TRANSPOSE, "main");

        // Upload every weight tensor as a storage buffer, named exactly as
        // cpu.rs's `get(w, name)` looks them up.
        let mut wbuf: HashMap<String, wgpu::Buffer> = HashMap::new();
        for (name, t) in w.iter() {
            let bytes: &[u8] = bytemuck::cast_slice(&t.data);
            let buf = buffers::upload(&device, &queue, name, bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
            wbuf.insert(name.clone(), buf);
        }
        let wb = |name: &str| wbuf.get(name).unwrap_or_else(|| panic!("missing tensor `{name}`"));

        let (tokens, dim, ff_inner, head_dim) = (cfg.tokens as u32, cfg.dim as u32, cfg.ff_inner as u32, cfg.head_dim as u32);
        let (ny, nx, channels, patch) = (cfg.ny as u32, cfg.nx as u32, cfg.channels as u32, cfg.patch as u32);
        let (tokens_y, tokens_x) = (cfg.tokens_y as u32, cfg.tokens_x as u32);

        let patches_buf = scratch(&device, "patches", cfg.tokens * cfg.patch_dim);
        let x_buf = scratch(&device, "x", cfg.tokens * cfg.dim);
        let h_buf = scratch(&device, "h", cfg.tokens * cfg.dim);
        let m_buf = scratch(&device, "m", cfg.dim);
        let qkv_buf = scratch(&device, "qkv", cfg.tokens * 3 * cfg.dim);
        let attn_out_buf = scratch(&device, "attn_out", cfg.tokens * cfg.dim);
        let ff1_out_buf = scratch(&device, "ff1_out", cfg.tokens * cfg.ff_inner);
        let xn_buf = scratch(&device, "xn", cfg.tokens * cfg.dim);
        let t0_buf = scratch(&device, "t0", cfg.dim * cfg.tokens);
        let mix_buf = scratch(&device, "mix", cfg.dim * cfg.tokens);
        let z0_buf = scratch(&device, "z0", cfg.head_dim * cfg.tokens);
        let z1_buf = scratch(&device, "z1", cfg.head_dim * cfg.ny * cfg.tokens_x);
        let z2_buf = scratch(&device, "z2", cfg.head_dim * cfg.ny * cfg.nx);
        let zc_buf = scratch(&device, "zc", (cfg.head_dim + cfg.channels) * cfg.ny * cfg.nx);
        let c1_buf = scratch(&device, "c1", cfg.head_dim * cfg.ny * cfg.nx);
        let state_buf = scratch(&device, "state", cfg.channels * cfg.ny * cfg.nx);
        let meta_buf = scratch(&device, "meta", cfg.meta_dim);
        let increment_buf = scratch(&device, "increment", cfg.channels * cfg.ny * cfg.nx);

        let patchify_bg = {
            let len = (cfg.tokens * cfg.patch_dim) as u32;
            let dims = uniform(&device, &queue, "patchify_dims", &PatchifyDims { tokens_x, patch, channels, nx, ny, patch_dim: cfg.patch_dim as u32, len, _pad: 0 });
            pipeline::bind_group(&device, &pl_patchify, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: patches_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
            ])
        };
        let embed_bg = linear_bg(&device, &queue, &pl_linear, &patches_buf, wb("embed.weight"), wb("embed.bias"), &x_buf, tokens, cfg.patch_dim as u32, dim, 0);
        let add_pos_bg = add_inplace_bg(&device, &queue, &pl_add_inplace, &x_buf, wb("pos"), tokens * dim);
        let meta_linear_bg = linear_bg(&device, &queue, &pl_linear, &meta_buf, wb("meta.weight"), wb("meta.bias"), &m_buf, 1, cfg.meta_dim as u32, dim, 0);
        let add_meta_bg = add_broadcast_bg(&device, &queue, &pl_add_broadcast, &x_buf, &m_buf, tokens, dim);

        let mut norm1 = Vec::new();
        let mut qkv = Vec::new();
        let mut proj = Vec::new();
        let mut norm2 = Vec::new();
        let mut ff1 = Vec::new();
        let mut ff2 = Vec::new();
        for i in 0..cfg.layers {
            let p = |s: &str| format!("blocks.{i}.{s}");
            norm1.push(layernorm_bg(&device, &queue, &pl_layernorm, &x_buf, wb(&p("norm1.gamma")), wb(&p("norm1.beta")), &h_buf, tokens, dim));
            // q, k and v share their input, so one matmul against the three
            // weights concatenated along the output axis ([in, 3*out]).
            let (wq, wk, wv) = (get(w, &p("q.weight")), get(w, &p("k.weight")), get(w, &p("v.weight")));
            let mut wqkv = Vec::with_capacity(3 * wq.data.len());
            for r in 0..cfg.dim {
                for t in [wq, wk, wv] {
                    wqkv.extend_from_slice(&t.data[r * cfg.dim..(r + 1) * cfg.dim]);
                }
            }
            let bqkv: Vec<f32> = [&p("q.bias"), &p("k.bias"), &p("v.bias")].iter().flat_map(|n| get(w, n).data.iter().copied()).collect();
            let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            let wqkv_buf = buffers::upload(&device, &queue, "qkv.weight", bytemuck::cast_slice(&wqkv), storage);
            let bqkv_buf = buffers::upload(&device, &queue, "qkv.bias", bytemuck::cast_slice(&bqkv), storage);
            qkv.push(linear_bg(&device, &queue, &pl_linear, &h_buf, &wqkv_buf, &bqkv_buf, &qkv_buf, tokens, dim, 3 * dim, 0));
            proj.push(linear_bg(&device, &queue, &pl_linear, &attn_out_buf, wb(&p("proj.weight")), wb(&p("proj.bias")), &x_buf, tokens, dim, dim, 2));
            norm2.push(layernorm_bg(&device, &queue, &pl_layernorm, &x_buf, wb(&p("norm2.gamma")), wb(&p("norm2.beta")), &h_buf, tokens, dim));
            ff1.push(linear_bg(&device, &queue, &pl_linear, &h_buf, wb(&p("ff1.weight")), wb(&p("ff1.bias")), &ff1_out_buf, tokens, dim, ff_inner, 1));
            ff2.push(linear_bg(&device, &queue, &pl_linear, &ff1_out_buf, wb(&p("ff2.weight")), wb(&p("ff2.bias")), &x_buf, tokens, ff_inner, dim, 2));
        }

        let (mask_kind, radius, dilation) = match cfg.mask {
            Mask::Full => (0u32, 0u32, 1u32),
            Mask::Stencil { radius, dilation } => (1u32, radius as u32, dilation as u32),
        };
        let attention_bg = {
            // Attention heads split `dim`; `cfg.head_dim` is the width of the cell head's convolutions.
            let attn_head_dim = (cfg.dim / cfg.heads) as u32;
            let scale = 1.0 / (attn_head_dim as f32).sqrt();
            let dims = uniform(&device, &queue, "attn_dims", &AttentionDims { tokens, dim, heads: cfg.heads as u32, head_dim: attn_head_dim, tokens_x, mask_kind, radius, dilation, scale, _p0: 0, _p1: 0, _p2: 0 });
            pipeline::bind_group(&device, &pl_attention, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: qkv_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: attn_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
            ])
        };

        let final_norm_bg = layernorm_bg(&device, &queue, &pl_layernorm, &x_buf, wb("norm.gamma"), wb("norm.beta"), &xn_buf, tokens, dim);
        let transpose_bg = {
            let dims = uniform(&device, &queue, "transpose_dims", &TransposeDims { tokens, dim, len: tokens * dim, _p0: 0 });
            pipeline::bind_group(&device, &pl_transpose, 0, &[
                wgpu::BindGroupEntry { binding: 0, resource: xn_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: t0_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
            ])
        };
        let token_mix_bg = conv2d_bg(&device, &queue, &pl_conv2d, &t0_buf, wb("head.token_mix.weight"), wb("head.token_mix.bias"), &mix_buf, dim, tokens_y, tokens_x, dim, 3, true);
        let t0_resid_bg = add_inplace_bg(&device, &queue, &pl_add_inplace, &t0_buf, &mix_buf, dim * tokens);
        let to_cells_bg = conv2d_bg(&device, &queue, &pl_conv2d, &t0_buf, wb("head.to_cells.weight"), wb("head.to_cells.bias"), &z0_buf, dim, tokens_y, tokens_x, head_dim, 1, false);
        let upsample_y_bg = upsample_bg(&device, &queue, &pl_upsample, &z0_buf, &z1_buf, head_dim, tokens_y, tokens_x, patch, 0);
        let upsample_x_bg = upsample_bg(&device, &queue, &pl_upsample, &z1_buf, &z2_buf, head_dim, ny, tokens_x, patch, 1);
        let head_conv1_bg = conv2d_bg(&device, &queue, &pl_conv2d, &zc_buf, wb("head.conv1.weight"), wb("head.conv1.bias"), &c1_buf, head_dim + channels, ny, nx, head_dim, 3, true);
        let head_conv2_bg = conv2d_bg(&device, &queue, &pl_conv2d, &c1_buf, wb("head.conv2.weight"), wb("head.conv2.bias"), &increment_buf, head_dim, ny, nx, channels, 3, false);

        let zc_z2_bytes = (cfg.head_dim * cfg.ny * cfg.nx * 4) as u64;
        let zc_state_bytes = (cfg.channels * cfg.ny * cfg.nx * 4) as u64;
        let readback_len = (cfg.channels * cfg.ny * cfg.nx) as u64;

        GpuModel {
            device, queue, cfg,
            pl_linear, pl_layernorm, pl_add_inplace, pl_add_broadcast, pl_patchify, pl_attention, pl_conv2d, pl_upsample, pl_transpose,
            state_buf, meta_buf, increment_buf, readback_len,
            norm1, qkv, proj, norm2, ff1, ff2,
            attention_bg,
            patchify_bg, embed_bg, add_pos_bg, meta_linear_bg, add_meta_bg, final_norm_bg, transpose_bg,
            token_mix_bg, t0_resid_bg, to_cells_bg, upsample_y_bg, upsample_x_bg,
            head_conv1_bg, head_conv2_bg,
            zc_z2_bytes, zc_state_bytes, z2_buf, zc_buf,
        }
    }

    fn dispatch1d(&self, encoder: &mut wgpu::CommandEncoder, pl: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, len: u32, wg_size: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pl);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(ceil_div(len, wg_size), 1, 1);
    }

    fn dispatch2d(&self, encoder: &mut wgpu::CommandEncoder, pl: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, x: u32, y: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pl);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }

    /// `state`/`meta`: same layout as `cpu::forward`. Returns the predicted
    /// increment, same shape as `state`.
    pub async fn forward(&self, state: &[f32], meta: &[f32]) -> Vec<f32> {
        self.queue.write_buffer(&self.state_buf, 0, bytemuck::cast_slice(state));
        self.queue.write_buffer(&self.meta_buf, 0, bytemuck::cast_slice(meta));

        let cfg = &self.cfg;
        let (tokens, dim, ff_inner) = (cfg.tokens as u32, cfg.dim as u32, cfg.ff_inner as u32);
        let (ny, nx, head_dim, channels) = (cfg.ny as u32, cfg.nx as u32, cfg.head_dim as u32, cfg.channels as u32);
        let (tokens_x, patch) = (cfg.tokens_x as u32, cfg.patch as u32);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("jacobi2000-forward") });

        self.dispatch1d(&mut encoder, &self.pl_patchify, &self.patchify_bg, tokens * cfg.patch_dim as u32, 64);
        self.dispatch2d(&mut encoder, &self.pl_linear, &self.embed_bg, ceil_div(dim, 32), ceil_div(tokens, 32));
        self.dispatch1d(&mut encoder, &self.pl_add_inplace, &self.add_pos_bg, tokens * dim, 64);
        self.dispatch2d(&mut encoder, &self.pl_linear, &self.meta_linear_bg, ceil_div(dim, 32), 1);
        self.dispatch1d(&mut encoder, &self.pl_add_broadcast, &self.add_meta_bg, tokens * dim, 64);

        for i in 0..cfg.layers {
            self.dispatch1d(&mut encoder, &self.pl_layernorm, &self.norm1[i], tokens, 1);
            self.dispatch2d(&mut encoder, &self.pl_linear, &self.qkv[i], ceil_div(3 * dim, 32), ceil_div(tokens, 32));
            self.dispatch2d(&mut encoder, &self.pl_attention, &self.attention_bg, cfg.heads as u32, tokens);
            self.dispatch2d(&mut encoder, &self.pl_linear, &self.proj[i], ceil_div(dim, 32), ceil_div(tokens, 32));

            self.dispatch1d(&mut encoder, &self.pl_layernorm, &self.norm2[i], tokens, 1);
            self.dispatch2d(&mut encoder, &self.pl_linear, &self.ff1[i], ceil_div(ff_inner, 32), ceil_div(tokens, 32));
            self.dispatch2d(&mut encoder, &self.pl_linear, &self.ff2[i], ceil_div(dim, 32), ceil_div(tokens, 32));
        }

        self.dispatch1d(&mut encoder, &self.pl_layernorm, &self.final_norm_bg, tokens, 1);
        self.dispatch1d(&mut encoder, &self.pl_transpose, &self.transpose_bg, tokens * dim, 64);
        self.dispatch2d(&mut encoder, &self.pl_conv2d, &self.token_mix_bg, ceil_div(tokens, 32), ceil_div(dim, 32));
        self.dispatch1d(&mut encoder, &self.pl_add_inplace, &self.t0_resid_bg, dim * tokens, 64);
        self.dispatch2d(&mut encoder, &self.pl_conv2d, &self.to_cells_bg, ceil_div(tokens, 32), ceil_div(head_dim, 32));
        self.dispatch1d(&mut encoder, &self.pl_upsample, &self.upsample_y_bg, head_dim * ny * tokens_x, 64);
        self.dispatch1d(&mut encoder, &self.pl_upsample, &self.upsample_x_bg, head_dim * ny * nx, 64);

        // zc = concat([z2, state], channel dim): two plain buffer copies,
        // no kernel needed.
        encoder.copy_buffer_to_buffer(&self.z2_buf, 0, &self.zc_buf, 0, self.zc_z2_bytes);
        encoder.copy_buffer_to_buffer(&self.state_buf, 0, &self.zc_buf, self.zc_z2_bytes, self.zc_state_bytes);

        self.dispatch2d(&mut encoder, &self.pl_conv2d, &self.head_conv1_bg, ceil_div(ny * nx, 32), ceil_div(head_dim, 32));
        self.dispatch2d(&mut encoder, &self.pl_conv2d, &self.head_conv2_bg, ceil_div(ny * nx, 32), ceil_div(channels, 32));
        let _ = patch;

        let staging = buffers::stage(&self.device, &mut encoder, &self.increment_buf, 0, self.readback_len * 4);
        self.queue.submit(Some(encoder.finish()));
        buffers::map_read(&self.device, &staging, |bytes| bytemuck::cast_slice::<u8, f32>(bytes).to_vec())
            .await
            .expect("GPU readback failed")
    }

    pub fn cfg(&self) -> &Config {
        &self.cfg
    }

    /// `state + forward(state, meta)`.
    pub async fn step(&self, state: &[f32], meta: &[f32]) -> Vec<f32> {
        let mut out = self.forward(state, meta).await;
        for (o, s) in out.iter_mut().zip(state) {
            *o += s;
        }
        out
    }
}
