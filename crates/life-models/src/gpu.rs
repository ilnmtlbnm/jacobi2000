//! WebGPU forward pass. One kernel per model (`src/shaders/`), one thread
//! per cell, so a generation is a
//! single dispatch over all cells; `run` chains generations inside one
//! command encoder, ping-ponging two cell buffers, and reads back once, at
//! the end, packed four cells to a word. Uploads use `queue.write_buffer`,
//! readback is awaited (`lean_runtime::buffers`), nothing blocks.
//!
//! Buffers are sized from the board: cells are one `u32` each (two
//! buffers), logits `outputs * 4` bytes per cell. Whatever exceeds the
//! device's storage binding limit is an `Error::Runtime(Limit)`, not a
//! validation panic.

use std::cell::RefCell;
use std::fmt::Write as _;

use bytemuck::{Pod, Zeroable};
use lean_runtime::{buffers, gpu as rtgpu, limits, pipeline};

use crate::model::Kind;
use crate::{check_board, Error, Model};

const COMMON: &str = include_str!("shaders/common.wgsl");
const MLP2: &str = include_str!("shaders/mlp2.wgsl");
const BERT: &str = include_str!("shaders/bert.wgsl");
const STENCIL: &str = include_str!("shaders/stencil.wgsl");
const PACK: &str = include_str!("shaders/pack.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Dims {
    w: u32,
    h: u32,
    n: u32,
    flags: u32,
}

const WRITE_LOGITS: u32 = 1;

struct Session {
    w: usize,
    h: usize,
    cells: [wgpu::Buffer; 2],
    logits: wgpu::Buffer,
    packed: wgpu::Buffer,
    dims: wgpu::Buffer,
    step: [wgpu::BindGroup; 2],
    pack: [wgpu::BindGroup; 2],
}

pub struct GpuModel {
    device: wgpu::Device,
    queue: wgpu::Queue,
    kind: Kind,
    label: String,
    weights: wgpu::Buffer,
    pl_step: wgpu::ComputePipeline,
    pl_pack: wgpu::ComputePipeline,
    session: RefCell<Option<Session>>,
}


/// The weights as the kernels read them: a vec4 array in which every
/// matrix is stored transposed (`[in][ceil(out / 4)]` vec4s, rows padded
/// with zeros), so one thread computes four outputs per weight load and a
/// product is `acc += W[j] * x[j]` over whole vec4s. Vectors (biases,
/// LayerNorm, embedding rows) are plain vec4 runs. Offsets are in vec4s.
fn pack_vec4(model: &Model) -> (Vec<f32>, Vec<(String, usize)>) {
    let mut data: Vec<f32> = Vec::new();
    let mut offs = Vec::new();
    for (name, shape) in model.cfg.tensors() {
        let src = model.params.get(&name);
        offs.push((name.clone(), data.len() / 4));
        if shape.len() == 2 && !name.ends_with("emb.weight") {
            let (o, i) = (shape[0], shape[1]);
            let o4 = o.div_ceil(4) * 4;
            for j in 0..i {
                for k in 0..o4 {
                    data.push(if k < o { src[k * i + j] } else { 0.0 });
                }
            }
        } else {
            data.extend_from_slice(src);
            while !data.len().is_multiple_of(4) {
                data.push(0.0);
            }
        }
    }
    // Per-model tables of what depends only on a cell's state (see
    // `cpu::stencil_kv`, `cpu::bert_tables`).
    match model.cfg.kind {
        Kind::Stencil => {
            offs.push(("kv".to_string(), data.len() / 4));
            data.extend(crate::cpu::stencil_kv(&model.cfg, &model.params));
        }
        Kind::Bert => {
            let (kv, q) = crate::cpu::bert_tables(&model.cfg, &model.params);
            offs.push(("kv".to_string(), data.len() / 4));
            data.extend(kv);
            offs.push(("q".to_string(), data.len() / 4));
            data.extend(q);
        }
        Kind::Mlp2 => {}
    }
    (data, offs)
}

fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
}

/// `lin_<k>_<n>(x, w, b)`: `b + W x` for the transposed vec4 matrix at `w`,
/// `x` of `k` vec4s in, `n` vec4s out (sizes are WGSL constant names).
fn lin_fn(k: &str, n: &str) -> String {
    format!(
        "fn lin_{k}_{n}(x: ptr<function, array<vec4<f32>, {k}>>, w: u32, b: u32) -> array<vec4<f32>, {n}> {{
    var o: array<vec4<f32>, {n}>;
    for (var n = 0u; n < {n}; n++) {{ o[n] = wt[b + n]; }}
    for (var j = 0u; j < {k}; j++) {{
        let xv = (*x)[j];
        for (var n = 0u; n < {n}; n++) {{
            let r = w + j * 4u * {n} + n;
            o[n] += wt[r] * xv.x + wt[r + {n}] * xv.y + wt[r + 2u * {n}] * xv.z + wt[r + 3u * {n}] * xv.w;
        }}
    }}
    return o;
}}
"
    )
}

fn wgsl_header(model: &Model, offs: &[(String, usize)]) -> String {
    let c = &model.cfg;
    let mut s = String::new();
    // The constants a kernel does not use are 1, not 0: WGSL rejects zero-sized arrays.
    let (d, heads, dff, hid) = (c.d_model.max(4), c.n_heads.max(1), c.d_ff.max(4), c.hidden.max(4));
    let dk = d / heads;
    let _ = writeln!(s, "const D: u32 = {d}u;\nconst H: u32 = {heads}u;\nconst DK: u32 = {dk}u;\nconst DFF: u32 = {dff}u;\nconst HID: u32 = {hid}u;");
    let _ = writeln!(s, "const DK4: u32 = {}u;\nconst EPS: f32 = {:e};\nconst SCALE: f32 = {:e};", dk / 4, c.eps, (dk as f32).sqrt());
    let _ = writeln!(s, "const D4: u32 = {}u;\nconst DFF4: u32 = {}u;\nconst HID4: u32 = {}u;", d / 4, dff / 4, hid / 4);
    let pairs: &[(&str, &str)] = match c.kind {
        Kind::Mlp2 => &[("HID4", "HID4"), ("HID4", "1")],
        Kind::Bert | Kind::Stencil => &[("D4", "D4"), ("D4", "DFF4"), ("DFF4", "D4"), ("D4", "1")],
    };
    for (k, n) in pairs {
        s.push_str(&lin_fn(k, n));
    }
    for (name, off) in offs {
        let id: String = name.chars().map(|ch| if ch.is_ascii_alphanumeric() { ch.to_ascii_uppercase() } else { '_' }).collect();
        let _ = writeln!(s, "const O_{id}: u32 = {off}u;");
    }
    s
}

/// Workgroups for `threads` threads of 64, as `(x, y)`; the kernels index
/// `(wid.y * num_workgroups.x + wid.x) * 64 + local index`.
fn groups_1d(threads: usize) -> (u32, u32) {
    let count = threads.div_ceil(64).max(1) as u32;
    let x = count.min(32768);
    (x, count.div_ceil(x))
}

impl Drop for GpuModel {
    /// Frees the device before the next one is requested. A dropped wgpu
    /// device is released lazily, and a process that builds many models in a
    /// row (a test binary) otherwise keeps their Vulkan devices alive while
    /// asking the driver for another.
    fn drop(&mut self) {
        self.device.destroy();
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait);
    }
}

impl GpuModel {
    /// A device on the best adapter and the model's pipelines and weights.
    /// `Error::Runtime(NoAdapter)` when there is none: the caller then takes
    /// the next backend of the ladder.
    pub async fn new(model: &Model) -> Result<GpuModel, Error> {
        let adapter = rtgpu::request_adapter().await?;
        let label = rtgpu::adapter_label(&adapter.get_info());
        let (device, queue) = rtgpu::request_device(&adapter, "life", wgpu::Features::empty()).await?;
        let c = &model.cfg;
        let lim = device.limits();
        limits::require_compute(&lim, 64, 0)?;
        for (what, v) in [("d_model", c.d_model), ("d_ff", c.d_ff), ("hidden", c.hidden)] {
            if !v.is_multiple_of(4) {
                return Err(Error::Config(format!("the WebGPU kernels read vec4s: {what} {v} is not a multiple of 4")));
            }
        }
        if c.n_heads > 0 && !(c.d_model / c.n_heads).is_multiple_of(4) {
            return Err(Error::Config(format!("the WebGPU kernels read vec4s: the head size {} is not a multiple of 4", c.d_model / c.n_heads)));
        }
        let (packed, offs) = pack_vec4(model);
        limits::check_binding(&lim, "weights", (packed.len() * 4) as u64)?;
        let body = match c.kind {
            Kind::Mlp2 => MLP2,
            Kind::Bert => BERT,
            Kind::Stencil => STENCIL,
        };
        let src = format!("{}\n{}\n{}", wgsl_header(model, &offs), COMMON, body);
        let pl_step = pipeline::compute_pipeline(&device, "life_step", &src, "main", &[]);
        let pl_pack = pipeline::compute_pipeline(&device, "life_pack", PACK, "main", &[]);
        let weights = buffers::upload(&device, &queue, "life_weights", bytemuck::cast_slice(&packed), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        Ok(GpuModel { device, queue, kind: c.kind, label, weights, pl_step, pl_pack, session: RefCell::new(None) })
    }

    /// "name (backend)" of the adapter in use.
    pub fn adapter(&self) -> &str {
        &self.label
    }

    fn session(&self, w: usize, h: usize) -> Result<std::cell::RefMut<'_, Session>, Error> {
        let mut slot = self.session.borrow_mut();
        if slot.as_ref().is_none_or(|s| s.w != w || s.h != h) {
            *slot = Some(self.make_session(w, h)?);
        }
        Ok(std::cell::RefMut::map(slot, |s| s.as_mut().unwrap()))
    }

    fn make_session(&self, w: usize, h: usize) -> Result<Session, Error> {
        let n = w * h;
        let lim = self.device.limits();
        let cell_bytes = (n * 4) as u64;
        let logit_bytes = (n * self.kind.outputs() * 4) as u64;
        limits::check_binding(&lim, "cells", cell_bytes)?;
        limits::check_binding(&lim, "logits", logit_bytes)?;
        let (gx, gy) = groups_1d(n);
        limits::check_workgroups(&lim, "model", (gx, gy, 1))?;
        if n > u32::MAX as usize / 4 {
            return Err(Error::Board(format!("{w}x{h} has more cells than a 32-bit index covers")));
        }
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let cells = [buffers::empty(&self.device, "cells_a", cell_bytes, rw), buffers::empty(&self.device, "cells_b", cell_bytes, rw)];
        let logits = buffers::empty(&self.device, "logits", logit_bytes, rw);
        let packed = buffers::empty(&self.device, "packed", (n.div_ceil(4) * 4) as u64, rw);
        let dims = buffers::uniform(&self.device, &self.queue, "dims", &Dims { w: w as u32, h: h as u32, n: n as u32, flags: 0 });
        let step = |i: usize| {
            pipeline::bind_group(&self.device, &self.pl_step, 0, &[entry(0, &self.weights), entry(1, &cells[i]), entry(2, &logits), entry(3, &cells[1 - i]), entry(4, &dims)])
        };
        let pack = |i: usize| pipeline::bind_group(&self.device, &self.pl_pack, 0, &[entry(0, &cells[i]), entry(1, &packed), entry(2, &dims)]);
        let (step, pack) = ([step(0), step(1)], [pack(0), pack(1)]);
        Ok(Session { w, h, cells, logits, packed, dims, step, pack })
    }

    fn upload(&self, s: &Session, cells: &[u8]) {
        let words: Vec<u32> = cells.iter().map(|c| (*c != 0) as u32).collect();
        self.queue.write_buffer(&s.cells[0], 0, bytemuck::cast_slice(&words));
    }

    fn dispatch_step(&self, pass: &mut wgpu::ComputePass, s: &Session, i: usize) {
        pass.set_pipeline(&self.pl_step);
        pass.set_bind_group(0, &s.step[i], &[]);
        let (gx, gy) = groups_1d(s.w * s.h);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    /// Logits of the current board, as the CPU path's `Model::logits`.
    pub async fn logits(&self, cells: &[u8], w: usize, h: usize) -> Result<Vec<f32>, Error> {
        let staging = self.submit_logits(cells, w, h)?;
        Ok(buffers::map_read(&self.device, &staging, bytemuck::pod_collect_to_vec::<u8, f32>).await?)
    }

    fn submit_logits(&self, cells: &[u8], w: usize, h: usize) -> Result<wgpu::Buffer, Error> {
        check_board(cells, w, h)?;
        let s = self.session(w, h)?;
        self.upload(&s, cells);
        self.queue.write_buffer(&s.dims, 0, bytemuck::bytes_of(&Dims { w: w as u32, h: h as u32, n: (w * h) as u32, flags: WRITE_LOGITS }));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("life_logits") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("step"), timestamp_writes: None });
            self.dispatch_step(&mut pass, &s, 0);
        }
        let staging = buffers::stage(&self.device, &mut enc, &s.logits, 0, (w * h * self.kind.outputs() * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        Ok(staging)
    }

    /// `generations` steps in one command encoder and one submit, one
    /// readback of the final board (0 or 1 per cell). With zero generations
    /// the board comes back normalised to 0/1.
    pub async fn run(&self, cells: &[u8], w: usize, h: usize, generations: u32) -> Result<Vec<u8>, Error> {
        let staging = self.submit_run(cells, w, h, generations)?;
        let n = w * h;
        buffers::map_read(&self.device, &staging, |b| b[..n].to_vec()).await.map_err(Error::from)
    }

    fn submit_run(&self, cells: &[u8], w: usize, h: usize, generations: u32) -> Result<wgpu::Buffer, Error> {
        check_board(cells, w, h)?;
        let s = self.session(w, h)?;
        self.upload(&s, cells);
        self.queue.write_buffer(&s.dims, 0, bytemuck::bytes_of(&Dims { w: w as u32, h: h as u32, n: (w * h) as u32, flags: 0 }));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("life_run") });
        for g in 0..generations as usize {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("step"), timestamp_writes: None });
            self.dispatch_step(&mut pass, &s, g % 2);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pack"), timestamp_writes: None });
            pass.set_pipeline(&self.pl_pack);
            pass.set_bind_group(0, &s.pack[generations as usize % 2], &[]);
            let (gx, gy) = groups_1d((w * h).div_ceil(4));
            pass.dispatch_workgroups(gx, gy, 1);
        }
        let staging = buffers::stage(&self.device, &mut enc, &s.packed, 0, (w * h).div_ceil(4) as u64 * 4);
        self.queue.submit(Some(enc.finish()));
        Ok(staging)
    }

    /// The next board.
    pub async fn step(&self, cells: &[u8], w: usize, h: usize) -> Result<Vec<u8>, Error> {
        self.run(cells, w, h, 1).await
    }
}
