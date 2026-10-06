//! GPU engine (wgpu: Metal natively, WebGPU in the browser). Bit-sliced
//! panels run the WGSL emitted by `Circuit::to_wgsl` on u32 words (the CPU
//! engine's u64 layout read as u32 pairs); the f32 panel runs `F32_WGSL`.

use crate::circuit::{Circuit, F32_WGSL};

pub struct GpuCtx {
    pub adapter: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipe_layout: wgpu::PipelineLayout,
    /// Per-dispatch uniform (dynamic offset), src, dst, an rw buffer and a
    /// read buffer: the three approaches' kernels.
    layout2: wgpu::BindGroupLayout,
    pipe_layout2: wgpu::PipelineLayout,
    limits: wgpu::Limits,
}

impl GpuCtx {
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, force_fallback_adapter: false, compatible_surface: None })
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let limits = adapter.limits();
        let info = adapter.get_info();
        let name = if info.name.is_empty() { String::new() } else { format!("{} ({:?})", info.name, info.backend) };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("jacobi"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("no WebGPU device: {e}"))?;
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jacobi"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("jacobi"), bind_group_layouts: &[&layout], push_constant_ranges: &[] });
        let layout2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jacobi2"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
                storage(3, false),
                storage(4, true),
            ],
        });
        let pipe_layout2 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("jacobi2"), bind_group_layouts: &[&layout2], push_constant_ranges: &[] });
        Ok(GpuCtx { adapter: name, device, queue, layout, pipe_layout, layout2, pipe_layout2, limits })
    }

    fn pipeline(&self, src: &str) -> wgpu::ComputePipeline {
        self.pipeline_with(src, &self.pipe_layout)
    }

    fn pipeline_with(&self, src: &str, layout: &wgpu::PipelineLayout) -> wgpu::ComputePipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Resolves when all submitted work is done.
    pub async fn done(&self) {
        let (tx, rx) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::Wait).expect("device poll failed");
        let _ = rx.await;
    }

    /// Wait for queued work (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn finish(&self) {
        self.device.poll(wgpu::PollType::Wait).expect("device poll failed");
    }
}

fn bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    F32,
    Bits(usize),
    /// b-bit circuit with stochastic rounding.
    Stochastic(usize),
    /// Ternary corrections, 32 planes; one iteration = two half-sweeps.
    Ternary,
    /// Lattice gas, 8 planes plus charge accumulators.
    Gas,
    /// f32 in red-black order; one iteration = two half-sweeps.
    F32RedBlack,
    /// Plain integer red-black, 32 planes; one iteration = two half-sweeps.
    Int32RedBlack,
}

fn words32(w64: &[u64]) -> Vec<u32> {
    w64.iter().flat_map(|w| [*w as u32, (*w >> 32) as u32]).collect()
}

/// Per-dispatch slots of the approaches' kernels.
struct Slots {
    buf: wgpu::Buffer,
    /// Rw buffer at binding 3 (ternary control words, gas accumulators).
    aux: wgpu::Buffer,
    aux_words: usize,
}

const MAX_SLOTS: usize = 256;

/// One field on the device, ping-ponged, with its kernels.
pub struct GpuPanel {
    pub n: usize,
    pub kind: Kind,
    pub round: bool,
    pub iter: u64,
    pub seed: u32,
    words: usize,
    slots: Option<Slots>,
    bufs: [wgpu::Buffer; 2],
    groups: [wgpu::BindGroup; 2],
    /// F32: one pipeline; Bits: truncating, round to nearest.
    pipes: Vec<wgpu::ComputePipeline>,
    dispatch: (u32, u32),
    cur: usize,
}

impl GpuPanel {
    pub fn f32(ctx: &GpuCtx, p: &crate::Problem) -> Result<Self, String> {
        let init: Vec<u32> = p.initial_f32().iter().map(|f| f.to_bits()).collect();
        Self::build(ctx, p.n, Kind::F32, &init, vec![ctx.pipeline(F32_WGSL)], ((p.n as u32).div_ceil(64), p.n as u32))
    }

    pub fn bits(ctx: &GpuCtx, p: &crate::Problem, bits: usize) -> Result<Self, String> {
        Self::from_q(ctx, p.n, bits, &p.initial_q(bits))
    }

    /// Bit-sliced panel from a row-major field of b-bit values.
    pub fn from_q(ctx: &GpuCtx, n: usize, bits: usize, q: &[u32]) -> Result<Self, String> {
        let w64 = crate::bitsliced::pack(n, bits, q);
        let init: Vec<u32> = w64.iter().flat_map(|w| [*w as u32, (*w >> 32) as u32]).collect();
        let pipes = vec![ctx.pipeline(&Circuit::jacobi(bits, false).to_wgsl()), ctx.pipeline(&Circuit::jacobi(bits, true).to_wgsl())];
        let wpr = (n / 32) as u32;
        Self::build(ctx, n, Kind::Bits(bits), &init, pipes, (wpr.div_ceil(64), n as u32))
    }

    fn build(ctx: &GpuCtx, n: usize, kind: Kind, init: &[u32], pipes: Vec<wgpu::ComputePipeline>, dispatch: (u32, u32)) -> Result<Self, String> {
        let size = (init.len() * 4) as u64;
        let max = lean_runtime::limits::max_binding_bytes(&ctx.limits);
        if size > max {
            return Err(format!("grid {n} needs a {size}-byte buffer, the adapter allows {max}"));
        }
        if dispatch.0.max(dispatch.1) > ctx.limits.max_compute_workgroups_per_dimension {
            return Err(format!("grid {n} needs {} workgroups in one dimension, the adapter allows {}", dispatch.0.max(dispatch.1), ctx.limits.max_compute_workgroups_per_dimension));
        }
        let d = &ctx.device;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let bufs = [0, 1].map(|_| {
            let b = d.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false });
            ctx.queue.write_buffer(&b, 0, &bytes(init));
            b
        });
        let wpr = (n / 32) as u32;
        let params = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        ctx.queue.write_buffer(&params, 0, &bytes(&[wpr, n as u32, wpr * n as u32, 0]));
        let groups = [0, 1].map(|k| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bufs[k].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bufs[1 - k].as_entire_binding() },
                ],
            })
        });
        Ok(GpuPanel { n, kind, round: false, iter: 0, seed: 1, words: init.len(), slots: None, bufs, groups, pipes, dispatch, cur: 0 })
    }

    /// Stochastic-rounding panel from a row-major field of b-bit values.
    pub fn stochastic(ctx: &GpuCtx, n: usize, bits: usize, q: &[u32], seed: u32) -> Result<Self, String> {
        let w64 = crate::bitsliced::pack(n, bits, q);
        let pipe = ctx.pipeline_with(&crate::kernels::stochastic(bits), &ctx.pipe_layout2);
        let wpr = (n / 32) as u32;
        let mut g = Self::build2(ctx, n, Kind::Stochastic(bits), &words32(&w64), pipe, &[0; 4], &[0; 4], (wpr.div_ceil(64), n as u32))?;
        g.seed = seed;
        Ok(g)
    }

    /// Ternary panel from a row-major i32 field.
    pub fn ternary(ctx: &GpuCtx, n: usize, u: &[i32]) -> Result<Self, String> {
        let q: Vec<u32> = u.iter().map(|&x| x as u32).collect();
        let w64 = crate::bitsliced::pack(n, crate::ternary::W, &q);
        let pipe = ctx.pipeline_with(&crate::kernels::ternary(), &ctx.pipe_layout2);
        let wpr = (n / 32) as u32;
        Self::build2(ctx, n, Kind::Ternary, &words32(&w64), pipe, &[0; 4], &[0; 4], (wpr.div_ceil(64), n as u32))
    }

    /// Lattice gas panel from site states (bits 0..3 +, 4..7 -).
    pub fn gas(ctx: &GpuCtx, p: &crate::Problem, state: &[u8], seed: u32, annihilation: bool) -> Result<Self, String> {
        let n = p.n;
        let q: Vec<u32> = state.iter().map(|&x| x as u32).collect();
        let w64 = crate::bitsliced::pack(n, 8, &q);
        let pipe = ctx.pipeline_with(&crate::kernels::gas(annihilation), &ctx.pipe_layout2);
        let wpr = (n / 32) as u32;
        let acc = vec![0u32; n * n / crate::gas::GROUP];
        let mut g = Self::build2(ctx, n, Kind::Gas, &words32(&w64), pipe, &acc, &crate::gas::thresholds(p), (wpr.div_ceil(64), n as u32))?;
        g.seed = seed;
        Ok(g)
    }

    /// f32 panel in red-black order.
    pub fn f32_red_black(ctx: &GpuCtx, p: &crate::Problem) -> Result<Self, String> {
        let init: Vec<u32> = p.initial_f32().iter().map(|f| f.to_bits()).collect();
        let pipe = ctx.pipeline_with(&crate::kernels::f32_rb(), &ctx.pipe_layout2);
        Self::build2(ctx, p.n, Kind::F32RedBlack, &init, pipe, &[0; 4], &[0; 4], ((p.n as u32).div_ceil(64), p.n as u32))
    }

    /// Plain integer red-black panel from a row-major i32 field.
    pub fn int32_red_black(ctx: &GpuCtx, n: usize, u: &[i32]) -> Result<Self, String> {
        let q: Vec<u32> = u.iter().map(|&x| x as u32).collect();
        let w64 = crate::bitsliced::pack(n, crate::ternary::W, &q);
        let pipe = ctx.pipeline_with(&crate::kernels::int32_rb(), &ctx.pipe_layout2);
        let wpr = (n / 32) as u32;
        Self::build2(ctx, n, Kind::Int32RedBlack, &words32(&w64), pipe, &[0; 4], &[0; 4], (wpr.div_ceil(64), n as u32))
    }

    #[allow(clippy::too_many_arguments)]
    fn build2(ctx: &GpuCtx, n: usize, kind: Kind, init: &[u32], pipe: wgpu::ComputePipeline, aux: &[u32], read: &[u32], dispatch: (u32, u32)) -> Result<Self, String> {
        let size = (init.len() * 4) as u64;
        let max = lean_runtime::limits::max_binding_bytes(&ctx.limits);
        if size > max {
            return Err(format!("grid {n} needs a {size}-byte buffer, the adapter allows {max}"));
        }
        if dispatch.0.max(dispatch.1) > ctx.limits.max_compute_workgroups_per_dimension {
            return Err(format!("grid {n} needs {} workgroups in one dimension, the adapter allows {}", dispatch.0.max(dispatch.1), ctx.limits.max_compute_workgroups_per_dimension));
        }
        let d = &ctx.device;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let storage = |data: &[u32]| {
            let b = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: (data.len() * 4) as u64, usage, mapped_at_creation: false });
            ctx.queue.write_buffer(&b, 0, &bytes(data));
            b
        };
        let bufs = [storage(init), storage(init)];
        let aux_buf = storage(aux);
        let read_buf = storage(read);
        let slot_buf = d.create_buffer(&wgpu::BufferDescriptor { label: Some("slots"), size: crate::kernels::SLOT * MAX_SLOTS as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let groups = [0, 1].map(|k| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.layout2,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &slot_buf, offset: 0, size: wgpu::BufferSize::new((crate::kernels::SLOT_WORDS * 4) as u64) }) },
                    wgpu::BindGroupEntry { binding: 1, resource: bufs[k].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bufs[1 - k].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: aux_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: read_buf.as_entire_binding() },
                ],
            })
        });
        let slots = Slots { buf: slot_buf, aux: aux_buf, aux_words: aux.len() };
        Ok(GpuPanel { n, kind, round: false, iter: 0, seed: 1, words: init.len(), slots: Some(slots), bufs, groups, pipes: vec![pipe], dispatch, cur: 0 })
    }

    /// Slot values of dispatch `d` of iteration `it` (approaches only).
    fn slot(&self, it: u64, d: usize) -> [u32; crate::kernels::SLOT_WORDS] {
        let n = self.n as u32;
        let wpr = n / 32;
        let mut v = [wpr, n, wpr * n, n, 0, 0, 0, 0, 0, 0, 0, 0];
        match self.kind {
            Kind::Stochastic(_) => {
                let k = crate::stochastic::keys(self.seed, it as u32);
                v[4] = k[0];
                v[5] = k[1];
            }
            Kind::Ternary => {
                let k = it % 3;
                v[4] = d as u32;
                v[5] = k as u32;
                v[6] = ((k + 2) % 3) as u32;
                v[7] = ((k + 1) % 3) as u32;
                v[8] = (it == 0) as u32;
            }
            Kind::Gas => {
                let k = crate::gas::keys(self.seed, it as u32);
                v[4..7].copy_from_slice(&k);
            }
            Kind::F32RedBlack | Kind::Int32RedBlack => v[4] = d as u32,
            _ => unreachable!(),
        }
        v
    }

    /// Queue `iters` iterations (does not wait).
    pub fn run(&mut self, ctx: &GpuCtx, iters: usize) {
        if self.slots.is_some() {
            return self.run_slots(ctx, iters);
        }
        let pipe = &self.pipes[if self.pipes.len() > 1 { self.round as usize } else { 0 }];
        let mut left = iters;
        while left > 0 {
            let k = left.min(256);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipe);
                for _ in 0..k {
                    pass.set_bind_group(0, &self.groups[self.cur], &[]);
                    pass.dispatch_workgroups(self.dispatch.0, self.dispatch.1, 1);
                    self.cur = 1 - self.cur;
                }
            }
            ctx.queue.submit(Some(enc.finish()));
            left -= k;
        }
        self.iter += iters as u64;
    }

    fn run_slots(&mut self, ctx: &GpuCtx, iters: usize) {
        let per = if matches!(self.kind, Kind::Ternary | Kind::F32RedBlack | Kind::Int32RedBlack) { 2 } else { 1 };
        let mut left = iters;
        while left > 0 {
            let k = left.min(MAX_SLOTS / per);
            let mut data = vec![0u8; crate::kernels::SLOT as usize * k * per];
            for t in 0..k {
                for d in 0..per {
                    let v = self.slot(self.iter + t as u64, d);
                    let o = (t * per + d) * crate::kernels::SLOT as usize;
                    data[o..o + v.len() * 4].copy_from_slice(&bytes(&v));
                }
            }
            let slots = self.slots.as_ref().unwrap();
            ctx.queue.write_buffer(&slots.buf, 0, &data);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipes[0]);
                for j in 0..k * per {
                    pass.set_bind_group(0, &self.groups[self.cur], &[(j as u64 * crate::kernels::SLOT) as u32]);
                    pass.dispatch_workgroups(self.dispatch.0, self.dispatch.1, 1);
                    self.cur = 1 - self.cur;
                }
            }
            ctx.queue.submit(Some(enc.finish()));
            self.iter += k as u64;
            left -= k;
        }
    }

    /// The rw auxiliary buffer (gas accumulators, ternary control words).
    pub async fn read_aux(&self, ctx: &GpuCtx) -> Vec<u32> {
        let s = self.slots.as_ref().expect("read_aux on a panel without one");
        read_buffer(ctx, &s.aux, s.aux_words).await
    }

    /// The current field's raw words.
    pub async fn read_words(&self, ctx: &GpuCtx) -> Vec<u32> {
        read_buffer(ctx, &self.bufs[self.cur], self.words).await
    }

    /// Bit-sliced panel: the field as row-major values of its plane count.
    pub async fn read_planes(&self, ctx: &GpuCtx, planes: usize) -> Vec<u32> {
        let w = self.read_words(ctx).await;
        let w64: Vec<u64> = w.chunks_exact(2).map(|p| p[0] as u64 | (p[1] as u64) << 32).collect();
        crate::bitsliced::unpack(self.n, planes, &w64)
    }

    /// Bit-sliced panel: the field as row-major b-bit values.
    pub async fn read_q(&self, ctx: &GpuCtx) -> Vec<u32> {
        let (Kind::Bits(bits) | Kind::Stochastic(bits)) = self.kind else { panic!("read_q on a panel without b-bit values") };
        self.read_planes(ctx, bits).await
    }

    pub async fn read_f32(&self, ctx: &GpuCtx) -> Vec<f32> {
        self.read_words(ctx).await.into_iter().map(f32::from_bits).collect()
    }
}

/// Copy `words` u32 of `buf` back to the host.
async fn read_buffer(ctx: &GpuCtx, buf: &wgpu::Buffer, words: usize) -> Vec<u32> {
    let size = (words * 4) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
    ctx.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    #[cfg(not(target_arch = "wasm32"))]
    ctx.device.poll(wgpu::PollType::Wait).expect("device poll failed");
    rx.await.expect("map_async channel dropped").expect("buffer map failed");
    let data = slice.get_mapped_range();
    let out = data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    drop(data);
    staging.unmap();
    out
}
