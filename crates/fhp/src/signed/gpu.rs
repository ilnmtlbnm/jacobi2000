//! Signed ternary FHP on the GPU (`shaders/signed.wgsl`), bit-identical to
//! `Signed`: one u32 per site (two bits per channel, see `two_bit`), the reference phi as 7 i32 per site, and 32
//! words per reference block (reservoir, fluid sites, the old reservoir's
//! share during a re-estimate, the sum of phi, two slots of block values).
//! A step is one dispatch; a re-estimate adds one (`requant`). Block sums
//! are workgroup reductions (workgroup atomics, exact for integers) over
//! block-aligned tiles, written by one workgroup each, with no global atomics.

#![cfg(feature = "gpu")]

use super::{fixed_values, Params, Signed, COLUMNS, NOKICK, WALLS, STREAM_COLLIDE, STREAM_REFRESH, STREAM_REQUANT, STREAM_ROUND};
use crate::average::Average;
use crate::coarse::BlockSums;
use crate::step_key;
use crate::ternary::{Rule, Table, NCH, NSTATE};

const BATCH: usize = 128;
/// Workgroup rows of `step` and `requant` (`WGY` in the shader).
const WGY: usize = 1;
const SLOT: u64 = 256;
const PARAMS: u64 = 80;
/// Words per reference block, and their layout (as in the shader).
const BW: usize = 32;
const RES: usize = 0;
const FLUID: usize = 7;
const NODE: usize = 24;
/// Flags (as in the shader).
const REFRESH: i32 = 1;
const REEST: i32 = 32;
const INIT: i32 = 64;
/// Bytes per site: the largest buffer (phi), and all buffers.
pub const PHI_BYTES: f64 = 28.0;
pub const TOTAL_BYTES: f64 = 36.0;

pub struct Gpu {
    pub w: usize,
    pub h: usize,
    pub params: Params,
    pub step: u32,
    pub adapter: String,
    nbx: usize,
    nby: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    step_pipe: wgpu::ComputePipeline,
    requant_pipe: wgpu::ComputePipeline,
    state: [wgpu::Buffer; 2],
    blk: wgpu::Buffer,
    phi: wgpu::Buffer,
    out_bgl: wgpu::BindGroupLayout,
    /// The coarse params' binding for `phi_out`, which does not read it.
    dummy: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    groups: Vec<[wgpu::BindGroup; 2]>,
    /// The cells are in state[cur]; the block values in node slot `slot`.
    cur: usize,
    slot: u32,
    /// Read-out: its own params slot, phi per site (tests), block sums.
    out_params: wgpu::Buffer,
    phi_pipe: wgpu::ComputePipeline,
    coarse_pipe: wgpu::ComputePipeline,
    coarse: Option<Coarse>,
}

struct Coarse {
    bw: usize,
    bh: usize,
    nx: usize,
    ny: usize,
    out: wgpu::Buffer,
    groups: [wgpu::BindGroup; 2],
    /// The running time average of the samples.
    avg: Average,
}

/// A cell from the CPU's form (base-3 code of delta + 1, flags in bits 14
/// and 15) to the GPU's (delta + 1 in two bits per channel, same flags).
fn two_bit(c: u32) -> u32 {
    let mut code = c & 0x3fff;
    let mut out = c & 0xc000;
    for ch in 0..NCH {
        out |= (code % 3) << (2 * ch);
        code /= 3;
    }
    out
}

/// The inverse of `two_bit`.
fn base3(c: u32) -> u32 {
    (0..NCH).rev().fold(0, |code, ch| code * 3 + ((c >> (2 * ch)) & 3)) | (c & 0xc000)
}

fn bytes<T: Copy + Into<i64>>(v: &[T]) -> Vec<u8> {
    v.iter().flat_map(|&x| (x.into() as u32).to_le_bytes()).collect()
}

/// The state read back: cells, phi per site, block values, reservoirs.
pub struct State {
    pub cells: Vec<u16>,
    pub phi: Vec<[i32; NCH]>,
    pub nodes: Vec<[i32; 3]>,
    pub res: Vec<[i32; NCH]>,
}

impl Gpu {
    /// A GPU lattice with these cells, block values and reservoirs; the
    /// reference phi is computed from the block values on the GPU (as
    /// `Signed::interpolated`). The lattice is checked against the adapter's limits before `state` is
    /// called.
    pub async fn new(w: usize, h: usize, params: Params, state: impl FnOnce() -> (Vec<u16>, Vec<[i32; 3]>, Vec<[i32; NCH]>)) -> Result<Self, String> {
        crate::check_shape(w, h)?;
        let b = params.block;
        if !(b.is_power_of_two() && (2..=64).contains(&b) && w.is_multiple_of(b) && h.is_multiple_of(b)) {
            return Err(format!("reference block {b}: a power of two from 2 to 64 dividing {w}x{h}"));
        }
        if !matches!(params.rule, Rule::Optimal | Rule::Chiral) {
            return Err("the GPU engine has rules optimal and chiral".into());
        }
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, force_fallback_adapter: false, compatible_surface: None })
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let limits = adapter.limits();
        crate::gpu::Gpu::size_limits(&limits).check_bytes(w, h, PHI_BYTES, TOTAL_BYTES)?;
        let info = adapter.get_info();
        let adapter_name = if info.name.is_empty() { format!("{:?}", info.backend) } else { format!("{} ({:?})", info.name, info.backend) };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fhp-signed"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("no WebGPU device: {e}"))?;
        let (cells, nodes, res) = state();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("signed"), source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/signed.wgsl").into()) });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let storage = |read_only| wgpu::BufferBindingType::Storage { read_only };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("signed"),
            entries: &[entry(0, wgpu::BufferBindingType::Uniform), entry(1, storage(true)), entry(2, storage(false)), entry(3, storage(true)), entry(4, storage(false)), entry(5, storage(false))],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("signed"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipe = |entry: &str, layout: Option<&wgpu::PipelineLayout>| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout, module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        let step_pipe = pipe("step", Some(&layout));
        let requant_pipe = pipe("requant", Some(&layout));
        let out_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("signed read-out"),
            entries: &[entry(0, wgpu::BufferBindingType::Uniform), entry(1, storage(true)), entry(4, storage(false)), entry(5, storage(false)), entry(6, wgpu::BufferBindingType::Uniform), entry(7, storage(false))],
        });
        let out_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("signed read-out"), bind_group_layouts: &[&out_bgl], push_constant_ranges: &[] });
        let phi_pipe = pipe("phi_out", Some(&out_layout));
        let coarse_pipe = pipe("coarse", Some(&out_layout));

        let t = Table::new();
        let mut tab: Vec<u32> = t.chiral.iter().map(|c| two_bit(c[0] as u32) | two_bit(c[1] as u32) << 16).collect();
        tab.extend((0..NSTATE as u16).map(|s| two_bit(crate::ternary::bounce(s) as u32)));
        tab.extend(t.opt.iter().flat_map(|o| o.chunks(2).map(|p| two_bit(p[0] as u32) | two_bit(p[1] as u32) << 16)));

        let (nbx, nby) = (w / b, h / b);
        let mut blk = vec![0i32; nbx * nby * BW];
        for (i, c) in cells.iter().enumerate() {
            if c & super::OBST == 0 {
                blk[((i / w / b) * nbx + (i % w) / b) * BW + FLUID] += 1;
            }
        }
        for k in 0..nbx * nby {
            let o = k * BW;
            blk[o + RES..o + RES + NCH].copy_from_slice(&res[k]);
            blk[o + NODE..o + NODE + 3].copy_from_slice(&nodes[k]);
        }

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, size: usize| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage, mapped_at_creation: false });
        let n = w * h;
        let state = [mk("state0", n * 4), mk("state1", n * 4)];
        let tabb = mk("table", tab.len() * 4);
        let blkb = mk("blocks", blk.len() * 4);
        let phib = mk("phi", n * NCH * 4);
        for (k, part) in cells.chunks(1 << 20).enumerate() {
            let part: Vec<u32> = part.iter().map(|&c| two_bit(c as u32)).collect();
            queue.write_buffer(&state[0], (k << 22) as u64, &bytes(&part));
        }
        queue.write_buffer(&tabb, 0, &bytes(&tab));
        queue.write_buffer(&blkb, 0, &bytes(&blk));
        drop(cells);
        let uniform = |label: &str, size: u64| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let uniforms = uniform("params", SLOT * BATCH as u64);
        let out_params = uniform("read-out params", PARAMS);
        let dummy = uniform("dummy", 16);
        let groups = (0..BATCH)
            .map(|j| {
                [0, 1].map(|k| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &uniforms, offset: j as u64 * SLOT, size: wgpu::BufferSize::new(PARAMS) }) },
                            wgpu::BindGroupEntry { binding: 1, resource: state[k].as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: state[1 - k].as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: tabb.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 4, resource: blkb.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 5, resource: phib.as_entire_binding() },
                        ],
                    })
                })
            })
            .collect();
        let g = Gpu {
            w,
            h,
            params,
            step: 0,
            adapter: adapter_name,
            nbx,
            nby,
            device,
            queue,
            step_pipe,
            requant_pipe,
            state,
            blk: blkb,
            phi: phib,
            out_bgl,
            dummy,
            uniforms,
            groups,
            cur: 0,
            slot: 0,
            out_params,
            phi_pipe,
            coarse_pipe,
            coarse: None,
        };
        // phi from the block values, and each block's sum of phi.
        let s = g.words(0, false, INIT);
        g.queue.write_buffer(&g.uniforms, 0, &bytes(&s));
        let mut enc = g.device.create_command_encoder(&Default::default());
        {
            // Group 1 of slot 0 has binding 2 = state[0].
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&g.requant_pipe);
            pass.set_bind_group(0, &g.groups[0][1], &[]);
            let (tx, ty) = g.tiles();
            pass.dispatch_workgroups(tx, ty, 1);
        }
        g.queue.submit(Some(enc.finish()));
        Ok(g)
    }

    /// The GPU copy of a CPU lattice (cells, block values, reservoirs; the
    /// CPU's phi must be the interpolation of its block values, as it is
    /// after `sample`, `from_state` and every re-estimate).
    pub async fn from_cpu(s: &Signed) -> Result<Self, String> {
        let mut g = Self::new(s.w, s.h, s.params, || (s.cells.clone(), s.nodes.clone(), s.res.clone())).await?;
        g.step = s.step;
        Ok(g)
    }

    /// Rows per invocation of `step` and `requant`: R = B up to 16 (an
    /// invocation's sites are one column of one block, so a tile has few
    /// blocks and few atomic additions per site), with at most 64 blocks per
    /// tile (R <= B^2 / WGY) and whole blocks in a tile (R >= B / WGY).
    fn rows(&self) -> usize {
        let b = self.params.block;
        b.min(16).min(b * b / WGY).max(b / WGY)
    }

    /// Workgroups of `step` and `requant`: tiles of 64 columns by WGY R rows
    /// (the last one partly empty when they do not divide H).
    fn tiles(&self) -> (u32, u32) {
        ((self.w / 64) as u32, self.h.div_ceil(WGY * self.rows()) as u32)
    }

    /// The uniform words of step `t` (node slot `self.slot`).
    fn words(&self, t: u32, refresh: bool, extra: i32) -> Vec<i32> {
        let p = &self.params;
        let b = p.block;
        let (r_in, jx_in) = p.boundary.map_or((0, 0), |bd| {
            let v = fixed_values(bd.rho, bd.u, 0.0);
            (v[0], v[1])
        });
        let (inlet, outlet_start) = p.boundary.map_or((0, 0), |bd| (bd.inlet, self.w - bd.outlet));
        let flags = (refresh as i32 * REFRESH)
            | (NOKICK.load(std::sync::atomic::Ordering::Relaxed) as i32) << 1
            | (p.boundary.is_some() as i32) << 2
            | (WALLS.load(std::sync::atomic::Ordering::Relaxed) as i32) << 3
            | (COLUMNS.load(std::sync::atomic::Ordering::Relaxed) as i32) << 4
            | extra;
        let key = |stream| step_key(p.seed, t, stream) as i32;
        vec![
            self.w as i32,
            self.h as i32,
            b as i32,
            (2 * b).trailing_zeros() as i32,
            self.nbx as i32,
            self.nby as i32,
            (p.rule == Rule::Chiral) as i32,
            flags,
            key(STREAM_COLLIDE),
            key(STREAM_ROUND),
            key(STREAM_REQUANT),
            key(STREAM_REFRESH),
            inlet as i32,
            outlet_start as i32,
            r_in,
            jx_in,
            p.rho0(),
            self.slot as i32,
            self.rows().trailing_zeros() as i32,
            0,
        ]
    }

    /// Queue `n` steps (does not wait).
    pub fn run(&mut self, n: usize) {
        let mut left = n;
        let tiles = self.tiles();
        while left > 0 {
            let k = left.min(BATCH);
            let mut u = vec![0i32; k * SLOT as usize / 4];
            let mut re = vec![false; k];
            let slot0 = self.slot;
            for j in 0..k {
                let t = self.step + j as u32;
                let (r, refresh) = Signed::reestimates(&self.params, t);
                re[j] = r;
                let o = j * SLOT as usize / 4;
                let s = self.words(t, refresh, if r { REEST } else { 0 });
                u[o..o + s.len()].copy_from_slice(&s);
                if r {
                    self.slot = 1 - self.slot;
                }
            }
            self.slot = slot0;
            self.queue.write_buffer(&self.uniforms, 0, &bytes(&u));
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for j in 0..k {
                    let g = &self.groups[j][self.cur];
                    pass.set_bind_group(0, g, &[]);
                    pass.set_pipeline(&self.step_pipe);
                    pass.dispatch_workgroups(tiles.0, tiles.1, 1);
                    self.cur = 1 - self.cur;
                    if re[j] {
                        // Same group, other direction: binding 2 is now the
                        // current cells.
                        pass.set_bind_group(0, &self.groups[j][1 - self.cur], &[]);
                        pass.set_pipeline(&self.requant_pipe);
                        pass.dispatch_workgroups(tiles.0, tiles.1, 1);
                        self.slot = 1 - self.slot;
                    }
                }
            }
            self.queue.submit(Some(enc.finish()));
            self.step += k as u32;
            left -= k;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn finish(&self) {
        self.device.poll(wgpu::PollType::Wait).expect("device poll failed");
    }

    async fn read_buffer(&self, buf: &wgpu::Buffer, size: u64) -> Vec<i32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::Wait).expect("device poll failed");
        rx.await.expect("map_async channel dropped").expect("buffer map failed");
        let data = slice.get_mapped_range();
        let out = data.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
        drop(data);
        staging.unmap();
        out
    }

    /// The read-out kernels' params: the current step's words.
    fn write_out_params(&self) {
        self.queue.write_buffer(&self.out_params, 0, &bytes(&self.words(self.step, false, 0)));
    }

    /// Cells, phi, block values and reservoirs (for tests).
    pub async fn read_state(&self) -> State {
        let n = self.w * self.h;
        let nb = self.nbx * self.nby;
        let phib = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("phi"), size: (n * NCH * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        self.write_out_params();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.out_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.out_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.state[self.cur].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.blk.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.phi.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: phib.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.phi_pipe);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((self.w / 64) as u32, self.h as u32, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let cells = self.read_buffer(&self.state[self.cur], (n * 4) as u64).await.into_iter().map(|v| base3(v as u32) as u16).collect();
        let p = self.read_buffer(&phib, (n * NCH * 4) as u64).await;
        let phi = (0..n).map(|i| std::array::from_fn(|c| p[c * n + i])).collect();
        let b = self.read_buffer(&self.blk, (nb * BW * 4) as u64).await;
        let s = NODE + self.slot as usize * 4;
        let nodes = (0..nb).map(|k| [b[k * BW + s], b[k * BW + s + 1], b[k * BW + s + 2]]).collect();
        let res = (0..nb).map(|k| std::array::from_fn(|c| b[k * BW + RES + c])).collect();
        State { cells, phi, nodes, res }
    }

    /// Add the current block sums into the GPU-side accumulator (queued, no
    /// wait). Changing the block size resets it.
    pub fn accumulate_sums(&mut self, bw: usize, bh: usize) {
        assert!(self.w.is_multiple_of(bw) && self.h.is_multiple_of(bh));
        if self.coarse.as_ref().is_none_or(|c| (c.bw, c.bh) != (bw, bh)) {
            self.coarse = Some(self.make_coarse(bw, bh));
        }
        self.write_out_params();
        let c = self.coarse.as_ref().unwrap();
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.coarse_pipe);
            pass.set_bind_group(0, &c.groups[self.cur], &[]);
            pass.dispatch_workgroups(c.nx as u32, c.ny as u32, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Read the accumulated block sums and clear the accumulator.
    pub async fn take_sums(&self) -> BlockSums {
        let c = self.coarse.as_ref().expect("accumulate_sums first");
        let size = (c.nx * c.ny * 16) as u64;
        let v = self.read_buffer(&c.out, size).await;
        self.queue.write_buffer(&c.out, 0, &vec![0u8; size as usize]);
        let col = |k: usize| v.iter().skip(k).step_by(4).copied().collect();
        BlockSums { nx: c.nx, ny: c.ny, bw: c.bw, bh: c.bh, jx: col(0), jy: col(1), mass: col(2), fluid: col(3) }
    }

    /// Sample the current block sums and fold them into the running time
    /// average on the GPU (queued, no wait, no read-back). Changing the
    /// block size resets the average.
    pub fn sample_sums(&mut self, bw: usize, bh: usize) {
        assert!(self.w.is_multiple_of(bw) && self.h.is_multiple_of(bh));
        if self.coarse.as_ref().is_none_or(|c| (c.bw, c.bh) != (bw, bh)) {
            self.coarse = Some(self.make_coarse(bw, bh));
        }
        self.write_out_params();
        let c = self.coarse.as_mut().unwrap();
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&c.out, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.coarse_pipe);
            pass.set_bind_group(0, &c.groups[self.cur], &[]);
            pass.dispatch_workgroups(c.nx as u32, c.ny as u32, 1);
        }
        c.avg.fold(&self.queue, &mut enc);
        self.queue.submit(Some(enc.finish()));
    }

    /// Queue a read-back of the running average unless one is in flight.
    pub fn request_average(&mut self) {
        if let Some(c) = self.coarse.as_mut() {
            c.avg.request(&self.device, &self.queue);
        }
    }

    /// The read-back running average (4 words per block) and its sample
    /// count, if it has arrived (does not wait).
    pub fn average_ready(&mut self) -> Option<(Vec<i32>, u32)> {
        self.coarse.as_mut()?.avg.ready(&self.device)
    }

    /// Read-backs of the average whose map failed.
    pub fn average_failures(&self) -> u32 {
        self.coarse.as_ref().map_or(0, |c| c.avg.failures)
    }

    /// Whether a read-back of the average is in flight.
    pub fn average_reading(&self) -> bool {
        self.coarse.as_ref().is_some_and(|c| c.avg.reading())
    }

    /// A receiver completed when the read-back of the average in flight
    /// lands (None if none is pending).
    pub fn average_waiter(&self) -> Option<futures_channel::oneshot::Receiver<()>> {
        self.coarse.as_ref()?.avg.waiter()
    }

    /// Wait for the read-back of the average in flight.
    pub async fn wait_average(&mut self) {
        if let Some(c) = self.coarse.as_mut() {
            c.avg.wait(&self.device).await;
        }
    }

    fn make_coarse(&self, bw: usize, bh: usize) -> Coarse {
        let (nx, ny) = (self.w / bw, self.h / bh);
        let u = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("coarse params"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.queue.write_buffer(&u, 0, &bytes(&[bw, bh, nx, ny].map(|v| v as i32)));
        let size = (nx * ny * 16) as u64;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("sums"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.queue.write_buffer(&out, 0, &vec![0u8; size as usize]);
        let groups = [0, 1].map(|k| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.out_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.out_params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.state[k].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.blk.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.phi.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: out.as_entire_binding() },
                ],
            })
        });
        let avg = Average::new(&self.device, &out, nx * ny);
        Coarse { bw, bh, nx, ny, out, groups, avg }
    }
}
