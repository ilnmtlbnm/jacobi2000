//! GPU ternary engines (rule Chiral), bit-identical to
//! `bitsliced::Bitsliced` and `engine::Ternary`: the bit-sliced step in WGSL
//! with u32 words (`Layout::Bitsliced`, periodic only), or one u32 per site
//! and the collision as a table lookup (`Layout::Table`, with the open
//! boundary of `engine::Boundary` and block sums). Same structure as
//! `crate::gpu` without the walls.

use super::bitsliced::PLANES;
use super::collide_gen::ENCODING;
use super::engine::{Boundary, EVERY, OBST, STREAM_BOUNDARY, STREAM_COLLIDE};
use crate::coarse::BlockSums;
use crate::step_key;

const BATCH: usize = 128;
const SLOT: u64 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Bitsliced,
    Table,
}

pub struct Gpu {
    pub layout: Layout,
    pub w: usize,
    pub h: usize,
    pub wn: usize,
    pub seed: u32,
    pub step: u32,
    pub adapter: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipe: wgpu::ComputePipeline,
    state: [wgpu::Buffer; 2],
    obst: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    groups: Vec<[wgpu::BindGroup; 2]>,
    cur: usize,
    boundary: Option<Boundary>,
    coarse_pipe: Option<wgpu::ComputePipeline>,
    coarse: Option<Coarse>,
}

struct Coarse {
    bw: usize,
    bh: usize,
    nx: usize,
    ny: usize,
    out: wgpu::Buffer,
    groups: [wgpu::BindGroup; 2],
}

fn bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Gpu {
    pub async fn new(w: usize, h: usize, cells: &[u16], seed: u32) -> Result<Self, String> {
        Self::with_layout(w, h, cells, seed, Layout::Bitsliced).await
    }

    pub async fn with_layout(w: usize, h: usize, cells: &[u16], seed: u32, layout: Layout) -> Result<Self, String> {
        Self::open(w, h, || cells.to_vec(), seed, layout, None).await
    }

    /// `with_layout` with an optional open boundary (`Layout::Table` only).
    /// The lattice is checked against the adapter's limits before `cells`
    /// is called.
    pub async fn open(w: usize, h: usize, cells: impl FnOnce() -> Vec<u16>, seed: u32, layout: Layout, boundary: Option<Boundary>) -> Result<Self, String> {
        crate::check_shape(w, h)?;
        if boundary.is_some() && layout != Layout::Table {
            return Err("open boundaries need the table layout".into());
        }
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, force_fallback_adapter: false, compatible_surface: None })
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let limits = adapter.limits();
        if layout == Layout::Table {
            crate::gpu::Gpu::size_limits(&limits).check_bytes(w, h, 4.0, 0.0)?;
        }
        let info = adapter.get_info();
        let adapter_name = if info.name.is_empty() { format!("{:?}", info.backend) } else { format!("{} ({:?})", info.name, info.backend) };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fhp-ternary"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("no WebGPU device: {e}"))?;
        let cells = cells();
        let src = match layout {
            Layout::Bitsliced => format!("{}\n{}", include_str!("../shaders/ternary_collide_gen.wgsl"), include_str!("../shaders/ternary_step.wgsl")),
            Layout::Table => include_str!("../shaders/ternary_table.wgsl").to_string(),
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ternary"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ternary step"),
            layout: None,
            module: &module,
            entry_point: Some("step"),
            compilation_options: Default::default(),
            cache: None,
        });
        let wn = w / 32;
        let (planes, ob) = match layout {
            Layout::Table => (vec![], {
                let t = super::Table::new();
                let mut v: Vec<u32> = t.chiral.iter().map(|c| c[0] as u32 | (c[1] as u32) << 16).collect();
                v.extend((0..super::NSTATE as u16).map(|s| super::bounce(s) as u32));
                v.extend(boundary.map_or([[0; 2]; super::NCH], |b| b.cdf).iter().flatten());
                v
            }),
            Layout::Bitsliced => {
                let mut planes = vec![0u32; h * PLANES * wn];
                let mut ob = vec![0u32; h * wn];
                for y in 0..h {
                    for x in 0..w {
                        let c = cells[y * w + x];
                        let p = ENCODING.planes(c & !OBST);
                        for k in 0..PLANES {
                            if (p >> k) & 1 == 1 {
                                planes[(y * PLANES + k) * wn + x / 32] |= 1 << (x % 32);
                            }
                        }
                        if c & OBST != 0 {
                            ob[y * wn + x / 32] |= 1 << (x % 32);
                        }
                    }
                }
                (planes, ob)
            }
        };
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, words: usize| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (words * 4) as u64, usage, mapped_at_creation: false });
        let words = if layout == Layout::Table { w * h } else { planes.len() };
        let state = [mk("state0", words), mk("state1", words)];
        let obst = mk("obst", ob.len());
        if layout == Layout::Table {
            // In pieces, so a large lattice has no second full copy on the host.
            for (k, part) in cells.chunks(1 << 20).enumerate() {
                let b: Vec<u8> = part.iter().flat_map(|&c| (c as u32).to_le_bytes()).collect();
                queue.write_buffer(&state[0], (k << 22) as u64, &b);
            }
        } else {
            queue.write_buffer(&state[0], 0, &bytes(&planes));
        }
        queue.write_buffer(&obst, 0, &bytes(&ob));
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor { label: Some("params"), size: SLOT * BATCH as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bgl = pipe.get_bind_group_layout(0);
        let groups = (0..BATCH)
            .map(|j| {
                [0, 1].map(|k| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &uniforms, offset: j as u64 * SLOT, size: wgpu::BufferSize::new(if layout == Layout::Table { 32 } else { 16 }) }),
                            },
                            wgpu::BindGroupEntry { binding: 1, resource: state[k].as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: state[1 - k].as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: obst.as_entire_binding() },
                        ],
                    })
                })
            })
            .collect();
        let coarse_pipe = (layout == Layout::Table).then(|| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ternary coarse"),
                layout: None,
                module: &module,
                entry_point: Some("coarse"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Ok(Gpu { layout, w, h, wn, seed, step: 0, adapter: adapter_name, device, queue, pipe, state, obst, uniforms, groups, cur: 0, boundary, coarse_pipe, coarse: None })
    }

    /// Queue `n` steps (does not wait).
    pub fn run(&mut self, n: usize) {
        let mut left = n;
        while left > 0 {
            let k = left.min(BATCH);
            let mut u = vec![0u32; k * SLOT as usize / 4];
            for j in 0..k {
                let o = j * SLOT as usize / 4;
                u[o] = if self.layout == Layout::Table { self.w as u32 } else { self.wn as u32 };
                u[o + 1] = self.h as u32;
                let st = self.step + j as u32;
                u[o + 2] = step_key(self.seed, st, STREAM_COLLIDE);
                if let (Some(b), true) = (self.boundary, st.is_multiple_of(EVERY)) {
                    u[o + 3] = step_key(self.seed, st, STREAM_BOUNDARY);
                    u[o + 4] = 1;
                    u[o + 5] = b.inlet as u32;
                    u[o + 6] = (self.w - b.outlet) as u32;
                }
            }
            self.queue.write_buffer(&self.uniforms, 0, &bytes(&u));
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipe);
                for j in 0..k {
                    pass.set_bind_group(0, &self.groups[j][self.cur], &[]);
                    let x = if self.layout == Layout::Table { self.w } else { self.wn };
                    pass.dispatch_workgroups(x.div_ceil(64) as u32, self.h as u32, 1);
                    self.cur = 1 - self.cur;
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

    async fn read_buffer(&self, buf: &wgpu::Buffer, size: u64) -> Vec<u32> {
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
        let out = data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data);
        staging.unmap();
        out
    }

    pub async fn read_cells(&self) -> Vec<u16> {
        if self.layout == Layout::Table {
            return self.read_buffer(&self.state[self.cur], (self.w * self.h * 4) as u64).await.into_iter().map(|v| v as u16).collect();
        }
        let words = self.read_buffer(&self.state[self.cur], (self.h * PLANES * self.wn * 4) as u64).await;
        let ob = self.read_buffer(&self.obst, (self.h * self.wn * 4) as u64).await;
        let mut cells = vec![0u16; self.w * self.h];
        for y in 0..self.h {
            for x in 0..self.w {
                let p = (0..PLANES).fold(0u16, |a, k| a | (((words[(y * PLANES + k) * self.wn + x / 32] >> (x % 32)) & 1) as u16) << k);
                let s = ENCODING.state(p).expect("planes hold a count");
                cells[y * self.w + x] = s | if (ob[y * self.wn + x / 32] >> (x % 32)) & 1 == 1 { OBST } else { 0 };
            }
        }
        cells
    }

    /// Add the current block sums into the GPU-side accumulator (queued, no
    /// wait); `Layout::Table` only. Changing the block size resets it.
    pub fn accumulate_sums(&mut self, bw: usize, bh: usize) {
        assert!(self.w.is_multiple_of(bw) && self.h.is_multiple_of(bh));
        if self.coarse.as_ref().is_none_or(|c| (c.bw, c.bh) != (bw, bh)) {
            self.coarse = Some(self.make_coarse(bw, bh));
        }
        let c = self.coarse.as_ref().unwrap();
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(self.coarse_pipe.as_ref().expect("block sums need the table layout"));
            pass.set_bind_group(0, &c.groups[self.cur], &[]);
            pass.dispatch_workgroups(c.nx.div_ceil(64) as u32, c.ny as u32, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Read the accumulated block sums and clear the accumulator.
    pub async fn take_sums(&self) -> BlockSums {
        let c = self.coarse.as_ref().expect("accumulate_sums first");
        let size = (c.nx * c.ny * 16) as u64;
        let v = self.read_buffer(&c.out, size).await;
        self.queue.write_buffer(&c.out, 0, &vec![0u8; size as usize]);
        let col = |k: usize| v.iter().skip(k).step_by(4).map(|&x| x as i32).collect();
        BlockSums { nx: c.nx, ny: c.ny, bw: c.bw, bh: c.bh, jx: col(0), jy: col(1), mass: col(2), fluid: col(3) }
    }

    fn make_coarse(&self, bw: usize, bh: usize) -> Coarse {
        let (nx, ny) = (self.w / bw, self.h / bh);
        let pipe = self.coarse_pipe.as_ref().expect("block sums need the table layout");
        let u = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("coarse params"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.queue.write_buffer(&u, 0, &bytes(&[self.w as u32, self.h as u32, bw as u32, bh as u32, nx as u32, ny as u32, 0, 0]));
        let size = (nx * ny * 16) as u64;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("sums"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.queue.write_buffer(&out, 0, &vec![0u8; size as usize]);
        let groups = [0, 1].map(|k| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.state[k].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
                ],
            })
        });
        Coarse { bw, bh, nx, ny, out, groups }
    }
}
