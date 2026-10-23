//! GPU engine: the bit-sliced step in WGSL (u32 words), via wgpu (Metal
//! natively, WebGPU in the browser). Bit-identical to the CPU engines.

use crate::bitsliced::PLANES;
use crate::average::Average;
use crate::coarse::BlockSums;
use crate::{step_key, Params, OBST, STREAM_CHIRALITY, STREAM_FORCING};

/// Steps encoded per submit; each has its own 256-byte uniform slot.
const BATCH: usize = 128;
const SLOT: u64 = 256;
const SLOT_USED: u64 = 32;
const NKEYS: usize = crate::boundary::NSEEDS;
/// Boundary keys of a refresh step: 96 u32 in a 512-byte slot.
const KEY_SLOT: u64 = 512;
/// Refresh steps per batch (BATCH / EVERY).
const KEY_SLOTS: usize = BATCH / crate::boundary::EVERY as usize;

pub struct Gpu {
    pub w: usize,
    pub h: usize,
    pub wn: usize,
    pub params: Params,
    pub step: u32,
    pub adapter: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    step_pipe: wgpu::ComputePipeline,
    coarse_pipe: wgpu::ComputePipeline,
    state: [wgpu::Buffer; 2],
    obst: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    groups: Vec<[wgpu::BindGroup; 2]>,
    dispatch_x: u32,
    boundary: Option<BoundaryPass>,
    seeds: [u32; crate::boundary::NSEEDS],
    keys: wgpu::Buffer,
    /// Which state buffer holds the current state.
    cur: usize,
    coarse: Option<Coarse>,
}

fn bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Gpu {
    pub async fn new(w: usize, h: usize, cells: &[u8], params: Params) -> Result<Self, String> {
        Self::build(w, h, params, |f| {
            for y in 0..h {
                f(y, &cells[y * w..][..w]);
            }
        })
        .await
    }

    /// Build from the setup's initial state, one row at a time.
    pub async fn from_setup(s: &crate::Setup) -> Result<Self, String> {
        Self::build(s.w, s.h, s.params, |f| s.initial_rows(f)).await
    }

    /// The device's lattice limits (adapter limits, which `build` requests).
    pub fn size_limits(l: &wgpu::Limits) -> crate::SizeLimits {
        let buf = lean_runtime::limits::max_binding_bytes(l);
        crate::SizeLimits { max_state_buffer: buf, max_total: u64::MAX, max_rows: l.max_compute_workgroups_per_dimension as u64 }
    }

    async fn build(w: usize, h: usize, params: Params, feed: impl FnOnce(&mut dyn FnMut(usize, &[u8]))) -> Result<Self, String> {
        crate::check_shape(w, h)?;
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let limits = adapter.limits();
        Self::size_limits(&limits).check(w, h)?;
        let info = adapter.get_info();
        let adapter_name = if info.name.is_empty() { format!("{:?}", info.backend) } else { format!("{} ({:?})", info.name, info.backend) };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fhp"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("no WebGPU device: {e}"))?;

        let mut step_src = include_str!("shaders/step.wgsl").to_string();
        if !params.walls {
            let (a, b) = (step_src.find("    // WALLS{").unwrap(), step_src.find("    // }WALLS").unwrap());
            step_src.replace_range(a..b, "");
        }
        let src = format!("{}\n{}", include_str!("shaders/collide_gen.wgsl"), step_src);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("fhp"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let step_pipe = pipe("step");
        let coarse_pipe = pipe("coarse");

        let wn = w / 32;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let empty = |label: &str, words: usize| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (words * 4) as u64, usage, mapped_at_creation: false });
        let mk = |label: &str, data: &[u32]| {
            let b = empty(label, data.len());
            queue.write_buffer(&b, 0, &bytes(data));
            b
        };
        let state = [empty("state0", h * PLANES * wn), empty("state1", h * PLANES * wn)];
        let obst = empty("obst", h * wn);
        // Pack and upload in batches of rows, so a large lattice never has a
        // second full copy on the host.
        let batch = (4 << 20) / (PLANES * wn * 4) + 1;
        let (mut planes, mut ob) = (vec![0u32; batch * PLANES * wn], vec![0u32; batch * wn]);
        let mut y0 = 0;
        let flush = |y0: usize, rows: usize, planes: &mut [u32], ob: &mut [u32]| {
            queue.write_buffer(&state[0], (y0 * PLANES * wn * 4) as u64, &bytes(&planes[..rows * PLANES * wn]));
            queue.write_buffer(&obst, (y0 * wn * 4) as u64, &bytes(&ob[..rows * wn]));
            planes.fill(0);
            ob.fill(0);
        };
        feed(&mut |y, row| {
            let r = y - y0;
            for (x, &s) in row.iter().enumerate() {
                for p in 0..PLANES {
                    if s & (1 << p) != 0 {
                        planes[(r * PLANES + p) * wn + x / 32] |= 1 << (x % 32);
                    }
                }
                if s & OBST != 0 {
                    ob[r * wn + x / 32] |= 1 << (x % 32);
                }
            }
            if r + 1 == batch || y + 1 == h {
                flush(y0, r + 1, &mut planes, &mut ob);
                y0 = y + 1;
            }
        });
        let mut bc = vec![0u32; 16];
        if let Some(b) = params.boundary {
            assert!(b.outlet < 1024 && b.outlet > 0 && b.outlet as usize <= wn);
            bc[..7].copy_from_slice(&b.eq);
            bc[7] = b.inlet;
            bc[8] = b.outlet;
            bc[11] = b.kind as u32;
            bc[9] = 1;
        }
        bc[10] = params.walls as u32;
        let bconst = mk("boundary", &bc);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step params"),
            size: SLOT * BATCH as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // One bind group per (uniform slot, ping-pong side).
        let layout = step_pipe.get_bind_group_layout(0);
        let groups = (0..BATCH)
            .map(|j| {
                [0, 1].map(|k| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &uniforms, offset: j as u64 * SLOT, size: wgpu::BufferSize::new(SLOT_USED) }),
                            },
                            wgpu::BindGroupEntry { binding: 1, resource: state[k].as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: state[1 - k].as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: obst.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 4, resource: bconst.as_entire_binding() },
                        ][..if params.walls { 5 } else { 4 }],
                    })
                })
            })
            .collect();
        let keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boundary keys"),
            size: KEY_SLOT * KEY_SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // The boundary refresh: its own pipeline and bind groups, writing the
        // step's output in place.
        let boundary = params.boundary.map(|b| {
            let pipeline = pipe("boundary");
            let layout = pipeline.get_bind_group_layout(0);
            let groups = (0..BATCH)
                .map(|j| {
                    [0, 1].map(|k| {
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &uniforms, offset: j as u64 * SLOT, size: wgpu::BufferSize::new(SLOT_USED) }),
                                },
                                wgpu::BindGroupEntry { binding: 2, resource: state[1 - k].as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 3, resource: obst.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 4, resource: bconst.as_entire_binding() },
                                wgpu::BindGroupEntry {
                                    binding: 5,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &keys,
                                        offset: (j / crate::boundary::EVERY as usize) as u64 * KEY_SLOT,
                                        size: wgpu::BufferSize::new(4 * NKEYS as u64),
                                    }),
                                },
                            ],
                        })
                    })
                })
                .collect();
            BoundaryPass { pipeline, groups, chunks: b.inlet + b.outlet }
        });
        let seeds = crate::boundary::stream_seeds(params.seed);
        let dispatch_x = wn.div_ceil(64) as u32;
        Ok(Gpu { w, h, wn, params, step: 0, adapter: adapter_name, device, queue, step_pipe, coarse_pipe, state, obst, uniforms, groups, dispatch_x, boundary, seeds, keys, cur: 0, coarse: None })
    }

    /// Queue `n` steps (does not wait).
    pub fn run(&mut self, n: usize) {
        let mut left = n;
        while left > 0 {
            let k = left.min(BATCH);
            let mut u = vec![0u32; BATCH * SLOT as usize / 4];
            let mut keys = vec![0u32; KEY_SLOTS * KEY_SLOT as usize / 4];
            for j in 0..k {
                let s = self.step + j as u32;
                let o = j * SLOT as usize / 4;
                u[o] = self.wn as u32;
                u[o + 1] = self.h as u32;
                u[o + 2] = step_key(self.params.seed, s, STREAM_CHIRALITY);
                u[o + 3] = step_key(self.params.seed, s, STREAM_FORCING);
                u[o + 4] = self.params.force_threshold;
                u[o + 5] = s;
                if self.boundary.is_some() && crate::boundary::Boundary::active(s) {
                    keys[j / crate::boundary::EVERY as usize * KEY_SLOT as usize / 4..][..NKEYS].copy_from_slice(&crate::boundary::step_keys(&self.seeds, s));
                }
            }
            if self.boundary.is_some() {
                self.queue.write_buffer(&self.keys, 0, &bytes(&keys));
            }
            self.queue.write_buffer(&self.uniforms, 0, &bytes(&u[..k * SLOT as usize / 4]));
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for j in 0..k {
                    pass.set_pipeline(&self.step_pipe);
                    pass.set_bind_group(0, &self.groups[j][self.cur], &[]);
                    pass.dispatch_workgroups(self.dispatch_x, self.h as u32, 1);
                    if let Some(b) = self.boundary.as_ref().filter(|_| crate::boundary::Boundary::active(self.step + j as u32) ) {
                        pass.set_pipeline(&b.pipeline);
                        pass.set_bind_group(0, &b.groups[j][self.cur], &[]);
                        pass.dispatch_workgroups(b.chunks.div_ceil(64), self.h as u32, 1);
                    }
                    self.cur = 1 - self.cur;
                }
            }
            self.queue.submit(Some(enc.finish()));
            self.step += k as u32;
            left -= k;
        }
    }

    async fn read_buffer(&self, buf: &wgpu::Buffer, size: u64) -> Vec<u32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
        let out = data.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
        drop(data);
        staging.unmap();
        out
    }

    /// Wait for every queued step (native only; the browser awaits a readback).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn finish(&self) {
        self.device.poll(wgpu::PollType::Wait).expect("device poll failed");
    }

    pub async fn read_cells(&self) -> Vec<u8> {
        let words = self.read_buffer(&self.state[self.cur], (self.h * PLANES * self.wn * 4) as u64).await;
        let ob = self.read_buffer(&self.obst, (self.h * self.wn * 4) as u64).await;
        let mut cells = vec![0u8; self.w * self.h];
        for y in 0..self.h {
            for x in 0..self.w {
                let mut s = 0u8;
                for p in 0..PLANES {
                    s |= (((words[(y * PLANES + p) * self.wn + x / 32] >> (x % 32)) & 1) as u8) << p;
                }
                s |= (((ob[y * self.wn + x / 32] >> (x % 32)) & 1) as u8) << 7;
                cells[y * self.w + x] = s;
            }
        }
        cells
    }

    /// Add the current block sums into the GPU-side accumulator (queued,
    /// no wait); `bw` divides 32. Changing the block size resets it.
    pub fn accumulate_sums(&mut self, bw: usize, bh: usize) {
        assert!(32 % bw == 0 && self.w.is_multiple_of(bw) && self.h.is_multiple_of(bh));
        if self.coarse.as_ref().is_none_or(|c| (c.bw, c.bh) != (bw, bh)) {
            self.coarse = Some(self.make_coarse(bw, bh));
        }
        let c = self.coarse.as_ref().unwrap();
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.coarse_pipe);
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

    /// Block sums of the current state.
    pub async fn block_sums(&mut self, bw: usize, bh: usize) -> BlockSums {
        if let Some(c) = self.coarse.as_ref() {
            if (c.bw, c.bh) == (bw, bh) {
                let _ = self.take_sums().await;
            }
        }
        self.accumulate_sums(bw, bh);
        self.take_sums().await
    }

    /// Sample the current block sums and fold them into the running time
    /// average on the GPU (queued, no wait, no read-back). Changing the
    /// block size resets the average.
    pub fn sample_sums(&mut self, bw: usize, bh: usize) {
        assert!(32 % bw == 0 && self.w.is_multiple_of(bw) && self.h.is_multiple_of(bh));
        if self.coarse.as_ref().is_none_or(|c| (c.bw, c.bh) != (bw, bh)) {
            self.coarse = Some(self.make_coarse(bw, bh));
        }
        let c = self.coarse.as_mut().unwrap();
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&c.out, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.coarse_pipe);
            pass.set_bind_group(0, &c.groups[self.cur], &[]);
            pass.dispatch_workgroups(c.nx.div_ceil(64) as u32, c.ny as u32, 1);
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
        let u = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coarse params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&u, 0, &bytes(&[self.wn as u32, self.h as u32, bw as u32, bh as u32, nx as u32, ny as u32, 0, 0]));
        let size = (nx * ny * 16) as u64;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sums"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&out, 0, &vec![0u8; size as usize]);
        let groups = [0, 1].map(|k| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.coarse_pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.state[k].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.obst.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                ],
            })
        });
        let avg = Average::new(&self.device, &out, nx * ny);
        Coarse { bw, bh, nx, ny, out, groups, avg }
    }
}

struct BoundaryPass {
    pipeline: wgpu::ComputePipeline,
    groups: Vec<[wgpu::BindGroup; 2]>,
    chunks: u32,
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
