//! The page's running time average on the GPU (`shaders/average.wgsl`):
//! folded after each sample of an engine's block sums with no read-back,
//! and read back on request, one read at a time, without blocking.

#![cfg(feature = "gpu")]

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use futures_channel::oneshot;

use crate::coarse::AVG_A;

const PENDING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

struct Read {
    buf: wgpu::Buffer,
    /// Samples folded when the read was queued.
    samples: u32,
    state: Arc<AtomicU8>,
    /// Woken when the map completes.
    wakers: Arc<Mutex<Vec<oneshot::Sender<()>>>>,
}

pub struct Average {
    n: usize,
    q: wgpu::Buffer,
    params: wgpu::Buffer,
    pipe: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    samples: u32,
    read: Option<Read>,
}

impl Average {
    /// The average of `n` blocks of the block sums in `sums` (4 words per
    /// block, as the engines' `coarse` kernels write them).
    pub fn new(device: &wgpu::Device, sums: &wgpu::Buffer, n: usize) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("average"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/average.wgsl").into()) });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("average"), layout: None, module: &module, entry_point: Some("fold"), compilation_options: Default::default(), cache: None });
        let q = device.create_buffer(&wgpu::BufferDescriptor { label: Some("average"), size: (n * 16) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("average params"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: q.as_entire_binding() },
            ],
        });
        Average { n, q, params, pipe, group, samples: 0, read: None }
    }

    /// Record the fold of the sums into `enc` (submitted by the caller);
    /// the first and second folds set the parameters.
    pub fn fold(&mut self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder) {
        if self.samples < 2 {
            let w = [self.n as u32, (self.samples == 0) as u32, AVG_A as u32, 0];
            queue.write_buffer(&self.params, 0, &w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        }
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipe);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.n.div_ceil(64) as u32, 1, 1);
        self.samples += 1;
    }

    /// Samples folded so far.
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Queue a read-back of the average unless one is in flight.
    pub fn request(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.read.is_some() {
            return;
        }
        let size = (self.n * 16) as u64;
        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("average read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.q, 0, &buf, 0, size);
        queue.submit(Some(enc.finish()));
        let state = Arc::new(AtomicU8::new(PENDING));
        let wakers: Arc<Mutex<Vec<oneshot::Sender<()>>>> = Arc::default();
        let (st, wk) = (state.clone(), wakers.clone());
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            st.store(if r.is_ok() { MAPPED } else { FAILED }, Ordering::Release);
            for tx in wk.lock().unwrap().drain(..) {
                let _ = tx.send(());
            }
        });
        self.read = Some(Read { buf, samples: self.samples, state, wakers });
    }

    /// Whether a read-back is in flight.
    pub fn reading(&self) -> bool {
        self.read.is_some()
    }

    /// The read-back average and its sample count, if it has arrived (does
    /// not wait).
    pub fn ready(&mut self, device: &wgpu::Device) -> Option<(Vec<i32>, u32)> {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = device.poll(wgpu::PollType::Poll);
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        match self.read.as_ref()?.state.load(Ordering::Acquire) {
            PENDING => return None,
            FAILED => panic!("buffer map failed"),
            _ => {}
        }
        let r = self.read.take().unwrap();
        let q = {
            let data = r.buf.slice(..).get_mapped_range();
            data.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect()
        };
        r.buf.unmap();
        Some((q, r.samples))
    }

    /// A receiver completed when the read-back in flight lands (None if
    /// there is none or it has landed). It borrows nothing, so the caller
    /// can keep stepping while it waits.
    pub fn waiter(&self) -> Option<oneshot::Receiver<()>> {
        let r = self.read.as_ref()?;
        let mut w = r.wakers.lock().unwrap();
        if r.state.load(Ordering::Acquire) != PENDING {
            return None;
        }
        let (tx, rx) = oneshot::channel();
        w.push(tx);
        Some(rx)
    }

    /// Wait for the read-back in flight (at once if none).
    pub async fn wait(&mut self, device: &wgpu::Device) {
        let Some(rx) = self.waiter() else { return };
        #[cfg(not(target_arch = "wasm32"))]
        device.poll(wgpu::PollType::Wait).expect("device poll failed");
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        let _ = rx.await;
    }
}
