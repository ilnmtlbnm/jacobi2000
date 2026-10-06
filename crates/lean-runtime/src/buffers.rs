//! Buffer creation and readback. Uploads go through `queue.write_buffer`,
//! never `mapped_at_creation`: on Chromium with Vulkan, buffers filled
//! through the mapping were seen holding different bytes on the device
//! (docs/runs/2026-10-02-webgpu-nondeterminism.md). Readback is always
//! awaited; a synchronous wait deadlocks a browser.

use crate::Error;

/// A buffer of `bytes.len()` bytes (at least 4) filled with `bytes`.
pub fn upload(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (bytes.len() as u64).max(4), usage, mapped_at_creation: false });
    if !bytes.is_empty() {
        queue.write_buffer(&buf, 0, bytes);
    }
    buf
}

/// A zeroed buffer of `size` bytes.
pub fn empty(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
}

/// A uniform buffer holding `data`, updatable with `queue.write_buffer`.
pub fn uniform<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, data: &T) -> wgpu::Buffer {
    upload(device, queue, label, bytemuck::bytes_of(data), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST)
}

/// Records a copy of `size` bytes of `src` from `offset` into a fresh
/// mappable buffer on `encoder`; read it with `map_read` after submitting.
/// Recording the copy with the work it reads saves a submit.
pub fn stage(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, offset: u64, size: u64) -> wgpu::Buffer {
    let staging = empty(device, "readback_staging", size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    encoder.copy_buffer_to_buffer(src, offset, &staging, 0, size);
    staging
}

/// Maps a submitted staging buffer, hands its bytes to `f`, unmaps it.
pub async fn map_read<R>(device: &wgpu::Device, staging: &wgpu::Buffer, f: impl FnOnce(&[u8]) -> R) -> Result<R, Error> {
    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    #[cfg(not(target_arch = "wasm32"))]
    device.poll(wgpu::PollType::Wait).map_err(|e| Error::Poll(e.to_string()))?;
    #[cfg(target_arch = "wasm32")]
    let _ = device;
    rx.await.map_err(|_| Error::Map("map_async callback dropped".into()))?.map_err(|e| Error::Map(e.to_string()))?;
    let data = slice.get_mapped_range();
    let out = f(&data);
    drop(data);
    staging.unmap();
    Ok(out)
}

/// One copy to staging, one submit, one map: `size` bytes of `src` from
/// `offset`, handed to `f`.
pub async fn read<R>(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, offset: u64, size: u64, f: impl FnOnce(&[u8]) -> R) -> Result<R, Error> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    let staging = stage(device, &mut encoder, src, offset, size);
    queue.submit(Some(encoder.finish()));
    map_read(device, &staging, f).await
}
