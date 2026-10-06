//! Adapter and device acquisition. The device is always requested with the
//! adapter's own limits: `wgpu::Limits::default()` caps a storage binding at
//! 128 MiB and the workgroup counts below what most adapters report, and an
//! engine should see what the device can actually do.

use crate::Error;

/// A high-performance adapter with no surface, or `Error::NoAdapter`.
pub async fn request_adapter() -> Result<wgpu::Adapter, Error> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    instance
        .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, force_fallback_adapter: false, compatible_surface: None })
        .await
        .map_err(|e| Error::NoAdapter(e.to_string()))
}

/// A device and queue on `adapter` with the adapter's limits and exactly
/// `features`, which must be a subset of `adapter.features()` (decide
/// optional features from that set before calling).
pub async fn request_device(adapter: &wgpu::Adapter, label: &str, features: wgpu::Features) -> Result<(wgpu::Device, wgpu::Queue), Error> {
    if !adapter.features().contains(features) {
        return Err(Error::NoDevice(format!("adapter lacks {:?}", features - adapter.features())));
    }
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: features,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| Error::NoDevice(e.to_string()))
}

/// "name (backend)", or the backend alone when the adapter has no name.
pub fn adapter_label(info: &wgpu::AdapterInfo) -> String {
    if info.name.is_empty() {
        format!("{:?}", info.backend)
    } else {
        format!("{} ({:?})", info.name, info.backend)
    }
}

/// JSON: adapter name and backend, adapter and device features, and the
/// device limits kernel selection or buffer chunking usually depends on.
pub fn adapter_report(adapter: &wgpu::Adapter, device: &wgpu::Device) -> String {
    let info = adapter.get_info();
    let l = device.limits();
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "{{\"name\":\"{}\",\"vendor\":{},\"device\":{},\"backend\":\"{:?}\",\"driver\":\"{}\",\"driverInfo\":\"{}\",\"adapterFeatures\":\"{}\",\"deviceFeatures\":\"{}\",\
\"maxStorageBufferBindingSize\":{},\"maxBufferSize\":{},\"maxStorageBuffersPerShaderStage\":{},\"maxComputeWorkgroupStorageSize\":{},\
\"maxComputeInvocationsPerWorkgroup\":{},\"maxComputeWorkgroupSizeX\":{},\"minStorageBufferOffsetAlignment\":{},\"minUniformBufferOffsetAlignment\":{},\
\"minSubgroupSize\":{},\"maxSubgroupSize\":{}}}",
        esc(&info.name),
        info.vendor,
        info.device,
        info.backend,
        esc(&info.driver),
        esc(&info.driver_info),
        esc(&format!("{:?}", adapter.features())),
        esc(&format!("{:?}", device.features())),
        l.max_storage_buffer_binding_size,
        l.max_buffer_size,
        l.max_storage_buffers_per_shader_stage,
        l.max_compute_workgroup_storage_size,
        l.max_compute_invocations_per_workgroup,
        l.max_compute_workgroup_size_x,
        l.min_storage_buffer_offset_alignment,
        l.min_uniform_buffer_offset_alignment,
        l.min_subgroup_size,
        l.max_subgroup_size,
    )
}
