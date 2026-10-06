//! Shader modules, compute pipelines and bind groups.

use std::borrow::Cow;

pub fn shader(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)) })
}

/// One WGSL source, one entry point, an auto-derived layout, and WGSL
/// `override` constants set at creation (one source can then compile to a
/// different fixed workgroup size per use).
pub fn compute_pipeline(device: &wgpu::Device, label: &str, src: &str, entry: &str, constants: &[(&str, f64)]) -> wgpu::ComputePipeline {
    compute_pipeline_in(device, label, &shader(device, label, src), entry, None, constants)
}

/// A pipeline on an existing module (several entry points share one), with
/// an explicit layout or, for `None`, an auto-derived one.
pub fn compute_pipeline_in(device: &wgpu::Device, label: &str, module: &wgpu::ShaderModule, entry: &str, layout: Option<&wgpu::PipelineLayout>, constants: &[(&str, f64)]) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout,
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
        cache: None,
    })
}

/// A bind group for `group` of a pipeline with an auto-derived layout.
pub fn bind_group(device: &wgpu::Device, pipeline: &wgpu::ComputePipeline, group: u32, entries: &[wgpu::BindGroupEntry]) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(group);
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries })
}
