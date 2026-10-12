//! The model backend both binaries run: CPU, or WebGPU when built with
//! `--features wgpu` and an adapter is granted. `auto` takes WebGPU when it is
//! available and the CPU otherwise (a capability check, never a benchmark).

use engine::Model;

#[allow(clippy::large_enum_variant)]
pub enum Engine {
    Cpu(Model),
    #[cfg(feature = "wgpu")]
    Gpu(Box<engine::gpu::GpuModel>),
}

impl Engine {
    pub fn step(&self, state: &[f32], meta: &[f32]) -> Vec<f32> {
        match self {
            Engine::Cpu(m) => m.step(state, meta),
            #[cfg(feature = "wgpu")]
            Engine::Gpu(m) => pollster::block_on(m.step(state, meta)),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Engine::Cpu(_) => "cpu",
            #[cfg(feature = "wgpu")]
            Engine::Gpu(_) => "webgpu",
        }
    }
}

pub fn pick_engine(model: Model, want: &str) -> Engine {
    match want {
        "cpu" => Engine::Cpu(model),
        "webgpu" => {
            #[cfg(feature = "wgpu")]
            {
                match pollster::block_on(model.to_gpu()) {
                    Ok(g) => Engine::Gpu(Box::new(g)),
                    Err(e) => {
                        eprintln!("--backend webgpu requested but no adapter was granted: {e:?}");
                        std::process::exit(1);
                    }
                }
            }
            #[cfg(not(feature = "wgpu"))]
            {
                eprintln!("--backend webgpu requires building with --features wgpu");
                std::process::exit(2);
            }
        }
        "auto" => {
            #[cfg(feature = "wgpu")]
            {
                if let Ok(g) = pollster::block_on(model.to_gpu()) {
                    return Engine::Gpu(Box::new(g));
                }
            }
            Engine::Cpu(model)
        }
        other => {
            eprintln!("unrecognized --backend `{other}` (want auto | cpu | webgpu)");
            std::process::exit(2);
        }
    }
}

