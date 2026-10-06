//! The wgpu and wasm plumbing an engine needs before its own kernels:
//! the backend ladder, device acquisition with the adapter's limits,
//! buffers and readback, pipelines, a clock, and the wasm threads pool.
//! No model, lattice or kernel types live here. See docs/lean-runtime.md.

pub mod backend;
pub mod buffers;
pub mod clock;
pub mod error;
pub mod gpu;
pub mod limits;
pub mod pipeline;
#[cfg(all(target_arch = "wasm32", feature = "wasm-mt"))]
pub mod threads;

pub use backend::{ladder, Backend, Capabilities};
pub use error::Error;
