//! Data stack, training objectives, rollout evaluation, checkpoint I/O.

pub mod checkpoint;
pub mod data;
pub mod eval;
pub mod fit;

pub use checkpoint::{Checkpoint, Meta};
pub use data::{Batch, Generator, GeneratorConfig, Normalizer, META_DIM};
pub use eval::{rollout_errors, RolloutErrors};
pub use fit::{fit, FitConfig};
