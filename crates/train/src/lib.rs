//! Data stack, training objectives, rollout evaluation, checkpoint I/O.

pub mod checkpoint;
pub mod data;
pub mod eval;
pub mod fit;
pub mod lagrangian;
pub mod metrics;
pub mod spectral;
pub mod xr;

pub use checkpoint::{load, save, Checkpoint, Meta};
pub use data::{Batch, Generator, GeneratorConfig, Normalizer, META_DIM};
pub use lagrangian::{LagrangianConfig, LagrangianGenerator};
pub use eval::{rollout_errors, RolloutErrors};
pub use fit::{fit, FitConfig};
pub use metrics::Metrics;
pub use spectral::band_mse;
pub use xr::{score_rollout_xr, xr_step};
