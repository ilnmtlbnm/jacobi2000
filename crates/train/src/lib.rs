//! Data stack: on-demand shallow-water trajectories, normalization stats.

pub mod data;

pub use data::{Batch, Generator, GeneratorConfig, Normalizer, META_DIM};
