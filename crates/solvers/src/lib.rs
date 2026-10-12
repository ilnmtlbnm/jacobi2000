//! Reference solvers. Pure Rust, no allocation in the hot loop.

pub mod shallow_water;
pub mod sph;
pub mod splat;

pub use shallow_water::{Params, ShallowWater, State};
pub use sph::{Initial, Sph};
pub use splat::{splat as splat_particles, SplatField};
