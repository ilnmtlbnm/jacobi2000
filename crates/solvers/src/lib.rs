//! Reference solvers. Pure Rust, no Burn, no allocation in the hot loop.

pub mod shallow_water;
pub mod sph;

pub use shallow_water::{Params, ShallowWater, State};
pub use sph::{Initial, Sph};
