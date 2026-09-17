//! Reference solvers. Pure Rust, no Burn, no allocation in the hot loop.

pub mod shallow_water;

pub use shallow_water::{Params, ShallowWater, State};
