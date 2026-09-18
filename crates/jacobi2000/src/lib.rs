//! The model: grid type, pluggable attention masks, patch transformer.

pub mod grid;
pub mod mask;
pub mod model;
pub mod project;

pub use grid::Grid;
pub use mask::Mask;
pub use model::{Jacobi2000, Jacobi2000Config};
pub use project::{energy, mass, project_energy, project_mass, Denorm};
