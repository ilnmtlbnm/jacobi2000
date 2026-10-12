//! The three small Game of Life models, run over a whole board:
//!
//! * `bert`: BERT of Life, one post-LN transformer encoder layer over the 9
//!   neighbourhood cells as tokens (token embedding, learned positions),
//!   read out at the centre token; 2 logits per cell.
//! * `mlp2`: the 9-number MLP, `Linear ReLU Linear ReLU Linear`; 2 logits.
//! * `stencil`: per-cell embedding, one pre-LN block whose attention reads
//!   each cell's 9 taps (itself and its 8 neighbours) on a toroidal grid,
//!   and a per-cell head; 1 logit. It has no position embedding, so a model
//!   trained at 16x16 runs on a board of any size and shape.
//!
//! Weights are a PyTorch `state_dict` in safetensors plus a `config.json`
//! (see [`model`]). A board is `width * height` bytes, row-major, nonzero
//! meaning alive; the grid is a torus. For `bert` and `mlp2` the logits are
//! `[dead, alive]` and a cell lives if `alive > dead`; for `stencil` a cell
//! lives if its logit is above 0.
//!
//! [`Model`] is the CPU path (rayon under the `threads` feature, one thread
//! without it); [`gpu::GpuModel`] (feature `wgpu`) runs the same math as
//! WGSL kernels, one dispatch per generation over all cells, with no readback
//! between generations. Which one to use is the caller's call from
//! `lean_runtime::backend::ladder`, by capability.

pub mod cpu;
#[cfg(feature = "wgpu")]
pub mod gpu;
pub mod model;
mod st;

use std::fmt;

pub use model::{Config, Kind, Params};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Config(String),
    Weights(String),
    /// A board whose cell count does not match `width * height`, or is empty.
    Board(String),
    Runtime(lean_runtime::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(e) => write!(f, "config.json: {e}"),
            Error::Weights(e) => write!(f, "weights: {e}"),
            Error::Board(e) => write!(f, "board: {e}"),
            Error::Runtime(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<lean_runtime::Error> for Error {
    fn from(e: lean_runtime::Error) -> Self {
        Error::Runtime(e)
    }
}

pub(crate) fn check_board(cells: &[u8], width: usize, height: usize) -> Result<(), Error> {
    if width == 0 || height == 0 {
        return Err(Error::Board(format!("{width}x{height} has no cells")));
    }
    if cells.len() != width * height {
        return Err(Error::Board(format!("{width}x{height} needs {} cells, got {}", width * height, cells.len())));
    }
    Ok(())
}

/// Whether a logit row means alive: `alive > dead` for two logits, above 0
/// for one.
pub fn is_alive(kind: Kind, logits: &[f32]) -> bool {
    match kind {
        Kind::Bert | Kind::Mlp2 => logits[1] > logits[0],
        Kind::Stencil => logits[0] > 0.0,
    }
}

pub struct Model {
    pub cfg: Config,
    pub params: Params,
}

impl Model {
    /// From the bytes of `config.json` and `model.safetensors`.
    pub fn from_bytes(safetensors: &[u8], config_json: &str) -> Result<Model, Error> {
        let cfg = Config::from_json(config_json)?;
        let params = Params::from_safetensors(&cfg, safetensors)?;
        Ok(Model { cfg, params })
    }

    pub fn kind(&self) -> Kind {
        self.cfg.kind
    }

    /// Logits for every cell: `[width * height * 2]` for `bert` and `mlp2`,
    /// `[width * height]` for `stencil`.
    pub fn logits(&self, cells: &[u8], width: usize, height: usize) -> Result<Vec<f32>, Error> {
        check_board(cells, width, height)?;
        Ok(cpu::logits(&self.cfg, &self.params, cells, width, height))
    }

    /// The next board (0 or 1 per cell).
    pub fn step(&self, cells: &[u8], width: usize, height: usize) -> Result<Vec<u8>, Error> {
        let lg = self.logits(cells, width, height)?;
        let k = self.cfg.kind.outputs();
        Ok(lg.chunks_exact(k).map(|l| is_alive(self.cfg.kind, l) as u8).collect())
    }

    /// The WebGPU twin of this model: same weights, same config.
    #[cfg(feature = "wgpu")]
    pub async fn to_gpu(&self) -> Result<gpu::GpuModel, Error> {
        gpu::GpuModel::new(self).await
    }
}
