//! The grid the model reasons on.
//!
//! Halo width and edge slots are carried from the start even though tonight's
//! model uses neither: the halo is what a stencil mask on a tile will read,
//! and the edge slots are where a flux-form head will write (new value = old
//! + in - out), so both have to exist in the type before either is a rewrite.

use burn::constant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    /// Fields per cell: h, u, v.
    pub channels: usize,
    /// Patch side, in cells.
    pub patch: usize,
    /// Ghost cells around the tile, read by a stencil, never predicted.
    pub halo: usize,
}

impl Default for Grid {
    fn default() -> Self {
        Grid {
            nx: 64,
            ny: 64,
            channels: 3,
            patch: 4,
            halo: 0,
        }
    }
}

impl Grid {
    pub fn cells(&self) -> usize {
        self.nx * self.ny
    }

    pub fn tokens_x(&self) -> usize {
        self.nx / self.patch
    }

    pub fn tokens_y(&self) -> usize {
        self.ny / self.patch
    }

    pub fn tokens(&self) -> usize {
        self.tokens_x() * self.tokens_y()
    }

    /// Floats per token: one patch of every field.
    pub fn patch_dim(&self) -> usize {
        self.channels * self.patch * self.patch
    }

    /// Slots on vertical cell faces, for a future flux head.
    pub fn edges_x(&self) -> usize {
        (self.nx + 1) * self.ny
    }

    /// Slots on horizontal cell faces.
    pub fn edges_y(&self) -> usize {
        self.nx * (self.ny + 1)
    }

    /// Grid extent including the halo on both sides.
    pub fn padded(&self) -> (usize, usize) {
        (self.nx + 2 * self.halo, self.ny + 2 * self.halo)
    }

    /// Token index -> token-grid coordinates.
    pub fn token_xy(&self, t: usize) -> (usize, usize) {
        (t % self.tokens_x(), t / self.tokens_x())
    }
}

impl core::fmt::Display for Grid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Grid {{ {}x{}, {} channels, patch {}, halo {} }}",
            self.nx, self.ny, self.channels, self.patch, self.halo
        )
    }
}

// Carried inside the model as a constant: shape metadata, never a parameter.
constant!(Grid);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_grid_is_the_64x64_case() {
        let g = Grid::default();
        assert_eq!(g.tokens(), 256);
        assert_eq!(g.patch_dim(), 48);
        assert_eq!(g.token_xy(17), (1, 1));
        assert_eq!(g.padded(), (64, 64));
        assert_eq!(Grid { halo: 2, ..g }.padded(), (68, 68));
    }
}
