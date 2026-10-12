//! The conditioning vector fed to the model's `meta` linear layer, pure
//! arithmetic over `solvers::Params`: `[ln dx, ln dt, ln g, ln depth,
//! ln(nu + 1e-6)]`.

use solvers::Params;

pub const META_DIM: usize = 5;

pub fn meta_of(p: &Params) -> [f32; META_DIM] {
    [p.dx.ln(), p.dt.ln(), p.g.ln(), p.depth.ln(), (p.nu + 1e-6).ln()]
}
