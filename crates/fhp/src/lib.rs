//! FHP-III lattice gas (Frisch, Hasslacher, Pomeau, PRL 56, 1505 (1986);
//! collision set of d'Humieres and Lallemand, Complex Systems 1, 599 (1987)).
//!
//! A hexagonal lattice stored on a W x H square grid, odd rows shifted half a
//! site to the right. Each site holds one bit per moving direction (0..6), a
//! rest bit (6) and an obstacle bit (7). Every path runs the same step,
//!
//!   state <- force(collide(stream(state), chirality(step)), forcing(step))
//!
//! so the stored state is always post-collision, and the random bits are
//! functions of (seed, step, row, 32-site chunk) only. That makes the per-site
//! reference, the bit-sliced CPU engine and the GPU engine bit-identical.

#![allow(clippy::needless_range_loop)]

pub mod bitsliced;
pub mod boundary;
#[cfg(feature = "gpu")]
pub mod average;
#[cfg(feature = "gpu")]
pub mod health;
pub mod coarse;
pub mod page;
pub mod collide_gen;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod reference;
pub mod rules;
pub mod signed;
pub mod synth;
pub mod ternary;
#[cfg(feature = "web-cpu")]
pub mod web;

#[cfg(feature = "wasm-mt")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Direction bits 0..6: E, NE, NW, W, SW, SE (counter-clockwise from east).
pub const NDIR: usize = 6;
pub const REST: u8 = 1 << 6;
pub const OBST: u8 = 1 << 7;
pub const E: usize = 0;
pub const W: usize = 3;

/// Velocities in integer units (2 c_x, 2 c_y / sqrt 3).
pub const CX2: [i32; 6] = [2, 1, -1, -2, -1, 1];
pub const CY2: [i32; 6] = [0, 1, 1, 0, -1, -1];

/// Row step of each direction.
pub const DY: [i32; 6] = [0, 1, 1, 0, -1, -1];
/// Column step of each direction leaving an even row, then an odd row.
pub const DX: [[i32; 6]; 2] = [[1, 0, -1, -1, -1, 0], [1, 1, 0, -1, 0, 1]];

/// The site a particle at (x, y) moving in `dir` reaches, periodic in both
/// directions (H must be even so the row parity is periodic too).
pub fn neighbor(x: usize, y: usize, dir: usize, w: usize, h: usize) -> (usize, usize) {
    let dx = DX[y & 1][dir];
    let nx = (x as i64 + dx as i64).rem_euclid(w as i64) as usize;
    let ny = (y as i64 + DY[dir] as i64).rem_euclid(h as i64) as usize;
    (nx, ny)
}

/// The site whose particle moving in `dir` arrives at (x, y).
pub fn source(x: usize, y: usize, dir: usize, w: usize, h: usize) -> (usize, usize) {
    neighbor(x, y, (dir + 3) % 6, w, h)
}

/// lowbias32 (Chris Wellons' hash-prospector), the same in WGSL.
#[inline(always)]
pub fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

pub const STREAM_CHIRALITY: u32 = 0;
pub const STREAM_FORCING: u32 = 1;

/// Seed of a random stream, constant over steps.
pub fn stream_seed(seed: u32, stream: u32) -> u32 {
    lowbias32(seed ^ lowbias32(stream.wrapping_add(0x9e37_79b9)))
}

/// Per-step key of a random stream.
pub fn step_key(seed: u32, step: u32, stream: u32) -> u32 {
    lowbias32(stream_seed(seed, stream) ^ step)
}

/// 32 random bits for the 32-site chunk `idx = y * (W / 32) + x / 32`.
#[inline(always)]
pub fn chunk_rand(key: u32, idx: u32) -> u32 {
    lowbias32(key ^ idx.wrapping_mul(0x9e37_79b9))
}

/// Forcing mask of a chunk: at most one site per chunk per step, the site
/// `h & 31` when `h >> 5 < threshold`, so each site is forced with
/// probability threshold / 2^32.
#[inline(always)]
pub fn force_mask(key: u32, idx: u32, threshold: u32) -> u32 {
    let h = chunk_rand(key, idx);
    if (h >> 5) < threshold {
        1 << (h & 31)
    } else {
        0
    }
}

/// Parameters every engine needs for a step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Params {
    pub seed: u32,
    /// Forcing probability per site per step is force_threshold / 2^32
    /// (at most 2^27, i.e. 1/32).
    pub force_threshold: u32,
    /// Open boundaries in x (inlet and outlet); None = periodic.
    pub boundary: Option<boundary::Boundary>,
    /// Free-slip walls on rows 0 and H-1; otherwise y is periodic.
    pub walls: bool,
}

impl Params {
    pub fn force_probability(&self) -> f64 {
        self.force_threshold as f64 / 4294967296.0
    }
    pub fn with_force_probability(seed: u32, p: f64) -> Self {
        let t = (p * 4294967296.0).round().clamp(0.0, (1u64 << 27) as f64) as u32;
        Params { seed, force_threshold: t, ..Default::default() }
    }
}

/// Obstacle shapes, in lattice coordinates (x, y) with the hexagonal row
/// spacing sqrt(3)/2 taken into account for the cylinder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Obstacle {
    None,
    /// Vertical plate: centre column, centre row, length in rows, thickness in columns.
    Plate { x: usize, y: usize, rows: usize, cols: usize },
    /// Cylinder: centre and diameter in lattice units.
    Cylinder { x: f64, y: f64, d: f64 },
}

impl Obstacle {
    pub fn contains(&self, x: usize, y: usize) -> bool {
        match *self {
            Obstacle::None => false,
            Obstacle::Plate { x: cx, y: cy, rows, cols } => {
                x >= cx && x < cx + cols && y + rows / 2 >= cy && y < cy + rows - rows / 2
            }
            Obstacle::Cylinder { x: cx, y: cy, d } => {
                let px = x as f64 + if y & 1 == 1 { 0.5 } else { 0.0 };
                let py = y as f64 * 3f64.sqrt() / 2.0;
                let (qx, qy) = (cx, cy * 3f64.sqrt() / 2.0);
                (px - qx).powi(2) + (py - qy).powi(2) <= (d / 2.0).powi(2)
            }
        }
    }
}

/// Initial state, one row at a time: fluid sites sampled from the
/// Fermi-Dirac equilibrium (d'Humieres and Lallemand eq. 2.2) at density `d`
/// per cell and velocity `ux` along x, obstacle sites empty with bit 7.
/// Shared by every engine, so it only has to be deterministic, not fast;
/// row by row so a large lattice never holds a byte per site.
pub fn initial_rows(w: usize, h: usize, d: f64, ux: f64, obstacle: &Obstacle, seed: u32, mut f: impl FnMut(usize, &[u8])) {
    check_shape(w, h).expect("lattice shape");
    let probs = boundary::equilibrium(d, ux);
    let mut row = vec![0u8; w];
    let mut ctr: u32 = 0;
    let base = lowbias32(seed ^ 0x5bd1_e995);
    for y in 0..h {
        for (x, out) in row.iter_mut().enumerate() {
            if obstacle.contains(x, y) {
                *out = OBST;
                continue;
            }
            let mut s = 0u8;
            for (b, &p) in probs.iter().enumerate() {
                ctr = ctr.wrapping_add(1);
                let r = lowbias32(base ^ lowbias32(ctr)) as f64 / 4294967296.0;
                if r < p {
                    s |= 1 << b;
                }
            }
            *out = s;
        }
        f(y, &row);
    }
}

/// Initial state as one byte per site (tests, the reference engine).
pub fn initial_state(w: usize, h: usize, d: f64, ux: f64, obstacle: &Obstacle, seed: u32) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    initial_rows(w, h, d, ux, obstacle, seed, |y, row| out[y * w..][..w].copy_from_slice(row));
    out
}

/// Coarse-graining block of the page and the default of `fhp run`.
pub const BLOCK: usize = 16;
/// Width granularity: 64-site CPU words (also 32-site GPU words, 16-site blocks).
pub const W_STEP: usize = 64;
/// Height granularity: even rows (row parity is periodic) and 16-row blocks.
pub const H_STEP: usize = 16;
/// Bytes per site the bit-sliced engines hold: two copies of 7 planes and
/// the obstacle plane.
pub const BYTES_PER_SITE: f64 = 15.0 / 8.0;

/// The shape every engine accepts: W a multiple of 64, H even, at least 2 rows.
pub fn check_shape(w: usize, h: usize) -> Result<(), String> {
    if w == 0 || !w.is_multiple_of(64) || h < 2 || !h.is_multiple_of(2) {
        return Err(format!("{w}x{h}: W must be a positive multiple of 64 and H even"));
    }
    Ok(())
}

/// Coarse-graining blocks the page offers.
pub const BLOCKS: [usize; 3] = [8, 16, 32];

/// Round a requested lattice up to the nearest size every engine and the
/// page's `block` x `block` coarse-graining blocks accept: W a multiple of
/// 64, H a multiple of 16 and of `block`, at least 64 x 16. `block` is one
/// of `BLOCKS` (all divide 64).
pub fn round_size(w: usize, h: usize, block: usize) -> (usize, usize) {
    let hs = H_STEP.max(block);
    (w.max(1).div_ceil(W_STEP) * W_STEP, h.max(1).div_ceil(hs) * hs)
}

/// What a device allows: the largest state buffer (7 bit-planes), the total
/// bytes of all lattice buffers, and the most rows (GPU dispatch limit).
#[derive(Clone, Copy, Debug)]
pub struct SizeLimits {
    pub max_state_buffer: u64,
    pub max_total: u64,
    pub max_rows: u64,
}

impl SizeLimits {
    /// CPU paths: the bytes the bit-sliced engine holds, under `max_total`.
    pub fn cpu(max_total: u64) -> Self {
        SizeLimits { max_state_buffer: max_total, max_total, max_rows: u64::MAX }
    }

    fn fits(&self, w: usize, h: usize, state: f64, total: f64) -> bool {
        let sites = w as f64 * h as f64;
        sites * state <= self.max_state_buffer as f64 && sites * total <= self.max_total as f64 && h as u64 <= self.max_rows
    }

    /// Ok if a (rounded) FHP-III lattice fits, else a message naming the
    /// largest lattice of the same aspect ratio that does.
    pub fn check(&self, w: usize, h: usize) -> Result<(), String> {
        self.check_bytes(w, h, 7.0 / 8.0, BYTES_PER_SITE)
    }

    /// `check` for an engine whose largest buffer holds `state` bytes per
    /// site and whose buffers together hold `total` bytes per site.
    pub fn check_bytes(&self, w: usize, h: usize, state: f64, total: f64) -> Result<(), String> {
        if self.fits(w, h, state, total) {
            return Ok(());
        }
        let mut k = 1.0f64;
        let (mut bw, mut bh) = (0, 0);
        while k > 1e-4 {
            k *= 0.98;
            let cw = ((w as f64 * k) as usize / W_STEP) * W_STEP;
            let ch = ((h as f64 * k) as usize / H_STEP) * H_STEP;
            if cw >= W_STEP && ch >= H_STEP && self.fits(cw, ch, state, total) {
                (bw, bh) = (cw, ch);
                break;
            }
        }
        let sites = w as f64 * h as f64;
        Err(format!(
            "{w}x{h} is too large for this device (state buffer {} MB, allowed {} MB; total {} MB, allowed {} MB; rows {h}, allowed {}). Largest at this aspect ratio: {bw}x{bh}",
            (sites * state) as u64 >> 20,
            self.max_state_buffer >> 20,
            (sites * total) as u64 >> 20,
            self.max_total >> 20,
            self.max_rows
        ))
    }
}

/// The outlet of `Setup::street`.
pub const DEFAULT_OUTLET: boundary::Outlet = boundary::Outlet::Randomize;

/// Memory the CPU paths may use in the browser (wasm32 has 4 GiB at most).
pub const WASM_CPU_MAX_BYTES: u64 = 2 << 30;

/// The page's setup for a requested size and inflow speed: the size rounded
/// up to what every engine and `block` accept (`round_size`), then checked
/// against `limits`. Never panics; errors are messages for the status line.
/// `mode` is an outlet (`sponge`, `randomize`, `delete`) or `periodic` (the
/// first, forced setup, which ignores `u`).
pub fn street_for(w: usize, h: usize, u: f64, mode: &str, block: usize, limits: &SizeLimits) -> Result<Setup, String> {
    if !(u > 0.0 && u <= 0.5) {
        return Err(format!("inflow speed {u} is outside (0, 0.5]"));
    }
    if !BLOCKS.contains(&block) {
        return Err(format!("block {block}: 8, 16 or 32"));
    }
    let (w, h) = round_size(w, h, block);
    limits.check(w, h)?;
    if mode == "periodic" {
        return Ok(Setup::street_periodic(w, h));
    }
    let kind = boundary::Outlet::parse(mode).ok_or_else(|| format!("unknown mode {mode}: periodic, sponge, randomize or delete"))?;
    Ok(Setup::street_with(w, h, u, kind))
}

/// Simulation setup shared by the CLI, the tests and the web page.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Setup {
    pub w: usize,
    pub h: usize,
    /// Occupation per cell.
    pub density: f64,
    /// Initial velocity along x.
    pub ux: f64,
    pub obstacle: Obstacle,
    pub params: Params,
}

impl Setup {
    /// Flow past a plate (d'Humieres and Lallemand 1987, section 6.3, scaled
    /// down): plate across H/6 rows at W/5, slightly off the centre line.
    pub fn plate(w: usize, h: usize) -> Self {
        let rows = (h / 6).max(4);
        Setup {
            w,
            h,
            density: 0.25,
            ux: 0.3,
            obstacle: Obstacle::Plate { x: w / 5, y: h / 2 + 1, rows, cols: 2 },
            params: Params::with_force_probability(1, 0.0),
        }
    }

    /// The vortex-street setup of the dev page and `fhp run`: density 0.28
    /// per cell, a plate across H/8 rows at W/6, one row off the centre
    /// line, open boundaries in x with inflow speed `u` (an inlet of 32
    /// columns and the default outlet), y periodic.
    pub fn street(w: usize, h: usize, u: f64) -> Self {
        Self::street_with(w, h, u, DEFAULT_OUTLET)
    }

    /// The street with a chosen outlet: `Sponge` over the last W/16 columns,
    /// `Randomize` or `Delete` over the last 64.
    pub fn street_with(w: usize, h: usize, u: f64, kind: boundary::Outlet) -> Self {
        let rows = h / 8;
        let w32 = (w / 32) as u32;
        let density = 0.28;
        let outlet = match kind {
            boundary::Outlet::Sponge => (w32 / 16).max(2),
            _ => 2,
        };
        Setup {
            w,
            h,
            density,
            ux: u,
            obstacle: Obstacle::Plate { x: w / 6, y: h / 2 + 1, rows, cols: 2 },
            params: Params { seed: 1, boundary: Some(boundary::Boundary::new(density, u, 1, outlet, kind)), ..Default::default() },
        }
    }

    /// The first, fully periodic setup: the flow held near 0.4 by forcing.
    /// The forcing probability balances the plate's drag (measured on
    /// 4096 x 1024: p = 1.1e-4 holds U = 0.40) and is scaled by plate size
    /// over lattice area for other sizes.
    pub fn street_periodic(w: usize, h: usize) -> Self {
        let rows = h / 8;
        let p = 1.1e-4 * (rows as f64 / 128.0) * (4096.0 * 1024.0) / (w * h) as f64;
        Setup {
            w,
            h,
            density: 0.28,
            ux: 0.35,
            obstacle: Obstacle::Plate { x: w / 6, y: h / 2 + 1, rows, cols: 2 },
            params: Params::with_force_probability(1, p),
        }
    }

    /// Mach number of a speed, c_s = sqrt(3/7) for the seven-bit models.
    pub fn mach(u: f64) -> f64 {
        u / (3.0f64 / 7.0).sqrt()
    }

    pub fn plate_rows(&self) -> usize {
        match self.obstacle {
            Obstacle::Plate { rows, .. } => rows,
            Obstacle::Cylinder { d, .. } => (d * 2.0 / 3f64.sqrt()) as usize,
            Obstacle::None => 0,
        }
    }

    /// Reynolds number U D g / nu for a mean speed U, D the obstacle's size across.
    pub fn reynolds(&self, u: f64) -> f64 {
        u * self.plate_rows() as f64 * 3f64.sqrt() / 2.0 * self.g() / self.viscosity()
    }

    pub fn initial_rows(&self, f: impl FnMut(usize, &[u8])) {
        initial_rows(self.w, self.h, self.density, self.ux, &self.obstacle, self.params.seed, f)
    }

    pub fn initial_state(&self) -> Vec<u8> {
        initial_state(self.w, self.h, self.density, self.ux, &self.obstacle, self.params.seed)
    }

    /// FHP-III kinematic shear viscosity at this density
    /// (d'Humieres and Lallemand 1987, eq. 2.13).
    pub fn viscosity(&self) -> f64 {
        let d = self.density;
        let q = d * (1.0 - d);
        1.0 / (28.0 * q * (1.0 - 8.0 * q / 7.0)) - 0.125
    }

    /// Galilean factor g(rho) of the seven-bit models (eq. 2.11).
    pub fn g(&self) -> f64 {
        let d = self.density;
        7.0 / 12.0 * (1.0 - 2.0 * d) / (1.0 - d)
    }
}
