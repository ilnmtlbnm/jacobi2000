//! The dev page's simulation without wasm-bindgen, so native tests can run
//! the same constructors: FHP-III (bit-sliced CPU or GPU engine), ternary
//! FHP (per-site table engine on the CPU, table-lookup engine on the GPU,
//! rule `chiral`, density 0.6) or signed ternary FHP (fixed-point engine on
//! the CPU or the GPU, rule `optimal`, reference block H/64, n0 = 4), on the
//! vortex-street setup, with the coarse-grained
//! vorticity on `block` x `block` blocks.

use crate::bitsliced::Bitsliced;
use crate::signed::{self, Signed};
use crate::coarse::{BlockSums, Field};
use crate::ternary::engine::{self, Ternary};
use crate::ternary::{self as tern, Rule};
use crate::{Obstacle, Setup, SizeLimits};

/// The block sums are sampled after every step that is a multiple of
/// SAMPLE_EVERY, counted from the start of the simulation, and each sample
/// is folded into the running average.
const SAMPLE_EVERY: u32 = 5;
/// Steps over which the running average decays by half (a weight of
/// 1 - 0.5^(SAMPLE_EVERY / HALF_LIFE) per sample).
const HALF_LIFE: f64 = 50.0;

/// Weight of a new sample in the running average.
fn alpha() -> f32 {
    (1.0 - 0.5f64.powf(SAMPLE_EVERY as f64 / HALF_LIFE)) as f32
}

/// Ternary density (particles per channel), the operating point.
pub const TERNARY_DENSITY: f64 = 0.6;
/// Ternary `chiral` at d = 0.6, measured:
/// g at U = 0.1 and nu from the shear wave at H = 256.
pub const TERNARY_G: f64 = 0.3796;
pub const TERNARY_NU: f64 = 0.0853;
/// FHP-III at d = 0.28, measured the same way (g at U = 0.1, nu from the shear wave at H = 256); Re 59 at 4096 x 1024.
pub const FHP3_G: f64 = 0.3648;
pub const FHP3_NU: f64 = 0.1371;
/// Bytes per site the per-site ternary engine holds (two u16 copies and
/// six u32 source indices).
const TERNARY_CPU_BYTES: f64 = 28.0;
/// Signed: reference occupation n0 of the page's street. The reference block is tied to the
/// plate (`signed_block`), independent of the display block.
pub const SIGNED_N0: f64 = 4.0;
/// Signed B16 N4 n0 4, measured (g at U = 0.1
/// and nu from the shear wave at H = 256); Re 199 at 4096 x 1024.
pub const SIGNED_G: f64 = 0.9941;
pub const SIGNED_NU: f64 = 0.1107;
/// Signed B8 N1, measured the same way (shear scan).
pub const SIGNED_B8_G: f64 = 0.9985;
pub const SIGNED_B8_NU: f64 = 0.1150;

/// The signed reference block for a W x H street: H/64, so the plate (H/8
/// rows) spans 8 blocks (wake bands at 3.5 blocks across the plate, a street
/// at 7), as a power of
/// two from 8 to 64 dividing W and H.
pub fn signed_block(w: usize, h: usize) -> usize {
    let mut b = 64;
    while b > 8 && (b > h / 64 || !w.is_multiple_of(b) || !h.is_multiple_of(b)) {
        b /= 2;
    }
    b
}

/// The re-estimate period for reference block b: 1 at B8 (N4 diverged
/// there), 4 above.
pub fn signed_every(b: usize) -> u32 {
    if b <= 8 {
        1
    } else {
        4
    }
}

/// Measured g and nu for reference block b (B8 N1, else B16 N4's).
pub fn signed_g_nu(b: usize) -> (f64, f64) {
    if b <= 8 {
        (SIGNED_B8_G, SIGNED_B8_NU)
    } else {
        (SIGNED_G, SIGNED_NU)
    }
}
/// Bytes per site the CPU signed engine holds (two u16 copies, six u32
/// source indices, the reference as 7 i32).
const SIGNED_CPU_BYTES: f64 = 56.0;

/// Signed street options beyond the size and the reference block. `Default`
/// is the stock street: plate H/8 rows at W/6, one row off the centre line,
/// outlet of 64 columns, uniform start.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Street {
    /// Plate (x, y, rows); None keeps the stock plate.
    pub plate: Option<(usize, usize, usize)>,
    /// Width of the outlet refresh zone in columns; 0 keeps 64.
    pub outlet: usize,
    /// Amplitude, in units of the inflow speed, of a uniform y push in a
    /// Gaussian patch in the plate's wake at the start (0: none). It breaks
    /// the mirror symmetry of a plate centred on the lattice's axis, so the
    /// street starts without waiting for noise to do it.
    pub kick: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Fhp3,
    Ternary,
    Signed,
}

impl Model {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "fhp3" => Ok(Model::Fhp3),
            "ternary" => Ok(Model::Ternary),
            "signed" => Ok(Model::Signed),
            _ => Err(format!("unknown model {s}: fhp3, ternary or signed")),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Model::Fhp3 => "fhp3",
            Model::Ternary => "ternary",
            Model::Signed => "signed",
        }
    }
}

enum Eng {
    Fhp3Cpu(Box<Bitsliced>),
    TernaryCpu(Box<Ternary>),
    SignedCpu(Box<Signed>),
    #[cfg(feature = "gpu")]
    Fhp3Gpu(Box<crate::gpu::Gpu>),
    #[cfg(feature = "gpu")]
    TernaryGpu(Box<tern::gpu::Gpu>),
    #[cfg(feature = "gpu")]
    SignedGpu(Box<signed::gpu::Gpu>),
}

pub struct Page {
    eng: Eng,
    pub model: Model,
    pub setup: Setup,
    pub block: usize,
    pub field: Field,
    /// The signed model's reference block (0 for the others).
    pub refblock: usize,
    /// Steps asked for by `advance` and not run yet: a GPU engine stops
    /// when its sample ring is full, until `sample` reads it back.
    pending: u32,
    /// Samples held on the GPU, not yet read back.
    held: usize,
    /// The last sample folded since the last `sample`.
    last: Option<BlockSums>,
    pub adapter: String,
}

/// The street for `model`: `street_for` (size rounded to the block, outlet
/// `mode`), at density 0.6 for ternary; ternary and signed only have the
/// `randomize` outlet. With `cpu_limits` the size is checked against them with the CPU
/// engine's footprint; without, the GPU engine checks it.
fn setup_for(model: Model, w: usize, h: usize, u: f64, mode: &str, block: usize, cpu_limits: Option<&SizeLimits>) -> Result<Setup, String> {
    let unlimited = SizeLimits { max_state_buffer: u64::MAX, max_total: u64::MAX, max_rows: u64::MAX };
    match model {
        Model::Fhp3 => crate::street_for(w, h, u, mode, block, cpu_limits.unwrap_or(&unlimited)),
        Model::Ternary | Model::Signed => {
            if mode != "randomize" {
                return Err(format!("{} has only the randomize outlet (got mode {mode})", model.name()));
            }
            let mut s = crate::street_for(w, h, u, mode, block, &unlimited)?;
            let bytes = if model == Model::Ternary { TERNARY_CPU_BYTES } else { SIGNED_CPU_BYTES };
            if let Some(l) = cpu_limits {
                l.check_bytes(s.w, s.h, bytes, bytes)?;
            }
            if model == Model::Ternary {
                s.density = TERNARY_DENSITY;
            }
            Ok(s)
        }
    }
}

/// The signed engine's parameters for a street setup: rule `optimal`,
/// reference block `block` (0: `signed_block`) with its `signed_every`,
/// n0 = 4, the open boundary at the reference density and the inflow speed.
pub fn signed_params(s: &Setup, block: usize) -> signed::Params {
    signed_params_with(s, block, &Street::default())
}

/// `signed_params` with the street options (outlet width).
pub fn signed_params_with(s: &Setup, block: usize, street: &Street) -> signed::Params {
    let rho = 7.0 * SIGNED_N0;
    let block = if block == 0 { signed_block(s.w, s.h) } else { block };
    signed::Params {
        seed: s.params.seed,
        rule: Rule::Optimal,
        block,
        every: signed_every(block),
        n0: SIGNED_N0,
        boundary: Some(signed::Boundary { rho, u: s.ux, inlet: 32, outlet: if street.outlet == 0 { 64 } else { street.outlet } }),
    }
}

/// The signed initial cells and block values: the uniform inflow sampled
/// around the block references, obstacle sites marked.
pub fn signed_state(s: &Setup, p: &signed::Params) -> (Vec<u16>, Vec<[i32; 3]>) {
    signed_state_with(s, p, &Street::default())
}

/// `signed_state` with the street options (start kick).
pub fn signed_state_with(s: &Setup, p: &signed::Params, street: &Street) -> (Vec<u16>, Vec<[i32; 3]>) {
    let rho = 7.0 * SIGNED_N0;
    let ux = s.ux;
    let kick = street.kick * ux;
    let (axis, len, rows) = match s.obstacle {
        Obstacle::Plate { y, rows, .. } => (y as f64 - 0.5, rows as f64 * 3f64.sqrt() / 2.0, rows as f64),
        _ => (s.h as f64 / 2.0, 1.0, 1.0),
    };
    let kx = match s.obstacle {
        Obstacle::Plate { x, .. } => x as f64 + 2.0 * len,
        _ => 0.0,
    };
    signed::initial(
        s.w,
        s.h,
        p,
        |x, y| s.obstacle.contains(x, y),
        |x, y| {
            let uy = if kick == 0.0 {
                0.0
            } else {
                let dx = (x - kx) / len;
                let dy = (y - axis) / (0.6 * rows);
                kick * (-(dx * dx) - dy * dy).exp()
            };
            (rho, ux, uy)
        },
        s.params.seed,
    )
}

/// The ternary engine's parameters for a street setup: rule `chiral`, the
/// open boundary at the inflow density and speed.
pub fn ternary_params(s: &Setup) -> engine::Params {
    engine::Params { seed: s.params.seed, rule: Rule::Chiral, boundary: Some(engine::Boundary::new(s.density, s.ux)) }
}

/// The ternary initial cells: fresh equilibrium samples at the inflow,
/// obstacle sites empty.
pub fn ternary_cells(s: &Setup) -> Vec<u16> {
    let mean = tern::equilibrium(2, 2, s.density, s.ux, 0.0);
    (0..s.w * s.h)
        .map(|i| if s.obstacle.contains(i % s.w, i / s.w) { engine::OBST } else { engine::sample_equilibrium(&mean, s.params.seed, i as u32) })
        .collect()
}

/// Apply the street's plate to the setup, checked to lie inside the lattice
/// and clear of the inlet and outlet zones.
fn place_plate(setup: &mut Setup, street: &Street) -> Result<(), String> {
    let Some((x, y, rows)) = street.plate else { return Ok(()) };
    if rows == 0 || y < rows / 2 + 1 || y + rows / 2 + 1 > setup.h || x < 64 || x + 64 > setup.w {
        return Err(format!("plate of {rows} rows at ({x}, {y}) does not fit a {}x{} lattice", setup.w, setup.h));
    }
    setup.obstacle = Obstacle::Plate { x, y, rows, cols: 2 };
    Ok(())
}

/// The flow page's street: a 1536x512 signed lattice with inflow 0.25, the
/// randomize outlet, reference block 8 and display block 16, a 65-row plate at
/// (256, 256), a 128-column outlet zone and a 0.3 start kick.
pub const FLOW: (usize, usize, f64, &str, usize, usize) = (1536, 512, 0.25, "randomize", 16, 8);

/// The street options of the flow page.
pub fn flow_street() -> Street {
    Street { plate: Some((256, 256, 65)), outlet: 128, kick: 0.3 }
}

impl Page {
    /// The flow page's street on the CPU engine.
    pub fn flow_cpu() -> Result<Self, String> {
        let (w, h, u, mode, block, refblock) = FLOW;
        Self::cpu_street(Model::Signed, w, h, u, mode, block, refblock, &flow_street())
    }

    /// The flow page's street on the WebGPU engine.
    #[cfg(feature = "gpu")]
    pub async fn flow_gpu() -> Result<Self, String> {
        let (w, h, u, mode, block, refblock) = FLOW;
        Self::gpu_street(Model::Signed, w, h, u, mode, block, refblock, &flow_street()).await
    }

    fn new(model: Model, setup: Setup, block: usize, refblock: usize, eng: Eng, adapter: String) -> Self {
        let field = Field::new(setup.w / block, setup.h / block, block, block);
        Page { eng, model, setup, block, field, refblock, pending: 0, held: 0, last: None, adapter }
    }

    /// The signed model's parameters with reference block `refblock` (0:
    /// the default), checked against the lattice.
    fn signed_for(setup: &Setup, refblock: usize, street: &Street) -> Result<signed::Params, String> {
        let p = signed_params_with(setup, refblock, street);
        let b = p.block;
        if !(2..=64).contains(&b) || !b.is_power_of_two() || !setup.w.is_multiple_of(b) || !setup.h.is_multiple_of(b) {
            return Err(format!("reference block {b}: a power of two from 2 to 64 dividing {}x{}", setup.w, setup.h));
        }
        Ok(p)
    }

    /// CPU engine (bit-sliced FHP-III or per-site ternary), checked against
    /// the browser's CPU memory bound. Errors are messages, never panics.
    pub fn cpu(model: Model, w: usize, h: usize, u: f64, mode: &str, block: usize) -> Result<Self, String> {
        Self::cpu_ref(model, w, h, u, mode, block, 0)
    }

    /// `cpu` with the signed model's reference block `refblock` (0: the
    /// default, `signed_block`).
    pub fn cpu_ref(model: Model, w: usize, h: usize, u: f64, mode: &str, block: usize, refblock: usize) -> Result<Self, String> {
        Self::cpu_street(model, w, h, u, mode, block, refblock, &Street::default())
    }

    /// `cpu_ref` with the signed street options.
    #[allow(clippy::too_many_arguments)]
    pub fn cpu_street(model: Model, w: usize, h: usize, u: f64, mode: &str, block: usize, refblock: usize, street: &Street) -> Result<Self, String> {
        let mut setup = setup_for(model, w, h, u, mode, block, Some(&SizeLimits::cpu(crate::WASM_CPU_MAX_BYTES)))?;
        place_plate(&mut setup, street)?;
        let mut rb = 0;
        let eng = match model {
            Model::Fhp3 => Eng::Fhp3Cpu(Box::new(Bitsliced::from_setup(&setup))),
            Model::Ternary => {
                Eng::TernaryCpu(Box::new(Ternary::new(setup.w, setup.h, ternary_cells(&setup), ternary_params(&setup))))
            }
            Model::Signed => {
                let p = Self::signed_for(&setup, refblock, street)?;
                rb = p.block;
                let (cells, nodes) = signed_state_with(&setup, &p, street);
                Eng::SignedCpu(Box::new(Signed::from_state(setup.w, setup.h, cells, nodes, p)))
            }
        };
        Ok(Self::new(model, setup, block, rb, eng, String::new()))
    }

    /// GPU engine (bit-sliced FHP-III or table-lookup ternary), checked
    /// against the adapter's limits.
    #[cfg(feature = "gpu")]
    pub async fn gpu(model: Model, w: usize, h: usize, u: f64, mode: &str, block: usize) -> Result<Self, String> {
        Self::gpu_ref(model, w, h, u, mode, block, 0).await
    }

    /// `gpu` with the signed model's reference block `refblock` (0: the
    /// default).
    #[cfg(feature = "gpu")]
    pub async fn gpu_ref(model: Model, w: usize, h: usize, u: f64, mode: &str, block: usize, refblock: usize) -> Result<Self, String> {
        Self::gpu_street(model, w, h, u, mode, block, refblock, &Street::default()).await
    }

    /// `gpu_ref` with the signed street options.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    pub async fn gpu_street(model: Model, w: usize, h: usize, u: f64, mode: &str, block: usize, refblock: usize, street: &Street) -> Result<Self, String> {
        // Rounded here, checked against the adapter's limits by the engine.
        let mut setup = setup_for(model, w, h, u, mode, block, None)?;
        place_plate(&mut setup, street)?;
        let mut rb = 0;
        let (eng, adapter) = match model {
            Model::Fhp3 => {
                let g = crate::gpu::Gpu::from_setup(&setup).await?;
                let a = g.adapter.clone();
                (Eng::Fhp3Gpu(Box::new(g)), a)
            }
            Model::Ternary => {
                let p = ternary_params(&setup);
                let g = tern::gpu::Gpu::open(setup.w, setup.h, || ternary_cells(&setup), p.seed, tern::gpu::Layout::Table, p.boundary).await?;
                let a = g.adapter.clone();
                (Eng::TernaryGpu(Box::new(g)), a)
            }
            Model::Signed => {
                let p = Self::signed_for(&setup, refblock, street)?;
                rb = p.block;
                let g = signed::gpu::Gpu::new(setup.w, setup.h, p, || {
                    let (cells, nodes) = signed_state_with(&setup, &p, street);
                    let blocks = nodes.len();
                    (cells, nodes, vec![[0; tern::NCH]; blocks])
                })
                .await?;
                let a = g.adapter.clone();
                (Eng::SignedGpu(Box::new(g)), a)
            }
        };
        Ok(Self::new(model, setup, block, rb, eng, adapter))
    }

    fn run(&mut self, n: u32) {
        match &mut self.eng {
            Eng::Fhp3Cpu(b) => b.run(n as usize),
            Eng::TernaryCpu(t) => t.run(n as usize),
            Eng::SignedCpu(t) => t.run(n as usize),
            #[cfg(feature = "gpu")]
            Eng::Fhp3Gpu(g) => g.run(n as usize),
            #[cfg(feature = "gpu")]
            Eng::TernaryGpu(g) => g.run(n as usize),
            #[cfg(feature = "gpu")]
            Eng::SignedGpu(g) => g.run(n as usize),
        }
    }

    /// Sample the block sums: folded on the spot on the CPU, queued into
    /// the engine's sample ring on the GPU.
    fn take_sample(&mut self) {
        let b = self.block;
        let s = match &mut self.eng {
            Eng::Fhp3Cpu(e) => e.block_sums(b, b),
            Eng::TernaryCpu(e) => e.block_sums(b, b),
            Eng::SignedCpu(e) => e.page_sums(b, b),
            #[cfg(feature = "gpu")]
            Eng::Fhp3Gpu(g) => return g.sample_sums(b, b),
            #[cfg(feature = "gpu")]
            Eng::TernaryGpu(g) => return g.sample_sums(b, b),
            #[cfg(feature = "gpu")]
            Eng::SignedGpu(g) => return g.sample_sums(b, b),
        };
        self.fold_sample(s);
    }

    fn is_gpu(&self) -> bool {
        !matches!(self.eng, Eng::Fhp3Cpu(_) | Eng::TernaryCpu(_) | Eng::SignedCpu(_))
    }

    pub fn step(&self) -> u32 {
        match &self.eng {
            Eng::Fhp3Cpu(e) => e.step,
            Eng::TernaryCpu(e) => e.step,
            Eng::SignedCpu(e) => e.step,
            #[cfg(feature = "gpu")]
            Eng::Fhp3Gpu(g) => g.step,
            #[cfg(feature = "gpu")]
            Eng::TernaryGpu(g) => g.step,
            #[cfg(feature = "gpu")]
            Eng::SignedGpu(g) => g.step,
        }
    }

    /// Run `n` steps. The block sums are sampled after every step that is
    /// a multiple of 5 and each sample is folded into the time average, so
    /// the average depends on the steps only, not on how they are split
    /// across calls. The GPU engines keep up to `SAMPLE_SLOTS` samples on
    /// the device; `sample` reads them back and folds them in order (a full
    /// ring stops the run until then).
    pub fn advance(&mut self, n: u32) {
        self.pending += n;
        self.drive();
    }

    /// Run pending steps, sampling on the way, until they are done or the
    /// GPU sample ring is full.
    fn drive(&mut self) {
        while self.pending > 0 && self.held < crate::coarse::SAMPLE_SLOTS {
            let to_sample = SAMPLE_EVERY - self.step() % SAMPLE_EVERY;
            let k = self.pending.min(to_sample);
            self.run(k);
            self.pending -= k;
            if self.step().is_multiple_of(SAMPLE_EVERY) {
                self.take_sample();
                if self.is_gpu() {
                    self.held += 1;
                }
            }
        }
    }

    /// Whether GPU samples are waiting to be read back.
    pub fn needs_sums(&self) -> bool {
        self.held > 0
    }

    /// The GPU samples taken since the last read, oldest first (none on the
    /// CPU).
    pub async fn take_gpu_samples(&mut self) -> Vec<BlockSums> {
        match &mut self.eng {
            #[cfg(feature = "gpu")]
            Eng::Fhp3Gpu(g) => g.take_samples().await,
            #[cfg(feature = "gpu")]
            Eng::TernaryGpu(g) => g.take_samples().await,
            #[cfg(feature = "gpu")]
            Eng::SignedGpu(g) => g.take_samples().await,
            _ => vec![],
        }
    }

    fn fold_sample(&mut self, s: BlockSums) {
        self.field.add(&s, alpha());
        let full = (s.bw * s.bh) as i32;
        for (solid, &f) in self.field.solid.iter_mut().zip(&s.fluid) {
            *solid = f < full;
        }
        self.last = Some(s);
    }

    /// Fold GPU samples (from `take_gpu_samples`) in order, then run the
    /// pending steps.
    pub fn fold(&mut self, samples: Vec<BlockSums>) {
        self.held = 0;
        for s in samples {
            self.fold_sample(s);
        }
        self.drive();
    }

    /// Read back and fold the GPU samples, running every pending step (one
    /// read-back unless more than `SAMPLE_SLOTS` samples were due); returns
    /// the last sample folded since the last call.
    pub async fn sample(&mut self) -> Option<BlockSums> {
        while self.needs_sums() {
            let s = self.take_gpu_samples().await;
            self.fold(s);
        }
        self.last.take()
    }

    /// RGBA vorticity image, one pixel per block, top row first; the scale
    /// saturates at 4U/D for both models.
    pub fn frame(&self) -> Vec<u8> {
        self.field.vorticity_rgba(self.vorticity_scale())
    }

    /// The vorticity per block that `frame` colours, divided by its scale
    /// (1 = full saturation, sign = rotation, not clamped), top row first.
    pub fn field(&self) -> Vec<f32> {
        self.field.vorticity_values(self.vorticity_scale())
    }

    /// 1 per block that `frame` paints as an obstacle, top row first.
    pub fn obstacle(&self) -> Vec<u8> {
        self.field.obstacle_mask()
    }

    fn vorticity_scale(&self) -> f32 {
        4.0 * self.setup.ux as f32 / self.plate_len() as f32
    }

    fn plate_len(&self) -> f64 {
        self.setup.plate_rows() as f64 * 3f64.sqrt() / 2.0
    }

    /// Reynolds number U D g / nu of the current mean flow, with the
    /// measured g and nu of each model (each setup uses one density).
    pub fn reynolds(&self) -> f64 {
        let (g, nu) = match self.model {
            Model::Fhp3 => (FHP3_G, FHP3_NU),
            Model::Ternary => (TERNARY_G, TERNARY_NU),
            Model::Signed => signed_g_nu(self.refblock),
        };
        self.field.mean_ux() as f64 * self.plate_len() * g / nu
    }

    /// The signed model's reference block (0 for the others).
    pub fn reference_block(&self) -> usize {
        self.refblock
    }

    /// The setup in use: periodic, or the outlet's name.
    pub fn mode(&self) -> &'static str {
        self.setup.params.boundary.map_or("periodic", |b| b.kind.name())
    }
}
