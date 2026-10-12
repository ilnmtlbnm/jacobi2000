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
use crate::{Setup, SizeLimits};

/// Steps between samples of the block sums inside a frame's window.
const SAMPLE_EVERY: u32 = 5;
/// Steps at the end of each `advance` that are averaged into a frame.
const WINDOW: u32 = 50;
/// Weight of a new frame in the running average across frames.
const ALPHA: f32 = 0.5;

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
    /// CPU-side accumulated block sums of the current frame.
    acc: Option<BlockSums>,
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
    let rho = 7.0 * SIGNED_N0;
    let block = if block == 0 { signed_block(s.w, s.h) } else { block };
    signed::Params {
        seed: s.params.seed,
        rule: Rule::Optimal,
        block,
        every: signed_every(block),
        n0: SIGNED_N0,
        boundary: Some(signed::Boundary { rho, u: s.ux, inlet: 32, outlet: 64 }),
    }
}

/// The signed initial cells and block values: the uniform inflow sampled
/// around the block references, obstacle sites marked.
pub fn signed_state(s: &Setup, p: &signed::Params) -> (Vec<u16>, Vec<[i32; 3]>) {
    let rho = 7.0 * SIGNED_N0;
    let ux = s.ux;
    signed::initial(s.w, s.h, p, |x, y| s.obstacle.contains(x, y), |_, _| (rho, ux, 0.0), s.params.seed)
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

impl Page {
    fn new(model: Model, setup: Setup, block: usize, refblock: usize, eng: Eng, adapter: String) -> Self {
        let field = Field::new(setup.w / block, setup.h / block, block, block);
        Page { eng, model, setup, block, field, refblock, acc: None, adapter }
    }

    /// The signed model's parameters with reference block `refblock` (0:
    /// the default), checked against the lattice.
    fn signed_for(setup: &Setup, refblock: usize) -> Result<signed::Params, String> {
        let p = signed_params(setup, refblock);
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
        let setup = setup_for(model, w, h, u, mode, block, Some(&SizeLimits::cpu(crate::WASM_CPU_MAX_BYTES)))?;
        let mut rb = 0;
        let eng = match model {
            Model::Fhp3 => Eng::Fhp3Cpu(Box::new(Bitsliced::from_setup(&setup))),
            Model::Ternary => {
                Eng::TernaryCpu(Box::new(Ternary::new(setup.w, setup.h, ternary_cells(&setup), ternary_params(&setup))))
            }
            Model::Signed => {
                let p = Self::signed_for(&setup, refblock)?;
                rb = p.block;
                let (cells, nodes) = signed_state(&setup, &p);
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
        // Rounded here, checked against the adapter's limits by the engine.
        let setup = setup_for(model, w, h, u, mode, block, None)?;
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
                let p = Self::signed_for(&setup, refblock)?;
                rb = p.block;
                let g = signed::gpu::Gpu::new(setup.w, setup.h, p, || {
                    let (cells, nodes) = signed_state(&setup, &p);
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

    fn accumulate(&mut self) {
        let b = self.block;
        let s = match &mut self.eng {
            Eng::Fhp3Cpu(e) => e.block_sums(b, b),
            Eng::TernaryCpu(e) => e.block_sums(b, b),
            Eng::SignedCpu(e) => e.page_sums(b, b),
            #[cfg(feature = "gpu")]
            Eng::Fhp3Gpu(g) => return g.accumulate_sums(b, b),
            #[cfg(feature = "gpu")]
            Eng::TernaryGpu(g) => return g.accumulate_sums(b, b),
            #[cfg(feature = "gpu")]
            Eng::SignedGpu(g) => return g.accumulate_sums(b, b),
        };
        match self.acc.as_mut() {
            Some(a) => a.add(&s),
            None => self.acc = Some(s),
        }
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

    /// Run `n` steps; the last min(n, 50) are sampled every 5 steps into
    /// the frame's block sums.
    pub fn advance(&mut self, n: u32) {
        let window = n.min(WINDOW);
        let samples = (window / SAMPLE_EVERY).max(1);
        let gap = window / samples;
        self.run(n - samples * gap);
        for _ in 0..samples {
            self.run(gap);
            self.accumulate();
        }
    }

    /// The GPU engine's accumulated block sums of the frame (None on the CPU).
    pub async fn take_gpu_sums(&self) -> Option<BlockSums> {
        match &self.eng {
            #[cfg(feature = "gpu")]
            Eng::Fhp3Gpu(g) => Some(g.take_sums().await),
            #[cfg(feature = "gpu")]
            Eng::TernaryGpu(g) => Some(g.take_sums().await),
            #[cfg(feature = "gpu")]
            Eng::SignedGpu(g) => Some(g.take_sums().await),
            _ => None,
        }
    }

    /// Fold the frame's block sums (the GPU's, else the CPU's) into the
    /// time-averaged field; returns them.
    pub fn fold(&mut self, gpu: Option<BlockSums>) -> Option<BlockSums> {
        let sums = gpu.or_else(|| self.acc.take());
        if let Some(s) = &sums {
            self.field.add(s, ALPHA);
        }
        sums
    }

    /// `take_gpu_sums` then `fold`.
    pub async fn sample(&mut self) -> Option<BlockSums> {
        let g = self.take_gpu_sums().await;
        self.fold(g)
    }

    /// RGBA vorticity image, one pixel per block, top row first; the scale
    /// saturates at 4U/D for both models.
    pub fn frame(&self) -> Vec<u8> {
        self.field.vorticity_rgba(4.0 * self.setup.ux as f32 / self.plate_len() as f32)
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
