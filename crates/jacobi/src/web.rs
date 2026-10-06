//! wasm-bindgen surface for the dev page: twelve panels (f32; the controls
//! f32 red-black and int32 red-black; ternary corrections; the lattice gas;
//! the circuit with stochastic rounding at 4, 8, 16 bits; the circuit at 4,
//! 8, 12, 16 bits) on WebGPU or on the single-thread CPU engines. The worker advances and syncs one panel at a time, then asks
//! for images and errors; all computation stays here.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use crate::bitsliced::Bitsliced;
use crate::gas::Estimator;
use crate::gpu::{GpuCtx, GpuPanel, Kind};
use crate::reference::F32Engine;
use crate::{gas, int32, ternary, Case, Problem};

/// Seed of the random streams (stochastic rounding, lattice gas).
const SEED: u32 = 1;
/// Bit widths of the stochastic-rounding panels.
const SR_BITS: [usize; 3] = [4, 8, 16];

#[wasm_bindgen(js_name = jacobiInit)]
pub fn jacobi_init() {
    console_error_panic_hook::set_once();
}

/// The grid size the engines use for a request, or the reason there is none.
#[wasm_bindgen(js_name = gridSize)]
pub fn grid_size(requested: f64) -> Result<usize, JsValue> {
    crate::grid_size(requested).map_err(|e| JsValue::from_str(&e))
}

enum Eng {
    CpuF32(F32Engine),
    CpuBits(Box<Bitsliced>),
    CpuTernary(Box<ternary::Bitsliced>),
    CpuInt32(Box<int32::Bitsliced>),
    CpuGas(Box<gas::Bitsliced>),
    Gpu(GpuPanel),
}

/// The last field read back, as v in [0, 1] per point.
enum Host {
    F32(Vec<f32>),
    Q(Vec<u32>, usize),
    /// i32 at the ternary scale (ternary, int32 red-black).
    I32(Vec<i32>),
    /// Lattice gas: v per 8 x 8 block over the estimator's window.
    Blocks(Vec<f64>),
}

impl Host {
    fn value(&self, n: usize, k: usize) -> f64 {
        match self {
            Host::F32(f) => f[k] as f64,
            Host::Q(q, bits) => q[k] as f64 / crate::qmax(*bits) as f64,
            Host::I32(u) => u[k] as f64 / ternary::Q as f64,
            Host::Blocks(b) => b[(k / n / gas::BLOCK) * (n / gas::BLOCK) + (k % n) / gas::BLOCK],
        }
    }
}

struct Panel {
    name: String,
    eng: Eng,
    host: Host,
    est: Option<Estimator>,
}

struct Inner {
    p: Problem,
    gpu: Option<GpuCtx>,
    panels: Vec<Panel>,
    round: bool,
}

impl Inner {
    fn new(p: Problem, gpu: Option<GpuCtx>) -> Result<Self, String> {
        let n = p.n;
        let mut panels = Vec::new();
        let panel = |name: String, eng: Eng, host: Host| Panel { name, eng, host, est: None };
        let f32_eng = match &gpu {
            Some(ctx) => Eng::Gpu(GpuPanel::f32(ctx, &p)?),
            None => Eng::CpuF32(F32Engine::new(n, p.initial_f32())),
        };
        panels.push(panel("f32".into(), f32_eng, Host::F32(p.initial_f32())));
        let eng = match &gpu {
            Some(ctx) => Eng::Gpu(GpuPanel::f32_red_black(ctx, &p)?),
            None => {
                let mut e = F32Engine::new(n, p.initial_f32());
                e.red_black = true;
                Eng::CpuF32(e)
            }
        };
        panels.push(panel("f32 red-black".into(), eng, Host::F32(p.initial_f32())));
        let eng = match &gpu {
            Some(ctx) => Eng::Gpu(GpuPanel::int32_red_black(ctx, n, &ternary::initial(&p))?),
            None => Eng::CpuInt32(Box::new(int32::Bitsliced::from_problem(&p))),
        };
        panels.push(panel("int32 red-black".into(), eng, Host::I32(ternary::initial(&p))));
        let eng = match &gpu {
            Some(ctx) => Eng::Gpu(GpuPanel::ternary(ctx, n, &ternary::initial(&p))?),
            None => Eng::CpuTernary(Box::new(ternary::Bitsliced::from_problem(&p))),
        };
        panels.push(panel("ternary".into(), eng, Host::I32(ternary::initial(&p))));
        let eng = match &gpu {
            Some(ctx) => Eng::Gpu(GpuPanel::gas(ctx, &p, &gas::initial(n), SEED, true)?),
            None => Eng::CpuGas(Box::new(gas::Bitsliced::from_problem(&p, SEED, true))),
        };
        let nb = n / gas::BLOCK;
        panels.push(Panel { name: "lattice gas".into(), eng, host: Host::Blocks(vec![0.0; nb * nb]), est: Some(Estimator::new(n)) });
        for bits in SR_BITS {
            let eng = match &gpu {
                Some(ctx) => Eng::Gpu(GpuPanel::stochastic(ctx, n, bits, &p.initial_q(bits), SEED)?),
                None => {
                    let mut e = Bitsliced::from_problem(&p, bits);
                    e.stochastic = true;
                    e.seed = SEED;
                    Eng::CpuBits(Box::new(e))
                }
            };
            panels.push(panel(format!("{bits} bits, stochastic"), eng, Host::Q(p.initial_q(bits), bits)));
        }
        for bits in crate::BITS {
            let eng = match &gpu {
                Some(ctx) => Eng::Gpu(GpuPanel::bits(ctx, &p, bits)?),
                None => Eng::CpuBits(Box::new(Bitsliced::from_problem(&p, bits))),
            };
            panels.push(panel(format!("{bits} bits"), eng, Host::Q(p.initial_q(bits), bits)));
        }
        Ok(Inner { p, gpu, panels, round: false })
    }
}

#[wasm_bindgen]
pub struct Viewer {
    inner: Rc<RefCell<Inner>>,
}

#[wasm_bindgen]
impl Viewer {
    /// All panels on the single-thread CPU engines; `n` as `gridSize`,
    /// `problem` "quadratic" (or "") or "sinh".
    #[wasm_bindgen(js_name = createCpu)]
    pub fn create_cpu(n: f64, problem: &str) -> Result<Viewer, JsValue> {
        let p = problem_of(n, problem)?;
        let inner = Inner::new(p, None).map_err(|e| JsValue::from_str(&e))?;
        Ok(Viewer { inner: Rc::new(RefCell::new(inner)) })
    }

    /// All panels on WebGPU.
    #[wasm_bindgen(js_name = createGpu)]
    pub async fn create_gpu(n: f64, problem: String) -> Result<Viewer, JsValue> {
        let p = problem_of(n, &problem)?;
        let ctx = GpuCtx::new().await.map_err(|e| JsValue::from_str(&e))?;
        let inner = Inner::new(p, Some(ctx)).map_err(|e| JsValue::from_str(&e))?;
        Ok(Viewer { inner: Rc::new(RefCell::new(inner)) })
    }

    /// Run (CPU) or queue (GPU) `iters` iterations of panel `i`.
    pub fn advance(&self, i: usize, iters: u32) {
        let s = &mut *self.inner.borrow_mut();
        let round = s.round;
        match &mut s.panels[i].eng {
            Eng::CpuF32(e) => e.run(iters as usize),
            Eng::CpuBits(e) => {
                e.round = round;
                e.run(iters as usize)
            }
            Eng::CpuTernary(e) => e.run(iters as usize),
            Eng::CpuInt32(e) => e.run(iters as usize),
            Eng::CpuGas(e) => e.run(iters as usize),
            Eng::Gpu(g) => {
                g.round = round;
                g.run(s.gpu.as_ref().unwrap(), iters as usize)
            }
        }
    }

    /// Resolves when the GPU has finished all queued iterations (at once on
    /// the CPU, where `advance` runs them).
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn done(&self) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let s = inner.borrow();
            if let Some(ctx) = s.gpu.as_ref() {
                ctx.done().await;
            }
            Ok(JsValue::UNDEFINED)
        })
    }

    /// Read panel `i`'s field back (waits for its queued GPU work). The
    /// worker awaits this before its next call, so the borrow held across
    /// the readback cannot collide with another.
    #[allow(clippy::await_holding_refcell_ref)]
    pub fn sync(&self, i: usize) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            // A gas panel yields its accumulators and step count instead.
            let (host, acc) = {
                let s = inner.borrow();
                match &s.panels[i].eng {
                    Eng::CpuF32(e) => (Some(Host::F32(e.cur.clone())), None),
                    Eng::CpuBits(e) => (Some(Host::Q(e.read_q(), e.bits)), None),
                    Eng::CpuTernary(e) => (Some(Host::I32(e.read())), None),
                    Eng::CpuInt32(e) => (Some(Host::I32(e.read())), None),
                    Eng::CpuGas(e) => (None, Some((e.acc.clone(), e.iter))),
                    Eng::Gpu(g) => {
                        let ctx = s.gpu.as_ref().unwrap();
                        match g.kind {
                            Kind::F32 | Kind::F32RedBlack => (Some(Host::F32(g.read_f32(ctx).await)), None),
                            Kind::Bits(b) | Kind::Stochastic(b) => (Some(Host::Q(g.read_q(ctx).await, b)), None),
                            Kind::Ternary | Kind::Int32RedBlack => (Some(Host::I32(g.read_planes(ctx, ternary::W).await.into_iter().map(|x| x as i32).collect())), None),
                            Kind::Gas => (None, Some((g.read_aux(ctx).await.into_iter().map(|x| x as i32).collect(), g.iter))),
                        }
                    }
                }
            };
            let s = &mut *inner.borrow_mut();
            let panel = &mut s.panels[i];
            if let Some(h) = host {
                panel.host = h;
            } else if let (Some((acc, it)), Some(est)) = (acc, panel.est.as_mut()) {
                if let Some((blocks, _)) = est.update(it, &acc) {
                    panel.host = Host::Blocks(blocks);
                }
            }
            Ok(JsValue::UNDEFINED)
        })
    }

    /// RGBA image of panel `i` as last synced, `imageSize` a side: the
    /// field, or its signed error against the analytic solution.
    pub fn frame(&self, i: usize, error: bool) -> Vec<u8> {
        let s = self.inner.borrow();
        let h = &s.panels[i].host;
        let n = s.p.n;
        crate::render::rgba(&s.p, |k| h.value(n, k), error)
    }

    /// Max |v - analytic| over the interior of panel `i` as last synced;
    /// for the lattice gas, over 8 x 8 blocks against the block mean.
    #[wasm_bindgen(js_name = maxError)]
    pub fn max_error(&self, i: usize) -> f64 {
        let s = self.inner.borrow();
        let h = &s.panels[i].host;
        let n = s.p.n;
        match h {
            Host::Blocks(b) => gas::block_error(&s.p, b),
            _ => s.p.max_error(|k| h.value(n, k)),
        }
    }

    /// Iterations of panel `i`.
    pub fn iteration(&self, i: usize) -> f64 {
        match &self.inner.borrow().panels[i].eng {
            Eng::CpuF32(e) => e.iter as f64,
            Eng::CpuBits(e) => e.iter as f64,
            Eng::CpuTernary(e) => e.iter as f64,
            Eng::CpuInt32(e) => e.iter as f64,
            Eng::CpuGas(e) => e.iter as f64,
            Eng::Gpu(g) => g.iter as f64,
        }
    }

    /// Round to nearest (true) or truncate (false) from the next iteration on.
    #[wasm_bindgen(js_name = setRound)]
    pub fn set_round(&self, round: bool) {
        self.inner.borrow_mut().round = round;
    }

    /// Back to the start: boundary set, interior zero, iteration 0.
    pub fn reset(&self) -> Result<(), JsValue> {
        let s = &mut *self.inner.borrow_mut();
        let gpu = s.gpu.take();
        let round = s.round;
        *s = Inner::new(s.p, gpu).map_err(|e| JsValue::from_str(&e))?;
        s.round = round;
        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn n(&self) -> usize {
        self.inner.borrow().p.n
    }

    #[wasm_bindgen(getter, js_name = imageSize)]
    pub fn image_size(&self) -> usize {
        crate::render::image_size(self.inner.borrow().p.n)
    }

    #[wasm_bindgen(getter)]
    pub fn panels(&self) -> usize {
        self.inner.borrow().panels.len()
    }

    /// "f32", "f32 red-black", "int32 red-black", "ternary", "lattice gas", "4 bits, stochastic", ..., "4 bits", ...
    #[wasm_bindgen(js_name = panelName)]
    pub fn panel_name(&self, i: usize) -> String {
        self.inner.borrow().panels[i].name.clone()
    }

    /// "quadratic" or "sinh".
    #[wasm_bindgen(getter)]
    pub fn problem(&self) -> String {
        self.inner.borrow().p.case.name().into()
    }

    #[wasm_bindgen(getter)]
    pub fn adapter(&self) -> String {
        self.inner.borrow().gpu.as_ref().map(|g| g.adapter.clone()).unwrap_or_default()
    }
}

fn problem_of(n: f64, problem: &str) -> Result<Problem, JsValue> {
    let case = Case::parse(problem).map_err(|e| JsValue::from_str(&e))?;
    crate::grid_size(n).and_then(|n| Problem::with_case(n, case)).map_err(|e| JsValue::from_str(&e))
}
