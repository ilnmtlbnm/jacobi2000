//! wasm-bindgen surface for the dev page: one `page::Page` (FHP-III,
//! ternary or signed ternary FHP, on the GPU or a CPU engine on one thread or a
//! wasm-bindgen-rayon pool in the threaded build), plus the coarse-grained
//! vorticity image. The page's worker only loads, loops and draws.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use crate::page::{Model, Page};

#[wasm_bindgen(js_name = fhpInit)]
pub fn fhp_init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct Sim {
    inner: Rc<RefCell<Page>>,
}

#[wasm_bindgen]
impl Sim {
    /// CPU engine for `model` (fhp3, ternary or signed) on the vortex-street
    /// setup, inflow speed `u`, `mode` periodic, sponge, randomize or delete
    /// (ternary and signed: randomize), coarse-graining `block` 8, 16 or 32; the size
    /// is rounded up to the block (see `street_for`) and checked against the
    /// CPU memory bound. `refblock` is the signed model's reference block,
    /// 0 for the default H/64. Errors come back as a message, never a panic.
    #[wasm_bindgen(js_name = createCpu)]
    pub fn create_cpu(model: &str, w: usize, h: usize, u: f64, mode: &str, block: usize, refblock: usize) -> Result<Sim, JsValue> {
        let p = Model::parse(model).and_then(|m| Page::cpu_ref(m, w, h, u, mode, block, refblock)).map_err(|e| JsValue::from_str(&e))?;
        Ok(Sim { inner: Rc::new(RefCell::new(p)) })
    }

    /// The flow page's street on the CPU engine (see `Page::flow_cpu`).
    #[wasm_bindgen(js_name = createCpuFlowStreet)]
    pub fn create_cpu_flow_street() -> Result<Sim, JsValue> {
        let p = Page::flow_cpu().map_err(|e| JsValue::from_str(&e))?;
        Ok(Sim { inner: Rc::new(RefCell::new(p)) })
    }

    /// The flow page's street on the WebGPU engine (see `Page::flow_cpu`).
    #[cfg(feature = "gpu")]
    #[wasm_bindgen(js_name = createGpuFlowStreet)]
    pub async fn create_gpu_flow_street() -> Result<Sim, JsValue> {
        let p = Page::flow_gpu().await.map_err(|e| JsValue::from_str(&e))?;
        Ok(Sim { inner: Rc::new(RefCell::new(p)) })
    }

    /// WebGPU engine, arguments as for `createCpu`.
    #[cfg(feature = "gpu")]
    #[wasm_bindgen(js_name = createGpu)]
    pub async fn create_gpu(model: String, w: usize, h: usize, u: f64, mode: String, block: usize, refblock: usize) -> Result<Sim, JsValue> {
        let m = Model::parse(&model).map_err(|e| JsValue::from_str(&e))?;
        let p = Page::gpu_ref(m, w, h, u, &mode, block, refblock).await.map_err(|e| JsValue::from_str(&e))?;
        Ok(Sim { inner: Rc::new(RefCell::new(p)) })
    }

    /// Run `n` steps (see `Page::advance`: the time average is sampled and
    /// folded every 5 steps, whatever the split into calls; nothing waits).
    pub fn advance(&self, n: u32) {
        self.inner.borrow_mut().advance(n);
    }

    /// Ask for the current time-averaged field: on the GPU a read-back is
    /// queued unless one is in flight (steps go on meanwhile); on the CPU
    /// the field is always current.
    #[wasm_bindgen(js_name = requestField)]
    pub fn request_field(&self) {
        self.inner.borrow_mut().request_field();
    }

    /// Whether `field` now holds a newer average (the read-back has landed;
    /// always true on the CPU). Does not wait.
    #[wasm_bindgen(js_name = fieldReady)]
    pub fn field_ready(&self) -> bool {
        self.inner.borrow_mut().field_ready()
    }

    /// Read-backs whose map failed (dropped; a new request starts a new one).
    #[wasm_bindgen(getter, js_name = readFailures)]
    pub fn read_failures(&self) -> u32 {
        self.inner.borrow().read_failures()
    }

    /// Whether a read-back is in flight.
    #[wasm_bindgen(getter)]
    pub fn reading(&self) -> bool {
        self.inner.borrow().reading()
    }

    /// Resolves when the read-back in flight lands (at once if none). Holds
    /// no borrow while pending, so `advance` can be called meanwhile.
    #[cfg(feature = "gpu")]
    #[wasm_bindgen(js_name = waitField)]
    pub fn wait_field(&self) -> js_sys::Promise {
        let rx = self.inner.borrow().field_waiter();
        wasm_bindgen_futures::future_to_promise(async move {
            if let Some(rx) = rx {
                let _ = rx.await;
            }
            Ok(JsValue::UNDEFINED)
        })
    }

    /// Bring the field up to the current step: wait for any read-back in
    /// flight, then read back the current average (on the CPU, at once).
    #[cfg(feature = "gpu")]
    pub fn sample(&self) -> js_sys::Promise {
        let inner = self.inner.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            for request in [false, true] {
                if request {
                    inner.borrow_mut().request_field();
                }
                loop {
                    let rx = {
                        let mut p = inner.borrow_mut();
                        if p.field_ready() || !p.reading() {
                            break;
                        }
                        p.field_waiter()
                    };
                    if let Some(rx) = rx {
                        let _ = rx.await;
                    }
                }
            }
            Ok(JsValue::UNDEFINED)
        })
    }

    /// The CPU field is always current.
    #[cfg(not(feature = "gpu"))]
    pub fn sample(&self) {}

    /// RGBA vorticity image, `imageWidth` x `imageHeight`, top row first.
    pub fn frame(&self) -> Vec<u8> {
        self.inner.borrow().frame()
    }

    /// Time-averaged vorticity per block over the frame's 4U/D scale, as
    /// `Float32Array`, `imageWidth` x `imageHeight`, top row first. 1 is the
    /// colour scale's saturation, the sign is the rotation; not clamped.
    pub fn field(&self) -> Vec<f32> {
        self.inner.borrow().field()
    }

    /// 1 where a block holds the obstacle, as `Uint8Array`, same grid as `field`.
    pub fn obstacle(&self) -> Vec<u8> {
        self.inner.borrow().obstacle()
    }

    #[wasm_bindgen(getter, js_name = imageWidth)]
    pub fn image_width(&self) -> usize {
        self.inner.borrow().field.nx
    }

    #[wasm_bindgen(getter, js_name = imageHeight)]
    pub fn image_height(&self) -> usize {
        self.inner.borrow().field.ny
    }

    #[wasm_bindgen(getter)]
    pub fn step(&self) -> u32 {
        self.inner.borrow().step()
    }

    /// Lattice width actually used (after rounding).
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.inner.borrow().setup.w
    }

    /// Lattice height actually used (after rounding).
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.inner.borrow().setup.h
    }

    /// The setup in use: periodic, or the outlet's name.
    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> String {
        self.inner.borrow().mode().to_string()
    }

    /// fhp3, ternary or signed.
    #[wasm_bindgen(getter)]
    pub fn model(&self) -> String {
        self.inner.borrow().model.name().to_string()
    }

    /// Coarse-graining block size.
    #[wasm_bindgen(getter)]
    pub fn block(&self) -> usize {
        self.inner.borrow().block
    }

    /// The signed model's reference block, independent of the display
    /// block (0 for the other models).
    #[wasm_bindgen(getter, js_name = refBlock)]
    pub fn ref_block(&self) -> usize {
        self.inner.borrow().reference_block()
    }

    #[wasm_bindgen(getter)]
    pub fn sites(&self) -> f64 {
        let s = self.inner.borrow();
        (s.setup.w * s.setup.h) as f64
    }

    /// Reynolds number of the current mean flow.
    #[wasm_bindgen(getter)]
    pub fn reynolds(&self) -> f64 {
        self.inner.borrow().reynolds()
    }

    #[wasm_bindgen(getter)]
    pub fn adapter(&self) -> String {
        self.inner.borrow().adapter.clone()
    }

    /// Threads in the CPU pool (1 without the threaded build).
    #[wasm_bindgen(getter)]
    pub fn threads(&self) -> usize {
        #[cfg(feature = "threads")]
        return rayon::current_num_threads();
        #[cfg(not(feature = "threads"))]
        1
    }
}
