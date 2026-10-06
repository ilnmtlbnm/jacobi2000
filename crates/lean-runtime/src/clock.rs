#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
}

/// Wall clock in milliseconds: `performance.now()` in the browser (window or
/// worker), a process-relative `Instant` natively, 0 on wasm without `web`.
pub fn now_ms() -> f64 {
    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    {
        performance_now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1e3
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "web")))]
    {
        0.0
    }
}
