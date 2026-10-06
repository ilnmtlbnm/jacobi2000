//! The wasm threads pool. wasm-bindgen-rayon's `initThreadPool` builds
//! rayon's global pool, which can be built only once per wasm instance (one
//! per worker): a second call fails. `initThreadPoolOnce` returns the first
//! call's promise to every later caller, so a worker that loads the threads
//! build twice (one run per backend, a model switch) is safe.

use std::cell::RefCell;

use wasm_bindgen::prelude::*;

/// wasm-bindgen-rayon's own entry point, re-exported so a consumer crate
/// keeps exporting `initThreadPool` to JS.
pub use wasm_bindgen_rayon::init_thread_pool;

thread_local! {
    static POOL: RefCell<Option<js_sys::Promise>> = const { RefCell::new(None) };
}

#[wasm_bindgen(js_name = initThreadPoolOnce)]
pub fn init_thread_pool_once(num_threads: usize) -> js_sys::Promise {
    POOL.with(|p| p.borrow_mut().get_or_insert_with(|| init_thread_pool(num_threads)).clone())
}
