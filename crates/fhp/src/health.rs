//! The first GPU error or device loss of an engine, kept for the page. In
//! the browser an uncaptured error would otherwise panic (wgpu's default
//! handler) and silently stop the worker; natively the default handler is
//! kept, so the tests still fail on any validation error.

#![cfg(feature = "gpu")]

use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub struct Health(Arc<Mutex<Option<String>>>);

impl Health {
    pub fn watch(device: &wgpu::Device) -> Self {
        let h = Health::default();
        #[cfg(target_arch = "wasm32")]
        {
            let e = h.0.clone();
            device.on_uncaptured_error(Box::new(move |err| {
                e.lock().unwrap().get_or_insert_with(|| format!("GPU error: {err}"));
            }));
            let l = h.0.clone();
            device.set_device_lost_callback(move |reason, msg| {
                l.lock().unwrap().get_or_insert_with(|| format!("GPU device lost ({reason:?}): {msg}"));
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = device;
        h
    }

    /// The first error or loss, if any.
    pub fn error(&self) -> Option<String> {
        self.0.lock().unwrap().clone()
    }
}
