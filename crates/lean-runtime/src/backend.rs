//! The backend ladder: WebGPU if an adapter is granted, else CPU threads if
//! the page is cross-origin isolated with SharedArrayBuffer and more than
//! one hardware thread, else one CPU thread. Chosen by capability only,
//! never by a benchmark. `js/capabilities.js` is the browser side of the
//! same rule.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    WebGpu,
    Threads,
    Single,
}

impl Backend {
    /// The id the pages use: `webgpu`, `threads` or `single`.
    pub fn id(self) -> &'static str {
        match self {
            Backend::WebGpu => "webgpu",
            Backend::Threads => "threads",
            Backend::Single => "single",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "webgpu" => Some(Backend::WebGpu),
            "threads" => Some(Backend::Threads),
            "single" => Some(Backend::Single),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub has_adapter: bool,
    pub cross_origin_isolated: bool,
    pub shared_array_buffer: bool,
    pub hardware_concurrency: usize,
}

impl Capabilities {
    /// A native process: threads need no isolation; `has_adapter` comes from
    /// the caller (e.g. whether `gpu::request_adapter` succeeded).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn native(has_adapter: bool) -> Self {
        Capabilities {
            has_adapter,
            cross_origin_isolated: true,
            shared_array_buffer: true,
            hardware_concurrency: std::thread::available_parallelism().map_or(1, |n| n.get()),
        }
    }

    pub fn threads_capable(&self) -> bool {
        self.cross_origin_isolated && self.shared_array_buffer && self.hardware_concurrency > 1
    }

    /// Why `b` is not available, or `None` if it is.
    pub fn missing(&self, b: Backend) -> Option<&'static str> {
        match b {
            Backend::WebGpu if !self.has_adapter => Some("no WebGPU adapter"),
            Backend::Threads if !self.cross_origin_isolated => Some("not cross-origin isolated"),
            Backend::Threads if !self.shared_array_buffer => Some("no SharedArrayBuffer"),
            Backend::Threads if self.hardware_concurrency <= 1 => Some("one hardware thread"),
            _ => None,
        }
    }
}

/// The available backends, best first. Single is always last and always there.
pub fn ladder(caps: &Capabilities) -> Vec<Backend> {
    [Backend::WebGpu, Backend::Threads, Backend::Single].into_iter().filter(|b| caps.missing(*b).is_none()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(has_adapter: bool, coi: bool, sab: bool, n: usize) -> Capabilities {
        Capabilities { has_adapter, cross_origin_isolated: coi, shared_array_buffer: sab, hardware_concurrency: n }
    }

    #[test]
    fn ladder_order() {
        assert_eq!(ladder(&caps(true, true, true, 8)), [Backend::WebGpu, Backend::Threads, Backend::Single]);
        assert_eq!(ladder(&caps(false, true, true, 8)), [Backend::Threads, Backend::Single]);
        assert_eq!(ladder(&caps(true, false, true, 8)), [Backend::WebGpu, Backend::Single]);
        assert_eq!(ladder(&caps(false, true, true, 1)), [Backend::Single]);
        assert_eq!(caps(false, true, false, 8).missing(Backend::Threads), Some("no SharedArrayBuffer"));
    }

    #[test]
    fn ids_round_trip() {
        for b in [Backend::WebGpu, Backend::Threads, Backend::Single] {
            assert_eq!(Backend::parse(b.id()), Some(b));
        }
    }
}
