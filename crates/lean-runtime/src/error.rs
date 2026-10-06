use std::fmt;

/// Every fallible call in this crate returns this; nothing here panics on a
/// missing adapter, a refused device, a failed map or a size over a limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    NoAdapter(String),
    NoDevice(String),
    /// A request above what the device allows: what, needed, allowed.
    Limit { what: String, need: u64, have: u64 },
    Map(String),
    Poll(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoAdapter(e) => write!(f, "no wgpu adapter: {e}"),
            Error::NoDevice(e) => write!(f, "no wgpu device: {e}"),
            Error::Limit { what, need, have } => write!(f, "{what} needs {need}, the device allows {have}"),
            Error::Map(e) => write!(f, "buffer map failed: {e}"),
            Error::Poll(e) => write!(f, "device poll failed: {e}"),
        }
    }
}

impl std::error::Error for Error {}
