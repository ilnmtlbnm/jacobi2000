//! Named-tensor exchange with the PyTorch training side
//! (`docs/pytorch-training.md`): every learnable tensor in [`Jacobi2000`] by
//! its burn field path, plus a minimal, dependency-free safetensors
//! reader/writer (the format is a JSON header plus a raw byte buffer; adding
//! the `safetensors` crate for this one F32-only use is not worth a new
//! dependency).
//!
//! This module, together with [`crate::model::Jacobi2000::from_weights`] and
//! [`crate::model::Jacobi2000::to_weights`], is the only place either side of
//! the exchange has to agree on names and shapes. Inference
//! (`Jacobi2000::forward`/`step`) and the Burn training loop (`crates/train`)
//! never see it.

use std::collections::HashMap;
use std::path::Path;

use burn::prelude::*;

/// One tensor: row-major shape plus its floats, in burn's own storage layout
/// for the field it came from (`nn::Linear` is `[in, out]`, `nn::Conv2d` is
/// `[out, in, kh, kw]`, `nn::LayerNorm` is `[dim]`).
#[derive(Debug, Clone)]
pub struct NamedTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub type Weights = HashMap<String, NamedTensor>;

pub(crate) fn get<const D: usize, B: Backend>(
    w: &Weights,
    name: &str,
    device: &B::Device,
) -> Tensor<B, D> {
    let t = w
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor `{name}`"));
    assert_eq!(
        t.shape.len(),
        D,
        "tensor `{name}`: expected rank {D}, got shape {:?}",
        t.shape
    );
    let mut shape = [0usize; D];
    shape.copy_from_slice(&t.shape);
    Tensor::<B, 1>::from_floats(t.data.as_slice(), device).reshape(shape)
}

pub(crate) fn put<const D: usize, B: Backend>(
    w: &mut Weights,
    name: &str,
    t: Tensor<B, D>,
) {
    let shape = t.dims().to_vec();
    let data = t.into_data().to_vec::<f32>().unwrap();
    w.insert(name.to_string(), NamedTensor { shape, data });
}

/// Write `weights` as a safetensors file: an 8-byte little-endian header
/// length, a JSON header (`{name: {dtype: "F32", shape, data_offsets}}`,
/// tensors sorted by name for a deterministic byte layout), then the raw
/// little-endian F32 buffer. Readable by the standard `safetensors` Python
/// package.
pub fn write_safetensors(path: &Path, weights: &Weights) {
    let mut names: Vec<&String> = weights.keys().collect();
    names.sort();

    let mut header = serde_json::Map::new();
    let mut buf = Vec::new();
    let mut offset = 0usize;
    for name in &names {
        let t = &weights[*name];
        let bytes: Vec<u8> = t.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let start = offset;
        let end = offset + bytes.len();
        header.insert(
            (*name).clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": t.shape,
                "data_offsets": [start, end],
            }),
        );
        buf.extend_from_slice(&bytes);
        offset = end;
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();

    let mut out = Vec::with_capacity(8 + header_bytes.len() + buf.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&buf);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(path, out).unwrap();
}

/// Read a safetensors file written by [`write_safetensors`] or by the
/// PyTorch export (`training/jacobi2000_pt/export.py`). F32 tensors only.
pub fn read_safetensors(path: &Path) -> Weights {
    let bytes = std::fs::read(path).unwrap();
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let data_start = 8 + header_len;

    let mut out = Weights::new();
    for (name, v) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        assert_eq!(
            v["dtype"].as_str().unwrap(),
            "F32",
            "tensor `{name}`: only F32 is supported"
        );
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let offsets = v["data_offsets"].as_array().unwrap();
        let start = data_start + offsets[0].as_u64().unwrap() as usize;
        let end = data_start + offsets[1].as_u64().unwrap() as usize;
        let data: Vec<f32> = bytes[start..end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        out.insert(name.clone(), NamedTensor { shape, data });
    }
    out
}
