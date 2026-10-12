//! A dependency-free safetensors reader. Reads the files the PyTorch export
//! (`training/jacobi2000_pt/export.py`) writes: an 8-byte little-endian header length, a JSON header
//! (`{name: {dtype, shape, data_offsets}}`), then the raw little-endian F32
//! buffer (docs/pytorch-training.md).

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn as_1(&self) -> &[f32] {
        assert_eq!(self.shape.len(), 1, "expected rank 1, got {:?}", self.shape);
        &self.data
    }
}

pub type Weights = HashMap<String, Tensor>;

pub fn read_safetensors(path: &Path) -> Weights {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    read_safetensors_bytes(&bytes)
}

pub fn read_safetensors_bytes(bytes: &[u8]) -> Weights {
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let data_start = 8 + header_len;

    let mut out = Weights::new();
    for (name, v) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        assert_eq!(v["dtype"].as_str().unwrap(), "F32", "tensor `{name}`: only F32 is supported");
        let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let offsets = v["data_offsets"].as_array().unwrap();
        let start = data_start + offsets[0].as_u64().unwrap() as usize;
        let end = data_start + offsets[1].as_u64().unwrap() as usize;
        let data: Vec<f32> = bytes[start..end].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        out.insert(name.clone(), Tensor { shape, data });
    }
    out
}

/// A named tensor, or panic naming what was missing (checkpoints are
/// expected to carry every tensor `docs/pytorch-training.md` lists).
pub fn get<'a>(w: &'a Weights, name: &str) -> &'a Tensor {
    w.get(name).unwrap_or_else(|| panic!("missing tensor `{name}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_minimal_safetensors_buffer() {
        // Build the safetensors byte layout by hand.
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes_le: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = serde_json::json!({
            "w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        buf.extend_from_slice(&bytes_le);

        let w = read_safetensors_bytes(&buf);
        let t = get(&w, "w");
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data, data);
    }
}
