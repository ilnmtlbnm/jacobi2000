//! A dependency-free safetensors reader: an 8-byte little-endian header
//! length, a JSON header (`{name: {dtype, shape, data_offsets}}`), then the
//! raw little-endian F32 buffer. Reads the files PyTorch's
//! `safetensors.torch.save_file` writes.

use std::collections::HashMap;

use crate::Error;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn read(bytes: &[u8]) -> Result<HashMap<String, Tensor>, Error> {
    let bad = |m: &str| Error::Weights(m.to_string());
    let head: [u8; 8] = bytes.get(0..8).ok_or_else(|| bad("file shorter than 8 bytes"))?.try_into().unwrap();
    let header_len = u64::from_le_bytes(head) as usize;
    let header_end = 8usize.checked_add(header_len).filter(|e| *e <= bytes.len()).ok_or_else(|| bad("header length past end of file"))?;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end]).map_err(|e| Error::Weights(format!("header is not JSON: {e}")))?;
    let header = header.as_object().ok_or_else(|| bad("header is not an object"))?;
    let mut out = HashMap::new();
    for (name, v) in header {
        if name == "__metadata__" {
            continue;
        }
        if v["dtype"].as_str() != Some("F32") {
            return Err(Error::Weights(format!("tensor `{name}`: only F32 is supported")));
        }
        let shape: Vec<usize> = v["shape"].as_array().ok_or_else(|| bad("shape missing"))?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
        let off = v["data_offsets"].as_array().ok_or_else(|| bad("data_offsets missing"))?;
        let s = off.first().and_then(|x| x.as_u64()).ok_or_else(|| bad("bad data_offsets"))? as usize;
        let e = off.get(1).and_then(|x| x.as_u64()).ok_or_else(|| bad("bad data_offsets"))? as usize;
        let raw = bytes.get(header_end.saturating_add(s)..header_end.saturating_add(e)).ok_or_else(|| Error::Weights(format!("tensor `{name}`: data past end of file")))?;
        if raw.len() != shape.iter().product::<usize>() * 4 {
            return Err(Error::Weights(format!("tensor `{name}`: {} bytes for shape {shape:?}", raw.len())));
        }
        let data = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        out.insert(name.clone(), Tensor { shape, data });
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) fn write(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut body = Vec::new();
    for (name, shape, data) in tensors {
        let start = body.len();
        for v in data {
            body.extend_from_slice(&v.to_le_bytes());
        }
        header.insert((*name).to_string(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, body.len()]}));
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let bytes = write(&[("a", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]), ("b", vec![1], vec![5.0])]);
        let w = read(&bytes).unwrap();
        assert_eq!(w["a"].shape, vec![2, 2]);
        assert_eq!(w["a"].data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(w["b"].data, vec![5.0]);
    }

    #[test]
    fn rejects_truncated() {
        let bytes = write(&[("a", vec![2], vec![1.0, 2.0])]);
        assert!(read(&bytes[..bytes.len() - 1]).is_err());
        assert!(read(&bytes[..4]).is_err());
    }
}
