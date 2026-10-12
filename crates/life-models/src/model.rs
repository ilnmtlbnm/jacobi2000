//! Model kinds, `config.json` and the packed weight vector.
//!
//! Tensor names and layouts are the PyTorch `state_dict` of the training
//! modules: `nn.Linear` weights are `[out, in]`, LayerNorm is
//! `weight`/`bias`, embeddings are `[rows, d]`.
//!
//! * `bert`: `tok_emb.weight [2,d]`, `pos_emb.weight [9,d]`,
//!   `mha.{query,key,value,output}.{weight,bias}`, `norm_1.*`,
//!   `pwff.linear_inner.*`, `pwff.linear_outer.*`, `norm_2.*`,
//!   `head.{weight [2,d], bias}`.
//! * `mlp2`: `fc1.{weight [h,9], bias}`, `fc2.*`, `fc3.{weight [2,h], bias}`.
//! * `stencil`: `embed.{weight [d,1], bias}`, `blocks.0.{norm1,q,k,v,proj,
//!   norm2,ff1,ff2}.*`, `head.{weight [1,d], bias}`.

use std::collections::HashMap;

use crate::{st, Error};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Post-LN transformer encoder over the 9 neighbourhood cells as tokens;
    /// 2 logits `[dead, alive]` per cell.
    Bert,
    /// `Linear(9,h) ReLU Linear(h,h) ReLU Linear(h,2)`; 2 logits per cell.
    Mlp2,
    /// Pre-LN block, attention over each cell's 9 toroidal taps; 1 logit
    /// per cell (alive iff above 0).
    Stencil,
}

impl Kind {
    pub fn id(self) -> &'static str {
        match self {
            Kind::Bert => "bert",
            Kind::Mlp2 => "mlp2",
            Kind::Stencil => "stencil",
        }
    }

    /// Logits per cell.
    pub fn outputs(self) -> usize {
        match self {
            Kind::Bert | Kind::Mlp2 => 2,
            Kind::Stencil => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub kind: Kind,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    /// `mlp2` hidden width.
    pub hidden: usize,
    pub eps: f32,
}

type Shapes = Vec<(String, Vec<usize>)>;

fn lin(t: &mut Shapes, name: &str, o: usize, i: usize) {
    t.push((format!("{name}.weight"), vec![o, i]));
    t.push((format!("{name}.bias"), vec![o]));
}

fn norm(t: &mut Shapes, name: &str, n: usize) {
    t.push((format!("{name}.weight"), vec![n]));
    t.push((format!("{name}.bias"), vec![n]));
}

impl Config {
    /// `{"model": "bert" | "mlp2" | "stencil", "d_model", "n_heads", "n_layers"
    /// (must be 1), "d_ff" (default 4 * d_model), "hidden" (mlp2),
    /// "layer_norm_eps" (default 1e-5)}`.
    pub fn from_json(s: &str) -> Result<Config, Error> {
        let j: serde_json::Value = serde_json::from_str(s).map_err(|e| Error::Config(format!("not JSON: {e}")))?;
        let kind = match j["model"].as_str() {
            Some("bert") => Kind::Bert,
            Some("mlp2") => Kind::Mlp2,
            Some("stencil") => Kind::Stencil,
            other => return Err(Error::Config(format!("`model` must be bert, mlp2 or stencil, got {other:?}"))),
        };
        let uint = |key: &str| j[key].as_u64().map(|v| v as usize);
        let need = |key: &str| uint(key).filter(|v| *v > 0).ok_or_else(|| Error::Config(format!("`{key}` missing or zero")));
        let eps = j["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        Ok(match kind {
            Kind::Mlp2 => Config { kind, d_model: 0, n_heads: 0, d_ff: 0, hidden: need("hidden")?, eps },
            Kind::Bert | Kind::Stencil => {
                let d = need("d_model")?;
                let heads = uint("n_heads").unwrap_or(1);
                if heads == 0 || d % heads != 0 {
                    return Err(Error::Config(format!("d_model {d} is not divisible by n_heads {heads}")));
                }
                if uint("n_layers").unwrap_or(1) != 1 {
                    return Err(Error::Config("only n_layers = 1 is supported (the kernels fuse one layer per cell)".into()));
                }
                Config { kind, d_model: d, n_heads: heads, d_ff: uint("d_ff").unwrap_or(4 * d), hidden: 0, eps }
            }
        })
    }

    /// `(name, shape)` of every tensor, in packing order.
    pub fn tensors(&self) -> Shapes {
        let (d, f, h) = (self.d_model, self.d_ff, self.hidden);
        let mut t: Shapes = Vec::new();
        match self.kind {
            Kind::Bert => {
                t.push(("tok_emb.weight".into(), vec![2, d]));
                t.push(("pos_emb.weight".into(), vec![9, d]));
                for n in ["mha.query", "mha.key", "mha.value", "mha.output"] {
                    lin(&mut t, n, d, d);
                }
                norm(&mut t, "norm_1", d);
                lin(&mut t, "pwff.linear_inner", f, d);
                lin(&mut t, "pwff.linear_outer", d, f);
                norm(&mut t, "norm_2", d);
                lin(&mut t, "head", 2, d);
            }
            Kind::Mlp2 => {
                lin(&mut t, "fc1", h, 9);
                lin(&mut t, "fc2", h, h);
                lin(&mut t, "fc3", 2, h);
            }
            Kind::Stencil => {
                lin(&mut t, "embed", d, 1);
                norm(&mut t, "blocks.0.norm1", d);
                for n in ["q", "k", "v", "proj"] {
                    lin(&mut t, &format!("blocks.0.{n}"), d, d);
                }
                norm(&mut t, "blocks.0.norm2", d);
                lin(&mut t, "blocks.0.ff1", f, d);
                lin(&mut t, "blocks.0.ff2", d, f);
                lin(&mut t, "head", 1, d);
            }
        }
        t
    }

    pub fn num_params(&self) -> usize {
        self.tensors().iter().map(|(_, s)| s.iter().product::<usize>()).sum()
    }
}

/// All weights of one model in a single f32 vector, tensors in
/// [`Config::tensors`] order, with each tensor's offset.
#[derive(Clone, Debug)]
pub struct Params {
    pub data: Vec<f32>,
    offsets: HashMap<String, (usize, usize)>,
}

impl Params {
    pub fn from_safetensors(cfg: &Config, bytes: &[u8]) -> Result<Params, Error> {
        let mut raw = st::read(bytes)?;
        let mut data = Vec::new();
        let mut offsets = HashMap::new();
        for (name, shape) in cfg.tensors() {
            let t = raw.remove(&name).ok_or_else(|| Error::Weights(format!("missing tensor `{name}`")))?;
            if t.shape != shape {
                return Err(Error::Weights(format!("tensor `{name}` has shape {:?}, the config needs {shape:?}", t.shape)));
            }
            offsets.insert(name, (data.len(), t.data.len()));
            data.extend_from_slice(&t.data);
        }
        if let Some(extra) = raw.keys().next() {
            return Err(Error::Weights(format!("unexpected tensor `{extra}`")));
        }
        Ok(Params { data, offsets })
    }

    pub fn get(&self, name: &str) -> &[f32] {
        let (o, n) = self.offsets[name];
        &self.data[o..o + n]
    }

    pub fn offset(&self, name: &str) -> usize {
        self.offsets[name].0
    }
}
