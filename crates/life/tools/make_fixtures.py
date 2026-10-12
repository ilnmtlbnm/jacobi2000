"""Builds crates/life/tests/fixtures/ from the published Burn checkpoints.

Input: a directory holding the safetensors and oracle files that llm-life's
`export` tool writes from its Burn checkpoints (Burn tensor names and
layouts: `Linear` weights `[in, out]`, LayerNorm `gamma`/`beta`):

    bert-d16-L1.safetensors  bert-oracle.json
    mlp2-32-lrfix.safetensors  mlp2-oracle.json
    stencil-d16-L1.safetensors  stencil-oracle.json

Output, per model, the PyTorch `state_dict` layout the `life` crate reads
(`Linear` weights `[out, in]`, LayerNorm `weight`/`bias`) as
`<model>/model.safetensors` + `<model>/config.json`, and `reference.json`:

  * `burn`: the Burn oracle logits (all 512 neighbourhoods for bert and mlp2;
    a 16x16 board for stencil, from llm-life's xorshift `Grid.random`, which
    is why `--llm-life-pytorch` is needed for that one).
  * `torch`: logits of the PyTorch modules below on boards of several sizes
    (non-square, tiny, wrap-around heavy), and on all 512 neighbourhoods.

The script also checks the PyTorch modules against the Burn oracles and
records the differences in `reference.json["checks"]`.

    python make_fixtures.py --burn-export DIR --out DIR [--llm-life-pytorch DIR]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


class BertOfLife(nn.Module):
    """Post-LN encoder layer (burn-nn 0.20 TransformerEncoderLayer, norm_first
    false), 1 head, no mask, 9 tokens, read out at token 8."""

    def __init__(self, d_model=16, n_heads=1, d_ff=64):
        super().__init__()
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.tok_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(9, d_model)
        self.mha = nn.ModuleDict({k: nn.Linear(d_model, d_model) for k in ("query", "key", "value", "output")})
        self.pwff = nn.ModuleDict({"linear_inner": nn.Linear(d_model, d_ff), "linear_outer": nn.Linear(d_ff, d_model)})
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def split(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, tokens):
        b, t = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(t))
        q, k, v = (self.split(self.mha[n](x)) for n in ("query", "key", "value"))
        a = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1) @ v
        a = self.mha["output"](a.transpose(1, 2).reshape(b, t, self.d_model))
        x = self.norm_1(x + a)
        x = self.norm_2(x + self.pwff["linear_outer"](F.gelu(self.pwff["linear_inner"](x))))
        return self.head(x[:, -1, :])


class Mlp2OfLife(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(9, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 2)

    def forward(self, bits):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(bits)))))


def stencil_neighbors(width, height):
    """Flat [n*9] indices: row i's 8 torus neighbours NW,N,NE,W,E,SW,S,SE,
    then i itself."""
    idx = []
    for i in range(width * height):
        x, y = i % width, i // width
        xm, xp, ym, yp = (x - 1) % width, (x + 1) % width, (y - 1) % height, (y + 1) % height
        idx += [ym * width + xm, ym * width + x, ym * width + xp, y * width + xm, y * width + xp,
                yp * width + xm, yp * width + x, yp * width + xp, i]
    return torch.tensor(idx, dtype=torch.long)


class StencilBlock(nn.Module):
    def __init__(self, d, d_ff, heads):
        super().__init__()
        self.heads, self.d_k = heads, d // heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.proj = (nn.Linear(d, d) for _ in range(4))
        self.ff1, self.ff2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)

    def forward(self, x, neighbors):
        b, t, d = x.shape
        h = self.norm1(x)
        sp = lambda y: y.reshape(b, t, self.heads, self.d_k).permute(0, 2, 1, 3)
        q, k, v = sp(self.q(h)), sp(self.k(h)), sp(self.v(h))
        k_nb = k[:, :, neighbors, :].reshape(b, self.heads, t, 9, self.d_k)
        v_nb = v[:, :, neighbors, :].reshape(b, self.heads, t, 9, self.d_k)
        scores = (q.reshape(b, self.heads, t, 1, self.d_k) * k_nb).sum(-1) / math.sqrt(self.d_k)
        a = (F.softmax(scores, dim=-1).unsqueeze(-1) * v_nb).sum(3).permute(0, 2, 1, 3).reshape(b, t, d)
        x = x + self.proj(a)
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x))))


class StencilOfLife(nn.Module):
    def __init__(self, d_model=16, heads=1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([StencilBlock(d_model, 4 * d_model, heads)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, cells, neighbors):
        b, n = cells.shape
        x = self.embed(cells.reshape(b, n, 1))
        for blk in self.blocks:
            x = blk(x, neighbors)
        return self.head(x).reshape(b, n)


def to_torch_layout(sd):
    out = {}
    for k, v in sd.items():
        if k.endswith(".gamma"):
            k = k[:-6] + ".weight"
        elif k.endswith(".beta"):
            k = k[:-5] + ".bias"
        elif k.endswith(".weight") and v.dim() == 2 and k not in ("tok_emb.weight", "pos_emb.weight"):
            v = v.t()
        out[k] = v.contiguous()
    return out


def exhaustive_bits():
    cases = []
    for k in range(512):
        cases.append([(k >> b) & 1 for b in range(8)] + [(k >> 8) & 1])
    return cases


def board_string(a):
    return "".join(str(int(v)) for v in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burn-export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm-life-pytorch", help="llm-life's pytorch/ directory (life.py: the Rust RNG port)")
    args = ap.parse_args()
    src, out = args.burn_export, args.out

    specs = {
        "bert": ("bert-d16-L1", BertOfLife(), {"model": "bert", "d_model": 16, "n_heads": 1, "n_layers": 1, "d_ff": 64}),
        "mlp2": ("mlp2-32-lrfix", Mlp2OfLife(), {"model": "mlp2", "hidden": 32}),
        "stencil": ("stencil-d16-L1", StencilOfLife(), {"model": "stencil", "d_model": 16, "n_heads": 1, "n_layers": 1, "d_ff": 64}),
    }
    models = {}
    for name, (stem, module, cfg) in specs.items():
        sd = to_torch_layout(load_file(os.path.join(src, stem + ".safetensors")))
        module.load_state_dict(sd, strict=True)
        module.eval()
        models[name] = module
        os.makedirs(os.path.join(out, name), exist_ok=True)
        save_file(sd, os.path.join(out, name, "model.safetensors"))
        with open(os.path.join(out, name, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        print(name, sum(p.numel() for p in module.parameters()), "parameters")

    ref = {"burn": {}, "torch": {"exhaustive": {}, "boards": []}, "checks": {}}
    bits = exhaustive_bits()
    with torch.no_grad():
        # 512 neighbourhoods
        lg = models["bert"](torch.tensor(bits, dtype=torch.long)).numpy()
        ref["torch"]["exhaustive"]["bert"] = lg.reshape(-1).tolist()
        lg_m = models["mlp2"](torch.tensor(bits, dtype=torch.float32)).numpy()
        ref["torch"]["exhaustive"]["mlp2"] = lg_m.reshape(-1).tolist()
        # stencil on 3x3 tori: the centre cell sees exactly the 9 cells
        nb3 = stencil_neighbors(3, 3)
        cent = []
        for bs in bits:
            grid = np.zeros(9, dtype=np.float32)
            for pos, v in zip([0, 1, 2, 3, 5, 6, 7, 8, 4], bs):
                grid[pos] = v
            cent.append(float(models["stencil"](torch.tensor(grid).unsqueeze(0), nb3)[0, 4]))
        ref["torch"]["exhaustive"]["stencil"] = cent

        for name, lgt in (("bert", lg), ("mlp2", lg_m)):
            want = np.array(json.load(open(os.path.join(src, name + "-oracle.json")))["logits"], dtype=np.float32).reshape(-1)
            ref["burn"][name] = want.tolist()
            diff = float(np.abs(lgt.reshape(-1) - want).max())
            ref["checks"][f"{name}_torch_vs_burn_512"] = diff
            print(f"{name}: torch vs Burn oracle, 512 neighbourhoods, max abs diff {diff:.3e}")

        if args.llm_life_pytorch:
            sys.path.insert(0, args.llm_life_pytorch)
            from life import Grid  # llm-life's port of crates/life (xorshift RNG)

            g = Grid.random(16, 16, 1_000_000, 0.28)
            want = np.array(json.load(open(os.path.join(src, "stencil-oracle.json")))["logits"], dtype=np.float32)
            cells = torch.tensor(g.cells, dtype=torch.float32).unsqueeze(0)
            got = models["stencil"](cells, stencil_neighbors(16, 16)).squeeze(0).numpy()
            diff = float(np.abs(got - want).max())
            ref["checks"]["stencil_torch_vs_burn_16x16"] = diff
            ref["burn"]["stencil"] = {"w": 16, "h": 16, "cells": board_string(g.cells), "logits": want.tolist()}
            print(f"stencil: torch vs Burn oracle, 16x16 held-out board, max abs diff {diff:.3e}")

        rng = np.random.RandomState(20261012)
        for w, h, density in [(16, 16, 0.28), (37, 23, 0.5), (5, 7, 0.4), (2, 5, 0.5), (1, 4, 0.5), (8, 8, 0.35), (9, 17, 0.45)]:
            cells = (rng.rand(w * h) < density).astype(np.int64)
            nb = stencil_neighbors(w, h)
            toks, bitsf = [], []
            for i in range(w * h):
                x, y = i % w, i // w
                xm, xp, ym, yp = (x - 1) % w, (x + 1) % w, (y - 1) % h, (y + 1) % h
                t = [cells[ym * w + xm], cells[ym * w + x], cells[ym * w + xp], cells[y * w + xm], cells[y * w + xp],
                     cells[yp * w + xm], cells[yp * w + x], cells[yp * w + xp], cells[i]]
                toks.append(t)
            toks = torch.tensor(np.array(toks), dtype=torch.long)
            ref["torch"]["boards"].append({
                "w": w, "h": h, "cells": board_string(cells),
                "bert": models["bert"](toks).numpy().reshape(-1).tolist(),
                "mlp2": models["mlp2"](toks.float()).numpy().reshape(-1).tolist(),
                "stencil": models["stencil"](torch.tensor(cells, dtype=torch.float32).unsqueeze(0), nb).numpy().reshape(-1).tolist(),
            })

    with open(os.path.join(out, "reference.json"), "w") as f:
        json.dump(ref, f, separators=(",", ":"))
    print("wrote", os.path.join(out, "reference.json"), os.path.getsize(os.path.join(out, "reference.json")), "bytes")


if __name__ == "__main__":
    main()
