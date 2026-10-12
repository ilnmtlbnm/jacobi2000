"""Reference fixture for the Rust inference engine.

`python -m jacobi2000_pt.parity --out <dir>` builds a tiny, seeded,
non-trivially initialised model, runs it in PyTorch, and writes

- `model.safetensors` and `config.json` (stencil mask), `config-full.json`
  (same weights, full attention): a checkpoint in the exchange format
  `export.py` writes,
- `reference.safetensors`: the input state, the conditioning vector, and
  PyTorch's outputs (one-step forward for both masks, and the stencil
  model's rollout after 1, 10 and 50 steps),

which `crates/engine/tests/parity.rs` loads and compares against the Rust
forward pass. The committed copy lives in `crates/engine/tests/fixtures/pt-tiny`;
regenerate it with the command above if the model definition changes.

`load_from_export` is also what `train.py --init` uses to warm-start.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .model import Jacobi2000, Jacobi2000Config, stencil_mask

ROLLOUT_STEPS = (1, 10, 50)


def load_from_export(export_dir: str | Path) -> tuple[Jacobi2000, dict]:
    export_dir = Path(export_dir)
    config = json.loads((export_dir / "config.json").read_text())
    cfg = Jacobi2000Config(
        nx=config["nx"], ny=config["ny"], patch=config["patch"], channels=config["channels"],
        dim=config["dim"], layers=config["layers"], heads=config["heads"],
        ff_mult=config["ff_mult"], meta_dim=config["meta_dim"],
    )
    model = Jacobi2000(cfg)
    tensors = load_file(str(export_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(tensors, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model, config


def probe_input(channels: int, nx: int, ny: int) -> np.ndarray:
    """A deterministic, non-trivial state: sinusoids per channel."""
    state = np.zeros((channels, ny, nx), dtype=np.float32)
    for c in range(channels):
        for y in range(ny):
            for x in range(nx):
                state[c, y, x] = (
                    (c + 1) * 0.05
                    * math.sin(2 * math.pi * (c + 1) * x / nx)
                    * math.cos(2 * math.pi * y / ny)
                )
    return state


def tiny_model(seed: int = 0) -> Jacobi2000:
    """Small enough to commit (about 150 kB of weights). Every parameter gets
    seeded noise on top of its init so no layer is an identity or zero."""
    torch.manual_seed(seed)
    model = Jacobi2000(Jacobi2000Config(nx=16, ny=16, patch=4, dim=32, layers=2, heads=4))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
        # Keep the increment small so a 50-step rollout stays bounded.
        model.head.conv2.weight.mul_(0.05)
        model.head.conv2.bias.mul_(0.05)
    return model.eval()


def write_fixture(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = tiny_model()
    cfg = model.cfg
    save_file(
        {k: v.detach().contiguous() for k, v in model.state_dict().items()},
        str(out_dir / "model.safetensors"),
    )
    config = {
        "nx": cfg.nx, "ny": cfg.ny, "patch": cfg.patch, "channels": cfg.channels,
        "dim": cfg.dim, "layers": cfg.layers, "heads": cfg.heads, "ff_mult": cfg.ff_mult,
        "meta_dim": cfg.meta_dim,
        "norm": {"mean": [0.1, -0.05, 0.02], "std": [0.3, 0.2, 0.2], "depth": 1.0},
        "dx": 1.0, "dt": 0.09, "g": 9.81, "depth": 1.0, "nu": 1e-3,
        "mask": {"Stencil": {"radius": 1, "dilation": 1}},
        "energy_r_max": 0.999,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "config-full.json").write_text(json.dumps({**config, "mask": "Full"}, indent=2))

    x0 = torch.from_numpy(probe_input(cfg.channels, cfg.nx, cfg.ny)).unsqueeze(0)
    meta = torch.tensor([[0.1 * (i + 1) for i in range(cfg.meta_dim)]], dtype=torch.float32)
    stencil = stencil_mask(cfg, 1, 1)
    ref = {"state": x0[0].contiguous(), "meta": meta[0].contiguous()}
    with torch.no_grad():
        ref["forward_stencil"] = model.forward(x0, meta, stencil)[0].contiguous()
        ref["forward_full"] = model.forward(x0, meta, None)[0].contiguous()
        x = x0
        for i in range(1, max(ROLLOUT_STEPS) + 1):
            x = model.step(x, meta, stencil)
            if i in ROLLOUT_STEPS:
                ref[f"rollout_{i}"] = x[0].contiguous().clone()
    save_file(ref, str(out_dir / "reference.safetensors"))


def check_fixture(fixture_dir: Path) -> float:
    """Reload the committed checkpoint into the PyTorch model and compare
    with the committed reference outputs; returns the max absolute difference."""
    model, _ = load_from_export(fixture_dir)
    model.eval()
    ref = load_file(str(fixture_dir / "reference.safetensors"))
    x = ref["state"].unsqueeze(0)
    meta = ref["meta"].unsqueeze(0)
    stencil = stencil_mask(model.cfg, 1, 1)
    worst = 0.0
    with torch.no_grad():
        for name, mask in (("forward_stencil", stencil), ("forward_full", None)):
            out = model.forward(x, meta, mask)[0]
            worst = max(worst, (out - ref[name]).abs().max().item())
        for i in range(1, max(ROLLOUT_STEPS) + 1):
            x = model.step(x, meta, stencil)
            if i in ROLLOUT_STEPS:
                worst = max(worst, (x[0] - ref[f"rollout_{i}"]).abs().max().item())
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the fixture to this directory")
    ap.add_argument("--check", help="check a fixture directory against this PyTorch model")
    args = ap.parse_args()
    if args.out:
        write_fixture(Path(args.out))
        print(f"wrote fixture to {args.out}")
    if args.check:
        print(f"max abs diff against reference: {check_fixture(Path(args.check)):.3e}")


if __name__ == "__main__":
    main()
