"""Step 3 of docs/pytorch-training.md: load a Burn checkpoint's weights
(exported by `jacobi export-weights`) into this PyTorch model, run the same
fixed synthetic input `jacobi forward-probe` uses, and report the max
absolute difference against the Rust forward pass's own dump.

Usage:
    jacobi export-weights --model artifacts/model20 --out /tmp/model20-pt
    jacobi forward-probe --model artifacts/model20 --out /tmp/model20-probe.f32
    python -m jacobi2000_pt.parity --export /tmp/model20-pt --rust-probe /tmp/model20-probe.f32
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .data import META_DIM
from .model import Jacobi2000, Jacobi2000Config, stencil_mask


def probe_input(nx: int, ny: int) -> np.ndarray:
    """Byte-for-byte the same formula as `crates/cli/src/main.rs::probe_input`."""
    state = np.zeros((3, ny, nx), dtype=np.float32)
    for c in range(3):
        for y in range(ny):
            for x in range(nx):
                state[c, y, x] = (
                    (c + 1) * 0.05
                    * math.sin(2 * math.pi * (c + 1) * x / nx)
                    * math.cos(2 * math.pi * y / ny)
                )
    return state


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="directory from `jacobi export-weights`")
    ap.add_argument("--rust-probe", required=True, help="file from `jacobi forward-probe`")
    args = ap.parse_args()

    model, config = load_from_export(args.export)
    model.eval()

    state = probe_input(config["nx"], config["ny"])
    x = torch.from_numpy(state).unsqueeze(0)
    meta_vec = np.array(
        [math.log(config["dx"]), math.log(config["dt"]), math.log(config["g"]),
         math.log(config["depth"]), math.log(config["nu"] + 1e-6)],
        dtype=np.float32,
    )
    meta = torch.from_numpy(meta_vec).unsqueeze(0)

    mask_cfg = config["mask"]
    mask = None
    if isinstance(mask_cfg, dict) and "Stencil" in mask_cfg:
        mask = stencil_mask(model.cfg, mask_cfg["Stencil"]["radius"], mask_cfg["Stencil"]["dilation"])

    with torch.no_grad():
        out = model.forward(x, meta, mask).squeeze(0).numpy().astype(np.float32)

    rust_out = np.fromfile(args.rust_probe, dtype="<f4").reshape(out.shape)
    diff = np.abs(out - rust_out)
    print(f"max abs diff: {diff.max():.6e}")
    print(f"mean abs diff: {diff.mean():.6e}")
    print(f"rust output range: [{rust_out.min():.4f}, {rust_out.max():.4f}]")
    print(f"torch output range: [{out.min():.4f}, {out.max():.4f}]")


if __name__ == "__main__":
    main()
