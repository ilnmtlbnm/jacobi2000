"""Export a trained Jacobi2000 to safetensors + config.json
(docs/pytorch-training.md's weight exchange format) — the format
`jacobi import-weights` reads to bring the checkpoint back into Burn's own
binary format for `eval-metrics`, `round-trip`, and `web/`."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .data import Normalizer
from .model import Jacobi2000


def state_dict_for_export(model: Jacobi2000) -> dict[str, torch.Tensor]:
    """The model's own named parameters are already named to match burn's
    field layout (see model.py's docstring), so this is the identity plus a
    contiguous-and-cpu pass safetensors requires."""
    return {name: p.detach().contiguous().cpu() for name, p in model.state_dict().items()}


def export(
    model: Jacobi2000,
    norm: Normalizer,
    dx: float,
    dt: float,
    g: float,
    depth: float,
    nu: float,
    mask_radius: int | None,
    mask_dilation: int | None,
    energy_r_max: float,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = state_dict_for_export(model)
    save_file(tensors, str(out_dir / "model.safetensors"))

    cfg = model.cfg
    mask = (
        {"Stencil": {"radius": mask_radius, "dilation": mask_dilation}}
        if mask_radius is not None
        else "Full"
    )
    config = {
        "nx": cfg.nx,
        "ny": cfg.ny,
        "patch": cfg.patch,
        "channels": cfg.channels,
        "dim": cfg.dim,
        "layers": cfg.layers,
        "heads": cfg.heads,
        "ff_mult": cfg.ff_mult,
        "meta_dim": cfg.meta_dim,
        "norm": norm.to_dict(),
        "dx": dx,
        "dt": dt,
        "g": g,
        "depth": depth,
        "nu": nu,
        "mask": mask,
        "energy_r_max": energy_r_max,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
