"""Step mode with a rollout loss, the model20 recipe
(docs/runs/2026-09-19-2.md), ported from crates/train/src/fit.rs::fit.
AdamW, warmup + cosine decay, pushforward curriculum (single-step before
`rollout_after`, then `rollout` steps), noise injection, mass + optional
energy projection in the loop, plain MSE plus a spectral term."""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import META_DIM, Normalizer, TrajectoryDataset
from .export import export as export_checkpoint
from .model import Jacobi2000, Jacobi2000Config, stencil_mask
from .parity import load_from_export
from .project import Denorm, energy, project_energy, project_mass
from .spectral import band_mse


@dataclass
class FitConfig:
    steps: int = 3000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup: int = 100
    rollout: int = 4
    rollout_after: int = 200
    noise: float = 0.04
    project: bool = True
    project_energy: bool = False
    r_max: float = 1.0001
    spectral: float = 1.0
    lr_floor: float = 1.0
    log_every: int = 100
    batch: int = 16
    mask_radius: int | None = 1
    mask_dilation: int = 1
    seed: int = 20
    reverse: bool = False


def lr_at(step: int, cfg: FitConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return cfg.lr * (cfg.lr_floor + (1.0 - cfg.lr_floor) * cos)


def fit(
    model: Jacobi2000,
    dataset: TrajectoryDataset,
    norm: Normalizer,
    cfg: FitConfig,
    device: torch.device,
    on_log=lambda row: None,
) -> Jacobi2000:
    model.to(device)
    mask = (
        stencil_mask(model.cfg, cfg.mask_radius, cfg.mask_dilation, device)
        if cfg.mask_radius is not None
        else None
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    denorm = Denorm(mean=norm.mean, std=norm.std, depth=norm.depth)
    meta_vec = torch.from_numpy(dataset.meta_vector()).to(device)
    rng = np.random.default_rng(cfg.seed)

    t0 = time.time()
    for step in range(cfg.steps):
        k = 1 if step < cfg.rollout_after else cfg.rollout
        window = dataset.batch(rng, norm, cfg.batch, k, device)  # [k+1, b, 3, ny, nx]
        if cfg.reverse:
            # mirrors `crates/train/src/data.rs`'s `reverse` flag: flip the
            # frame order so the model learns state(t+1) -> state(t), mask
            # and solver unchanged.
            window = window.flip(0)
        b = window.shape[1]
        meta = meta_vec.unsqueeze(0).expand(b, META_DIM).contiguous()

        x = window[0]
        mass0 = x[:, 0:1].mean(dim=(2, 3)).reshape(b, 1) if cfg.project else None
        loss = torch.zeros((), device=device)
        for step_k in range(k):
            if cfg.noise > 0.0:
                x = x + torch.randn_like(x) * cfg.noise
            prev_energy = energy(x, denorm, dataset.g) if cfg.project_energy else None
            x = model.step(x, meta, mask)
            if cfg.project:
                x = project_mass(x, mass0)
            if prev_energy is not None:
                x = project_energy(x, prev_energy, denorm, dataset.g, cfg.r_max)
            target = window[step_k + 1]
            err = x - target
            loss = loss + err.pow(2).mean()
            if cfg.spectral > 0.0:
                loss = loss + band_mse(err) * cfg.spectral
        loss = loss / k

        opt.zero_grad(set_to_none=True)
        loss.backward()
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()

        if (step + 1) % cfg.log_every == 0 or step == 0:
            on_log({"step": step + 1, "loss": float(loss.item()), "secs": time.time() - t0})

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="directory written by `jacobi dump-dataset`")
    ap.add_argument("--out", required=True, help="output directory (safetensors + config.json)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=4)
    ap.add_argument("--rollout-after", type=int, default=200)
    ap.add_argument("--noise", type=float, default=0.04)
    ap.add_argument("--project", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--project-energy", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--r-max", type=float, default=1.0001)
    ap.add_argument("--spectral", type=float, default=1.0)
    ap.add_argument("--lr-floor", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--reverse", type=lambda s: s.lower() == "true", default=False,
                     help="flip the frame order so the model learns state(t+1) -> state(t), "
                          "mirroring `train --reverse` (docs/runs/2026-09-19-rewind.md)")
    ap.add_argument("--init", default=None, help="directory from a previous export to warm-start from")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrajectoryDataset(args.dataset)
    norm = dataset.fit_normalizer()

    if args.init is not None:
        model, _ = load_from_export(args.init)
    else:
        cfg = Jacobi2000Config(dim=args.dim, layers=args.layers)
        model = Jacobi2000(cfg)
    print(f"model: {model.num_params()} params, device {device}")

    fit_cfg = FitConfig(
        steps=args.steps,
        lr=args.lr,
        batch=args.batch,
        rollout=args.rollout,
        rollout_after=args.rollout_after,
        noise=args.noise,
        project=args.project,
        project_energy=args.project_energy,
        r_max=args.r_max,
        spectral=args.spectral,
        lr_floor=args.lr_floor,
        seed=args.seed,
        reverse=args.reverse,
    )

    log_rows = []

    def on_log(row):
        log_rows.append(row)
        print(f"step {row['step']:6d}  loss {row['loss']:.6f}  {row['secs']:.1f}s")

    t0 = time.time()
    fit(model, dataset, norm, fit_cfg, device, on_log)
    wall = time.time() - t0
    print(f"training wall time: {wall:.1f}s")

    export_checkpoint(
        model,
        norm,
        dx=dataset.dx,
        dt=dataset.dt,
        g=dataset.g,
        depth=dataset.depth,
        nu=dataset.nu,
        mask_radius=fit_cfg.mask_radius,
        mask_dilation=fit_cfg.mask_dilation,
        energy_r_max=fit_cfg.r_max,
        out_dir=args.out,
    )
    print(f"exported to {args.out}")

    if args.log_file:
        Path(args.log_file).write_text(
            json.dumps({"rows": log_rows, "wall_seconds": wall, "params": model.num_params()}, indent=2)
        )


if __name__ == "__main__":
    main()
