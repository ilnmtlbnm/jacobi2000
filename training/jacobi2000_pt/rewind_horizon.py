"""Roadmap item 4: negative time, building on the rewind model.

Design notes, section 4.3 (the Janssen loop, "ensemble over discretizations"):
"jitter the patch-grid offset per realization, same weights, average +
spread... Gives uncertainty for free." Section 11 lists as a measurable,
not yet done anywhere in this repo's run log: "rewind horizon (ticks until
ensemble spread exceeds threshold) vs viscosity." Section 6's rewind exhibit
predicts the shape: "the moment it starts smearing is the information
horizon of that physics... switch viscosity on: the horizon collapses to
almost nothing."

This script is that measurement. It takes an already-trained reverse
checkpoint (`train.py --reverse true`, as `docs/runs/2026-09-20-rewind-round-
trip.md` and `docs/runs/2026-09-29-pytorch-training.md` section 8 already
trained one at nu=0.001), starts an ensemble of K members at the same
"present" state, steps each one backward with an independently jittered
patch-grid offset each tick (`torch.roll` by a random `(dy, dx)` in
`[0, patch)` before the forward pass, rolled back after -- the same
jittered-discretization idea, without needing a mask/patch-offset argument
threaded through the Rust model), and reports the tick at which the
per-cell ensemble standard deviation first exceeds a fixed threshold
(relative to the dataset's own normalized-field std, so the threshold means
the same thing at every viscosity).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import Jacobi2000
from .parity import load_from_export


def jittered_step(model: Jacobi2000, x: torch.Tensor, meta: torch.Tensor, mask, patch: int, rng: np.random.Generator):
    dy = int(rng.integers(0, patch))
    dx = int(rng.integers(0, patch))
    xr = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    out = model.step(xr, meta, mask)
    return torch.roll(out, shifts=(-dy, -dx), dims=(2, 3))


def rewind_horizon(model: Jacobi2000, start: torch.Tensor, meta: torch.Tensor, mask,
                    k: int, max_ticks: int, threshold: float, seed: int, device: torch.device):
    """`start`: `[1, 3, ny, nx]` normalized state, the "present". Returns
    (spreads per tick, first tick spread exceeds threshold or None)."""
    patch = model.cfg.patch
    rngs = [np.random.default_rng(seed * 1000 + m) for m in range(k)]
    members = [start.clone() for _ in range(k)]
    spreads = []
    horizon = None
    for tick in range(1, max_ticks + 1):
        members = [jittered_step(model, members[m], meta, mask, patch, rngs[m]) for m in range(k)]
        stacked = torch.stack(members, dim=0)  # [k, 1, 3, ny, nx]
        spread = float(stacked.std(dim=0).mean().item())
        spreads.append(spread)
        if horizon is None and spread > threshold:
            horizon = tick
    return spreads, horizon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reverse", required=True, help="export dir of a --reverse true checkpoint")
    ap.add_argument("--dataset", required=True, help="raw dataset dump to draw a starting trajectory from")
    ap.add_argument("--traj-index", type=int, default=0)
    ap.add_argument("--start-tick", type=int, default=40, help="tick in the trajectory to call 'the present'")
    ap.add_argument("--k", type=int, default=8, help="ensemble size")
    ap.add_argument("--max-ticks", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.3, help="in normalized field units")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from .data import META_DIM, TrajectoryDataset
    from .model import stencil_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_from_export(args.reverse)
    model.to(device)
    model.eval()

    mask_cfg = config["mask"]
    mask = stencil_mask(model.cfg, mask_cfg["Stencil"]["radius"], mask_cfg["Stencil"]["dilation"], device) \
        if isinstance(mask_cfg, dict) and "Stencil" in mask_cfg else None

    dataset = TrajectoryDataset(args.dataset)
    norm = dataset.fit_normalizer()
    window = dataset.window(norm, args.traj_index, args.start_tick, 0)
    start = torch.from_numpy(window[0:1]).to(device)
    meta_vec = torch.from_numpy(dataset.meta_vector()).to(device)
    meta = meta_vec.unsqueeze(0)

    with torch.no_grad():
        spreads, horizon = rewind_horizon(
            model, start, meta, mask, args.k, args.max_ticks, args.threshold, args.seed, device
        )

    print(f"nu={dataset.nu}  ensemble K={args.k}  threshold={args.threshold}")
    for tick, spread in enumerate(spreads, start=1):
        flag = " <-- horizon" if tick == horizon else ""
        print(f"tick {tick:4d}  spread {spread:.5f}{flag}")
    print(f"rewind horizon: {horizon if horizon is not None else f'> {args.max_ticks} (not reached)'}")

    Path(args.out).write_text(json.dumps({
        "nu": dataset.nu, "k": args.k, "threshold": args.threshold,
        "spreads": spreads, "horizon": horizon,
    }, indent=2))


if __name__ == "__main__":
    main()
