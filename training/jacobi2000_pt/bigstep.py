"""Roadmap item 3: forward time with bigger steps.

the design notes, section 15 (`Multi-horizon operators`): "A solver must step; a
learned operator need not... Horizon as input: dt is already conditioning
-- train on (state_t, state_{t+k}) with random k... Measure: error vs
horizon for direct k-step prediction against k one-step applications; wall
clock per physical time. If direct-8 ~= 8x single-step, we do 8x less work
per unit of physical time -- a claim a solver cannot make."

Minimal form: fixed k=8 (not random k as input -- that is the
natural next rung, flagged in the run doc), same `Jacobi2000` architecture
and mask as the existing model, trained so one `model.step()` call jumps 8
solver ticks instead of 1. Compared against the existing curriculum
checkpoint (`exports/curriculum/s6-mass-energy-proj`) applied 8 times, at
the same total physical horizon, on both accuracy (correlation against the
dataset's own ground truth) and wall clock.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import META_DIM, Normalizer, TrajectoryDataset
from .export import export as export_checkpoint
from .metrics import correlation, rmse
from .model import Jacobi2000, stencil_mask
from .parity import load_from_export
from .project import Denorm, energy, project_energy, project_mass


def fit_direct_k(model: Jacobi2000, dataset: TrajectoryDataset, norm: Normalizer, k: int,
                  steps: int, lr: float, batch: int, mask, device: torch.device,
                  project: bool, project_energy_on: bool, r_max: float,
                  on_log=lambda row: None, seed: int = 0):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    denorm = Denorm(mean=norm.mean, std=norm.std, depth=norm.depth)
    meta_vec = torch.from_numpy(dataset.meta_vector()).to(device)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    for step in range(steps):
        picks = dataset.sample_windows(rng, batch, k)
        windows = np.stack([dataset.window(norm, ti, start, k) for ti, start in picks], axis=1)
        window = torch.from_numpy(windows).to(device)  # [k+1, b, 3, ny, nx]
        b = window.shape[1]
        meta = meta_vec.unsqueeze(0).expand(b, META_DIM).contiguous()

        x = window[0]
        mass0 = x[:, 0:1].mean(dim=(2, 3)).reshape(b, 1) if project else None
        prev_e = energy(x, denorm, dataset.g) if project_energy_on else None

        pred = model.step(x, meta, mask)  # direct jump of k ticks in one call
        if project:
            pred = project_mass(pred, mass0)
        if project_energy_on:
            pred = project_energy(pred, prev_e, denorm, dataset.g, r_max)

        target = window[k]
        loss = (pred - target).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(steps // 20, 1) == 0 or step == 0:
            on_log({"step": step + 1, "loss": float(loss.item()), "secs": time.time() - t0})
    return model


def compare(direct_model: Jacobi2000, baseline_model: Jacobi2000, dataset: TrajectoryDataset,
            norm: Normalizer, k: int, n_repeats: int, mask, device: torch.device,
            n_traj: int):
    """direct_model.step() once vs baseline_model.step() applied k times, same
    total physical horizon = k * n_repeats ticks."""
    meta_vec = torch.from_numpy(dataset.meta_vector()).to(device)
    horizon = k * n_repeats
    rows = []

    direct_model.eval()
    baseline_model.eval()
    with torch.no_grad():
        for ti in range(min(n_traj, len(dataset.trajectories))):
            window = dataset.window(norm, ti, 0, horizon)
            x0 = torch.from_numpy(window[0:1]).to(device)
            meta = meta_vec.unsqueeze(0)

            x = x0
            t0 = time.time()
            for _ in range(n_repeats):
                x = direct_model.step(x, meta, mask)
            direct_secs = time.time() - t0
            pred_direct = x.cpu().numpy().ravel()

            x = x0
            t0 = time.time()
            for _ in range(n_repeats * k):
                x = baseline_model.step(x, meta, mask)
            baseline_secs = time.time() - t0
            pred_baseline = x.cpu().numpy().ravel()

            truth = window[horizon].ravel()
            rows.append({
                "traj": ti,
                "direct_corr": correlation(pred_direct, truth),
                "direct_rmse": rmse(pred_direct, truth),
                "direct_secs": direct_secs,
                "baseline_corr": correlation(pred_baseline, truth),
                "baseline_rmse": rmse(pred_baseline, truth),
                "baseline_secs": baseline_secs,
            })
    return rows, horizon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", required=True, help="export dir of the model to compare against (e.g. curriculum s6)")
    ap.add_argument("--k", type=int, default=8, help="fixed horizon this model jumps in one step")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--project-energy", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--r-max", type=float, default=1.0001)
    ap.add_argument("--init", default=None)
    ap.add_argument("--n-repeats", type=int, default=5, help="how many k-jumps for the comparison horizon")
    ap.add_argument("--eval-trajectories", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0, help="controls minibatch sampling order during training")
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrajectoryDataset(args.dataset)
    norm = dataset.fit_normalizer()

    baseline_model, base_cfg = load_from_export(args.baseline)
    baseline_model.to(device)
    mask_cfg = base_cfg["mask"]
    mask = stencil_mask(baseline_model.cfg, mask_cfg["Stencil"]["radius"], mask_cfg["Stencil"]["dilation"], device) \
        if isinstance(mask_cfg, dict) and "Stencil" in mask_cfg else None

    if args.init is not None:
        model, _ = load_from_export(args.init)
    else:
        model, _ = load_from_export(args.baseline)  # warm start from the same architecture/weights
    print(f"direct-{args.k} model: {model.num_params()} params, device {device}, seed {args.seed}")

    log_rows = []

    def on_log(row):
        log_rows.append(row)
        print(f"step {row['step']:6d}  loss {row['loss']:.6f}  {row['secs']:.1f}s", flush=True)

    t0 = time.time()
    fit_direct_k(model, dataset, norm, args.k, args.steps, args.lr, args.batch, mask, device,
                 args.project, args.project_energy, args.r_max, on_log, seed=args.seed)
    wall = time.time() - t0
    print(f"training wall time: {wall:.1f}s")

    export_checkpoint(
        model, norm, dx=dataset.dx, dt=dataset.dt, g=dataset.g, depth=dataset.depth,
        nu=dataset.nu, mask_radius=mask_cfg["Stencil"]["radius"] if isinstance(mask_cfg, dict) else None,
        mask_dilation=mask_cfg["Stencil"]["dilation"] if isinstance(mask_cfg, dict) else None,
        energy_r_max=args.r_max, out_dir=args.out,
    )

    rows, horizon = compare(model, baseline_model, dataset, norm, args.k, args.n_repeats, mask, device,
                             args.eval_trajectories)
    mean_direct_corr = float(np.mean([r["direct_corr"] for r in rows]))
    mean_baseline_corr = float(np.mean([r["baseline_corr"] for r in rows]))
    total_direct_secs = float(np.sum([r["direct_secs"] for r in rows]))
    total_baseline_secs = float(np.sum([r["baseline_secs"] for r in rows]))
    print(f"horizon {horizon} ticks ({args.n_repeats} x direct-{args.k} vs {args.n_repeats * args.k} x baseline-1)")
    print(f"direct:   mean corr {mean_direct_corr:.4f}  total wall {total_direct_secs:.4f}s")
    print(f"baseline: mean corr {mean_baseline_corr:.4f}  total wall {total_baseline_secs:.4f}s")

    if args.log_file:
        Path(args.log_file).write_text(json.dumps({
            "rows": log_rows, "compare": rows, "horizon": horizon,
            "wall_seconds": wall, "params": model.num_params(),
            "mean_direct_corr": mean_direct_corr, "mean_baseline_corr": mean_baseline_corr,
            "total_direct_secs": total_direct_secs, "total_baseline_secs": total_baseline_secs,
        }, indent=2))


if __name__ == "__main__":
    main()
