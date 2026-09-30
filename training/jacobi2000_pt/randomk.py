"""Follow-up to roadmap item 3: random-k conditioning, per CONCEPT.md
section 15's actual proposal ("Horizon as input: dt is already conditioning
-- train on (state_t, state_{t+k}) with random k in the metadata; one model
answers 'next', '10th', '40th' from the same weights"), rather than
`bigstep.py`'s single fixed k=8.

Meta_dim is extended from 5 to 6 (`log(k)` appended to the existing
`[ln dx, ln dt, ln g, ln depth, ln(nu+eps)]` vector) -- a deliberate,
disclosed deviation from `Jacobi2000Config`'s fixed meta_dim=5, so this
checkpoint (like `bigstep.py`'s) cannot be imported into Burn without a
config-schema change on the Rust side; scored in Python only, same as the
other new-architecture-shaped experiments tonight.

Warm start: the body (embed/blocks/norm/head) is loaded from the baseline
export unchanged; only `meta.weight` is padded with one freshly-initialized
row for the new `log(k)` input, so the horizon-conditioned run starts from
the same converged body as `bigstep.py`'s fixed-k8 run, not from scratch.
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
from .model import Jacobi2000, Jacobi2000Config, stencil_mask
from .project import Denorm, energy, project_energy, project_mass


def load_from_export_padded_meta(export_dir: str, extra_dims: int = 1):
    from safetensors.torch import load_file

    export_dir = Path(export_dir)
    config = json.loads((export_dir / "config.json").read_text())
    base_meta_dim = config["meta_dim"]
    cfg = Jacobi2000Config(
        nx=config["nx"], ny=config["ny"], patch=config["patch"], channels=config["channels"],
        dim=config["dim"], layers=config["layers"], heads=config["heads"],
        ff_mult=config["ff_mult"], meta_dim=base_meta_dim + extra_dims,
    )
    model = Jacobi2000(cfg)
    tensors = load_file(str(export_dir / "model.safetensors"))
    old_w = tensors["meta.weight"]  # [base_meta_dim, dim]
    k = 1.0 / (cfg.meta_dim ** 0.5)
    pad = torch.empty(extra_dims, old_w.shape[1]).uniform_(-k, k)
    tensors["meta.weight"] = torch.cat([old_w, pad], dim=0)
    missing, unexpected = model.load_state_dict(tensors, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model, config, base_meta_dim


def fit_random_k(model: Jacobi2000, dataset: TrajectoryDataset, norm: Normalizer, ks: list[int],
                  steps: int, lr: float, batch: int, mask, device: torch.device,
                  project: bool, project_energy_on: bool, r_max: float,
                  on_log=lambda row: None):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    denorm = Denorm(mean=norm.mean, std=norm.std, depth=norm.depth)
    base_meta = torch.from_numpy(dataset.meta_vector()).to(device)
    rng = np.random.default_rng(0)

    t0 = time.time()
    for step in range(steps):
        k = int(rng.choice(ks))
        picks = dataset.sample_windows(rng, batch, k)
        windows = np.stack([dataset.window(norm, ti, start, k) for ti, start in picks], axis=1)
        window = torch.from_numpy(windows).to(device)  # [k+1, b, 3, ny, nx]
        b = window.shape[1]
        meta = torch.cat([base_meta, torch.tensor([np.log(k)], device=device, dtype=base_meta.dtype)])
        meta = meta.unsqueeze(0).expand(b, META_DIM + 1).contiguous()

        x = window[0]
        mass0 = x[:, 0:1].mean(dim=(2, 3)).reshape(b, 1) if project else None
        prev_e = energy(x, denorm, dataset.g) if project_energy_on else None

        pred = model.step(x, meta, mask)
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
            on_log({"step": step + 1, "loss": float(loss.item()), "secs": time.time() - t0, "k": k})
    return model


def compare(random_k_model: Jacobi2000, baseline_model: Jacobi2000, dataset: TrajectoryDataset,
            norm: Normalizer, k: int, n_repeats: int, mask, device: torch.device, n_traj: int):
    """random_k_model conditioned at horizon k, stepped n_repeats times, vs.
    baseline_model (1 tick/call) stepped k*n_repeats times, same physical
    horizon -- identical shape to bigstep.py's `compare`, for a direct
    fixed-k8-vs-random-k comparison at the same eval horizon."""
    base_meta = torch.from_numpy(dataset.meta_vector()).to(device)
    meta_k = torch.cat([base_meta, torch.tensor([np.log(k)], device=device, dtype=base_meta.dtype)]).unsqueeze(0)
    horizon = k * n_repeats
    rows = []

    random_k_model.eval()
    baseline_model.eval()
    base_meta_1 = base_meta.unsqueeze(0)
    with torch.no_grad():
        for ti in range(min(n_traj, len(dataset.trajectories))):
            window = dataset.window(norm, ti, 0, horizon)
            x0 = torch.from_numpy(window[0:1]).to(device)

            x = x0
            t0 = time.time()
            for _ in range(n_repeats):
                x = random_k_model.step(x, meta_k, mask)
            rk_secs = time.time() - t0
            pred_rk = x.cpu().numpy().ravel()

            x = x0
            t0 = time.time()
            for _ in range(n_repeats * k):
                x = baseline_model.step(x, base_meta_1, mask)
            baseline_secs = time.time() - t0
            pred_baseline = x.cpu().numpy().ravel()

            truth = window[horizon].ravel()
            rows.append({
                "traj": ti,
                "randomk_corr": correlation(pred_rk, truth),
                "randomk_rmse": rmse(pred_rk, truth),
                "randomk_secs": rk_secs,
                "baseline_corr": correlation(pred_baseline, truth),
                "baseline_rmse": rmse(pred_baseline, truth),
                "baseline_secs": baseline_secs,
            })
    return rows, horizon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", required=True, help="export dir to warm-start the body from and compare against")
    ap.add_argument("--ks", default="1,2,4,8", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--eval-k", type=int, default=8, help="horizon to condition on for the comparison")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--project-energy", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--r-max", type=float, default=1.0001)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--eval-trajectories", type=int, default=16)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrajectoryDataset(args.dataset)
    norm = dataset.fit_normalizer()

    from .parity import load_from_export
    baseline_model, base_cfg = load_from_export(args.baseline)
    baseline_model.to(device)
    mask_cfg = base_cfg["mask"]
    mask = stencil_mask(baseline_model.cfg, mask_cfg["Stencil"]["radius"], mask_cfg["Stencil"]["dilation"], device) \
        if isinstance(mask_cfg, dict) and "Stencil" in mask_cfg else None

    model, _, base_meta_dim = load_from_export_padded_meta(args.baseline, extra_dims=1)
    print(f"random-k model: {model.num_params()} params, device {device}, "
          f"meta_dim {base_meta_dim} -> {model.cfg.meta_dim}, ks={args.ks}")

    log_rows = []

    def on_log(row):
        log_rows.append(row)
        print(f"step {row['step']:6d}  k {row['k']}  loss {row['loss']:.6f}  {row['secs']:.1f}s", flush=True)

    t0 = time.time()
    fit_random_k(model, dataset, norm, args.ks, args.steps, args.lr, args.batch, mask, device,
                 args.project, args.project_energy, args.r_max, on_log)
    wall = time.time() - t0
    print(f"training wall time: {wall:.1f}s")

    export_checkpoint(
        model, norm, dx=dataset.dx, dt=dataset.dt, g=dataset.g, depth=dataset.depth,
        nu=dataset.nu, mask_radius=mask_cfg["Stencil"]["radius"] if isinstance(mask_cfg, dict) else None,
        mask_dilation=mask_cfg["Stencil"]["dilation"] if isinstance(mask_cfg, dict) else None,
        energy_r_max=args.r_max, out_dir=args.out,
    )

    rows, horizon = compare(model, baseline_model, dataset, norm, args.eval_k, args.n_repeats, mask, device,
                             args.eval_trajectories)
    mean_rk_corr = float(np.mean([r["randomk_corr"] for r in rows]))
    mean_baseline_corr = float(np.mean([r["baseline_corr"] for r in rows]))
    mean_rk_rmse = float(np.mean([r["randomk_rmse"] for r in rows]))
    mean_baseline_rmse = float(np.mean([r["baseline_rmse"] for r in rows]))
    total_rk_secs = float(np.sum([r["randomk_secs"] for r in rows]))
    total_baseline_secs = float(np.sum([r["baseline_secs"] for r in rows]))
    print(f"horizon {horizon} ticks ({args.n_repeats} x random-k(k={args.eval_k}) vs {args.n_repeats * args.eval_k} x baseline-1)")
    print(f"random-k: mean corr {mean_rk_corr:.4f}  mean rmse {mean_rk_rmse:.4f}  total wall {total_rk_secs:.4f}s")
    print(f"baseline: mean corr {mean_baseline_corr:.4f}  mean rmse {mean_baseline_rmse:.4f}  total wall {total_baseline_secs:.4f}s")

    if args.log_file:
        Path(args.log_file).write_text(json.dumps({
            "rows": log_rows, "compare": rows, "horizon": horizon,
            "wall_seconds": wall, "params": model.num_params(),
            "mean_randomk_corr": mean_rk_corr, "mean_baseline_corr": mean_baseline_corr,
            "mean_randomk_rmse": mean_rk_rmse, "mean_baseline_rmse": mean_baseline_rmse,
            "total_randomk_secs": total_rk_secs, "total_baseline_secs": total_baseline_secs,
        }, indent=2))


if __name__ == "__main__":
    main()
