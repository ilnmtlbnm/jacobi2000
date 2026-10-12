"""Roadmap item 2: mode models.

CONCEPT.md does not define "mode models" anywhere (grepped; the word "mode"
only appears for step/block/control/rewind *modes*, a different sense).
Interpretation written before implementing, per the brief: a "mode model"
here means a spectral operator -- decompose the field into its leading 2D
Fourier modes (a truncated `rfft2`), and learn the dynamics of the retained
mode coefficients directly, instead of the patch-transformer's pixel/patch
tokens. This is the concrete, minimal reading of CONCEPT.md section 2's
"spectral term" and the prior-art line "FNO ... (mesh operators)" -- a small
Koopman/DMD-style operator on a truncated spectral basis, testing whether
shallow water's dominant (long-wavelength) modes evolve with a much smaller
model than the full patch-transformer needs.

Model: `rfft2(h, u, v)` truncated to the lowest `k x k` wavenumbers per
channel, flattened to a real vector (concat real/imag), one linear-plus-GELU
residual block predicts the increment in mode space, `irfft2` reconstructs
the field. No grid attention, no patches, no mask.

This is a new architecture, not `Jacobi2000`, so it has none of the patch-transformer's
tensors, so `crates/engine` cannot load it, and it is scored here in Python only, against held-out trajectories from
the same dataset dump `training/jacobi2000_pt/curriculum.py` trains on
(`metrics.py`).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import Normalizer, TrajectoryDataset
from .metrics import correlation, rmse


class ModeModel(nn.Module):
    def __init__(self, ny: int, nx: int, k: int, channels: int = 3, hidden: int = 256):
        super().__init__()
        self.ny, self.nx, self.k, self.channels = ny, nx, k, channels
        # rfft2 of an (ny, nx) real field has shape (ny, nx//2+1) complex;
        # keep only the lowest positive-frequency rows/cols (rows 0..ky-1,
        # cols 0..kx-1). Asymmetric (misses the negative-y-frequency half of
        # the same physical wavelength) but simple and bug-free -- still a
        # low-pass truncation, which is the property this experiment tests.
        self.kx = min(k, nx // 2 + 1)
        self.ky = min(k, ny)
        dim = channels * self.kx * self.ky * 2  # real + imag
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def to_modes(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `[b, c, ny, nx]` -> `[b, dim]` real vector of truncated
        rfft2 coefficients."""
        b = x.shape[0]
        f = torch.fft.rfft2(x, norm="ortho")  # [b, c, ny, nx//2+1] complex
        f = f[:, :, : self.ky, : self.kx]
        return torch.view_as_real(f).reshape(b, -1)

    def from_modes(self, m: torch.Tensor) -> torch.Tensor:
        b = m.shape[0]
        c = self.channels
        f = m.reshape(b, c, self.ky, self.kx, 2)
        f = torch.view_as_complex(f.contiguous())
        full = torch.zeros(b, c, self.ny, self.nx // 2 + 1, dtype=f.dtype, device=f.device)
        full[:, :, : self.ky, : self.kx] = f
        return torch.fft.irfft2(full, s=(self.ny, self.nx), norm="ortho")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.to_modes(x)
        dm = self.net(m)
        dx = self.from_modes(dm)
        return x + dx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def fit(model: ModeModel, dataset: TrajectoryDataset, norm: Normalizer, steps: int, lr: float,
        rollout: int, batch: int, device: torch.device, on_log=lambda row: None):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(steps):
        window = dataset.batch(rng, norm, batch, rollout, device)  # [rollout+1, b, 3, ny, nx]
        x = window[0]
        loss = torch.zeros((), device=device)
        for k in range(rollout):
            x = model(x)
            loss = loss + (x - window[k + 1]).pow(2).mean()
        loss = loss / rollout

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(steps // 20, 1) == 0 or step == 0:
            on_log({"step": step + 1, "loss": float(loss.item()), "secs": time.time() - t0})
    return model


def evaluate(model: ModeModel, dataset: TrajectoryDataset, norm: Normalizer, horizons: list[int],
             n_traj: int, device: torch.device):
    model.eval()
    rows = []
    with torch.no_grad():
        for ti in range(min(n_traj, len(dataset.trajectories))):
            window = dataset.window(norm, ti, 0, max(horizons))
            x = torch.from_numpy(window[0:1]).to(device)
            for tick in range(1, max(horizons) + 1):
                x = model(x)
                if tick in horizons:
                    pred = x.cpu().numpy().ravel()
                    truth = window[tick].ravel()
                    rows.append({
                        "traj": ti, "tick": tick,
                        "corr": correlation(pred, truth),
                        "rmse": rmse(pred, truth),
                    })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollout", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k", type=int, default=8, help="retained wavenumbers per axis")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--horizons", default="10,25,50", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--eval-trajectories", type=int, default=16)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrajectoryDataset(args.dataset)
    norm = dataset.fit_normalizer()

    model = ModeModel(dataset.ny, dataset.nx, args.k, hidden=args.hidden)
    print(f"model: {model.num_params()} params, device {device}, kx={model.kx} ky={model.ky}")

    log_rows = []

    def on_log(row):
        log_rows.append(row)
        print(f"step {row['step']:6d}  loss {row['loss']:.6f}  {row['secs']:.1f}s", flush=True)

    t0 = time.time()
    fit(model, dataset, norm, args.steps, args.lr, args.rollout, args.batch, device, on_log)
    wall = time.time() - t0
    print(f"training wall time: {wall:.1f}s")

    eval_rows = evaluate(model, dataset, norm, args.horizons, args.eval_trajectories, device)
    by_horizon = {}
    for row in eval_rows:
        by_horizon.setdefault(row["tick"], []).append(row)
    for tick, rows in sorted(by_horizon.items()):
        mean_corr = float(np.mean([r["corr"] for r in rows]))
        mean_rmse = float(np.mean([r["rmse"] for r in rows]))
        print(f"horizon {tick:4d}  mean corr {mean_corr:.4f}  mean rmse {mean_rmse:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    if args.log_file:
        Path(args.log_file).write_text(
            json.dumps({"rows": log_rows, "eval": eval_rows, "wall_seconds": wall, "params": model.num_params()}, indent=2)
        )


if __name__ == "__main__":
    main()
