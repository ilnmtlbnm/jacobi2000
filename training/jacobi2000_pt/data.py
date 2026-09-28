"""Reads the trajectory files written once by
`jacobi dump-dataset --out <dir>` (crates/train/src/data.rs::dump_trajectories):
one `traj-NNNNN.json` header and one `traj-NNNNN.f32` raw-field file per
trajectory, plus a `manifest.json` listing every header.

Physics never happens here: the Rust solver is the one source of it. This
module only reshapes what it already computed into normalized training
windows, and a `Normalizer` fit the same way `train::data::Generator`'s
`fit_normalizer` is (mean/std over every dumped physical value), so a step-
mode window here is what `crates/train/src/fit.rs::fit` would have handed a
Burn model.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

META_DIM = 5


@dataclass
class Normalizer:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    depth: float

    def to_dict(self):
        return {"mean": list(self.mean), "std": list(self.std), "depth": self.depth}


class TrajectoryDataset:
    """Loads every dumped trajectory into memory (a few thousand trajectories
    of a few hundred ticks at 64x64x3 floats is tens to low hundreds of MB,
    same order of magnitude the run doc for the Rust generator quotes)."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.headers = manifest
        self.trajectories = []  # list of (h, u, v) each [ticks, ny, nx]
        for h in manifest:
            nx, ny, ticks = h["nx"], h["ny"], h["ticks"]
            raw = np.fromfile(self.dir / f"traj-{h['index']:05d}.f32", dtype="<f4")
            n = nx * ny
            expected = ticks * n * 3
            assert raw.size == expected, f"traj {h['index']}: {raw.size} floats, expected {expected}"
            hh = raw[: ticks * n].reshape(ticks, ny, nx)
            uu = raw[ticks * n : 2 * ticks * n].reshape(ticks, ny, nx)
            vv = raw[2 * ticks * n :].reshape(ticks, ny, nx)
            self.trajectories.append((hh, uu, vv))
        first = manifest[0]
        self.nx, self.ny = first["nx"], first["ny"]
        self.dx, self.dt, self.g, self.depth, self.nu = (
            first["dx"], first["dt"], first["g"], first["depth"], first["nu"],
        )

    def meta_vector(self) -> np.ndarray:
        """`[ln dx, ln dt, ln g, ln depth, ln(nu + 1e-6)]`, matching
        `train::data::meta_of` exactly. Constant across this dataset (one set
        of solver params), unlike the per-trajectory drop/(cx,cy) fields."""
        return np.array(
            [
                np.log(self.dx),
                np.log(self.dt),
                np.log(self.g),
                np.log(self.depth),
                np.log(self.nu + 1e-6),
            ],
            dtype=np.float32,
        )

    def fit_normalizer(self) -> Normalizer:
        sums = np.zeros(3, dtype=np.float64)
        sqs = np.zeros(3, dtype=np.float64)
        count = 0
        for hh, uu, vv in self.trajectories:
            for c, field in enumerate((hh - self.depth, uu, vv)):
                sums[c] += field.sum(dtype=np.float64)
                sqs[c] += (field.astype(np.float64) ** 2).sum()
            count += hh.size
        mean = sums / count
        var = np.maximum(sqs / count - mean ** 2, 0.0)
        std = np.maximum(np.sqrt(var), 1e-8)
        return Normalizer(tuple(mean.tolist()), tuple(std.tolist()), self.depth)

    def sample_windows(self, rng: np.random.Generator, batch: int, rollout: int) -> list[tuple[int, int]]:
        """`(trajectory index, start tick)` pairs with `rollout + 1` frames
        available from `start`."""
        out = []
        n_traj = len(self.trajectories)
        for _ in range(batch):
            for _ in range(200):  # bounded retries against very short trajectories
                ti = int(rng.integers(0, n_traj))
                ticks = self.trajectories[ti][0].shape[0]
                if ticks > rollout:
                    start = int(rng.integers(0, ticks - rollout))
                    out.append((ti, start))
                    break
            else:
                raise RuntimeError("no trajectory long enough for this rollout")
        return out

    def window(self, norm: Normalizer, ti: int, start: int, rollout: int) -> np.ndarray:
        """`[rollout + 1, 3, ny, nx]`, normalized."""
        hh, uu, vv = self.trajectories[ti]
        sl = slice(start, start + rollout + 1)
        h = (hh[sl] - self.depth - norm.mean[0]) / norm.std[0]
        u = (uu[sl] - norm.mean[1]) / norm.std[1]
        v = (vv[sl] - norm.mean[2]) / norm.std[2]
        return np.stack([h, u, v], axis=1).astype(np.float32)

    def batch(self, rng: np.random.Generator, norm: Normalizer, batch: int, rollout: int, device) -> torch.Tensor:
        """`[rollout + 1, batch, 3, ny, nx]`."""
        picks = self.sample_windows(rng, batch, rollout)
        windows = [self.window(norm, ti, start, rollout) for ti, start in picks]
        stacked = np.stack(windows, axis=1)  # [rollout+1, batch, 3, ny, nx]
        return torch.from_numpy(stacked).to(device)
