"""Excess-energy band (project.py), the same guarantees the Rust tests in
crates/engine/src/project.rs checks.
Run from training/: `python -m unittest discover tests`. The solver test
needs the release `dump-dataset` binary
(`cargo build --release -p engine-cli --bin dump-dataset`) and fails if it
is missing."""

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from jacobi2000_pt.data import TrajectoryDataset
from jacobi2000_pt.project import Denorm, excess_energy, mass, project_excess_energy

DT = 0.0903
G = 9.81
D20 = Denorm(mean=(0.0008, 0.0004, 0.0005), std=(0.0063, 0.0138, 0.0135), depth=1.0)
REPO = Path(__file__).resolve().parents[2]


def wavy(b: int) -> torch.Tensor:
    n = 3 * 8 * 8
    i = torch.arange(b * n, dtype=torch.float32)
    amp = 1.0 + (i // n)
    return (amp * (torch.sin(i * 0.37) + 0.5 * torch.cos(i * 1.91))).reshape(b, 3, 8, 8)


class ExcessBand(unittest.TestCase):
    def test_never_leaves_the_band_and_keeps_mass(self):
        x = wavy(6)
        e = excess_energy(x, D20, G, DT)
        r_min, r_max = 0.9999, 1.0004
        # prev puts members far below, just below, inside, just above, far above the band
        prev = e * torch.tensor([[2.0], [1.0002], [1.0], [0.9995], [0.5], [0.0]])
        y = project_excess_energy(x, prev, D20, G, DT, r_min, r_max)
        e1 = excess_energy(y, D20, G, DT)
        for i in range(6):
            self.assertGreaterEqual(e1[i].item(), prev[i].item() * r_min, f"member {i}")
            self.assertLessEqual(e1[i].item(), prev[i].item() * r_max + 1e-12, f"member {i}")
        self.assertLess((mass(y) - mass(x)).abs().max().item(), 1e-6)

    def test_upper_only_form_with_r_min_zero(self):
        x = wavy(2)
        e = excess_energy(x, D20, G, DT)
        prev = e * torch.tensor([[2.0], [0.9]])
        y = project_excess_energy(x, prev, D20, G, DT, 0.0, 1.0)
        self.assertTrue(torch.equal(y[0], x[0]))
        self.assertLessEqual(excess_energy(y, D20, G, DT)[1].item(), prev[1].item())

    def test_identity_inside_the_band(self):
        x = wavy(3)
        e = excess_energy(x, D20, G, DT)
        y = project_excess_energy(x, e.clone(), D20, G, DT, 0.9999, 1.0004)
        self.assertTrue(torch.equal(y, x))

    def test_gradient_finite_at_zero_budget_and_when_lifting(self):
        x = wavy(2).requires_grad_(True)
        e = excess_energy(x.detach(), D20, G, DT)
        prev = torch.cat([torch.zeros(1, 1), e[1:] * 2.0])
        y = project_excess_energy(x, prev, D20, G, DT, 0.9999, 1.0004)
        y.pow(2).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_solver_trajectory_untouched_by_its_measured_band(self):
        exe = REPO / "target" / "release" / "dump-dataset"
        self.assertTrue(exe.exists(), f"{exe} missing: cargo build --release -p engine-cli --bin dump-dataset")
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [str(exe), "--pool", "2", "--nu", "0.01", "--seed", "3", "--out", tmp],
                check=True, capture_output=True,
            )
            ds = TrajectoryDataset(tmp)
            norm = ds.fit_normalizer()
            dn = Denorm(mean=norm.mean, std=norm.std, depth=norm.depth)
            dt_dx = ds.dt / ds.dx
            # nu = 0.01, 39,411 solver ticks (dump seed 11): per-tick ratio in [0.9996733, 1.0002059]
            r_min, r_max = 0.99965, 1.00022
            checked = 0
            for ti, (hh, _, _) in enumerate(ds.trajectories):
                x = torch.from_numpy(ds.window(norm, ti, 0, hh.shape[0] - 1))
                m = mass(x).ravel()
                # skip ticks where the generator injected a drop
                same = (m[1:] - m[:-1]).abs() * norm.std[0] < 1e-6
                e = excess_energy(x, dn, ds.g, dt_dx)
                x1, prev = x[1:][same], e[:-1][same]
                y = project_excess_energy(x1, prev, dn, ds.g, dt_dx, r_min, r_max)
                self.assertTrue(torch.equal(y, x1))
                checked += len(x1)
            self.assertGreater(checked, 100)


if __name__ == "__main__":
    unittest.main()
