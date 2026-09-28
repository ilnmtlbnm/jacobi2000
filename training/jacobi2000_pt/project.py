"""Mirrors crates/jacobi2000/src/project.rs exactly: hard constraints applied
after the model proposes, not soft loss terms."""

from dataclasses import dataclass

import torch


@dataclass
class Denorm:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    depth: float


def _potential_kinetic(x: torch.Tensor, d: Denorm, g: float):
    b = x.shape[0]
    n = x.shape[2] * x.shape[3]
    h = x[:, 0:1] * d.std[0] + (d.mean[0] + d.depth)
    u = x[:, 1:2] * d.std[1] + d.mean[1]
    v = x[:, 2:3] * d.std[2] + d.mean[2]
    potential = (h.pow(2) * (0.5 * g)).reshape(b, n).sum(dim=1, keepdim=True)
    kinetic = (h * (u.pow(2) + v.pow(2)) * 0.5).reshape(b, n).sum(dim=1, keepdim=True)
    return potential, kinetic


def energy(x: torch.Tensor, d: Denorm, g: float) -> torch.Tensor:
    potential, kinetic = _potential_kinetic(x, d, g)
    return potential + kinetic


def project_energy(x: torch.Tensor, prev_energy: torch.Tensor, d: Denorm, g: float, r_max: float) -> torch.Tensor:
    b = x.shape[0]
    bound = prev_energy * r_max
    potential, kinetic = _potential_kinetic(x, d, g)
    budget = (bound.detach() - potential.detach()).clamp_min(0.0)
    scale = (budget / kinetic.detach().clamp_min(1e-12)).sqrt().clamp(0.0, 1.0)
    scale = scale.reshape(b, 1, 1, 1)

    u = x[:, 1:2] * d.std[1] + d.mean[1]
    v = x[:, 2:3] * d.std[2] + d.mean[2]
    u = (u * scale - d.mean[1]) / d.std[1]
    v = (v * scale - d.mean[2]) / d.std[2]
    x = x.clone()
    x[:, 1:2] = u
    x[:, 2:3] = v
    return x


def mass(x: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    n = x.shape[2] * x.shape[3]
    return x[:, 0:1].reshape(b, n).mean(dim=1, keepdim=True)


def project_mass(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    delta = (target - mass(x)).reshape(b, 1, 1, 1)
    x = x.clone()
    x[:, 0:1] = x[:, 0:1] + delta
    return x
