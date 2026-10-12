"""Hard constraints (`crates/engine/src/project.rs` does the same at inference), applied
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


def _anomaly_parts(x: torch.Tensor, d: Denorm):
    m = mass(x)
    eta = (x[:, 0:1] - m.reshape(-1, 1, 1, 1)) * d.std[0]
    hbar = m * d.std[0] + (d.mean[0] + d.depth)
    u = x[:, 1:2] * d.std[1] + d.mean[1]
    v = x[:, 2:3] * d.std[2] + d.mean[2]
    return hbar, eta, u, v


def _face_outflow(f: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    fx = (f[..., :-1] + f[..., 1:]) * (u[..., :-1] + u[..., 1:]) * 0.25
    fy = (f[..., :-1, :] + f[..., 1:, :]) * (v[..., :-1, :] + v[..., 1:, :]) * 0.25
    fx = torch.nn.functional.pad(fx, (1, 1))
    fy = torch.nn.functional.pad(fy, (0, 0, 1, 1))
    return fx[..., 1:] - fx[..., :-1] + fy[..., 1:, :] - fy[..., :-1, :]


def _excess_terms(x: torch.Tensor, d: Denorm, g: float, dt_dx: float):
    b = x.shape[0]
    hbar, eta, u, v = _anomaly_parts(x, d)

    def total(t):
        return t.reshape(b, -1).sum(dim=1, keepdim=True)

    speed2 = u.pow(2) + v.pow(2)
    a = total(eta.pow(2)) * (0.5 * g) + hbar * total(speed2) * 0.5
    q = total(eta * speed2) * 0.5
    if dt_dx != 0.0:
        k = -0.5 * g * dt_dx
        a = a + hbar * total(eta * _face_outflow(torch.ones_like(eta), u, v)) * k
        q = q + total(eta * _face_outflow(eta, u, v)) * k
    return a, q


def excess_energy(x: torch.Tensor, d: Denorm, g: float, dt_dx: float) -> torch.Tensor:
    """Mirrors `excess_energy` in project.rs: `sum 1/2 g eta^2 + 1/2 h |u|^2
    - (dt/dx)(g/2) sum eta div(h u)` per batch member, `[b, 1]`. The cross
    term is the forward-backward solver's own; `dt_dx = 0` gives the plain
    excess energy."""
    a, q = _excess_terms(x, d, g, dt_dx)
    return a + q


def project_excess_energy(
    x: torch.Tensor,
    prev_excess: torch.Tensor,
    d: Denorm,
    g: float,
    dt_dx: float,
    r_min: float,
    r_max: float,
) -> torch.Tensor:
    """Mirrors `project_excess_energy` in project.rs: scale the fluctuation
    `(eta, u, v)` by one detached `s` per member so that `prev * r_min <=
    excess <= prev * r_max`; mass exact, members inside the band untouched.
    `r_min = 0` is the upper-only form."""
    b = x.shape[0]
    prev = prev_excess.detach()
    upper = prev * r_max
    lower = prev * r_min
    with torch.no_grad():
        a, q = _excess_terms(x, d, g, dt_dx)
        e = a + q
        active = (e > upper) | (e < lower)
        target = torch.minimum(torch.maximum(e, lower * (1.0 + 1e-5)), upper * (1.0 - 1e-5))
        s = (target / e.clamp_min(1e-30)).clamp_min(0.0).sqrt()
        for _ in range(2):
            f = s.pow(2) * a + s.pow(3) * q - target
            df = s * a * 2.0 + s.pow(2) * q * 3.0
            s = s - f / df.clamp_min(1e-30)
        s = s.clamp(0.0, 2.0).reshape(b, 1, 1, 1)
    m = mass(x).reshape(b, 1, 1, 1)
    h = (x[:, 0:1] - m) * s + m
    u = ((x[:, 1:2] * d.std[1] + d.mean[1]) * s - d.mean[1]) / d.std[1]
    v = ((x[:, 2:3] * d.std[2] + d.mean[2]) * s - d.mean[2]) / d.std[2]
    projected = torch.cat([h, u, v], dim=1)
    return torch.where(active.reshape(b, 1, 1, 1), projected, x)
