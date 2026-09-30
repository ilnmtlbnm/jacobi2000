"""Roadmap item 1: a Lagrangian formulation, per CONCEPT.md section 9:

    "mask = neighbor list (rebuilt each tick, spatial hash); position = edge
    feature (relative offset, distance) -- mandatory; mass per particle
    constant by construction; momentum exact via antisymmetrized pairwise
    predictions (Newton III as a layer)."

This is a genuinely new model class, not the grid patch-transformer:
particles carry (x, y, vx, vy), a k-nearest-neighbor graph is rebuilt every
tick (`torch.cdist`, small particle counts here so O(n^2) is fine -- no
spatial hash needed at this scale), the edge feature is the relative offset
and its length, and a small MLP predicts one message per directed edge that
is applied as `+m_ij` to particle i and `-m_ij` to particle j (the Newton
III layer: the net predicted force is antisymmetric by construction, so
total momentum from pairwise interactions is exactly conserved regardless of
what the MLP weights are). Mass is a per-particle constant never touched by
the model, so it is conserved by construction too, trivially.

Data: `crates/solvers/examples/sph_dump.rs`, raw SPH particle trajectories
(position, velocity, mass every tick), not the splatted-to-grid fields
`crates/train/src/lagrangian.rs` uses for the existing grid model
(`artifacts/model-sph-smoke`, `docs/runs/2026-09-19-sph-smoke.md`).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ParticleTrajectory:
    def __init__(self, directory: str | Path, name: str):
        directory = Path(directory)
        header = json.loads((directory / f"traj-{name}.json").read_text())
        self.n = header["n"]
        self.ticks = header["ticks"]
        self.dt = header["dt_effective"]
        self.g = header["g"]
        raw = np.fromfile(directory / f"traj-{name}.f32", dtype="<f4")
        raw = raw.reshape(self.ticks, 5, self.n)
        self.x = raw[:, 0]
        self.y = raw[:, 1]
        self.vx = raw[:, 2]
        self.vy = raw[:, 3]
        self.mass = raw[:, 4]


class NewtonLayer(nn.Module):
    """One message-passing layer over the k-NN graph. The MLP sees the edge
    feature (relative offset, distance, both particles' velocities) and
    predicts a 2D message; particle i receives `+m`, particle j receives
    `-m` for the same edge -- Newton's third law as a layer, not a loss
    term."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, pos: torch.Tensor, vel: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """`pos`, `vel`: `[n, 2]`. `edges`: `[e, 2]` directed pairs (i, j)
        with i < j (each undirected pair listed once). Returns `[n, 2]` net
        message per particle."""
        i, j = edges[:, 0], edges[:, 1]
        rel = pos[i] - pos[j]
        dist = rel.norm(dim=-1, keepdim=True)
        feat = torch.cat([rel, dist, vel[i], vel[j]], dim=-1)
        m = self.mlp(feat)  # [e, 2]
        out = torch.zeros_like(pos)
        out.index_add_(0, i, m)
        out.index_add_(0, j, -m)
        return out


class LagrangianModel(nn.Module):
    def __init__(self, k: int = 8, layers: int = 2, hidden: int = 64):
        super().__init__()
        self.k = k
        self.layers = nn.ModuleList([NewtonLayer(hidden) for _ in range(layers)])

    def neighbors(self, pos: torch.Tensor) -> torch.Tensor:
        n = pos.shape[0]
        d = torch.cdist(pos, pos)
        d.fill_diagonal_(float("inf"))
        k = min(self.k, n - 1)
        idx = d.topk(k, largest=False).indices  # [n, k]
        i = torch.arange(n, device=pos.device).unsqueeze(1).expand(n, k).reshape(-1)
        j = idx.reshape(-1)
        pair = torch.stack([torch.minimum(i, j), torch.maximum(i, j)], dim=1)
        pair = torch.unique(pair, dim=0)
        return pair

    def forward(self, pos: torch.Tensor, vel: torch.Tensor, g: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """One tick. `pos`, `vel`: `[n, 2]`. Returns `(pos', vel')`."""
        edges = self.neighbors(pos)
        force = torch.zeros_like(pos)
        for layer in self.layers:
            force = force + layer(pos, vel, edges)
        force = force + torch.tensor([0.0, -g], device=pos.device)
        vel_new = vel + force * dt
        pos_new = pos + vel_new * dt
        return pos_new, vel_new

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_all(data_dir: str) -> list[ParticleTrajectory]:
    names = [l.strip() for l in (Path(data_dir) / "manifest.txt").read_text().splitlines() if l.strip()]
    return [ParticleTrajectory(data_dir, name) for name in names]


def fit(model: LagrangianModel, trajs: list[ParticleTrajectory], steps: int, lr: float,
        rollout: int, device: torch.device, on_log=lambda row: None):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(steps):
        traj = trajs[int(rng.integers(0, len(trajs)))]
        start = int(rng.integers(0, traj.ticks - rollout - 1))
        pos = torch.tensor(
            np.stack([traj.x[start], traj.y[start]], axis=-1), dtype=torch.float32, device=device
        )
        vel = torch.tensor(
            np.stack([traj.vx[start], traj.vy[start]], axis=-1), dtype=torch.float32, device=device
        )
        loss = torch.zeros((), device=device)
        for k in range(rollout):
            pos, vel = model(pos, vel, traj.g, traj.dt)
            target_pos = torch.tensor(
                np.stack([traj.x[start + k + 1], traj.y[start + k + 1]], axis=-1),
                dtype=torch.float32, device=device,
            )
            loss = loss + (pos - target_pos).pow(2).mean()
        loss = loss / rollout

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % max(steps // 20, 1) == 0 or step == 0:
            on_log({"step": step + 1, "loss": float(loss.item()), "secs": time.time() - t0})
    return model


def evaluate(model: LagrangianModel, trajs: list[ParticleTrajectory], horizons: list[int], device: torch.device):
    model.eval()
    rows = []
    with torch.no_grad():
        for traj in trajs:
            pos = torch.tensor(
                np.stack([traj.x[0], traj.y[0]], axis=-1), dtype=torch.float32, device=device
            )
            vel = torch.tensor(
                np.stack([traj.vx[0], traj.vy[0]], axis=-1), dtype=torch.float32, device=device
            )
            for tick in range(1, max(horizons) + 1):
                pos, vel = model(pos, vel, traj.g, traj.dt)
                if tick in horizons:
                    truth = np.stack([traj.x[tick], traj.y[tick]], axis=-1)
                    pred = pos.cpu().numpy()
                    corr = float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1])
                    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
                    rows.append({"traj": traj_name_of(traj), "tick": tick, "corr": corr, "rmse": rmse})
    return rows


def traj_name_of(traj: ParticleTrajectory) -> str:
    return getattr(traj, "name", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollout", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--horizons", default="5,10,20,40", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajs = load_all(args.data)
    names = [l.strip() for l in (Path(args.data) / "manifest.txt").read_text().splitlines() if l.strip()]
    for name, traj in zip(names, trajs):
        traj.name = name

    model = LagrangianModel(k=args.k, layers=args.layers, hidden=args.hidden)
    print(f"model: {model.num_params()} params, device {device}")

    log_rows = []

    def on_log(row):
        log_rows.append(row)
        print(f"step {row['step']:6d}  loss {row['loss']:.6f}  {row['secs']:.1f}s", flush=True)

    t0 = time.time()
    fit(model, trajs, args.steps, args.lr, args.rollout, device, on_log)
    wall = time.time() - t0
    print(f"training wall time: {wall:.1f}s")

    eval_rows = evaluate(model, trajs, args.horizons, device)
    for row in eval_rows:
        print(f"eval {row['traj']:16s} tick {row['tick']:4d}  corr {row['corr']:.4f}  rmse {row['rmse']:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    if args.log_file:
        Path(args.log_file).write_text(
            json.dumps({"rows": log_rows, "eval": eval_rows, "wall_seconds": wall, "params": model.num_params()}, indent=2)
        )


if __name__ == "__main__":
    main()
