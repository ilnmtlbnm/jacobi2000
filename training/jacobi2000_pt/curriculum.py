"""A PyTorch reconstruction of model20's training lineage
(docs/runs/2026-09-18-{9..17}.md, docs/runs/2026-09-19-{1,2}.md), run as a
sequence of stages on one dataset dump instead of the original's ~18
separately-launched Burn/wgpu runs.

What's reconstructed exactly: the *shape* of the lineage documented in the
run log -- rollout curriculum (1 -> 4 -> 8 -> 16 ticks), constant noise
injection (0.04), the switch from no projection to mass-only projection
(A1, docs/runs/2026-09-19-1.md) to mass+energy projection (A2,
docs/runs/2026-09-19-2.md, the recipe that produced model20), each stage
warm-started from the previous stage's exported weights, exactly as
`--init artifacts/modelNN` warm-started each Burn run.

What differs, stated plainly (docs/pytorch-training.md's acceptance test
already allows "design the shortest curriculum that reaches the target and
say what differs" when the lineage can't be replayed exactly):
  - The Burn lineage used a different, more slowly-widened dataset
    distribution at each stage (rung-3 pool -> single-drop -> extended
    coverage with stacked/corner drops, docs/runs/2026-09-18-14..17.md); this
    curriculum instead trains every stage on one dataset dump (the
    already-widest "coverage" distribution, matching
    docs/pytorch-training.md section 3's `dump-dataset` call), so the
    normalizer is identical across stages and no per-stage renormalization
    is needed. This makes every stage strictly harder than the Burn
    lineage's early stages, not easier.
  - It is 6 stages, not ~18: the Burn lineage includes several abandoned or
    superseded attempts (rung-5 "coverage attempt 1", "attempt 2",
    "attempt 2b") kept in the log for the record but not part of the
    surviving chain into model19/model20; this curriculum only replays the
    surviving shape (rollout ramp, then coverage-style training throughout,
    then the two projection stages).
  - Step counts per stage are not copied from the Burn log (those were
    tuned interactively against a GPU that was also running other jobs);
    they're chosen fresh for a from-scratch run on one dataset, generous
    enough on a fast GPU to substitute for the
    warm-start budget the original lineage got from its rung-3/4 stages
    this curriculum skips.
"""

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .data import Normalizer, TrajectoryDataset
from .export import export as export_checkpoint
from .model import Jacobi2000, Jacobi2000Config
from .parity import load_from_export
from .train import FitConfig, fit


@dataclass
class Stage:
    name: str
    steps: int
    lr: float
    rollout: int
    rollout_after: int
    noise: float
    project: bool
    project_energy: bool
    r_max: float
    spectral: float
    lr_floor: float
    init_from: str | None  # previous stage name, or None for scratch


STAGES: list[Stage] = [
    Stage("s1-rollout1", steps=3000, lr=3e-4, rollout=1, rollout_after=3000,
          noise=0.04, project=False, project_energy=False, r_max=1.0001,
          spectral=1.0, lr_floor=1.0, init_from=None),
    Stage("s2-rollout4", steps=2000, lr=3e-4, rollout=4, rollout_after=0,
          noise=0.04, project=False, project_energy=False, r_max=1.0001,
          spectral=1.0, lr_floor=1.0, init_from="s1-rollout1"),
    Stage("s3-rollout8", steps=2000, lr=2e-4, rollout=8, rollout_after=0,
          noise=0.04, project=False, project_energy=False, r_max=1.0001,
          spectral=1.0, lr_floor=1.0, init_from="s2-rollout4"),
    Stage("s4-rollout16", steps=1000, lr=1e-4, rollout=16, rollout_after=0,
          noise=0.04, project=False, project_energy=False, r_max=1.0001,
          spectral=1.0, lr_floor=1.0, init_from="s3-rollout8"),
    Stage("s5-mass-proj", steps=500, lr=1e-4, rollout=8, rollout_after=0,
          noise=0.04, project=True, project_energy=False, r_max=1.0001,
          spectral=1.0, lr_floor=0.05, init_from="s4-rollout16"),
    Stage("s6-mass-energy-proj", steps=1500, lr=1e-4, rollout=8, rollout_after=0,
          noise=0.04, project=True, project_energy=True, r_max=1.0001,
          spectral=1.0, lr_floor=1.0, init_from="s5-mass-proj"),
]


def run_curriculum(dataset_dir: str, out_root: str, dim: int, layers: int, seed: int, log_path: str | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrajectoryDataset(dataset_dir)
    norm = dataset.fit_normalizer()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_logs = {}
    t_all0 = time.time()
    for stage in STAGES:
        torch.manual_seed(seed)
        cfg = Jacobi2000Config(dim=dim, layers=layers)
        model = Jacobi2000(cfg)

        if stage.init_from is not None:
            prev_dir = out_root / stage.init_from
            model, _ = load_from_export(prev_dir)

        fit_cfg = FitConfig(
            steps=stage.steps, lr=stage.lr, rollout=stage.rollout,
            rollout_after=stage.rollout_after, noise=stage.noise,
            project=stage.project, project_energy=stage.project_energy,
            r_max=stage.r_max, spectral=stage.spectral, lr_floor=stage.lr_floor,
            seed=seed,
        )

        rows = []

        def on_log(row, name=stage.name):
            rows.append(row)
            print(f"[{name}] step {row['step']:6d}  loss {row['loss']:.6f}  {row['secs']:.1f}s", flush=True)

        t0 = time.time()
        model = fit(model, dataset, norm, fit_cfg, device, on_log)
        wall = time.time() - t0
        print(f"[{stage.name}] wall {wall:.1f}s", flush=True)

        stage_out = out_root / stage.name
        export_checkpoint(
            model, norm, dx=dataset.dx, dt=dataset.dt, g=dataset.g,
            depth=dataset.depth, nu=dataset.nu, mask_radius=1, mask_dilation=1,
            energy_r_max=stage.r_max, out_dir=stage_out,
        )
        all_logs[stage.name] = {"rows": rows, "wall_seconds": wall, "params": model.num_params()}

    total_wall = time.time() - t_all0
    print(f"curriculum total wall: {total_wall:.1f}s", flush=True)
    if log_path:
        Path(log_path).write_text(json.dumps({"stages": all_logs, "total_wall_seconds": total_wall}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, help="root dir; one subdir per stage")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()
    run_curriculum(args.dataset, args.out, args.dim, args.layers, args.seed, args.log_file)


if __name__ == "__main__":
    main()
