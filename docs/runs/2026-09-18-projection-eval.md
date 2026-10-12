# Eval 2026-09-18 — mass projection at inference only

No training. The three checkpoints from runs 1-3 re-evaluated with the
constant-mass projection switched on in the rollout, to separate mass drift
from the rest of the 50-tick error.

Machine: a 16 GB consumer machine. Timings **provisional** (the GPU is shared).

- commit: `d6fdc68`
- backend: wgpu (Metal)
- command: `./target/release/jacobi eval --model artifacts/<model> --trajectories 4 --seed 1000 --project <true|false>`
- seed 1000 is the seed `train` uses for its own post-run eval, so the
  `--project false` column reproduces the tables in runs 1-3 exactly.

## Rollout error (RMSE, normalized units, 4 held-out trajectories)

| ticks | run 1 off | run 1 on | run 2 off | run 2 on | run 3 off | run 3 on | frozen field |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.11168 | 0.11166 | 0.11036 | 0.11029 | 0.14673 | 0.14666 | 0.52369 |
| 50 | 3.01984 | 2.96443 | 1.26863 | 1.26233 | 1.59181 | 1.58292 | 1.15976 |
| 200 | 67.12000 | 66.29755 | 38.00577 | 33.96453 | 36.18235 | 35.98714 | 0.97018 |
