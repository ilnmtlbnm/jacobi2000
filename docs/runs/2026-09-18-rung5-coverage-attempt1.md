# Rung-5 coverage attempt 1 (task 8b, run 14 -> model16)

Machine: a 16 GB consumer machine. Backend: wgpu (Metal). GPU uncontended
(`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` empty).
Commit: `103cdc6`. Checkpoint: `artifacts/model16` (run 14,
docs/runs/2026-09-18-14.md, warm start from model15 on extended-coverage
data, rollout 16, 450 steps).

## Standard held-out set (24 single-drop trajectories, pre-reflection)

Command: `./target/release/jacobi eval-metrics --model artifacts/model16 --horizons 10,25,50 --trajectories 24 --seed 0 --project false`

| model | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) |
|---|---:|---:|---:|---:|---:|---:|---:|
| model16 | 10 | 0.9743 | 0.9989 | 0.08214 | 0.9577 | -0.000 | -0.097 |
| model15 | 10 | 0.9994 | 0.9998 | 0.09019 | 1.0034 | +0.000 | +0.107 |
| model16 | 25 | 0.8947 | 0.9800 | 0.20033 | 1.0323 | -0.605 | -0.493 |
| model15 | 25 | 0.9989 | 0.9991 | 0.07739 | 1.0250 | +0.765 | +0.043 |
| model16 | 50 | 0.7194 | 0.9142 | 0.40107 | 1.1084 | -6.755 | -0.102 |
| model15 | 50 | 0.9974 | 0.9981 | 0.08654 | 1.0335 | +0.081 | +0.052 |

## Stress set (task 8b): single high-amp / stacked / corner, 8 trajectories each

Command: `./target/release/jacobi eval-metrics --model artifacts/model16 --horizons 10,25,50 --trajectories 24 --seed 0 --project false --stress true`

| category | model | ticks | corr (h,u,v) | corr (h) | amplitude ratio |
|---|---|---:|---:|---:|---:|
| single high-amp | model16 | 50 | 0.9463 | 0.9599 | 0.9101 |
| single high-amp | model15 | 50 | 0.8083 | 0.8100 | 0.8694 |
| stacked | model16 | 50 | 0.9123 | 0.9826 | 1.0328 |
| stacked | model15 | 50 | 0.8831 | 0.8951 | 1.2564 |
| corner | model16 | 50 | 0.7311 | 0.8445 | 1.0102 |
| corner | model15 | 50 | 0.9837 | 0.9777 | 0.8265 |

(Full tables with all horizons and both models are appended to
docs/runs/2026-09-18-14.md.)

## Verdict

**Regression on standard set: yes.** corr@50 (h,u,v) dropped from model15's
0.9974 to model16's 0.7194, a drop of 0.2780 — far past the 0.02 no-regression
bar in task 8b's convergence criteria. This alone fails the criterion,
regardless of the stress-set outcome.

**Stress set corr@50 (h,u,v), model16 vs model15 baseline:** single
high-amplitude improved (0.8083 -> 0.9463, +0.1380); stacked improved
(0.8831 -> 0.9123, +0.0292); corner regressed (0.9837 -> 0.7311, -0.2526).
The extended-coverage training helped the two categories the author named as
failing, but broke the easy corner category and the standard held-out set —
net regression, not an improvement across the board.
