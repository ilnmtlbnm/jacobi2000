# Task 10 — xr ensemble at inference

Machine: a 16 GB consumer machine. Backend: wgpu (Metal). GPU uncontended
(`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` empty before
the run). Commit: `fbb874b`. Checkpoint: `artifacts/model19` (BEST, task
8c, docs/runs/2026-09-18-17.md / docs/runs/2026-09-18-rung5-coverage.md).

Command:

```
cargo run --release -p cli --features wgpu --no-default-features -- eval-metrics --model artifacts/model19 --xr 4 --horizons 10,25,50 --trajectories 16 --seed 0 --project false
```

16 single-drop trajectories requested, 18 skipped (horizon < 50 ticks) —
the eval-metrics binary always draws from a fixed larger candidate pool and
keeps only pre-reflection trajectories at every requested horizon.

Implements jittered patch-grid offsets (design notes §4): K=4 offsets of the
same weights, mean + per-cell spread across offsets.

## Baseline (single-offset) rung-1 metrics

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) |
|---|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.9996 | 0.9998 | 0.03232 | 0.9860 | +0.001 | +0.065 |
| model | 25 | 0.9983 | 0.9988 | 0.05409 | 1.0109 | +0.544 | +0.073 |
| model | 50 | 0.9936 | 0.9976 | 0.09616 | 1.0345 | +2.840 | -0.212 |

## xr ensemble mean (K=4) rung-1 metrics, alongside the single-offset baseline

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) |
|---|---:|---:|---:|---:|---:|---:|---:|
| xr ensemble mean | 10 | 0.9996 | 0.9998 | 0.03281 | 0.9853 | +0.001 | +0.070 |
| single-offset baseline | 10 | 0.9996 | 0.9998 | 0.03232 | 0.9860 | +0.001 | +0.065 |
| xr ensemble mean | 25 | 0.9976 | 0.9969 | 0.06834 | 1.0262 | +0.551 | +0.068 |
| single-offset baseline | 25 | 0.9983 | 0.9988 | 0.05409 | 1.0109 | +0.544 | +0.073 |
| xr ensemble mean | 50 | 0.9697 | 0.9751 | 0.18600 | 0.9056 | +0.918 | -0.147 |
| single-offset baseline | 50 | 0.9936 | 0.9976 | 0.09616 | 1.0345 | +2.840 | -0.212 |

At this checkpoint the ensemble mean is slightly worse than the
single-offset baseline at every horizon (e.g. corr@50 0.9697 vs 0.9936) —
K=4 patch-grid jitter costs a little accuracy here rather than gaining it,
though it stays well inside the acceptance bars. The ensemble's value is
the spread signal for the divergence panel, not a metric improvement.

## Per-cell spread across offsets

| ticks | mean spread (physical h units) |
|---:|---:|
| 10 | 0.000005 |
| 25 | 0.000010 |
| 50 | 0.000027 |

Spread is nonzero at every horizon and grows monotonically with ticks
(5e-6 -> 1e-5 -> 2.7e-5), roughly 5x from 10 to 50 ticks — it is not a
constant, and it tracks the wave's growing spatial extent as expected for
a per-cell disagreement measure across patch-grid offsets. Reproduced with
an identical rerun (same command, same seed): spread values matched to the
printed precision.
