# Rung-4 stencil (radius 1, dilation 1)

Mask: stencil, radius 1, dilation 1 (confirmed by eval-metrics output: `mask: --mask stencil --radius 1 --dilation 1`). See `docs/runs/2026-09-18-11.md` for the training run (warm start from `artifacts/model11`, +1350 steps, equal budget to run 10, run 11 of `stay-true-for-longer`). Checkpoint: `artifacts/model13`.

## Rung-1 metrics (eval-metrics, 24 held-out single-drop pre-reflection trajectories)

Command:

```
./target/release/jacobi eval-metrics --model artifacts/model13 --horizons 10,25,50 --trajectories 24 --seed 0 --project false
```

24 single-drop trajectories, 22 skipped (horizon < 50 ticks).

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.9990 | 0.9997 | 0.13917 | 1.0358 | -0.001 | 0.01020 | 0.01719 | 0.02259 | 0.04572 |
| frozen field | 10 | 0.6835 | 0.9687 | 2.04643 | 1.6385 | -0.012 | 0.07556 | 0.15268 | 0.21831 | 0.63595 |
| model | 25 | 0.9982 | 0.9983 | 0.12367 | 1.0262 | +0.440 | 0.01212 | 0.02117 | 0.02558 | 0.06459 |
| frozen field | 25 | 0.0639 | 0.1387 | 2.43889 | 3.6404 | -5.195 | 0.23757 | 0.49596 | 0.84250 | 1.15908 |
| model | 50 | 0.9958 | 0.9963 | 0.12123 | 1.0147 | -2.838 | 0.01875 | 0.03492 | 0.04482 | 0.08896 |
| frozen field | 50 | -0.2451 | -0.4485 | 2.14787 | 3.7638 | -10.787 | 0.22387 | 0.46414 | 0.92344 | 1.32435 |

## Acceptance bars (docs/GOAL-2026-09-18.md)

Measured on held-out single-drop trajectories, pre-reflection.

| bar | required | measured (model) | pass/fail |
|---|---|---|---|
| correlation at 25 ticks | ≥ 0.9 | 0.9982 | pass |
| correlation at 50 ticks | ≥ 0.8 | 0.9958 | pass |
| amplitude ratio at 50 ticks | 0.7–1.3 | 1.0147 | pass |
| wavefront radius error at 50 ticks | < 2 cells | -2.838 | fail |

Three of four bars pass; wavefront radius error at 50 ticks fails (magnitude 2.838 > 2 cells), though it is a large improvement over the run-10 full-attention baseline's +4.705.
