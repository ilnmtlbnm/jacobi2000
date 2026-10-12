# Rung-5 curriculum stage 1 (rollout 8)

Training run: `docs/runs/2026-09-18-12.md` (warm start from `artifacts/model13`, stencil radius 1 dilation 1, rollout 8, 700 steps, equal-ish continuation budget). Checkpoint: `artifacts/model14`.

## Rung-1 metrics (eval-metrics, 24 held-out single-drop pre-reflection trajectories)

Command:

```
./target/release/jacobi eval-metrics --model artifacts/model14 --horizons 10,25,50 --trajectories 24 --seed 0 --project false
```

backend: wgpu (Metal), mass projection: false, mask: --mask stencil --radius 1 --dilation 1
24 single-drop trajectories, 22 skipped (horizon < 50 ticks)

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.9991 | 0.9998 | 0.10347 | 1.0277 | -0.001 | 0.00852 | 0.01421 | 0.01832 | 0.03818 |
| frozen field | 10 | 0.6947 | 0.9687 | 1.97857 | 1.6385 | -0.012 | 0.07538 | 0.15232 | 0.21779 | 0.61477 |
| model | 25 | 0.9976 | 0.9987 | 0.09964 | 1.0279 | +0.683 | 0.00976 | 0.01631 | 0.01971 | 0.06344 |
| frozen field | 25 | 0.0647 | 0.1387 | 2.38726 | 3.6404 | -5.195 | 0.23701 | 0.49479 | 0.84052 | 1.13509 |
| model | 50 | 0.9925 | 0.9964 | 0.12951 | 1.0561 | +3.196 | 0.01469 | 0.02639 | 0.03375 | 0.10577 |
| frozen field | 50 | -0.2532 | -0.4485 | 2.12042 | 3.7638 | -10.787 | 0.22334 | 0.46304 | 0.92126 | 1.30727 |

## Acceptance bars

Measured on held-out single-drop trajectories, pre-reflection.

| bar | required | measured (model, run 12) | pass/fail |
|---|---|---|---|
| correlation at 25 ticks | ≥ 0.9 | 0.9976 | pass |
| correlation at 50 ticks | ≥ 0.8 | 0.9925 | pass |
| amplitude ratio at 50 ticks | 0.7–1.3 | 1.0561 | pass |
| wavefront radius error at 50 ticks | < 2 cells | +3.196 | fail |

Three of four bars pass; wavefront radius error at 50 ticks fails (magnitude 3.196 > 2 cells).

## Did the radius error at 50 ticks improve vs run 11 (-2.84 cells)?

No. Run 11 (model13, rollout 4) measured -2.838 cells at 50 ticks; run 12 (model14, rollout 8) measures +3.196 cells. The magnitude got worse (2.838 → 3.196 cells) and the sign flipped from undershoot to overshoot — the rollout-8 curriculum stage did not improve wavefront radius fidelity at 50 ticks; if anything it traded one direction of error for a slightly larger one in the other direction.
