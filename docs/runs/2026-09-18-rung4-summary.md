# Rung-4 summary: full attention (run 10) vs stencil r1 (run 11)

Cites `docs/runs/2026-09-18-rung4-full.md` (run 10, `artifacts/model12`) and `docs/runs/2026-09-18-rung4-stencil-r1.md` (run 11, `artifacts/model13`).

## Rung-1 metrics, held-out single-drop pre-reflection trajectories

| config | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | band 2-4 | band 4-8 | band 8-16 | global RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full (run 10) | 10 | 0.9948 | 0.9990 | 0.25165 | 1.0733 | -0.001 | 0.01303 | 0.02302 | 0.03244 | 0.15854 |
| stencil r1 (run 11) | 10 | 0.9990 | 0.9997 | 0.13917 | 1.0358 | -0.001 | 0.01020 | 0.01719 | 0.02259 | 0.04572 |
| full (run 10) | 25 | 0.9755 | 0.9905 | 0.43042 | 1.1498 | +2.479 | 0.01749 | 0.03216 | 0.04718 | 0.36952 |
| stencil r1 (run 11) | 25 | 0.9982 | 0.9983 | 0.12367 | 1.0262 | +0.440 | 0.01212 | 0.02117 | 0.02558 | 0.06459 |
| full (run 10) | 50 | 0.9097 | 0.9670 | 0.74583 | 1.4359 | +4.705 | 0.02606 | 0.04975 | 0.08072 | 0.70140 |
| stencil r1 (run 11) | 50 | 0.9958 | 0.9963 | 0.12123 | 1.0147 | -2.838 | 0.01875 | 0.03492 | 0.04482 | 0.08896 |

## Training-log rollout RMSE (normalized units, 4 held-out trajectories)

| config | ticks | model RMSE | frozen field RMSE |
|---|---:|---:|---:|
| full (run 10) | 10 | 0.17349 | 0.89232 |
| stencil r1 (run 11) | 10 | 0.07041 | 0.89232 |
| full (run 10) | 50 | 0.71186 | 1.82026 |
| stencil r1 (run 11) | 50 | 0.13950 | 1.82026 |
| full (run 10) | 200 | 2.48337 | 1.62202 |
| stencil r1 (run 11) | 200 | 0.36497 | 1.62202 |

## Acceptance bars (docs/GOAL-2026-09-18.md)

| bar | required | full (run 10) | pass/fail | stencil r1 (run 11) | pass/fail |
|---|---|---|---|---|---|
| correlation at 25 ticks | ≥ 0.9 | 0.9755 | pass | 0.9982 | pass |
| correlation at 50 ticks | ≥ 0.8 | 0.9097 | pass | 0.9958 | pass |
| amplitude ratio at 50 ticks | 0.7–1.3 | 1.4359 | fail | 1.0147 | pass |
| wavefront radius error at 50 ticks | < 2 cells | +4.705 | fail | -2.838 | fail |

Neither configuration clears all four acceptance bars; the stencil r1 model (run 11) proceeds to rung 5.
