# Acceptance check — stay-true-for-longer

Machine: Apple M2, 16 GB. Backend: wgpu (Metal). GPU uncontended
(`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` empty
immediately before this run). Commit: `4e8673b`.

Fresh held-out single-drop pre-reflection trajectories, seed `2026`
(never used in any training or eval run in this queue — all prior runs
used seed 0 or 5/7/8 for training and seed 0 for eval). 24 trajectories
requested per model, 20 skipped for horizon < 50 ticks (post-reflection),
leaving trajectories scored at every requested horizon that reached it
pre-reflection. No xr ensemble (`--xr` omitted, default K=1), no
projection (`--project false`).

Models scored independently:
- (a) **artifacts/model12** — best full-attention checkpoint (rung 4
  baseline, docs/runs/2026-09-18-rung4-full.md), reported as baseline, not
  required to pass.
- (b) **artifacts/model19** — best stencil-trained checkpoint (task 8c,
  BEST), the model the acceptance bar applies to per the lead's 2026-09-18
  14:55 amendment ("locality is the thesis").

Commands:

```
./target/release/jacobi eval-metrics --model artifacts/model12 --horizons 10,25,50 --trajectories 24 --seed 2026 --project false
./target/release/jacobi eval-metrics --model artifacts/model19 --horizons 10,25,50 --trajectories 24 --seed 2026 --project false
```

## (a) artifacts/model12 — full attention (baseline, reported)

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells, argmax) | wavefront lag (xcorr) | band 2-4 | band 4-8 | band 8-16 | global RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.9947 | 0.9990 | 0.23913 | 1.0611 | -0.000 | +0.042 | 0.01194 | 0.02132 | 0.03180 | 0.15728 |
| frozen field | 10 | 0.6904 | 0.9720 | 1.93410 | 1.5973 | -0.011 | -1.194 | 0.06799 | 0.13842 | 0.20387 | 0.61014 |
| model | 25 | 0.9741 | 0.9901 | 0.42418 | 1.1928 | +2.175 | +0.201 | 0.01657 | 0.03070 | 0.04653 | 0.36895 |
| frozen field | 25 | 0.0731 | 0.1574 | 2.31864 | 3.7441 | -5.934 | -7.351 | 0.22444 | 0.46922 | 0.80376 | 1.12276 |
| model | 50 | 0.9037 | 0.9650 | 0.74799 | 1.4504 | +4.494 | +0.989 | 0.02401 | 0.04592 | 0.07910 | 0.70210 |

## (b) artifacts/model19 — stencil r1 d1 (BEST, bar applies)

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells, argmax) | wavefront lag (xcorr) | band 2-4 | band 4-8 | band 8-16 | global RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.9995 | 0.9998 | 0.03287 | 0.9868 | +0.001 | +0.055 | 0.00414 | 0.00601 | 0.00764 | 0.02175 |
| frozen field | 10 | 0.6872 | 0.9720 | 1.34429 | 1.5973 | -0.011 | -1.194 | 0.04682 | 0.09533 | 0.14041 | 0.42408 |
| model | 25 | 0.9980 | 0.9987 | 0.05520 | 1.0188 | +0.438 | +0.067 | 0.00627 | 0.01046 | 0.01393 | 0.04888 |
| frozen field | 25 | 0.0732 | 0.1574 | 1.60607 | 3.7441 | -5.934 | -7.351 | 0.15457 | 0.32316 | 0.55356 | 0.77757 |
| model | 50 | 0.9924 | 0.9975 | 0.09672 | 1.0419 | +2.609 | -0.191 | 0.00938 | 0.01696 | 0.02401 | 0.09242 |

Note: the argmax radius error column is included for the record only, per
task 8a's finding (docs/runs/2026-09-18-radius-estimator.md) that the
argmax estimator is bistable at 50 ticks (both models here show a sizeable
argmax radius error at 50 ticks — +4.494 cells for model12, +2.609 for
model19 — despite very high correlation and a small, stable xcorr lag).
The acceptance bar uses the xcorr lag column, not this one.

## Acceptance bars (docs/GOAL-2026-09-18.md)

| bar | required | model12 (baseline) | pass/fail | model19 (BEST, bar applies) | pass/fail |
|---|---|---:|---|---:|---|
| correlation at 25 ticks | ≥ 0.9 | 0.9741 | pass | 0.9980 | pass |
| correlation at 50 ticks | ≥ 0.8 | 0.9037 | pass | 0.9924 | pass |
| amplitude ratio at 50 ticks | 0.7–1.3 | 1.4504 | **fail** | 1.0419 | pass |
| wavefront lag at 50 ticks (xcorr) | < 2 cells | +0.989 | pass | -0.191 | pass |

## Verdict: PASS

The bar applies to the best stencil-trained checkpoint (model19), per the
lead's 2026-09-18 14:55 amendment ("locality is the thesis, not an
option"). model19 passes all four acceptance bars independently, on fresh
held-out single-drop pre-reflection trajectories at a seed (2026) never
used anywhere else in this queue, with the GPU uncontended.

The full-attention baseline (model12) is reported alongside it as
required, but is not itself required to pass — and in fact it fails the
amplitude-ratio bar at 50 ticks (1.4504, outside 0.7–1.3), consistent with
its earlier characterization in docs/runs/2026-09-18-rung4-full.md. This is
exactly the gap the stencil-training rung (rung 4 onward) was meant to
close, and it is closed: model19's amplitude ratio at 50 ticks (1.0419) is
comfortably inside the bar where model12's is not.

**PASS.**
