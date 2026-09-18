# Stress-set baseline (task 8b)

Machine: Apple M2, 16 GB. Backend: wgpu (Metal). GPU uncontended
(`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` empty before
the run). Commit: `bc34aac` (adds the `--stress` eval set, 9c821db, and the
`xcorr_lag` NaN-on-short-profile fix it surfaced, bc34aac).

Checkpoint: `artifacts/model15` (previous best stencil checkpoint, run 13 /
rung-5 curriculum stage 2, docs/runs/2026-09-18-rung5-rollout16.md), scored
**before** any training on the extended-coverage data, so this is the
before-picture the 8b curriculum run is meant to improve on.

Command:

```
./target/release/jacobi eval-metrics --model artifacts/model15 --stress true --horizons 10,25,50 --project false
```

The stress set is fixed (seed 0 per category, 8 trajectories each), not
drawn from the coverage-extended generator: single high-amplitude drops
(0.15-0.3), stacked drops (2-3 at the same spot), and corner drops (within
6 cells of two walls). For corner drops the pre-reflection horizon filter
used by the standard held-out set is meaningless — the wave meets a wall
almost immediately — so every corner trajectory is scored at every horizon
regardless of reflection, per the eval binary's own printed note.

## Single high-amplitude (0.15-0.3)

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.9742 | 0.9928 | 3.77113 | 1.1676 | -0.180 | -0.631 | 0.16911 | 0.32848 | 0.43556 | 1.10824 |
| frozen field | 10 | 0.6709 | 0.9620 | 10.34005 | 1.6902 | -0.182 | -1.154 | 0.38862 | 0.78353 | 1.06278 | 3.04103 |
| model | 25 | 0.9279 | 0.8938 | 4.09038 | 0.9179 | +1.830 | -1.096 | 0.37453 | 0.74804 | 1.09049 | 1.80448 |
| frozen field | 25 | 0.0591 | 0.0953 | 12.25560 | 3.0809 | -3.856 | -5.948 | 1.22492 | 2.55732 | 4.12975 | 5.37963 |
| model | 50 | 0.8083 | 0.8100 | 4.42821 | 0.8694 | -5.567 | +2.348 | 0.56230 | 1.08980 | 1.49233 | 2.67508 |
| frozen field | 50 | -0.1462 | -0.3264 | 10.14007 | 3.3155 | -7.263 | -1.634 | 1.07835 | 2.19898 | 4.18155 | 5.99521 |

## Stacked (2-3 drops, same spot)

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.9920 | 0.9964 | 1.38860 | 1.2411 | -0.002 | -0.704 | 0.12696 | 0.24349 | 0.34915 | 0.45877 |
| frozen field | 10 | 0.7348 | 0.9753 | 5.92770 | 1.5399 | +0.284 | -1.512 | 0.24151 | 0.48828 | 0.70871 | 1.97845 |
| model | 25 | 0.9658 | 0.9520 | 2.14791 | 1.1317 | -1.037 | -0.254 | 0.17468 | 0.32895 | 0.51834 | 1.05745 |
| frozen field | 25 | 0.1653 | 0.2893 | 7.47285 | 3.2713 | -7.657 | -4.262 | 0.73100 | 1.54299 | 2.68056 | 3.69450 |
| model | 50 | 0.8831 | 0.8951 | 2.84591 | 1.2564 | -6.846 | -0.292 | 0.31069 | 0.63743 | 0.96725 | 1.88653 |
| frozen field | 50 | -0.2011 | -0.4056 | 7.12112 | 2.9747 | -8.660 | +1.765 | 0.70117 | 1.45390 | 2.91083 | 4.32714 |

## Corner (within 6 cells of two walls)

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.9965 | 0.9981 | 0.31178 | 0.9289 | -0.236 | -0.046 | 0.03932 | 0.03980 | 0.02973 | 0.06411 |
| frozen field | 10 | 0.7785 | 0.9636 | 2.30725 | 1.2124 | -0.544 | -0.111 | 0.11861 | 0.18267 | 0.22421 | 0.49405 |
| model | 25 | 0.9886 | 0.9884 | 0.47588 | 0.8895 | +0.126 | +0.387 | 0.05890 | 0.06564 | 0.06079 | 0.13161 |
| frozen field | 25 | 0.3494 | 0.6037 | 3.16801 | 1.6368 | -1.887 | -0.394 | 0.18674 | 0.38504 | 0.64199 | 0.90988 |
| model | 50 | 0.9837 | 0.9777 | 0.40418 | 0.8265 | -0.593 | +0.049 | 0.05277 | 0.08865 | 0.10770 | 0.15947 |
| frozen field | 50 | -0.0553 | -0.1099 | 2.98801 | 2.5512 | -1.888 | +0.092 | 0.17853 | 0.35215 | 0.67501 | 1.17923 |

## Reading it

Corner drops are the easiest of the three for model15 (correlation 0.98-0.99
at every horizon) — the domain is small enough and the stencil local enough
that a wall-adjacent drop is not, by itself, a hard case for this
checkpoint. The two categories that stress the model are the ones the
named: single high-amplitude drops and stacked drops both show
correlation dropping into the 0.81-0.88 range by 50 ticks (vs 0.997 on the
standard held-out set, docs/runs/2026-09-18-rung5-rollout16.md) and
amplitude ratios drifting further from 1.0 (0.83-1.26 vs 1.03). This is the
baseline the task-8b curriculum run (warm-started on the extended-coverage
data) is meant to improve, most visibly on these two categories.
