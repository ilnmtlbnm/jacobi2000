# Rung-5 coverage attempt 2 (task 8b, run 15 -> model17)

Machine: Apple M2, 16 GB. Backend: wgpu (Metal). GPU uncontended
(`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` empty before
scoring). Commit: `6fa22bb`. Checkpoint: `artifacts/model17` (run 15,
docs/runs/2026-09-18-15.md, warm start from model15 on extended-coverage
data, rollout 4, lr 2e-4, 1350 steps).

Full tables (model, model16, model15 rows) are appended to
docs/runs/2026-09-18-15.md. Summary at 50 ticks below.

## Standard held-out set (24 single-drop trajectories, pre-reflection)

Command: `./target/release/jacobi eval-metrics --model artifacts/model17 --horizons 10,25,50 --trajectories 24 --seed 0 --project false`

| model | ticks | corr (h,u,v) | amplitude ratio | radius err (cells) | wavefront lag (xcorr) |
|---|---:|---:|---:|---:|---:|
| model17 | 50 | 0.9695 | 1.0134 | -4.905 | +0.191 |
| model16 | 50 | 0.7194 | 1.1084 | -6.755 | -0.102 |
| model15 | 50 | 0.9974 | 1.0335 | +0.081 | +0.052 |

## Stress set (task 8b): single high-amp / stacked / corner, 8 trajectories each

Command: `./target/release/jacobi eval-metrics --model artifacts/model17 --horizons 10,25,50 --trajectories 24 --seed 0 --project false --stress true`

| category | model | ticks | corr (h,u,v) | amplitude ratio |
|---|---|---:|---:|---:|
| single high-amp | model17 | 50 | 0.9833 | 0.9583 |
| single high-amp | model16 | 50 | 0.9463 | 0.9101 |
| single high-amp | model15 | 50 | 0.8083 | 0.8694 |
| stacked | model17 | 50 | 0.9853 | 1.0717 |
| stacked | model16 | 50 | 0.9123 | 1.0328 |
| stacked | model15 | 50 | 0.8831 | 1.2564 |
| corner | model17 | 50 | 0.9590 | 0.7700 |
| corner | model16 | 50 | 0.7311 | 1.0102 |
| corner | model15 | 50 | 0.9837 | 0.8265 |

## Task 8b criteria

**Criterion 1 — standard set, no regression beyond 0.02 in corr@50 vs
model15.** model15 = 0.9974, model17 = 0.9695, delta = -0.0279. **FAIL**
(exceeds the 0.02 tolerance by 0.0079). model17 is a large improvement over
model16's disastrous 0.7194, and closer to model15 than model16 was, but
still misses the bar.

**Criterion 2 — stress set corr@50 improved per category vs the model15
baseline (high-amp 0.8083, stacked 0.8831, corner 0.9837).**

| category | model15 baseline | model17 | delta | pass/fail |
|---|---:|---:|---:|---|
| single high-amp | 0.8083 | 0.9833 | +0.1750 | pass |
| stacked | 0.8831 | 0.9853 | +0.1022 | pass |
| corner | 0.9837 | 0.9590 | -0.0247 | fail |

Two of three stress categories improve substantially; corner regresses
slightly (still a strong absolute score, 0.959, but below model15's 0.9837).
Per-category criterion requires all three to improve. **FAIL.**

## Task 8b verdict: FAIL

Both criteria must pass; neither does. Criterion 1 fails by 0.0079 past
tolerance on the standard set. Criterion 2 fails because the corner category
regresses (-0.0247) even though high-amp and stacked both improve
substantially (+0.175, +0.102). model17 is a much better trade-off than
model16 (which failed both criteria far worse: -0.278 on standard, and
regressed corner by -0.2526), but does not clear the bar set by task 8b.

## Acceptance bars (docs/GOAL-2026-09-18.md) on model17, standard set

| bar | required | measured (model17) | pass/fail |
|---|---|---|---|
| correlation at 25 ticks | ≥ 0.9 | 0.9920 | pass |
| correlation at 50 ticks | ≥ 0.8 | 0.9695 | pass |
| amplitude ratio at 50 ticks | 0.7-1.3 | 1.0134 | pass |
| wavefront lag at 50 ticks (xcorr) | < 2 cells | +0.191 | pass |

All four acceptance bars still pass on model17. Per the run-15 task
instructions, since task 8b's own two convergence criteria fail, web/ is
left untouched (still serving model14).
