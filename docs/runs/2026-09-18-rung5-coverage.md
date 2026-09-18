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

## Attempt 2b (run 16, model18)

Machine: Apple M2, 16 GB. Backend: wgpu (Metal). GPU uncontended
(`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` empty before
scoring). Checkpoint: `artifacts/model18` (run 16,
docs/runs/2026-09-18-16.md, warm start from model17, extended-coverage data,
no-horizon-fraction 0.15, rollout 8, lr 2e-4, 700 steps).

Full tables (model18, model17, model15 rows) are appended to
docs/runs/2026-09-18-16.md. Summary at 50 ticks below.

### Standard held-out set (24 single-drop trajectories, pre-reflection)

Command: `./target/release/jacobi eval-metrics --model artifacts/model18 --horizons 10,25,50 --trajectories 24 --seed 0 --project false`

| model | ticks | corr (h,u,v) | amplitude ratio | radius err (cells) | wavefront lag (xcorr) |
|---|---:|---:|---:|---:|---:|
| model18 | 50 | 0.9460 | 1.2318 | +4.614 | -0.396 |
| model17 | 50 | 0.9695 | 1.0134 | -4.905 | +0.191 |
| model15 | 50 | 0.9974 | 1.0335 | +0.081 | +0.052 |

### Stress set (task 8b): single high-amp / stacked / corner, 8 trajectories each

Command: `./target/release/jacobi eval-metrics --model artifacts/model18 --horizons 10,25,50 --trajectories 24 --seed 0 --project false --stress true`

| category | model | ticks | corr (h,u,v) | amplitude ratio |
|---|---|---:|---:|---:|
| single high-amp | model18 | 50 | 0.9856 | 1.0288 |
| single high-amp | model17 | 50 | 0.9833 | 0.9583 |
| single high-amp | model15 | 50 | 0.8083 | 0.8694 |
| stacked | model18 | 50 | 0.9887 | 1.1102 |
| stacked | model17 | 50 | 0.9853 | 1.0717 |
| stacked | model15 | 50 | 0.8831 | 1.2564 |
| corner | model18 | 50 | 0.9468 | 0.9417 |
| corner | model17 | 50 | 0.9590 | 0.7700 |
| corner | model15 | 50 | 0.9837 | 0.8265 |

### Task 8b criteria

**Criterion 1 — standard set, no regression beyond 0.02 in corr@50 vs
model15 (0.9974, so bar is corr@50 ≥ 0.9774).** model18 = 0.9460, delta =
-0.0514. **FAIL** — a bigger regression than model17's -0.0279, and further
past the 0.02 tolerance.

**Criterion 2 — stress set corr@50 improved per category vs the model15
baseline (high-amp 0.8083, stacked 0.8831, corner 0.9837).**

| category | model15 baseline | model18 | delta | pass/fail |
|---|---:|---:|---:|---|
| single high-amp | 0.8083 | 0.9856 | +0.1773 | pass |
| stacked | 0.8831 | 0.9887 | +0.1056 | pass |
| corner | 0.9837 | 0.9468 | -0.0369 | fail |

Same failure mode as model17: corner regresses (worse than model17's
-0.0247), while high-amp and stacked both improve further than model17 did.
Per-category criterion requires all three to improve. **FAIL.**

### Task 8b verdict (attempt 2b): FAIL

Both criteria fail, and by a wider margin than model17 on the standard set
(-0.0514 vs -0.0279) and on corner (-0.0369 vs -0.0247), though high-amp and
stacked both edge out model17. Extending the extra-drop-prob/no-horizon
training with warm start from model17 did not fix the corner regression —
it deepened it — while trading away standard-set accuracy relative to
model17.

### Acceptance bars (docs/GOAL-2026-09-18.md) on model18, standard set

| bar | required | measured (model18) | pass/fail |
|---|---|---|---|
| correlation at 25 ticks | ≥ 0.9 | 0.9866 | pass |
| correlation at 50 ticks | ≥ 0.8 | 0.9460 | pass |
| amplitude ratio at 50 ticks | 0.7-1.3 | 1.2318 | pass |
| wavefront lag at 50 ticks (xcorr) | < 2 cells | -0.396 | pass |

All four acceptance bars pass on model18, same as model17.

### Verdict: best checkpoint overall

Neither model17 nor model18 (nor model15/model16) passes task 8b's two
criteria strictly. Of the four, **model17 remains the best overall
checkpoint for the acceptance check**: it has the smallest standard-set
regression from model15 (-0.0279 vs model18's -0.0514), the best corner
stress score of the coverage-trained checkpoints (0.9590 vs model18's
0.9468, both still short of model15's 0.9837), and stress gains on
high-amp/stacked close to model18's (+0.175/+0.102 vs model18's
+0.177/+0.106) without model18's extra standard-set cost. **Task 8b is
still failed**; model18 is not an improvement over model17 and is not
promoted.
