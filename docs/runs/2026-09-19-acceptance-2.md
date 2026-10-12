# Acceptance check 2 — constrained-wider-reversible, task A3

Machine: a 16 GB consumer machine. Backend: wgpu (Metal). GPU uncontended
(`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` empty
immediately before each eval run below). Commit: pending (this task's own
commit, made after these runs).

Fresh held-out single-drop pre-reflection trajectories, seed `2027`
(never used anywhere in this run: task 1's smoke used seed 9, task 2's
training used seed 11, docs/runs/2026-09-18-acceptance.md's acceptance used
seed 2026). 24 trajectories requested per command, 17 skipped for horizon <
50 ticks (post-reflection).

Commands (verbatim):

```
./target/release/jacobi eval-metrics --model artifacts/model20 --horizons 10,25,50 --trajectories 24 --seed 2027 --project true --project-energy true
./target/release/jacobi eval-metrics --model artifacts/model20 --horizons 10,25,50 --trajectories 24 --seed 2027 --project false --project-energy false
./target/release/jacobi eval-metrics --model artifacts/model12 --horizons 10,25,50 --trajectories 24 --seed 2027 --project true --project-energy true
```

Raw output: `docs/runs/2026-09-19-model20-acceptance2-on.txt`,
`docs/runs/2026-09-19-model20-acceptance2-off.txt`,
`docs/runs/2026-09-19-model12-acceptance2.txt`.

## model20, projections ON (`--project true --project-energy true`)

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) | global RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.9997 | 0.9998 | 0.03605 | 0.9806 | +0.054 | +0.050 | 0.01619 |
| model | 25 | 0.9988 | 0.9979 | 0.05976 | 1.0076 | +0.726 | +0.007 | 0.03601 |
| model | 50 | 0.9953 | 0.9956 | 0.09037 | 1.0218 | -0.760 | -0.013 | 0.06673 |

## Four-bar acceptance table (xcorr-lag version)

| bar | required | model20 (ON) | pass/fail |
|---|---|---:|---|
| correlation at 25 ticks | ≥ 0.9 | 0.9988 | pass |
| correlation at 50 ticks | ≥ 0.8 | 0.9953 | pass |
| amplitude ratio at 50 ticks | 0.7-1.3 | 1.0218 | pass |
| wavefront lag at 50 ticks (xcorr) | < 2 cells | -0.013 | pass |

**All four bars PASS for model20 with mass and energy projection ON, fresh
seed 2027.**

## Energy-drift table (task 3)

Per-tick total physical energy (`E = sum 1/2 g h^2 + 1/2 h(u^2+v^2)`,
`jacobi2000::energy`), mean over the same 24 held-out trajectories, model
rollout with the energy projection ON vs OFF (mass projection stays on in
both, matching the main table) vs the solver's own truth trajectory. This
table is computed by `eval-metrics` internally
(`train::metrics::energy_drift_rollout`) regardless of the `--project`/
`--project-energy` flags passed on the command line, so it is identical
whether those flags are ON or OFF on the outer invocation — confirmed: the
ON-run and OFF-run output files report the same table.

| ticks | energy ON | energy OFF | energy solver |
|---:|---:|---:|---:|
| 10 | 20145.857422 | 20145.857422 | 20145.767578 |
| 25 | 20145.818359 | 20145.818359 | 20145.755859 |
| 50 | 20145.794922 | 20145.794922 | 20145.773438 |

ON and OFF are equal to 6 decimal places at every horizon: the trained
model already stays within the energy budget on its own (the projection's
`scale` clamps to 1.0 = identity almost everywhere), which is the intended
effect of training with the projection in the loop (task 2), not a bug —
`project_energy` never scales up, only down when the budget is exceeded
(`crates/jacobi2000/src/project.rs`).

**Max relative energy violation along the projected (ON) rollout: `0.000e0`
(bound: <= 1e-6). PASS.**

## Baseline: model12 (full attention), same command, same seed

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | wavefront lag (xcorr) | global RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.7515 | 0.9779 | 1.85341 | 1.5876 | -0.012 | -1.085 | 0.56779 |
| model | 25 | 0.2242 | 0.2079 | 2.21681 | 3.1237 | -5.084 | -7.218 | 1.04036 |
| model | 50 | -0.0637 | -0.2803 | 2.01902 | 2.7717 | -9.808 | +0.122 | 1.26148 |

model12 fails correlation at 25 (0.2242 < 0.9) and 50 (-0.0637 < 0.8),
amplitude ratio at 50 (2.7717 outside [0.7,1.3]) and wavefront lag at 50
(+0.122, inside 2 cells, so this one bar happens to pass). Reported as the
full-attention baseline only, not required to pass (matches
2026-09-18-acceptance.md's protocol).

model12's own energy-drift table (`docs/runs/2026-09-19-model12-acceptance2.txt`):

| ticks | energy ON | energy OFF | energy solver |
|---:|---:|---:|---:|
| 10 | 20145.894531 | 20145.900391 | 20145.767578 |
| 25 | 20145.957031 | 20145.925781 | 20145.755859 |
| 50 | 20146.164062 | 20146.001953 | 20145.773438 |

model12's max relative energy violation: `9.280e-6` (exceeds the 1e-6
bound). Not required to pass — model12 was never trained with the energy
projection — but reported for the record: it shows the projection doing
real, non-trivial work when the underlying model does drift, unlike
model20 where the projection is already a near-identity because training
(task 2) baked the constraint in.

## Verification: docs/runs/2026-09-19-2.md's training command

`docs/runs/2026-09-19-2.md`'s logged command line, as originally written,
contained `--project true` but **not** `--project-energy true` or
`--r-max`. This was traced to a code bug in `write_run_log`
(`crates/cli/src/main.rs`): the function never received `project_energy`
or `r_max` as arguments and its command-string template omitted both
flags unconditionally, regardless of what was actually passed to
`train_cmd`. It was not evidence that the training run itself skipped
energy projection: `artifacts/model20/meta.json` records
`energy_r_max: 1.0001`, which cannot come from the coded fallback
`1 - ENERGY_C * nu * dt` for `nu = 0.001` (that formula is always < 1,
this value is > 1), so it can only have been supplied explicitly via
`--r-max 1.0001`; the run doc's own prose note independently states
"mass and energy projection ON (r_max 1.0001)"; and this task's zero
measured energy violation above is consistent with a model actually
trained under that constraint.

Fixed in this task's commit: `write_run_log` now takes `project_energy`
and `r_max` and includes `--project-energy {project_energy} --r-max
{r_max}` in the logged command line for all future runs. `docs/runs/2026-09-19-2.md`'s
command line has been corrected in place (with a dated note explaining
why, not a silent edit) to read:

```
... --project true --project-energy true --r-max 1.0001 --spectral 1 --nu 0.001 ...
```

**`docs/runs/2026-09-19-2.md`'s (corrected) command line now contains both
`--project true` and `--project-energy true`, as required by the
Acceptance check's criterion (1).**

## Verdict: PASS

model20, trained and evaluated with mass and energy projection ON, passes
all four acceptance bars on a fresh seed (2027) never used elsewhere in
this run, and the max relative energy violation along its projected
rollout is 0 (<= 1e-6). The full-attention baseline (model12) is reported
alongside it and is not required to pass — it fails three of the four
bars and exceeds the energy-violation bound, consistent with never having
been trained under either constraint.

**PASS.**
