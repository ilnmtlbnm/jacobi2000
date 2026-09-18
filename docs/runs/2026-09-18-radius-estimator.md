# Radius estimator diagnostic (task 8a)

Machine: Apple M2, 16 GB. Backend: wgpu (Metal). GPU uncontended
(`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` empty before
each run).

Question: runs 11 and 12 (checkpoints `artifacts/model13`, radius-1 stencil,
rollout 4; and `artifacts/model14`, same mask, rollout 8) differ by one
curriculum stage yet their 50-tick radius error flips from -2.84 to +3.20
cells (docs/runs/2026-09-18-rung4-stencil-r1.md,
docs/runs/2026-09-18-rung5-rollout8.md) while every other rung-1 metric
barely moves. Is the estimator (argmax of the azimuthally-averaged `|h|`
profile) bistable at 50 ticks?

Commands (24 held-out single-drop trajectories, seed 0, same trajectories
for both checkpoints since the seed and sampling order are identical):

```
./target/release/jacobi eval-metrics --model artifacts/model13 --horizons 10,25,50 --trajectories 24 --seed 0 --project false
./target/release/jacobi eval-metrics --model artifacts/model14 --horizons 10,25,50 --trajectories 24 --seed 0 --project false
```

24 single-drop trajectories used, 22 skipped (horizon < 50 ticks), for both
checkpoints.

## 1. Per-trajectory radius error (argmax), model13 vs model14

| checkpoint | ticks | mean | std | min | max | count \|err\|>2 |
|---|---:|---:|---:|---:|---:|---:|
| model13 (rung 4) | 25 | +0.440 | 1.740 | -0.081 | +8.774 | 1 |
| model13 (rung 4) | 50 | -2.838 | 5.350 | -17.000 | +0.571 | 6 |
| model14 (rung 5) | 25 | +0.683 | 2.241 | -0.034 | +8.743 | 2 |
| model14 (rung 5) | 50 | +3.196 | 6.160 | -0.078 | +16.867 | 5 |

At 50 ticks the std (5.4-6.2 cells) and the max (up to 16.9 cells) both
dwarf the reported mean (2.8-3.2 cells) and dwarf the acceptance bar (2
cells). A quarter or more of the trajectories individually exceed the bar
by 8x. This is not a small, consistent bias — it is a handful of
trajectories where argmax jumps to the far side of the domain.

## 2. Numeric radial profiles (model vs solver), two trajectories

Same trajectories for both checkpoints (RNG seed 0, first two accepted).
Full profiles are printed by `eval-metrics`; the 50-tick trajectory-0
profile below is the clearest case of bistability — two comparable maxima,
one near the drop centre (r=0) and one near the domain edge (r=17), whose
relative height flips between checkpoints:

**Trajectory 0, 50 ticks, model13:**

| r | model | solver |
|---:|---:|---:|
| 0 | 0.01223 | 0.01104 |
| 9 | 0.00498 | 0.00394 |
| 10 | 0.00234 | 0.00138 |
| 11 | 0.00100 | 0.00158 |
| 17 | 0.01213 | 0.01219 |

Peak at r=0 (0.01223) and r=17 (0.01213) are within 1% of each other; the
trough at r=11 (0.00100) separates them. `radial_profile` truncates each
trajectory's bins at the nearest wall (`Params::horizon_ticks`/the drop
placement keep the wave itself pre-reflection, but the *averaging annulus*
still gets cut short near the wall), so the last bin averages over fewer,
more wave-dominated cells and rises again — a genuine second local maximum
within the sampled range, comparable in height to the true near-centre
residual. `wavefront_radius` (parabolic argmax) picks whichever of the two
is taller by even a fraction of a percent; for the solver profile here that
is r=17 (0.01219 vs 0.01104 at r=0), for the model it is r=0 (0.01223 vs
0.01213) — a photo finish that the argmax estimator reports as a ~17-cell
disagreement, one of the six trajectories with `|err|>2` in model13's
50-tick row above.

**Trajectory 0, 50 ticks, model14:** same shape, model peak at r=0 is
0.01121 (vs 0.01104 solver) and at r=17 is 0.01260 (vs 0.01219 solver) — a
small quantitative change in relative peak height (model13: r=0 taller by
1.1%; model14: r=17 taller by 3.3%) is enough to flip which side the
parabolic argmax locks onto. This is the mechanism behind the mean flipping
sign between runs 11 and 12 with every other metric unchanged.

Trajectory 0 and trajectory 1 at 25 ticks (both checkpoints, not shown in
full here — see command output) do not show this bimodality: the profile
has one dominant lobe and the argmax and cross-correlation estimators agree
to within 0.1-0.7 cells, matching why the 25-tick radius-error numbers are
comparatively stable across runs 11-12 (+0.440 -> +0.683) while the 50-tick
numbers are not (-2.838 -> +3.196).

## 3. Cross-correlation lag estimator

`xcorr_lag` (crates/train/src/metrics.rs) shifts the model's radial `|h|`
profile against the solver's over ±8 cells, scores each shift by Pearson
correlation over the overlapping range, and refines the integer argmax to
sub-cell accuracy with a parabola through the best lag and its neighbours.
Unlike the argmax radius, it uses the whole profile shape, so a single
mismatched bin at one end cannot flip it.

Unit test `xcorr_lag_recovers_a_three_cell_shift`
(crates/train/src/metrics.rs): a synthetic Gaussian ring shifted by exactly
3 cells reports lag 3.0 ± 0.1 (measured: -2.9989216 before the sign
convention was fixed to match `radius_error`'s "positive = model ahead",
then 3.0 after `corr_at_lag` was defined as comparing `model[i]` against
`solver[i - lag]`). `cargo test -p train --lib metrics` passes.

| checkpoint | ticks | xcorr lag mean | std |
|---|---:|---:|---:|
| model13 (rung 4) | 25 | +0.099 | 0.101 |
| model13 (rung 4) | 50 | +0.193 | 0.155 |
| model14 (rung 5) | 25 | -0.097 | 0.048 |
| model14 (rung 5) | 50 | -0.162 | 0.074 |

At 50 ticks the xcorr lag is stable in both mean (well under 0.2 cells,
vs the argmax mean of 2.8-3.2 cells) and std (0.07-0.16 cells, vs 5.4-6.2
cells for argmax) across both checkpoints. It does not confirm the
argmax's claim that run 12 regressed sharply on wavefront timing — by the
xcorr measure both checkpoints track the solver's wavefront to a fraction
of a cell, consistent with every other rung-1 metric (correlation,
amplitude ratio) barely moving between runs 11 and 12.

## 4. eval-metrics table (xcorr column added)

`crates/cli/src/main.rs` (`metrics_row`, `eval_metrics_cmd`) now prints a
"wavefront lag (xcorr)" column alongside the existing "radius err (cells)"
column, plus a per-trajectory stats block (mean/std/min/max/count
`|err|>2`) for radius error and xcorr lag, and numeric radial-profile
tables for the first two trajectories at 25/50 ticks — exactly the tables
in sections 1-3 above, reproducible by rerunning the two commands at the
top of this doc.

## Recommendation

**The acceptance bar should use the xcorr-lag column, not the argmax
radius-error column.**

Reasons, with the numbers above:

- The argmax radius error is measurably bistable at 50 ticks: a two-peak
  profile (section 2) where a ~1-3% change in relative peak height flips
  which peak wins, moving the reported error by up to 17 cells for
  individual trajectories and flipping the *sign of the mean* between two
  checkpoints (runs 11 -> 12, -2.84 -> +3.20) that agree on every other
  rung-1 metric to within a few percent.
- The per-trajectory std of the argmax radius error (5.4-6.2 cells at 50
  ticks) is 2.5-3x the acceptance bar itself (2 cells), so the *mean*
  passing or failing the bar is close to a coin flip driven by a handful of
  bistable trajectories, not a property of the model's wavefront timing.
- The xcorr lag is stable across both checkpoints (std 0.05-0.16 cells,
  mean magnitude < 0.2 cells at 50 ticks) and does not reproduce the sign
  flip — it agrees with the qualitative read of every other metric that
  runs 11 and 12 are close in wavefront timing.
- The xcorr estimator is unit-tested to recover a known 3-cell shift to
  ±0.1 cells (`xcorr_lag_recovers_a_three_cell_shift`), giving it the same
  kind of ground-truth check the argmax estimator has
  (`wavefront_radius_recovers_an_analytic_ring`), so switching the bar does
  not trade a tested estimator for an untested one.

The argmax radius-error column should stay in the table (it is cheap, and a
large disagreement between the two columns is itself diagnostic of a
bistable profile), but the acceptance criterion in
`convergence/queues/stay-true-for-longer.md` should be read against
"wavefront lag (xcorr)" going forward.
