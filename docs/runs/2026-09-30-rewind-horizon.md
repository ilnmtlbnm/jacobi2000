# Roadmap item 4 — negative time, building on the rewind model

Machine: data generation on an Apple M2 (native, CPU); training and eval on
a Linux desktop with an RTX 3080 (CUDA), under `flock`, no other GPU job
running.

## What this is

CONCEPT.md section 11 lists a measurable never yet produced in this repo's
run log: "rewind horizon (ticks until ensemble spread exceeds threshold) vs
viscosity." Section 4.3 (the Janssen loop): "Ensemble over discretizations,
not over models: jitter the patch-grid offset per realization, same weights,
average + spread... Gives uncertainty for free." Section 6's rewind exhibit
predicts the shape: dragging time backward, ripples collapse to a point,
then "the moment it starts smearing is the information horizon of that
physics... switch viscosity on: the horizon collapses to almost nothing."
This builds directly on the existing reverse checkpoint and round-trip work
(`docs/runs/2026-09-20-rewind-round-trip.md`,
`docs/runs/2026-09-29-pytorch-training.md` section 8), which measured
correlation vs horizon but never the ensemble-spread threshold-crossing
horizon itself.

## Method

New: `training/jacobi2000_pt/rewind_horizon.py`. Starts `K=8` ensemble
members at the same "present" state (tick 40 of a held-out trajectory),
steps each one backward with an independently jittered patch-grid offset
every tick (`torch.roll` by a random `(dy, dx)` in `[0, patch)` before the
forward pass, rolled back after -- the jittered-discretization idea from
section 4.3, without threading a patch-offset argument through the Rust
model), and reports the tick at which the per-cell ensemble standard
deviation first exceeds a fixed threshold (0.08, in normalized field units,
so the same number means the same thing at every viscosity).

## Checkpoints

Three forward+reverse pairs at nu in {0, 0.001, 0.01}, **not matched in
training budget** (stated plainly, see observations):

| nu | forward checkpoint | forward recipe | reverse checkpoint | reverse recipe |
|---:|---|---|---|---|
| 0 | `results/fwd-nu0` | 1000 steps from scratch, 300-trajectory dataset, 77.4 s | `results/rev-nu0` | 700 steps warm-started from fwd-nu0, 55.0 s |
| 0.001 | `exports/curriculum/s6-mass-energy-proj` | 6-stage curriculum, 1000-trajectory dataset, 565.1 s total (`docs/runs/2026-09-29-pytorch-training.md` section 6) | `exports/reverse` | 700 steps warm-started from s6, 1000-trajectory dataset, 54.3 s (section 8 of the same doc) |
| 0.01 | `results/fwd-nu01` | 1000 steps from scratch, 300-trajectory dataset, 78.2 s | `results/rev-nu01` | 700 steps warm-started from fwd-nu01, 55.2 s |

```
./target/release/jacobi dump-dataset --pool 300 --nu 0.0 --seed 20 --out data/sw-nu0
./target/release/jacobi dump-dataset --pool 300 --nu 0.01 --seed 20 --out data/sw-nu01
python -m jacobi2000_pt.train --dataset data/sw-nu0 --out results/fwd-nu0 \
  --steps 1000 --lr 1e-4 --rollout 8 --project true --project-energy true --dim 128 --layers 4 --seed 100
python -m jacobi2000_pt.train --dataset data/sw-nu0 --out results/rev-nu0 \
  --init results/fwd-nu0 --steps 700 --reverse true --seed 30 [same other flags]
# nu=0.01 identical shape
```

## Results

```
python -m jacobi2000_pt.rewind_horizon --reverse <checkpoint> --dataset <matching dataset> \
  --traj-index 0 --start-tick 40 --k 8 --max-ticks 80 --threshold 0.08 --seed 0
```

| nu | spread @ 10 | @ 20 | @ 40 | @ 60 | @ 80 | rewind horizon (threshold 0.08) |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.00531 | 0.00799 | 0.01421 | 0.02401 | 0.04054 | not reached by tick 80 |
| 0.001 | 0.00883 | 0.02209 | 0.04210 | 0.07193 | 0.10228 | **tick 62** |
| 0.01 | 0.00494 | 0.00782 | 0.01610 | 0.02844 | 0.05308 | not reached by tick 80 |

Raw spread curves: `results/rewind-nu0.json`, `results/rewind-nu001.json`,
`results/rewind-nu01.json`. Full run log: `results/rewind-run.log`.

## Observations

- **This is not the clean "horizon vs viscosity" result CONCEPT predicts**,
  and the likely reason is visible in the checkpoint table above, not in the
  physics: nu=0.001's reverse model is warm-started from the curriculum's
  fully-converged, 6-stage, 1000-trajectory s6 checkpoint (the repo's best),
  while nu=0 and nu=0.01's reverse models are warm-started from a single
  1000-step from-scratch forward checkpoint on a 300-trajectory dataset --
  much less trained. The nu=0.001 ensemble's spread is 2-3x every other
  run's at every tick (0.10 vs 0.04-0.05 at tick 80), which is at least as
  consistent with "the better-trained model is sharper and more sensitive to
  patch-grid jitter" as with "nu=0.001 is less reversible" -- the two are
  confounded here, and this run cannot separate them. **A clean version of
  this experiment needs three forward+reverse pairs trained to the same
  budget** (ideally a matched short curriculum, not the full 6-stage one
  for nu=0.001 alone); not done tonight, time budget.
- Taken at face value anyway: nu=0 and nu=0.01 track each other closely at
  every tick (0.0050 vs 0.0049 at tick 10, 0.0405 vs 0.0531 at tick 80) --
  no signal in this run that inviscid (theoretically well-posed, reversible)
  and viscous (theoretically ill-posed) rewind behave differently at this
  training budget and threshold. That absence of a signal is itself
  consistent with the confound above: two undertrained models of similar
  quality look similar regardless of the physics.
- The spread threshold (0.08) and the single held-out trajectory / start
  tick were chosen once and not tuned; a threshold sweep or averaging over
  several start trajectories would tighten this measurement. Not attempted.
- What would make this a real test of section 11's measurable: matched
  training budgets across nu, a threshold sweep, and comparing the ensemble-
  spread horizon against the correlation-based rewind horizon
  (`docs/runs/2026-09-20-rewind-round-trip.md`'s approach) on the same
  checkpoints, to see whether the two horizon definitions agree.
