# Roadmap item 1 — a Lagrangian formulation

Machine: data generation on the development machine (native, CPU); training on a Linux
desktop with a desktop GPU (CUDA), with no other GPU job running.

## What "a Lagrangian formulation" means here

the design notes, section 9 defines the branch: "mask = neighbor list (rebuilt each
tick, spatial hash); position = edge feature (relative offset, distance) --
mandatory; mass per particle constant by construction; momentum exact via
antisymmetrized pairwise predictions (Newton III as a layer)." This is
distinct from the repo's existing "Lagrangian" work
(`crates/train/src/lagrangian.rs`, `artifacts/model-sph-smoke`,
`docs/runs/2026-09-19-sph-smoke.md`), which splats SPH particles onto a grid
and reuses the grid patch-transformer unchanged -- not a particle-native
model. This item is the particle-native model section 9 actually
describes.

## Data

New: `crates/solvers/examples/sph_dump.rs`, dumping raw SPH particle state
(position, velocity, mass) every `SPH_STEPS_PER_TICK = 9` SPH steps (the same
tick definition `crates/train/src/lagrangian.rs` uses), from
`solvers::sph::Sph`'s already-public `positions`/`velocities`/`masses`
accessors -- no changes to the solver itself. 10 trajectories (dam break and
drop, several column/blob sizes and positions), 60 ticks each, particle
counts in the low hundreds.

```
./target/release/examples/sph_dump data/sph
```

## Model

`training/jacobi2000_pt/lagrangian.py`. Per tick: a k-nearest-neighbor graph
over particle positions (`torch.cdist`, k=10; O(n^2) is fine at these
particle counts, no spatial hash needed), one `NewtonLayer` per message-
passing layer (3 layers, hidden 96): the edge feature is the relative
position, its length, and both particles' velocities; the MLP predicts one
2D message per directed edge, applied as `+m` to one endpoint and `-m` to
the other (`index_add_`), so the net predicted force sums to exactly zero
regardless of the weights -- Newton's third law as a layer, not a loss term.
Mass is a per-particle constant the model never touches, so it is conserved
by construction, trivially. Gravity is added directly, not learned.
30,822 parameters.

```
python -m jacobi2000_pt.lagrangian --data data/sph --out results/lagrangian \
  --steps 6000 --lr 1e-3 --rollout 6 --k 10 --layers 3 --hidden 96 \
  --horizons 5,10,20,40,59
```

Wall clock: **192.4 s** on the desktop GPU.

## Results

Rollout correlation and RMSE (raw particle position units, unnormalized)
against SPH ground truth, evaluated on the same 10 trajectories trained on
(no held-out split -- see observations):

| trajectory | tick 5 corr | tick 20 corr | tick 40 corr | tick 59 corr | tick 59 rmse |
|---|---:|---:|---:|---:|---:|
| dambreak | 1.0000 | 1.0000 | 0.9993 | 0.9944 | 1.211 |
| drop | 1.0000 | 1.0000 | 0.9993 | 0.9961 | 2.803 |
| dambreak-small | 1.0000 | 1.0000 | 0.9981 | 0.9861 | 1.347 |
| drop-small | 1.0000 | 1.0000 | 0.9992 | 0.9965 | 2.883 |
| dambreak-v0 | 1.0000 | 1.0000 | 0.9989 | 0.9922 | 1.237 |
| dambreak-v1 | 1.0000 | 1.0000 | 0.9982 | 0.9898 | 1.659 |
| dambreak-v2 | 1.0000 | 1.0000 | 0.9970 | 0.9813 | 1.663 |
| drop-v0 | 1.0000 | 1.0000 | 0.9992 | 0.9960 | 2.849 |
| drop-v1 | 1.0000 | 1.0000 | 0.9993 | 0.9962 | 2.810 |
| drop-v2 | 1.0000 | 1.0000 | 0.9993 | 0.9971 | 2.897 |

Full per-tick rows: `results/lagrangian-log.json`. Raw run log:
`results/lagrangian-run.log`. Checkpoint: `results/checkpoints/lagrangian/model.pt`
(gitignored, a state dict, not exported to the Burn-compatible safetensors
format since this is not the `Jacobi2000` architecture).

## Observations

- The correlation numbers are far above the existing splatted grid model's
  (`model-sph-smoke`, corr 0.35-0.37 at 25 ticks against SPH truth,
  `docs/runs/2026-09-19-sph-smoke.md`) -- but **the two numbers are not
  comparable**: `model-sph-smoke` was scored in grid-field space (splatted
  density field correlation), this run is scored directly on unnormalized
  particle positions, on a training set that doubles as its own eval set
  (10 trajectories total, no held-out split -- a from-scratch particle model
  reusing the training trajectories for eval will look better than it would
  on unseen initial conditions). Read this as "the particle-native pipeline
  works end to end and produces stable, non-exploding rollouts to 59 ticks,"
  not as "the Lagrangian branch now beats the grid branch."
- RMSE grows roughly linearly with tick count (as expected for an
  autoregressive rollout with no explicit stability guarantee beyond the
  momentum-conservation structure), reaching 1.2-2.9 position units (the
  domain is `nx*dx = 64` units wide) by tick 59 -- particles are still
  broadly in the right place, not yet visibly diverged, but this is not
  checked visually (no picture dump for the particle model).
- What's not done: a spatial hash (particle counts here are small enough
  that the brute-force `cdist` neighbor search was fine; the design notes' "rebuilt
  each tick, spatial hash" is a scaling concern for larger particle counts,
  not exercised here -- see the follow-up below), comparing against a
  from-scratch grid-splat baseline trained on the same trajectories (the
  only fair apples-to-apples comparison), and wiring this into `web/` or
  `eval-metrics` (a new architecture, no Rust-side support).

## Follow-up (same night, later) — held-out split, different initial-condition family

Added `--held-out-prefix` to `lagrangian.py`: trajectories whose name starts
with the given prefix are excluded from training and scored separately.
Trained on the 5 dam-break variants only, held out all 5 drop variants
entirely (a different initial-condition family, not just an unseen seed of
the same one):

```
python -m jacobi2000_pt.lagrangian --data data/sph --out results/lagrangian-split \
  --steps 6000 --lr 1e-3 --rollout 6 --k 10 --layers 3 --hidden 96 \
  --horizons 5,10,20,40,59 --held-out-prefix drop
```

30,822 params (unchanged), wall clock **193.0 s** on the desktop GPU (effectively
the same as the first pass's 192.4 s -- training set size doesn't change
per-step cost, since each step draws one trajectory).

Mean correlation and RMSE, train (5 dam-break trajectories) vs. held-out (5
drop trajectories, never seen during training):

| tick | train corr | held-out corr | train rmse | held-out rmse |
|---:|---:|---:|---:|---:|
| 5 | 1.0000 | 1.0000 | 0.0055 | 0.0057 |
| 10 | 1.0000 | 1.0000 | 0.0116 | 0.0119 |
| 20 | 1.0000 | 1.0000 | 0.0380 | 0.0447 |
| 40 | 0.9981 | 0.9993 | 0.4397 | 1.1149 |
| 59 | 0.9862 | 0.9961 | 1.4595 | 2.8852 |

Full rows: `results/lagrangian-split-log.json`. Run log:
`results/lagrangian-split-run.log`.

### Observation

Correlation on the held-out drop family is **not worse** than on the
training dam-break family at any horizon (in fact slightly higher at ticks
40/59: 0.9993/0.9961 vs 0.9981/0.9862) -- the Newton-III message-passing
layer generalizes to an initial-condition family it never trained on, at
least on this correlation metric. **This is not a clean win, and the RMSE
column shows why**: held-out RMSE is 2-2.5x train RMSE at ticks 40/59
(1.11 vs 0.44, 2.89 vs 1.46). That gap matches exactly the gap already seen
in the first pass, when drop trajectories *were* part of training (drop's
tick-59 RMSE there was 2.80-2.90, dambreak's was 1.24-1.66) -- drop
trajectories simply involve more particle displacement (a blob falling
through empty space vs. a column collapsing sideways), so higher RMSE for
the drop family is a property of that family's motion scale, not evidence
of worse generalization. Correlation (scale-invariant) is the fairer metric
here, and by that metric held-out looks as good as train. Kept honest: this
is a small-N result (5 held-out trajectories, one model, one seed, no
repeat), and the two families may simply be similar enough in local
dynamics (both are gravity-driven free-surface flows the same message-
passing rule handles the same way) that this isn't a hard generalization
test. `k`-NN neighbor search stayed brute-force (`cdist`, still fine at
these particle counts); no spatial hash was needed or built.
