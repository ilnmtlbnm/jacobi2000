# Excess-energy band: a two-sided energy constraint, in Rust and PyTorch, and the width sweep it is for

Date: 2026-10-06. Branch `energy-band`, off `roadmap-night`.

Machines:
- Apple M2, for the solver statistics, the tests, the Rust/PyTorch match and a pipeline smoke run.
- RTX 3080, for the planned sweep in section 6. It has not been run yet.

Raw `energy-stats` outputs are in `docs/runs/2026-10-06-energy-band-raw/`.

## 1. Why the existing energy projection never acts

`project_energy` bounds the total energy: `E_new <= E_prev * r_max`, with `r_max = 1.0001`.

The total energy is about 20,150 on the training distribution, and almost all of it is the
flat pond's `1/2 g H^2`. The part a wave carries is about 0.6 (p50). So the bound lets the
total rise by about 2 units per tick, which is more than the whole wave's energy, every tick.

The "max relative energy violation 0" rows in `2026-09-19-rung6.md` and
`2026-09-29-pytorch-training.md` come from this looseness, not from a model that conserves
energy.

## 2. The quantity that is actually conserved: excess energy with the scheme's cross term

The plain excess energy is `E' = sum 1/2 g (h - hbar)^2 + 1/2 h |u|^2`. With mass held fixed,
it changes by exactly as much as the total energy does, but it has no f32 cancellation against
the flat-pond baseline.

On the solver's own trajectories, `E'` is not monotone at any viscosity this repo trains: it
swings by up to ±0.3 % per tick. The reason is the forward-backward scheme
(`crates/solvers/src/shallow_water.rs`). As in symplectic Euler, it conserves a modified,
time-staggered energy rather than the collocated one.

Adding the cross term `-(dt/dx)(g/2) sum (h - hbar) div(h u)` restores that conserved quantity.
Here `div(h u)` is the solver's own face flux. The table shows the per-tick ratio
`E'_{t+1}/E'_t` on the training mixture.

`energy-stats --pool 32 --ticks 200 --seed 42` gives 6,393 windows; 7 windows with a
mid-window drop injection are skipped:

| nu | form | ticks up | min | p50 | p99 | max |
|---:|---|---:|---:|---:|---:|---:|
| 0 | plain | 3125 | 0.9930590 | 0.9999935 | 1.0012876 | 1.0028242 |
| 0 | cross term | 4422 | 0.9999289 | 1.0000052 | 1.0001487 | 1.0003629 |
| 0.001 | plain | 3045 | 0.9930389 | 0.9999867 | 1.0012794 | 1.0028234 |
| 0.001 | cross term | 3149 | 0.9999205 | 0.9999998 | 1.0001411 | 1.0003480 |
| 0.01 | plain | 2459 | 0.9928510 | 0.9999201 | 1.0011934 | 1.0028177 |
| 0.01 | cross term | 427 | 0.9997814 | 0.9999518 | 1.0000855 | 1.0002143 |

Larger samples of the cross-term ratio, with drop injections skipped at |Δmass| > 1e-6:

| nu | sample | ticks | min | p50 | max |
|---:|---|---:|---:|---:|---:|
| 0.001 | `energy-stats --pool 128 --ticks 400 --seed 7` | 51,140 | 0.9997747 | 0.9999989 | 1.0003457 |
| 0.001 | `dump-dataset --pool 300 --seed 11`, scored with `project.py` | 39,411 | 0.9998135 | 0.9999982 | 1.0003122 |
| 0.01 | `dump-dataset --pool 300 --seed 11`, scored with `project.py` | 39,411 | 0.9996733 | 0.9999523 | 1.0002059 |

Viscous decay at these `nu` is small. With small drops at `nu = 0.01`, the cross-term `E'` never
rises over 200 ticks and ends at 0.991 of its start (`raw/pub-stats-nu0.01.txt`). At
`nu = 0.001` the per-tick swing (±3e-4) is far larger than the decay.

## 3. Implementation

- **Rust.** `crates/jacobi2000/src/project.rs` defines `excess_energy(x, d, g, dt_dx)`,
  `EnergyBand { r_min, r_max }` and `project_excess_energy(x, prev, d, g, dt_dx, band)`.
  - The fluctuation `(eta, u, v)` is scaled by one detached `s` per batch member: down when
    the proposal is above the band, up when it is below.
  - `eta` keeps zero mean, so mass is exact.
  - The excess energy as a function of `s` is `s^2 a + s^3 q`. `s` starts from the closed form
    `sqrt(T / (a + q))` and two Newton steps reach the root, where `T` is the violated edge
    pulled 1e-5 inside the band.
  - Members already inside the band are returned bit-for-bit.
  - `EnergyBand::upper(r)` (`r_min = 0`) is the upper-only form.
- **Rust flags.** In `train` and `eval` / `eval-metrics`, `--project-excess <r_max>` and
  `--excess-r-min <r_min>`. `train` stores them in `meta.json` as `excess_r` and
  `excess_r_min`; eval uses the checkpoint's values unless given others, and
  `--project-excess 0` turns the projection off.
  - `eval-metrics` prints `E'_t/E'_0` for the rollout with the constraint ON, the rollout
    with it OFF, and the solver, plus the number of ticks that were outside the band before
    projection.
  - The total-energy drift section now runs only with `--project-energy true`. Old commands
    pass that flag, so their output is unchanged.
  - `energy-stats` prints the tables in section 2 and suggests `--project-excess` and
    `--excess-r-min`.
- **PyTorch.** `training/jacobi2000_pt/project.py` has `excess_energy` and
  `project_excess_energy(x, prev, d, g, dt_dx, r_min, r_max)`, line for line the Rust version.
  - `train.py` adds `--excess-r-max` (off when absent) and `--excess-r-min` (default 0).
  - The training log reports the fraction of proposals outside the band.
  - `export.py` writes `excess_r` and `excess_r_min` to `config.json`, so
    `jacobi import-weights` carries them into the Burn checkpoint's metadata.

## 4. Rust and PyTorch agree

Six 64x64 solver states: two drops, 7 to 42 ticks, `nu = 0.001`, model20's normalizer.
`prev` is set so that members fall below, above, inside, far below and far above the band
`[0.9998, 1.00035]`. The Rust (ndarray) outputs were dumped and recomputed in PyTorch (CPU):

| quantity | max difference |
|---|---:|
| `E'`, cross term (relative) | 1.8e-6 |
| `E'`, plain (relative) | 1.9e-6 |
| projected fields (absolute, normalized units; field max 24.9) | 2.3e-5 (9.2e-7 relative) |
| active members | identical (5 of 6 in both) |

Both differences are at the level of f32 summation order.

## 5. Tests

All pass.

Rust (`cargo test --release -p jacobi2000 -p train -p solvers -p cli`), in
`crates/jacobi2000/src/project.rs` and `crates/train/src/fit.rs`:
- the upper-only projection never raises `E'` and keeps mass (bounds at 0 %, 10 %, 90 % and
  99.99 %);
- a mixed batch: members inside the band come back bit-for-bit;
- finite gradient at a zero budget;
- `E'` has the same increments as the total energy at fixed mass;
- `excess_band_lifts_a_proposal_below_the_band_and_keeps_mass`: one member below, one inside,
  one above the band;
- `solver_excess_energy_with_cross_term_never_rises_at_nu_0_01`: the cross-term `E'` never
  rises beyond 1e-5 on a solver trajectory, the plain form rises on more than 50 of 200 ticks,
  and the band `[0.99965, 1.00022]` from the measured swing leaves every state untouched.

PyTorch (`cd training && python -m unittest discover tests`, `training/tests/test_project.py`):
- `never_leaves_the_band_and_keeps_mass`: members far below, just below, inside, just above
  and far above the band, plus a zero budget;
- `upper_only_form_with_r_min_zero`;
- `identity_inside_the_band` (bit-for-bit);
- `gradient_finite_at_zero_budget_and_when_lifting`;
- `solver_trajectory_untouched_by_its_measured_band`: it calls `jacobi dump-dataset --pool 2
  --nu 0.01 --seed 3`, normalizes the way training does, and checks that the `[0.99965,
  1.00022]` band is the identity on every solver tick. It needs the release binary and fails
  if it is missing.

Caution: a band taken from the measured extremes is an envelope, not a guarantee. A fresh
trajectory can step just outside it. In the first draft of this test, an `energy-stats` band
from 6,393 ticks was beaten by one tick of a new trajectory (0.999773 vs 0.999780). The sweep
therefore uses edges from about 90,000 ticks (section 6).

## 6. 3080 sweep (prepared, not run)

**Question.** In `2026-09-19-rung6.md`, widening 128/4 to 256/6 bought nothing. At a matched
budget, with seeds, does a hard energy constraint change that verdict or reduce seed-to-seed
spread?

Measured on the Burn path the same day:
- model20 loses energy: `E'` is 0.94 of its start by tick 50 while the solver holds. An upper
  bound never touches it, which is why this sweep includes the lower edge.
- A 200-step warm start of the same weights went the other way: +14 % by tick 50.
- Applying the upper bound only at inference cut that checkpoint's active RMSE@50 from 0.102
  to 0.082.

**Design.**
- 2 sizes × 3 training arms × 3 seeds = 18 trainings:
  - sizes: 128/4 and 256/6;
  - arms: off, upper (`r_max`), band (`r_min`, `r_max`);
  - seeds: 20, the seed of `model-wide-1`, plus 21 and 22.
- All on the rung6 width-comparison recipe (`2026-09-19-3.md`): 3000 steps, lr 3e-4, batch 16,
  rollout 4 after 200 steps, noise 0.04, spectral 1, constant lr, mass projection on, total
  energy projection off, from scratch. `train.py`'s defaults already match this.
- Data: the section-3 dump of `2026-09-29-pytorch-training.md` (`dump-dataset --pool 1000 --nu
  0.001 --seed 20`).
- Band: `r_min = 0.99975`, `r_max = 1.00036`, the extremes of the two `nu = 0.001` samples in
  section 2 with a small margin.
- Every checkpoint is imported into Burn and scored with the standard rung6 eval (`eval-metrics
  --horizons 10,25,50 --trajectories 24 --seed 0 --project true`):
  - with the constraint off and with its own constraint on;
  - the off-trained checkpoints also get both the upper and the band constraint at inference.
  - That is 42 evals.
- Script: `tools/energy-band-sweep.sh`. It is resumable, refuses to start if the GPU is in use,
  and writes `results/band/summary.md`, which holds the per-run metrics table plus `E'_50/E'_0`
  (ON, OFF, solver) and the active-tick counts.
- Smoke-tested end to end on the M2 (`STEPS=3 TRAJ=1`, 20-trajectory dump): 18 trainings,
  18 imports, 42 evals, and the summary.

**Time estimate (3080).**

| item | basis | estimate |
|---|---|---:|
| 128/4 training | 0.018 s/step at rollout 1, 0.0357 s/step at rollout 4 (curriculum stages s1/s2, section 6 of `2026-09-29-pytorch-training.md`): 200 × 0.018 + 2800 × 0.0357 | ~104 s, ~115 s with the constraint |
| 256/6 training | no PyTorch measurement; compute per token scales 6x (dim² × layers), the small model is partly overhead-bound, so 4x (range 2-6x) | ~430 s (210-640) |
| 18 trainings | 3 × 3 × (110 + 450) | ~1.4 h (0.8-2.0) |
| evals, wgpu | M2 GPU: 178 s for a 128/4 band eval at 24 trajectories with 5 rollouts per trajectory (provisional: another process held part of the GPU); with the drift section gated, off = 1 rollout, on = 3; 256/6 is 2.5x; 3080 assumed no faster (batch-1 rollouts are launch-bound) | ~1.6 h (1-2.5) |
| remote build (wgpu, fresh target) | | ~10 min |
| total | | **~3.2 h (range 2-5 h)** |

To check the estimate early, the first `train d128l4-s20-off` and `eval ...` lines in the log
carry timestamps.

**Commands.** `<box>` is the GPU host, `<box-dir>` a fresh directory under its agent work area,
`<venv>` an existing virtualenv there with torch 2.14, `<box-lock>` the lock file the box's GPU jobs serialize on, and `<dump>` the section-3 1000-trajectory
dump already on the box.

```
# from this worktree
rsync -az --delete --exclude target/ --exclude target-wgpu/ --exclude results/ --exclude data/ \
  --exclude web/pkg/ --exclude web/node_modules/ ./ <box>:<box-dir>/
ssh <box> 'nvidia-smi; systemctl --user list-units --no-pager | grep -E "chain-|bench-|judge-|dl-|jacobi"; pgrep -af llama-server'
ssh <box> 'cd <box-dir> && source ~/.cargo/env && CARGO_TARGET_DIR=target-wgpu cargo build --release -p cli --features wgpu --no-default-features'
ssh <box> 'cd <box-dir> && systemd-run --user --unit=jacobi-band --same-dir --collect --property=Type=simple \
  --setenv=DATASET=<dump> --setenv=PYTHON=<venv>/bin/python --setenv=JACOBI=./target-wgpu/release/jacobi \
  /usr/bin/flock <box-lock> /bin/bash tools/energy-band-sweep.sh'
# status
ssh <box> 'systemctl --user status jacobi-band --no-pager; journalctl --user -u jacobi-band -n 20 --no-pager; ls <box-dir>/results/band | tail'
# fetch (summaries, logs, eval outputs; checkpoints optional)
rsync -az --exclude '*-burn/' --exclude '*.safetensors' <box>:<box-dir>/results/band/ results/band/
```

**How to read the result.** For each size, training arm and inference arm, take the mean ±
std over the 3 seeds of:
- active RMSE@50,
- amp@50,
- corr@50,
- `E'_50/E'_0`.

The constraint earns its place if either holds:
- the band (or upper) arms have a smaller seed std than off;
- 256/6 beats 128/4 under a constraint but not without one.

If every difference sits inside the seed std, energy drift is not what holds the wider model
back, and the next suspect is the training budget itself.
