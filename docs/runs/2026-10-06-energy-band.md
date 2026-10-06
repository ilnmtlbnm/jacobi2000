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
`<venv>` an existing virtualenv there with torch 2.14, `<box-lock>` the lock file GPU jobs serialize on, and `<dump>` the section-3 1000-trajectory
dump already there.

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

## 7. Results of the 3080 sweep (run 2026-10-06, 1 h 12 min wall)

Raw outputs are in `results/band/`: per-run eval outputs, training logs, the training-log
JSONs, and `summary.md`. The box path in the training logs' "exported to" line is replaced by
`<box-dir>`.

The script died in its summary tail. With `set -euo pipefail`, the grep for the `E'` section
returned nonzero on `*.eval-off.txt` files. That is fixed in `89b5eef`. `summary.md` was
regenerated locally from the raw files with the fixed tail; nothing was rerun.

The table below is at 50 ticks, mean ± sd over seeds 20/21/22: 24 held-out trajectories per
eval, the rung6 protocol.

| size | train | inference | n | corr@50 | active RMSE@50 | amp@50 | global RMSE@50 | E'_50/E'_0 (ON) | active ticks % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 128/4 | off | off | 3 | 0.9817 ± 0.0074 | 0.1312 ± 0.0250 | 1.043 ± 0.033 | 0.1077 ± 0.0183 | – | – |
| 128/4 | off | upper | 3 | 0.9810 ± 0.0076 | 0.1199 ± 0.0199 | 0.996 ± 0.014 | 0.1006 ± 0.0160 | 1.0154 ± 0.0019 | 92.7 ± 4.8 |
| 128/4 | off | band | 3 | 0.9810 ± 0.0076 | 0.1199 ± 0.0199 | 0.997 ± 0.014 | 0.1006 ± 0.0160 | 1.0162 ± 0.0010 | 95.5 ± 2.4 |
| 128/4 | upper | off | 3 | 0.9913 ± 0.0012 | 0.1138 ± 0.0344 | 1.034 ± 0.078 | 0.0839 ± 0.0195 | – | – |
| 128/4 | upper | upper | 3 | 0.9912 ± 0.0011 | 0.0943 ± 0.0078 | 0.997 ± 0.018 | 0.0720 ± 0.0065 | 0.9775 ± 0.0403 | 41.0 ± 49.1 |
| 128/4 | band | off | 3 | 0.0679 ± 0.0060 | 0.8841 ± 0.0002 | 0.146 ± 0.003 | 0.5520 ± 0.0003 | – | – |
| 128/4 | band | band | 3 | 0.0083 ± 0.0010 | 0.9148 ± 0.0057 | 16.054 ± 7.536 | 0.7833 ± 0.0402 | 0.9881 ± 0.0000 | 100.0 ± 0.0 |
| 256/6 | off | off | 3 | 0.9822 ± 0.0151 | 0.1470 ± 0.0711 | 1.029 ± 0.027 | 0.1205 ± 0.0674 | – | – |
| 256/6 | off | upper | 3 | 0.9813 ± 0.0163 | 0.1305 ± 0.0717 | 0.974 ± 0.033 | 0.1095 ± 0.0667 | 1.0078 ± 0.0135 | 76.6 ± 31.8 |
| 256/6 | off | band | 3 | 0.9813 ± 0.0163 | 0.1304 ± 0.0718 | 0.976 ± 0.036 | 0.1095 ± 0.0666 | 1.0124 ± 0.0073 | 88.8 ± 14.9 |
| 256/6 | upper | off | 3 | 0.9866 ± 0.0060 | 0.1361 ± 0.0583 | 1.029 ± 0.059 | 0.1083 ± 0.0454 | – | – |
| 256/6 | upper | upper | 3 | 0.9859 ± 0.0070 | 0.1222 ± 0.0368 | 0.984 ± 0.010 | 0.0999 ± 0.0326 | 1.0037 ± 0.0113 | 62.1 ± 31.4 |
| 256/6 | band | off | 3 | 0.0647 ± 0.0094 | 0.8846 ± 0.0005 | 0.149 ± 0.009 | 0.5523 ± 0.0008 | – | – |
| 256/6 | band | band | 3 | 0.0057 ± 0.0012 | 0.9111 ± 0.0055 | 11.180 ± 3.325 | 0.8187 ± 0.0394 | 0.9881 ± 0.0000 | 100.0 ± 0.0 |

`E'_50/E'_0 (ON)` is the excess energy of the projected rollout; the solver's own value is
1.0005. "Active ticks" counts the ticks whose proposal was outside the bound before projection.

### 7a. The band-trained checkpoints collapse: a training bug, fixed

The symptoms appear on all 6 band-trained runs:
- corr@50 is 0.004-0.009 under band inference, and the amplitude ratio is 7.4-24.4.
- The raw model (inference off) is down to an amplitude ratio of 0.15 and `E'_50/E'_0` of
  0.0004-0.005.
- Under band inference `E'` sits exactly on the lower edge every tick (0.99975^48 = 0.98807)
  and the projection acts on 100 % of ticks.
- Training did not diverge: the loss curves match the off and upper arms
  (`results/band/*-band.train.log`).
- The constraint was active on 88-100 % of training proposals at the first log row and on
  100 % from step 100 on. In the upper arm, activity wandered between 0 % and 100 % from one
  log row to the next.

Cause: `train.py` (and the Rust `fit`) measured the band's reference energy `E'_prev` after
the input-noise injection (`--noise 0.04`). That noise carries 0.6 % of `E'` at p50 and up to
6 % at the tail, about 20 times the band's width (6e-4). Measured on 256 training windows,
95 % of the true next states fall below `0.99975 · E'(noisy input)`, against 0 % below
`0.99975 · E'(clean input)`.

So the lower edge lifted almost every target-matching prediction. The model learned to emit an
energy-starved field and leave the amplitude to the lift: the uniform up-scaling makes the loss
blind to the proposal's amplitude whenever it binds. At inference, the lift multiplies whatever
the model does not damp by `s > 1` every tick, and that grows into the spike.

It is not a sign or ratio bug in the band itself:
- the off-trained checkpoints under band inference match upper inference (table above);
- the projection tests pass, gradient finiteness included;
- with the fix below, the same band in training no longer collapses (B).

Fix (`1dc2138`): measure `E'_prev` on the state before noise is added, in both `train.py` and
`fit.rs`. The upper arm was affected the same way, but mildly: a noisy reference only loosens
an upper bound.

Local reproduction on the M2 (CPU, 1000 steps, 128/4, seed 20, 300-trajectory dump
`--nu 0.001 --seed 11`, otherwise the sweep recipe):

| run | inference | corr@50 | active RMSE@50 | amp@50 | global RMSE@50 | `E'_50/E'_0` raw / ON | active ticks |
|---|---|---:|---:|---:|---:|---:|---:|
| A: band, reference after noise (old) | off | 0.0692 | 0.9960 | 0.199 | 0.6223 | | |
| A | band | 0.7393 | 0.3987 | 2.142 | 0.4356 | 0.0031 / 0.9881 | 100 % |
| B: band, reference before noise (fixed) | off | 0.9468 | 0.4047 | 0.662 | 0.2700 | | |
| B | band | **0.9624** | **0.2081** | **0.968** | **0.1764** | 0.435 / 0.9881 | 100 % |
| C: constraint off in training | off | 0.9573 | 0.2720 | 1.028 | 0.2316 | | |
| C | upper | 0.9554 | 0.2582 | 0.942 | 0.2193 | 1.214 / 1.0130 | 94.9 % |
| C | band | 0.9556 | 0.2580 | 0.944 | 0.2192 | 1.214 / 1.0163 | 98.8 % |

A reproduces the collapse already at 1000 steps. With the fix (B), the band-trained model is
the best of the three on active RMSE@50 (0.208 vs 0.258-0.272), global RMSE and corr@50. This
is one seed and one short budget.

B still leans on the lift: its raw rollout keeps only 43 % of the energy by tick 50, and the
projection acts on every tick. The band does its job at inference, but the model is not
learning energy conservation on its own.

### 7b. Verdict (off and upper only, since the band arms of this sweep are invalid)

**Does the constraint make the small model as good as the large one?** The small model was
already as good: 128/4 off matches 256/6 off (corr@50 0.982 vs 0.982, active RMSE 0.131 vs
0.147). The upper bound in training makes the small model better than both:
- corr@50 0.9913 ± 0.0011, against 0.9817 ± 0.0074 trained without it;
- active RMSE@50 0.094 ± 0.008 with upper at inference, against 0.131 ± 0.025 off/off (−28 %);
- global RMSE 0.072 against 0.108;
- seed-to-seed sd shrinks 3-7x.

**Does width help once the constraint is on?** No. 256/6 trained with the upper bound scores
corr@50 0.9859 ± 0.0070 and active RMSE 0.122 ± 0.037, worse than 128/4 with the same
constraint, and its seed spread stays large. As in rung6, width does not help at this budget;
the constraint does, for the small model.

Applying the constraint at inference only (off-trained checkpoints) trims active RMSE by about
10 % and pulls amp@50 to 0.97-1.00, but leaves corr unchanged. Most of the gain comes from
training with it.

Caveats:
- 3 seeds, 24 trajectories per eval.
- The upper arm's reference energy was the noisy one (a slightly looser bound than intended).
- Activity in upper/upper varies a lot between seeds (1-96 %): seed 22 at 128/4 hardly ever
  touched the bound, and its model loses energy (`E'_50/E'_0` 0.94).

### 7c. Next run (prepared, not launched)

Rerun the upper and band arms with the fixed reference: 12 trainings plus 24 evals, about
40-50 min at this sweep's measured pace. The script is resumable, so deleting those outputs
reruns just them:

```
rsync -az --delete --exclude target/ --exclude target-wgpu/ --exclude results/ --exclude data/ \
  --exclude web/pkg/ --exclude web/node_modules/ ./ <box>:<box-dir>/
ssh <box> 'cd <box-dir> && rm -rf results/band/*-upper* results/band/*-band* results/band/summary.md'
ssh <box> 'cd <box-dir> && systemd-run --user --unit=jacobi-band2 --same-dir --collect --property=Type=simple \
  --setenv=DATASET=<dump> --setenv=PYTHON=<venv>/bin/python --setenv=JACOBI=./target-wgpu/release/jacobi \
  /usr/bin/flock <box-lock> /bin/bash tools/energy-band-sweep.sh'
```

The Rust binary needs no rebuild for this. `fit.rs` changed, but the sweep trains in PyTorch,
and the eval and import paths are unchanged.

## 8. Rerun with the fixed reference energy (2026-10-06, unit jacobi-band2, 47 min)

Upper and band arms retrained with E'_prev measured before noise (commit 1dc2138); off arms kept from the first run. 50 ticks, mean ± sd over seeds 20/21/22, 24 trajectories.

| size | train | inference | corr | active RMSE | amp | global RMSE |
|---|---|---|---:|---:|---:|---:|
| 128/4 | off | off | 0.9817 ± 0.0074 | 0.131 ± 0.025 | 1.043 ± 0.033 | 0.108 ± 0.018 |
| 128/4 | off | upper | 0.9810 ± 0.0076 | 0.120 ± 0.020 | 0.996 ± 0.014 | 0.101 ± 0.016 |
| 128/4 | upper (fixed) | upper | 0.9657 ± 0.0349 | 0.151 ± 0.054 | 0.985 ± 0.032 | 0.131 ± 0.056 |
| 128/4 | band (fixed) | band | 0.9614 ± 0.0242 | 0.167 ± 0.051 | 0.952 ± 0.027 | 0.143 ± 0.056 |
| 256/6 | off | off | 0.9822 ± 0.0151 | 0.147 ± 0.071 | 1.029 ± 0.027 | 0.121 ± 0.067 |
| 256/6 | off | upper | 0.9813 ± 0.0163 | 0.131 ± 0.072 | 0.974 ± 0.033 | 0.110 ± 0.067 |
| 256/6 | upper (fixed) | upper | 0.9702 ± 0.0198 | 0.143 ± 0.043 | 0.995 ± 0.017 | 0.120 ± 0.038 |
| 256/6 | band (fixed) | band | 0.9732 ± 0.0204 | 0.138 ± 0.037 | 0.966 ± 0.024 | 0.113 ± 0.039 |

Revised verdict:
- The band no longer collapses (corr 0.96-0.97 instead of 0.007), so the fix works.
- With the tight (fixed) reference, training with either constraint is worse and noisier across seeds than training without it. The section 7 result (upper-trained 128/4 at corr 0.9913 ± 0.0011, active RMSE 0.094) came from the noisy reference, i.e. a bound loosened by about 20x the band width. The slack of the bound during training matters; the tight bound hurts.
- The robust gain is the constraint at inference on unconstrained checkpoints: active RMSE -8 to -11 %, amplitude to 0.97-1.00, correlation unchanged, at both sizes.
- Width still does not help.
- Next: sweep the training slack (r_max from tight to the effective looseness of the first run) with more seeds, keeping inference-time projection.
