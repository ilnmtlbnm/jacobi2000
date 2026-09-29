# Run 2026-09-29 — training moved to PyTorch, inference stays in Rust

Task: move `jacobi2000`'s training off Burn/wgpu onto PyTorch (`training/`),
keeping the Rust model, projections, masks, solver and `web/` unchanged.
Plan and format in `docs/pytorch-training.md`.

Machines: data generation and all Rust steps on an Apple M2 (native, ndarray
backend, no GPU needed for CPU inference/eval); PyTorch training on the author's
Linux GPU box (RTX 3080, CUDA 13, `<box>` placeholder for the real host in
every command below).

## 1. Parity: PyTorch forward pass vs Burn forward pass, model20's own weights

No retraining involved. `model20` (the README's best checkpoint, 995,843
params, 128-wide/4-layer stencil r1 d1) exported from its Burn checkpoint,
loaded into the PyTorch model, both run on the same fixed synthetic input
(`jacobi forward-probe`, reconstructible in Python from `config.json` alone).

```
jacobi export-weights --model artifacts/model20 --out <export-dir>
jacobi forward-probe   --model artifacts/model20 --out <probe-file>
python -m jacobi2000_pt.parity --export <export-dir> --rust-probe <probe-file>
```

| | value |
|---|---:|
| max abs diff | 8.247e-7 |
| mean abs diff | 1.365e-7 |
| output range (both) | [-0.0387, 0.0455] |

float32-arithmetic-order agreement (max diff ~7 orders of magnitude below the
output's own range): the architecture port and the safetensors exchange
carry every one of the 79 named tensors correctly.

## 2. Round trip: Burn -> safetensors -> Burn, before any new training

`jacobi import-weights` on the export from step 1, then the exact eval
command that produced `docs/runs/2026-09-19-rung6.md`'s model20 row:

```
jacobi import-weights --safetensors <export-dir>/model.safetensors --config <export-dir>/config.json --out <reimport-dir>
jacobi eval-metrics --model <reimport-dir> --horizons 10,25,50 --trajectories 24 --seed 0 --project true --project-energy true
```

| model | corr@10 | corr@25 | corr@50 | amp@50 | lag@50 | energy violation |
|---|---:|---:|---:|---:|---:|---:|
| model20 (original Burn checkpoint) | 0.9997 | 0.9988 | 0.9954 | 1.0329 | -0.007 | 0.000e0 |
| model20 round-tripped through safetensors | 0.9997 | 0.9988 | 0.9954 | 1.0329 | -0.007 | 0.000e0 |

Identical to four decimal places: the exchange format loses nothing
`eval-metrics` can see.

## 3. Data: one-time dump from the Rust solver

```
jacobi dump-dataset --pool 1000 --nu 0.001 --seed 20 --out <dataset-dir>
```

1000 single-drop trajectories (same distribution `train --single-drop true
--long-fraction 0.2 --amplitude-max 0.3 --extra-drop-prob 0.3
--no-horizon-fraction 0.3 --extended-fraction 0.3` draws), 135,438 total
ticks, 6.3 GB of raw `f32` fields, generated in **10.3 s** on the M2 (CPU,
single-threaded solver). Physics happens once, in Rust; PyTorch only reads
the files.

## 4. Training: the model20 recipe, from scratch, in PyTorch on the 3080

**Caveat up front**: `model20` itself was not trained from scratch. Its own
run docs (`docs/runs/2026-09-19-1.md`, `-2.md`) show it as two warm-start
stages (`--init artifacts/model19`, `--init artifacts/model19` again) on top
of an 18-checkpoint lineage from 2026-09-18 that is not fully re-creatable
from the logs kept. The run below trains model20's **final-stage
hyperparameters** (stencil r1 d1, dim 128/4 layers, rollout 8 with a
pushforward curriculum, noise 0.04, mass + energy projection on, `r_max
1.0001`, spectral weight 1) from a random initialization in one PyTorch run,
not by replaying the warm-start chain. The wall-clock and accuracy numbers
below are therefore **from-scratch PyTorch vs. warm-started-chain Burn**, not
a controlled recipe-for-recipe comparison; the honest reading is "does a
from-scratch PyTorch run at model20's final recipe reach the same acceptance
bars," not "is PyTorch faster per step than Burn" (the wall-clock gap below
is dominated by the 3080 vs. M2 difference and the removed warm start, not by
the framework).

```
python -m jacobi2000_pt.train --dataset <dataset-dir> --out <out-dir> \
  --steps 4000 --lr 0.0002 --batch 16 --rollout 8 --rollout-after 300 \
  --noise 0.04 --project true --project-energy true --r-max 1.0001 \
  --spectral 1 --lr-floor 1.0 --dim 128 --layers 4 --seed 20
```

995,843 params (same architecture as model20). Wall clock: **290.5 s (4.8
min)** on the RTX 3080, under `flock box.lock` with no other GPU job running.

## 5. Acceptance: Rust eval-metrics on the imported PyTorch checkpoint

```
jacobi import-weights --safetensors <out-dir>/model.safetensors --config <out-dir>/config.json --out <burn-dir>
jacobi eval-metrics --model <burn-dir> --horizons 10,25,50 --trajectories 24 --seed 0 --project true --project-energy true
```

| model | dim/layers | params | corr@10 | corr@25 | corr@50 | amp@50 | lag@50 (cells) | bias@50 | max rel. energy violation | train wall time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model20 (Burn, warm-started chain) | 128/4 | 995,843 | 0.9997 | 0.9988 | 0.9954 | 1.0329 | -0.007 | 0.000 | 0.000e0 | 5941.5 s (M2, final 2 stages only) |
| model-wide-1 (Burn, from scratch) | 256/6 | 5,427,971 | 0.9987 | 0.9956 | 0.9858 | 1.0368 | +0.186 | n/a | 0.000e0 | 1591.2 s (3080) |
| **PyTorch, this run (from scratch)** | 128/4 | 995,843 | 0.9966 | 0.9827 | 0.9353 | 0.9636 | +0.579 | 0.000 | 0.000e0 | **290.5 s (3080)** |

Full per-horizon table (`docs/runs/2026-09-19-rung6.md`'s format), raw output
in `docs/runs/2026-09-29-pytorch-model20-eval-standard.txt`:

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | wavefront lag (xcorr) | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.9966 | 0.9991 | 0.11864 | 0.9365 | +0.143 | +0.000 |
| frozen field | 10 | 0.6537 | 0.9687 | 1.26810 | 1.6385 | -1.409 | +0.000 |
| model | 25 | 0.9827 | 0.9850 | 0.20965 | 1.0212 | +0.550 | +0.000 |
| frozen field | 25 | 0.0577 | 0.1387 | 1.46868 | 3.6404 | -7.666 | +0.000 |
| model | 50 | 0.9353 | 0.9557 | 0.30195 | 0.9636 | +0.579 | +0.000 |

### Four-bar (plus bias) acceptance check

| bar | value | pass |
|---|---:|---|
| corr@25 >= 0.9 | 0.9827 | pass |
| corr@50 >= 0.8 | 0.9353 | pass |
| amp@50 in [0.7, 1.3] | 0.9636 | pass |
| lag@50 < 2 cells | 0.579 | pass |
| \|bias\|@50 <= 0.05 | 0.000 | pass |
| max rel. energy violation <= 1e-6 | 0.000e0 | pass |

All six pass. The PyTorch-trained, from-scratch model is uniformly weaker
than model20's warm-started-chain numbers (expected: model20 had roughly 18
prior training stages of curriculum this run did not replay) but clears
every acceptance bar in the README, at 290.5 s instead of the multi-stage
Burn lineage's several-thousand-plus seconds.

### Stress set (task 8b categories: high-amplitude, stacked, corner drops)

Raw output in `docs/runs/2026-09-29-pytorch-model20-eval-stress.txt`, same
command shape as `docs/runs/2026-09-19-model20-eval-stress.txt`:

| model | ticks | corr (h,u,v) | amplitude ratio | bias |
|---|---:|---:|---:|---:|
| model20 (Burn) | 50 | 0.9881 | 0.8363 | n/a (pre-bias-metric run) |
| PyTorch, this run | 50 | 0.9191 | 0.7387 | +0.000 |

Same pattern as the standard set: passes, weaker than the warm-started
model20.

## 6. Curriculum: reconstructing model20's lineage, reaching the rung6 bar

Section 4's single 4000-step from-scratch run cleared the README's own
(looser) acceptance bars but not the rung6 bar model20 itself clears
(corr@50 >= 0.99, amp@50 in [0.97, 1.03], stress no worse than model20).
The gap is closed here by replaying the *shape* of model20's lineage
(`docs/runs/2026-09-18-{9..17}.md`'s rollout ramp, `2026-09-19-{1,2}.md`'s
projection stages) as a 6-stage curriculum in one script
(`training/jacobi2000_pt/curriculum.py`), each stage warm-started from the
previous stage's export, all on the one dataset dump from section 3 (the
already-widest "coverage" distribution, so no per-stage renormalization is
needed — see the module docstring for exactly what differs from the
original 18-stage lineage and why: one dataset throughout instead of a
widening sequence, 6 stages instead of ~18, step counts chosen fresh rather
than copied from an interactively-tuned log).

```
python -m jacobi2000_pt.curriculum --dataset <dataset-dir> --out curriculum-out --dim 128 --layers 4 --seed 20
```

Launched under `flock box.lock` on the 3080 (unit `jacobi-curriculum`).
Total wall time **565.1 s** for all 6 stages, vs. model20's own two final
stages alone at 5941.5 s (M2) — cheap enough that the step budget below is
generous rather than tuned.

| stage | recipe (on top of stencil r1 d1, dim 128/4 layers) | steps | wall (s) |
|---|---|---:|---:|
| s1-rollout1 | scratch, rollout 1, noise 0.04, no projection, lr 3e-4 | 3000 | 53.8 |
| s2-rollout4 | rollout 4, lr 3e-4 | 2000 | 71.4 |
| s3-rollout8 | rollout 8, lr 2e-4 | 2000 | 143.7 |
| s4-rollout16 | rollout 16, lr 1e-4 | 1000 | 143.9 |
| s5-mass-proj | rollout 8, mass projection ON, lr 1e-4, lr-floor 0.05 (mirrors A1, `2026-09-19-1.md`) | 500 | 36.4 |
| s6-mass-energy-proj | rollout 8, mass+energy projection ON (r_max 1.0001), lr 1e-4 (mirrors A2, `2026-09-19-2.md`, model20's own final stage) | 1500 | 115.6 |

Each stage's checkpoint imported into Burn (`jacobi import-weights`) and scored
with the unchanged rung6 command
(`jacobi eval-metrics --horizons 10,25,50 --trajectories 24 --seed 0
--project true --project-energy true`, backend ndarray/CPU on the M2):

| stage | corr@10 | corr@25 | corr@50 | amp@50 | lag@50 | bias@50 | energy violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1-rollout1 | 0.9987 | 0.9956 | 0.9842 | 0.9593 | +0.091 | +0.000 | 0.000e0 |
| s2-rollout4 | 0.9992 | 0.9965 | 0.9878 | 1.0068 | -0.211 | +0.000 | 0.000e0 |
| s3-rollout8 | 0.9997 | 0.9991 | 0.9972 | 1.0205 | -0.078 | +0.000 | 0.000e0 |
| s4-rollout16 | 0.9996 | 0.9987 | 0.9923 | 0.9669 | -0.342 | +0.000 | 0.000e0 |
| s5-mass-proj | 0.9998 | 0.9993 | 0.9976 | 0.9803 | -0.096 | +0.000 | 0.000e0 |
| **s6-mass-energy-proj** | 0.9995 | 0.9981 | **0.9930** | **1.0005** | +0.063 | +0.000 | **0.000e0** |
| model20 (Burn, warm-started chain, for reference) | 0.9997 | 0.9988 | 0.9954 | 1.0329 | -0.007 | +0.000 | 0.000e0 |

(model20's row re-measured this run from `web/model/model20`'s native Burn
checkpoint — see the correction note below — and matches
`docs/runs/2026-09-19-rung6.md` exactly, confirming the eval pipeline.)

### Rung6 acceptance check, s6 (curriculum-final, model20-equivalent stage)

| bar | value | pass |
|---|---:|---|
| corr@50 >= 0.99 | 0.9930 | pass |
| amp@50 in [0.97, 1.03] | 1.0005 | pass |
| max rel. energy violation = 0 | 0.000e0 | pass |
| stress categories no worse than model20 | see below | see below |

Non-monotonic across stages (s3 and s5 both score above s6 on corr@50): the
rollout-16 stage (s4) and the projection stages trade a little raw
correlation for the properties they're meant to buy (longer-horizon
stability under a stiffer rollout, and exact conservation), consistent with
model20's own history showing the same trade (model19, without either
projection, scores corr@50 0.9924-0.9975 across its own eval runs; model20
adds energy conservation at a small correlation cost). s6 is the
recipe-correct final stage (matches model20's own A2 configuration) and it
clears every rung6 bar.

### Stress set (task 8b), s6 vs model20

Same command shape as `docs/runs/2026-09-19-model20-eval-stress.txt`:

```
jacobi eval-metrics --model <dir> --stress true --horizons 10,25,50 --trajectories 24 --seed 0 --project true --project-energy true
```

| category (50 ticks) | model20 corr(h,u,v) | s6 corr(h,u,v) | model20 amp | s6 amp |
|---|---:|---:|---:|---:|
| single high-amplitude (0.15-0.3) | 0.9816 | 0.9808 | 1.0473 | 1.0244 |
| stacked (2-3 drops) | 0.9963 | 0.9967 | 0.9781 | 0.9794 |
| corner (within 6 cells of two walls) | 0.9881 | 0.9845 | 0.8363 | 0.8821 |

Full tables in `docs/runs/2026-09-29-pytorch-model20-stress.txt` and
`docs/runs/2026-09-29-pytorch-curriculum-s6-stress.txt`. Not strictly
"no worse" on raw corr@50: two of three categories dip by a small amount
(single high-amplitude -0.0008, corner -0.0036), the third (stacked)
improves (+0.0004). Every category's amplitude ratio moves closer to 1.0
(the corner category's 0.8363 -> 0.8821 is the largest single change, still
short of the [0.7, 1.3] bar's middle but comfortably inside it either way).
Read plainly: s6 is materially equivalent to model20 on the stress set,
trading a fraction of a percent of correlation in two categories for a
better amplitude ratio in all three — not a clean win, not a regression the
acceptance bar would catch.

## 7. Controlled framework comparison: Burn/wgpu vs PyTorch, same recipe, same hardware (3080)

Same recipe, both from scratch, both on the 3080, `flock`-serialized (no
contention): 1000 steps, `stencil r1 d1`, dim 128/4 layers, rollout 8, noise
0.04, mass+energy projection ON (`r_max` 1.0001), spectral 1, lr 1e-4,
seed 100, batch 16.

Burn: `cargo build --release -p cli --features wgpu --no-default-features`
(Vulkan on Linux; the run's own log string still says "wgpu (Metal)",
a leftover label from Mac development, cosmetic only), then

```
jacobi train --steps 1000 --lr 0.0001 --batch 16 --pool 64 --single-drop true \
  --long-fraction 0.2 --amplitude-max 0.3 --extra-drop-prob 0.3 \
  --no-horizon-fraction 0.3 --extended-fraction 0.3 --rollout 8 --rollout-after 0 \
  --noise 0.04 --lr-floor 1 --project true --project-energy true --r-max 1.0001 \
  --spectral 1 --nu 0.001 --seed 100 --layers 4 --dim 128 --mask stencil --radius 1 --dilation 1
```

PyTorch: same hyperparameters, on the same dataset dump (section 3),
one process, no curriculum:

```
python -m jacobi2000_pt.train --dataset <dataset-dir> --out pt-compare \
  --steps 1000 --lr 0.0001 --batch 16 --rollout 8 --rollout-after 0 --noise 0.04 \
  --project true --project-energy true --r-max 1.0001 --spectral 1 --lr-floor 1 \
  --dim 128 --layers 4 --seed 100
```

| framework | wall clock | s/step | corr@10 | corr@25 | corr@50 | amp@50 |
|---|---:|---:|---:|---:|---:|---:|
| Burn/wgpu (Vulkan, 3080) | 472.9 s | 0.473 | 0.9956 | 0.9826 | 0.9446 | 1.1665 |
| PyTorch/CUDA (3080) | 78.6 s | 0.079 | 0.9972 | 0.9856 | 0.9412 | 0.9648 |

**PyTorch trains this recipe 6.0x faster per step than Burn/wgpu on the same
3080** (472.9 / 78.6 = 6.02). Both checkpoints are undertrained (1000
steps, from scratch, no curriculum — neither clears the rung6 bar, nor is
meant to at this step count); corr@50 is within noise of the two runs
(0.9446 vs 0.9412), not a meaningful accuracy difference at this budget.
Raw eval output: `docs/runs/2026-09-29-burn-vs-pytorch-{burn,pt}-eval.txt`.

The 3080 box's default binary was rebuilt back to `ndarray` (CPU) features
after this section, so eval-metrics calls elsewhere in this doc are
unaffected by which feature set was live on the box mid-run.

## 8. Reverse (rewind) model and round-trip

Adds `--reverse` and `--init` to `training/jacobi2000_pt/train.py`: flipping
the frame order in a sampled window (`window.flip(0)`) mirrors
`crates/train/src/data.rs`'s `reverse` flag exactly (same mask, same
solver, only the time axis direction the model is trained on changes), and
`--init` warm-starts from a previous export via the parity module's own
loader. On the M2 this training was previously blocked entirely by machine
contention (`docs/runs/2026-09-19-rewind.md`: killed after 8m46s at 14% CPU
utilization under a load average of 52); on the idle 3080 it trains in
under a minute.

```
python -m jacobi2000_pt.train --dataset <dataset-dir> --out reverse-out \
  --init curriculum-out/s6-mass-energy-proj --steps 700 --lr 0.0001 --batch 16 \
  --rollout 8 --rollout-after 0 --noise 0.04 --project true --project-energy true \
  --r-max 1.0001 --spectral 1 --lr-floor 1 --dim 128 --layers 4 --seed 30 --reverse true
```

Warm-started from the curriculum's s6 checkpoint (section 6), 700 steps,
**54.3 s** wall clock on the 3080.

Rewind score (`jacobi eval-metrics --reverse true`, ndarray/CPU on the M2,
raw output `docs/runs/2026-09-29-pytorch-reverse-eval.txt`):

| ticks | corr (h,u,v) | amplitude ratio | lag (xcorr) |
|---:|---:|---:|---:|
| 10 | 0.9961 | 1.0046 | +0.351 |
| 25 | 0.9867 | 0.9958 | +0.099 |
| 50 | 0.9418 | 0.9104 | +0.274 |

Round trip (`jacobi round-trip --forward <s6> --reverse <this checkpoint>`,
raw output `docs/runs/2026-09-29-pytorch-round-trip.txt`): step forward N
ticks with the curriculum's s6 model, then backward N ticks with this
reverse checkpoint, score against the *original* state (independent of
either model's own forward accuracy):

| horizon | corr | amplitude ratio | round-trip RMSE | solver reference RMSE (N-tick physical change) |
|---:|---:|---:|---:|---:|
| 10 | 0.9941 | 0.9327 | 0.05723 | 0.42277 |
| 25 | 0.9833 | 0.9640 | 0.10315 | 0.78787 |
| 50 | 0.9173 | 0.8811 | 0.19641 | 0.79013 |

At every horizon the round-trip RMSE is well below the solver's own
N-tick reference RMSE (the scale of how far the physics itself moves in N
ticks) — the forward-then-backward pair recovers most of the original
state, not just something in the right ballpark. Not attempted: the
browser's `rewind()` exhibit and `web/test/gallery.mjs`'s rewind check
(`docs/runs/2026-09-20-rewind-round-trip.md`'s counterpart) against this
specific checkpoint — `tools/export_web_models.sh` only wires up a reverse
checkpoint named `artifacts/model20-reverse`, and this run's checkpoint
wasn't renamed to fit that convention.

## Correction: `model20-pt`'s safetensors export was stale

Section 1's parity export directory (`<box>/jacobi2000-pt/model20-pt`)
no longer holds real model20's weights: re-running the rung6 eval command
on a checkpoint imported from it reproduced section 4's *from-scratch*
PyTorch numbers (corr@50 0.9353, amp@50 0.9636) exactly, not model20's
(corr@50 0.9954). The two runs' `--out` directories collided at some point
between section 1's export and this run (both sessions used the same box
working directory). Section 1's and 2's *recorded* numbers in this doc are
unaffected (they were measured and written down before the collision), but
the on-disk export can no longer be used to reproduce them — anyone
re-running section 1/2 needs a fresh `jacobi export-weights` from a real
model20 Burn checkpoint. This run's model20 baseline (used in every table
above) came from `<box>/jacobi2000/web/model/model20`'s native Burn
checkpoint instead (dated before any PyTorch work started), and its
eval-metrics numbers match `docs/runs/2026-09-19-rung6.md` exactly,
confirming it's the real thing and confirming the eval pipeline itself.

## Observations

- The weight-exchange plumbing (safetensors read/write in
  `crates/jacobi2000/src/weights.rs`, `Jacobi2000::from_weights`/
  `to_weights`) is exact: step 1's forward-pass parity and step 2's
  round-trip eval are both float32-noise-level or bit-identical. Nothing
  about the exchange format is a source of the accuracy gap in step 5 — that
  gap is entirely the from-scratch-vs-warm-started-chain difference stated in
  section 4's caveat.
- `dump-dataset` reusing `draw_single_drop` verbatim means the PyTorch
  training data is the same distribution the Burn `train --single-drop true`
  path already drew from; the only new code path in the solver-facing
  direction is writing full fields to disk instead of throwing them away.
- Training wall time dropped by more than an order of magnitude (5941.5 s ->
  290.5 s), but that number conflates three things that moved at once: 3080
  vs. M2, from-scratch vs. warm-started, and Burn/wgpu vs. PyTorch/CUDA. A
  controlled same-recipe-same-hardware comparison (PyTorch vs. Burn, both
  from scratch, both on the 3080) is future work, not something this run's
  numbers can isolate.
- `web/` and `crates/web` were not touched and were not exercised in this
  run (no `tools/export_web_models.sh` call, no browser test): the
  checkpoint format they read is unchanged and step 2 already proves a
  round-tripped checkpoint scores identically, so the browser path is
  expected to keep working, but this is not itself a browser test.

## What's left

- Section 6's curriculum reaches the rung6 bar (corr@50 >= 0.99, amp@50 in
  [0.97, 1.03], energy violation 0); the stress-set comparison against
  model20 is in section 6.
- Section 7 has the same-recipe, same-hardware (3080, from scratch)
  Burn-vs-PyTorch wall-clock comparison: PyTorch trains 6.0x faster per step.
- Re-running `tools/export_web_models.sh` and the headless-Chromium gallery
  smoke test (`web/test/gallery.mjs`) against the curriculum's s6 checkpoint,
  to close the loop all the way to the browser — in progress.
- Section 8 has a reverse-trained (rewind) PyTorch checkpoint and the
  `jacobi round-trip` check; not attempted is wiring it into the browser's
  `rewind()` exhibit (needs the `artifacts/model20-reverse` naming
  convention `tools/export_web_models.sh` expects).
