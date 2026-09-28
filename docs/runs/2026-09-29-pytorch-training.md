# Run 2026-09-29 — training moved to PyTorch, inference stays in Rust

Task: move `jacobi2000`'s training off Burn/wgpu onto PyTorch (`training/`),
keeping the Rust model, projections, masks, solver and `web/` unchanged.
Plan and format in `docs/pytorch-training.md`.

Machines: data generation and all Rust steps on an Apple M2 (native, ndarray
backend, no GPU needed for CPU inference/eval); PyTorch training on my
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

- A reverse-trained (rewind) PyTorch checkpoint and the `jacobi round-trip`
  check (`docs/runs/2026-09-20-rewind-round-trip.md`'s counterpart) — not
  attempted this run; the forward-only acceptance path above is what the
  task's acceptance test names.
- A same-recipe, same-hardware (3080, from scratch) Burn-vs-PyTorch
  wall-clock comparison, to isolate the framework's own speed from the other
  two variables changed at once (see Observations).
- Re-running `tools/export_web_models.sh` and the headless-Chromium gallery
  smoke test (`web/test/gallery.mjs`) against the PyTorch-trained,
  Rust-imported checkpoint, to close the loop all the way to the browser.
- Matching model20's actual accuracy would need replaying its full 18-stage
  warm-start lineage (or an equivalent from-scratch step budget large enough
  to substitute for it) in PyTorch, not a single 4000-step run.
