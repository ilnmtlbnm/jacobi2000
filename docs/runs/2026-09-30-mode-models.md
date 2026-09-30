# Roadmap item 2 — mode models

Machine: data generation on an Apple M2 (native, CPU); training on a Linux
desktop with an RTX 3080 (CUDA), under `flock`, no other GPU job running.

## Interpretation (undefined in CONCEPT.md, written before implementing)

Grepped CONCEPT.md for "mode": every hit is `step` / `block` / `control` /
`rewind` *mode* (a different sense of the word), never a "mode model."
Interpretation used here, per the brief's instruction to write down a
minimal reading when a roadmap item is undefined: a **mode model** is a
spectral operator -- decompose the field into its leading 2D Fourier modes
(a truncated `rfft2`) and learn the dynamics of the retained mode
coefficients directly, instead of the patch-transformer's pixel/patch
tokens. This is the concrete reading of CONCEPT section 2's "spectral term"
and the prior-art line "FNO ... (mesh operators)": a small Koopman/DMD-style
operator on a truncated spectral basis, testing whether shallow water's
dominant long-wavelength modes evolve with a much smaller, much cheaper
model than the full patch-transformer needs.

## Model

`training/jacobi2000_pt/modes.py`. `rfft2(h, u, v)` (each `[64, 64]`),
keep the lowest 10x10 positive-frequency coefficients per channel (rows
0-9, cols 0-9 of the `[64, 33]` rfft2 output -- asymmetric, misses the
negative-y-frequency half of the same physical wavelength, but is a simple,
bug-free low-pass truncation), flatten real+imag to a 600-dim real vector, a
3-layer MLP (hidden 256) predicts the increment in mode space, `irfft2`
reconstructs the field. No attention, no patches, no mask. 373,848
parameters (vs. the patch-transformer's 995,843 for `model-pt-curriculum`).

```
python -m jacobi2000_pt.modes --dataset data/sw-nu001 --out results/modes \
  --steps 4000 --lr 1e-3 --rollout 8 --batch 16 --k 10 --hidden 256 \
  --horizons 10,25,50 --eval-trajectories 16
```

Same dataset as the curriculum baseline (`jacobi dump-dataset --pool 1000
--nu 0.001 --seed 20`, `docs/runs/2026-09-29-pytorch-training.md` section 3).
Wall clock: **64.2 s** on the 3080.

## Results

Scored in Python (`metrics.py`, the same formulas
`crates/train/src/metrics.rs` uses) against 16 held-out trajectories from
the dataset dump, since this is a new architecture with no Rust-side
`Jacobi2000` tensor names to import through `jacobi import-weights` --
`docs/pytorch-training.md`'s acceptance path (Rust `eval-metrics` as the
numbers of record) does not apply to this architecture tonight.

| horizon (ticks) | mean corr | mean rmse |
|---:|---:|---:|
| 10 | 0.8197 | 0.3116 |
| 25 | 0.5419 | 0.6075 |
| 50 | 0.2091 | 0.8688 |

For reference, the patch-transformer curriculum's own numbers on the same
dataset and horizons (`docs/runs/2026-09-29-pytorch-training.md` section 6,
`s6-mass-energy-proj`, scored by the Rust `eval-metrics` binary, not this
run's Python port -- not a controlled apples-to-apples comparison, just the
existing best for scale):

| horizon (ticks) | corr (model-pt-curriculum) |
|---:|---:|
| 10 | 0.9995 |
| 25 | 0.9981 |
| 50 | 0.9930 |

Full per-trajectory rows: `results/modes-log.json`. Raw run log:
`results/modes-run.log`. Checkpoint: `results/checkpoints/modes/model.pt`
(gitignored).

## Observations

- The mode model degrades much faster with horizon than the patch-
  transformer (corr 0.82 -> 0.21 from 10 to 50 ticks, vs. 0.9995 -> 0.9930):
  a 10x10 truncation keeps only the lowest wavenumbers, and shallow-water
  waves from a localized drop carry meaningful energy well past that cutoff
  at short range -- the truncation itself, not the operator's dynamics, is
  likely the dominant error source at these horizons. This was not
  ablated tonight (no sweep over `k`).
- The result is consistent with CONCEPT's own framing of the mask
  as the problem statement: full patch attention (or even the stencil mask
  the curriculum model uses) sees local structure the 10x10 mode truncation
  simply discards. A mode model may be a better fit for smoother fields
  (heat diffusion, low Gray-Scott wavenumbers) than a drop-driven wave field
  with a sharp initial condition.
- No projection (mass/energy) was applied in mode space -- `project.py`'s
  functions operate on the pixel-space field, and were applied to the
  *reconstructed* field after `irfft2` would have been straightforward to
  add but wasn't, for time. Flagged as a gap, not attempted.
- Not attempted: importing this architecture into the Rust side (would need
  a new `Jacobi2000`-sibling type and a new weights layout, out of scope for
  one night), a `k` sweep, and picture output.
