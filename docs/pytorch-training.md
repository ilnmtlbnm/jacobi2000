# Moving training to PyTorch, keeping inference in Rust

## What moves

- **Model definition**: a PyTorch mirror of `crates/jacobi2000/src/model.rs`
  (`training/jacobi2000_pt/model.py`), same layer graph, same tensor shapes,
  named to match burn's own field paths so weights move without a name
  translation table on the training side.
- **Training loop**: `training/jacobi2000_pt/train.py`, the model20 recipe
  (`docs/runs/2026-09-19-2.md`): AdamW, warmup + cosine decay, pushforward
  rollout curriculum, noise injection, mass + energy projection in the loop,
  spectral loss term.
- **Eval metrics used during training** (correlation, amplitude ratio,
  wavefront lag, bias, energy violation): ported from
  `crates/train/src/metrics.rs` to `training/jacobi2000_pt/metrics.py`, same
  formulas, for fast iteration without leaving Python. The acceptance numbers
  in this doc's run log are still the Rust `eval-metrics` binary's own
  output, not this port's.
- **Data generation stays behind the Rust solver.** `crates/solvers` is the
  one place the shallow-water equations are written down, and it stays that
  way. Training data is produced once by a new CLI subcommand,
  `jacobi dump-dataset`, which reuses `train::data`'s existing single-drop
  generator (the exact distribution `train --single-drop true` already
  trains on) and writes full raw trajectories to disk instead of throwing the
  fields away and keeping only the `Record` metadata `gen-data` wrote before.
  Python never touches the physics; it reads flat `f32` files.

## What stays in Rust

- The solver (`crates/solvers`), the model's inference path
  (`Jacobi2000::forward`/`step`, unchanged), the constraint projections
  (`crates/jacobi2000/src/project.rs`, unchanged), the masks
  (`crates/jacobi2000/src/mask.rs`, unchanged).
- The acceptance eval (`jacobi eval-metrics`) and the round-trip check
  (`jacobi round-trip`): both run unchanged, on a checkpoint imported from
  the PyTorch export.
- `web/` and `crates/web`: unchanged, since the checkpoint format they read
  (`train::checkpoint::{save, load}`, a Burn binary record plus
  `meta.json`) is untouched. A PyTorch-trained model reaches the browser the
  same way a Burn-trained one always did: through `tools/export_web_models.sh`,
  once its safetensors export has been imported into that format.
- The Burn training path itself (`crates/train/src/fit.rs`, `crates/cli`'s
  `train` command) is left in place, not deleted. It is not the recipe this
  run trained with going forward, but nothing about the model or the eval
  requires removing it, and the goal is to stop *using* Burn autodiff, not to
  make the old commands fail.

## Weight exchange format

**safetensors + config.json**, in both directions:

- `config.json` is exactly `train::checkpoint::Meta`'s JSON (the same schema
  already written as `meta.json` next to every Burn checkpoint): `nx, ny,
  patch, channels, dim, layers, heads, ff_mult, meta_dim, norm (mean, std,
  depth), dx, dt, g, depth, nu, mask, energy_r_max`.
- `model.safetensors` holds one F32 tensor per learnable parameter, named
  after burn's own field layout (checked against `burn-nn` 0.20.1 source:
  `nn::Linear` stores `weight: [in, out]` + `bias: [out]`, row layout, not
  PyTorch's `[out, in]`; `nn::Conv2d` stores `weight: [out, in, kh, kw]`,
  same as PyTorch; `nn::LayerNorm` stores `gamma`/`beta`, both `[dim]`):

  ```
  embed.weight [patch_dim, dim]   embed.bias [dim]
  meta.weight  [meta_dim, dim]    meta.bias  [dim]
  pos [1, tokens, dim]
  blocks.<i>.norm1.{gamma,beta} [dim]
  blocks.<i>.{q,k,v,proj}.weight [dim, dim]   .bias [dim]
  blocks.<i>.norm2.{gamma,beta} [dim]
  blocks.<i>.ff1.weight [dim, dim*ff_mult]    .bias [dim*ff_mult]
  blocks.<i>.ff2.weight [dim*ff_mult, dim]    .bias [dim]
  norm.{gamma,beta} [dim]
  head.token_mix.weight [dim, dim, 3, 3]      .bias [dim]
  head.to_cells.weight  [head_dim, dim, 1, 1] .bias [head_dim]
  head.conv1.weight [head_dim, head_dim+channels, 3, 3]  .bias [head_dim]
  head.conv2.weight [channels, head_dim, 3, 3]           .bias [channels]
  ```

  The PyTorch model (`training/jacobi2000_pt/model.py`) stores every "Linear"
  weight as an `[in, out]` `nn.Parameter` and does `x @ W + b` directly
  (never `nn.Linear`, whose PyTorch-native `[out, in]` layout would need a
  transpose on every export) so the safetensors bytes need no reshuffling in
  either direction.

- **Rust reads/writes safetensors itself** (`crates/jacobi2000/src/weights.rs`):
  the format is a small enough JSON-header-plus-buffer layout that adding the
  `safetensors` crate as a dependency for one F32-only read/write pair wasn't
  worth it; the files it produces and consumes are byte-identical to what the
  standard Python `safetensors` package writes and reads.
- `Jacobi2000::from_weights`/`to_weights`
  (`crates/jacobi2000/src/model.rs`) are the only place either side of the
  exchange has to agree on names; `Jacobi2000::forward` and the constraint
  projections never see a name. New CLI subcommands wrap them:
  `jacobi export-weights` (Burn checkpoint -> safetensors + config.json, for
  a parity baseline) and `jacobi import-weights` (safetensors + config.json
  -> Burn checkpoint, for training's output to reach `eval-metrics`, `web/`,
  and `round-trip` unchanged).

## Acceptance test

1. **Parity** (forward-pass agreement, not yet trained anything new): export
   an existing Burn checkpoint's weights (`model20`) with `export-weights`,
   load them into the PyTorch model, run both models' `forward` on the same
   fixed synthetic input (`jacobi forward-probe`, reconstructible in Python
   from `config.json` alone) and report max absolute difference. Must be at
   float32-arithmetic-order scale (`< 1e-4` on this model's activation
   range), not merely "close".
2. **Training**: train the model20 recipe in PyTorch on the dumped dataset,
   export the result, `import-weights` it into a Burn checkpoint, and run
   `jacobi eval-metrics --horizons 10,25,50 --project true --project-energy
   true` on it, unchanged from how model20 itself was scored
   (`docs/runs/2026-09-19-rung6.md`). The acceptance bars are the ones
   already in the README: corr@25 >= 0.9, corr@50 >= 0.8, amplitude ratio at
   50 ticks in [0.7, 1.3], wavefront lag at 50 ticks < 2 cells, `|bias| <=
   0.05` at 50 ticks, max relative energy violation <= 1e-6 with the
   projection on.
3. **Round trip**: `jacobi eval` (the plain rollout-error check) and, if a
   reverse checkpoint is trained, `jacobi round-trip`, both run on the
   imported checkpoint exactly as they run on a native Burn one — proof the
   exchange format loses nothing `web/`, `eval-metrics`, or `round-trip`
   depend on.

Numbers for all three are in `docs/runs/2026-09-29-pytorch-training.md`.
