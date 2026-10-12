# PyTorch training, Rust inference

## Who does what

- **Training** is PyTorch (`training/jacobi2000_pt`): the model
  (`model.py`), the training loop (`train.py`, the model20 recipe of
  `docs/runs/2026-09-19-2.md`: AdamW, warmup + cosine decay, pushforward
  rollout curriculum, noise injection, mass + energy projection in the loop,
  spectral loss term), the six-stage curriculum (`curriculum.py`), and the
  scoring functions used during training (`metrics.py`).
- **Data generation** stays behind the Rust solver. `crates/solvers` is the
  one place the shallow-water equations are written down. `dump-dataset`
  (`crates/engine-cli`) draws single-drop trajectories from it and writes
  full raw fields to disk; Python reads flat `f32` files and never touches
  the physics.
- **Inference** is `crates/engine`: a hand-written CPU (optionally
  rayon-threaded) and WebGPU forward pass over the exported checkpoint, with
  the mass and energy projections. `rollout` steps a checkpoint next to the
  solver; `eval-metrics` scores it at several horizons on fresh drops.

## Weight exchange format

**safetensors + config.json**, in both directions:

- `config.json` holds: `nx, ny,
  patch, channels, dim, layers, heads, ff_mult, meta_dim, norm (mean, std,
  depth), dx, dt, g, depth, nu, mask, energy_r_max`.
- `model.safetensors` holds one F32 tensor per learnable parameter, named
  as below. Linear layers store `weight: [in, out]` + `bias: [out]` (row
  layout, not PyTorch's `[out, in]`), convolutions store `weight: [out, in,
  kh, kw]` as in PyTorch, layer norms store `gamma`/`beta`, both `[dim]`:

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

- `crates/engine` reads safetensors itself (`src/weights.rs`): the format is
  a small enough JSON-header-plus-buffer layout that a dependency for one
  F32-only reader was not worth it. The files it consumes are byte-identical
  to what the standard Python `safetensors` package writes.

## Parity with PyTorch

`python -m jacobi2000_pt.parity --out crates/engine/tests/fixtures/pt-tiny`
(run from `training/`) writes a tiny seeded checkpoint plus PyTorch's outputs
on a fixed input: the one-step forward pass for the stencil and full masks,
and the stencil rollout after 1, 10 and 50 steps. `cargo test -p engine
--test parity` checks the CPU forward pass against them (max abs difference
below 1e-5 for one step, below 1e-4 over 50 steps); with `--features wgpu` it
also checks the WebGPU pass. The fixture is 224 kB.

## Acceptance

`eval-metrics --model <dir> --horizons 10,25,50 --project-energy true` scores
a checkpoint against the solver. The bars: corr@25 >= 0.9, corr@50 >= 0.8,
amplitude ratio at 50 ticks in [0.7, 1.3], wavefront lag at 50 ticks < 2
cells, `|bias| <= 0.05` at 50 ticks, max relative energy violation <= 1e-6
with the projection on. The research log under `docs/runs/` records the
numbers for the checkpoints trained so far
(`docs/runs/2026-09-29-pytorch-training.md` for the PyTorch runs).
