# jacobi2000

A method for turning a PDE solver into one masked transformer over the
whole spacetime grid. Given some values (an initial condition, a set of
sensor readings, a target state) and blanks for the rest, the model fills
in the blanks by iterating to a fixed point. Forward simulation, rewinding
a state to see its past, and reconstructing hidden fields from sparse
measurements are the same operation with different values marked as known.

Named after Jacobi's iterative method: every point on the grid is updated at
once from its neighbors, repeated until the field stops changing.

The first system it has been trained and evaluated on is 2D shallow water.

## What it does

- Trains one transformer on a PDE solver's trajectories, with a stencil
  attention mask so each cell only looks at nearby cells, matching how the
  underlying physics propagates locally.
- Runs that same model forward (simulate ahead), in reverse (rewind a
  state to reconstruct its past), or filling in unknown regions of a field
  from partial observations: forward, inverse and rewind are the same call
  with different values marked as known.
- Enforces physical constraints (conserved mass, bounded energy) as a
  projection step after each prediction, not as a soft loss term.
- On 2D shallow water: simulates height and velocity fields with a
  reference solver written in Rust that runs natively and in the browser,
  and trains the transformer on its trajectories.

## Results on 2D shallow water

Numbers below are from the run logs under `docs/runs/`, referenced by path.

- Best trained model (`model20`, 128-wide, 4 layers, 995,843 parameters,
  trained with mass and energy projection on) scores correlation 0.9997 at
  10 ticks, 0.9988 at 25 ticks, and 0.9954 at 50 ticks against the solver,
  with an amplitude ratio of 1.03 and a wavefront lag under 0.02 cells at
  50 ticks (`docs/runs/2026-09-19-rung6.md`).
- Its maximum relative energy violation along a projected rollout is
  0.000e0, against a bound of 1e-6 (`docs/runs/2026-09-19-acceptance-2.md`).
- A 5x wider model (256-wide, 6 layers, 5,427,971 parameters) trained on a
  separate RTX 3080 machine does not beat `model20` on this evaluation at matched
  horizons, despite more parameters and more training steps
  (`docs/runs/2026-09-19-rung6.md`).
- Rewinding a state works well at short horizons and degrades gracefully:
  correlation to the true past is 0.98 at 10 ticks back, 0.93 at 25 ticks,
  and 0.80 at 50 ticks, using a reverse model trained at the same viscosity
  as the forward one (`docs/runs/2026-09-20-rewind-round-trip.md`).
- A full-attention baseline model (`model12`, no stencil mask, not trained
  with the energy projection) fails the same evaluation at 25 and 50 ticks,
  showing the stencil mask and the projection both matter
  (`docs/runs/2026-09-19-acceptance-2.md`).

See `CONCEPT.md` for the full design note, including the longer-term idea:
that iterating a masked model to a fixed point can act as a parallel-in-time
solver, converging in a small, roughly constant number of passes regardless
of how many time steps the trajectory covers.

## Running it

### Native

Requires a Rust toolchain. Training and evaluation use a GPU backend
(Metal on macOS, Vulkan or CUDA on Linux) through Burn/wgpu; tests and CPU
inference use the ndarray backend and need no GPU.

```
cargo build --release
cargo test --workspace
```

The `cli` binary (`jacobi`) generates training data, trains models, and
evaluates checkpoints:

```
cargo run --release -p cli -- train --help
cargo run --release -p cli -- eval-metrics --help
```

### Browser

`web/` is a self-contained demo page (solver and model side by side, a
divergence panel, and controls for time and viscosity). It needs a
populated `web/model/` directory of exported checkpoints, which is not
committed (see `web/README.md` and `tools/export_web_models.sh`).

```
cd web
npm install
./tools/export_web_models.sh   # from the repo root, after training a model
```

Then serve `web/` with any static file server and open `index.html`.

### Run it from the command line

`crates/engine-cli` (binary `rollout`) runs a checkpoint's inference
off Burn entirely — no Burn anywhere in its dependency tree
(`cargo tree -p engine-cli | grep -c burn` is `0`). It loads a
`model.safetensors` + `config.json` checkpoint (the format
`jacobi export-weights` and the PyTorch training export both write,
`docs/pytorch-training.md`), steps it next to the reference solver from a
built-in initial condition, and prints the model's correlation against
the solver every tick.

There is no public pretrained checkpoint to download yet (`model20` is a
local training artifact, gitignored under `results/checkpoints` and
`artifacts/`, never published). Train one with the `cli` crate's `train`
command (CPU-only `ndarray` backend by default, no GPU needed), export it
to the `rollout` format, then run it:

```
cargo run --release -p cli -- train --steps 3000 --single-drop true --out artifacts/model
cargo run --release -p cli -- export-weights --model artifacts/model --out web/model/model20
cargo run -p engine-cli --release --bin rollout -- \
    --model web/model/model20 --steps 50 --backend cpu
```

`train`'s defaults (`--dim 128 --layers 4`) already match `model20`'s
architecture; `--steps 3000` is a short run, not the full model20 recipe
(`docs/runs/2026-09-19-2.md` has the exact flags that one was trained
with, warm-started from earlier checkpoints). Raise `--steps` and add the
projection/mask flags from that run log for a model that matches the
correlations quoted above.

`--backend auto|cpu|webgpu` picks the backend (`auto`, the default, tries
WebGPU and falls back to CPU — capability, not a benchmark); `--backend
webgpu`/`threads` need this crate built with `--features wgpu`/`threads`
respectively. `--case drop|dam_break` picks the initial condition, and
`--out field.npy` dumps the final decoded height field as a NumPy `.npy`
file.

## Layout

- `crates/jacobi2000` — grid type, attention masks, model, constraint
  projection, the fill iteration.
- `crates/solvers` — the reference shallow-water solver.
- `crates/train` — data generation, training loop, evaluation metrics
  (Burn).
- `crates/cli` — the native training/evaluation command-line tool (Burn).
- `crates/engine` — inference off Burn: a hand-written CPU (optionally
  `rayon`-threaded) and WebGPU (hand-written WGSL) forward pass, reading
  `model.safetensors` + `config.json` checkpoints.
- `crates/engine-cli` — `rollout`, a command-line runner for `engine`
  (no Burn in its dependency tree; see "Run it from the command line").
- `crates/web` — the WASM build of `engine` and the solver for the
  browser, also off Burn.
- `web/` — the demo page.
- `docs/runs/` — dated research-log entries: parameters, commands, results.

## License

MIT. See `LICENSE`.
