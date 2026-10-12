# jacobi2000

A PDE solver, learned as one transformer over the whole space-time grid.

Give the model some values (an initial condition, a few sensor readings, a target state) and blanks for the rest, and it fills in the blanks by iterating to a fixed point. Running the simulation forward, rewinding a state to see its past, and reconstructing hidden fields from sparse measurements are the same call with different values marked as known. The name comes from Jacobi's iterative method: every point of the grid is updated at once from its neighbors, and this is repeated until the field stops changing.

The model is trained in PyTorch. Inference is a Rust engine with hand-written WGSL kernels. It runs on WebGPU where there is one, on CPU threads otherwise, and on a single thread as the last resort, natively and in the browser from the same source. The first system I trained it on is 2D shallow water.

![The reference solver (left), the model (middle) and their difference (right), 40 ticks after a drop of water, running in the browser on WebGPU](docs/images/web-demo.png)

*The demo page after 40 ticks. Left: the classical solver. Middle: the model, with its correlation to the solver (0.997 here). Right: the difference between the two.*

## Results on 2D shallow water

- The best model (`model20`: 128 wide, 4 layers, 995,843 parameters, trained with mass and energy projection on) has a correlation with the solver of 0.9997 at 10 ticks, 0.9988 at 25 ticks and 0.9954 at 50 ticks. At 50 ticks its amplitude ratio is 1.03 and its wavefront lag is -0.007 cells (`docs/runs/2026-09-19-step6.md`).
- Along a projected rollout its maximum relative energy violation is 0.000e0, against a bound of 1e-6 (`docs/runs/2026-09-19-acceptance-2.md`).
- Rewinding works over short horizons. A forward model steps a state N ticks ahead, and a reverse model, trained with the time axis flipped, steps it back. The correlation with the original state is 0.98 after 10 ticks, 0.93 after 25 and 0.80 after 50, with the reverse model trained at the same viscosity as the forward one (0.9817, 0.9263 and 0.8025 in `docs/runs/2026-09-20-rewind-round-trip.md`). The same table shows what is hard: amplitude comes back 10 to 25 percent low, and a reverse model trained without viscosity does worse at every horizon.
- Two choices in the architecture carry the long-horizon accuracy. A full-attention model of the same kind (`model12`, no stencil mask, no energy projection) fails the same evaluation: its correlation is 0.2242 at 25 ticks and -0.0637 at 50 ticks, where `model20` stays above 0.99 (`docs/runs/2026-09-19-acceptance-2.md`). That baseline differs from `model20` in both choices at once, so this run does not separate the contribution of each.
- A 5x wider model (256 wide, 6 layers, 5,427,971 parameters), trained on a desktop GPU, does not beat `model20` on this evaluation at matched horizons, despite more parameters and more training steps (`docs/runs/2026-09-19-step6.md`).

## Running it

### Native

Requires a Rust toolchain. Training is PyTorch (below); the Rust crates run
inference on the CPU, or on a GPU through wgpu (Metal on macOS, Vulkan on
Linux) when built with `--features wgpu`. Tests need no GPU.

```
cargo build --release
cargo test --workspace
```

`crates/engine-cli` has three binaries: `dump-dataset` writes training
trajectories from the reference solver, `rollout` steps a checkpoint next to
the solver, and `eval-metrics` scores a checkpoint's rollouts against the
solver at several horizons:

```
cargo run --release -p engine-cli --bin eval-metrics -- --model <checkpoint-dir> --project-energy true
```

### Browser

`web/` is a self-contained demo page (solver and model side by side, a
divergence panel, and controls for time and viscosity). It needs the wasm
build in `web/pkg` and a populated `web/model/` directory of exported
checkpoints, neither of which is committed (see `web/README.md`).

```
scripts/build_web.sh                                  # needs wasm-pack
tools/export_web_models.sh model20=models/model20     # a checkpoint directory, see below
python3 -m http.server 8000 --directory web
```

Then open `http://localhost:8000`. (`cd web && npm install && npm test` runs
the headless check of the page.)

### Run it from the command line

`crates/engine-cli` (binary `rollout`) runs a checkpoint's inference. It loads
a `model.safetensors` + `config.json` checkpoint (the format the PyTorch
training export writes, `docs/pytorch-training.md`), steps it next to the
reference solver from a built-in initial condition, and prints the model's
correlation against the solver every tick.

The trained `model20` is on Hugging Face. Download it and run it:

```
pip install huggingface_hub
hf download ilnmtlbnm/jacobi2000-model20 --local-dir models/model20
cargo run -p engine-cli --release --bin rollout -- \
    --model models/model20 --steps 50 --backend cpu
```

`--backend auto|cpu|webgpu` picks the backend (`auto`, the default, tries
WebGPU and falls back to CPU); `--backend
webgpu`/`threads` need this crate built with `--features wgpu`/`threads`
respectively. `--case drop|dam_break` picks the initial condition, and
`--out field.npy` dumps the final decoded height field as a NumPy `.npy`
file.

### Train one yourself

Training is PyTorch, in `training/jacobi2000_pt`. The solver stays in Rust:
`dump-dataset` writes the trajectories once, and Python reads the flat files.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e training
cargo run --release -p engine-cli --bin dump-dataset -- --pool 24 --out data/ds
python -m jacobi2000_pt.train --dataset data/ds --steps 300 --out artifacts/model
cargo run -p engine-cli --release --bin rollout -- \
    --model artifacts/model --steps 50 --backend cpu
```

## Also in this repo

- **A lattice gas.** `crates/fhp` is the FHP-III lattice gas (Frisch, Hasslacher and Pomeau), with ternary and signed ternary variants, as a per-site reference and bit-sliced CPU and WGSL engines that agree bit for bit. Flow past a plate forms a von Karman vortex street (below: the signed ternary variant on WebGPU, a 4096 x 1024 lattice after 13,000 steps, Reynolds number about 200, coloured by vorticity). The dev page at `crates/fhp/www` runs it in the browser; `crates/fhp/README.md` has the model, the collision circuit and the sources.

![The wake behind a plate in the signed ternary lattice gas: two vortices of opposite sign have formed and a third is starting to shed](docs/images/fhp-street.png)

- **A bit-sliced Jacobi stencil.** `crates/jacobi` solves the 2D Laplace equation by Jacobi iteration as a gate circuit at 4, 8, 12 and 16 bits, next to integer and f32 references.
- **Three small Game of Life models.** `crates/life-models` has a BERT-style model, a 9-number MLP and a toroidal stencil model, with CPU and WGSL forward passes.


## Layout

- `crates/solvers`: the reference shallow-water solver.
- `crates/engine`: inference: a hand-written CPU (optionally
  `rayon`-threaded) and WebGPU (hand-written WGSL) forward pass, reading
  `model.safetensors` + `config.json` checkpoints.
- `crates/engine-cli`: `rollout`, `eval-metrics` and `dump-dataset`, the
  command-line tools around `engine` and `solvers`.
- `crates/fhp`: the FHP-III lattice gas (and ternary and signed variants) as per-site reference, bit-sliced CPU and wgpu/WGSL engines, native and wasm from one source; see its README.
- `crates/jacobi`: 2D Laplace by Jacobi iteration as a bit-sliced gate circuit at 4, 8, 12 and 16 bits, with integer and f32 references and bit-sliced CPU and wgpu/WGSL engines.
- `crates/life-models`: the three small Game of Life models (BERT of Life, the 9-number MLP, the toroidal stencil model), CPU and WGSL forward passes reading safetensors + config.json.
- `crates/web`: the WASM build of `engine` and the solver for the browser.
- `crates/lean-runtime`: device acquisition, buffers, pipelines and the
  backend ladder (WebGPU, CPU threads, single thread, by capability).
- `training/jacobi2000_pt`: PyTorch model, training loop, curriculum and
  export.
- `web/`: the demo page.
- `docs/runs/`: dated research-log entries: parameters, commands, results.

## License

MIT. See `LICENSE`.
