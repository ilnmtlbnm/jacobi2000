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

## Results on 2D shallow water

- Best trained model (`model20`, 128-wide, 4 layers, 995,843 parameters,
  trained with mass and energy projection on) scores correlation 0.9997 at
  10 ticks, 0.9988 at 25 ticks, and 0.9954 at 50 ticks against the solver,
  with an amplitude ratio of 1.03 and a wavefront lag of -0.007 cells at
  50 ticks (`docs/runs/2026-09-19-step6.md`).
- Its maximum relative energy violation along a projected rollout is
  0.000e0, against a bound of 1e-6 (`docs/runs/2026-09-19-acceptance-2.md`).
- A 5x wider model (256-wide, 6 layers, 5,427,971 parameters) trained on a
  desktop GPU does not beat `model20` on this evaluation at matched
  horizons, despite more parameters and more training steps
  (`docs/runs/2026-09-19-step6.md`).

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
