# Jacobi2000 inference off Burn

Step 3 of the engine consolidation (1: lean-runtime extracted; 2: fhp/jacobi
moved onto it in `physics-engine`; 3: this step; 4: a trucs.ai lattice-gas
demo, later). Branch `inference-lean`, from `physics-engine`.

## What moved

- New crate `crates/engine`: a hand-written CPU forward pass over flat
  `f32` buffers, no Burn dependency. Reads the same `model.safetensors` +
  `config.json` checkpoints the PyTorch export already writes
  (`docs/pytorch-training.md`), which `jacobi2000::Jacobi2000::from_weights`
  on the Burn side also reads — the exchange format did not change, only
  who reads it for inference.
- `crates/web` (the browser demo) now depends on `engine` + `solvers` +
  `lean-runtime` only. `jacobi2000`, `train` and `burn` are gone from its
  dependency tree. `LoadedModel` wraps `engine::Model`; state is `Vec<f32>`;
  mass/energy projection goes through `engine::project`, ported from
  `jacobi2000::project` as plain slice arithmetic. Every `#[wasm_bindgen]`
  function keeps its old name and `async`-ness, so `web/app.js` needed only
  a file-extension change: `model.bin`/`meta.json` ->
  `model.safetensors`/`config.json` (two `fetch()` calls), matching
  `tools/export_web_models.sh`, which now shells out to `jacobi
  export-weights` (Burn checkpoint -> safetensors + config.json) instead of
  copying the old Burn binary record.
- `crates/cli` is unchanged. Its Burn-dependent commands
  (`train`/`eval`/`eval-metrics`/`round-trip`/`forward-probe`/
  `export-weights`/`import-weights`) are training- and acceptance-tooling,
  not a shipped inference path, and `docs/pytorch-training.md` already
  commits to keeping them (eval-metrics and round-trip are the acceptance
  gate for a PyTorch-trained checkpoint). `cli` has no standalone inference
  command outside that tooling, so there was nothing in it to move.
- `spread_field` (the per-cell xr-ensemble spread, CONCEPT sec. 4 task 10)
  is not ported. It was never wired into `web/app.js` (a pre-existing TODO
  in the old `crates/web/src/lib.rs`), so it is not part of the page's
  acceptance checks; it now returns an empty vector instead of running
  Burn's `train::xr_step`.

## Kernels reused vs new

No WGSL. This model (995,843 params at the shipped config, 256 tokens,
dim 128, 4 layers) is small enough that the forward pass is plain
loops over flat buffers — patchify, linear, layernorm, masked multi-head
attention, GELU (exact erf, via `libm`), 3x3/1x1 conv2d, and the two-axis
linear upsample the per-cell head uses, each hand-written once in
`crates/engine/src/cpu.rs`, matching `jacobi2000::model::Jacobi2000::forward`
operation for operation (same patchify/head-split/mask/conv-padding/
upsample-weight conventions). `crates/lean-runtime`'s `backend` module
(the WebGPU/Threads/Single capability ladder) is reused as-is for backend
selection; its `gpu`/`pipeline`/`buffers` helpers are not used yet because
no WebGPU kernel exists for this model (see "What's left").

The `threads` feature parallelizes the hot loops (linear, layernorm,
attention per query row, conv2d per output row) over `rayon`'s
`par_chunks_mut`, picked by the capability ladder, never autotuned.

## Parity

Burn has no trained checkpoint committed to this repo (weights are never
committed, per the README); the gate is numerical agreement between the
two *implementations* on a random-init checkpoint, not model quality
(`crates/engine/tests/parity.rs`). Exchanged through the same
safetensors + config.json format the PyTorch export writes. Reference:
`jacobi2000::Jacobi2000::forward`/`step`, Burn's NdArray (CPU) backend.

| case | mask | max abs diff | max rel diff | corr vs Burn |
|---|---|---:|---:|---:|
| 1-step, CPU single | Stencil r1 d1 | 2.53e-7 | 1.20e-2 | - |
| 1-step, CPU single | Full | 3.43e-7 | 3.26e-4 | - |
| 50-step rollout, CPU single | Stencil r1 d1 | 4.28e-5 | 5.93e-2 | 1.000000000 |
| 1-step, CPU threads | Stencil r1 d1 | 2.37e-7 | 2.25e-3 | - |
| 1-step, CPU threads | Full | 2.98e-7 | 2.86e-3 | - |
| 50-step rollout, CPU threads | Stencil r1 d1 | 4.29e-6 | 2.28e-2 | 1.000000000 |

(`cargo test -p engine --test parity -- --nocapture`, and again with
`--features threads`; numbers are each run's own stdout.) Max relative
diff is taken pointwise against `max(|a|,|b|,1e-6)`, so it is dominated by
near-zero activations and is not itself a useful error measure; max
absolute diff and the 50-step correlation are. No PyTorch-side fixture was
exercised in this pass (none is wired to run offline here); `jacobi
forward-probe` already exists as the byte-identical cross-language fixture
described in `docs/pytorch-training.md` and would be the next thing to
point this crate's own forward pass at.

Burn's **wgpu** backend (as `crates/web` ran on before this change) was
not separately checked against this engine — only NdArray. The two Burn
backends are expected to agree with each other to float-order-of-operations
tolerance (that was never a question this task raised), so this is a gap
in the parity chain, not a known discrepancy.

## Backends tested

- **CPU single thread**: `cargo test -p engine`, `cargo test -p web -p
  jacobi2000 -p solvers -p lean-runtime`, `cargo clippy -p web -p engine --
  -D warnings` — all on the M2, native, no GPU. All pass.
- **CPU threads**: same, with `--features threads` — all pass (parity
  table above).
- **wasm32**: `wasm-pack build crates/web --target web` succeeds (M2,
  native build, no browser run). `web_bg.wasm` is 207,104 bytes
  (release, wasm-opt'd).
- **WebGPU native (Vulkan on the Linux test machine)**: not run. There is
  no WebGPU kernel for this model yet (see below), so there is nothing
  backend-specific to test there; this step's GPU-box work was limited to
  native `cargo test`/`clippy`, which do not need a GPU and were run on the
  M2 instead since the box's GPU was in use by the Jacobi2000 training job
  (energy/slack units) and this task's own CPU-only work never touches it.
- **Headless browser page check**: not run (no browser invoked this
  session, per the M2-is-the-dev-machine-right-now rule; the Linux test
  machine's browser availability was not checked this session).

## What's left

1. **WebGPU kernel.** No WGSL exists for this model. The forward pass is
   small (256 tokens, dim 128) — a naive one-dispatch-per-op sequence
   (linear/layernorm/attention/conv2d/upsample, each a small compute
   shader built on `lean-runtime`'s `gpu`/`pipeline`/`buffers` helpers)
   would close the WebGPU slot in the capability ladder. Parity gate before
   any of it ships: same `tests/parity.rs` fixture, run through the GPU
   path.
2. **Gallery checkpoints.** `web/model/` is gitignored and was not
   regenerated here (no `artifacts/` checkpoints exist in this worktree to
   export). `tools/export_web_models.sh` is updated to the new format but
   was not run end-to-end against a real checkpoint.
3. **`jacobi forward-probe` cross-check.** Add an `engine`-side probe that
   reads the same deterministic input and diffs against an existing
   `forward-probe` dump, closing the parity chain to the PyTorch reference
   docs/pytorch-training.md already describes.
4. **`spread_field`/xr ensemble**, if `web/app.js` ever wires it in.
5. **Headless page check** on whichever machine has a browser available,
   once `web/model/` is populated.
