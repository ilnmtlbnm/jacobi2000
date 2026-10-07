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
not separately checked against this engine on the random-init fixture —
only NdArray. The real-checkpoint section below closes that gap (engine's
WebGPU path is checked directly against PyTorch, not against Burn/wgpu).

## WebGPU kernels, real-checkpoint parity, gallery export, headless check

Follow-up in the same session, after the lead asked for a WebGPU path,
parity on the trained model rather than a random-init fixture, the gallery
re-exported, a headless check, and a command-line entry point. All of the
below ran on the Linux test machine (RTX 3080, Vulkan), via a git bundle
into an isolated `RUSTUP_HOME`/`CARGO_HOME` and `systemd-run` for the one
GPU-dispatching test, per the GPU-box runner's rules.

### WebGPU kernels

`crates/engine/src/shaders/*.wgsl`: `linear`, `layernorm`, `gelu`,
`add_inplace`, `add_broadcast_row`, `patchify`, `attention` (bidirectional,
the Chebyshev-stencil mask as a predicate rather than a materialized
`[tokens,tokens]` matrix), `conv2d`, `upsample`, `transpose_tc` — the same
naive, one-dispatch-per-op shapes already used across this project's other
kernels (lean's own `linear.wgsl`/`rmsnorm.wgsl`/`attn_prefill.wgsl`
patterns, t0-fast's `add_inplace.wgsl`), adapted to this model's tensor
layouts (burn's `[in,out]` linear weight, not GGUF's `[out,in]`; a
bidirectional mask predicate instead of attn_prefill's causal online
softmax, since every token here attends to a small fixed non-causal
neighbourhood or to everything). `crates/engine/src/gpu.rs` allocates every
weight buffer, every scratch buffer and every bind group once, at load
time (layer-specific bind groups for the weight-bearing ops; attention and
the two residual adds are shape-only and share one bind group across every
layer); `forward()` only `queue.write_buffer`s state/meta and replays one
command encoder (~64 dispatches for 4 layers), one submit, one async
readback (`lean_runtime::buffers::read`).

One real bug caught by actually running this on the box: `patchify.wgsl`'s
`Dims` struct named a field `patch`, a WGSL reserved keyword — wgpu's
shader validation rejected it (`Validation Error ... name 'patch' is a
reserved keyword`). Renamed to `patch_size`. (It was also caught once by
an accidental GPU dispatch on the M2's Metal adapter, from cargo unifying
`engine`'s `wgpu` feature across a multi-package test invocation — not
repeated; `cargo test -p engine` is run alone on the M2 from here on.)

### Parity, random-init fixture, GPU vs CPU (Vulkan, 3080)

`cargo test -p engine --features wgpu --test parity -- --nocapture`
(`gpu_matches_cpu_on_random_init`, plus the three CPU-only tests, all pass):

| case | max abs diff | max rel diff |
|---|---:|---:|
| 1-step, GPU vs CPU | 4.17e-7 | 4.19e-5 |
| 50-step rollout, GPU vs CPU | 3.50e-5 | — |

### Parity, real checkpoint (model20), engine vs PyTorch

`model20` (the README's best checkpoint) exists on the box already
exported to safetensors+config.json by the PyTorch training side
(`jacobi2000_pt.export`); `engine-cli`'s `--probe-out` dumps one forward
pass on the deterministic sinusoidal probe input (the same fixture
`jacobi forward-probe` and `jacobi2000_pt.parity` use), compared with
`python -m jacobi2000_pt.parity`:

| backend | max abs diff vs PyTorch | mean abs diff |
|---|---:|---:|
| engine CPU | 1.043e-06 | 1.806e-07 |
| engine WebGPU | 1.013e-06 | 1.788e-07 |

Output range both backends and PyTorch: `[-0.0410, 0.1616]`.

### Parity, real checkpoint, 50-step rollout against the solver

`rollout --model model20-pt --steps 50 --case drop --backend {cpu,webgpu}`
(center drop, amplitude 0.05, sigma 4, mass projection only — not the
official `eval-metrics` single-drop sampling or its energy projection, so
not directly comparable to the README's 0.9954; see below for a
same-checkpoint number under those closer conditions):

| backend | corr@50 vs solver | ms/step |
|---|---:|---:|
| CPU | 0.96587 | 108.87 |
| WebGPU | 0.96587 | 17.40 |

CPU and WebGPU agree to 5 decimal places on the real trained model, not
just the random-init fixture.

For a same-checkpoint number under the official methodology: `jacobi
import-weights` round-tripped `model20-pt` back into a Burn checkpoint,
then `jacobi eval-metrics --horizons 10,25,50 --project true
--project-energy true` (Burn, default seed — not the seed the README's
0.9954 was measured with, so not expected to match digit-for-digit):
corr(h)@10 = 0.9993, @25 = 0.9852, @50 = 0.9573 (corr(h,u,v)@50 = 0.9394),
max relative energy violation 0.000e0. This is in the range the training
run doc's own repeated `eval-metrics` runs already show for this
checkpoint under different seeds (0.9353-0.9954 across runs,
`docs/runs/2026-09-29-pytorch-training.md` secs. 5 and 7) — seed
sensitivity in the eval sampling, not a regression.

### Gallery export

`tools/export_web_models.sh`'s new `jacobi export-weights`-based form
exported 6 of the 7 checkpoints `web/model/index.json` lists
(`model-pt-curriculum`'s Burn checkpoint was not found on the box; the
other six were): `model20` (copied directly from the PyTorch export,
already in the right format), `model19`, `model-wide-1`, `model-wide-2`,
`model-sph-smoke`, `model-energy-smoke-2`, plus `model20`'s reverse
checkpoint (copied from the PyTorch export's `reverse-out`). Not committed
(`web/model/` is gitignored, as before).

### Headless (headed, real display) page check

Playwright's bundled Chromium cannot reach a real Vulkan adapter (falls
back to SwiftShader, software) — confirmed by running `web/test/smoke.mjs`
unmodified: it passes, `BACKEND: webgpu`, but at 1097 ms for 10 ticks vs
20 ms headed (below), consistent with software rendering. The real check
is headed Chromium on the box's own Xwayland display
(`SMOKE_HEADED=1 SMOKE_DISPLAY=:0 node test/smoke.mjs`, `/usr/bin/chromium`,
the same recipe as llm-web's lean-browser-box run):

- **WebGPU** (`BACKEND: webgpu`): every check passes — HORIZON, MASK,
  HIGHLIGHT, HEALTH, STEP, RACE, REWIND (peak error 0.00 cells, exact).
- **CPU forced** (`SMOKE_FORCE_CPU=1`, `--disable-gpu` so
  `requestAdapter()` genuinely fails rather than succeeding on
  SwiftShader): `BACKEND: cpu` confirmed; rollout errors match the WebGPU
  run to ~1e-7 (0.05167862 vs 0.05167858 at tick 10 — more real-browser
  evidence the two backends agree); HORIZON/MASK/HIGHLIGHT/HEALTH/STEP/
  REWIND all pass. **RACE fails**: `expected tick counter to advance by
  30, 0 -> 0`. Root cause, not a data race or a panic: the CPU backend's
  `step()` has no internal `.await` that actually suspends (the engine
  call is synchronous), so a `step(30)` call runs to completion in one
  browser microtask turn with no yield back to the JS event loop; the
  three `drop_at` calls the test fires "mid-flight" only run after
  `step(30)` has already finished, landing as ordinary immediate drops
  (each of which resets `tick` to 0) rather than queued ones. The WebGPU
  backend's real async GPU readback is what let this test's premise hold
  before. Not fixed here — flagged as the one open item.

### Timing vs the old Burn/wgpu build, same box, same model, same methodology

Built the pre-this-task `physics-engine` branch's `crates/web` (Burn/wgpu)
in a separate checkout, same box, loaded the same `model20` Burn
checkpoint, same headed-Chromium-on-Xwayland recipe, same probe script
(drop at centre, 5 samples of `step(20)`, median of `ms_per_tick()`):

| build | median ms/tick |
|---|---:|
| Burn/wgpu (pre-this-task) | 7.60 |
| This engine's WebGPU kernels | 15.75 |

**The new engine is slower here, about 2.1x, not faster** — the opposite
of what this section set out to show. Measured under some CPU contention
from an unrelated job on the box (a `tts_generate` process at ~720% CPU
during the run; not re-measured quiet, so take the absolute numbers as
approximate, though the 2x gap between two runs taken back-to-back under
the same contention is not explained by that alone). Likely cause, not
yet investigated: dispatch count. This engine issues ~64 separate compute
passes per forward call (naive, one op per dispatch); Burn's cubecl/wgpu
backend fuses and caches kernels across calls in ways this naive sequence
does not. Optimizing dispatch count (fusing the per-layer linear+residual
sequence, or batching the four attention heads into fewer passes) is the
next step before this can be called a win on speed — right now it is a
correct, portable, Burn-free, but slower replacement.

## What's left

1. **Dispatch-count optimization.** The WebGPU path is slower than the
   Burn build it replaces (measured above); reducing the ~64
   dispatches/forward is the likely fix, not yet done.
2. **`model-pt-curriculum`** gallery entry: its Burn checkpoint was not
   found on the box this session; the other six exported.
3. **The RACE headless-check regression** (CPU backend never yields
   mid-rollout, so a concurrent drop_at lands immediately rather than
   queued) — a real behavioral difference from the WebGPU path, not fixed.
4. **`spread_field`/xr ensemble**, if `web/app.js` ever wires it in.
5. A quiet (uncontended) re-measurement of the timing section above.
