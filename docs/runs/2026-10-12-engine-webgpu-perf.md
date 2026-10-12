# Engine WebGPU path: re-measurement and easy wins

Branch `perf`, from `inference-lean`. Follows the timing section of
`2026-10-07-inference-lean.md`, which recorded the engine's WebGPU path as
about 2.1x slower than the Burn/wgpu build it replaced, measured while the
GPU was busy. This run re-measures on an idle GPU, finds where the time
goes, and applies the changes that needed no new architecture. Every change
is its own commit; kernels are portable (workgroups of 64 to 256, no
subgroups, no per-device variants, nothing chosen by measurement at run
time).

## Parameters

| item | value |
|---|---|
| model | shipped config: dim 128, 4 layers, 4 heads, 64x64 grid, patch 4 (256 tokens), 3 channels, Stencil r1 d1 |
| engine weights | `model20-pt` export (safetensors); timing does not depend on weight values |
| Burn weights | `model20` Burn record, same architecture |
| native machine | the M2 laptop (Metal) |
| browser | Playwright's bundled Chromium, headless, ANGLE Metal; adapter reported vendor apple, architecture metal-3 (without the ANGLE flag the same build reports a SwiftShader software adapter, which was not used) |
| native engine metric | `GpuModel::step` + host mass projection per tick (upload, one submit, readback), 20 steps x 5 repeats, median of repeats; no reference solver |
| native Burn metric | `Jacobi2000::step` + `project_mass` on GPU tensors, 50 steps x 7 repeats after a warmup rollout. "chained" reads back once at the end; "readback each" reads the state back after every step (what the engine does) |
| browser metric | `ms_per_tick()` of the page's wasm: `step(20)` wall time / 20, 7 repeats, median; includes the CPU reference solver step, so it is not comparable to the native rows |
| rounds | variants run round-robin, 3 rounds; the table shows the median of the 3 per-round medians. Round 1 ran on a GPU that had been idle and is 5 to 15 percent slower for the later variants |
| load | 1-minute load average 1.4 to 1.8 during the native run, no rustc running; GPU and browser locks held |

## Results: native, ms per step

| variant | commit | dispatches per step | ms per step (median) |
|---|---|---:|---:|
| engine as of `inference-lean` | a669dd8 | 64 | 137.9 |
| + cooperative attention (64 threads per head/query) | 06e0ae0 | 64 | 13.87 |
| + tiled 32x32 matmul, 2x2 register blocking | c15200f | 64 | 11.59 |
| + attention visits stencil candidates only | 4e42d0a | 64 | 9.64 |
| + conv2d as tiled implicit GEMM | 3014181 | 64 | 8.24 |
| + readback copy in the compute encoder | 1c95f17 | 64 | 8.19 |
| + GELU and residual adds fused into epilogues | d097b12 | 50 | 8.09 |
| + weights uploaded with `write_buffer` | 8dad31f | 50 | 8.13 |
| + q, k, v as one matmul | 6b810aa | 42 | 7.97 |
| Burn wgpu, chained (one readback at the end) | - | n/a | 11.17 |
| Burn wgpu, readback each step | - | n/a | 11.85 |

Per step, before and after: 1 submit before the readback change and 1 after
(the readback used to be a second submit), 1 readback, 0 buffer allocations
in `forward()` apart from the readback staging buffer, 0 bind group
creations in `forward()` (all built at load).

## Results: browser, ms per tick (engine includes the CPU reference solver)

| build | ms per tick, round 1 | ms per tick, round 2 |
|---|---:|---:|
| engine at a669dd8 | 125.1 | 124.5 |
| engine at 6b810aa | 3.8 | 3.8 |
| Burn/wgpu (`physics-engine`) | 11.95 | 12.05 |

The model field after 20 ticks from the same drop agrees between the two
engine builds (sum 5.026633 against 5.026631 over 8192 values, max abs
0.013667 in both).

## Where the time went (GPU timestamps, not committed)

Timestamp queries around each pass, read at the variant of 3014181 (total
GPU 7.05 ms of an 8.2 ms step):

| op | passes | GPU microseconds |
|---|---:|---:|
| attention | 4 | 1489 |
| conv1 (per-cell head) | 1 | 738 |
| token_mix conv | 1 | 674 |
| ff1 | 4 | 904 |
| ff2 | 4 | 915 |
| q, k, v | 12 | 777 |
| proj | 4 | 250 |
| layernorm | 9 | 361 |
| everything else (adds, gelu, transpose, upsample, patchify, to_cells, conv2) | 25 | 945 |

Before the attention change the same measurement (skipping the dispatch and
differencing the step time) put attention at about 105 of 131 ms.

## Observations

- The slowdown recorded on 2026-10-07 was not the dispatch count. Native, the
  engine was 12x slower than Burn on an idle GPU (137.9 against 11.17), and
  about four fifths of it was one kernel: attention ran one single-thread
  workgroup per (head, query) with a 1024-float private array, and scanned
  all 256 keys. Replacing it with a 64-thread workgroup took the step from
  137.9 to 13.87 ms.
- Matmul and conv were the next cost. Both are compute-bound at this size
  (a 256x128x512 matmul is 17M multiply-adds). The 32x32 tile with 2x2
  register blocking took the step from 13.87 to 11.59 ms (matmul), and from
  9.64 to 8.24 ms (conv). The conv is the same tiled kernel with the
  k*k*cin gather as the B tile.
- Visiting only the stencil neighbours in attention (9 keys instead of a
  256-key scan with a mask test) took the step from 11.59 to 9.64 ms.
- Fusing GELU, both residual adds and q/k/v removed 22 of 64 dispatches and
  moved the native step by 0.2 ms in total, because on Metal an empty pass
  costs a few microseconds. Those changes are worth keeping for the browser
  and for devices with a higher per-pass cost, but they are not where the
  gap was.
- The remaining floor is about 1.3 to 1.7 ms per step for upload, submit,
  readback and the host-side add and mass projection (measured by skipping
  every dispatch: 1.65 ms before the readback change).
- The final native engine is 7.97 ms per step against Burn's 11.85 with the
  same per-step readback, and 11.17 chained. In the browser it is 3.8 against
  12.0 ms per tick with the reference solver included on both sides. The
  browser number being lower than the native one is a measured difference
  between the two runtimes' Metal paths (not investigated; Dawn may merge
  consecutive compute passes), not a different workload: the output field
  matches.
- What did not help: a direct (non-tiled) conv kept for the 3-channel output
  conv, which a timestamp run had flagged as slower under the tiled kernel
  (663 against 139 microseconds). The full-step time did not move
  (8.19 to 8.34 ms, within run-to-run noise), so it was dropped.
- Parity after each change: `cargo test -p engine --features wgpu --test parity`,
  1-step GPU against CPU max abs diff 2.5e-7 to 3.6e-7 (gate 1e-2), 50-step
  rollout max abs diff 5e-6 to 6e-5 (gate 1.0). Two cases were added for the
  new attention enumeration: full attention and a dilated radius-2 stencil
  (clipped at the grid edge), 1-step max abs diff below 5e-7 (gate 1e-3).
- Weights were being uploaded with mapped-at-creation buffers
  (`create_buffer_init`); they now go through `queue.write_buffer` like every
  other upload, since mapped-at-creation buffers were seen holding wrong
  bytes in Chromium on Vulkan.
- Not measured: Chromium on Vulkan/NVIDIA. The native and browser numbers above are from one
  Metal GPU; the kernels use no Apple-specific features, but the gains are
  not validated on the other device families here.
