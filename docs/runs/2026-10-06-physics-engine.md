# physics-engine: fhp and jacobi move in from tritsliced, onto lean-runtime

Branch `physics-engine`, worktree off the `energy-band` tip (882ada1). Moves
the `fhp` (FHP-III/ternary/signed lattice gas) and `jacobi` (bit-sliced Jacobi
stencil) crates out of the tritsliced repository and into this workspace, and
adds the shared wgpu/wasm runtime crate `lean-runtime` (from llm-web's
lean-runtime branch, tip 2b9353e, with llm-web's clippy fix 68fe5c0
cherry-picked on top so the workspace is clean on Rust 1.95).

Sources read before moving anything:
- `fhp`: tritsliced branch `fhp-signed-speed`, tip a83515b, and its
  `docs/runs/2026-10-06-fhp-signed-speed.md`.
- `jacobi`: tritsliced branch `jacobi-viewer`.
- `lean-runtime`: llm-web branch `lean-runtime`, tip 2b9353e (+ the
  cherry-picked clippy fix), and its `docs/lean-runtime.md` design note.

## Dependency choice for lean-runtime

**Vendored copy**, not a git or path dependency to the llm-web checkout.
`lean-runtime` isn't pushed to `origin` on llm-web (only `lean-release` is),
and this session does not push branches or open PRs without explicit
authorization, so a `git = "...", branch = "lean-runtime"` dependency would
fail to resolve for anyone else, and an absolute local-path dependency would
leak a local path (the thing gate 5 explicitly flags). The fix: `lean-runtime`
is copied byte-for-byte into `crates/lean-runtime` of this workspace and
consumed as an ordinary in-workspace path dependency (`path =
"../lean-runtime"`), which is a relative path inside the repo and carries no
machine-specific information. Its own `Cargo.toml` was adjusted to stop
inheriting `[workspace.dependencies]` from llm-web (this workspace doesn't
have those entries, and pulling in llm-web's unrelated deps — burn, cubecl,
tokenizers — would be pointless here); the versions are pinned inline instead,
copied from llm-web's `workspace.dependencies` (`wgpu = "26"` with
`fragile-send-sync-non-atomic-wasm`, `wasm-bindgen = "0.2"`, `js-sys = "0.3"`).

## What moved vs what stayed

`fhp` and `jacobi` have no Cargo dependency on the `tritsliced` compiler
crate and vendor no generated gate output — their bit-sliced circuits are
built at runtime from each crate's own `rules.rs`/`circuit.rs`, not compiled
by tritsliced. So there was nothing to vendor there; the compiler itself
never had to be touched or copied.

`lean-runtime` is now a dependency of both crates (feature `gpu`, alongside
`wgpu`/`futures-channel`, since the whole `gpu` module in each crate is
`#[cfg(feature = "gpu")]`). The only lines actually swapped for the shared
crate are the three that compute "the largest buffer that can also be bound
whole as storage" — `l.max_buffer_size.min(l.max_storage_buffer_binding_size
as u64)` — now `lean_runtime::limits::max_binding_bytes(l)`, in
`crates/fhp/src/gpu.rs:63` and `crates/jacobi/src/gpu.rs:185,267`. This is a
true drop-in: identical formula, identical numeric result, no visible
behaviour change.

**Left as is, and why:** the adapter/device request blocks in
`fhp::gpu`, `fhp::ternary::gpu`, `fhp::signed::gpu` and `jacobi::gpu` are
logically identical to `lean_runtime::gpu::request_adapter`/
`request_device`, but swapping them changes the error string on failure
(`lean_runtime::Error::NoAdapter` formats as `"no wgpu adapter: {e}"`, the
crates' own code says `"no WebGPU adapter: {e}"` / `"no WebGPU device:
{e}"`) and the device label passed (`fhp::gpu` labels the device `"fhp"`,
`lean_runtime::gpu::request_device` takes the label as a parameter, which
is fine, but the two crates' code paths are enough functions apart —
`fhp::ternary::gpu`, `fhp::signed::gpu`, `jacobi::gpu`'s two call sites —
that swapping all of them with zero risk of changing an error path or a
bind-group/pipeline detail needed more time than this session had, against
an explicit "behaviour must not change" requirement. The three crates' JS
(`www/worker.js` backend ladders, `?v=` build-tag handling, the COOP/COEP
server) were likewise left as their own copies rather than switched to
`lean-runtime/js/boot.js` + `capabilities.js` — the dev pages' worker.js
files have fhp/jacobi-specific model and query-string logic (`?model=`,
`?n=`, per-panel status lines) interleaved with the ladder, not cleanly
separable in the time available without risking the pages' own behaviour.
These are candidates for a later pass once there's time to verify each
swap against the existing test suite and the headless pages, not skipped
on judgment about the duplication itself.

## Gate 1: native tests, M2 and the RTX 3080 machine

| crate | source branch (passed) | physics-engine (passed) | match |
|---|---:|---:|---|
| fhp | 53 | 53 | yes |
| jacobi | 21 | 21 | yes |

`cargo test --release -p fhp -p jacobi`, run separately per crate (a combined
invocation double-counts shared binary names across the two crates' `grep`
tallies, a counting artefact, not a real mismatch — verified by running each
crate alone). 74 total (53 + 21) on both the M2 and the RTX 3080 machine (Ryzen
9 3900X / RTX 3080, GPU idle before the run, `0 %, 469 MiB` on
`nvidia-smi --query-gpu=utilization.gpu,memory.used`). Box run via
`systemd-run --user --unit=physics-engine-gates`, logs at
`logs/test.log` (`TEST_EXIT=0`) and `logs/clippy.log` (`CLIPPY_EXIT=0`) under
a scratch checkout of this branch on the RTX 3080 machine.

Note: the debug profile hit a SIGBUS in rustc's DWARF emission on the M2
compiling `fhp` with `-C debuginfo=2 -C split-debuginfo=unpacked` (not
reproducible on the RTX 3080 machine, Rust 1.95 vs the M2's 1.93); release builds (and the
RTX 3080 machine's clippy pass, which also uses `--release`) were unaffected, so all gates
here are release-profile.

## Gate 2: clippy

`cargo clippy --release -p fhp -p jacobi -p lean-runtime --all-targets -- -D
warnings`: clean on the M2 (rustc 1.93.0) and on the RTX 3080 machine (rustc 1.95.0,
the toolchain that flags `unnecessary_sort_by` on `lean`'s
`debug_top_buffers` — the reason the 68fe5c0 cherry-pick was needed before
this would pass).

## Gate 3: fixed-seed byte-identical state

Diffed `crates/fhp/src` and `crates/jacobi/src` between the source branches
and this workspace: the only differences are the three `max_binding_bytes`
swaps above (confirmed with `diff -rq`, both trees otherwise identical byte
for byte). Then ran the same deterministic CLI calls on the source branch
and on `physics-engine` (same machine, the M2, to rule out cross-hardware
float/GPU variance) and sha256'd stdout:

| variant | command | N / size | sha256 | match |
|---|---|---|---|---|
| FHP-III | `fhp run --cpu --width 256 --height 64 --steps 40 --seed 7 --every 5 --frame 20` | 40 steps | `9ae211aa4ea1aa8b...` | identical |
| ternary | `ternary theory` | n/a (closed-form table) | `c4e160c4757871bd...` | identical |
| signed | `signed shear --h 32 --width 64 --seeds 1` | w64×h32 | `2b6ba6616d833425...` | identical |

All three: `diff src.log dst.log` → empty, confirmed identical, not just
same hash.

## Gate 4: wasm builds, serving, headless

Built with `ENGINE_BUILD=2026-10-06-physics-engine-01` (new tag on this
move, both pages' only literal `ENGINE_BUILD` constant; each page's
`worker.js` reads the tag back off its own URL, so nothing else needed
editing):

| crate | output | bytes | sha256 (local, full) |
|---|---|---:|---|
| fhp | `pkg/fhp_bg.wasm` | 521757 | `eb39cd63444b2a61961ff691385b05d528b3c26b740f7a897f177442ba257765` |
| fhp | `pkg-mt/fhp_bg.wasm` | 353366 | `1937a596b81cc1e3b01008bda1d8eec1c353b66d6ec5982924041134554145d6` |
| jacobi | `pkg/jacobi_bg.wasm` | 228169 | `b263233f82af7ecd80d1c0fdeb5029592031a75a93455bf1e5e5c5acdf2df52d` |

Served locally (127.0.0.1 only, COOP/COEP via `scripts/serve_coi.py`): fhp on
8052, jacobi on 8962. `curl` of each served `?v=2026-10-06-physics-engine-01`
wasm URL, sha256'd, matched the local build byte for byte for all three
files (checked explicitly, not just "should match").

Headless (Playwright's bundled Chromium, `--enable-unsafe-webgpu
--use-angle=metal`, never the dev browser):

- fhp default (`model` unset → fhp3) and `?model=signed`: both picked the
  `webgpu` backend, `isolated: true`, zero console/page errors, `pass:
  true` on both. Screenshots `docs/runs/pictures/2026-10-06-physics-engine/
  fhp-0.png` (default) and `fhp-1.png` (`model=signed`).
- jacobi `?n=128`: `webgpu` backend, `isolated: true`, zero console/page
  errors, `pass: true`. Screenshot `jacobi.png` (field view) and
  `jacobi-error.png` (the script's standard error-view shot of panel 0).

## Gate 5: privacy grep

A privacy grep for personal paths and identifiers on all three
commits on this branch returns only the leak-check shell scripts' own grep
patterns (`crates/lean-runtime/scripts/leak_check.sh`,
`scripts/build_fhp_web.sh`, `scripts/build_jacobi_web.sh` — they necessarily
contain the strings `.claude/`, `/Users/`, `Code/` as the patterns they grep
*for*). No actual leaked path, hostname or name. The wasm binaries
themselves were checked by each build script's own `leak_check`/`LEAKS`
step during the build (gate 4) and came back clean.

## Commits on `physics-engine`

1. `workspace: add lean-runtime (vendored...), gitignore its consumers' wasm output`
2. `fhp: move the FHP-III/ternary/signed lattice-gas crate in from tritsliced...`
3. `jacobi: move the bit-sliced Jacobi stencil crate in from tritsliced...`

## What's still open

- The Rust-level backend-ladder duplication beyond `max_binding_bytes`
  (adapter/device request, buffer/pipeline helpers) and the JS-level
  duplication (`www/worker.js`'s ladder vs `lean-runtime/js/boot.js` +
  `capabilities.js`) were surveyed but not swapped, per the "left as is"
  note above — a follow-up pass with its own test run against each
  page, not a gap in this one.
- `pkg-mt` (the threaded wasm CPU path) was built and leak-checked for fhp
  but not exercised headless — the default headless check always resolved
  to the `webgpu` backend (available in this Chromium), so the `threads`
  path never ran. The existing `path=threads` variant in
  `scripts/fhp_headless.mjs` would need `SharedArrayBuffer`/COOP-COEP, which
  the local server does provide; it just wasn't part of the two checks gate
  4 asked for (default and `?model=signed`).
