# lean-runtime

The wgpu and wasm plumbing an engine needs before its own kernels: the
backend ladder (WebGPU, then CPU threads, then one CPU thread, chosen by
capability only), device acquisition with the adapter's own limits, buffers
and awaited readback, pipelines, size checks against the device limits, a
clock, and the wasm threads pool. No model or physics types.

Origin: this crate is the runtime layer written for
[idle-intelligence/llm-web](https://github.com/idle-intelligence/llm-web)
(`crates/lean-runtime`, commit 648a659). llm-web's main branch (checked at
6daf322) does not contain it as a crate: its `lean-core` acquires the device
inside `Engine`, next to the GGUF and tokenizer code, so there is nothing
there to depend on without pulling in a language-model stack. The copy here is
kept as its own crate and synced by hand with what llm-web main does in the
same places: the minimum compute-limit check at device acquisition
(`limits::require_compute`, called by `engine`), uploads through
`queue.write_buffer` only, and the `mtBuilt` flag in `js/capabilities.js`
(a page tells the ladder whether it shipped a threads build).

- `src/`: the Rust crate.
- `js/capabilities.js`: the same ladder rule for a browser page or worker.
- `js/boot.js`: loads a wasm-bindgen build once per worker, with a build tag
  on every URL so a rebuilt module is never served from an older cache.
- `scripts/`: wasm output checks (`leak_check.sh`), the rayon helper patch,
  build-tag and served-bytes checks, and a local server that sets the headers
  threads need (`serve_coi.py`).
