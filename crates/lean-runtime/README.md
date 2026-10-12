# lean-runtime

The wgpu and wasm plumbing an engine needs before its own kernels: the
backend ladder (WebGPU, then CPU threads, then one CPU thread, chosen by
capability only), device acquisition with the adapter's own limits, buffers
and awaited readback, pipelines, size checks against the device limits, a
clock, and the wasm threads pool. No model or physics types.

This crate was derived from the runtime work in
[idle-intelligence/llm-web](https://github.com/idle-intelligence/llm-web) and
is synced with the device handling of llm-web main at commit 6daf322: the
minimum compute-limit check at device acquisition (`limits::require_compute`),
uploads through `queue.write_buffer` only, and the `mtBuilt` flag in
`js/capabilities.js`.

- `src/`: the Rust crate.
- `js/capabilities.js`: the same ladder rule for a browser page or worker.
- `js/boot.js`: loads a wasm-bindgen build once per worker, with a build tag
  on every URL so a rebuilt module is never served from an older cache.
- `scripts/`: wasm output checks (`leak_check.sh`), the rayon helper patch,
  build-tag and served-bytes checks, and a local server that sets the headers
  threads need (`serve_coi.py`).
