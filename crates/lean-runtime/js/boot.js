// Loads a wasm-bindgen build (--target web) in a worker, once per worker.
//
//   const mod = await loadBuild({ glue: "../lean/pkg/lean.js", wasm: "../lean/pkg/lean_bg.wasm", tag });
//   const mod = await loadBuild({ glue, wasm, tag, threads: navigator.hardwareConcurrency });
//
// Relative URLs resolve against the worker script. Every URL gets `?v=<tag>`,
// the build tag the page loads this build with, so a rebuilt module is never
// served from an older build's cache. `threads` (a pool size, threads builds
// only) builds the wasm-bindgen-rayon pool after init; `init(mod)` runs once
// after that (a panic hook, say). Later calls for the same glue URL and tag
// return the first call's module: a threads pool can be built only once per
// worker, so a second run on the same build must not initialise it again.
// `timing`, if given, gets `wasmInitMs` and `threadPoolMs` from the call
// that actually loaded the build.

const builds = new Map();

function tagged(url, tag) {
  const u = new URL(url, self.location.href);
  u.searchParams.set("v", tag);
  return u.href;
}

export function loadBuild({ glue, wasm, tag, threads = 0, init = null, timing = null }) {
  const key = tagged(glue, tag);
  if (!builds.has(key)) {
    builds.set(
      key,
      (async () => {
        let t0 = performance.now();
        const mod = await import(key);
        await mod.default(tagged(wasm, tag));
        if (timing) timing.wasmInitMs = performance.now() - t0;
        if (threads) {
          t0 = performance.now();
          await (mod.initThreadPoolOnce || mod.initThreadPool)(threads);
          if (timing) timing.threadPoolMs = performance.now() - t0;
        }
        if (init) init(mod);
        return mod;
      })()
    );
  }
  return builds.get(key);
}
