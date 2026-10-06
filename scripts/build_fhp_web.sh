#!/usr/bin/env bash
# Build the fhp engine for the browser into crates/fhp/pkg (WebGPU and the
# single-thread bit-sliced CPU path, SIMD128) and crates/fhp/pkg-mt (the
# threaded CPU path: nightly build-std, atomics, wasm-bindgen-rayon), for
# the dev page crates/fhp/www/index.html. Same recipe as llm-web's
# scripts/build_lean.sh and scripts/build_lean_mt.sh.
#
# Requires wasm-pack, a nightly toolchain with rust-src, and wasm-bindgen-cli
# at the version pinned in crates/fhp/Cargo.toml (0.2.128).
#
# Usage: ENGINE_BUILD=<tag> scripts/build_fhp_web.sh
# ENGINE_BUILD is the `?v=` tag the page puts on every loading URL (worker,
# glue, wasm, rayon helpers); the script fails unless the page already
# carries it, so a rebuild always comes with a tag bump.
set -euo pipefail

: "${ENGINE_BUILD:?set ENGINE_BUILD to the ?v= tag crates/fhp/www/index.html loads this build with}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CRATE=crates/fhp

if ! grep -qF "const ENGINE_BUILD = '$ENGINE_BUILD';" $CRATE/www/index.html; then
    echo "error: $CRATE/www/index.html does not load ENGINE_BUILD=$ENGINE_BUILD; bump its ENGINE_BUILD constant first" >&2
    exit 1
fi

# Compiled-in source paths would carry the home directory; map it away.
REMAP="--remap-path-prefix=$HOME=/home"

leak_check() {
    local wasm=$1
    LEAKS=$(strings "$wasm" | grep -F -e "$HOME" -e "Code/" -e ".claude/" -e "/Users/" || true)
    USER_HITS=$(strings "$wasm" | grep -Fw -e "$(id -un)" || true)
    if [ -n "$LEAKS$USER_HITS" ]; then
        echo "error: $wasm contains local paths or the user name:" >&2
        printf '%s\n%s\n' "$LEAKS" "$USER_HITS" | grep -v '^$' | head -20 >&2
        exit 1
    fi
}

echo "==> pkg: WebGPU + single-thread CPU"
RUSTFLAGS="-C target-feature=+simd128 $REMAP" \
    wasm-pack build $CRATE --target web --release --out-dir "$REPO_ROOT/$CRATE/pkg" --no-default-features --features web
leak_check $CRATE/pkg/fhp_bg.wasm

echo "==> pkg-mt: threaded CPU (nightly build-std)"
# 2.5 GiB: the 2 GiB CPU lattice bound (WASM_CPU_MAX_BYTES) plus headroom.
MAX_MEMORY="${MAX_MEMORY:-2684354560}"
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
-C link-arg=--shared-memory -C link-arg=--max-memory=$MAX_MEMORY \
-C link-arg=--import-memory \
-C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align -C link-arg=--export=__tls_base $REMAP" \
  cargo +nightly build -p fhp --lib \
    --target wasm32-unknown-unknown \
    --release \
    --no-default-features --features wasm-mt \
    -Z build-std=panic_abort,std
OUT_DIR=$CRATE/pkg-mt
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
wasm-bindgen --target web --out-dir "$OUT_DIR" --out-name fhp target/wasm32-unknown-unknown/release/fhp.wasm

# wasm-bindgen-rayon's workerHelpers.js imports the main module as
# '../../..', which a plain browser import cannot resolve; point it at
# fhp.js, and put the build tag on every URL in the chain.
HELPER=$(find "$OUT_DIR/snippets" -name workerHelpers.js | head -1)
[ -n "$HELPER" ] || { echo "error: workerHelpers.js not found under $OUT_DIR/snippets" >&2; exit 1; }
sed -i.bak \
    -e "s#await import('\.\./\.\./\.\.')#await import('../../../fhp.js?v=$ENGINE_BUILD')#" \
    -e "s#new URL('\./workerHelpers\.js', import\.meta\.url)#new URL('./workerHelpers.js?v=$ENGINE_BUILD', import.meta.url)#" \
    "$HELPER"
sed -i.bak "s#\(from '\./snippets/[^']*/workerHelpers\.js\)'#\1?v=$ENGINE_BUILD'#" "$OUT_DIR/fhp.js"
rm -f "$HELPER.bak" "$OUT_DIR/fhp.js.bak"
for want in "fhp.js?v=$ENGINE_BUILD" "workerHelpers.js?v=$ENGINE_BUILD"; do
    grep -qF "$want" "$HELPER" || { echo "error: $HELPER lacks $want" >&2; exit 1; }
done
grep -qF "workerHelpers.js?v=$ENGINE_BUILD'" "$OUT_DIR/fhp.js" || { echo "error: fhp.js does not import workerHelpers.js?v=$ENGINE_BUILD" >&2; exit 1; }
leak_check $OUT_DIR/fhp_bg.wasm

for d in pkg pkg-mt; do
    echo "$ENGINE_BUILD" > $CRATE/$d/BUILD
    echo "==> $CRATE/$d: $(wc -c < $CRATE/$d/fhp_bg.wasm | tr -d ' ') bytes, sha256 $(shasum -a 256 $CRATE/$d/fhp_bg.wasm | cut -c1-16)"
done
