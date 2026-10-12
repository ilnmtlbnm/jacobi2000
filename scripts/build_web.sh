#!/usr/bin/env bash
# Build the demo page's wasm (crates/web: the solver and the engine's WebGPU
# and CPU paths) into web/pkg, then check the output: no local paths or user
# name inside, and 128-bit SIMD compiled in.
#
# Requires wasm-pack and the wasm32-unknown-unknown target
# (rust-toolchain.toml installs it with rustup).
#
# Usage: scripts/build_web.sh
# The page loads web/pkg/web.js; if you serve it from a cache that outlives a
# rebuild, put a new ?v=<tag> on that URL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Compiled-in source paths would carry the home directory; map it away. The
# flags go in with --config, which merges with .cargo/config.toml
# (target-feature=+simd128) instead of replacing it the way RUSTFLAGS would.
CARGO_HOME_PATH="${CARGO_HOME:-$HOME/.cargo}"
REMAP="target.'cfg(all())'.rustflags=[\"--remap-path-prefix=$HOME=/home\", \"--remap-path-prefix=$CARGO_HOME_PATH=/cargo\"]"

wasm-pack build crates/web --target web --release --out-dir "$REPO_ROOT/web/pkg" --config "$REMAP"

WASM=web/pkg/web_bg.wasm
bash crates/lean-runtime/scripts/leak_check.sh "$WASM"
python3 crates/lean-runtime/scripts/check_wasm_features.py "$WASM" +simd128

echo "==> web/pkg: $(wc -c < "$WASM" | tr -d ' ') bytes, sha256 $(shasum -a 256 "$WASM" | cut -c1-16)"
