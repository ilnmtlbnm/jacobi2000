#!/usr/bin/env bash
# Build the jacobi engine for the browser into crates/jacobi/pkg (WebGPU and
# the single-thread bit-sliced CPU engines, SIMD128), for the dev page
# crates/jacobi/www/index.html. Same recipe as the fhp crate's build script.
#
# Usage: ENGINE_BUILD=<tag> scripts/build_jacobi_web.sh
# ENGINE_BUILD is the `?v=` tag the page puts on every loading URL (worker,
# glue, wasm); the script fails unless the page already carries it, so a
# rebuild always comes with a tag bump.
set -euo pipefail

: "${ENGINE_BUILD:?set ENGINE_BUILD to the ?v= tag crates/jacobi/www/index.html loads this build with}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CRATE=crates/jacobi

if ! grep -qF "const ENGINE_BUILD = '$ENGINE_BUILD';" $CRATE/www/index.html; then
    echo "error: $CRATE/www/index.html does not load ENGINE_BUILD=$ENGINE_BUILD; bump its ENGINE_BUILD constant first" >&2
    exit 1
fi

# Compiled-in source paths would carry the home directory; map it away.
REMAP="--remap-path-prefix=$HOME=/home"

RUSTFLAGS="-C target-feature=+simd128 $REMAP" \
    wasm-pack build $CRATE --target web --release --out-dir "$REPO_ROOT/$CRATE/pkg" --no-default-features --features web

WASM=$CRATE/pkg/jacobi_bg.wasm
LEAKS=$(strings "$WASM" | grep -F -e "$HOME" -e "Code/" -e ".claude/" -e "/Users/" || true)
USER_HITS=$(strings "$WASM" | grep -Fw -e "$(id -un)" || true)
if [ -n "$LEAKS$USER_HITS" ]; then
    echo "error: $WASM contains local paths or the user name:" >&2
    printf '%s\n%s\n' "$LEAKS" "$USER_HITS" | grep -v '^$' | head -20 >&2
    exit 1
fi

echo "$ENGINE_BUILD" > $CRATE/pkg/BUILD
echo "==> $CRATE/pkg: $(wc -c < $WASM | tr -d ' ') bytes, sha256 $(shasum -a 256 $WASM | cut -c1-16)"
