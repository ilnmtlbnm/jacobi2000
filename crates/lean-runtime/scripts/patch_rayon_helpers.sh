#!/usr/bin/env bash
# Rewrites a threads build's wasm-bindgen-rayon glue for a plain browser
# import (wasm-bindgen --target web, no bundler) and puts the build tag on
# every URL in the chain: <name>.js imports workerHelpers.js, which spawns
# its own worker script, which imports <name>.js. workerHelpers.js's
# bundler-relative import('../../..') does not resolve in a browser; it is
# pointed at <name>.js. Fails if a rewrite did not apply.
#
# Usage: patch_rayon_helpers.sh <out_dir> <name> <tag>
#   out_dir: wasm-bindgen's --out-dir; name: its --out-name; tag: ENGINE_BUILD.
set -euo pipefail
OUT_DIR=$1 NAME=$2 TAG=$3

HELPER=$(find "$OUT_DIR/snippets" -name workerHelpers.js 2>/dev/null | head -1)
if [ -z "$HELPER" ]; then
    echo "error: workerHelpers.js not found under $OUT_DIR/snippets" >&2
    exit 1
fi
sed -i.bak \
    -e "s#await import('\.\./\.\./\.\.')#await import('../../../$NAME.js?v=$TAG')#" \
    -e "s#new URL('\./workerHelpers\.js', import\.meta\.url)#new URL('./workerHelpers.js?v=$TAG', import.meta.url)#" \
    "$HELPER"
sed -i.bak "s#\(from '\./snippets/[^']*/workerHelpers\.js\)'#\1?v=$TAG'#" "$OUT_DIR/$NAME.js"
rm -f "$HELPER.bak" "$OUT_DIR/$NAME.js.bak"
for want in "$NAME.js?v=$TAG" "workerHelpers.js?v=$TAG"; do
    grep -qF "$want" "$HELPER" || { echo "error: $HELPER lacks $want" >&2; exit 1; }
done
grep -qF "workerHelpers.js?v=$TAG'" "$OUT_DIR/$NAME.js" || { echo "error: $NAME.js does not import workerHelpers.js?v=$TAG" >&2; exit 1; }
echo "==> Patched $HELPER and $NAME.js: imports carry ?v=$TAG"
