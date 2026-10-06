#!/usr/bin/env bash
# Checks that a server returns the local bytes for each file at the build
# tag's URL: fetches <base>/<path>?v=<tag> and compares its sha256 with the
# local file's. Run it against the server a browser test is about to use,
# before trusting a result from a rebuilt wasm.
#
# Usage: check_served.sh <base-url> <tag> <local-file>=<served-path>...
#   e.g. check_served.sh http://127.0.0.1:8030/web 2026-10-06-x \
#          crates/lean/pkg/lean_bg.wasm=lean/pkg/lean_bg.wasm
set -euo pipefail
BASE=${1%/} TAG=$2; shift 2
FAILED=0
for pair in "$@"; do
    LOCAL=${pair%%=*} SERVED=${pair#*=}
    WANT=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
    GOT=$(curl -fsS "$BASE/$SERVED?v=$TAG" | shasum -a 256 | cut -d' ' -f1)
    if [ "$WANT" = "$GOT" ]; then
        echo "ok   $SERVED ${WANT:0:16}"
    else
        echo "DIFF $SERVED local ${WANT:0:16} served ${GOT:0:16}" >&2
        FAILED=1
    fi
done
exit $FAILED
