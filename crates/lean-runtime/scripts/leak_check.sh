#!/usr/bin/env bash
# Fails if a built wasm file carries local paths or the user name.
# Usage: leak_check.sh <file.wasm>...
set -euo pipefail
for wasm in "$@"; do
    LEAKS=$(strings "$wasm" | grep -F -e "$HOME" -e "Code/" -e ".claude/" -e "/Users/" || true)
    USER_HITS=$(strings "$wasm" | grep -Fw -e "$(id -un)" || true)
    if [ -n "$LEAKS$USER_HITS" ]; then
        echo "error: $wasm contains local paths or the user name:" >&2
        printf '%s\n%s\n' "$LEAKS" "$USER_HITS" | grep -v '^$' | head -20 >&2
        exit 1
    fi
done
