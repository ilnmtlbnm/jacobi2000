#!/usr/bin/env bash
# Checks that every build tag in the given files is <tag>: each
# `ENGINE_BUILD = "..."` (or '...') constant, and each literal `?v=...` in a
# URL (template URLs like `?v=${ENGINE_BUILD}` take the constant's value).
# Directories are searched for .js and .html files. Fails on any other tag,
# or if no ENGINE_BUILD constant is found at all.
#
# Usage: check_build_tag.sh <tag> <file-or-dir>...
set -euo pipefail
TAG=$1; shift
FILES=$(find "$@" -type f \( -name '*.js' -o -name '*.html' \))
[ -n "$FILES" ] || { echo "error: no .js or .html files under $*" >&2; exit 1; }

# shellcheck disable=SC2086
CONSTS=$(grep -HoE "ENGINE_BUILD = [\"'][^\"']*[\"']" $FILES || true)
[ -n "$CONSTS" ] || { echo "error: no ENGINE_BUILD constant in $*" >&2; exit 1; }
BAD_CONSTS=$(printf '%s\n' "$CONSTS" | grep -vE "ENGINE_BUILD = [\"']$TAG[\"']$" || true)
# shellcheck disable=SC2086
BAD_URLS=$(grep -HoE "\?v=[A-Za-z0-9._-]+" $FILES | grep -vE ":\?v=$TAG$" || true)
if [ -n "$BAD_CONSTS$BAD_URLS" ]; then
    echo "error: build tags other than $TAG:" >&2
    printf '%s\n%s\n' "$BAD_CONSTS" "$BAD_URLS" | grep -v '^$' >&2
    exit 1
fi
echo "==> build tag $TAG: $(printf '%s\n' "$CONSTS" | wc -l | tr -d ' ') constants, no other tag"
