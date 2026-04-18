#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# fetch_assets.sh
#
# Fetches the six cd-rules design assets at the commit recorded in
# assets/SOURCE.md and places them into assets/. Run tools/hash_assets.sh
# afterwards to verify integrity against the recorded SHA-256 table.
#
# Usage:        tools/fetch_assets.sh [--force]
# Context:      iontrap-dynamics repository root
# Requires:     curl, grep
#
# Exit codes:
#   0  success (including "skipped because destination exists")
#   1  SOURCE.md missing or malformed
#   2  network fetch failed
#
# Behaviour:
#   - By default, skips files already present in assets/ (prints a note, exits 0).
#     Existing files are treated as intentional and left untouched so the script
#     is safe to re-run after a partial fetch.
#   - With --force, overwrites existing files.
#   - Reads the pinned commit SHA from assets/SOURCE.md (same contract as
#     hash_assets.sh), so bumping the commit is a single-point edit.

set -euo pipefail

SOURCE_MD="assets/SOURCE.md"
REPO_SLUG="threehouse-plus-ec/cd-rules"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

if [[ ! -f "$SOURCE_MD" ]]; then
    echo "ERROR: $SOURCE_MD not found. Run from repository root." >&2
    exit 1
fi

COMMIT_SHA=$(grep -oE '[a-f0-9]{40}' "$SOURCE_MD" | head -n 1 || true)
if [[ -z "$COMMIT_SHA" ]]; then
    echo "ERROR: no 40-character commit SHA found in $SOURCE_MD" >&2
    exit 1
fi
echo "Pinned commit: $COMMIT_SHA"

FILES=(
    "emblem-16.svg"
    "emblem-32.svg"
    "emblem-64.svg"
    "tokens.css"
    "wordmark-full.svg"
    "wordmark-silent.svg"
)

mkdir -p assets
# Note: docs/stylesheets/ is created only if we actually mirror tokens.css
# into it (see loop below). Creating it unconditionally would leave an
# empty directory on repos that haven't set up a docs site yet.

echo "Fetching from ${REPO_SLUG}..."
for f in "${FILES[@]}"; do
    dest="assets/${f}"
    if [[ -f "$dest" && $FORCE -eq 0 ]]; then
        echo "  $f: already present, skipping (use --force to overwrite)"
    else
        url="https://raw.githubusercontent.com/${REPO_SLUG}/${COMMIT_SHA}/${f}"
        if ! curl -sSfL --max-redirs 5 -o "$dest" "$url"; then
            echo "  $f: FETCH FAILED from $url" >&2
            exit 2
        fi
        echo "  $f: fetched"
    fi

    # tokens.css is also consumed by the docs site (workplan §0.H step 4).
    # Maintain a second vendored copy under docs/stylesheets/ so mkdocs,
    # which only serves files from docs_dir, can load it via extra_css.
    # The hash check in tools/hash_assets.sh operates on assets/ only; if
    # the docs copy drifts from the assets copy, fetch_assets.sh (or a
    # small CI step) will restore it at its next run.
    # Only create docs/stylesheets/ when we actually write tokens.css
    # — avoids leaving an empty directory on pre-docs-site repositories.
    if [[ "$f" == "tokens.css" ]]; then
        mkdir -p docs/stylesheets
        cp -f "assets/$f" "docs/stylesheets/$f"
        echo "    -> mirrored to docs/stylesheets/$f for mkdocs consumption"
    fi
done

echo ""
echo "SUCCESS: assets placed in assets/ (tokens.css also mirrored to docs/stylesheets/)."
echo ""
echo "Next: run tools/hash_assets.sh to verify integrity against SOURCE.md,"
echo "      then review each file visually before committing."
