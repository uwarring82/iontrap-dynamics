#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# hash_assets.sh
#
# Finalises assets/SOURCE.md by fetching the six cd-rules design assets at the
# commit recorded in SOURCE.md, computing SHA-256 and byte sizes, and rewriting
# the "Files and checksums" table in place.
#
# Usage:   tools/hash_assets.sh
# Context: iontrap-dynamics repository root
# Requires: curl, sha256sum (coreutils) or shasum (macOS)
#
# Exit codes:
#   0  success
#   1  SOURCE.md missing or malformed
#   2  network fetch failed
#   3  required commit SHA not found in SOURCE.md

set -euo pipefail

SOURCE_MD="assets/SOURCE.md"
REPO_SLUG="threehouse-plus-ec/cd-rules"

if [[ ! -f "$SOURCE_MD" ]]; then
    echo "ERROR: $SOURCE_MD not found. Run from repository root." >&2
    exit 1
fi

# Extract the resolved commit SHA from SOURCE.md
COMMIT_SHA=$(grep -oE '[a-f0-9]{40}' "$SOURCE_MD" | head -n 1 || true)
if [[ -z "$COMMIT_SHA" ]]; then
    echo "ERROR: no 40-character commit SHA found in $SOURCE_MD" >&2
    exit 3
fi
echo "Resolved commit: $COMMIT_SHA"

# Pick sha256 tool
if command -v sha256sum >/dev/null 2>&1; then
    SHA_CMD="sha256sum"
    SHA_FIELD=1
elif command -v shasum >/dev/null 2>&1; then
    SHA_CMD="shasum -a 256"
    SHA_FIELD=1
else
    echo "ERROR: neither sha256sum nor shasum available" >&2
    exit 1
fi

# Canonical file list — must match workplan 0.H step 2
FILES=(
    "emblem-16.svg"
    "emblem-32.svg"
    "emblem-64.svg"
    "tokens.css"
    "wordmark-full.svg"
    "wordmark-silent.svg"
)

# Working directory
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "Fetching assets at commit $COMMIT_SHA..."
declare -A HASHES
declare -A SIZES
for f in "${FILES[@]}"; do
    url="https://raw.githubusercontent.com/${REPO_SLUG}/${COMMIT_SHA}/${f}"
    out="$WORK/$f"
    if ! curl -sSfL --max-redirs 5 -o "$out" "$url"; then
        echo "  $f: FETCH FAILED from $url" >&2
        exit 2
    fi
    size=$(wc -c < "$out" | tr -d ' ')
    hash=$($SHA_CMD "$out" | awk '{print $1}')
    HASHES[$f]="$hash"
    SIZES[$f]="$size"
    echo "  $f: $size bytes, $hash"
done

# Rewrite the "Files and checksums" table in SOURCE.md.
# Strategy: generate the new table as a temp file, then replace the region
# between the table header and the next blank-line-terminated section.

NEW_TABLE=$(mktemp)
cat > "$NEW_TABLE" <<EOF
| File | SHA-256 | Size (bytes) |
|---|---|---|
EOF
for f in "${FILES[@]}"; do
    printf '| `%s` | `%s` | %s |\n' "$f" "${HASHES[$f]}" "${SIZES[$f]}" >> "$NEW_TABLE"
done

# Use python for the in-place substitution (more reliable than sed across platforms)
python3 <<PYEOF
import re, sys
path = "$SOURCE_MD"
with open(path) as fh:
    content = fh.read()

# Match from the table header "| File | SHA-256 | Size (bytes) |" through the
# end of the last row.
pattern = re.compile(
    r"\| File \| SHA-256 \| Size \(bytes\) \|.*?(?=\n\nUpstream path)",
    re.DOTALL,
)

with open("$NEW_TABLE") as fh:
    new_table = fh.read().rstrip("\n")

new_content, count = pattern.subn(new_table, content, count=1)
if count != 1:
    sys.stderr.write("ERROR: could not locate checksum table in SOURCE.md\n")
    sys.exit(1)

# This script only rewrites the checksum table. Other governance fields
# (Source tag, Source commit, Status banner) are manual edits.
with open(path, "w") as fh:
    fh.write(new_content)
print(f"Updated {path}")
PYEOF

echo ""
echo "SUCCESS: SOURCE.md checksum table populated."
echo ""
echo "Next steps:"
echo "  1. Review the updated $SOURCE_MD"
echo "  2. Verify local copies in assets/ match these hashes:"
echo "     for f in ${FILES[*]}; do $SHA_CMD \"assets/\$f\"; done"
echo "  3. When satisfied, commit the Source commit / tag field updates"
echo "     alongside the regenerated checksum table."
