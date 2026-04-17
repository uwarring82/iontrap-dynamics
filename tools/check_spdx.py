# SPDX-License-Identifier: MIT
"""Enforce SPDX-License-Identifier headers on source files.

Runs as a pre-commit hook. Given a list of file paths as CLI arguments,
exits non-zero if any file lacks an `SPDX-License-Identifier:` line within
its first 10 physical lines (enough to tolerate shebangs and short file-
level docstrings but not so generous that a header buried mid-file counts).

Covers Python source, shell scripts, and GitHub Actions workflow YAML.
Markdown, TOML, and JSON are excluded at the pre-commit `files:` filter.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "SPDX-License-Identifier:"
HEAD_LINES = 10


def check(path: Path) -> str | None:
    """Return an error message if `path` lacks the SPDX marker, else None."""
    try:
        with path.open(encoding="utf-8") as fh:
            head = [next(fh, "") for _ in range(HEAD_LINES)]
    except OSError as exc:
        return f"{path}: could not read ({exc})"
    if not any(MARKER in line for line in head):
        return f"{path}: missing '{MARKER} <SPDX>' in first {HEAD_LINES} lines"
    return None


def main(argv: list[str]) -> int:
    errors = [msg for msg in (check(Path(p)) for p in argv[1:]) if msg]
    for msg in errors:
        print(msg, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
