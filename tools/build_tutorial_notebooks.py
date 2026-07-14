#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# =============================================================================
# tools/build_tutorial_notebooks.py
#
# Convert the Markdown tutorials under ``docs/tutorials/`` into Google Colab
# executable notebooks under ``docs/tutorials/notebooks/``.
#
# The Markdown tutorials are the SINGLE SOURCE OF TRUTH. The generated
# notebooks are a derived artifact: each ``` ```python ``` block becomes a
# runnable code cell, the surrounding prose becomes Markdown cells, and a
# Colab "Setup" cell that ``pip install``s the package is prepended. Because
# the tutorials are written as linear scripts (later steps reuse names bound
# in earlier steps), the cells run top-to-bottom in a shared kernel and the
# embedded ``assert`` statements become the notebook's built-in self-test.
#
# Dependency-free on purpose (no nbformat): emits nbformat 4.5 JSON directly,
# so it runs anywhere the repo's own toolchain runs.
#
# Usage
# -----
#   python tools/build_tutorial_notebooks.py                # build all tutorials
#   python tools/build_tutorial_notebooks.py docs/tutorials/18_*.md   # subset
#   python tools/build_tutorial_notebooks.py --check        # CI: fail on drift
#
# The ``--check`` mode regenerates in memory and diffs against the committed
# notebooks without writing — a freshness guard for CI, in the spirit of the
# repo's other "regenerate and compare" oracle checks.
# =============================================================================
from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TUTORIALS_DIR = REPO_ROOT / "docs" / "tutorials"
NOTEBOOKS_DIR = TUTORIALS_DIR / "notebooks"

# Repository coordinates. ``OPEN_REF`` is the branch the *notebook file* lives
# on (what the Colab badge opens). ``INSTALL_REF`` is the git ref the Setup cell
# installs the *package* from.
#
# INSTALL_REF tracks ``main`` (not a release tag) so the live Colab notebooks
# always pick up the newest fixes — a tag lags behind every post-tag fix, which
# silently served stale code on Colab. ``main`` is guarded by the ``tutorials``
# (execute) and ``wheel-smoke`` (packaging) CI jobs. To return to reproducible
# tag-pinning after cutting a release, set ``INSTALL_REF = "v0.6.1"`` (etc.) and
# regenerate *after* the tag is pushed, or the git install URL will not resolve.
GITHUB_SLUG = "uwarring82/iontrap-dynamics"
OPEN_REF = "main"
INSTALL_REF = "main"
SITE_URL = "https://uwarring82.github.io/iontrap-dynamics/"

# Only note/tip/warning appear in the tutorials today; any other kind falls
# back to a plain bold label (see ``transform_admonitions``).
ADMONITION_LABELS = {
    "note": "📝 **Note**",
    "tip": "💡 **Tip**",
    "warning": "⚠️ **Warning**",
    "example": "🔬 **Example**",
}


# -----------------------------------------------------------------------------
# Markdown → cell segmentation
# -----------------------------------------------------------------------------
_FENCE = re.compile(r"^(\s*)(`{3,})\s*([^\s`]*)\s*$")


def segment(markdown: str) -> list[tuple[str, str]]:
    """Split a tutorial into ordered ``("md"|"code", text)`` segments.

    ``` ```python ``` (or ``` ```py ```) fences become ``code`` segments; every
    other fence is copied verbatim into the surrounding ``md`` segment (so a
    ``` ```json ``` block still renders as a fenced block in the notebook).
    """
    lines = markdown.split("\n")
    segments: list[tuple[str, str]] = []
    md_buf: list[str] = []

    def flush_md() -> None:
        if md_buf:
            segments.append(("md", "\n".join(md_buf)))
            md_buf.clear()

    i, n = 0, len(lines)
    while i < n:
        m = _FENCE.match(lines[i])
        if m:
            fence, lang = m.group(2), m.group(3).lower()
            close = re.compile(r"^\s*" + re.escape(fence) + r"\s*$")
            if lang in ("python", "py"):
                flush_md()
                i += 1
                code: list[str] = []
                while i < n and not close.match(lines[i]):
                    code.append(lines[i])
                    i += 1
                i += 1  # consume closing fence
                segments.append(("code", "\n".join(code)))
                continue
            # Non-python fence: keep verbatim inside the markdown segment.
            md_buf.append(lines[i])
            i += 1
            while i < n and not close.match(lines[i]):
                md_buf.append(lines[i])
                i += 1
            if i < n:
                md_buf.append(lines[i])
                i += 1
            continue
        md_buf.append(lines[i])
        i += 1
    flush_md()
    return segments


# -----------------------------------------------------------------------------
# Markdown transforms (applied to "md" segments only)
# -----------------------------------------------------------------------------
def transform_admonitions(text: str) -> str:
    """Rewrite mkdocs-material ``!!!`` / ``???`` / ``???+`` blocks as blockquotes.

    Colab has no admonition support; a blockquote with a bold label is the
    portable equivalent. Collapsible blocks (``???`` collapsed, ``???+`` open —
    ``pymdownx.details``) degrade to the same static blockquote as the always-open
    ``!!!`` form, since a notebook cell has no collapse affordance. Indented
    (4-space) admonition bodies are dedented.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = re.match(r'^(?:!!!|\?\?\?\+?)\s+(\w+)(?:\s+"([^"]*)")?\s*$', lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        kind, title = m.group(1).lower(), m.group(2)
        label = ADMONITION_LABELS.get(kind, f"**{kind.capitalize()}**")
        head = label + (f" — {title}" if title else "")
        i += 1
        body: list[str] = []
        while i < len(lines) and (
            lines[i].strip() == "" or lines[i].startswith("    ") or lines[i].startswith("\t")
        ):
            body.append("" if lines[i].strip() == "" else re.sub(r"^(    |\t)", "", lines[i]))
            i += 1
        while body and body[-1] == "":
            body.pop()
        out.append("> " + head)
        out.append(">")
        out.extend("> " + b if b else ">" for b in body)
        out.append("")
    return "\n".join(out)


def rewrite_links(text: str) -> str:
    """Point relative ``*.md`` links at the published docs site.

    A relative ``08_full_lamb_dicke.md`` (resolved against ``docs/tutorials/``)
    becomes ``{SITE_URL}tutorials/08_full_lamb_dicke/`` — the directory-URL form
    mkdocs-material serves by default. Absolute URLs and non-``.md`` targets
    (e.g. the ``.png`` figure) are left untouched.
    """

    def repl(m: re.Match[str]) -> str:
        label, target = m.group(1), m.group(2)
        path, _, anchor = target.partition("#")
        if "://" in path or not path.endswith(".md"):
            return m.group(0)
        resolved = posixpath.normpath(posixpath.join("docs/tutorials", path))
        if resolved.startswith("docs/"):
            resolved = resolved[len("docs/") :]
        url = SITE_URL + resolved[:-3] + "/"
        if anchor:
            url += "#" + anchor
        return f"[{label}]({url})"

    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", repl, text)


def strip_badge(text: str) -> str:
    """Drop any pre-existing Colab badge line (the notebook adds its own)."""
    return "\n".join(ln for ln in text.split("\n") if "colab.research.google.com" not in ln)


def clean_md(text: str) -> str:
    return rewrite_links(transform_admonitions(strip_badge(text))).strip("\n")


# -----------------------------------------------------------------------------
# Notebook assembly (nbformat 4.5)
# -----------------------------------------------------------------------------
def _source(text: str) -> list[str]:
    """nbformat ``source``: a list of lines, each newline-terminated but the last."""
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]] if parts else []


def build_notebook(md_path: Path) -> str:
    stem = md_path.stem
    nb_rel = f"docs/tutorials/notebooks/{stem}.ipynb"
    md_rel = f"docs/tutorials/{stem}.md"
    colab_url = f"https://colab.research.google.com/github/{GITHUB_SLUG}/blob/{OPEN_REF}/{nb_rel}"
    md_url = f"https://github.com/{GITHUB_SLUG}/blob/{OPEN_REF}/{md_rel}"

    header = (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({colab_url})\n\n"
        "**Run the _Setup_ cell below first**, then run the remaining cells "
        "top-to-bottom. The `assert` statements are the tutorial's built-in "
        "checks — if every cell runs without error, every step passed.\n\n"
        f"> _Auto-generated from [`{md_rel}`]({md_url}) by "
        "`tools/build_tutorial_notebooks.py`. Edit the Markdown tutorial, not "
        "this notebook._"
    )
    setup = (
        "# Setup — install iontrap-dynamics and its dependencies (qutip, numpy, scipy).\n"
        "# First run on Colab takes ~1-2 min; safe to re-run (a no-op once installed).\n"
        f'%pip install -q "iontrap-dynamics[plot] @ '
        f'git+https://github.com/{GITHUB_SLUG}.git@{INSTALL_REF}"'
    )

    cells: list[dict[str, object]] = [_md_cell(header), _code_cell(setup)]
    for kind, text in segment(md_path.read_text(encoding="utf-8")):
        if kind == "code":
            if text.strip():
                cells.append(_code_cell(text.strip("\n")))
        else:
            cleaned = clean_md(text)
            if cleaned:
                cells.append(_md_cell(cleaned))

    for idx, cell in enumerate(cells):
        cell["id"] = f"cell-{idx}"

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def _md_cell(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def _code_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _source(text),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path, help="Tutorial .md files (default: all numbered tutorials)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify committed notebooks are up to date; do not write",
    )
    args = parser.parse_args(argv)

    if args.paths:
        md_files = sorted(p if p.is_absolute() else REPO_ROOT / p for p in args.paths)
    elif args.check:
        # CI freshness guard: check exactly the notebooks that are committed —
        # a tutorial without a notebook is simply not covered yet, not a
        # failure. New notebooks come under the guard as soon as they land.
        md_files = sorted(TUTORIALS_DIR / f"{nb.stem}.md" for nb in NOTEBOOKS_DIR.glob("*.ipynb"))
    else:
        md_files = sorted(TUTORIALS_DIR.glob("[0-9]*.md"))

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    drifted: list[str] = []
    for md_path in md_files:
        nb_path = NOTEBOOKS_DIR / f"{md_path.stem}.ipynb"
        rel = nb_path.relative_to(REPO_ROOT)
        if not md_path.exists():
            drifted.append(str(rel))
            print(f"ORPHAN {rel} — no source {md_path.relative_to(REPO_ROOT)}")
            continue
        content = build_notebook(md_path)
        if args.check:
            current = nb_path.read_text(encoding="utf-8") if nb_path.exists() else None
            if current != content:
                drifted.append(str(rel))
                print(f"DRIFT  {rel}")
            else:
                print(f"ok     {rel}")
        else:
            nb_path.write_text(content, encoding="utf-8")
            print(f"wrote  {rel}")

    if args.check and drifted:
        print(
            f"\n{len(drifted)} notebook(s) out of date. "
            "Run `python tools/build_tutorial_notebooks.py` and commit.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
