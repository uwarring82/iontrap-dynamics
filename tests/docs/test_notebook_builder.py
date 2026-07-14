# SPDX-License-Identifier: MIT
"""Unit tests for ``tools/build_tutorial_notebooks.py`` — the Markdown → Colab converter.

The tutorials are the single source of truth and the ``--check`` freshness guard
pins ``notebook == regeneration``, but it never exercises the transforms in
isolation. This module pins the non-obvious one — admonition rewriting — including
the ``???`` / ``???+`` collapsibles added under WP-06 dispatch TA7a, so a builder
change cannot silently corrupt what a reader runs on Colab.

``tools/`` is not on the pytest ``pythonpath`` (only ``src`` and ``tests`` are), so
the module is loaded by path via ``importlib``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_BUILDER_PATH = Path(__file__).resolve().parents[2] / "tools" / "build_tutorial_notebooks.py"
_spec = importlib.util.spec_from_file_location("build_tutorial_notebooks", _BUILDER_PATH)
assert _spec is not None and _spec.loader is not None
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def test_bang_admonition_becomes_labelled_blockquote() -> None:
    md = '!!! note "Why it matters"\n    body line one\n    body line two\n'
    out = build.transform_admonitions(md)
    assert "> 📝 **Note** — Why it matters" in out
    assert "> body line one" in out
    assert "> body line two" in out
    assert "!!!" not in out


def test_collapsible_question_admonition_becomes_blockquote() -> None:
    # ??? (collapsed) and ???+ (open) both degrade to the same static blockquote —
    # a Colab cell has no collapse affordance (WP-06 TA7a).
    for marker in ("???", "???+"):
        md = f'{marker} tip "Setup"\n    install the thing\n    then run it\n'
        out = build.transform_admonitions(md)
        assert "> 💡 **Tip** — Setup" in out, marker
        assert "> install the thing" in out, marker
        assert "> then run it" in out, marker
        assert f'{marker} tip "Setup"' not in out, f"{marker}: literal header leaked to notebook"


def test_collapsible_without_title() -> None:
    out = build.transform_admonitions("??? warning\n    watch out\n")
    assert "> ⚠️ **Warning**" in out
    assert "> watch out" in out


def test_unknown_kind_falls_back_to_bold_label() -> None:
    out = build.transform_admonitions('??? danger "Careful"\n    boom\n')
    assert "> **Danger** — Careful" in out  # not in ADMONITION_LABELS → capitalised fallback
    assert "> boom" in out


def test_non_admonition_text_passes_through_unchanged() -> None:
    md = "Just a paragraph with a ??? question mark inline, not an admonition."
    assert build.transform_admonitions(md) == md


def test_bare_triple_question_without_kind_is_not_an_admonition() -> None:
    # A lone ``???`` with no admonition kind must not be swallowed.
    md = "???"
    assert build.transform_admonitions(md) == md
