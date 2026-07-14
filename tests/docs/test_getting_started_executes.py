# SPDX-License-Identifier: MIT
"""Execute the runnable python blocks in ``docs/getting-started.md`` end-to-end.

The Getting Started page carries a runnable first-simulation example (WP-06 dispatch
TA5) that the tutorial-execution guard does NOT cover — that guard only globs
``docs/tutorials/[0-9][0-9]_*.md``. This test keeps that example, and the hand-built
result-object snippet beside it, from silently rotting against the public API.

Each block runs from a fresh temporary working directory: a reader who ``pip install``s
the package has no repo checkout, so a block that read a repo-relative file would fail
here — exactly the regression we want to catch. The page's example is plot-free, so
this needs only the base dependencies (no ``[plot]`` extra, hence no ``tutorial`` mark).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

_PAGE = Path(__file__).resolve().parents[2] / "docs" / "getting-started.md"
_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def test_getting_started_python_blocks_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = _PYTHON_BLOCK.findall(_PAGE.read_text(encoding="utf-8"))
    assert blocks, "getting-started.md: no python blocks found"

    monkeypatch.chdir(tmp_path)  # no repo checkout — repo-relative reads must fail
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, block in enumerate(blocks):
            exec(compile(block, f"getting-started:block{i}", "exec"), namespace)
