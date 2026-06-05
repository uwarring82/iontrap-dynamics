# SPDX-License-Identifier: MIT
"""Execute every docs/tutorials notebook end-to-end and let its asserts judge it.

The tutorials are linear scripts whose embedded ``assert`` statements are the
oracle (e.g. ``assert np.max(pop_red) < 1e-4``). ``tools/build_tutorial_notebooks.py``
copies each ``` ```python ``` block verbatim into a notebook cell, so executing the
Markdown blocks in order — sharing one namespace, exactly as a notebook kernel
does — is a faithful test of what a reader runs on Colab.

This is the guard the ``notebooks`` freshness job does NOT provide: ``--check``
only verifies ``notebook == regeneration``, never *runs* the code. Running these
caught real latent errors the test suite had never executed (tutorials are not
otherwise collected by pytest).

Each tutorial runs from a fresh temporary working directory: a Colab user has no
repository checkout, so a tutorial that reads a repo-relative file would fail
here — which is exactly the regression we want to catch.

Marked ``tutorial`` and skipped without the ``[plot]`` extra (the tutorials
``import matplotlib``); CI runs them in a dedicated job that installs it.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")  # tutorials import matplotlib.pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is fixed)

TUTORIALS_DIR = Path(__file__).resolve().parents[2] / "docs" / "tutorials"
_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _python_blocks(markdown: str) -> list[str]:
    return _PYTHON_BLOCK.findall(markdown)


pytestmark = pytest.mark.tutorial


@pytest.mark.parametrize(
    "tutorial",
    sorted(TUTORIALS_DIR.glob("[0-9][0-9]_*.md")),
    ids=lambda p: p.stem,
)
def test_tutorial_runs_end_to_end(
    tutorial: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All ``` ```python ``` blocks run in order in one namespace with no error."""
    blocks = _python_blocks(tutorial.read_text(encoding="utf-8"))
    assert blocks, f"{tutorial.name}: no python blocks found"

    # Colab has no repo checkout — run from an empty cwd so repo-relative reads fail.
    monkeypatch.chdir(tmp_path)
    # ``plt.show()`` is a no-op under Agg; silence its non-interactive warning.
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    namespace: dict[str, object] = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # tutorials 06/etc. emit on purpose
            for i, block in enumerate(blocks):
                exec(compile(block, f"{tutorial.name}:block{i}", "exec"), namespace)
    finally:
        plt.close("all")
