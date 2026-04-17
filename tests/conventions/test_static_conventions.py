# SPDX-License-Identifier: MIT
"""Static convention-enforcement tests (workplan §0.D).

These tests parse the package source tree and verify compliance with rules
from ``CONVENTIONS.md`` that can be checked without executing the code.
They are permanent anchors: rules applicable to current code (e.g. §3's
ban on ``qutip.sigmaz``) and guards against drift as Phase 1 builders land.

Rules checked
-------------

**CONVENTIONS.md §3 — QuTiP discipline.** The atomic-physics Pauli
convention (|↓⟩ with σ_z eigenvalue −1) requires the library to expose its
own sign-flipped ``sigma_z_ion``. The §3 rule explicitly states:

    "`from qutip import sigmaz` is banned and flagged by the
    convention-enforcement tests."

Three scans enforce this:

1. No ``from qutip import *`` wildcards (explicit imports only, so reviewers
   can tell what leaks from the QuTiP namespace).
2. No ``from qutip import sigmaz`` (named import of the banned symbol).
3. No ``qutip.sigmaz`` attribute access (same symbol, different syntax).

Scope
-----

Only ``src/iontrap_dynamics/`` is scanned. Tests, tools, and examples are
outside the library's public contract and may use QuTiP idioms directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.convention

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "iontrap_dynamics"


def _package_source_files() -> list[Path]:
    """Every ``.py`` file under the package, excluding ``__pycache__``."""
    return sorted(path for path in _PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _format_violations(violations: list[str]) -> str:
    """Render a violation list as a human-readable message for assert output."""
    return "\n  " + "\n  ".join(violations)


# ----------------------------------------------------------------------------
# QuTiP discipline — CONVENTIONS.md §3
# ----------------------------------------------------------------------------


class TestQutipDiscipline:
    def test_no_wildcard_qutip_import(self) -> None:
        """``from qutip import *`` is banned: imports must be explicit so
        reviewers can see exactly which QuTiP symbols are in scope and
        spot any banned ones (§3)."""
        violations: list[str] = []
        for path in _package_source_files():
            tree = _parse(path)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "qutip"
                    and any(alias.name == "*" for alias in node.names)
                ):
                    violations.append(
                        f"{path.relative_to(_PACKAGE_ROOT.parent.parent)}:{node.lineno}"
                    )
        assert not violations, (
            "wildcard qutip imports (CONVENTIONS.md §3 forbids):" + _format_violations(violations)
        )

    def test_no_named_qutip_sigmaz_import(self) -> None:
        """``from qutip import sigmaz`` is banned (CONVENTIONS.md §3).

        QuTiP's ``sigmaz()`` returns ``diag(+1, −1)`` relative to
        ``basis(2, 0)`` — the quantum-information convention. Under our
        Wineland convention (``|↓⟩ ≡ basis(2, 0)`` with σ_z eigenvalue
        −1), using QuTiP's default would silently flip the sign.

        The library exposes its own ``sigma_z_ion`` with the correct sign.
        This test catches direct-import bypasses.
        """
        violations: list[str] = []
        for path in _package_source_files():
            tree = _parse(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "qutip":
                    for alias in node.names:
                        if alias.name == "sigmaz":
                            violations.append(
                                f"{path.relative_to(_PACKAGE_ROOT.parent.parent)}:{node.lineno}"
                            )
        assert not violations, (
            "banned `from qutip import sigmaz` (CONVENTIONS.md §3):"
            + _format_violations(violations)
            + "\n  Use `iontrap_dynamics.operators.sigma_z_ion` (Phase 1+) instead."
        )

    def test_no_qutip_sigmaz_attribute_access(self) -> None:
        """``qutip.sigmaz`` attribute access is banned (CONVENTIONS.md §3).

        Catches ``qutip.sigmaz()`` calls and bare ``qutip.sigmaz``
        references, which bypass the named-import gate but have the same
        sign-convention problem. Companion to
        :meth:`test_no_named_qutip_sigmaz_import`.
        """
        violations: list[str] = []
        for path in _package_source_files():
            tree = _parse(path)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "sigmaz"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "qutip"
                ):
                    violations.append(
                        f"{path.relative_to(_PACKAGE_ROOT.parent.parent)}:{node.lineno}"
                    )
        assert not violations, (
            "banned `qutip.sigmaz` attribute access (CONVENTIONS.md §3):"
            + _format_violations(violations)
            + "\n  Use `iontrap_dynamics.operators.sigma_z_ion` (Phase 1+) instead."
        )


# ----------------------------------------------------------------------------
# Self-test: the scanner actually detects violations when they exist.
# ----------------------------------------------------------------------------
#
# Without a self-test, a test file that scans an initially-empty package for
# bad imports will appear to pass even if the scanner is broken. These
# self-tests construct synthetic AST trees and verify the detection logic.


class TestScannerSelfTest:
    def test_wildcard_detection_is_not_vacuous(self) -> None:
        """Construct an in-memory module containing a banned import and
        confirm the same visitor logic flags it. Rules out "passes because
        no source files exist" false positives."""
        synthetic = ast.parse("from qutip import *\n")
        flagged = [
            n
            for n in ast.walk(synthetic)
            if isinstance(n, ast.ImportFrom)
            and n.module == "qutip"
            and any(a.name == "*" for a in n.names)
        ]
        assert len(flagged) == 1

    def test_named_sigmaz_detection_is_not_vacuous(self) -> None:
        synthetic = ast.parse("from qutip import sigmaz, sigmax\n")
        flagged = [
            (n, a)
            for n in ast.walk(synthetic)
            if isinstance(n, ast.ImportFrom) and n.module == "qutip"
            for a in n.names
            if a.name == "sigmaz"
        ]
        assert len(flagged) == 1

    def test_attribute_sigmaz_detection_is_not_vacuous(self) -> None:
        synthetic = ast.parse("import qutip\nx = qutip.sigmaz()\n")
        flagged = [
            n
            for n in ast.walk(synthetic)
            if isinstance(n, ast.Attribute)
            and n.attr == "sigmaz"
            and isinstance(n.value, ast.Name)
            and n.value.id == "qutip"
        ]
        assert len(flagged) == 1


# ----------------------------------------------------------------------------
# Sanity: the package has source files to scan (defensive — ensures the
# above tests are not silently passing on an empty tree).
# ----------------------------------------------------------------------------


class TestPackagePresence:
    def test_package_source_tree_is_non_empty(self) -> None:
        files = _package_source_files()
        assert len(files) > 0, (
            f"expected at least one .py file under {_PACKAGE_ROOT}; "
            "convention-enforcement tests would be vacuously true on empty input."
        )
