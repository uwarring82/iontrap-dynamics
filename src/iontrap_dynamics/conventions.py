# SPDX-License-Identifier: MIT
"""Authoritative convention version marker for :mod:`iontrap_dynamics`.

Every :class:`~iontrap_dynamics.results.TrajectoryResult` records
``CONVENTION_VERSION`` in its metadata, so that downstream analyses carry
an unambiguous pointer back to the ``CONVENTIONS.md`` revision under which
the result was produced.

The full convention-enforcement surface (``ConventionSet`` object, unit
conversions, canonical constants) will land in a later Phase 0 step. For
v0.1 this module provides only the version constant — enough to populate
result metadata ahead of the builder work.
"""

from __future__ import annotations

#: Version of ``CONVENTIONS.md`` this release of the library is built against.
#: Bumps follow CONVENTIONS.md Convention Freeze gates: additions are free,
#: changes require a minor-version bump and a CHANGELOG entry.
CONVENTION_VERSION: str = "0.1-draft"

#: Default Fock-truncation convergence tolerance ε (CONVENTIONS.md §13).
#: The top-Fock-level population ``p_top = max_t ⟨N_Fock−1|ρ_m|N_Fock−1⟩``
#: drives the §15 warnings ladder:
#:
#:   - ``p_top < ε/10``     → OK, silent
#:   - ``ε/10 ≤ p_top < ε`` → :class:`FockConvergenceWarning` (Level 1)
#:   - ``ε ≤ p_top < 10·ε`` → :class:`FockQualityWarning`     (Level 2)
#:   - ``p_top ≥ 10·ε``     → :class:`ConvergenceError`       (Level 3)
#:
#: Callers may override per call via ``solve(fock_tolerance=…)``.
FOCK_CONVERGENCE_TOLERANCE: float = 1e-4

__all__ = ["CONVENTION_VERSION", "FOCK_CONVERGENCE_TOLERANCE"]
