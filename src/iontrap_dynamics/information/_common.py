# SPDX-License-Identifier: MIT
"""Shared nonlinear-in-ρ helpers for the ``information`` sub-package.

Private utilities reused across :mod:`~iontrap_dynamics.information.fisher`,
:mod:`~iontrap_dynamics.information.redundancy`, and
:mod:`~iontrap_dynamics.information.recoverability` — the shared helper layer
that motivated the single ``information/`` umbrella (WP-01 §3). Everything here
is private; nothing is re-exported from the package.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import qutip

from ..hilbert import HilbertSpace

_ENTROPY_EIGENVALUE_CUTOFF: float = 1e-12
"""Eigenvalue floor below which a density-matrix eigenvalue is treated as 0."""


def _ensure_density(state: qutip.Qobj) -> qutip.Qobj:
    """Promote a ket to a density matrix if needed."""
    if state.isket:
        return state * state.dag()
    return state


def _von_neumann_entropy_bits(rho: qutip.Qobj) -> float:
    """von Neumann entropy ``S(ρ) = −Σ λ log₂ λ`` in bits."""
    eigenvalues = np.clip(np.asarray(rho.eigenenergies(), dtype=np.float64), 0.0, None)
    nonzero = eigenvalues[eigenvalues > _ENTROPY_EIGENVALUE_CUTOFF]
    if nonzero.size == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _validate_indices(
    indices: Sequence[int],
    hilbert: HilbertSpace,
    name: str,
) -> list[int]:
    """Validate a set of subsystem indices against ``hilbert``'s subsystem count."""
    idx = list(indices)
    if not idx:
        raise ValueError(f"{name}: must name at least one subsystem")
    n_subsystems = hilbert.n_ions + hilbert.n_modes
    for i in idx:
        if not 0 <= i < n_subsystems:
            raise ValueError(f"{name}: index {i} out of range [0, {n_subsystems})")
    if len(set(idx)) != len(idx):
        raise ValueError(f"{name}: indices must be distinct; got {list(indices)}")
    return idx
