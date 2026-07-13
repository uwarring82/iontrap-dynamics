# SPDX-License-Identifier: MIT
"""Phase-space façades over the Gaussian core: Wigner function + readout entry point.

CONVENTIONS.md §26.3 / WP-05 R4 (no-fork rule). This module holds **only**
Wigner and readout *façades* over :mod:`iontrap_dynamics.gaussian`; the covariance
/ symplectic arithmetic lives there (the ``N = 1`` core), never here. It adds:

* :func:`wigner` — a Wigner-function wrapper with the **scaling pinned** to the
  §26.2 vacuum-variance-1 convention (``x̂ = â + â†``), i.e. QuTiP's ``g = 1``
  (its default ``g = √2`` gives vacuum variance ½). The pin is part of the sealed
  §26.3 convention, not a free display parameter.
* :func:`phase_space_readout` — a convenience that partial-traces a full
  spin⊗mode state down to one mode (if needed) and returns the
  :class:`~iontrap_dynamics.gaussian.GaussianReadout` (§26.4, observable-only).

The Gaussian readout primitives are re-exported here so callers have a single
phase-space import surface.
"""

from __future__ import annotations

import numpy as np
import qutip

from .conventions import FOCK_CONVERGENCE_TOLERANCE
from .gaussian import (
    GaussianReadout,
    check_fock_truncation,
    coherent_amplitude,
    covariance_matrix,
    gaussian_readout,
    mean_occupation,
    mean_squeezed_occupation,
    phonon_number_distribution,
    pure_squeezed_vacuum_pn,
    quadrature_operators,
    reduced_single_mode,
    squeezing_parameter,
    symplectic_eigenvalue,
)
from .hilbert import HilbertSpace

#: QuTiP ``wigner`` scaling giving the §26.2 vacuum-variance-1 quadratures
#: (``x̂ = â + â†``). QuTiP's default ``g = √2`` gives vacuum variance ½; ``g = 1``
#: puts the vacuum Wigner variance at 1, matching the covariance readout.
WIGNER_G: float = 1.0


def _as_single_mode(
    state: qutip.Qobj, hilbert: HilbertSpace | None, mode_label: str | None
) -> qutip.Qobj:
    """Return ``state`` if it is already single-mode, else partial-trace to the named mode."""
    if len(state.dims[0]) == 1:
        return state
    if hilbert is None or mode_label is None:
        raise ValueError(
            "phase-space readout of a multi-subsystem state requires hilbert= and "
            f"mode_label= to reduce to one mode; got dims {state.dims}."
        )
    return reduced_single_mode(state, hilbert, mode_label)


def wigner(
    state: qutip.Qobj,
    xvec: np.ndarray,
    pvec: np.ndarray | None = None,
    *,
    hilbert: HilbertSpace | None = None,
    mode_label: str | None = None,
) -> np.ndarray:
    """Return the Wigner function ``W(x, p)`` on the §26.2 vacuum-variance-1 grid.

    Thin wrapper over :func:`qutip.wigner` with the scaling **pinned** to
    :data:`WIGNER_G` (``= 1``) so the vacuum Wigner has variance 1 and its
    principal variances coincide with the covariance eigenvalues (§26.3). A full
    spin⊗mode ``state`` is partial-traced to ``mode_label`` first (supply
    ``hilbert`` and ``mode_label``); a single-mode state is used directly.

    Parameters
    ----------
    state
        A single-mode ket/density matrix, or a full spin⊗mode state (with
        ``hilbert`` + ``mode_label`` for the reduction).
    xvec
        1-D grid of ``x̂`` values.
    pvec
        1-D grid of ``p̂`` values. Defaults to ``xvec``.
    hilbert, mode_label
        Required only when ``state`` spans more than one subsystem.

    Returns
    -------
    numpy.ndarray
        ``W`` with shape ``(len(pvec), len(xvec))`` (QuTiP's convention).

    Notes
    -----
    Marginal variances read back from ``W`` on a finite grid are
    **grid-resolution-limited** (spacing + extent); use a sufficiently fine, wide
    grid when comparing them to the covariance eigenvalues.
    """
    mode_state = _as_single_mode(state, hilbert, mode_label)
    if pvec is None:
        pvec = xvec
    return np.asarray(qutip.wigner(mode_state, xvec, pvec, g=WIGNER_G), dtype=float)


def phase_space_readout(
    state: qutip.Qobj,
    *,
    hilbert: HilbertSpace | None = None,
    mode_label: str | None = None,
    check_truncation: bool = True,
    tolerance: float = FOCK_CONVERGENCE_TOLERANCE,
    window: int = 2,
) -> GaussianReadout:
    """Return the single-mode :class:`GaussianReadout` (§26.4) of a state.

    Convenience façade: partial-traces a full spin⊗mode ``state`` to ``mode_label``
    (supply ``hilbert`` + ``mode_label``) then applies
    :func:`iontrap_dynamics.gaussian.gaussian_readout`, forwarding the
    Fock-truncation guard controls unchanged.
    """
    return gaussian_readout(
        _as_single_mode(state, hilbert, mode_label),
        check_truncation=check_truncation,
        tolerance=tolerance,
        window=window,
    )


__all__ = [
    "WIGNER_G",
    "GaussianReadout",
    "check_fock_truncation",
    "coherent_amplitude",
    "covariance_matrix",
    "gaussian_readout",
    "mean_occupation",
    "mean_squeezed_occupation",
    "phase_space_readout",
    "phonon_number_distribution",
    "pure_squeezed_vacuum_pn",
    "quadrature_operators",
    "reduced_single_mode",
    "squeezing_parameter",
    "symplectic_eigenvalue",
    "wigner",
]
