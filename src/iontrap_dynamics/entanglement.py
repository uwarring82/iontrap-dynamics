# SPDX-License-Identifier: MIT
"""Registered entanglement observables for spin-motion trajectories.

Phase 1 v0.2 additions named in ``WORKPLAN_v0.3.md`` §5 Phase 1
(Core physics and measurement layer — "logarithmic negativity /
concurrence / EoF as registered observables"). These three measures
are intrinsically **nonlinear in the density matrix** — they involve
eigenvalues or singular values of operators built from ρ — and so do
not fit the ``operator → qutip.expect`` shape consumed by
:func:`iontrap_dynamics.observables.expectations_over_time`. Instead
they are exposed here as *trajectory evaluators*: functions that take
a sequence of quantum states (kets or density matrices) and return a
1-D array with one scalar per state.

Usage pattern
-------------

The solver in :mod:`iontrap_dynamics.sequences` must run with
``storage_mode=StorageMode.EAGER`` so the states are retained; the
evaluators are then applied as post-processing::

    from iontrap_dynamics.results import StorageMode
    from iontrap_dynamics.sequences import solve
    from iontrap_dynamics.entanglement import (
        concurrence_trajectory,
        log_negativity_trajectory,
    )

    result = solve(..., storage_mode=StorageMode.EAGER)
    c = concurrence_trajectory(
        result.states, hilbert=hilbert, ion_indices=(0, 1),
    )
    e_N = log_negativity_trajectory(
        result.states, hilbert=hilbert, partition="spins",
    )

Subsystem conventions follow ``CONVENTIONS.md`` §2 (spin subsystems
first, mode subsystems after). The evaluators consume
``HilbertSpace`` as the ground-truth for tensor-ordering instead of
inferring it from QuTiP dims.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import qutip
from numpy.typing import NDArray

from .hilbert import HilbertSpace


def concurrence_trajectory(
    states: Sequence[qutip.Qobj],
    *,
    hilbert: HilbertSpace,
    ion_indices: tuple[int, int],
) -> NDArray[np.float64]:
    """Wootters concurrence of two ions' spin-spin reduced state.

    For each state in ``states``, partial-trace over everything except
    the two named spin subsystems, compute Wootters' concurrence on
    the resulting ``4 × 4`` density matrix, return the trajectory.

    Parameters
    ----------
    states
        Sequence of QuTiP states — kets or density matrices — all on
        the Hilbert space described by ``hilbert``.
    hilbert
        :class:`HilbertSpace` used to determine the subsystem ordering
        (§2: spins before modes). The i-th ion's spin sits at
        tensor-subsystem index ``i`` by convention.
    ion_indices
        Two-tuple ``(i, j)`` of ion indices to keep. Must be distinct
        and inside ``[0, hilbert.n_ions)``.

    Returns
    -------
    NDArray[np.float64]
        Concurrence trajectory, length ``len(states)``, entries in
        ``[0, 1]``. ``0`` for separable states, ``1`` for maximally
        entangled Bell states.

    Notes
    -----
    Uses :func:`qutip.concurrence` for the 4 × 4 Wootters calculation.
    Partial trace keeps only the two spin subsystems; motional
    subsystems and any other ion spins are traced out (which can
    turn a pure global state into a mixed reduced state — the
    concurrence handles that).

    Raises
    ------
    ValueError
        If ``ion_indices`` names fewer than two distinct ions, or if
        either index is out of range ``[0, hilbert.n_ions)``.
    """
    i, j = _validate_ion_pair(ion_indices, hilbert.n_ions)
    values = np.empty(len(states), dtype=np.float64)
    for idx, state in enumerate(states):
        rho = _ensure_density(state)
        reduced = rho.ptrace([i, j])
        values[idx] = float(qutip.concurrence(reduced))
    return values


def entanglement_of_formation_trajectory(
    states: Sequence[qutip.Qobj],
    *,
    hilbert: HilbertSpace,
    ion_indices: tuple[int, int],
) -> NDArray[np.float64]:
    """Wootters entanglement of formation for two ions' reduced state.

    Derived from :func:`concurrence_trajectory` via the closed-form
    Wootters relation:

        E_F(ρ) = h((1 + √(1 − C²)) / 2)
        h(x)   = −x log₂ x − (1 − x) log₂(1 − x)   (binary entropy)

    Bounds: ``0 ≤ E_F ≤ 1`` for two-qubit states, with ``1`` on Bell
    states and ``0`` on separable states. Exact for the two-qubit
    case only; for larger bipartite systems :func:`log_negativity_trajectory`
    is the appropriate measure.

    Parameters and raises are identical to :func:`concurrence_trajectory`.
    """
    c = concurrence_trajectory(states, hilbert=hilbert, ion_indices=ion_indices)
    return _binary_entropy(0.5 * (1.0 + np.sqrt(np.maximum(0.0, 1.0 - c * c))))


def log_negativity_trajectory(
    states: Sequence[qutip.Qobj],
    *,
    hilbert: HilbertSpace,
    partition: str = "spins",
) -> NDArray[np.float64]:
    """Logarithmic negativity across a named bipartition.

    The logarithmic negativity ``E_N(ρ) = log₂ ‖ρ^{T_A}‖₁`` is a
    mixed-state entanglement measure that works for arbitrary
    bipartite splits — including spin-vs-motion, which concurrence
    cannot address.

    Parameters
    ----------
    states
        Sequence of QuTiP states on ``hilbert``'s Hilbert space.
    hilbert
        :class:`HilbertSpace` instance used to enumerate subsystems.
    partition
        Which subsystems form subsystem A (the partial-transposed
        side). Supported values:

        - ``"spins"``: A = all spins, B = all modes.
        - ``"modes"``: A = all modes, B = all spins. Gives the same
          numerical value as ``"spins"`` (log-negativity is symmetric
          in the bipartition choice) — provided for callers who find
          the mode-side framing more natural.

    Returns
    -------
    NDArray[np.float64]
        Log-negativity trajectory, length ``len(states)``, entries
        ``≥ 0``.  Returns ``0`` when the state is separable across
        the chosen partition.

    Raises
    ------
    ValueError
        If ``partition`` is not one of ``"spins"`` / ``"modes"``, or
        if the Hilbert space has no modes (the partition is ill-
        defined when one side is empty).

    Notes
    -----
    Uses :func:`qutip.negativity` with ``logarithmic=True``. The
    subsystem mask is built from ``hilbert.subsystem_dims`` per
    CONVENTIONS.md §2 ordering (spins then modes).
    """
    if partition not in {"spins", "modes"}:
        raise ValueError(
            f"log_negativity_trajectory: partition must be 'spins' or 'modes'; got {partition!r}"
        )
    if hilbert.n_modes == 0:
        raise ValueError(
            "log_negativity_trajectory: Hilbert space has no modes — "
            "bipartite negativity across a spin/mode cut requires at "
            "least one mode subsystem."
        )

    # Partial-transpose mask in §2 ordering: spins come first (indices
    # 0 .. n_ions − 1), modes after (n_ions .. n_ions + n_modes − 1).
    # Entry = 1 means transpose that subsystem; = 0 leave alone. Log-
    # negativity is symmetric under bipartition choice, so transposing
    # either side gives the same value.
    n_ions = hilbert.n_ions
    n_modes = hilbert.n_modes
    mask = [1] * n_ions + [0] * n_modes if partition == "spins" else [0] * n_ions + [1] * n_modes

    values = np.empty(len(states), dtype=np.float64)
    for idx, state in enumerate(states):
        rho = _ensure_density(state)
        rho_pt = qutip.partial_transpose(rho, mask)
        # Log-negativity = log₂ ‖ρ^{T_A}‖₁ = log₂ Σ|λᵢ(ρ^{T_A})|.
        # Take abs of the eigenvalues (which can be negative for
        # entangled states) and sum; the diagonal is real for a
        # Hermitian-transpose result.
        eigs = rho_pt.eigenenergies()
        trace_norm = float(np.sum(np.abs(eigs)))
        # log₂(1) = 0 cleanly at separable states; guard against
        # sub-unit trace norm due to floating-point slop.
        values[idx] = float(np.log2(max(trace_norm, 1.0)))
    return values


def _ensure_density(state: qutip.Qobj) -> qutip.Qobj:
    """Promote a ket to a density matrix if needed."""
    if state.isket:
        return state * state.dag()
    return state


def _validate_ion_pair(
    ion_indices: tuple[int, int],
    n_ions: int,
) -> tuple[int, int]:
    if len(ion_indices) != 2:
        raise ValueError(f"ion_indices must be a 2-tuple; got {ion_indices}")
    i, j = ion_indices
    if i < 0 or j < 0 or i >= n_ions or j >= n_ions:
        raise ValueError(f"ion_indices must lie in [0, {n_ions}); got {ion_indices}")
    if i == j:
        raise ValueError(f"ion_indices must be distinct; got {ion_indices}")
    return i, j


def _binary_entropy(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Binary Shannon entropy ``h(p) = −p log₂ p − (1 − p) log₂(1 − p)``.

    Handles ``p ∈ {0, 1}`` boundaries — ``h(0) = h(1) = 0`` — without
    spraying NaN / log(0) warnings.
    """
    p = np.asarray(p, dtype=np.float64)
    # 0·log(0) = 0 by convention; np.where short-circuits the log call.
    h_p = np.where(p > 0.0, -p * np.log2(np.where(p > 0.0, p, 1.0)), 0.0)
    q = 1.0 - p
    h_q = np.where(q > 0.0, -q * np.log2(np.where(q > 0.0, q, 1.0)), 0.0)
    return h_p + h_q


__all__ = [
    "concurrence_trajectory",
    "entanglement_of_formation_trajectory",
    "log_negativity_trajectory",
]
