# SPDX-License-Identifier: MIT
"""Trace distance and the BLP non-Markovianity measure.

Phase A / WI-1 of the non-Markovianity + spectral-density task card
(``task cards/TC-non-markovianity-spectral-density.md``; WP-04 candidate). Every
symbol here is **application-agnostic** — defined on generic quantum states with
no reference to any consuming application (e.g. the Wittemer-2018 reproduction
lives in a benchmark/tutorial, not here).

These are **observable-only** primitives that add **no new convention symbol**:
the trace distance is the standard distinguishability functional and the BLP
measure is a fixed functional of it (the MCF probe-QFI precedent — ship a derived
observable, cite the existing frozen conventions, edit nothing; cf. §19's
information-observable family and the §15 no-silent-degradation ladder).

The module provides:

- :func:`trace_distance` — ``D(ρ₁, ρ₂) = ½‖ρ₁ − ρ₂‖₁`` between two states, the
  distinguishability of an equiprobable pair (Breuer–Laine–Piilo, PRL 103,
  210401 (2009), Eq. 1).
- :func:`trace_distance_trajectory` — ``D(t)`` along a pair of trajectories,
  optionally tracing onto a subsystem (the reduced *open system*) first.
- :func:`blp_non_markovianity` — the accumulated measure
  ``𝒩 = Σ_i [D(tᵢ) − D(tᵢ₋₁)]_{>0}`` (the discretised, **fixed-pair** form used by
  Wittemer et al., PRA 97, 020102(R) (2018), Eq. 1). A monotone non-increasing
  ``D(t)`` (Markovian / divisible dynamics) gives ``𝒩 = 0``; any temporary
  increase — a back-flow of information from environment to system — gives
  ``𝒩 > 0``.

Conventions
-----------

For two density matrices the trace distance is ``D = ½ Σ_k |λ_k|`` over the
eigenvalues ``λ_k`` of the Hermitian difference ``ρ₁ − ρ₂``; for valid states
``D ∈ [0, 1]``, with ``D = 0`` iff ``ρ₁ = ρ₂`` and ``D = 1`` for orthogonal pure
states. The **original** BLP measure (BLP 2009, Eqs. 11–12) additionally
maximises over all initial state pairs; that optimisation is deliberately *not*
performed here — PRA 97, 020102 fixes one representative orthogonal pair, which
is what :func:`blp_non_markovianity` consumes. (A sup-over-pairs variant is a
separate, deferred addition; see the task card §9.)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import qutip
from numpy.typing import NDArray

from ..hilbert import HilbertSpace
from ._common import _ensure_density, _validate_indices, _validate_state_dim

_TRACE_DISTANCE_TOLERANCE: float = 1e-9
"""Slack on the ``D ∈ [0, 1]`` bound; a gross violation signals a malformed state."""


def trace_distance(state_1: qutip.Qobj, state_2: qutip.Qobj) -> float:
    """Trace distance ``D(ρ₁, ρ₂) = ½‖ρ₁ − ρ₂‖₁`` between two states.

    Kets are promoted to density matrices. The value is the distinguishability
    of the equiprobable pair: a guesser's optimal success probability is
    ``½(1 + D)``.

    Parameters
    ----------
    state_1, state_2
        QuTiP states — kets or density matrices — on the **same** Hilbert space.

    Returns
    -------
    float
        ``D ∈ [0, 1]`` for valid states (``0`` iff identical).

    Raises
    ------
    ValueError
        If the two states have different shapes, or the difference has
        non-finite eigenvalues, or ``D`` exceeds 1 beyond round-off (a malformed,
        non-physical input — surfaced, not silently clipped; §15 ladder).

    Notes
    -----
    Inputs are assumed **physical** — Hermitian, positive-semidefinite,
    trace-one (solver outputs or the state factories). Physicality is **not**
    exhaustively re-validated beyond the shape, finiteness, and ``D ≤ 1`` guards
    above — consistent with the rest of :mod:`information` (the QFI evaluator
    clips rather than rejects ρ's tiny negative eigenvalues) and deliberate: a
    strict trace/PSD gate would reject legitimate ``mesolve`` outputs whose trace
    drifts by ~round-off over long evolutions. The non-finite-difference and
    ``D > 1`` checks are the §15 boundary that is surfaced; a subnormalised or
    non-positive input that still yields ``D ≤ 1`` is caller error, not caught
    here.
    """
    rho_1 = _ensure_density(state_1)
    rho_2 = _ensure_density(state_2)
    if rho_1.shape != rho_2.shape:
        raise ValueError(
            f"trace_distance: states have shapes {rho_1.shape} and {rho_2.shape}; "
            "they must be equal"
        )
    diff = np.asarray((rho_1 - rho_2).full(), dtype=np.complex128)
    # ρ₁ − ρ₂ is Hermitian; symmetrise to absorb round-off before the eigensolve.
    diff = 0.5 * (diff + diff.conj().T)
    eigenvalues = np.linalg.eigvalsh(diff)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(
            "trace_distance: the state difference has non-finite eigenvalues "
            "(malformed state); refusing to return a silently-masked distance"
        )
    distance = 0.5 * float(np.sum(np.abs(eigenvalues)))
    if distance > 1.0 + _TRACE_DISTANCE_TOLERANCE:
        raise ValueError(
            f"trace_distance: D = {distance} exceeds 1; the inputs are not valid "
            "trace-one density matrices"
        )
    return distance


def trace_distance_trajectory(
    states_1: Sequence[qutip.Qobj],
    states_2: Sequence[qutip.Qobj],
    *,
    hilbert: HilbertSpace,
    subsystem_indices: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    """Trace distance ``D(t)`` along a pair of trajectories.

    For each time index the two states are (optionally) reduced to
    ``subsystem_indices`` by a partial trace — the **open system** whose memory
    effects are probed — and the trace distance of the reduced states is
    returned.

    Parameters
    ----------
    states_1, states_2
        Equal-length sequences of QuTiP states (kets or density matrices), each
        on the Hilbert space described by ``hilbert``. They are the two
        trajectories evolved from the two initial conditions.
    hilbert
        :class:`HilbertSpace` ground-truth for the total dimension and subsystem
        ordering; every state is checked against it.
    subsystem_indices
        Subsystems to **retain** (partial-trace onto), in ``hilbert``'s ordering
        — e.g. ``[0]`` keeps the first spin and traces out the motional
        environment. ``None`` keeps the full state (no reduction).

    Returns
    -------
    NDArray[np.float64]
        ``D(t)`` of length ``len(states_1)``.

    Raises
    ------
    ValueError
        If the trajectories differ in length, are empty, a state does not match
        ``hilbert.total_dim``, or ``subsystem_indices`` is invalid.
    """
    if len(states_1) != len(states_2):
        raise ValueError(
            f"trace_distance_trajectory: trajectories differ in length "
            f"({len(states_1)} vs {len(states_2)})"
        )
    if len(states_1) == 0:
        raise ValueError("trace_distance_trajectory: trajectories must be non-empty")
    keep: list[int] | None = None
    if subsystem_indices is not None:
        keep = _validate_indices(subsystem_indices, hilbert, "trace_distance_trajectory")
    values = np.empty(len(states_1), dtype=np.float64)
    for idx, (state_1, state_2) in enumerate(zip(states_1, states_2, strict=True)):
        rho_1 = _ensure_density(state_1)
        rho_2 = _ensure_density(state_2)
        _validate_state_dim(rho_1, hilbert, f"trace_distance_trajectory (state_1[{idx}])")
        _validate_state_dim(rho_2, hilbert, f"trace_distance_trajectory (state_2[{idx}])")
        if keep is not None:
            rho_1 = rho_1.ptrace(keep)
            rho_2 = rho_2.ptrace(keep)
        values[idx] = trace_distance(rho_1, rho_2)
    return values


def blp_non_markovianity(distances: NDArray[np.float64] | Sequence[float]) -> float:
    """Accumulated BLP non-Markovianity ``𝒩 = Σ_i [D(tᵢ) − D(tᵢ₋₁)]_{>0}``.

    The discretised, fixed-pair measure of Wittemer et al., PRA 97, 020102(R)
    (2018), Eq. 1: the sum of the **positive** increments of a sampled trace
    distance. ``𝒩 = 0`` for monotone non-increasing ``D`` (Markovian / divisible
    dynamics); ``𝒩 > 0`` signals information back-flow.

    Parameters
    ----------
    distances
        A 1-D sequence of trace-distance samples ``D(t₀), D(t₁), …`` in time
        order (e.g. from :func:`trace_distance_trajectory`).

    Returns
    -------
    float
        ``𝒩 ≥ 0``. Fewer than two samples gives ``0`` (no increments).

    Raises
    ------
    ValueError
        If ``distances`` is not 1-D, contains non-finite values, or has any
        sample outside ``[0, 1]`` beyond round-off (not a valid trace distance) —
        the §15 ladder: an unphysical input is surfaced, not summed over.

    Notes
    -----
    This is the *fixed-pair* measure (one supplied trajectory pair). The original
    BLP measure also maximises over initial pairs; that optimisation is out of
    scope here (task card §9).
    """
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError(f"blp_non_markovianity: distances must be 1-D; got ndim {d.ndim}")
    if not np.all(np.isfinite(d)):
        raise ValueError(
            "blp_non_markovianity: distances contain non-finite values; refusing to "
            "accumulate a silently-masked measure"
        )
    if d.size and (
        float(np.min(d)) < -_TRACE_DISTANCE_TOLERANCE
        or float(np.max(d)) > 1.0 + _TRACE_DISTANCE_TOLERANCE
    ):
        raise ValueError(
            "blp_non_markovianity: trace-distance samples must lie in [0, 1]; got range "
            f"[{float(np.min(d))}, {float(np.max(d))}] — these are not valid trace distances"
        )
    if d.size < 2:
        return 0.0
    increments = np.diff(d)
    return float(np.sum(increments[increments > 0.0]))


__all__ = [
    "blp_non_markovianity",
    "trace_distance",
    "trace_distance_trajectory",
]
