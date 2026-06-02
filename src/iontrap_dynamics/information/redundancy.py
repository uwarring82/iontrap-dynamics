# SPDX-License-Identifier: MIT
"""Quantum-Darwinism redundancy measures.

WI-2 of WP-01 (dispatch EDB). Application-agnostic: every symbol is defined on a
generic multipartite state with a designated *system* partition and an
*environment* partition (sets of subsystem indices in the ``HilbertSpace`` §2
ordering), with no reference to any consuming application.

The module provides:

- :func:`fragment_mutual_information` — the quantum mutual information
  ``I(S:F)`` between the system and a chosen environment fragment.
- :func:`partial_information_plot` — the ``I(S:F)`` curve versus fragment size
  (the partial-information plot of quantum Darwinism), built over nested
  fragments in the given environment-index order.
- :func:`redundancy` — the redundancy ``R_δ = N / f_δ`` of the system
  information across the environment.

These measures are **nonlinear in ρ** (von Neumann entropies of reduced
states), so they follow the *trajectory-evaluator* posture of
:mod:`iontrap_dynamics.entanglement` rather than the ``operator → expect``
observable shape: run the solver with ``storage_mode=StorageMode.EAGER`` and
apply these as post-processing.

Conventions
-----------

(CONVENTIONS.md §20, staged for the shared v0.3 Convention Freeze — see
``WP/FREEZE-v0.3.md``.) Entropies are von Neumann entropies in **bits** (base
2). The quantum mutual information is

    I(S:F) = S(ρ_S) + S(ρ_F) − S(ρ_{S∪F}).

The redundancy is ``R_δ = N / f_δ`` where ``N`` is the number of environment
subsystems, ``H_S = S(ρ_S)`` the system entropy, and ``f_δ`` the smallest
environment fraction whose fragment carries at least ``(1 − δ)·H_S`` of the
system information (the classical-information-deficit convention of Zurek,
*Nat. Phys.* 2009; Ollivier, Poulin & Zurek). Equivalently ``R_δ`` counts how
many disjoint fragments each suffice to learn the system to within deficit δ.

For the canonical decoherence model — one system qubit perfectly copied onto N
environment qubits (a GHZ cascade) — ``I(S:F)`` reaches the **plateau**
``H_S = 1`` bit for every non-empty proper fragment, and ``R_δ = N``
(maximally redundant: each single fragment already carries the whole bit).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import qutip
from numpy.typing import NDArray

from ..hilbert import HilbertSpace
from ._common import (
    _ENTROPY_EIGENVALUE_CUTOFF,
    _ensure_density,
    _validate_indices,
    _validate_state_dim,
    _von_neumann_entropy_bits,
)


def fragment_mutual_information(
    state: qutip.Qobj,
    *,
    hilbert: HilbertSpace,
    system_indices: Sequence[int],
    fragment_indices: Sequence[int],
) -> float:
    """Quantum mutual information ``I(S:F)`` between system and fragment.

    Parameters
    ----------
    state
        A QuTiP state — ket or density matrix — on ``hilbert``.
    hilbert
        :class:`HilbertSpace` fixing the subsystem ordering (§2: spins first,
        then modes). Subsystem indices run ``0 .. n_ions + n_modes − 1``.
    system_indices, fragment_indices
        Disjoint sets of subsystem indices forming the system ``S`` and the
        fragment ``F``.

    Returns
    -------
    float
        ``I(S:F) = S(ρ_S) + S(ρ_F) − S(ρ_{S∪F})`` in bits (``≥ 0``).

    Raises
    ------
    ValueError
        If the index sets are empty, out of range, contain repeats, or
        overlap.
    """
    rho = _ensure_density(state)
    _validate_state_dim(rho, hilbert, "fragment_mutual_information")
    sys_idx = _validate_indices(system_indices, hilbert, "system_indices")
    frag_idx = _validate_indices(fragment_indices, hilbert, "fragment_indices")
    if set(sys_idx) & set(frag_idx):
        raise ValueError(
            "fragment_mutual_information: system_indices and fragment_indices "
            f"must be disjoint; got {system_indices} and {fragment_indices}"
        )
    s_system = _von_neumann_entropy_bits(rho.ptrace(sys_idx))
    s_fragment = _von_neumann_entropy_bits(rho.ptrace(frag_idx))
    s_joint = _von_neumann_entropy_bits(rho.ptrace(sorted(set(sys_idx) | set(frag_idx))))
    return s_system + s_fragment - s_joint


def partial_information_plot(
    state: qutip.Qobj,
    *,
    hilbert: HilbertSpace,
    system_indices: Sequence[int],
    environment_indices: Sequence[int],
) -> NDArray[np.float64]:
    """The partial-information curve ``I(S:F)`` versus fragment size.

    Builds nested fragments ``F_f = environment_indices[:f]`` for
    ``f = 0 .. N`` (``N = len(environment_indices)``) and returns
    ``[I(S:F_0), I(S:F_1), …, I(S:F_N)]`` — the partial-information plot of
    quantum Darwinism. ``I(S:F_0) = 0`` by convention. For a symmetric
    decoherence model the nested curve equals the fragment-averaged curve.

    Returns
    -------
    NDArray[np.float64]
        Array of length ``N + 1``; entry ``f`` is ``I(S:F_f)`` in bits.

    Raises
    ------
    ValueError
        If the index sets are empty, out of range, contain repeats, or
        overlap.
    """
    rho = _ensure_density(state)
    _validate_state_dim(rho, hilbert, "partial_information_plot")
    sys_idx = _validate_indices(system_indices, hilbert, "system_indices")
    env_idx = _validate_indices(environment_indices, hilbert, "environment_indices")
    if set(sys_idx) & set(env_idx):
        raise ValueError(
            "partial_information_plot: system_indices and environment_indices "
            f"must be disjoint; got {system_indices} and {environment_indices}"
        )
    return _partial_information_from_density(rho, sys_idx=sys_idx, env_idx=env_idx)


def _partial_information_from_density(
    rho: qutip.Qobj,
    *,
    sys_idx: Sequence[int],
    env_idx: Sequence[int],
) -> NDArray[np.float64]:
    """PIP loop on an already-converted, already-validated density matrix.

    Shared by :func:`partial_information_plot` and :func:`redundancy` so the
    latter converts the input ket to a density matrix exactly once.
    """
    n_env = len(env_idx)
    pip = np.empty(n_env + 1, dtype=np.float64)
    pip[0] = 0.0
    s_system = _von_neumann_entropy_bits(rho.ptrace(list(sys_idx)))
    for f in range(1, n_env + 1):
        fragment = list(env_idx[:f])
        s_fragment = _von_neumann_entropy_bits(rho.ptrace(fragment))
        s_joint = _von_neumann_entropy_bits(rho.ptrace(sorted(set(sys_idx) | set(fragment))))
        pip[f] = s_system + s_fragment - s_joint
    return pip


def redundancy(
    state: qutip.Qobj,
    *,
    hilbert: HilbertSpace,
    system_indices: Sequence[int],
    environment_indices: Sequence[int],
    delta: float = 0.1,
) -> float:
    """The redundancy ``R_δ = N / f_δ`` of the system information.

    ``f_δ`` is the smallest environment fraction ``k / N`` whose nested
    fragment carries at least ``(1 − δ)·H_S`` of the system information
    ``H_S = S(ρ_S)``; ``R_δ = N / k`` then counts how many such fragments fit in
    the environment.

    Parameters
    ----------
    delta
        Information deficit in ``(0, 1)``. Default ``0.1``.

    Returns
    -------
    float
        ``R_δ ≥ 0``. Returns ``0.0`` if the system carries no information
        (``H_S ≈ 0``) or if no fragment reaches the ``(1 − δ)·H_S`` threshold.

    Raises
    ------
    ValueError
        If ``delta`` is not in ``(0, 1)``, or the index sets are invalid /
        overlapping.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"redundancy: delta must be in (0, 1); got {delta}")
    rho = _ensure_density(state)
    _validate_state_dim(rho, hilbert, "redundancy")
    sys_idx = _validate_indices(system_indices, hilbert, "system_indices")
    env_idx = _validate_indices(environment_indices, hilbert, "environment_indices")
    if set(sys_idx) & set(env_idx):
        raise ValueError(
            "redundancy: system_indices and environment_indices must be disjoint; "
            f"got {system_indices} and {environment_indices}"
        )
    h_system = _von_neumann_entropy_bits(rho.ptrace(sys_idx))
    if h_system < _ENTROPY_EIGENVALUE_CUTOFF:
        return 0.0
    pip = _partial_information_from_density(rho, sys_idx=sys_idx, env_idx=env_idx)
    n_env = len(env_idx)
    threshold = (1.0 - delta) * h_system
    for k in range(1, n_env + 1):
        if pip[k] >= threshold:
            return float(n_env / k)
    return 0.0


__all__ = [
    "fragment_mutual_information",
    "partial_information_plot",
    "redundancy",
]
