# SPDX-License-Identifier: MIT
"""Recoverability of system information from accessible fragments.

WI-2 of WP-01 (dispatch EDB). Application-agnostic: defined on a generic
multipartite state with a designated *system* partition and an *accessible*
partition (sets of subsystem indices in the ``HilbertSpace`` §2 ordering).

The recoverability / residual-information measure is the **coherent
information** (Schumacher & Nielsen, ~1996), ratified 2026-06-02 as the §20
convention:

    recoverability(S | A) = max(0, S(ρ_A) − S(ρ_{S∪A}))   [bits]

i.e. the clamped coherent information ``I_c(S⟩A)``. It is a lower bound on the
quantum information about the system that is recoverable from the accessible
subsystems ``A``:

- **perfect recovery** — when ``S`` is maximally entangled with ``A`` (``ρ_{SA}``
  pure), it equals the full system entropy ``H_S = S(ρ_S)``;
- **full decoherence** — when ``A`` is uncorrelated with ``S``, the coherent
  information is ``≤ 0`` and the (clamped) measure is ``0``;
- it is monotone between those limits as a channel is dephased.

Like the Darwinism measures it is **nonlinear in ρ** (von Neumann entropies of
reduced states) — apply it as post-processing on states retained under
``StorageMode.EAGER``.

Conventions: CONVENTIONS.md §20 (staged for the shared v0.3 Convention Freeze —
see ``WP/FREEZE-v0.3.md``), ratified to the coherent-information measure on
2026-06-02 (WP-01 §5; Schumacher–Nielsen).
"""

from __future__ import annotations

from collections.abc import Sequence

import qutip

from ..hilbert import HilbertSpace
from ._common import (
    _ensure_density,
    _validate_indices,
    _validate_state_dim,
    _von_neumann_entropy_bits,
)


def recoverability(
    state: qutip.Qobj,
    *,
    hilbert: HilbertSpace,
    system_indices: Sequence[int],
    accessible_indices: Sequence[int],
) -> float:
    """Clamped coherent information ``max(0, S(ρ_A) − S(ρ_{S∪A}))`` in bits.

    Parameters
    ----------
    state
        A QuTiP state — ket or density matrix — on ``hilbert``.
    hilbert
        :class:`HilbertSpace` fixing the subsystem ordering (§2). Subsystem
        indices run ``0 .. n_ions + n_modes − 1``.
    system_indices, accessible_indices
        Disjoint sets of subsystem indices forming the system ``S`` and the
        accessible part ``A``.

    Returns
    -------
    float
        ``max(0, S(ρ_A) − S(ρ_{S∪A}))`` in bits: ``H_S`` at perfect recovery,
        ``0`` at full decoherence, monotone between.

    Raises
    ------
    ValueError
        If the index sets are empty, out of range, contain repeats, or
        overlap.
    """
    rho = _ensure_density(state)
    _validate_state_dim(rho, hilbert, "recoverability")
    sys_idx = _validate_indices(system_indices, hilbert, "system_indices")
    acc_idx = _validate_indices(accessible_indices, hilbert, "accessible_indices")
    if set(sys_idx) & set(acc_idx):
        raise ValueError(
            "recoverability: system_indices and accessible_indices must be "
            f"disjoint; got {system_indices} and {accessible_indices}"
        )
    s_accessible = _von_neumann_entropy_bits(rho.ptrace(acc_idx))
    s_joint = _von_neumann_entropy_bits(rho.ptrace(sorted(set(sys_idx) | set(acc_idx))))
    coherent_information = s_accessible - s_joint
    return max(0.0, coherent_information)


__all__ = ["recoverability"]
