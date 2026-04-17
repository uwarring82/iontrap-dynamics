# SPDX-License-Identifier: MIT
"""State-preparation helpers for a :class:`HilbertSpace`.

Two entry points:

- :func:`ground_state` — the pure ket with every spin in |↓⟩ and every mode
  in |0⟩. The "cold start" used by most builders as a reference initial
  condition.
- :func:`compose_density` — the general composition: given one state per
  ion and one state per mode, return the full-space density matrix.

Per-subsystem state construction is delegated to QuTiP. The library does
not re-wrap ``qutip.thermal_dm``, ``qutip.coherent``, ``qutip.squeeze``,
``qutip.fock_dm`` etc. — those are stable, well-documented primitives
that accept a dimension and return a Qobj of the right shape. This module
only handles the tensor-product ordering (CONVENTIONS.md §2: spins first,
then modes, left-to-right in the order they appear in ``system.modes``).

Design notes
------------

- All returns are QuTiP ``Qobj`` instances on the Hilbert space specified
  by the passed :class:`HilbertSpace`. ``ground_state`` returns a ket;
  ``compose_density`` always returns a density matrix (converting ket
  inputs via ``ket2dm`` as needed, so a mix of pure and mixed inputs
  composes uniformly).
- Every dimension check raises :class:`ConventionError` on mismatch —
  CONVENTIONS.md §15 Level 3 hard failures.
- No caching — state construction is cheap relative to solve time and
  callers typically build one state per simulation.

Example
-------

Reproducing the Phase 0.F scenario 1 (single-ion carrier flopping from
thermal motion at ``n_bar = 0.5``)::

    from iontrap_dynamics.states import compose_density
    from iontrap_dynamics.operators import spin_down
    import qutip

    rho_0 = compose_density(
        hilbert,
        spin_states_per_ion=[spin_down()],
        mode_states_by_label={"axial": qutip.thermal_dm(30, n=0.5)},
    )
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import qutip

from .exceptions import ConventionError
from .hilbert import HilbertSpace
from .operators import spin_down

# ----------------------------------------------------------------------------
# Ground state — canonical cold start
# ----------------------------------------------------------------------------


def ground_state(hilbert: HilbertSpace) -> qutip.Qobj:
    """Return the pure ket ``|↓⟩^⊗N ⊗ |0⟩^⊗M`` on ``hilbert``.

    Every ion in |↓⟩ (atomic ground state per CONVENTIONS.md §3), every
    mode in the motional vacuum |0⟩. Dimensions follow the §2 tensor
    ordering baked into :class:`HilbertSpace`.

    Returned ket is normalised by construction.
    """
    subsystems: list[qutip.Qobj] = [spin_down() for _ in range(hilbert.n_ions)]
    for mode in hilbert.system.modes:
        subsystems.append(qutip.basis(hilbert.mode_dim(mode.label), 0))
    return qutip.tensor(*subsystems)


# ----------------------------------------------------------------------------
# General composition — from per-subsystem states to a full density matrix
# ----------------------------------------------------------------------------


def _to_density_matrix(state: qutip.Qobj, expected_dim: int, context: str) -> qutip.Qobj:
    """Convert a ket to a density matrix if needed; validate dimension."""
    if state.dims[0] != [expected_dim]:
        raise ConventionError(
            f"{context}: state has dims {state.dims}, expected single "
            f"subsystem of dimension {expected_dim}."
        )

    if state.isket:
        return qutip.ket2dm(state)
    if state.isoper:
        if state.dims[1] != [expected_dim]:
            raise ConventionError(
                f"{context}: density-matrix operator has non-square dims "
                f"{state.dims}; expected [[{expected_dim}], [{expected_dim}]]."
            )
        return state
    raise ConventionError(
        f"{context}: expected a ket or a density-matrix operator; got a Qobj "
        f"of type {state.type!r}."
    )


def compose_density(
    hilbert: HilbertSpace,
    *,
    spin_states_per_ion: Sequence[qutip.Qobj],
    mode_states_by_label: Mapping[str, qutip.Qobj],
) -> qutip.Qobj:
    """Compose a full-space density matrix from per-subsystem states.

    Parameters
    ----------
    hilbert
        The :class:`HilbertSpace` the composed state lives on.
    spin_states_per_ion
        Sequence of one state per ion, in ion-index order. Each element
        may be a ket (dim 2) or a ``2 × 2`` density matrix; kets are
        converted to density matrices via ``ket2dm``.
    mode_states_by_label
        Mapping from mode label to per-mode state. Every mode in
        ``hilbert.system.modes`` must have an entry; no extras are
        permitted. Each value may be a ket or a density matrix of the
        mode's declared Fock truncation.

    Returns
    -------
    qutip.Qobj
        Density matrix on the full tensor-product space, with dims
        matching :meth:`HilbertSpace.qutip_dims`.

    Raises
    ------
    ConventionError
        If the number of spin states does not match ``hilbert.n_ions``,
        if a mode entry is missing or extra, or if any per-subsystem
        state has the wrong dimension or type.
    """
    if len(spin_states_per_ion) != hilbert.n_ions:
        raise ConventionError(
            f"spin_states_per_ion has {len(spin_states_per_ion)} entries; "
            f"expected {hilbert.n_ions} (one per ion)."
        )

    system_mode_labels = {m.label for m in hilbert.system.modes}
    given_labels = set(mode_states_by_label.keys())
    missing = system_mode_labels - given_labels
    extra = given_labels - system_mode_labels
    if missing:
        raise ConventionError(
            f"mode_states_by_label is missing entries for modes: {sorted(missing)}."
        )
    if extra:
        raise ConventionError(
            f"mode_states_by_label has entries for unknown modes: {sorted(extra)}."
        )

    subsystems: list[qutip.Qobj] = []
    for i, spin in enumerate(spin_states_per_ion):
        subsystems.append(
            _to_density_matrix(spin, hilbert.spin_dim, context=f"spin_states_per_ion[{i}]")
        )
    for mode in hilbert.system.modes:
        cutoff = hilbert.mode_dim(mode.label)
        subsystems.append(
            _to_density_matrix(
                mode_states_by_label[mode.label],
                cutoff,
                context=f"mode_states_by_label[{mode.label!r}]",
            )
        )

    return qutip.tensor(*subsystems)


__all__ = [
    "compose_density",
    "ground_state",
]
