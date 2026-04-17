# SPDX-License-Identifier: MIT
"""Canonical single-ion Pauli operators and spin-basis kets.

Implements ``CONVENTIONS.md`` §3 — the atomic-physics Pauli convention
under the basis labelling ``|↓⟩ ≡ basis(2, 0)``, ``|↑⟩ ≡ basis(2, 1)``.
Every Hamiltonian builder, state-prep routine, and observable in the
library should reach for its Pauli operators through this module rather
than through QuTiP's native ``sigmaz``, which has the opposite sign
under our basis mapping (see §3 for the derivation).

Design
------

Every operator is derived from first principles from the two basis
kets, matching the CONVENTIONS.md §3 definitions verbatim:

    σ_+ ≡ |↑⟩⟨↓|,    σ_− ≡ |↓⟩⟨↑|
    σ_z ≡ |↑⟩⟨↑| − |↓⟩⟨↓|
    σ_x = σ_+ + σ_−
    σ_y = −i(σ_+ − σ_−)

This means the module never touches ``qutip.sigmaz``, ``qutip.sigmay``,
``qutip.sigmap``, or ``qutip.sigmam``. The one remaining QuTiP
dependency is ``qutip.basis(2, k)`` for the kets themselves, which is
convention-neutral. As a result, ``operators.py`` contains the
CONVENTIONS.md §3 definitions as executable code — the definitions are
not rewrapped QuTiP operators, they ARE the operators.

Correspondence with QuTiP (for reference only — these mappings MUST NOT
be used directly in library code; use the functions below):

===================  ================================
This module          Equivalent raw QuTiP expression
===================  ================================
``sigma_x_ion()``    ``qutip.sigmax()``              (unchanged)
``sigma_y_ion()``    ``-qutip.sigmay()``             (sign flip)
``sigma_z_ion()``    ``-qutip.sigmaz()``             (sign flip)
``sigma_plus_ion()`` ``qutip.sigmam()``              (operators swap)
``sigma_minus_ion()`` ``qutip.sigmap()``             (operators swap)
===================  ================================

API
---

Operators are returned as fresh :class:`qutip.Qobj` instances from each
call, matching QuTiP's own idiom (``qutip.sigmax()``, not a module-level
constant). Tensor-product builders over a (spins, modes) Hilbert space
will live in a separate ``hilbert.py`` / ``system.py`` module in Phase
1+.
"""

from __future__ import annotations

import qutip

# ----------------------------------------------------------------------------
# Spin-basis kets (CONVENTIONS.md §3)
# ----------------------------------------------------------------------------


def spin_down() -> qutip.Qobj:
    """Return |↓⟩ — the physical ground state.

    Equivalent to ``qutip.basis(2, 0)``. Under the atomic-physics Pauli
    convention, σ_z_ion |↓⟩ = −|↓⟩.
    """
    return qutip.basis(2, 0)


def spin_up() -> qutip.Qobj:
    """Return |↑⟩ — the physical excited state.

    Equivalent to ``qutip.basis(2, 1)``. Under the atomic-physics Pauli
    convention, σ_z_ion |↑⟩ = +|↑⟩.
    """
    return qutip.basis(2, 1)


# ----------------------------------------------------------------------------
# Ladder operators — first-principles from the basis kets
# ----------------------------------------------------------------------------


def sigma_plus_ion() -> qutip.Qobj:
    """Return σ_+_ion ≡ |↑⟩⟨↓|.

    Raises |↓⟩ to |↑⟩ and annihilates |↑⟩ — direct transcription of the
    CONVENTIONS.md §3 definition.
    """
    up = spin_up()
    down = spin_down()
    return up * down.dag()


def sigma_minus_ion() -> qutip.Qobj:
    """Return σ_−_ion ≡ |↓⟩⟨↑|.

    Lowers |↑⟩ to |↓⟩ and annihilates |↓⟩.
    """
    up = spin_up()
    down = spin_down()
    return down * up.dag()


# ----------------------------------------------------------------------------
# Pauli operators — derived from the ladder and basis definitions
# ----------------------------------------------------------------------------


def sigma_z_ion() -> qutip.Qobj:
    """Return σ_z_ion ≡ |↑⟩⟨↑| − |↓⟩⟨↓|.

    In the ordered basis (|↓⟩, |↑⟩) the matrix form is ``diag(−1, +1)``.
    This is the operator CONVENTIONS.md §3 mandates; ``qutip.sigmaz()``
    has the opposite sign under our basis mapping and is banned in
    library code (static check in
    ``tests/conventions/test_static_conventions.py``).
    """
    up = spin_up()
    down = spin_down()
    return up * up.dag() - down * down.dag()


def sigma_x_ion() -> qutip.Qobj:
    """Return σ_x_ion ≡ σ_+_ion + σ_−_ion.

    Coincides with :func:`qutip.sigmax` (σ_x is basis-flip-invariant),
    but built here from the §3 ladder definitions so the whole module
    reads as a single consistent derivation.
    """
    return sigma_plus_ion() + sigma_minus_ion()


def sigma_y_ion() -> qutip.Qobj:
    """Return σ_y_ion ≡ −i(σ_+_ion − σ_−_ion).

    Equals ``-qutip.sigmay()`` under the basis mapping; built here from
    the §3 ladder definitions rather than QuTiP's primitive so the
    sign is self-evident.
    """
    return -1j * (sigma_plus_ion() - sigma_minus_ion())


__all__ = [
    "sigma_minus_ion",
    "sigma_plus_ion",
    "sigma_x_ion",
    "sigma_y_ion",
    "sigma_z_ion",
    "spin_down",
    "spin_up",
]
