# SPDX-License-Identifier: MIT
"""Shared constructors for the WP-01 information / states tests.

Importable because ``tests`` is on the pytest ``pythonpath`` (pyproject
``[tool.pytest.ini_options]``). Not a test module (no ``test_`` prefix), so
pytest does not collect it. Keeps the spin-only Hilbert space, the collective
``J_z`` generator, and the product-``|+⟩`` probe in one place instead of
re-deriving them in every test file.
"""

from __future__ import annotations

import qutip

from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.operators import sigma_z_ion, spin_down, spin_up
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem


def _spin_hilbert(n_ions: int) -> HilbertSpace:
    """A spin-only Hilbert space of ``n_ions`` qubits (no motional modes)."""
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})


def _collective_jz(hilbert: HilbertSpace) -> qutip.Qobj:
    """The collective generator ``J_z = 0.5 * sum_i sigma_z`` on the spins."""
    ops = [hilbert.spin_op_for_ion(sigma_z_ion(), i) for i in range(hilbert.n_ions)]
    total = ops[0]
    for op in ops[1:]:
        total = total + op
    return 0.5 * total


def _product_plus(n_ions: int) -> qutip.Qobj:
    """The product probe ``|+>^{tensor n}`` with ``|+> = (|up> + |down>)/sqrt(2)``."""
    plus = (spin_up() + spin_down()).unit()
    return qutip.tensor([plus] * n_ions)
