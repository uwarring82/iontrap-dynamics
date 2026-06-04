# SPDX-License-Identifier: MIT
"""Shared constructors for the WP-01 information / states tests.

Importable because ``tests`` is on the pytest ``pythonpath`` (pyproject
``[tool.pytest.ini_options]``). Not a test module (no ``test_`` prefix), so
pytest does not collect it. Keeps the spin-only Hilbert space, the collective
``J_z`` generator, and the product-``|+⟩`` probe in one place instead of
re-deriving them in every test file.
"""

from __future__ import annotations

import numpy as np
import qutip

from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.operators import sigma_z_ion, spin_down, spin_up
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem


def _spin_hilbert(n_ions: int) -> HilbertSpace:
    """A spin-only Hilbert space of ``n_ions`` qubits (no motional modes)."""
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})


def _single_mode_hilbert(fock_dim: int, *, label: str = "b") -> HilbertSpace:
    """A one-ion, one-motional-mode Hilbert space (spin ⊗ Fock(``fock_dim``)).

    The mode is axial (unit eigenvector along z, satisfying the §11 per-mode
    normalisation) at 1 MHz. Shared by the WP-02 motional-channel tests.
    """
    mode = ModeConfig(
        label=label,
        frequency_rad_s=2.0 * np.pi * 1.0e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={label: fock_dim})


def _two_mode_hilbert(fock_dim: int, *, labels: tuple[str, str] = ("a", "b")) -> HilbertSpace:
    """A one-ion, two-motional-mode Hilbert space (spin ⊗ Fock ⊗ Fock).

    Both modes axial (unit eigenvector along z, §11 normalisation) at slightly
    different frequencies, each truncated to ``fock_dim``. Shared by the WP-02
    two-mode SU(1,1) squeezing tests.
    """
    eigenvector = np.array([[0.0, 0.0, 1.0]])
    modes = (
        ModeConfig(
            label=labels[0], frequency_rad_s=2.0 * np.pi * 1.0e6, eigenvector_per_ion=eigenvector
        ),
        ModeConfig(
            label=labels[1], frequency_rad_s=2.0 * np.pi * 1.1e6, eigenvector_per_ion=eigenvector
        ),
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=modes)
    return HilbertSpace(system=system, fock_truncations={labels[0]: fock_dim, labels[1]: fock_dim})


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
