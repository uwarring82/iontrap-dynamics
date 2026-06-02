# SPDX-License-Identifier: MIT
"""Unit tests for the GHZ and cat-state factories (WI-3 of WP-01, dispatch EDC).

Covers states.ghz_state and states.cat_mode: normalisation, the GHZ population
structure and maximal entanglement, the cat parity eigenvalue and Fock support,
and the ConventionError guards. Includes a cross-check that ghz_state feeds the
WI-1 quantum Fisher information to the Heisenberg limit F_Q = N^2 (a miniature
of the keystone benchmark).
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics import (
    ConventionError,
    cat_mode,
    concurrence_trajectory,
    ghz_state,
    quantum_fisher_information_trajectory,
)
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.operators import sigma_z_ion, spin_down, spin_up
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem


def _spin_hilbert(n_ions: int) -> HilbertSpace:
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})


def _collective_jz(h: HilbertSpace) -> qutip.Qobj:
    ops = [h.spin_op_for_ion(sigma_z_ion(), i) for i in range(h.n_ions)]
    total = ops[0]
    for op in ops[1:]:
        total = total + op
    return 0.5 * total


def _parity_operator(fock_dim: int) -> qutip.Qobj:
    return qutip.Qobj(np.diag([(-1.0) ** n for n in range(fock_dim)]))


# ----------------------------------------------------------------------------
# GHZ
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("n_ions", [1, 2, 3])
def test_ghz_state_is_normalised_ket(n_ions: int) -> None:
    psi = ghz_state(_spin_hilbert(n_ions))
    assert psi.isket
    assert psi.norm() == pytest.approx(1.0)


@pytest.mark.parametrize("n_ions", [2, 3])
def test_ghz_population_split_equally_between_all_up_and_all_down(n_ions: int) -> None:
    psi = ghz_state(_spin_hilbert(n_ions))
    all_up = qutip.tensor([spin_up()] * n_ions)
    all_down = qutip.tensor([spin_down()] * n_ions)
    assert abs(all_up.overlap(psi)) ** 2 == pytest.approx(0.5)
    assert abs(all_down.overlap(psi)) ** 2 == pytest.approx(0.5)


def test_ghz_two_ion_is_maximally_entangled() -> None:
    h = _spin_hilbert(2)
    c = concurrence_trajectory([ghz_state(h)], hilbert=h, ion_indices=(0, 1))
    assert c[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n_ions", [1, 2, 3])
def test_ghz_qfi_reaches_heisenberg_limit(n_ions: int) -> None:
    # Cross-check the WI-3 factory against the WI-1 QFI primitive: F_Q = N^2.
    h = _spin_hilbert(n_ions)
    qfi = quantum_fisher_information_trajectory(
        [ghz_state(h)], hilbert=h, generator=_collective_jz(h)
    )
    assert qfi[0] == pytest.approx(float(n_ions**2), abs=1e-9)


# ----------------------------------------------------------------------------
# Cat
# ----------------------------------------------------------------------------


def test_cat_mode_is_normalised_ket() -> None:
    for parity in ("even", "odd"):
        cat = cat_mode(12, 1.2, parity=parity)
        assert cat.isket
        assert cat.norm() == pytest.approx(1.0)


def test_cat_mode_parity_eigenvalue() -> None:
    fock = 12
    parity_op = _parity_operator(fock)
    even = cat_mode(fock, 1.0, parity="even")
    odd = cat_mode(fock, 1.0, parity="odd")
    assert qutip.expect(parity_op, even) == pytest.approx(1.0, abs=1e-8)
    assert qutip.expect(parity_op, odd) == pytest.approx(-1.0, abs=1e-8)


def test_cat_mode_fock_support_is_single_parity() -> None:
    fock = 12
    even = cat_mode(fock, 1.0, parity="even")
    odd = cat_mode(fock, 1.0, parity="odd")
    for n in range(fock):
        amp_even = abs(qutip.basis(fock, n).overlap(even))
        amp_odd = abs(qutip.basis(fock, n).overlap(odd))
        if n % 2 == 0:
            assert amp_odd == pytest.approx(0.0, abs=1e-8)
        else:
            assert amp_even == pytest.approx(0.0, abs=1e-8)


def test_cat_mode_validation() -> None:
    with pytest.raises(ConventionError):
        cat_mode(0, 1.0)
    with pytest.raises(ConventionError):
        cat_mode(8, 1.0, parity="weird")
    # An odd cat at alpha = 0 is the zero vector (degenerate).
    with pytest.raises(ConventionError):
        cat_mode(8, 0.0, parity="odd")
