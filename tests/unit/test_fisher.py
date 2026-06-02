# SPDX-License-Identifier: MIT
"""Unit tests for the Fisher-information surface (WI-1 of WP-01, dispatch EDA).

Covers the four public symbols of ``iontrap_dynamics.information.fisher``:
quantum Fisher information (SLD-QFI) as a trajectory evaluator, classical
Fisher information, the Cramer-Rao bound, and the linear-Gaussian Fisher
matrix. The keystone oracle is the GHZ-vs-product QFI scaling under the
collective generator J_z = 0.5 * sum_i sigma_z_ion(i): F_Q(GHZ) = N^2
(Heisenberg) and F_Q(product) = N (standard quantum limit). The
Braunstein-Caves inequality CFI <= QFI is checked at a saturating and a
sub-optimal measurement.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics import (
    classical_fisher_information,
    cramer_rao_bound,
    linear_gaussian_fisher,
    quantum_fisher_information_trajectory,
)
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.operators import sigma_z_ion, spin_down, spin_up
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

# ----------------------------------------------------------------------------
# Fixtures / builders (spin-only Hilbert spaces; no motional modes)
# ----------------------------------------------------------------------------


def _spin_hilbert(n_ions: int) -> HilbertSpace:
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})


def _ghz(n_ions: int) -> qutip.Qobj:
    """(|up...up> + |down...down>) / sqrt(2)."""
    up = qutip.tensor([spin_up()] * n_ions)
    down = qutip.tensor([spin_down()] * n_ions)
    return (up + down).unit()


def _product_plus(n_ions: int) -> qutip.Qobj:
    """|+>^{tensor n}, with |+> = (|up> + |down>) / sqrt(2)."""
    plus = (spin_up() + spin_down()).unit()
    return qutip.tensor([plus] * n_ions)


def _collective_jz(h: HilbertSpace) -> qutip.Qobj:
    """J_z = 0.5 * sum_i sigma_z on ion i, embedded in the full space."""
    ops = [h.spin_op_for_ion(sigma_z_ion(), i) for i in range(h.n_ions)]
    total = ops[0]
    for op in ops[1:]:
        total = total + op
    return 0.5 * total


# ----------------------------------------------------------------------------
# Quantum Fisher information: the keystone GHZ (N^2) vs product (N) oracle
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("n_ions", [1, 2, 3])
def test_qfi_ghz_reaches_heisenberg_limit(n_ions: int) -> None:
    h = _spin_hilbert(n_ions)
    jz = _collective_jz(h)
    qfi = quantum_fisher_information_trajectory([_ghz(n_ions)], hilbert=h, generator=jz)
    assert qfi.shape == (1,)
    assert qfi[0] == pytest.approx(float(n_ions**2), abs=1e-9)


@pytest.mark.parametrize("n_ions", [1, 2, 3])
def test_qfi_product_reaches_standard_quantum_limit(n_ions: int) -> None:
    h = _spin_hilbert(n_ions)
    jz = _collective_jz(h)
    qfi = quantum_fisher_information_trajectory([_product_plus(n_ions)], hilbert=h, generator=jz)
    assert qfi[0] == pytest.approx(float(n_ions), abs=1e-9)


def test_qfi_pure_state_equals_four_times_variance() -> None:
    h = _spin_hilbert(1)
    sigma_z = h.spin_op_for_ion(sigma_z_ion(), 0)
    plus = (spin_up() + spin_down()).unit()
    # Var(sigma_z) on |+> is 1, so F_Q = 4.
    qfi_plus = quantum_fisher_information_trajectory([plus], hilbert=h, generator=sigma_z)
    assert qfi_plus[0] == pytest.approx(4.0, abs=1e-9)
    # An eigenstate of the generator carries no parameter sensitivity.
    qfi_eig = quantum_fisher_information_trajectory([spin_up()], hilbert=h, generator=sigma_z)
    assert qfi_eig[0] == pytest.approx(0.0, abs=1e-9)


def test_qfi_maximally_mixed_state_is_zero() -> None:
    h = _spin_hilbert(1)
    jz = _collective_jz(h)
    maximally_mixed = qutip.qeye(2) / 2.0
    qfi = quantum_fisher_information_trajectory([maximally_mixed], hilbert=h, generator=jz)
    assert qfi[0] == pytest.approx(0.0, abs=1e-9)


def test_qfi_trajectory_length_and_dimension_check() -> None:
    h = _spin_hilbert(2)
    jz = _collective_jz(h)
    qfi = quantum_fisher_information_trajectory(
        [_ghz(2), _product_plus(2)], hilbert=h, generator=jz
    )
    assert qfi.shape == (2,)
    # A generator that does not match the Hilbert dimension is rejected.
    h_one = _spin_hilbert(1)
    with pytest.raises(ValueError):
        quantum_fisher_information_trajectory([_ghz(1)], hilbert=h_one, generator=jz)


# ----------------------------------------------------------------------------
# Classical Fisher information and the Cramer-Rao bound
# ----------------------------------------------------------------------------


def test_classical_fisher_information_values() -> None:
    assert classical_fisher_information(
        [0.5, 0.5], parameter_derivative=[-0.5, 0.5]
    ) == pytest.approx(1.0)
    assert classical_fisher_information(
        [0.5, 0.5], parameter_derivative=[0.0, 0.0]
    ) == pytest.approx(0.0)
    # A zero-probability outcome contributes nothing.
    assert classical_fisher_information(
        [1.0, 0.0], parameter_derivative=[0.0, 0.0]
    ) == pytest.approx(0.0)


def test_classical_fisher_information_validation() -> None:
    with pytest.raises(ValueError):
        classical_fisher_information([0.5, 0.5], parameter_derivative=[0.1])
    with pytest.raises(ValueError):
        classical_fisher_information([0.6, 0.6], parameter_derivative=[0.0, 0.0])
    with pytest.raises(ValueError):
        classical_fisher_information([-0.1, 1.1], parameter_derivative=[0.0, 0.0])


def test_cfi_never_exceeds_qfi_braunstein_caves() -> None:
    h = _spin_hilbert(1)
    jz = _collective_jz(h)
    plus = (spin_up() + spin_down()).unit()
    qfi = quantum_fisher_information_trajectory([plus], hilbert=h, generator=jz)[0]
    assert qfi == pytest.approx(1.0)
    # Optimal measurement saturates the bound: CFI == QFI.
    cfi_optimal = classical_fisher_information([0.5, 0.5], parameter_derivative=[-0.5, 0.5])
    assert cfi_optimal <= qfi + 1e-9
    assert cfi_optimal == pytest.approx(qfi, abs=1e-9)
    # A sub-optimal (insensitive) measurement: CFI < QFI.
    cfi_suboptimal = classical_fisher_information([0.5, 0.5], parameter_derivative=[0.0, 0.0])
    assert cfi_suboptimal <= qfi + 1e-9


def test_cramer_rao_bound() -> None:
    assert cramer_rao_bound(4.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        cramer_rao_bound(0.0)
    with pytest.raises(ValueError):
        cramer_rao_bound(-1.0)


# ----------------------------------------------------------------------------
# Linear-Gaussian Fisher matrix  F = A^T Sigma^-1 A
# ----------------------------------------------------------------------------


def test_linear_gaussian_fisher_matrix() -> None:
    # Two independent observations of one parameter; F = 1/var1 + 1/var2.
    A = np.array([[1.0], [1.0]])
    sigma = np.diag([4.0, 1.0])
    fisher = linear_gaussian_fisher(A=A, sigma=sigma)
    assert fisher.shape == (1, 1)
    assert fisher[0, 0] == pytest.approx(1.0 / 4.0 + 1.0 / 1.0)
    assert cramer_rao_bound(float(fisher[0, 0])) == pytest.approx(1.0 / 1.25)


def test_linear_gaussian_fisher_validation() -> None:
    with pytest.raises(ValueError):
        linear_gaussian_fisher(A=np.array([1.0, 1.0]), sigma=np.eye(2))
    with pytest.raises(ValueError):
        linear_gaussian_fisher(A=np.array([[1.0], [1.0]]), sigma=np.eye(3))
    with pytest.raises(ValueError):
        linear_gaussian_fisher(A=np.array([[1.0], [1.0]]), sigma=np.array([[1.0, 0.5], [0.6, 1.0]]))
