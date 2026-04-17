# SPDX-License-Identifier: MIT
"""Runtime convention-enforcement tests for the canonical Pauli operators.

CONVENTIONS.md §3 mandates the atomic-physics Pauli convention:

    |↓⟩ ≡ basis(2, 0),   |↑⟩ ≡ basis(2, 1)
    σ_z_ion |↓⟩ = −|↓⟩,  σ_z_ion |↑⟩ = +|↑⟩
    σ_+_ion |↓⟩ = |↑⟩,   σ_−_ion |↑⟩ = |↓⟩

These tests verify that the operators returned by
:mod:`iontrap_dynamics.operators` actually satisfy those equations at
runtime, in addition to the static scan in
``tests/conventions/test_static_conventions.py`` that bans the raw
``qutip.sigmaz`` import.

The pair of tests — static ban + runtime equations — catches both
failure modes: writing the banned import (static test fires) AND the
operator returned under the library's own name having the wrong sign
(runtime test fires).
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics.operators import (
    sigma_minus_ion,
    sigma_plus_ion,
    sigma_x_ion,
    sigma_y_ion,
    sigma_z_ion,
    spin_down,
    spin_up,
)

pytestmark = pytest.mark.convention


# ----------------------------------------------------------------------------
# Basis kets
# ----------------------------------------------------------------------------


class TestSpinBasis:
    def test_down_is_basis_2_0(self) -> None:
        assert spin_down() == qutip.basis(2, 0)

    def test_up_is_basis_2_1(self) -> None:
        assert spin_up() == qutip.basis(2, 1)

    def test_down_is_normalised(self) -> None:
        assert spin_down().norm() == pytest.approx(1.0)

    def test_up_is_normalised(self) -> None:
        assert spin_up().norm() == pytest.approx(1.0)

    def test_down_and_up_are_orthogonal(self) -> None:
        assert abs(spin_down().overlap(spin_up())) < 1e-15


# ----------------------------------------------------------------------------
# σ_z_ion — the principal sign-flip (CONVENTIONS.md §3)
# ----------------------------------------------------------------------------


class TestSigmaZion:
    def test_sigma_z_ion_acts_on_down_with_eigenvalue_minus_one(self) -> None:
        """σ_z_ion |↓⟩ = −|↓⟩ per CONVENTIONS.md §3.

        This is the headline sign-flip: QuTiP's native ``sigmaz`` would
        give +|↓⟩, violating the atomic-physics convention. Using the
        library's ``sigma_z_ion`` must give the physics-correct sign.
        """
        eigval = qutip.expect(sigma_z_ion(), spin_down())
        assert eigval == pytest.approx(-1.0)

    def test_sigma_z_ion_acts_on_up_with_eigenvalue_plus_one(self) -> None:
        """σ_z_ion |↑⟩ = +|↑⟩."""
        eigval = qutip.expect(sigma_z_ion(), spin_up())
        assert eigval == pytest.approx(+1.0)

    def test_sigma_z_ion_is_sign_flipped_qutip_sigmaz(self) -> None:
        """Explicit sign-flip check: σ_z_ion = −qutip.sigmaz()."""
        diff = sigma_z_ion() + qutip.sigmaz()
        assert diff.norm() < 1e-15

    def test_sigma_z_ion_matrix_form(self) -> None:
        """In ordered basis (|↓⟩, |↑⟩) = (basis(2,0), basis(2,1)), σ_z_ion
        is ``diag(−1, +1)``."""
        matrix = sigma_z_ion().full()
        np.testing.assert_allclose(matrix, np.diag([-1.0, +1.0]))


# ----------------------------------------------------------------------------
# σ_+_ion, σ_−_ion — ladder operators
# ----------------------------------------------------------------------------


class TestLadderOperators:
    def test_sigma_plus_raises_down_to_up(self) -> None:
        """σ_+_ion |↓⟩ = |↑⟩ per CONVENTIONS.md §3."""
        result = sigma_plus_ion() * spin_down()
        assert (result - spin_up()).norm() < 1e-15

    def test_sigma_plus_annihilates_up(self) -> None:
        """σ_+_ion |↑⟩ = 0 (cannot raise beyond |↑⟩)."""
        result = sigma_plus_ion() * spin_up()
        assert result.norm() < 1e-15

    def test_sigma_minus_lowers_up_to_down(self) -> None:
        """σ_−_ion |↑⟩ = |↓⟩."""
        result = sigma_minus_ion() * spin_up()
        assert (result - spin_down()).norm() < 1e-15

    def test_sigma_minus_annihilates_down(self) -> None:
        """σ_−_ion |↓⟩ = 0 (cannot lower below |↓⟩)."""
        result = sigma_minus_ion() * spin_down()
        assert result.norm() < 1e-15

    def test_sigma_plus_is_hermitian_conjugate_of_sigma_minus(self) -> None:
        """σ_+_ion = (σ_−_ion)†."""
        diff = sigma_plus_ion() - sigma_minus_ion().dag()
        assert diff.norm() < 1e-15


# ----------------------------------------------------------------------------
# σ_x_ion and σ_y_ion — derived forms
# ----------------------------------------------------------------------------


class TestSigmaXY:
    def test_sigma_x_equals_sigma_plus_minus_sum(self) -> None:
        """σ_x_ion = σ_+_ion + σ_−_ion."""
        diff = sigma_x_ion() - (sigma_plus_ion() + sigma_minus_ion())
        assert diff.norm() < 1e-15

    def test_sigma_x_ion_is_unchanged_vs_qutip(self) -> None:
        """σ_x is basis-flip-invariant: σ_x_ion = qutip.sigmax()."""
        diff = sigma_x_ion() - qutip.sigmax()
        assert diff.norm() < 1e-15

    def test_sigma_y_equals_minus_i_ladder_difference(self) -> None:
        """σ_y_ion = −i(σ_+_ion − σ_−_ion)."""
        diff = sigma_y_ion() - (-1j) * (sigma_plus_ion() - sigma_minus_ion())
        assert diff.norm() < 1e-15

    def test_sigma_y_ion_is_sign_flipped_qutip_sigmay(self) -> None:
        """σ_y_ion = −qutip.sigmay() under the basis mapping."""
        diff = sigma_y_ion() + qutip.sigmay()
        assert diff.norm() < 1e-15


# ----------------------------------------------------------------------------
# SU(2) algebra — commutation relations [σ_i, σ_j] = 2iε_{ijk} σ_k
# ----------------------------------------------------------------------------


class TestCommutators:
    def test_xy_commutator(self) -> None:
        """[σ_x, σ_y] = 2i σ_z."""
        lhs = sigma_x_ion() * sigma_y_ion() - sigma_y_ion() * sigma_x_ion()
        rhs = 2j * sigma_z_ion()
        assert (lhs - rhs).norm() < 1e-14

    def test_yz_commutator(self) -> None:
        """[σ_y, σ_z] = 2i σ_x."""
        lhs = sigma_y_ion() * sigma_z_ion() - sigma_z_ion() * sigma_y_ion()
        rhs = 2j * sigma_x_ion()
        assert (lhs - rhs).norm() < 1e-14

    def test_zx_commutator(self) -> None:
        """[σ_z, σ_x] = 2i σ_y."""
        lhs = sigma_z_ion() * sigma_x_ion() - sigma_x_ion() * sigma_z_ion()
        rhs = 2j * sigma_y_ion()
        assert (lhs - rhs).norm() < 1e-14

    def test_pauli_squares_are_identity(self) -> None:
        """σ_i² = I for i ∈ {x, y, z}."""
        identity = qutip.qeye(2)
        for op, name in [
            (sigma_x_ion(), "x"),
            (sigma_y_ion(), "y"),
            (sigma_z_ion(), "z"),
        ]:
            diff = (op * op) - identity
            assert diff.norm() < 1e-14, f"σ_{name}² ≠ I"


# ----------------------------------------------------------------------------
# Fresh-instance contract: each call returns a new Qobj, matching QuTiP's idiom
# ----------------------------------------------------------------------------


class TestInstanceContract:
    def test_sigma_z_returns_fresh_instance_each_call(self) -> None:
        """The API mirrors qutip.sigmax / sigmaz — each call returns a new
        Qobj, so downstream code can compose without aliasing risk."""
        a = sigma_z_ion()
        b = sigma_z_ion()
        assert a is not b
        assert (a - b).norm() < 1e-15
