# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.spectrum_observables`."""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics import (
    ConventionError,
    SpectrumResult,
    effective_dimension,
    eth_diagonal,
    inverse_participation_ratio,
    phonon_number_diagonals,
    solve_spectrum,
)
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.operators import sigma_z_ion
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem


def _identity_spectrum(dimension: int) -> SpectrumResult:
    return SpectrumResult(
        metadata=solve_spectrum(np.diag(np.arange(dimension, dtype=np.float64))).metadata,
        eigenvalues=np.arange(dimension, dtype=np.float64),
        eigenvectors=np.eye(dimension, dtype=np.complex128),
    )


def _single_ion_hilbert(*, fock: int = 4) -> HilbertSpace:
    mode = ModeConfig(
        label="axial",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={"axial": fock})


class TestParticipationMetrics:
    def test_single_eigenstate_has_unit_ipr_and_deff_one(self) -> None:
        spectrum = _identity_spectrum(4)
        initial_state = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)

        assert inverse_participation_ratio(spectrum, initial_state) == pytest.approx(1.0)
        assert effective_dimension(spectrum, initial_state) == pytest.approx(1.0)

    def test_uniform_superposition_over_d_states_has_ipr_one_over_d(self) -> None:
        spectrum = _identity_spectrum(4)
        initial_state = np.ones(4, dtype=np.complex128)

        assert inverse_participation_ratio(spectrum, initial_state) == pytest.approx(0.25)
        assert effective_dimension(spectrum, initial_state) == pytest.approx(4.0)

    def test_loader_backed_spectrum_is_supported(self) -> None:
        basis = np.eye(3, dtype=np.complex128)
        dense = _identity_spectrum(3)
        loader_backed = SpectrumResult(
            metadata=dense.metadata,
            eigenvalues=dense.eigenvalues,
            eigenvectors_loader=lambda i: basis[:, i],
        )
        initial_state = np.array([1.0, 1.0, 0.0], dtype=np.complex128)

        assert inverse_participation_ratio(loader_backed, initial_state) == pytest.approx(0.5)

    def test_density_matrix_input_is_supported(self) -> None:
        spectrum = _identity_spectrum(2)
        rho = np.diag([0.25, 0.75]).astype(np.complex128)

        assert inverse_participation_ratio(spectrum, rho) == pytest.approx(0.25**2 + 0.75**2)
        assert effective_dimension(spectrum, rho) == pytest.approx(1.0 / (0.25**2 + 0.75**2))


class TestEthDiagonals:
    def test_eth_diagonal_matches_number_operator_in_harmonic_basis(self) -> None:
        spectrum = _identity_spectrum(5)
        number = qutip.num(5)

        np.testing.assert_allclose(eth_diagonal(spectrum, number), np.arange(5, dtype=np.float64))

    def test_non_hermitian_operator_rejected(self) -> None:
        spectrum = _identity_spectrum(2)
        with pytest.raises(ConventionError, match="Hermitian"):
            eth_diagonal(spectrum, np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128))

    def test_operator_dimension_mismatch_rejected(self) -> None:
        spectrum = _identity_spectrum(3)
        with pytest.raises(ConventionError, match="does not match spectrum dimension"):
            eth_diagonal(spectrum, np.eye(2, dtype=np.float64))


class TestPhononNumberDiagonals:
    def test_single_mode_diagonals_match_legacy_nphdiag_cell_formula(self) -> None:
        hilbert = _single_ion_hilbert(fock=4)
        sigma_z = hilbert.spin_op_for_ion(sigma_z_ion(), 0)
        number = hilbert.number_for_mode("axial")
        hamiltonian = 0.3 * sigma_z + 1.7 * number

        beta = 0.8
        rho_mode = np.diag(np.exp(-beta * np.arange(4, dtype=np.float64)))
        rho_mode = rho_mode / np.trace(rho_mode)
        rho_spin = 0.5 * np.eye(2, dtype=np.complex128)
        rho0 = qutip.Qobj(np.kron(rho_spin, rho_mode), dims=hamiltonian.dims)

        spectrum = solve_spectrum(hamiltonian, initial_state=rho0, fock_truncations={"axial": 4})
        diagonals = phonon_number_diagonals(spectrum, hilbert)

        np.testing.assert_allclose(diagonals["axial"], np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))

    def test_mode_subset_can_be_requested(self) -> None:
        hilbert = _single_ion_hilbert(fock=3)
        spectrum = solve_spectrum(hilbert.number_for_mode("axial"))

        diagonals = phonon_number_diagonals(spectrum, hilbert, mode_labels=("axial",))

        assert tuple(diagonals.keys()) == ("axial",)
