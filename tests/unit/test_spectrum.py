# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.spectrum`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import qutip
import scipy

from iontrap_dynamics import (
    CONVENTION_VERSION,
    ConventionError,
    Result,
    SpectrumResult,
    solve_spectrum,
)
from iontrap_dynamics.spectrum import SpectrumMetadata


def _metadata() -> SpectrumMetadata:
    return SpectrumMetadata(
        convention_version=CONVENTION_VERSION,
        request_hash="0" * 64,
        backend_name="test-spectrum",
        backend_version="0.0.0",
    )


def _harmonic_oscillator_hamiltonian(dim: int = 10, *, spacing: float = 2.5) -> qutip.Qobj:
    return spacing * qutip.num(dim)


class TestSpectrumResultSchema:
    def test_result_is_frozen_and_part_of_result_family(self) -> None:
        result = SpectrumResult(
            metadata=_metadata(),
            eigenvalues=np.array([0.0, 1.0]),
            eigenvectors=np.eye(2, dtype=np.complex128),
        )
        assert isinstance(result, Result)
        with pytest.raises(FrozenInstanceError):
            result.method = "shift_invert"  # type: ignore[misc]

    def test_requires_one_eigenvector_source(self) -> None:
        with pytest.raises(ConventionError, match="requires either"):
            SpectrumResult(
                metadata=_metadata(),
                eigenvalues=np.array([0.0, 1.0]),
            )

    def test_materialized_vectors_and_loader_are_mutually_exclusive(self) -> None:
        with pytest.raises(ConventionError, match="forbids setting both"):
            SpectrumResult(
                metadata=_metadata(),
                eigenvalues=np.array([0.0, 1.0]),
                eigenvectors=np.eye(2, dtype=np.complex128),
                eigenvectors_loader=lambda i: np.array([1.0, 0.0], dtype=np.complex128),
            )

    def test_vector_column_count_must_match_eigenvalue_count(self) -> None:
        with pytest.raises(ConventionError, match="columns must match"):
            SpectrumResult(
                metadata=_metadata(),
                eigenvalues=np.array([0.0, 1.0]),
                eigenvectors=np.ones((2, 3), dtype=np.complex128),
            )


class TestSolveSpectrum:
    def test_dense_harmonic_oscillator_spectrum_matches_known_values(self) -> None:
        spacing = 1.7
        H = _harmonic_oscillator_hamiltonian(10, spacing=spacing)

        result = solve_spectrum(
            H,
            request_hash="a" * 64,
            fock_truncations={"axial": 10},
            provenance_tags=("unit-test",),
        )

        assert isinstance(result, SpectrumResult)
        np.testing.assert_allclose(result.eigenvalues, spacing * np.arange(10))
        assert result.eigenvectors is not None
        assert result.eigenvectors.shape == (10, 10)
        assert result.method == "dense"
        assert result.metadata.backend_name == "spectrum-scipy"
        assert result.metadata.backend_version == scipy.__version__
        assert result.metadata.request_hash == "a" * 64
        assert result.metadata.fock_truncations == {"axial": 10}
        assert result.metadata.provenance_tags == ("unit-test",)

    def test_accepts_dense_numpy_arrays(self) -> None:
        matrix = np.diag(np.array([0.0, 2.0, 5.0], dtype=np.float64))
        result = solve_spectrum(matrix)
        np.testing.assert_allclose(result.eigenvalues, [0.0, 2.0, 5.0])

    def test_records_initial_state_energy_moments_for_ket(self) -> None:
        H = _harmonic_oscillator_hamiltonian(6, spacing=0.75)
        psi = qutip.basis(6, 3)

        result = solve_spectrum(H, initial_state=psi)

        assert result.initial_state_mean_energy == pytest.approx(3 * 0.75)
        assert result.initial_state_energy_std == pytest.approx(0.0)

    def test_records_initial_state_energy_moments_for_density_matrix(self) -> None:
        H = _harmonic_oscillator_hamiltonian(4, spacing=2.0)
        rho = 0.25 * qutip.ket2dm(qutip.basis(4, 0)) + 0.75 * qutip.ket2dm(qutip.basis(4, 2))

        result = solve_spectrum(H, initial_state=rho)

        assert result.initial_state_mean_energy == pytest.approx(3.0)
        assert result.initial_state_energy_std == pytest.approx(np.sqrt(3.0))

    def test_non_hermitian_hamiltonian_rejected(self) -> None:
        with pytest.raises(ConventionError, match="Hermitian"):
            solve_spectrum(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128))

    def test_non_square_hamiltonian_rejected(self) -> None:
        with pytest.raises(ConventionError, match="square operator"):
            solve_spectrum(np.ones((2, 3), dtype=np.float64))

    def test_unsupported_method_rejected(self) -> None:
        with pytest.raises(ConventionError, match="not implemented yet"):
            solve_spectrum(np.diag([0.0, 1.0]), method="shift_invert")

    def test_initial_state_dimension_mismatch_rejected(self) -> None:
        with pytest.raises(ConventionError, match="expected 4"):
            solve_spectrum(np.diag([0.0, 1.0, 2.0, 3.0]), initial_state=np.array([1.0, 0.0]))
