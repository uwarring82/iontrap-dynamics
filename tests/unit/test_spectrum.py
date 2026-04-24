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

    def test_unknown_backend_name_rejected(self) -> None:
        with pytest.raises(ConventionError, match="not recognised"):
            solve_spectrum(np.diag([0.0, 1.0]), backend_name="spectrum-made-up")

    def test_unknown_device_value_rejected(self) -> None:
        with pytest.raises(ConventionError, match="not recognised"):
            solve_spectrum(np.diag([0.0, 1.0]), backend_name="spectrum-jax", device="tpu")

    def test_device_rejected_with_scipy_backend(self) -> None:
        with pytest.raises(ConventionError, match="only applicable to backend_name='spectrum-jax'"):
            solve_spectrum(np.diag([0.0, 1.0]), device="cpu")


class TestSpectrumJaxBackend:
    """BBA: numeric equivalence and device-dispatch contract for the JAX path."""

    def test_jax_and_scipy_eigenvalues_agree_on_small_hermitian(self) -> None:
        pytest.importorskip("jax")
        rng = np.random.default_rng(0xBBA)
        n = 16
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = 0.5 * (a + a.conj().T)

        scipy_result = solve_spectrum(H, backend_name="spectrum-scipy")
        jax_result = solve_spectrum(H, backend_name="spectrum-jax")

        np.testing.assert_allclose(
            jax_result.eigenvalues, scipy_result.eigenvalues, rtol=1e-10, atol=1e-10
        )

    def test_jax_and_scipy_eigenvector_projectors_agree(self) -> None:
        pytest.importorskip("jax")
        rng = np.random.default_rng(0xBBA1)
        n = 8
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = 0.5 * (a + a.conj().T)

        scipy_result = solve_spectrum(H, backend_name="spectrum-scipy")
        jax_result = solve_spectrum(H, backend_name="spectrum-jax")

        # Compare eigenvector-derived projectors per eigenvalue; phase/sign-insensitive.
        assert scipy_result.eigenvectors is not None
        assert jax_result.eigenvectors is not None
        for column in range(n):
            overlap = np.abs(
                np.vdot(scipy_result.eigenvectors[:, column], jax_result.eigenvectors[:, column])
            )
            assert overlap == pytest.approx(1.0, abs=1e-9)

    def test_jax_backend_records_device_in_provenance(self) -> None:
        pytest.importorskip("jax")
        result = solve_spectrum(
            np.diag(np.arange(4, dtype=np.float64)),
            backend_name="spectrum-jax",
            provenance_tags=("sweep-label",),
        )
        assert result.metadata.backend_name == "spectrum-jax"
        assert result.metadata.backend_version.startswith("jax-")
        # Caller's own tags preserved; device tag appended as provenance.
        assert "sweep-label" in result.metadata.provenance_tags
        device_tags = [t for t in result.metadata.provenance_tags if t.startswith("device:")]
        assert len(device_tags) == 1
        assert device_tags[0].split(":", 1)[1] in {"cpu", "gpu"}

    def test_jax_backend_accepts_explicit_cpu_device(self) -> None:
        pytest.importorskip("jax")
        result = solve_spectrum(
            np.diag(np.arange(3, dtype=np.float64)),
            backend_name="spectrum-jax",
            device="cpu",
        )
        assert result.metadata.backend_name == "spectrum-jax"
        assert "device:cpu" in result.metadata.provenance_tags

    def test_jax_backend_rejects_unavailable_gpu_device(self) -> None:
        """On CPU-only CI, device='gpu' must surface a clear ConventionError.

        Gated: skipped cleanly on machines where JAX *does* see a GPU platform,
        since there the request is legitimate and the diagnostic path is
        untestable without explicitly disabling GPU visibility.
        """
        jax = pytest.importorskip("jax")
        platforms = {d.platform for d in jax.devices()}
        if "gpu" in platforms:
            pytest.skip("GPU platform available on this machine; diagnostic path not testable.")
        with pytest.raises(ConventionError, match=r"device='gpu'\) requested but unavailable"):
            solve_spectrum(
                np.diag(np.arange(3, dtype=np.float64)),
                backend_name="spectrum-jax",
                device="gpu",
            )
