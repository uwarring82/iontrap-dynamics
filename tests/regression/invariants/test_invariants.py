# SPDX-License-Identifier: MIT
"""Permanent invariant regression tests for synthetic quantum states."""

from __future__ import annotations

import numpy as np
import pytest

from iontrap_dynamics.exceptions import IntegrityError
from iontrap_dynamics.invariants import (
    DensityMatrixDiagnostics,
    StateVectorDiagnostics,
    density_matrix_diagnostics,
    state_vector_diagnostics,
    symmetry_deviation,
    validate_density_matrix,
    validate_state_vector,
)

pytestmark = pytest.mark.regression_invariant


def _ket0() -> np.ndarray:
    return np.array([1.0, 0.0], dtype=np.complex128)


def _ket1() -> np.ndarray:
    return np.array([0.0, 1.0], dtype=np.complex128)


def _density_matrix(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def _bell_phi_plus() -> np.ndarray:
    return (1.0 / np.sqrt(2.0)) * np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128)


def _swap_operator() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


def test_density_matrix_diagnostics_for_valid_state() -> None:
    rho = _density_matrix(_ket0())

    diagnostics = density_matrix_diagnostics(rho)

    assert isinstance(diagnostics, DensityMatrixDiagnostics)
    assert diagnostics.trace_deviation == pytest.approx(0.0)
    assert diagnostics.hermiticity_deviation == pytest.approx(0.0)
    assert diagnostics.minimum_eigenvalue == pytest.approx(0.0)


def test_validate_density_matrix_accepts_physical_mixed_state() -> None:
    rho = np.diag([0.75, 0.25]).astype(np.complex128)

    diagnostics = validate_density_matrix(rho)

    assert diagnostics.trace_deviation == pytest.approx(0.0)
    assert diagnostics.minimum_eigenvalue == pytest.approx(0.25)


def test_validate_density_matrix_rejects_trace_drift() -> None:
    rho = np.diag([1.0 + 2.0e-6, 0.0]).astype(np.complex128)

    with pytest.raises(IntegrityError, match="trace deviation"):
        validate_density_matrix(rho)


def test_validate_density_matrix_rejects_hermiticity_violation() -> None:
    rho = np.array([[0.5, 1.0e-7], [0.0, 0.5]], dtype=np.complex128)

    with pytest.raises(IntegrityError, match="Hermiticity deviation"):
        validate_density_matrix(rho)


def test_validate_density_matrix_rejects_negative_eigenvalue() -> None:
    rho = np.diag([1.0 + 5.0e-8, -5.0e-8]).astype(np.complex128)

    with pytest.raises(IntegrityError, match="minimum eigenvalue"):
        validate_density_matrix(rho)


def test_state_vector_diagnostics_for_normalized_state() -> None:
    psi = (1.0 / np.sqrt(2.0)) * (_ket0() + _ket1())

    diagnostics = state_vector_diagnostics(psi)

    assert isinstance(diagnostics, StateVectorDiagnostics)
    assert diagnostics.norm_deviation == pytest.approx(0.0)


def test_validate_state_vector_rejects_norm_drift() -> None:
    psi = np.array([np.sqrt(1.0 + 2.0e-6), 0.0], dtype=np.complex128)

    with pytest.raises(IntegrityError, match="norm deviation"):
        validate_state_vector(psi)


def test_swap_symmetry_deviation_vanishes_for_bell_state() -> None:
    bell_state = _bell_phi_plus()

    deviation = symmetry_deviation(bell_state, _swap_operator())

    assert deviation == pytest.approx(0.0)


def test_swap_symmetry_deviation_detects_asymmetric_state() -> None:
    product_state = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)

    deviation = symmetry_deviation(product_state, _swap_operator())

    assert deviation == pytest.approx(1.0)
