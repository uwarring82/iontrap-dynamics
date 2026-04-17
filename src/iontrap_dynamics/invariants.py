# SPDX-License-Identifier: MIT
"""Invariant diagnostics for quantum states and density matrices.

These helpers provide the permanent physics anchors named in
``WORKPLAN_v0.3.md`` section 0.B and the failure thresholds declared in
``CONVENTIONS.md`` section 15. They are backend-agnostic and operate on
NumPy-convertible arrays, so they can be reused by both QuTiP-backed and
future JAX-backed solver layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import IntegrityError

TRACE_HARD_FAILURE_TOLERANCE = 1.0e-6
HERMITICITY_HARD_FAILURE_TOLERANCE = 1.0e-10
POSITIVITY_HARD_FAILURE_TOLERANCE = 1.0e-8
NORM_HARD_FAILURE_TOLERANCE = 1.0e-6

_ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True, kw_only=True)
class DensityMatrixDiagnostics:
    """Summary diagnostics for a candidate density matrix."""

    trace_deviation: float
    hermiticity_deviation: float
    minimum_eigenvalue: float


@dataclass(frozen=True, slots=True, kw_only=True)
class StateVectorDiagnostics:
    """Summary diagnostics for a candidate pure-state vector."""

    norm_deviation: float


def _as_square_matrix(matrix: ArrayLike) -> _ComplexArray:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("expected a square 2-D array")
    return array


def _as_state_vector(state: ArrayLike) -> _ComplexArray:
    array = np.asarray(state, dtype=np.complex128)
    if array.ndim != 1:
        raise ValueError("expected a 1-D state vector")
    return array


def density_matrix_diagnostics(matrix: ArrayLike) -> DensityMatrixDiagnostics:
    r"""Compute trace, Hermiticity, and positivity diagnostics for ``matrix``.

    Positivity is evaluated on the Hermitian projection
    ``(rho + rho^\dagger) / 2``. This is intentional: for numerically noisy but
    nearly Hermitian inputs, it measures the physically meaningful spectrum
    rather than an arbitrary complex one.
    """

    rho = _as_square_matrix(matrix)
    trace_deviation = float(abs(np.trace(rho) - 1.0))
    hermiticity_deviation = float(np.max(np.abs(rho - rho.conj().T)))
    hermitian_projection = 0.5 * (rho + rho.conj().T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian_projection)))
    return DensityMatrixDiagnostics(
        trace_deviation=trace_deviation,
        hermiticity_deviation=hermiticity_deviation,
        minimum_eigenvalue=minimum_eigenvalue,
    )


def validate_density_matrix(
    matrix: ArrayLike,
    *,
    trace_tolerance: float = TRACE_HARD_FAILURE_TOLERANCE,
    hermiticity_tolerance: float = HERMITICITY_HARD_FAILURE_TOLERANCE,
    positivity_tolerance: float = POSITIVITY_HARD_FAILURE_TOLERANCE,
) -> DensityMatrixDiagnostics:
    """Return diagnostics for ``matrix`` or raise ``IntegrityError``."""

    diagnostics = density_matrix_diagnostics(matrix)
    failures: list[str] = []

    if diagnostics.trace_deviation > trace_tolerance:
        failures.append(
            f"trace deviation {diagnostics.trace_deviation:.3e} exceeds {trace_tolerance:.3e}"
        )
    if diagnostics.hermiticity_deviation > hermiticity_tolerance:
        failures.append(
            "Hermiticity deviation "
            f"{diagnostics.hermiticity_deviation:.3e} exceeds {hermiticity_tolerance:.3e}"
        )
    if diagnostics.minimum_eigenvalue < -positivity_tolerance:
        failures.append(
            "minimum eigenvalue "
            f"{diagnostics.minimum_eigenvalue:.3e} is below -{positivity_tolerance:.3e}"
        )

    if failures:
        raise IntegrityError("; ".join(failures))

    return diagnostics


def state_vector_diagnostics(state: ArrayLike) -> StateVectorDiagnostics:
    """Compute norm-conservation diagnostics for ``state``."""

    psi = _as_state_vector(state)
    norm_deviation = float(abs(np.vdot(psi, psi) - 1.0))
    return StateVectorDiagnostics(norm_deviation=norm_deviation)


def validate_state_vector(
    state: ArrayLike,
    *,
    norm_tolerance: float = NORM_HARD_FAILURE_TOLERANCE,
) -> StateVectorDiagnostics:
    """Return diagnostics for ``state`` or raise ``IntegrityError``."""

    diagnostics = state_vector_diagnostics(state)
    if diagnostics.norm_deviation > norm_tolerance:
        raise IntegrityError(
            "state-vector norm deviation "
            f"{diagnostics.norm_deviation:.3e} exceeds {norm_tolerance:.3e}"
        )
    return diagnostics


def symmetry_deviation(state_or_matrix: ArrayLike, symmetry: ArrayLike) -> float:
    r"""Return the max-entry deviation from invariance under ``symmetry``.

    For state vectors, this computes ``max(abs(S @ psi - psi))``.
    For density matrices, this computes ``max(abs(S @ rho @ S^\dagger - rho))``.
    """

    operator = _as_square_matrix(symmetry)
    object_array = np.asarray(state_or_matrix, dtype=np.complex128)

    if object_array.ndim == 1:
        if operator.shape[0] != object_array.shape[0]:
            raise ValueError("symmetry operator dimension does not match state vector")
        return float(np.max(np.abs(operator @ object_array - object_array)))

    if object_array.ndim == 2:
        candidate = _as_square_matrix(object_array)
        if operator.shape != candidate.shape:
            raise ValueError("symmetry operator shape does not match density matrix")
        return float(np.max(np.abs(operator @ candidate @ operator.conj().T - candidate)))

    raise ValueError("expected a 1-D state vector or 2-D density matrix")


__all__ = [
    "HERMITICITY_HARD_FAILURE_TOLERANCE",
    "NORM_HARD_FAILURE_TOLERANCE",
    "POSITIVITY_HARD_FAILURE_TOLERANCE",
    "TRACE_HARD_FAILURE_TOLERANCE",
    "DensityMatrixDiagnostics",
    "StateVectorDiagnostics",
    "density_matrix_diagnostics",
    "state_vector_diagnostics",
    "symmetry_deviation",
    "validate_density_matrix",
    "validate_state_vector",
]
