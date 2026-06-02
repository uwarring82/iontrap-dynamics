# SPDX-License-Identifier: MIT
"""Analytic-regression oracle for the CFI / linear-Gaussian benchmark.

Binding closed-form anchors for ``tools/run_benchmark_cfi_linear_gaussian.py``
(WP-01 §7 row 2, dispatch EDE), independent of the tool:

(a) the linear-Gaussian Fisher matrix from
    :func:`iontrap_dynamics.information.linear_gaussian_fisher` equals the
    hand-computed ``AᵀΣ⁻¹A`` across a variance sweep, and the single-parameter
    Cramér–Rao bound equals ``1 / F``;
(b) the Braunstein–Caves inequality CFI ≤ QFI for a single-qubit phase model —
    saturated by the optimal σ_x-basis measurement (CFI = QFI = 1) and strict
    (CFI = 0) for a phase-insensitive one.

This file is kept separate from test_analytic.py: the latter is deliberately
QuTiP-free, whereas the QFI half of this oracle constructs a quantum state.
Both live in the regression_analytic tier. The tolerance is a named symbolic
constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import _spin_hilbert
from iontrap_dynamics.information import (
    classical_fisher_information,
    cramer_rao_bound,
    linear_gaussian_fisher,
    quantum_fisher_information_trajectory,
)
from iontrap_dynamics.operators import sigma_z_ion, spin_down, spin_up

pytestmark = pytest.mark.regression_analytic

ATOL_CFI_LINEAR_GAUSSIAN = 1e-9

_DESIGN_MATRIX = np.array([[1.0, 0.0], [0.5, 2.0]], dtype=np.float64)


@pytest.mark.parametrize("variance", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_linear_gaussian_fisher_matches_hand_computed(variance: float) -> None:
    """``linear_gaussian_fisher`` equals ``AᵀΣ⁻¹A`` to machine precision."""
    sigma = variance * np.eye(_DESIGN_MATRIX.shape[0], dtype=np.float64)
    f_lib = linear_gaussian_fisher(A=_DESIGN_MATRIX, sigma=sigma)
    f_hand = _DESIGN_MATRIX.T @ np.linalg.inv(sigma) @ _DESIGN_MATRIX
    assert np.allclose(f_lib, f_hand, atol=ATOL_CFI_LINEAR_GAUSSIAN)


def test_cramer_rao_bound_is_inverse_fisher_one_parameter() -> None:
    """On a 1-parameter case the CRB equals ``1 / F[0,0]``."""
    sigma = np.eye(_DESIGN_MATRIX.shape[0], dtype=np.float64)
    f = linear_gaussian_fisher(A=_DESIGN_MATRIX, sigma=sigma)
    f00 = float(f[0, 0])
    assert cramer_rao_bound(f00) == pytest.approx(1.0 / f00, abs=ATOL_CFI_LINEAR_GAUSSIAN)


def test_braunstein_caves_saturation_single_qubit_phase() -> None:
    """CFI = QFI = 1 (optimal σ_x basis) and CFI = 0 < QFI (phase-insensitive)."""
    hilbert = _spin_hilbert(1)
    plus = (spin_up() + spin_down()).unit()
    jz = 0.5 * hilbert.spin_op_for_ion(sigma_z_ion(), 0)
    qfi = quantum_fisher_information_trajectory([plus], hilbert=hilbert, generator=jz)[0]
    assert qfi == pytest.approx(1.0, abs=ATOL_CFI_LINEAR_GAUSSIAN)

    cfi_optimal = classical_fisher_information([0.5, 0.5], parameter_derivative=[-0.5, 0.5])
    assert cfi_optimal == pytest.approx(qfi, abs=ATOL_CFI_LINEAR_GAUSSIAN)
    assert cfi_optimal == pytest.approx(1.0, abs=ATOL_CFI_LINEAR_GAUSSIAN)

    cfi_insensitive = classical_fisher_information([0.5, 0.5], parameter_derivative=[0.0, 0.0])
    assert cfi_insensitive == pytest.approx(0.0, abs=ATOL_CFI_LINEAR_GAUSSIAN)
    assert cfi_insensitive < qfi
