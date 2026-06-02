# SPDX-License-Identifier: MIT
"""Analytic-regression oracle for the common-mode channel benchmark.

Binding closed-form anchor for ``tools/run_benchmark_common_mode.py`` (WP-01
§7 row 6, dispatch EDE): the difference-observable variance of a two-subsystem
correlated shared-latent phase channel,
:class:`iontrap_dynamics.CommonModePhase`, follows

    Var(offset_0 - offset_1) = 2 sigma^2 (1 - c)

for correlation ``c in [0, 1]``. At ``c = 0`` the offsets are independent and
the variance is ``2 sigma^2`` (incoherent per-drive sum); at ``c = 1`` the
shared latent cancels exactly so the difference variance is ``0`` (common-mode
rejection), and the per-shot phase difference of two drives perturbed via
:func:`iontrap_dynamics.perturb_common_mode` is invariant. The variance is
monotone decreasing in ``c``. The benchmark tool reports
``max_numerical_vs_analytic_error``; this is the independent binding assertion,
with the tolerance as a named symbolic constant.

The sampling uses a fixed ``np.random.default_rng(seed)`` so the assertions are
deterministic. The exact ``c = 1`` cancellation is asserted at machine
precision, independent of any sampling.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from iontrap_dynamics import CommonModePhase, perturb_common_mode
from iontrap_dynamics.drives import DriveConfig

pytestmark = pytest.mark.regression_analytic

ATOL_COMMON_MODE_INTERIOR = 5e-3
ATOL_COMMON_MODE_EXACT = 1e-12

_SIGMA_RAD = 0.3
_SHOTS = 4_000_000
_SEED = 20260602


def _difference_variance(correlation: float) -> float:
    """Sampled Var(offset_0 - offset_1) at the given correlation, fixed seed."""
    spec = CommonModePhase(sigma_rad=_SIGMA_RAD, correlation=correlation)
    offsets = spec.sample_offsets(n_subsystems=2, shots=_SHOTS, rng=np.random.default_rng(_SEED))
    return float(np.var(offsets[:, 0] - offsets[:, 1]))


@pytest.mark.parametrize("correlation", [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
def test_difference_variance_matches_2_sigma_sq_one_minus_c(correlation: float) -> None:
    analytic = 2.0 * _SIGMA_RAD**2 * (1.0 - correlation)
    assert _difference_variance(correlation) == pytest.approx(
        analytic, abs=ATOL_COMMON_MODE_INTERIOR
    )


def test_c1_difference_variance_cancels_exactly() -> None:
    """At c=1 the shared latent is identical across subsystems: difference is 0."""
    spec = CommonModePhase(sigma_rad=_SIGMA_RAD, correlation=1.0)
    offsets = spec.sample_offsets(n_subsystems=2, shots=_SHOTS, rng=np.random.default_rng(_SEED))
    difference = offsets[:, 0] - offsets[:, 1]
    assert np.max(np.abs(difference)) == pytest.approx(0.0, abs=ATOL_COMMON_MODE_EXACT)
    assert float(np.var(difference)) == pytest.approx(0.0, abs=ATOL_COMMON_MODE_EXACT)


def test_difference_variance_monotone_decreasing_in_correlation() -> None:
    """The difference variance is monotone decreasing in c across the grid."""
    grid = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)
    variances = [_difference_variance(c) for c in grid]
    for earlier, later in itertools.pairwise(variances):
        assert later <= earlier + ATOL_COMMON_MODE_INTERIOR


def test_perturb_common_mode_phase_difference_invariant_at_c1() -> None:
    """At c=1 the per-shot phase difference of two perturbed drives is invariant."""
    k = np.array([0.0, 0.0, 1.0e7], dtype=np.float64)
    drive_a = DriveConfig(k_vector_m_inv=k, carrier_rabi_frequency_rad_s=1.0e6, phase_rad=0.2)
    drive_b = DriveConfig(k_vector_m_inv=k, carrier_rabi_frequency_rad_s=1.0e6, phase_rad=-0.5)
    base_diff = drive_a.phase_rad - drive_b.phase_rad
    spec = CommonModePhase(sigma_rad=_SIGMA_RAD, correlation=1.0)
    perturbed = perturb_common_mode([drive_a, drive_b], spec, shots=256, seed=_SEED)
    for shot in perturbed:
        diff = shot[0].phase_rad - shot[1].phase_rad
        assert diff == pytest.approx(base_diff, abs=ATOL_COMMON_MODE_EXACT)
