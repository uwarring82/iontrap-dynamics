# SPDX-License-Identifier: MIT
"""Unit tests for the common-mode phase channel (WI-4 of WP-01, dispatch EDD).

Verifies the correlation interpolation: at correlation = 1 a single offset is
shared across all drives (it cancels in the phase difference -> common-mode
rejection); at correlation = 0 the per-drive offsets are independent (the
channel reduces to PhaseJitter); the marginal std is sigma at every
correlation; and the difference-observable variance is monotone decreasing in
correlation, reaching exactly zero at full correlation.
"""

from __future__ import annotations

import numpy as np
import pytest

from iontrap_dynamics import CommonModePhase, perturb_common_mode
from iontrap_dynamics.drives import DriveConfig

_WAVENUMBER = 2 * np.pi / 280e-9


def _drive(phase: float) -> DriveConfig:
    return DriveConfig(
        k_vector_m_inv=[0.0, 0.0, _WAVENUMBER],
        carrier_rabi_frequency_rad_s=2 * np.pi * 1e6,
        phase_rad=phase,
    )


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_full_correlation_shares_one_offset() -> None:
    spec = CommonModePhase(sigma_rad=0.3, correlation=1.0)
    offsets = spec.sample_offsets(n_subsystems=3, shots=2000, rng=_rng())
    assert offsets.shape == (2000, 3)
    # Every subsystem in a shot gets the identical offset.
    assert np.allclose(offsets[:, 0], offsets[:, 1])
    assert np.allclose(offsets[:, 0], offsets[:, 2])


def test_full_correlation_cancels_in_difference() -> None:
    spec = CommonModePhase(sigma_rad=0.4, correlation=1.0)
    drives = [_drive(0.0), _drive(0.5)]
    perturbed = perturb_common_mode(drives, spec, shots=100, seed=1)
    assert len(perturbed) == 100
    for shot in perturbed:
        assert len(shot) == 2
        # The shared offset cancels: the phase difference is invariant.
        assert shot[0].phase_rad - shot[1].phase_rad == pytest.approx(-0.5, abs=1e-12)


def test_zero_correlation_is_independent() -> None:
    spec = CommonModePhase(sigma_rad=0.2, correlation=0.0)
    offsets = spec.sample_offsets(n_subsystems=2, shots=50000, rng=_rng(7))
    cross = np.corrcoef(offsets[:, 0], offsets[:, 1])[0, 1]
    assert abs(cross) < 0.05  # columns statistically independent
    assert offsets[:, 0].std() == pytest.approx(0.2, rel=0.05)


@pytest.mark.parametrize("correlation", [0.0, 0.5, 1.0])
def test_marginal_std_independent_of_correlation(correlation: float) -> None:
    spec = CommonModePhase(sigma_rad=0.25, correlation=correlation)
    offsets = spec.sample_offsets(n_subsystems=2, shots=50000, rng=_rng(3))
    assert offsets.std() == pytest.approx(0.25, rel=0.05)


def test_difference_variance_monotone_in_correlation() -> None:
    sigma = 0.3
    variances = []
    for correlation in (0.0, 0.25, 0.5, 0.75, 1.0):
        spec = CommonModePhase(sigma_rad=sigma, correlation=correlation)
        offsets = spec.sample_offsets(n_subsystems=2, shots=50000, rng=_rng(11))
        variances.append(float((offsets[:, 0] - offsets[:, 1]).var()))
    # var(diff) = 2 sigma^2 (1 - c): non-increasing in c, exactly 0 at c = 1.
    assert all(variances[i] >= variances[i + 1] - 1e-9 for i in range(len(variances) - 1))
    assert variances[0] == pytest.approx(2 * sigma**2, rel=0.05)
    assert variances[-1] == pytest.approx(0.0, abs=1e-12)


def test_perturb_structure_and_immutability() -> None:
    spec = CommonModePhase(sigma_rad=0.1, correlation=0.5)
    drives = [_drive(0.0), _drive(1.0)]
    perturbed = perturb_common_mode(drives, spec, shots=5, seed=2)
    assert len(perturbed) == 5
    assert all(len(shot) == 2 for shot in perturbed)
    # Originals not mutated; non-phase fields pass through.
    assert drives[0].phase_rad == 0.0
    assert drives[1].phase_rad == 1.0
    for shot in perturbed:
        for original, pert in zip(drives, shot, strict=True):
            assert pert.carrier_rabi_frequency_rad_s == original.carrier_rabi_frequency_rad_s


def test_reproducible_with_seed() -> None:
    spec = CommonModePhase(sigma_rad=0.2, correlation=0.6)
    drives = [_drive(0.0), _drive(0.3)]
    a = perturb_common_mode(drives, spec, shots=10, seed=42)
    b = perturb_common_mode(drives, spec, shots=10, seed=42)
    for shot_a, shot_b in zip(a, b, strict=True):
        for drive_a, drive_b in zip(shot_a, shot_b, strict=True):
            assert drive_a.phase_rad == drive_b.phase_rad


def test_validation() -> None:
    with pytest.raises(ValueError):
        CommonModePhase(sigma_rad=-0.1)
    with pytest.raises(ValueError):
        CommonModePhase(sigma_rad=0.1, correlation=1.5)
    with pytest.raises(ValueError):
        CommonModePhase(sigma_rad=0.1, correlation=-0.1)
    with pytest.raises(ValueError):
        CommonModePhase(sigma_rad=float("nan"))
    with pytest.raises(ValueError):
        CommonModePhase(sigma_rad=0.1, correlation=float("inf"))
    spec = CommonModePhase(sigma_rad=0.1)
    with pytest.raises(ValueError):
        perturb_common_mode([], spec, shots=5)
    with pytest.raises(ValueError):
        perturb_common_mode([_drive(0.0)], spec, shots=0)
