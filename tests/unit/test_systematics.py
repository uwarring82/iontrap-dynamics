# SPDX-License-Identifier: MIT
"""Unit tests for the systematics layer (Dispatch R scope).

Covers :class:`RabiJitter` construction surface and the multiplier-
sampling contract, plus :func:`perturb_carrier_rabi` and its integration
against :class:`DriveConfig`. Backend-agnostic — no QuTiP imports.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iontrap_dynamics import (
    DetuningJitter,
    PhaseJitter,
    RabiJitter,
    perturb_carrier_rabi,
    perturb_detuning,
    perturb_phase,
)
from iontrap_dynamics.drives import DriveConfig


def _drive(
    *,
    rabi: float = 1.0e6,
    detuning: float = 0.0,
    phase: float = 0.0,
) -> DriveConfig:
    return DriveConfig(
        k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
        carrier_rabi_frequency_rad_s=rabi,
        detuning_rad_s=detuning,
        phase_rad=phase,
    )


# ----------------------------------------------------------------------------
# RabiJitter construction + multiplier contract
# ----------------------------------------------------------------------------


class TestRabiJitterConstruction:
    def test_minimal_construction(self) -> None:
        j = RabiJitter(sigma=0.05)
        assert j.sigma == 0.05
        assert j.label == "rabi_jitter"

    def test_zero_sigma_is_no_op(self) -> None:
        """σ = 0 is a valid no-op (pipeline tests / ideal-limit checks)."""
        j = RabiJitter(sigma=0.0)
        rng = np.random.default_rng(0)
        mult = j.sample_multipliers(shots=100, rng=rng)
        np.testing.assert_array_equal(mult, np.ones(100))

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            RabiJitter(sigma=-0.01)

    def test_frozen(self) -> None:
        j = RabiJitter(sigma=0.01)
        with pytest.raises(FrozenInstanceError):
            j.sigma = 0.05  # type: ignore[misc]

    def test_custom_label(self) -> None:
        j = RabiJitter(sigma=0.02, label="ao_amp_noise")
        assert j.label == "ao_amp_noise"


class TestRabiJitterMultipliers:
    def test_output_shape_and_dtype(self) -> None:
        j = RabiJitter(sigma=0.02)
        rng = np.random.default_rng(0)
        mult = j.sample_multipliers(shots=1000, rng=rng)
        assert mult.shape == (1000,)
        assert mult.dtype == np.float64

    def test_mean_converges_to_one(self) -> None:
        """Large-shot mean of (1 + ε) converges to 1."""
        j = RabiJitter(sigma=0.05)
        rng = np.random.default_rng(0)
        mult = j.sample_multipliers(shots=50_000, rng=rng)
        # 1σ error of the mean ≈ 0.05 / √50k ≈ 2e-4.
        np.testing.assert_allclose(float(mult.mean()), 1.0, atol=1e-3)

    def test_std_matches_sigma(self) -> None:
        """Large-shot std matches the configured sigma."""
        j = RabiJitter(sigma=0.03)
        rng = np.random.default_rng(0)
        mult = j.sample_multipliers(shots=50_000, rng=rng)
        # 1σ error of the std ≈ sigma / √(2(N−1)) ≈ 3e-5.
        np.testing.assert_allclose(float(mult.std(ddof=1)), 0.03, atol=1e-3)

    def test_deterministic_given_rng_seed(self) -> None:
        j = RabiJitter(sigma=0.02)
        first = j.sample_multipliers(shots=64, rng=np.random.default_rng(42))
        second = j.sample_multipliers(shots=64, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)

    def test_zero_shots_raises(self) -> None:
        j = RabiJitter(sigma=0.02)
        with pytest.raises(ValueError, match="shots must be >= 1"):
            j.sample_multipliers(shots=0, rng=np.random.default_rng(0))


# ----------------------------------------------------------------------------
# perturb_carrier_rabi — composition with DriveConfig
# ----------------------------------------------------------------------------


class TestPerturbCarrierRabi:
    def test_output_length(self) -> None:
        drives = perturb_carrier_rabi(_drive(), RabiJitter(sigma=0.02), shots=7, seed=0)
        assert len(drives) == 7
        for d in drives:
            assert isinstance(d, DriveConfig)

    def test_non_rabi_fields_preserved(self) -> None:
        base = _drive(rabi=5e6, phase=0.3)
        drives = perturb_carrier_rabi(base, RabiJitter(sigma=0.02), shots=3, seed=0)
        for d in drives:
            assert d.phase_rad == 0.3
            np.testing.assert_array_equal(d.k_vector_m_inv, base.k_vector_m_inv)

    def test_mean_perturbed_rabi_near_base(self) -> None:
        base = _drive(rabi=1.0e6)
        drives = perturb_carrier_rabi(base, RabiJitter(sigma=0.05), shots=5000, seed=0)
        rabis = np.array([d.carrier_rabi_frequency_rad_s for d in drives])
        # 1σ error of the mean ≈ 5e4 / √5000 ≈ 707 Hz.
        np.testing.assert_allclose(float(rabis.mean()), 1.0e6, atol=3.0e3)

    def test_std_matches_sigma_times_base(self) -> None:
        base = _drive(rabi=2.0e6)
        drives = perturb_carrier_rabi(base, RabiJitter(sigma=0.04), shots=5000, seed=0)
        rabis = np.array([d.carrier_rabi_frequency_rad_s for d in drives])
        expected_std = 0.04 * 2.0e6
        np.testing.assert_allclose(float(rabis.std(ddof=1)), expected_std, rtol=0.05)

    def test_seed_reproducible(self) -> None:
        base = _drive(rabi=1.0e6)
        first = perturb_carrier_rabi(base, RabiJitter(sigma=0.02), shots=10, seed=99)
        second = perturb_carrier_rabi(base, RabiJitter(sigma=0.02), shots=10, seed=99)
        np.testing.assert_array_equal(
            np.array([d.carrier_rabi_frequency_rad_s for d in first]),
            np.array([d.carrier_rabi_frequency_rad_s for d in second]),
        )

    def test_different_seeds_differ(self) -> None:
        base = _drive(rabi=1.0e6)
        first = perturb_carrier_rabi(base, RabiJitter(sigma=0.02), shots=10, seed=1)
        second = perturb_carrier_rabi(base, RabiJitter(sigma=0.02), shots=10, seed=2)
        assert not np.array_equal(
            np.array([d.carrier_rabi_frequency_rad_s for d in first]),
            np.array([d.carrier_rabi_frequency_rad_s for d in second]),
        )

    def test_zero_sigma_passes_through_unchanged(self) -> None:
        base = _drive(rabi=3.14e6)
        drives = perturb_carrier_rabi(base, RabiJitter(sigma=0.0), shots=5, seed=0)
        for d in drives:
            assert d.carrier_rabi_frequency_rad_s == base.carrier_rabi_frequency_rad_s

    def test_zero_shots_raises(self) -> None:
        with pytest.raises(ValueError, match="shots must be >= 1"):
            perturb_carrier_rabi(_drive(), RabiJitter(sigma=0.02), shots=0, seed=0)

    def test_returns_frozen_driveconfigs(self) -> None:
        drives = perturb_carrier_rabi(_drive(), RabiJitter(sigma=0.02), shots=2, seed=0)
        with pytest.raises(FrozenInstanceError):
            drives[0].carrier_rabi_frequency_rad_s = 9999.0  # type: ignore[misc]


# ----------------------------------------------------------------------------
# DetuningJitter — additive offsets on detuning
# ----------------------------------------------------------------------------


class TestDetuningJitterConstruction:
    def test_minimal_construction(self) -> None:
        j = DetuningJitter(sigma_rad_s=500.0)
        assert j.sigma_rad_s == 500.0
        assert j.label == "detuning_jitter"

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma_rad_s must be >= 0"):
            DetuningJitter(sigma_rad_s=-100.0)

    def test_zero_sigma_is_no_op(self) -> None:
        j = DetuningJitter(sigma_rad_s=0.0)
        offsets = j.sample_offsets(shots=50, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(offsets, np.zeros(50))

    def test_frozen(self) -> None:
        j = DetuningJitter(sigma_rad_s=100.0)
        with pytest.raises(FrozenInstanceError):
            j.sigma_rad_s = 200.0  # type: ignore[misc]


class TestDetuningJitterOffsets:
    def test_output_shape_dtype(self) -> None:
        j = DetuningJitter(sigma_rad_s=1000.0)
        offsets = j.sample_offsets(shots=500, rng=np.random.default_rng(0))
        assert offsets.shape == (500,)
        assert offsets.dtype == np.float64

    def test_statistics(self) -> None:
        """Large-shot mean → 0 and std → sigma_rad_s."""
        j = DetuningJitter(sigma_rad_s=750.0)
        offsets = j.sample_offsets(shots=50_000, rng=np.random.default_rng(0))
        np.testing.assert_allclose(float(offsets.mean()), 0.0, atol=20.0)
        np.testing.assert_allclose(float(offsets.std(ddof=1)), 750.0, rtol=0.02)

    def test_seed_reproducible(self) -> None:
        j = DetuningJitter(sigma_rad_s=500.0)
        first = j.sample_offsets(shots=64, rng=np.random.default_rng(42))
        second = j.sample_offsets(shots=64, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)

    def test_zero_shots_raises(self) -> None:
        j = DetuningJitter(sigma_rad_s=500.0)
        with pytest.raises(ValueError, match="shots must be >= 1"):
            j.sample_offsets(shots=0, rng=np.random.default_rng(0))


class TestPerturbDetuning:
    def test_adds_offsets_to_base(self) -> None:
        base = _drive(detuning=2.0e3)
        drives = perturb_detuning(base, DetuningJitter(sigma_rad_s=500.0), shots=5000, seed=0)
        detunings = np.array([d.detuning_rad_s for d in drives])
        # Mean of perturbed detunings → base (within SE of mean).
        np.testing.assert_allclose(float(detunings.mean()), 2.0e3, atol=30.0)
        np.testing.assert_allclose(float(detunings.std(ddof=1)), 500.0, rtol=0.05)

    def test_non_detuning_fields_preserved(self) -> None:
        base = _drive(rabi=1.5e6, detuning=1.0e3, phase=0.3)
        drives = perturb_detuning(base, DetuningJitter(sigma_rad_s=100.0), shots=3, seed=0)
        for d in drives:
            assert d.carrier_rabi_frequency_rad_s == 1.5e6
            assert d.phase_rad == 0.3

    def test_seed_reproducible(self) -> None:
        base = _drive(detuning=0.0)
        first = perturb_detuning(base, DetuningJitter(sigma_rad_s=500.0), shots=10, seed=99)
        second = perturb_detuning(base, DetuningJitter(sigma_rad_s=500.0), shots=10, seed=99)
        np.testing.assert_array_equal(
            np.array([d.detuning_rad_s for d in first]),
            np.array([d.detuning_rad_s for d in second]),
        )

    def test_zero_sigma_passes_through(self) -> None:
        base = _drive(detuning=1.234e3)
        drives = perturb_detuning(base, DetuningJitter(sigma_rad_s=0.0), shots=5, seed=0)
        for d in drives:
            assert d.detuning_rad_s == 1.234e3

    def test_zero_shots_raises(self) -> None:
        with pytest.raises(ValueError, match="shots must be >= 1"):
            perturb_detuning(_drive(), DetuningJitter(sigma_rad_s=500.0), shots=0, seed=0)


# ----------------------------------------------------------------------------
# PhaseJitter — additive offsets on phase
# ----------------------------------------------------------------------------


class TestPhaseJitterConstruction:
    def test_minimal_construction(self) -> None:
        j = PhaseJitter(sigma_rad=0.05)
        assert j.sigma_rad == 0.05
        assert j.label == "phase_jitter"

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma_rad must be >= 0"):
            PhaseJitter(sigma_rad=-0.01)

    def test_zero_sigma_is_no_op(self) -> None:
        j = PhaseJitter(sigma_rad=0.0)
        offsets = j.sample_offsets(shots=50, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(offsets, np.zeros(50))

    def test_frozen(self) -> None:
        j = PhaseJitter(sigma_rad=0.05)
        with pytest.raises(FrozenInstanceError):
            j.sigma_rad = 0.1  # type: ignore[misc]


class TestPhaseJitterOffsets:
    def test_output_shape_dtype(self) -> None:
        j = PhaseJitter(sigma_rad=0.1)
        offsets = j.sample_offsets(shots=500, rng=np.random.default_rng(0))
        assert offsets.shape == (500,)
        assert offsets.dtype == np.float64

    def test_statistics(self) -> None:
        j = PhaseJitter(sigma_rad=0.08)
        offsets = j.sample_offsets(shots=50_000, rng=np.random.default_rng(0))
        np.testing.assert_allclose(float(offsets.mean()), 0.0, atol=1e-3)
        np.testing.assert_allclose(float(offsets.std(ddof=1)), 0.08, rtol=0.02)

    def test_seed_reproducible(self) -> None:
        j = PhaseJitter(sigma_rad=0.05)
        first = j.sample_offsets(shots=64, rng=np.random.default_rng(42))
        second = j.sample_offsets(shots=64, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)


class TestPerturbPhase:
    def test_adds_offsets_to_base(self) -> None:
        base = _drive(phase=0.5)
        drives = perturb_phase(base, PhaseJitter(sigma_rad=0.05), shots=5000, seed=0)
        phases = np.array([d.phase_rad for d in drives])
        np.testing.assert_allclose(float(phases.mean()), 0.5, atol=1e-3)
        np.testing.assert_allclose(float(phases.std(ddof=1)), 0.05, rtol=0.05)

    def test_non_phase_fields_preserved(self) -> None:
        base = _drive(rabi=2.0e6, detuning=3.0e3, phase=0.0)
        drives = perturb_phase(base, PhaseJitter(sigma_rad=0.02), shots=3, seed=0)
        for d in drives:
            assert d.carrier_rabi_frequency_rad_s == 2.0e6
            assert d.detuning_rad_s == 3.0e3

    def test_seed_reproducible(self) -> None:
        base = _drive(phase=0.0)
        first = perturb_phase(base, PhaseJitter(sigma_rad=0.1), shots=10, seed=99)
        second = perturb_phase(base, PhaseJitter(sigma_rad=0.1), shots=10, seed=99)
        np.testing.assert_array_equal(
            np.array([d.phase_rad for d in first]),
            np.array([d.phase_rad for d in second]),
        )

    def test_zero_shots_raises(self) -> None:
        with pytest.raises(ValueError, match="shots must be >= 1"):
            perturb_phase(_drive(), PhaseJitter(sigma_rad=0.05), shots=0, seed=0)

    def test_large_offsets_accepted_no_wrap(self) -> None:
        """Phase is not wrapped — values outside [−π, π] are legal per §3 / DriveConfig docs."""
        base = _drive(phase=np.pi)
        drives = perturb_phase(base, PhaseJitter(sigma_rad=1.5), shots=100, seed=0)
        phases = np.array([d.phase_rad for d in drives])
        # Some phases will land > π or < 0 — check the spread reaches outside [0, 2π].
        assert phases.min() < 0.5 or phases.max() > 2 * np.pi
