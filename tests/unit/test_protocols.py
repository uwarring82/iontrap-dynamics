# SPDX-License-Identifier: MIT
"""Unit tests for the protocol layer (Dispatch M scope).

Covers :class:`SpinReadout` construction surface, the projective-shot
sampling contract (envelope convergence, seed reproducibility, edge
probabilities), metadata propagation, and trajectory-lookup errors.
Tests are backend-agnostic and do not import QuTiP.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iontrap_dynamics import (
    CONVENTION_VERSION,
    ConventionError,
    DetectorConfig,
    MeasurementResult,
    ResultMetadata,
    SpinReadout,
    StorageMode,
    TrajectoryResult,
)

# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _trajectory(
    sigma_z: np.ndarray,
    *,
    ion_index: int = 0,
    provenance_tags: tuple[str, ...] = ("demo", "carrier_rabi"),
    request_hash: str = "a" * 64,
) -> TrajectoryResult:
    """Build a TrajectoryResult whose expectations carry sigma_z_{ion_index}."""
    times = np.linspace(0.0, 1.0e-6, sigma_z.size)
    meta = ResultMetadata(
        convention_version=CONVENTION_VERSION,
        request_hash=request_hash,
        backend_name="qutip-mesolve",
        backend_version="5.0.0",
        storage_mode=StorageMode.OMITTED,
        provenance_tags=provenance_tags,
    )
    return TrajectoryResult(
        metadata=meta,
        times=times,
        expectations={f"sigma_z_{ion_index}": sigma_z},
    )


def _ideal_detector(threshold: int = 1) -> DetectorConfig:
    """η=1, γ_d=0 — perfect detector with threshold at 1 photon."""
    return DetectorConfig(
        efficiency=1.0,
        dark_count_rate=0.0,
        threshold=threshold,
    )


def _noisy_detector() -> DetectorConfig:
    """η=0.4, γ_d=0.3, N̂=4 — finite-fidelity detector."""
    return DetectorConfig(
        efficiency=0.4,
        dark_count_rate=0.3,
        threshold=4,
    )


# ----------------------------------------------------------------------------
# SpinReadout construction surface
# ----------------------------------------------------------------------------


class TestSpinReadoutConstruction:
    def test_minimal_construction(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        assert protocol.ion_index == 0
        assert protocol.lambda_bright == 10.0
        assert protocol.lambda_dark == 0.0
        assert protocol.label == "spin_readout"
        assert protocol.detector.efficiency == 1.0

    def test_negative_ion_index_raises(self) -> None:
        with pytest.raises(ValueError, match="ion_index must be >= 0"):
            SpinReadout(
                ion_index=-1,
                detector=_ideal_detector(),
                lambda_bright=10.0,
                lambda_dark=0.0,
            )

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rates must be >= 0"):
            SpinReadout(
                ion_index=0,
                detector=_ideal_detector(),
                lambda_bright=10.0,
                lambda_dark=-0.1,
            )

    def test_dark_greater_than_bright_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_dark must be <= lambda_bright"):
            SpinReadout(
                ion_index=0,
                detector=_ideal_detector(),
                lambda_bright=2.0,
                lambda_dark=5.0,
            )

    def test_frozen(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            protocol.ion_index = 1  # type: ignore[misc]

    def test_positional_construction_forbidden(self) -> None:
        with pytest.raises(TypeError):
            SpinReadout(0, _ideal_detector(), 10.0, 0.0)  # type: ignore[misc]


# ----------------------------------------------------------------------------
# SpinReadout.run — result shape, dual-view, and metadata surface
# ----------------------------------------------------------------------------


class TestSpinReadoutRunSurface:
    def test_result_is_measurement_result(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        sigma_z = np.array([-1.0, 0.0, 1.0])
        result = protocol.run(_trajectory(sigma_z), shots=100, seed=0)
        assert isinstance(result, MeasurementResult)
        assert result.shots == 100
        assert result.rng_seed == 0

    def test_sampled_outcome_shapes(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        sigma_z = np.linspace(-1.0, 1.0, 5)
        result = protocol.run(_trajectory(sigma_z), shots=200, seed=0)
        counts = result.sampled_outcome["spin_readout_counts"]
        bits = result.sampled_outcome["spin_readout_bits"]
        fraction = result.sampled_outcome["spin_readout_bright_fraction"]
        assert counts.shape == (200, 5)
        assert counts.dtype == np.int64
        assert bits.shape == (200, 5)
        assert bits.dtype == np.int8
        assert fraction.shape == (5,)
        assert fraction.dtype == np.float64

    def test_ideal_outcome_carries_p_up_and_envelope(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_noisy_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        sigma_z = np.array([-1.0, 0.0, 1.0])
        result = protocol.run(_trajectory(sigma_z), shots=100, seed=0)
        np.testing.assert_allclose(result.ideal_outcome["p_up"], np.array([0.0, 0.5, 1.0]))
        assert result.ideal_outcome["bright_fraction_envelope"].shape == (3,)

    def test_envelope_is_linear_in_p_up(self) -> None:
        """Envelope must satisfy env = TP·p + (1−TN)·(1−p) — slope = TP − (1−TN)."""
        detector = _noisy_detector()
        protocol = SpinReadout(
            ion_index=0,
            detector=detector,
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        sigma_z = np.linspace(-1.0, 1.0, 11)
        result = protocol.run(_trajectory(sigma_z), shots=10, seed=0)
        p_up = result.ideal_outcome["p_up"]
        env = result.ideal_outcome["bright_fraction_envelope"]

        fid = detector.classification_fidelity(lambda_bright=25.0, lambda_dark=0.0)
        expected = fid["true_positive_rate"] * p_up + (1.0 - fid["true_negative_rate"]) * (
            1.0 - p_up
        )
        np.testing.assert_allclose(env, expected)

    def test_trajectory_hash_inherited(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        trajectory = _trajectory(np.array([0.0]), request_hash="c" * 64)
        result = protocol.run(trajectory, shots=10, seed=0)
        assert result.trajectory_hash == "c" * 64

    def test_provenance_chain(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        trajectory = _trajectory(np.array([0.0]), provenance_tags=("demo", "rabi"))
        result = protocol.run(trajectory, shots=10, seed=0, provenance_tags=("unit-test",))
        assert result.metadata.provenance_tags == (
            "demo",
            "rabi",
            "measurement",
            "spin_readout",
            "unit-test",
        )

    def test_storage_mode_omitted(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        result = protocol.run(_trajectory(np.array([0.0])), shots=10, seed=0)
        assert result.metadata.storage_mode is StorageMode.OMITTED

    def test_label_routes_sampled_outcome_keys(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
            label="readout_ion_0",
        )
        result = protocol.run(_trajectory(np.array([0.0])), shots=10, seed=0)
        assert "readout_ion_0_counts" in result.sampled_outcome
        assert "readout_ion_0_bits" in result.sampled_outcome
        assert "readout_ion_0_bright_fraction" in result.sampled_outcome
        assert "spin_readout_counts" not in result.sampled_outcome


# ----------------------------------------------------------------------------
# SpinReadout.run — error paths
# ----------------------------------------------------------------------------


class TestSpinReadoutRunErrors:
    def test_missing_observable_raises(self) -> None:
        """If trajectory lacks the expected σ_z observable, raise ConventionError."""
        protocol = SpinReadout(
            ion_index=5,  # trajectory only has sigma_z_0
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        trajectory = _trajectory(np.array([0.0]))
        with pytest.raises(ConventionError, match="sigma_z_5"):
            protocol.run(trajectory, shots=1, seed=0)

    def test_zero_shots_raises(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        with pytest.raises(ValueError, match="shots must be >= 1"):
            protocol.run(_trajectory(np.array([0.0])), shots=0, seed=0)

    def test_out_of_range_sigma_z_raises(self) -> None:
        """An ODE trajectory with |⟨σ_z⟩| >> 1 indicates an upstream bug."""
        protocol = SpinReadout(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        bad_trajectory = _trajectory(np.array([1.5]))  # p_up = 1.25
        with pytest.raises(ValueError, match="p_up lies outside"):
            protocol.run(bad_trajectory, shots=10, seed=0)


# ----------------------------------------------------------------------------
# Projective-shot sampling — physics contract
# ----------------------------------------------------------------------------


class TestSpinReadoutProjectiveSampling:
    def test_p_up_zero_gives_all_dark(self) -> None:
        """p_↑ = 0 everywhere → every shot projects dark → Poisson(λ_dark_eff)."""
        protocol = SpinReadout(
            ion_index=0,
            detector=DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        sigma_z = np.full(4, -1.0)  # p_up = 0 everywhere
        result = protocol.run(_trajectory(sigma_z), shots=500, seed=0)
        # With λ_dark = 0 and γ_d = 0, every count is exactly 0, so every
        # bit is 0 (dark).
        assert result.sampled_outcome["spin_readout_counts"].sum() == 0
        assert result.sampled_outcome["spin_readout_bits"].sum() == 0
        np.testing.assert_array_equal(
            result.sampled_outcome["spin_readout_bright_fraction"], np.zeros(4)
        )

    def test_p_up_one_ideal_detector_gives_all_bright(self) -> None:
        """p_↑ = 1 everywhere, threshold=1, λ_bright ≫ 0 → all shots bright."""
        protocol = SpinReadout(
            ion_index=0,
            detector=DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        sigma_z = np.full(3, 1.0)
        result = protocol.run(_trajectory(sigma_z), shots=500, seed=0)
        np.testing.assert_array_equal(
            result.sampled_outcome["spin_readout_bright_fraction"], np.ones(3)
        )

    def test_bright_fraction_converges_to_envelope(self) -> None:
        """Shot-averaged estimator must track the analytic envelope at 3σ."""
        detector = _noisy_detector()
        protocol = SpinReadout(
            ion_index=0,
            detector=detector,
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        sigma_z = np.linspace(-1.0, 1.0, 21)  # p_up sweep 0 → 1
        shots = 10_000
        result = protocol.run(_trajectory(sigma_z), shots=shots, seed=0)
        estimate = result.sampled_outcome["spin_readout_bright_fraction"]
        envelope = result.ideal_outcome["bright_fraction_envelope"]
        # Per-point shot-noise std ≈ sqrt(p(1−p)/N); p ∈ [0, 1] gives
        # a max std ~ 5e-3 at 10k shots. Use 3σ ≈ 1.5e-2.
        np.testing.assert_allclose(estimate, envelope, atol=1.5e-2)

    def test_seed_reproducible(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_noisy_detector(),
            lambda_bright=20.0,
            lambda_dark=0.1,
        )
        sigma_z = np.linspace(-1.0, 1.0, 7)
        first = protocol.run(_trajectory(sigma_z), shots=100, seed=42)
        second = protocol.run(_trajectory(sigma_z), shots=100, seed=42)
        np.testing.assert_array_equal(
            first.sampled_outcome["spin_readout_counts"],
            second.sampled_outcome["spin_readout_counts"],
        )
        np.testing.assert_array_equal(
            first.sampled_outcome["spin_readout_bits"],
            second.sampled_outcome["spin_readout_bits"],
        )

    def test_different_seeds_differ(self) -> None:
        protocol = SpinReadout(
            ion_index=0,
            detector=_noisy_detector(),
            lambda_bright=20.0,
            lambda_dark=0.1,
        )
        sigma_z = np.linspace(-1.0, 1.0, 7)
        first = protocol.run(_trajectory(sigma_z), shots=100, seed=1)
        second = protocol.run(_trajectory(sigma_z), shots=100, seed=2)
        assert not np.array_equal(
            first.sampled_outcome["spin_readout_counts"],
            second.sampled_outcome["spin_readout_counts"],
        )

    def test_ideal_detector_bright_fraction_matches_p_up(self) -> None:
        """Ideal detector (F=1) → envelope = p_↑ exactly → estimate tracks p_↑."""
        detector = DetectorConfig(
            efficiency=1.0,
            dark_count_rate=0.0,
            threshold=1,
        )
        protocol = SpinReadout(
            ion_index=0,
            detector=detector,
            lambda_bright=50.0,  # saturates Poisson — TP ≈ 1 at any threshold ≤ 50
            lambda_dark=0.0,
        )
        sigma_z = np.linspace(-1.0, 1.0, 11)
        p_up = 0.5 * (1.0 + sigma_z)
        shots = 20_000
        result = protocol.run(_trajectory(sigma_z), shots=shots, seed=0)
        # Shot noise std ≈ sqrt(p(1−p)/N) ≤ 3.5e-3 at 20k shots; 3σ ≈ 1.1e-2.
        np.testing.assert_allclose(
            result.sampled_outcome["spin_readout_bright_fraction"], p_up, atol=1.1e-2
        )

    def test_multi_ion_trajectory_reads_correct_observable(self) -> None:
        """When trajectory has σ_z_0 and σ_z_1, ion_index=1 reads sigma_z_1."""
        times = np.linspace(0.0, 1.0e-6, 3)
        meta = ResultMetadata(
            convention_version=CONVENTION_VERSION,
            request_hash="d" * 64,
            backend_name="qutip-mesolve",
            backend_version="5.0.0",
            storage_mode=StorageMode.OMITTED,
        )
        trajectory = TrajectoryResult(
            metadata=meta,
            times=times,
            expectations={
                "sigma_z_0": np.array([-1.0, -1.0, -1.0]),  # ion 0 always dark
                "sigma_z_1": np.array([1.0, 1.0, 1.0]),  # ion 1 always bright
            },
        )
        readout_ion0 = SpinReadout(
            ion_index=0,
            detector=DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        readout_ion1 = SpinReadout(
            ion_index=1,
            detector=DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        r0 = readout_ion0.run(trajectory, shots=50, seed=0)
        r1 = readout_ion1.run(trajectory, shots=50, seed=0)
        # Ion 0: always dark → bright fraction 0.
        np.testing.assert_array_equal(
            r0.sampled_outcome["spin_readout_bright_fraction"], np.zeros(3)
        )
        # Ion 1: always bright → bright fraction 1.
        np.testing.assert_array_equal(
            r1.sampled_outcome["spin_readout_bright_fraction"], np.ones(3)
        )
