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
    ParityScan,
    ResultMetadata,
    SidebandInference,
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


# ----------------------------------------------------------------------------
# ParityScan — joint two-ion readout
# ----------------------------------------------------------------------------


def _two_ion_trajectory(
    *,
    sigma_z_0: np.ndarray,
    sigma_z_1: np.ndarray,
    parity_01: np.ndarray,
    provenance_tags: tuple[str, ...] = ("demo", "ms_gate"),
    request_hash: str = "b" * 64,
) -> TrajectoryResult:
    """Two-ion trajectory carrying sigma_z_0, sigma_z_1, parity_0_1."""
    assert sigma_z_0.shape == sigma_z_1.shape == parity_01.shape
    times = np.linspace(0.0, 1.0e-6, sigma_z_0.size)
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
        expectations={
            "sigma_z_0": sigma_z_0,
            "sigma_z_1": sigma_z_1,
            "parity_0_1": parity_01,
        },
    )


def _bell_phi_plus_trajectory(n_times: int = 4) -> TrajectoryResult:
    """|Φ+⟩ Bell state: ⟨σ_z^i⟩ = 0, ⟨σ_z^0 σ_z^1⟩ = +1."""
    return _two_ion_trajectory(
        sigma_z_0=np.zeros(n_times),
        sigma_z_1=np.zeros(n_times),
        parity_01=np.ones(n_times),
    )


def _bell_psi_plus_trajectory(n_times: int = 4) -> TrajectoryResult:
    """|Ψ+⟩ Bell state: ⟨σ_z^i⟩ = 0, ⟨σ_z^0 σ_z^1⟩ = −1."""
    return _two_ion_trajectory(
        sigma_z_0=np.zeros(n_times),
        sigma_z_1=np.zeros(n_times),
        parity_01=-np.ones(n_times),
    )


class TestParityScanConstruction:
    def test_minimal_construction(self) -> None:
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        assert protocol.ion_indices == (0, 1)
        assert protocol.label == "parity_scan"

    def test_non_distinct_indices_raises(self) -> None:
        with pytest.raises(ValueError, match="must be distinct"):
            ParityScan(
                ion_indices=(0, 0),
                detector=_ideal_detector(),
                lambda_bright=20.0,
                lambda_dark=0.0,
            )

    def test_negative_index_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            ParityScan(
                ion_indices=(-1, 1),
                detector=_ideal_detector(),
                lambda_bright=20.0,
                lambda_dark=0.0,
            )

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rates must be >= 0"):
            ParityScan(
                ion_indices=(0, 1),
                detector=_ideal_detector(),
                lambda_bright=20.0,
                lambda_dark=-0.1,
            )

    def test_dark_greater_than_bright_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_dark must be <= lambda_bright"):
            ParityScan(
                ion_indices=(0, 1),
                detector=_ideal_detector(),
                lambda_bright=2.0,
                lambda_dark=5.0,
            )

    def test_frozen(self) -> None:
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            protocol.lambda_bright = 10.0  # type: ignore[misc]


class TestParityScanRunSurface:
    def test_result_is_measurement_result(self) -> None:
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        result = protocol.run(_bell_phi_plus_trajectory(), shots=100, seed=0)
        assert isinstance(result, MeasurementResult)
        assert result.shots == 100

    def test_sampled_outcome_shapes(self) -> None:
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        result = protocol.run(_bell_phi_plus_trajectory(n_times=5), shots=200, seed=0)
        assert result.sampled_outcome["parity_scan_counts_0"].shape == (200, 5)
        assert result.sampled_outcome["parity_scan_counts_1"].shape == (200, 5)
        assert result.sampled_outcome["parity_scan_bits_0"].shape == (200, 5)
        assert result.sampled_outcome["parity_scan_bits_1"].shape == (200, 5)
        assert result.sampled_outcome["parity_scan_parity"].shape == (200, 5)
        assert result.sampled_outcome["parity_scan_parity"].dtype == np.int8
        assert result.sampled_outcome["parity_scan_parity_estimate"].shape == (5,)

    def test_ideal_outcome_carries_joint_and_parity(self) -> None:
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        trajectory = _bell_phi_plus_trajectory(n_times=3)
        result = protocol.run(trajectory, shots=10, seed=0)
        assert result.ideal_outcome["p_up_0"].shape == (3,)
        assert result.ideal_outcome["p_up_1"].shape == (3,)
        np.testing.assert_array_equal(result.ideal_outcome["parity"], np.ones(3))
        np.testing.assert_allclose(
            result.ideal_outcome["joint_probabilities"].sum(axis=0), np.ones(3)
        )

    def test_trajectory_hash_and_provenance(self) -> None:
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        trajectory = _two_ion_trajectory(
            sigma_z_0=np.array([0.0]),
            sigma_z_1=np.array([0.0]),
            parity_01=np.array([1.0]),
            provenance_tags=("demo", "ms_gate"),
            request_hash="d" * 64,
        )
        result = protocol.run(trajectory, shots=10, seed=0, provenance_tags=("unit-test",))
        assert result.trajectory_hash == "d" * 64
        assert result.metadata.provenance_tags == (
            "demo",
            "ms_gate",
            "measurement",
            "parity_scan",
            "unit-test",
        )

    def test_label_routes_sampled_outcome_keys(self) -> None:
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
            label="bell_scan",
        )
        result = protocol.run(_bell_phi_plus_trajectory(n_times=2), shots=10, seed=0)
        assert "bell_scan_counts_0" in result.sampled_outcome
        assert "bell_scan_parity" in result.sampled_outcome
        assert "parity_scan_parity" not in result.sampled_outcome


class TestParityScanRunErrors:
    def test_missing_marginal_raises(self) -> None:
        times = np.linspace(0.0, 1.0e-6, 3)
        meta = ResultMetadata(
            convention_version=CONVENTION_VERSION,
            request_hash="a" * 64,
            backend_name="qutip-mesolve",
            backend_version="5.0.0",
            storage_mode=StorageMode.OMITTED,
        )
        trajectory = TrajectoryResult(
            metadata=meta,
            times=times,
            expectations={
                "sigma_z_0": np.zeros(3),
                # missing sigma_z_1 and parity_0_1
            },
        )
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        with pytest.raises(ConventionError, match="sigma_z_1"):
            protocol.run(trajectory, shots=10, seed=0)

    def test_missing_parity_raises(self) -> None:
        times = np.linspace(0.0, 1.0e-6, 2)
        meta = ResultMetadata(
            convention_version=CONVENTION_VERSION,
            request_hash="a" * 64,
            backend_name="qutip-mesolve",
            backend_version="5.0.0",
            storage_mode=StorageMode.OMITTED,
        )
        trajectory = TrajectoryResult(
            metadata=meta,
            times=times,
            expectations={
                "sigma_z_0": np.zeros(2),
                "sigma_z_1": np.zeros(2),
            },
        )
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        with pytest.raises(ConventionError, match="parity_0_1"):
            protocol.run(trajectory, shots=10, seed=0)

    def test_unphysical_joint_raises(self) -> None:
        """Marginals + parity that imply negative joint probabilities → error."""
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        # ⟨σ_z_0⟩ = ⟨σ_z_1⟩ = 0 but ⟨σ_z_0 σ_z_1⟩ = +1 → P(↑↓) = P(↓↑) = 0
        # which is legal. Push parity past +1 to force negatives.
        trajectory = _two_ion_trajectory(
            sigma_z_0=np.zeros(1),
            sigma_z_1=np.zeros(1),
            parity_01=np.array([1.5]),
        )
        with pytest.raises(ValueError, match="unphysical"):
            protocol.run(trajectory, shots=10, seed=0)

    def test_zero_shots_raises(self) -> None:
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        with pytest.raises(ValueError, match="shots must be >= 1"):
            protocol.run(_bell_phi_plus_trajectory(), shots=0, seed=0)


class TestParityScanJointSampling:
    def test_phi_plus_all_agree(self) -> None:
        """|Φ+⟩: perfect correlation → every shot parity is +1."""
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1),
            lambda_bright=30.0,
            lambda_dark=0.0,
        )
        result = protocol.run(_bell_phi_plus_trajectory(n_times=3), shots=300, seed=0)
        parities = result.sampled_outcome["parity_scan_parity"]
        # With ideal detector + clean Poisson saturation, every shot
        # should produce parity = +1 for |Φ+⟩.
        assert np.all(parities == 1)
        np.testing.assert_array_equal(
            result.sampled_outcome["parity_scan_parity_estimate"],
            np.ones(3),
        )

    def test_psi_plus_all_anticorrelated(self) -> None:
        """|Ψ+⟩: perfect anti-correlation → every shot parity is −1."""
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1),
            lambda_bright=30.0,
            lambda_dark=0.0,
        )
        result = protocol.run(_bell_psi_plus_trajectory(n_times=3), shots=300, seed=0)
        parities = result.sampled_outcome["parity_scan_parity"]
        assert np.all(parities == -1)
        np.testing.assert_array_equal(
            result.sampled_outcome["parity_scan_parity_estimate"],
            -np.ones(3),
        )

    def test_product_state_factorises(self) -> None:
        """Product state |↑↑⟩: sigma_z_i = +1, parity = +1 → parity est +1."""
        trajectory = _two_ion_trajectory(
            sigma_z_0=np.ones(3),
            sigma_z_1=np.ones(3),
            parity_01=np.ones(3),
        )
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1),
            lambda_bright=30.0,
            lambda_dark=0.0,
        )
        result = protocol.run(trajectory, shots=200, seed=0)
        np.testing.assert_array_equal(
            result.sampled_outcome["parity_scan_parity_estimate"],
            np.ones(3),
        )

    def test_parity_estimator_converges_to_envelope(self) -> None:
        """Noisy detector: parity estimate tracks the linear-fidelity envelope."""
        detector = _noisy_detector()
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=detector,
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        # Sweep the parity observable from −1 to +1 while keeping marginals
        # at 0 (both ions maximally mixed σ_z basis) — pure entanglement
        # contrast with no single-ion bias.
        n_times = 21
        trajectory = _two_ion_trajectory(
            sigma_z_0=np.zeros(n_times),
            sigma_z_1=np.zeros(n_times),
            parity_01=np.linspace(-1.0, 1.0, n_times),
        )
        shots = 10_000
        result = protocol.run(trajectory, shots=shots, seed=0)
        estimate = result.sampled_outcome["parity_scan_parity_estimate"]
        envelope = result.ideal_outcome["parity_envelope"]
        # Per-point shot std of a ±1 sample estimator ≈ sqrt(1/N); use 4σ.
        tolerance = 4.0 / np.sqrt(shots)
        np.testing.assert_allclose(estimate, envelope, atol=tolerance)

    def test_envelope_contracts_by_fidelity_squared(self) -> None:
        """Envelope = (TP + TN − 1)² · ⟨σ_z σ_z⟩ + (TP − TN)² at marginals = 0.

        Expansion of the 4-term projective envelope at σ_z_0 = σ_z_1 = 0
        separates into (a) a parity contrast scaled by the fidelity-
        contrast squared (TP + TN − 1)² — the entanglement-visibility
        shrinkage — and (b) a detector-asymmetry offset (TP − TN)²
        that vanishes for symmetric detectors (TP = TN) but is nonzero
        whenever dark-count and threshold choices leave TP ≠ TN.
        """
        detector = _noisy_detector()
        fid = detector.classification_fidelity(lambda_bright=25.0, lambda_dark=0.0)
        tp = fid["true_positive_rate"]
        tn = fid["true_negative_rate"]
        contrast = tp + tn - 1.0
        asymmetry = tp - tn

        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=detector,
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        trajectory = _two_ion_trajectory(
            sigma_z_0=np.zeros(5),
            sigma_z_1=np.zeros(5),
            parity_01=np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
        )
        result = protocol.run(trajectory, shots=10, seed=0)
        envelope = result.ideal_outcome["parity_envelope"]
        expected = contrast**2 * trajectory.expectations["parity_0_1"] + asymmetry**2
        np.testing.assert_allclose(envelope, expected)

    def test_seed_reproducible(self) -> None:
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_noisy_detector(),
            lambda_bright=20.0,
            lambda_dark=0.1,
        )
        trajectory = _bell_phi_plus_trajectory(n_times=4)
        first = protocol.run(trajectory, shots=100, seed=42)
        second = protocol.run(trajectory, shots=100, seed=42)
        np.testing.assert_array_equal(
            first.sampled_outcome["parity_scan_parity"],
            second.sampled_outcome["parity_scan_parity"],
        )

    def test_joint_reconstruction_correctness(self) -> None:
        """Reconstructed joint for |Φ+⟩: P(↑↑)=P(↓↓)=1/2, P(↑↓)=P(↓↑)=0."""
        protocol = ParityScan(
            ion_indices=(0, 1),
            detector=_ideal_detector(),
            lambda_bright=10.0,
            lambda_dark=0.0,
        )
        result = protocol.run(_bell_phi_plus_trajectory(n_times=2), shots=10, seed=0)
        joint = result.ideal_outcome["joint_probabilities"]  # shape (4, 2)
        np.testing.assert_allclose(joint[0], 0.5)  # P(↑↑)
        np.testing.assert_allclose(joint[1], 0.0, atol=1e-12)  # P(↑↓)
        np.testing.assert_allclose(joint[2], 0.0, atol=1e-12)  # P(↓↑)
        np.testing.assert_allclose(joint[3], 0.5)  # P(↓↓)


# ----------------------------------------------------------------------------
# SidebandInference — RSB / BSB ratio-based n̄ inference
# ----------------------------------------------------------------------------


def _sideband_trajectory(
    sigma_z: np.ndarray,
    *,
    ion_index: int = 0,
    provenance_tags: tuple[str, ...] = ("demo", "sideband"),
    request_hash: str = "e" * 64,
    times: np.ndarray | None = None,
) -> TrajectoryResult:
    t = times if times is not None else np.linspace(0.0, 1.0e-6, sigma_z.size)
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
        times=t,
        expectations={f"sigma_z_{ion_index}": sigma_z},
    )


def _from_nbar(
    nbar: float,
    *,
    n_times: int = 5,
    scale: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Build matched-time (RSB, BSB) ⟨σ_z⟩ trajectories for a given n̄.

    Uses the short-time ratio p_up_rsb / p_up_bsb = n̄/(n̄+1). We pick
    p_up_bsb(t) = scale · t_index (a linear ramp into the short-time
    regime), then derive p_up_rsb from the ratio. Returns
    (sigma_z_rsb, sigma_z_bsb) with ⟨σ_z⟩ = 2·p_up − 1.
    """
    ramp = np.linspace(scale, scale * n_times, n_times)
    p_up_bsb = ramp
    p_up_rsb = (nbar / (nbar + 1.0)) * ramp
    return 2.0 * p_up_rsb - 1.0, 2.0 * p_up_bsb - 1.0


class TestSidebandInferenceConstruction:
    def test_minimal_construction(self) -> None:
        protocol = SidebandInference(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        assert protocol.ion_index == 0
        assert protocol.label == "sideband_inference"

    def test_negative_ion_index_raises(self) -> None:
        with pytest.raises(ValueError, match="ion_index must be >= 0"):
            SidebandInference(
                ion_index=-1,
                detector=_ideal_detector(),
                lambda_bright=20.0,
                lambda_dark=0.0,
            )

    def test_dark_greater_than_bright_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_dark must be <= lambda_bright"):
            SidebandInference(
                ion_index=0,
                detector=_ideal_detector(),
                lambda_bright=2.0,
                lambda_dark=5.0,
            )

    def test_frozen(self) -> None:
        protocol = SidebandInference(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=20.0,
            lambda_dark=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            protocol.ion_index = 1  # type: ignore[misc]


class TestSidebandInferenceRunSurface:
    def test_result_is_measurement_result(self) -> None:
        rsb, bsb = _from_nbar(0.5, n_times=4)
        protocol = SidebandInference(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        result = protocol.run(
            rsb_trajectory=_sideband_trajectory(rsb),
            bsb_trajectory=_sideband_trajectory(bsb),
            shots=100,
            seed=0,
        )
        assert isinstance(result, MeasurementResult)
        assert result.shots == 100

    def test_sampled_outcome_shapes(self) -> None:
        rsb, bsb = _from_nbar(0.3, n_times=5)
        protocol = SidebandInference(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        result = protocol.run(
            rsb_trajectory=_sideband_trajectory(rsb),
            bsb_trajectory=_sideband_trajectory(bsb),
            shots=200,
            seed=0,
        )
        assert result.sampled_outcome["sideband_inference_rsb_counts"].shape == (200, 5)
        assert result.sampled_outcome["sideband_inference_bsb_counts"].shape == (200, 5)
        assert result.sampled_outcome["sideband_inference_rsb_bright_fraction"].shape == (5,)
        assert result.sampled_outcome["sideband_inference_nbar_estimate"].shape == (5,)
        assert result.sampled_outcome["sideband_inference_nbar_from_raw_ratio"].shape == (5,)

    def test_ideal_outcome_carries_p_up_and_ideal_ratio(self) -> None:
        rsb, bsb = _from_nbar(1.0, n_times=4)  # n̄ = 1
        protocol = SidebandInference(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        result = protocol.run(
            rsb_trajectory=_sideband_trajectory(rsb),
            bsb_trajectory=_sideband_trajectory(bsb),
            shots=10,
            seed=0,
        )
        np.testing.assert_allclose(result.ideal_outcome["p_up_bsb"], 0.5 * (1.0 + bsb))
        np.testing.assert_allclose(result.ideal_outcome["p_up_rsb"], 0.5 * (1.0 + rsb))
        # n̄ = 1 everywhere since the trajectory was constructed that way.
        np.testing.assert_allclose(result.ideal_outcome["nbar_from_ideal_ratio"], 1.0)

    def test_trajectory_hashes_and_provenance(self) -> None:
        rsb, bsb = _from_nbar(0.5)
        protocol = SidebandInference(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        result = protocol.run(
            rsb_trajectory=_sideband_trajectory(
                rsb,
                provenance_tags=("demo", "rsb_scan"),
                request_hash="f" * 64,
            ),
            bsb_trajectory=_sideband_trajectory(
                bsb,
                provenance_tags=("demo", "bsb_scan"),
                request_hash="9" * 64,
            ),
            shots=10,
            seed=0,
            provenance_tags=("unit-test",),
        )
        assert result.trajectory_hash == "f" * 64
        assert str(result.ideal_outcome["bsb_trajectory_hash"]) == "9" * 64
        assert result.metadata.provenance_tags == (
            "demo",
            "rsb_scan",
            "measurement",
            "sideband_inference",
            "unit-test",
        )

    def test_label_routing(self) -> None:
        rsb, bsb = _from_nbar(0.5)
        protocol = SidebandInference(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
            label="thermometry",
        )
        result = protocol.run(
            rsb_trajectory=_sideband_trajectory(rsb),
            bsb_trajectory=_sideband_trajectory(bsb),
            shots=10,
            seed=0,
        )
        assert "thermometry_nbar_estimate" in result.sampled_outcome
        assert "thermometry_rsb_counts" in result.sampled_outcome
        assert "thermometry_bsb_counts" in result.sampled_outcome
        assert "sideband_inference_nbar_estimate" not in result.sampled_outcome


class TestSidebandInferenceRunErrors:
    def test_missing_observable_rsb_raises(self) -> None:
        protocol = SidebandInference(
            ion_index=5,
            detector=_ideal_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        with pytest.raises(ConventionError, match="sigma_z_5"):
            protocol.run(
                rsb_trajectory=_sideband_trajectory(np.zeros(3)),
                bsb_trajectory=_sideband_trajectory(np.zeros(3)),
                shots=10,
                seed=0,
            )

    def test_mismatched_time_grids_raise(self) -> None:
        protocol = SidebandInference(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        rsb_traj = _sideband_trajectory(np.zeros(3), times=np.linspace(0.0, 1e-6, 3))
        bsb_traj = _sideband_trajectory(np.zeros(3), times=np.linspace(0.0, 2e-6, 3))
        with pytest.raises(ValueError, match="same time grid"):
            protocol.run(
                rsb_trajectory=rsb_traj,
                bsb_trajectory=bsb_traj,
                shots=10,
                seed=0,
            )

    def test_zero_contrast_detector_raises(self) -> None:
        """TP + TN − 1 ≤ 0 makes fidelity inversion ill-defined."""
        degenerate = DetectorConfig(
            efficiency=0.0,
            dark_count_rate=0.0,
            threshold=1,
        )
        protocol = SidebandInference(
            ion_index=0,
            detector=degenerate,
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        rsb, bsb = _from_nbar(0.5)
        with pytest.raises(ValueError, match="contrast"):
            protocol.run(
                rsb_trajectory=_sideband_trajectory(rsb),
                bsb_trajectory=_sideband_trajectory(bsb),
                shots=10,
                seed=0,
            )

    def test_zero_shots_raises(self) -> None:
        protocol = SidebandInference(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        rsb, bsb = _from_nbar(0.5)
        with pytest.raises(ValueError, match="shots must be >= 1"):
            protocol.run(
                rsb_trajectory=_sideband_trajectory(rsb),
                bsb_trajectory=_sideband_trajectory(bsb),
                shots=0,
                seed=0,
            )


class TestSidebandInferenceRatioRecovery:
    def test_ground_state_nbar_zero(self) -> None:
        """RSB is dark (p_up_rsb = 0 everywhere) → inferred n̄ = 0."""
        # p_up_bsb small-but-nonzero; p_up_rsb identically 0.
        p_up_bsb = np.linspace(0.01, 0.1, 5)
        p_up_rsb = np.zeros_like(p_up_bsb)
        rsb = 2.0 * p_up_rsb - 1.0
        bsb = 2.0 * p_up_bsb - 1.0
        protocol = SidebandInference(
            ion_index=0,
            detector=DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1),
            lambda_bright=50.0,  # saturate → TP ≈ 1
            lambda_dark=0.0,
        )
        shots = 50_000
        result = protocol.run(
            rsb_trajectory=_sideband_trajectory(rsb),
            bsb_trajectory=_sideband_trajectory(bsb),
            shots=shots,
            seed=0,
        )
        nbar_est = result.sampled_outcome["sideband_inference_nbar_estimate"]
        # Shot noise on p̂_rsb is ~ sqrt(p(1-p)/N) ≈ 0 for true p=0 but
        # non-zero when fidelity-inverted. Expect very small nbar.
        assert np.nanmax(np.abs(nbar_est)) < 0.05

    def test_thermal_like_nbar_recovery(self) -> None:
        """Constructed trajectory with ratio = n̄/(n̄+1) → inferred n̄ = n̄."""
        true_nbar = 0.75
        rsb, bsb = _from_nbar(true_nbar, n_times=5, scale=0.02)
        protocol = SidebandInference(
            ion_index=0,
            detector=DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1),
            lambda_bright=50.0,
            lambda_dark=0.0,
        )
        shots = 50_000
        result = protocol.run(
            rsb_trajectory=_sideband_trajectory(rsb),
            bsb_trajectory=_sideband_trajectory(bsb),
            shots=shots,
            seed=0,
        )
        # ideal ratio trajectory should match true n̄ exactly (construction)
        np.testing.assert_allclose(result.ideal_outcome["nbar_from_ideal_ratio"], true_nbar)
        # Sampled estimate converges to n̄ within shot noise (at low p,
        # the ratio variance is large; use a generous tolerance for this
        # deliberately-small-signal fixture).
        nbar_est = result.sampled_outcome["sideband_inference_nbar_estimate"]
        np.testing.assert_allclose(nbar_est, true_nbar, atol=0.5)

    def test_seed_reproducible(self) -> None:
        rsb, bsb = _from_nbar(0.5)
        protocol = SidebandInference(
            ion_index=0,
            detector=_noisy_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        first = protocol.run(
            rsb_trajectory=_sideband_trajectory(rsb),
            bsb_trajectory=_sideband_trajectory(bsb),
            shots=100,
            seed=42,
        )
        second = protocol.run(
            rsb_trajectory=_sideband_trajectory(rsb),
            bsb_trajectory=_sideband_trajectory(bsb),
            shots=100,
            seed=42,
        )
        np.testing.assert_array_equal(
            first.sampled_outcome["sideband_inference_rsb_counts"],
            second.sampled_outcome["sideband_inference_rsb_counts"],
        )
        np.testing.assert_array_equal(
            first.sampled_outcome["sideband_inference_bsb_counts"],
            second.sampled_outcome["sideband_inference_bsb_counts"],
        )

    def test_rsb_and_bsb_streams_are_independent(self) -> None:
        """Same p_up on RSB and BSB should yield different bit patterns."""
        # Matching p_up → ratio = 1 → n̄ undefined, but that's OK — we're
        # testing stream independence here, not inference correctness.
        matching = 0.3 * np.ones(5)
        sigma_z = 2.0 * matching - 1.0
        protocol = SidebandInference(
            ion_index=0,
            detector=_ideal_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        result = protocol.run(
            rsb_trajectory=_sideband_trajectory(sigma_z),
            bsb_trajectory=_sideband_trajectory(sigma_z),
            shots=200,
            seed=7,
        )
        assert not np.array_equal(
            result.sampled_outcome["sideband_inference_rsb_counts"],
            result.sampled_outcome["sideband_inference_bsb_counts"],
        )

    def test_fidelity_correction_improves_over_raw_ratio(self) -> None:
        """At finite fidelity, the corrected estimator should be closer to truth.

        At low-visibility trajectories where both sidebands are deep in
        the short-time regime, the raw-ratio estimate is shrunk by the
        detector contrast while the corrected estimate is not. Compare
        mean absolute errors against the true n̄.
        """
        true_nbar = 0.5
        rsb, bsb = _from_nbar(true_nbar, n_times=8, scale=0.05)
        protocol = SidebandInference(
            ion_index=0,
            detector=_noisy_detector(),
            lambda_bright=25.0,
            lambda_dark=0.0,
        )
        shots = 50_000
        result = protocol.run(
            rsb_trajectory=_sideband_trajectory(rsb),
            bsb_trajectory=_sideband_trajectory(bsb),
            shots=shots,
            seed=0,
        )
        corrected = result.sampled_outcome["sideband_inference_nbar_estimate"]
        raw = result.sampled_outcome["sideband_inference_nbar_from_raw_ratio"]
        # Compare MAE vs truth across the trajectory; corrected should
        # win on average in this regime.
        err_corrected = float(np.nanmean(np.abs(corrected - true_nbar)))
        err_raw = float(np.nanmean(np.abs(raw - true_nbar)))
        assert err_corrected < err_raw
