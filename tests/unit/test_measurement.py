# SPDX-License-Identifier: MIT
"""Unit tests for the measurement layer (Dispatch H scope).

Covers :class:`MeasurementResult` schema enforcement, the
:class:`BernoulliChannel` stochastic contract, and the
:func:`sample_outcome` orchestrator's metadata plumbing. Tests are
backend-agnostic and do not import QuTiP.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iontrap_dynamics import (
    CONVENTION_VERSION,
    BernoulliChannel,
    BinomialChannel,
    ConventionError,
    DetectorConfig,
    IonTrapError,
    MeasurementResult,
    PoissonChannel,
    Result,
    ResultMetadata,
    StorageMode,
    TrajectoryResult,
    sample_outcome,
)

# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _measurement_metadata(storage_mode: StorageMode = StorageMode.OMITTED) -> ResultMetadata:
    return ResultMetadata(
        convention_version=CONVENTION_VERSION,
        request_hash="0" * 64,
        backend_name="test-backend",
        backend_version="0.0.0",
        storage_mode=storage_mode,
    )


def _trajectory_metadata() -> ResultMetadata:
    return ResultMetadata(
        convention_version=CONVENTION_VERSION,
        request_hash="a" * 64,
        backend_name="qutip-mesolve",
        backend_version="5.0.0",
        storage_mode=StorageMode.OMITTED,
        provenance_tags=("demo", "carrier_rabi"),
    )


# ----------------------------------------------------------------------------
# MeasurementResult construction surface
# ----------------------------------------------------------------------------


class TestMeasurementResultConstruction:
    def test_minimal_construction(self) -> None:
        result = MeasurementResult(
            metadata=_measurement_metadata(),
            shots=100,
            rng_seed=42,
        )
        assert result.shots == 100
        assert result.rng_seed == 42
        assert result.ideal_outcome == {}
        assert result.sampled_outcome == {}
        assert result.trajectory_hash is None

    def test_measurement_result_is_a_result(self) -> None:
        """Phase 1 sibling hook — blanket ``except Result`` must catch it."""
        result = MeasurementResult(
            metadata=_measurement_metadata(),
            shots=1,
            rng_seed=None,
        )
        assert isinstance(result, Result)

    def test_shots_zero_raises(self) -> None:
        with pytest.raises(ConventionError, match="shots >= 1"):
            MeasurementResult(
                metadata=_measurement_metadata(),
                shots=0,
                rng_seed=None,
            )

    def test_non_omitted_storage_mode_raises(self) -> None:
        with pytest.raises(ConventionError, match="storage_mode=OMITTED"):
            MeasurementResult(
                metadata=_measurement_metadata(StorageMode.EAGER),
                shots=1,
                rng_seed=None,
            )

    def test_conventionerror_subclasses_iontraperror(self) -> None:
        with pytest.raises(IonTrapError):
            MeasurementResult(
                metadata=_measurement_metadata(),
                shots=-1,
                rng_seed=None,
            )

    def test_attribute_assignment_raises(self) -> None:
        result = MeasurementResult(
            metadata=_measurement_metadata(),
            shots=1,
            rng_seed=None,
        )
        with pytest.raises(FrozenInstanceError):
            result.shots = 2  # type: ignore[misc]

    def test_positional_construction_forbidden(self) -> None:
        with pytest.raises(TypeError):
            MeasurementResult(_measurement_metadata(), 1, None)  # type: ignore[misc]


# ----------------------------------------------------------------------------
# BernoulliChannel stochastic contract
# ----------------------------------------------------------------------------


class TestBernoulliChannelSample:
    def test_output_shape_and_dtype(self) -> None:
        channel = BernoulliChannel()
        probs = np.array([0.2, 0.5, 0.8])
        bits = channel.sample(probs, shots=1000, rng=np.random.default_rng(0))
        assert bits.shape == (1000, 3)
        assert bits.dtype == np.int8
        assert set(np.unique(bits).tolist()).issubset({0, 1})

    def test_deterministic_given_seed(self) -> None:
        """Same (seed, probs, shots) must reproduce the exact bit pattern."""
        channel = BernoulliChannel()
        probs = np.array([0.1, 0.5, 0.9])
        first = channel.sample(probs, shots=64, rng=np.random.default_rng(123))
        second = channel.sample(probs, shots=64, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(first, second)

    def test_mean_converges_to_probability(self) -> None:
        """Shot-averaged rate must track the input probability in the large-shot limit."""
        channel = BernoulliChannel()
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        bits = channel.sample(probs, shots=20_000, rng=np.random.default_rng(7))
        estimated = bits.mean(axis=0)
        np.testing.assert_allclose(estimated, probs, atol=0.01)

    def test_p_zero_is_all_zeros(self) -> None:
        channel = BernoulliChannel()
        bits = channel.sample(np.array([0.0, 0.0]), shots=50, rng=np.random.default_rng(0))
        assert bits.sum() == 0

    def test_p_one_is_all_ones(self) -> None:
        channel = BernoulliChannel()
        bits = channel.sample(np.array([1.0, 1.0]), shots=50, rng=np.random.default_rng(0))
        assert bits.sum() == 100

    def test_out_of_range_probability_raises(self) -> None:
        channel = BernoulliChannel()
        with pytest.raises(ValueError, match=r"probabilities must lie in \[0, 1\]"):
            channel.sample(np.array([0.5, 1.5]), shots=1, rng=np.random.default_rng(0))

    def test_negative_probability_raises(self) -> None:
        channel = BernoulliChannel()
        with pytest.raises(ValueError, match=r"probabilities must lie in \[0, 1\]"):
            channel.sample(np.array([-0.1, 0.5]), shots=1, rng=np.random.default_rng(0))

    def test_non_1d_input_raises(self) -> None:
        channel = BernoulliChannel()
        with pytest.raises(ValueError, match="must be 1-D"):
            channel.sample(np.array([[0.2, 0.5]]), shots=1, rng=np.random.default_rng(0))

    def test_shots_zero_raises(self) -> None:
        channel = BernoulliChannel()
        with pytest.raises(ValueError, match="shots must be >= 1"):
            channel.sample(np.array([0.5]), shots=0, rng=np.random.default_rng(0))


# ----------------------------------------------------------------------------
# BinomialChannel stochastic contract
# ----------------------------------------------------------------------------


class TestBinomialChannelSample:
    def test_output_shape_and_dtype(self) -> None:
        channel = BinomialChannel()
        probs = np.array([0.2, 0.5, 0.8])
        counts = channel.sample(probs, shots=500, rng=np.random.default_rng(0))
        assert counts.shape == (3,)
        assert counts.dtype == np.int64

    def test_counts_in_range(self) -> None:
        channel = BinomialChannel()
        probs = np.linspace(0.0, 1.0, 11)
        counts = channel.sample(probs, shots=200, rng=np.random.default_rng(0))
        assert np.all(counts >= 0)
        assert np.all(counts <= 200)

    def test_deterministic_given_seed(self) -> None:
        channel = BinomialChannel()
        probs = np.array([0.1, 0.5, 0.9])
        first = channel.sample(probs, shots=256, rng=np.random.default_rng(123))
        second = channel.sample(probs, shots=256, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(first, second)

    def test_mean_converges_to_shots_times_probability(self) -> None:
        """Binomial(n, p) has mean n·p and std sqrt(n·p·(1−p))."""
        channel = BinomialChannel()
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        shots = 5_000
        # Average over many repeats to estimate the distribution mean.
        rng = np.random.default_rng(7)
        trials = np.stack(
            [channel.sample(probs, shots=shots, rng=rng) for _ in range(200)],
            axis=0,
        )
        estimated_rate = trials.mean(axis=0) / shots
        np.testing.assert_allclose(estimated_rate, probs, atol=0.005)

    def test_distributional_equivalence_to_summed_bernoulli(self) -> None:
        """Binomial counts have the same distribution as sum-of-Bernoulli bits.

        Not bit-identical (different RNG consumption) but the empirical
        rate agrees to within Monte-Carlo tolerance.
        """
        bernoulli = BernoulliChannel()
        binomial = BinomialChannel()
        probs = np.array([0.25, 0.5, 0.75])
        shots = 2_000

        rng_b = np.random.default_rng(42)
        bernoulli_counts = (
            bernoulli.sample(probs, shots=shots, rng=rng_b).sum(axis=0).astype(np.int64)
        )
        rng_p = np.random.default_rng(84)
        binomial_counts = binomial.sample(probs, shots=shots, rng=rng_p)

        # Shot-noise std is ≈ sqrt(N·p·(1−p)); compare to well within 5σ.
        std_bound = np.sqrt(shots * probs * (1.0 - probs))
        np.testing.assert_array_less(np.abs(bernoulli_counts - binomial_counts), 5.0 * std_bound)

    def test_p_zero_is_all_zeros(self) -> None:
        channel = BinomialChannel()
        counts = channel.sample(np.array([0.0, 0.0]), shots=100, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(counts, np.array([0, 0]))

    def test_p_one_equals_shots(self) -> None:
        channel = BinomialChannel()
        counts = channel.sample(np.array([1.0, 1.0]), shots=100, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(counts, np.array([100, 100]))

    def test_out_of_range_probability_raises(self) -> None:
        channel = BinomialChannel()
        with pytest.raises(ValueError, match=r"probabilities must lie in \[0, 1\]"):
            channel.sample(np.array([0.5, 1.5]), shots=1, rng=np.random.default_rng(0))

    def test_non_1d_input_raises(self) -> None:
        channel = BinomialChannel()
        with pytest.raises(ValueError, match="must be 1-D"):
            channel.sample(np.array([[0.2, 0.5]]), shots=1, rng=np.random.default_rng(0))

    def test_shots_zero_raises(self) -> None:
        channel = BinomialChannel()
        with pytest.raises(ValueError, match="shots must be >= 1"):
            channel.sample(np.array([0.5]), shots=0, rng=np.random.default_rng(0))

    def test_default_label_is_binomial(self) -> None:
        assert BinomialChannel().label == "binomial"


# ----------------------------------------------------------------------------
# PoissonChannel stochastic contract
# ----------------------------------------------------------------------------


class TestPoissonChannelSample:
    def test_output_shape_and_dtype(self) -> None:
        channel = PoissonChannel()
        rates = np.array([0.5, 2.0, 10.0])
        counts = channel.sample(rates, shots=200, rng=np.random.default_rng(0))
        assert counts.shape == (200, 3)
        assert counts.dtype == np.int64
        assert np.all(counts >= 0)

    def test_deterministic_given_seed(self) -> None:
        channel = PoissonChannel()
        rates = np.array([1.0, 5.0, 12.0])
        first = channel.sample(rates, shots=50, rng=np.random.default_rng(123))
        second = channel.sample(rates, shots=50, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(first, second)

    def test_empirical_mean_matches_rate(self) -> None:
        """Shot-averaged count must track the Poisson rate as shots grow."""
        channel = PoissonChannel()
        rates = np.array([0.5, 2.0, 5.0, 10.0])
        counts = channel.sample(rates, shots=20_000, rng=np.random.default_rng(7))
        estimated = counts.mean(axis=0)
        # Standard error of the empirical mean is sqrt(λ / N); 20k shots
        # gives ~1% on rate ≈ 10.
        np.testing.assert_allclose(estimated, rates, atol=0.05)

    def test_empirical_variance_matches_rate(self) -> None:
        """Poisson has variance = mean; shot-wise variance should track rate."""
        channel = PoissonChannel()
        rates = np.array([1.0, 5.0, 15.0])
        counts = channel.sample(rates, shots=50_000, rng=np.random.default_rng(42))
        estimated_var = counts.var(axis=0, ddof=1)
        # Sampling std of variance estimator ≈ sqrt(2/(N−1)) · λ; large N
        # gives sub-1% relative accuracy — use 5% tolerance for safety.
        np.testing.assert_allclose(estimated_var, rates, rtol=0.05)

    def test_rate_zero_is_all_zeros(self) -> None:
        channel = PoissonChannel()
        counts = channel.sample(np.array([0.0, 0.0]), shots=100, rng=np.random.default_rng(0))
        assert counts.sum() == 0

    def test_negative_rate_raises(self) -> None:
        channel = PoissonChannel()
        with pytest.raises(ValueError, match="rates must be >= 0"):
            channel.sample(np.array([1.0, -0.1]), shots=1, rng=np.random.default_rng(0))

    def test_non_1d_input_raises(self) -> None:
        channel = PoissonChannel()
        with pytest.raises(ValueError, match="must be 1-D"):
            channel.sample(np.array([[1.0, 2.0]]), shots=1, rng=np.random.default_rng(0))

    def test_shots_zero_raises(self) -> None:
        channel = PoissonChannel()
        with pytest.raises(ValueError, match="shots must be >= 1"):
            channel.sample(np.array([1.0]), shots=0, rng=np.random.default_rng(0))

    def test_ideal_label_is_rate(self) -> None:
        assert PoissonChannel.ideal_label == "rate"

    def test_default_label_is_poisson(self) -> None:
        assert PoissonChannel().label == "poisson"


# ----------------------------------------------------------------------------
# DetectorConfig construction + rate transform + thresholding + fidelity
# ----------------------------------------------------------------------------


class TestDetectorConfigConstruction:
    def test_minimal_construction(self) -> None:
        det = DetectorConfig(efficiency=0.3, dark_count_rate=0.1, threshold=3)
        assert det.efficiency == 0.3
        assert det.dark_count_rate == 0.1
        assert det.threshold == 3
        assert det.label == "detector"

    def test_efficiency_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match=r"efficiency must lie in \[0, 1\]"):
            DetectorConfig(efficiency=1.5, dark_count_rate=0.0, threshold=1)
        with pytest.raises(ValueError, match=r"efficiency must lie in \[0, 1\]"):
            DetectorConfig(efficiency=-0.1, dark_count_rate=0.0, threshold=1)

    def test_negative_dark_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="dark_count_rate must be >= 0"):
            DetectorConfig(efficiency=0.5, dark_count_rate=-0.2, threshold=1)

    def test_threshold_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be >= 1"):
            DetectorConfig(efficiency=0.5, dark_count_rate=0.0, threshold=0)

    def test_frozen(self) -> None:
        det = DetectorConfig(efficiency=0.3, dark_count_rate=0.1, threshold=3)
        with pytest.raises(FrozenInstanceError):
            det.efficiency = 0.5  # type: ignore[misc]

    def test_positional_construction_forbidden(self) -> None:
        with pytest.raises(TypeError):
            DetectorConfig(0.3, 0.1, 3)  # type: ignore[misc]


class TestDetectorApply:
    def test_rate_transform(self) -> None:
        det = DetectorConfig(efficiency=0.4, dark_count_rate=0.5, threshold=1)
        rates = np.array([0.0, 1.0, 10.0])
        detected = det.apply(rates)
        np.testing.assert_allclose(detected, np.array([0.5, 0.9, 4.5]))

    def test_unit_efficiency_no_dark_passes_through(self) -> None:
        det = DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1)
        rates = np.array([0.5, 2.0, 10.0])
        np.testing.assert_array_equal(det.apply(rates), rates)

    def test_zero_efficiency_floor_is_dark_rate(self) -> None:
        det = DetectorConfig(efficiency=0.0, dark_count_rate=0.3, threshold=1)
        rates = np.array([0.0, 5.0, 100.0])
        np.testing.assert_array_equal(det.apply(rates), np.full_like(rates, 0.3))

    def test_output_is_float64(self) -> None:
        det = DetectorConfig(efficiency=0.5, dark_count_rate=0.1, threshold=1)
        detected = det.apply(np.array([1.0, 2.0]))
        assert detected.dtype == np.float64

    def test_negative_rate_raises(self) -> None:
        det = DetectorConfig(efficiency=0.5, dark_count_rate=0.0, threshold=1)
        with pytest.raises(ValueError, match="emitted_rate must be >= 0"):
            det.apply(np.array([1.0, -0.2]))


class TestDetectorDiscriminate:
    def test_thresholding(self) -> None:
        det = DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=3)
        counts = np.array([0, 1, 2, 3, 4, 100])
        bits = det.discriminate(counts)
        np.testing.assert_array_equal(bits, np.array([0, 0, 0, 1, 1, 1], dtype=np.int8))

    def test_threshold_one_any_click_is_bright(self) -> None:
        det = DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1)
        counts = np.array([[0, 1, 5], [2, 0, 0]])
        bits = det.discriminate(counts)
        np.testing.assert_array_equal(bits, np.array([[0, 1, 1], [1, 0, 0]], dtype=np.int8))

    def test_output_is_int8_preserves_shape(self) -> None:
        det = DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=2)
        counts = np.ones((7, 4), dtype=np.int64) * 3
        bits = det.discriminate(counts)
        assert bits.shape == (7, 4)
        assert bits.dtype == np.int8

    def test_negative_counts_raises(self) -> None:
        det = DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=2)
        with pytest.raises(ValueError, match="counts must be >= 0"):
            det.discriminate(np.array([0, 1, -1]))


class TestDetectorClassificationFidelity:
    def test_ideal_detector_high_contrast(self) -> None:
        """η=1, γ_d=0, threshold=5, λ_bright=15, λ_dark=0.3 → fidelity ~ 1."""
        det = DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=5)
        fid = det.classification_fidelity(lambda_bright=15.0, lambda_dark=0.3)
        assert fid["true_positive_rate"] > 0.99
        assert fid["true_negative_rate"] > 0.99
        assert fid["fidelity"] > 0.99
        assert fid["effective_bright_rate"] == 15.0
        assert fid["effective_dark_rate"] == 0.3

    def test_rate_transform_visible_in_effective_rates(self) -> None:
        det = DetectorConfig(efficiency=0.5, dark_count_rate=0.4, threshold=3)
        fid = det.classification_fidelity(lambda_bright=10.0, lambda_dark=0.0)
        assert fid["effective_bright_rate"] == 0.5 * 10.0 + 0.4
        assert fid["effective_dark_rate"] == 0.5 * 0.0 + 0.4

    def test_no_contrast_gives_chance_performance(self) -> None:
        """λ_bright = λ_dark → classifier cannot distinguish; fidelity < 0.6."""
        det = DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=3)
        fid = det.classification_fidelity(lambda_bright=2.0, lambda_dark=2.0)
        # TP + FN = 1 and TN + FP = 1, with TP = 1 − TN for equal rates,
        # so fidelity = (TP + TN) / 2 = (1 − TN + TN) / 2 = 0.5 exactly.
        np.testing.assert_allclose(fid["fidelity"], 0.5)

    def test_matches_empirical_classification(self) -> None:
        """Analytic fidelity must match Monte-Carlo classification rate at 1σ."""
        det = DetectorConfig(efficiency=0.6, dark_count_rate=0.5, threshold=4)
        lam_bright, lam_dark = 12.0, 0.2
        fid = det.classification_fidelity(lambda_bright=lam_bright, lambda_dark=lam_dark)
        rng = np.random.default_rng(2026)
        shots = 50_000
        bright_counts = rng.poisson(det.efficiency * lam_bright + det.dark_count_rate, size=shots)
        dark_counts = rng.poisson(det.efficiency * lam_dark + det.dark_count_rate, size=shots)
        empirical_tp = (bright_counts >= det.threshold).mean()
        empirical_tn = (dark_counts < det.threshold).mean()
        # 1σ empirical error ≈ sqrt(p(1−p)/N) ≈ 2e-3 at 50k; allow 5e-3.
        np.testing.assert_allclose(empirical_tp, fid["true_positive_rate"], atol=5e-3)
        np.testing.assert_allclose(empirical_tn, fid["true_negative_rate"], atol=5e-3)

    def test_negative_rate_raises(self) -> None:
        det = DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1)
        with pytest.raises(ValueError, match="rates must be >= 0"):
            det.classification_fidelity(lambda_bright=1.0, lambda_dark=-0.1)

    def test_dark_greater_than_bright_raises(self) -> None:
        det = DetectorConfig(efficiency=1.0, dark_count_rate=0.0, threshold=1)
        with pytest.raises(ValueError, match="lambda_dark must be"):
            det.classification_fidelity(lambda_bright=0.5, lambda_dark=5.0)


# ----------------------------------------------------------------------------
# Detector + PoissonChannel composition (end-to-end pipeline)
# ----------------------------------------------------------------------------


class TestDetectorPoissonComposition:
    def test_end_to_end_pipeline(self) -> None:
        """apply → Poisson → discriminate should reach the expected fidelity."""
        det = DetectorConfig(efficiency=0.5, dark_count_rate=0.3, threshold=3)
        lam_bright, lam_dark = 12.0, 0.0

        bright_rate = det.apply(np.array([lam_bright]))
        dark_rate = det.apply(np.array([lam_dark]))

        bright_result = sample_outcome(
            channel=PoissonChannel(),
            inputs=bright_rate,
            shots=10_000,
            seed=1,
        )
        dark_result = sample_outcome(
            channel=PoissonChannel(),
            inputs=dark_rate,
            shots=10_000,
            seed=2,
        )

        bright_bits = det.discriminate(bright_result.sampled_outcome["poisson"])
        dark_bits = det.discriminate(dark_result.sampled_outcome["poisson"])

        empirical_fid = 0.5 * (bright_bits.mean() + (1 - dark_bits).mean())
        analytic = det.classification_fidelity(lambda_bright=lam_bright, lambda_dark=lam_dark)
        # 10k shots on each of two rates — 1σ ~ 5e-3.
        np.testing.assert_allclose(empirical_fid, analytic["fidelity"], atol=1e-2)


# ----------------------------------------------------------------------------
# sample_outcome orchestrator
# ----------------------------------------------------------------------------


class TestSampleOutcome:
    def test_standalone_probability_array(self) -> None:
        probs = np.array([0.25, 0.5, 0.75])
        result = sample_outcome(
            channel=BernoulliChannel(),
            inputs=probs,
            shots=100,
            seed=0,
        )
        assert isinstance(result, MeasurementResult)
        assert result.shots == 100
        assert result.rng_seed == 0
        assert result.trajectory_hash is None
        np.testing.assert_array_equal(result.ideal_outcome["probability"], probs)
        assert result.sampled_outcome["bernoulli"].shape == (100, 3)

    def test_seed_is_reproducible(self) -> None:
        probs = np.array([0.3, 0.7])
        first = sample_outcome(channel=BernoulliChannel(), inputs=probs, shots=32, seed=99)
        second = sample_outcome(channel=BernoulliChannel(), inputs=probs, shots=32, seed=99)
        np.testing.assert_array_equal(
            first.sampled_outcome["bernoulli"],
            second.sampled_outcome["bernoulli"],
        )

    def test_upstream_metadata_inherited(self) -> None:
        times = np.linspace(0.0, 1.0e-6, 3)
        traj = TrajectoryResult(
            metadata=_trajectory_metadata(),
            times=times,
            expectations={"sigma_z_0": np.array([-1.0, 0.0, 1.0])},
        )
        probs = (1.0 + traj.expectations["sigma_z_0"]) / 2.0
        result = sample_outcome(
            channel=BernoulliChannel(),
            inputs=probs,
            shots=50,
            seed=1,
            upstream=traj,
            provenance_tags=("unit-test",),
        )
        assert result.trajectory_hash == traj.metadata.request_hash
        assert result.metadata.backend_name == "qutip-mesolve"
        assert result.metadata.storage_mode is StorageMode.OMITTED
        assert result.metadata.provenance_tags == (
            "demo",
            "carrier_rabi",
            "measurement",
            "unit-test",
        )

    def test_freestanding_metadata_is_tagged_measurement(self) -> None:
        result = sample_outcome(
            channel=BernoulliChannel(),
            inputs=np.array([0.5]),
            shots=1,
            seed=0,
        )
        assert "measurement" in result.metadata.provenance_tags
        assert result.metadata.backend_name == "iontrap-dynamics.measurement"

    def test_channel_label_routed_into_sampled_outcome(self) -> None:
        custom = BernoulliChannel(label="readout_ion_0")
        result = sample_outcome(
            channel=custom,
            inputs=np.array([0.5]),
            shots=1,
            seed=0,
        )
        assert "readout_ion_0" in result.sampled_outcome
        assert "bernoulli" not in result.sampled_outcome

    def test_binomial_channel_dispatch(self) -> None:
        probs = np.array([0.25, 0.5, 0.75])
        result = sample_outcome(
            channel=BinomialChannel(),
            inputs=probs,
            shots=200,
            seed=0,
        )
        counts = result.sampled_outcome["binomial"]
        assert counts.shape == (3,)
        assert counts.dtype == np.int64
        assert np.all(counts >= 0)
        assert np.all(counts <= 200)
        np.testing.assert_array_equal(result.ideal_outcome["probability"], probs)

    def test_binomial_seed_reproducible(self) -> None:
        probs = np.array([0.3, 0.7])
        first = sample_outcome(channel=BinomialChannel(), inputs=probs, shots=500, seed=99)
        second = sample_outcome(channel=BinomialChannel(), inputs=probs, shots=500, seed=99)
        np.testing.assert_array_equal(
            first.sampled_outcome["binomial"],
            second.sampled_outcome["binomial"],
        )

    def test_binomial_label_routed_into_sampled_outcome(self) -> None:
        custom = BinomialChannel(label="population_ion_0")
        result = sample_outcome(
            channel=custom,
            inputs=np.array([0.5]),
            shots=10,
            seed=0,
        )
        assert "population_ion_0" in result.sampled_outcome
        assert "binomial" not in result.sampled_outcome

    def test_poisson_channel_dispatch(self) -> None:
        rates = np.array([0.5, 2.0, 10.0])
        result = sample_outcome(
            channel=PoissonChannel(),
            inputs=rates,
            shots=100,
            seed=0,
        )
        counts = result.sampled_outcome["poisson"]
        assert counts.shape == (100, 3)
        assert counts.dtype == np.int64
        np.testing.assert_array_equal(result.ideal_outcome["rate"], rates)
        assert "probability" not in result.ideal_outcome

    def test_poisson_ideal_label_routing(self) -> None:
        """ideal_outcome key follows channel.ideal_label, not a hard-coded string."""
        result = sample_outcome(
            channel=PoissonChannel(),
            inputs=np.array([1.0]),
            shots=1,
            seed=0,
        )
        assert "rate" in result.ideal_outcome
        assert "probability" not in result.ideal_outcome

    def test_bernoulli_ideal_label_routing(self) -> None:
        result = sample_outcome(
            channel=BernoulliChannel(),
            inputs=np.array([0.5]),
            shots=1,
            seed=0,
        )
        assert "probability" in result.ideal_outcome
        assert "rate" not in result.ideal_outcome

    def test_poisson_seed_reproducible(self) -> None:
        rates = np.array([0.5, 5.0, 15.0])
        first = sample_outcome(channel=PoissonChannel(), inputs=rates, shots=200, seed=99)
        second = sample_outcome(channel=PoissonChannel(), inputs=rates, shots=200, seed=99)
        np.testing.assert_array_equal(
            first.sampled_outcome["poisson"],
            second.sampled_outcome["poisson"],
        )

    def test_poisson_label_routed_into_sampled_outcome(self) -> None:
        custom = PoissonChannel(label="photon_window_pmt0")
        result = sample_outcome(
            channel=custom,
            inputs=np.array([5.0]),
            shots=10,
            seed=0,
        )
        assert "photon_window_pmt0" in result.sampled_outcome
        assert "poisson" not in result.sampled_outcome
