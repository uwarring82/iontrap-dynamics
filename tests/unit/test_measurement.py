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
    IonTrapError,
    MeasurementResult,
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
# sample_outcome orchestrator
# ----------------------------------------------------------------------------


class TestSampleOutcome:
    def test_standalone_probability_array(self) -> None:
        probs = np.array([0.25, 0.5, 0.75])
        result = sample_outcome(
            channel=BernoulliChannel(),
            probabilities=probs,
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
        first = sample_outcome(channel=BernoulliChannel(), probabilities=probs, shots=32, seed=99)
        second = sample_outcome(channel=BernoulliChannel(), probabilities=probs, shots=32, seed=99)
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
            probabilities=probs,
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
            probabilities=np.array([0.5]),
            shots=1,
            seed=0,
        )
        assert "measurement" in result.metadata.provenance_tags
        assert result.metadata.backend_name == "iontrap-dynamics.measurement"

    def test_channel_label_routed_into_sampled_outcome(self) -> None:
        custom = BernoulliChannel(label="readout_ion_0")
        result = sample_outcome(
            channel=custom,
            probabilities=np.array([0.5]),
            shots=1,
            seed=0,
        )
        assert "readout_ion_0" in result.sampled_outcome
        assert "bernoulli" not in result.sampled_outcome

    def test_binomial_channel_dispatch(self) -> None:
        probs = np.array([0.25, 0.5, 0.75])
        result = sample_outcome(
            channel=BinomialChannel(),
            probabilities=probs,
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
        first = sample_outcome(channel=BinomialChannel(), probabilities=probs, shots=500, seed=99)
        second = sample_outcome(channel=BinomialChannel(), probabilities=probs, shots=500, seed=99)
        np.testing.assert_array_equal(
            first.sampled_outcome["binomial"],
            second.sampled_outcome["binomial"],
        )

    def test_binomial_label_routed_into_sampled_outcome(self) -> None:
        custom = BinomialChannel(label="population_ion_0")
        result = sample_outcome(
            channel=custom,
            probabilities=np.array([0.5]),
            shots=10,
            seed=0,
        )
        assert "population_ion_0" in result.sampled_outcome
        assert "binomial" not in result.sampled_outcome
