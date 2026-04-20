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
