# SPDX-License-Identifier: MIT
"""Unit tests for the canonical result schema.

Covers ``TrajectoryResult`` construction, storage-mode consistency, frozenness,
keyword-only enforcement, and metadata surface — the contract Phase 1 builders
will consume. Tests here are backend-agnostic and do not require QuTiP.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iontrap_dynamics import (
    CONVENTION_VERSION,
    ConventionError,
    IonTrapError,
    Result,
    ResultMetadata,
    ResultWarning,
    StorageMode,
    TrajectoryResult,
    WarningSeverity,
)

# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _metadata(storage_mode: StorageMode = StorageMode.OMITTED) -> ResultMetadata:
    """Minimal metadata for construction-surface tests."""
    return ResultMetadata(
        convention_version=CONVENTION_VERSION,
        request_hash="0" * 64,
        backend_name="test-backend",
        backend_version="0.0.0",
        storage_mode=storage_mode,
    )


def _times(n: int = 3) -> np.ndarray:
    return np.linspace(0.0, 1.0e-6, n)


# ----------------------------------------------------------------------------
# Construction happy paths
# ----------------------------------------------------------------------------


class TestConstructionHappyPaths:
    def test_omitted_mode_requires_nothing(self) -> None:
        result = TrajectoryResult(
            metadata=_metadata(StorageMode.OMITTED),
            times=_times(),
        )
        assert result.states is None
        assert result.states_loader is None

    def test_eager_mode_with_states(self) -> None:
        fake_states = ("psi_0", "psi_1", "psi_2")  # backend-opaque stand-ins
        result = TrajectoryResult(
            metadata=_metadata(StorageMode.EAGER),
            times=_times(),
            states=fake_states,
        )
        assert result.states == fake_states
        assert result.states_loader is None

    def test_lazy_mode_with_loader(self) -> None:
        loader = lambda i: f"psi_{i}"  # noqa: E731
        result = TrajectoryResult(
            metadata=_metadata(StorageMode.LAZY),
            times=_times(),
            states_loader=loader,
        )
        assert result.states is None
        assert result.states_loader is loader

    def test_expectations_roundtrip(self) -> None:
        expectations = {"sigma_z": np.array([1.0, 0.5, -0.5])}
        result = TrajectoryResult(
            metadata=_metadata(),
            times=_times(),
            expectations=expectations,
        )
        assert "sigma_z" in result.expectations
        np.testing.assert_array_equal(result.expectations["sigma_z"], expectations["sigma_z"])


# ----------------------------------------------------------------------------
# Storage-mode consistency (CONVENTIONS.md §0.E)
# ----------------------------------------------------------------------------


class TestStorageModeEnforcement:
    def test_eager_without_states_raises(self) -> None:
        with pytest.raises(ConventionError, match="EAGER requires"):
            TrajectoryResult(
                metadata=_metadata(StorageMode.EAGER),
                times=_times(),
                states=None,
            )

    def test_eager_with_loader_raises(self) -> None:
        with pytest.raises(ConventionError, match="EAGER forbids"):
            TrajectoryResult(
                metadata=_metadata(StorageMode.EAGER),
                times=_times(),
                states=("a", "b", "c"),
                states_loader=lambda i: f"psi_{i}",
            )

    def test_lazy_with_states_raises(self) -> None:
        with pytest.raises(ConventionError, match="LAZY forbids"):
            TrajectoryResult(
                metadata=_metadata(StorageMode.LAZY),
                times=_times(),
                states=("a", "b", "c"),
                states_loader=lambda i: f"psi_{i}",
            )

    def test_lazy_without_loader_raises(self) -> None:
        with pytest.raises(ConventionError, match="LAZY requires"):
            TrajectoryResult(
                metadata=_metadata(StorageMode.LAZY),
                times=_times(),
            )

    def test_omitted_with_states_raises(self) -> None:
        with pytest.raises(ConventionError, match="OMITTED requires"):
            TrajectoryResult(
                metadata=_metadata(StorageMode.OMITTED),
                times=_times(),
                states=("a", "b", "c"),
            )

    def test_omitted_with_loader_raises(self) -> None:
        with pytest.raises(ConventionError, match="OMITTED requires"):
            TrajectoryResult(
                metadata=_metadata(StorageMode.OMITTED),
                times=_times(),
                states_loader=lambda i: f"psi_{i}",
            )

    def test_conventionerror_subclasses_iontraperror(self) -> None:
        """Downstream code should be able to catch `IonTrapError` as a blanket."""
        with pytest.raises(IonTrapError):
            TrajectoryResult(
                metadata=_metadata(StorageMode.EAGER),
                times=_times(),
            )


# ----------------------------------------------------------------------------
# Immutability + keyword-only invariants (Design Principle 6)
# ----------------------------------------------------------------------------


class TestImmutability:
    def test_attribute_assignment_raises(self) -> None:
        result = TrajectoryResult(metadata=_metadata(), times=_times())
        with pytest.raises(FrozenInstanceError):
            result.times = _times(5)  # type: ignore[misc]

    def test_metadata_attribute_assignment_raises(self) -> None:
        meta = _metadata()
        with pytest.raises(FrozenInstanceError):
            meta.backend_name = "other"  # type: ignore[misc]

    def test_positional_construction_forbidden(self) -> None:
        """kw_only=True: even one positional arg should fail."""
        with pytest.raises(TypeError):
            TrajectoryResult(_metadata())  # type: ignore[misc]


# ----------------------------------------------------------------------------
# Warning record surface
# ----------------------------------------------------------------------------


class TestWarnings:
    def test_default_warnings_empty(self) -> None:
        result = TrajectoryResult(metadata=_metadata(), times=_times())
        assert result.warnings == ()

    def test_warning_record_construction(self) -> None:
        warn = ResultWarning(
            severity=WarningSeverity.CONVERGENCE,
            category="fock-truncation",
            message="top Fock population 5e-5, between eps/10 and eps",
            diagnostics={"p_top": 5e-5, "eps": 1e-4},
        )
        result = TrajectoryResult(
            metadata=_metadata(),
            times=_times(),
            warnings=(warn,),
        )
        assert len(result.warnings) == 1
        assert result.warnings[0].severity is WarningSeverity.CONVERGENCE

    def test_warning_severity_enum_values(self) -> None:
        """Only Level 1 and Level 2 live on results; Level 3 raises."""
        assert WarningSeverity.CONVERGENCE.value == "convergence"
        assert WarningSeverity.QUALITY.value == "quality"
        assert {m.name for m in WarningSeverity} == {"CONVERGENCE", "QUALITY"}


# ----------------------------------------------------------------------------
# Abstract base and result-family (decision D5)
# ----------------------------------------------------------------------------


class TestResultFamily:
    def test_trajectory_result_is_a_result(self) -> None:
        """Phase 1 siblings (stochastic, measurement) extend `Result` too."""
        result = TrajectoryResult(metadata=_metadata(), times=_times())
        assert isinstance(result, Result)

    def test_metadata_convention_version_recorded(self) -> None:
        result = TrajectoryResult(metadata=_metadata(), times=_times())
        assert result.metadata.convention_version == CONVENTION_VERSION

    def test_storage_mode_enum_values(self) -> None:
        assert {m.name for m in StorageMode} == {"EAGER", "LAZY", "OMITTED"}
