# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.modes`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iontrap_dynamics.exceptions import ConventionError, IonTrapError
from iontrap_dynamics.modes import ModeConfig


def _single_ion_axial_eigenvector() -> np.ndarray:
    """A single-ion crystal: eigenvector = (0, 0, 1), normalised."""
    return np.array([[0.0, 0.0, 1.0]])


def _two_ion_axial_com_eigenvector() -> np.ndarray:
    """A two-ion centre-of-mass axial mode: both ions move in phase with
    equal amplitudes 1/√2 along z."""
    return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]) / np.sqrt(2.0)


def _two_ion_axial_stretch_eigenvector() -> np.ndarray:
    """A two-ion stretch axial mode: opposite-phase motion at ±1/√2."""
    return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]) / np.sqrt(2.0)


# ----------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------


class TestConstruction:
    def test_single_ion_mode(self) -> None:
        mode = ModeConfig(
            label="axial",
            frequency_rad_s=2 * np.pi * 1.5e6,
            eigenvector_per_ion=_single_ion_axial_eigenvector(),
        )
        assert mode.label == "axial"
        assert mode.n_ions == 1
        assert mode.frequency_rad_s == pytest.approx(2 * np.pi * 1.5e6)

    def test_two_ion_com_mode(self) -> None:
        mode = ModeConfig(
            label="com",
            frequency_rad_s=2 * np.pi * 1.5e6,
            eigenvector_per_ion=_two_ion_axial_com_eigenvector(),
        )
        assert mode.n_ions == 2

    def test_two_ion_stretch_mode(self) -> None:
        mode = ModeConfig(
            label="stretch",
            frequency_rad_s=2 * np.pi * 2.6e6,
            eigenvector_per_ion=_two_ion_axial_stretch_eigenvector(),
        )
        assert mode.n_ions == 2

    def test_construction_from_list_coerces(self) -> None:
        mode = ModeConfig(
            label="axial",
            frequency_rad_s=1.0,
            eigenvector_per_ion=[[0.0, 0.0, 1.0]],
        )
        assert isinstance(mode.eigenvector_per_ion, np.ndarray)
        assert mode.eigenvector_per_ion.shape == (1, 3)


# ----------------------------------------------------------------------------
# Validation — including CONVENTIONS.md §11 normalisation rule
# ----------------------------------------------------------------------------


class TestValidation:
    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ConventionError, match="label"):
            ModeConfig(
                label="",
                frequency_rad_s=1.0,
                eigenvector_per_ion=_single_ion_axial_eigenvector(),
            )

    def test_zero_frequency_rejected(self) -> None:
        with pytest.raises(ConventionError, match="frequency_rad_s"):
            ModeConfig(
                label="axial",
                frequency_rad_s=0.0,
                eigenvector_per_ion=_single_ion_axial_eigenvector(),
            )

    def test_negative_frequency_rejected(self) -> None:
        with pytest.raises(ConventionError, match="frequency_rad_s"):
            ModeConfig(
                label="axial",
                frequency_rad_s=-1.0,
                eigenvector_per_ion=_single_ion_axial_eigenvector(),
            )

    def test_wrong_shape_rejected(self) -> None:
        """Eigenvector must have shape (N_ions, 3), not flat (3,)."""
        with pytest.raises(ConventionError, match="shape"):
            ModeConfig(
                label="axial",
                frequency_rad_s=1.0,
                eigenvector_per_ion=np.array([0.0, 0.0, 1.0]),  # 1-D, not 2-D
            )

    def test_empty_eigenvector_rejected(self) -> None:
        with pytest.raises(ConventionError, match="at least one ion"):
            ModeConfig(
                label="axial",
                frequency_rad_s=1.0,
                eigenvector_per_ion=np.zeros((0, 3)),
            )

    def test_non_unit_normalisation_rejected(self) -> None:
        """CONVENTIONS.md §11: Σᵢ ||b_i,m||² = 1. A factor-of-two-off
        normalisation must raise at construction time."""
        with pytest.raises(ConventionError, match="§11"):
            ModeConfig(
                label="axial",
                frequency_rad_s=1.0,
                eigenvector_per_ion=np.array([[0.0, 0.0, 2.0]]),  # norm² = 4
            )

    def test_two_ion_unnormalised_rejected(self) -> None:
        """Two ions with raw (1,1) amplitudes: norm² = 2, not 1."""
        with pytest.raises(ConventionError, match="§11"):
            ModeConfig(
                label="com",
                frequency_rad_s=1.0,
                eigenvector_per_ion=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),  # unnormalised
            )

    def test_validation_errors_subclass_iontraperror(self) -> None:
        with pytest.raises(IonTrapError):
            ModeConfig(
                label="",
                frequency_rad_s=1.0,
                eigenvector_per_ion=_single_ion_axial_eigenvector(),
            )


# ----------------------------------------------------------------------------
# Derived methods
# ----------------------------------------------------------------------------


class TestEigenvectorAccessors:
    def test_eigenvector_at_ion_returns_3_vector(self) -> None:
        mode = ModeConfig(
            label="com",
            frequency_rad_s=1.0,
            eigenvector_per_ion=_two_ion_axial_com_eigenvector(),
        )
        b0 = mode.eigenvector_at_ion(0)
        assert b0.shape == (3,)
        np.testing.assert_allclose(b0, [0.0, 0.0, 1.0 / np.sqrt(2.0)])

    def test_eigenvector_at_ion_returns_copy_not_view(self) -> None:
        """Mutating the returned array must not affect the stored mode."""
        mode = ModeConfig(
            label="com",
            frequency_rad_s=1.0,
            eigenvector_per_ion=_two_ion_axial_com_eigenvector(),
        )
        b0 = mode.eigenvector_at_ion(0)
        b0[2] = 999.0  # tamper
        # The stored mode must be unaffected.
        np.testing.assert_allclose(mode.eigenvector_at_ion(0), [0.0, 0.0, 1.0 / np.sqrt(2.0)])

    def test_eigenvector_at_ion_out_of_range_raises(self) -> None:
        mode = ModeConfig(
            label="com",
            frequency_rad_s=1.0,
            eigenvector_per_ion=_two_ion_axial_com_eigenvector(),
        )
        with pytest.raises(IndexError):
            mode.eigenvector_at_ion(2)

    def test_negative_ion_index_raises(self) -> None:
        mode = ModeConfig(
            label="com",
            frequency_rad_s=1.0,
            eigenvector_per_ion=_two_ion_axial_com_eigenvector(),
        )
        with pytest.raises(IndexError):
            mode.eigenvector_at_ion(-1)


# ----------------------------------------------------------------------------
# Immutability
# ----------------------------------------------------------------------------


class TestImmutability:
    def test_attribute_assignment_raises(self) -> None:
        mode = ModeConfig(
            label="axial",
            frequency_rad_s=1.0,
            eigenvector_per_ion=_single_ion_axial_eigenvector(),
        )
        with pytest.raises(FrozenInstanceError):
            mode.frequency_rad_s = 2.0  # type: ignore[misc]

    def test_positional_construction_forbidden(self) -> None:
        with pytest.raises(TypeError):
            ModeConfig("axial", 1.0, _single_ion_axial_eigenvector())  # type: ignore[misc]
