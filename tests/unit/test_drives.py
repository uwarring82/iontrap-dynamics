# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.drives`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.exceptions import ConventionError, IonTrapError

# ----------------------------------------------------------------------------
# Construction happy paths
# ----------------------------------------------------------------------------


class TestConstruction:
    def test_minimal_construction_from_ndarray(self) -> None:
        drive = DriveConfig(
            k_vector_m_inv=np.array([1.0, 0.0, 0.0]) * 2e7,
            carrier_rabi_frequency_rad_s=2 * np.pi * 1e6,
        )
        assert drive.carrier_rabi_frequency_rad_s == pytest.approx(2 * np.pi * 1e6)
        assert drive.detuning_rad_s == 0.0
        assert drive.phase_rad == 0.0
        assert drive.polarisation is None
        assert drive.transition_label is None

    def test_construction_from_list_coerces(self) -> None:
        """k_vector input as a list is coerced to a 3-element ndarray."""
        drive = DriveConfig(
            k_vector_m_inv=[1.0, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
        )
        assert isinstance(drive.k_vector_m_inv, np.ndarray)
        assert drive.k_vector_m_inv.shape == (3,)

    def test_construction_with_all_optionals(self) -> None:
        drive = DriveConfig(
            k_vector_m_inv=[2e7, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=2 * np.pi * 0.5e6,
            detuning_rad_s=-2 * np.pi * 1e3,  # red-detuned 1 kHz
            phase_rad=np.pi / 4,
            polarisation=[0.0, 1.0, 0.0],
            transition_label="S_P12",
        )
        assert drive.detuning_rad_s == pytest.approx(-2 * np.pi * 1e3)
        assert drive.phase_rad == pytest.approx(np.pi / 4)
        assert drive.transition_label == "S_P12"
        np.testing.assert_allclose(drive.polarisation, [0.0, 1.0, 0.0])


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


class TestValidation:
    def test_2_vector_k_rejected(self) -> None:
        with pytest.raises(ConventionError, match="k_vector_m_inv"):
            DriveConfig(
                k_vector_m_inv=[1.0, 0.0],  # too short
                carrier_rabi_frequency_rad_s=1.0,
            )

    def test_4_vector_k_rejected(self) -> None:
        with pytest.raises(ConventionError, match="k_vector_m_inv"):
            DriveConfig(
                k_vector_m_inv=[1.0, 0.0, 0.0, 0.0],  # too long
                carrier_rabi_frequency_rad_s=1.0,
            )

    def test_2_vector_polarisation_rejected(self) -> None:
        with pytest.raises(ConventionError, match="polarisation"):
            DriveConfig(
                k_vector_m_inv=[1.0, 0.0, 0.0],
                carrier_rabi_frequency_rad_s=1.0,
                polarisation=[0.0, 1.0],  # too short
            )

    def test_zero_rabi_rejected(self) -> None:
        with pytest.raises(ConventionError, match="carrier_rabi_frequency_rad_s"):
            DriveConfig(
                k_vector_m_inv=[1.0, 0.0, 0.0],
                carrier_rabi_frequency_rad_s=0.0,
            )

    def test_negative_rabi_rejected(self) -> None:
        """Overall sign of the coupling lives in phase_rad, not in Ω.
        Negative Rabi would invite two conflicting sign carriers."""
        with pytest.raises(ConventionError, match="carrier_rabi_frequency_rad_s"):
            DriveConfig(
                k_vector_m_inv=[1.0, 0.0, 0.0],
                carrier_rabi_frequency_rad_s=-1.0,
            )

    def test_validation_errors_subclass_iontraperror(self) -> None:
        with pytest.raises(IonTrapError):
            DriveConfig(
                k_vector_m_inv=[1.0, 0.0, 0.0],
                carrier_rabi_frequency_rad_s=-1.0,
            )


# ----------------------------------------------------------------------------
# Detuning sign conventions (CONVENTIONS.md §4)
# ----------------------------------------------------------------------------


class TestDetuningSign:
    def test_positive_detuning_is_blue(self) -> None:
        """CONVENTIONS.md §4: δ = ω_laser − ω_atom; positive = blue."""
        drive = DriveConfig(
            k_vector_m_inv=[1.0, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
            detuning_rad_s=+1e6,
        )
        assert drive.detuning_rad_s > 0

    def test_negative_detuning_is_red(self) -> None:
        drive = DriveConfig(
            k_vector_m_inv=[1.0, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
            detuning_rad_s=-1e6,
        )
        assert drive.detuning_rad_s < 0

    def test_zero_detuning_is_on_resonance(self) -> None:
        drive = DriveConfig(
            k_vector_m_inv=[1.0, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
            detuning_rad_s=0.0,
        )
        assert drive.detuning_rad_s == 0.0


# ----------------------------------------------------------------------------
# Derived properties
# ----------------------------------------------------------------------------


class TestDerivedProperties:
    def test_wavenumber_scalar(self) -> None:
        """|k| matches sqrt(kx² + ky² + kz²)."""
        drive = DriveConfig(
            k_vector_m_inv=[3.0, 4.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
        )
        assert drive.wavenumber_m_inv == pytest.approx(5.0)


# ----------------------------------------------------------------------------
# Immutability + keyword-only
# ----------------------------------------------------------------------------


class TestImmutability:
    def test_attribute_assignment_raises(self) -> None:
        drive = DriveConfig(
            k_vector_m_inv=[1.0, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
        )
        with pytest.raises(FrozenInstanceError):
            drive.phase_rad = 0.5  # type: ignore[misc]

    def test_positional_construction_forbidden(self) -> None:
        with pytest.raises(TypeError):
            DriveConfig([1.0, 0.0, 0.0], 1.0)  # type: ignore[misc]
