# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.system` — the composition layer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iontrap_dynamics.conventions import CONVENTION_VERSION
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.exceptions import ConventionError, IonTrapError
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.species import (
    IonSpecies,
    Transition,
    TransitionType,
    ca40_plus,
    mg25_plus,
)
from iontrap_dynamics.system import IonSystem


# ----------------------------------------------------------------------------
# Fixture helpers
# ----------------------------------------------------------------------------


def _single_ion_axial_mode(*, label: str = "axial", freq: float = 2 * np.pi * 1.5e6) -> ModeConfig:
    return ModeConfig(
        label=label,
        frequency_rad_s=freq,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )


def _two_ion_com_mode() -> ModeConfig:
    return ModeConfig(
        label="com",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]) / np.sqrt(2.0),
    )


def _two_ion_stretch_mode() -> ModeConfig:
    return ModeConfig(
        label="stretch",
        frequency_rad_s=2 * np.pi * 2.6e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]) / np.sqrt(2.0),
    )


def _simple_drive() -> DriveConfig:
    return DriveConfig(
        k_vector_m_inv=[2e7, 0.0, 0.0],
        carrier_rabi_frequency_rad_s=2 * np.pi * 0.1e6,
    )


# ----------------------------------------------------------------------------
# Construction happy paths
# ----------------------------------------------------------------------------


class TestConstruction:
    def test_single_ion_no_drives_no_modes(self) -> None:
        system = IonSystem(species_per_ion=(mg25_plus(),))
        assert system.n_ions == 1
        assert system.n_drives == 0
        assert system.n_modes == 0
        assert system.convention_version == CONVENTION_VERSION

    def test_single_ion_with_mode_and_drive(self) -> None:
        system = IonSystem(
            species_per_ion=(mg25_plus(),),
            drives=(_simple_drive(),),
            modes=(_single_ion_axial_mode(),),
        )
        assert system.n_ions == 1
        assert system.n_modes == 1
        assert system.n_drives == 1

    def test_two_ion_two_modes(self) -> None:
        system = IonSystem(
            species_per_ion=(mg25_plus(), mg25_plus()),
            modes=(_two_ion_com_mode(), _two_ion_stretch_mode()),
        )
        assert system.n_ions == 2
        assert system.n_modes == 2

    def test_mixed_species(self) -> None:
        system = IonSystem(
            species_per_ion=(mg25_plus(), ca40_plus()),
            modes=(_two_ion_com_mode(),),
        )
        assert not system.is_homogeneous
        assert system.species(0).element == "Mg"
        assert system.species(1).element == "Ca"


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------


class TestValidation:
    def test_empty_crystal_rejected(self) -> None:
        with pytest.raises(ConventionError, match="at least one ion"):
            IonSystem(species_per_ion=())

    def test_mode_with_wrong_n_ions_rejected(self) -> None:
        """Single-ion crystal + two-ion mode → shape mismatch at construction."""
        with pytest.raises(ConventionError, match="n_ions"):
            IonSystem(
                species_per_ion=(mg25_plus(),),
                modes=(_two_ion_com_mode(),),
            )

    def test_two_ion_crystal_with_single_ion_mode_rejected(self) -> None:
        """Reverse mismatch is also caught."""
        with pytest.raises(ConventionError, match="n_ions"):
            IonSystem(
                species_per_ion=(mg25_plus(), mg25_plus()),
                modes=(_single_ion_axial_mode(),),
            )

    def test_duplicate_mode_labels_rejected(self) -> None:
        axial_1 = _single_ion_axial_mode(label="axial", freq=2 * np.pi * 1.5e6)
        axial_2 = _single_ion_axial_mode(label="axial", freq=2 * np.pi * 2.6e6)
        with pytest.raises(ConventionError, match="duplicate mode labels"):
            IonSystem(
                species_per_ion=(mg25_plus(),),
                modes=(axial_1, axial_2),
            )

    def test_drive_with_unknown_transition_label_rejected(self) -> None:
        drive = DriveConfig(
            k_vector_m_inv=[2e7, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
            transition_label="fictional_transition",
        )
        with pytest.raises(ConventionError, match="transition_label"):
            IonSystem(species_per_ion=(mg25_plus(),), drives=(drive,))

    def test_drive_with_known_transition_label_accepted(self) -> None:
        drive = DriveConfig(
            k_vector_m_inv=[2e7, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
            transition_label="3s²S₁/₂ → 3p²P₁/₂",  # exists in Mg25+
        )
        system = IonSystem(species_per_ion=(mg25_plus(),), drives=(drive,))
        assert system.n_drives == 1

    def test_drive_label_satisfied_by_any_species(self) -> None:
        """In a mixed-species crystal, a drive's transition_label is valid
        if ANY species in the crystal carries that transition."""
        drive = DriveConfig(
            k_vector_m_inv=[2e7, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
            transition_label="4s²S₁/₂ → 3d²D₅/₂",  # exists in Ca40+ only
        )
        system = IonSystem(
            species_per_ion=(mg25_plus(), ca40_plus()),
            drives=(drive,),
            modes=(_two_ion_com_mode(),),
        )
        assert system.n_drives == 1

    def test_drive_with_no_transition_label_accepted(self) -> None:
        """None transition_label skips the cross-reference check."""
        system = IonSystem(
            species_per_ion=(mg25_plus(),),
            drives=(_simple_drive(),),  # transition_label defaults None
        )
        assert system.n_drives == 1

    def test_validation_errors_subclass_iontraperror(self) -> None:
        with pytest.raises(IonTrapError):
            IonSystem(species_per_ion=())


# ----------------------------------------------------------------------------
# Accessors
# ----------------------------------------------------------------------------


class TestAccessors:
    def test_species_by_ion_index(self) -> None:
        system = IonSystem(species_per_ion=(mg25_plus(), ca40_plus()))
        assert system.species(0).element == "Mg"
        assert system.species(1).element == "Ca"

    def test_species_out_of_range_raises(self) -> None:
        system = IonSystem(species_per_ion=(mg25_plus(),))
        with pytest.raises(IndexError):
            system.species(1)

    def test_negative_ion_index_raises(self) -> None:
        system = IonSystem(species_per_ion=(mg25_plus(),))
        with pytest.raises(IndexError):
            system.species(-1)

    def test_mode_by_label(self) -> None:
        system = IonSystem(
            species_per_ion=(mg25_plus(), mg25_plus()),
            modes=(_two_ion_com_mode(), _two_ion_stretch_mode()),
        )
        com = system.mode("com")
        assert com.frequency_rad_s == pytest.approx(2 * np.pi * 1.5e6)

    def test_mode_unknown_label_raises(self) -> None:
        system = IonSystem(
            species_per_ion=(mg25_plus(), mg25_plus()),
            modes=(_two_ion_com_mode(),),
        )
        with pytest.raises(ConventionError, match="unknown mode"):
            system.mode("stretch")

    def test_is_homogeneous_true_for_same_species(self) -> None:
        system = IonSystem(
            species_per_ion=(mg25_plus(), mg25_plus(), mg25_plus()),
            modes=(),
        )
        assert system.is_homogeneous is True

    def test_is_homogeneous_false_for_mixed_species(self) -> None:
        system = IonSystem(
            species_per_ion=(mg25_plus(), ca40_plus()),
            modes=(_two_ion_com_mode(),),
        )
        assert system.is_homogeneous is False

    def test_is_homogeneous_true_for_single_ion(self) -> None:
        system = IonSystem(species_per_ion=(mg25_plus(),))
        assert system.is_homogeneous is True


# ----------------------------------------------------------------------------
# IonSystem.homogeneous classmethod
# ----------------------------------------------------------------------------


class TestHomogeneousFactory:
    def test_homogeneous_three_ions(self) -> None:
        sp = mg25_plus()
        system = IonSystem.homogeneous(species=sp, n_ions=3)
        assert system.n_ions == 3
        assert system.is_homogeneous
        assert system.species(0) == sp
        assert system.species(2) == sp

    def test_homogeneous_with_modes_and_drives(self) -> None:
        # Two-ion homogeneous crystal with com + stretch modes
        system = IonSystem.homogeneous(
            species=mg25_plus(),
            n_ions=2,
            drives=(_simple_drive(),),
            modes=(_two_ion_com_mode(), _two_ion_stretch_mode()),
        )
        assert system.n_ions == 2
        assert system.n_modes == 2
        assert system.n_drives == 1

    def test_homogeneous_zero_ions_rejected(self) -> None:
        with pytest.raises(ConventionError, match="n_ions"):
            IonSystem.homogeneous(species=mg25_plus(), n_ions=0)

    def test_homogeneous_negative_ions_rejected(self) -> None:
        with pytest.raises(ConventionError, match="n_ions"):
            IonSystem.homogeneous(species=mg25_plus(), n_ions=-1)


# ----------------------------------------------------------------------------
# Immutability
# ----------------------------------------------------------------------------


class TestImmutability:
    def test_attribute_assignment_raises(self) -> None:
        system = IonSystem(species_per_ion=(mg25_plus(),))
        with pytest.raises(FrozenInstanceError):
            system.species_per_ion = ()  # type: ignore[misc]

    def test_positional_construction_forbidden(self) -> None:
        with pytest.raises(TypeError):
            IonSystem((mg25_plus(),))  # type: ignore[misc]


# ----------------------------------------------------------------------------
# Convention version snapshot
# ----------------------------------------------------------------------------


class TestConventionVersion:
    def test_default_matches_module_constant(self) -> None:
        system = IonSystem(species_per_ion=(mg25_plus(),))
        assert system.convention_version == CONVENTION_VERSION

    def test_explicit_override(self) -> None:
        """Explicit override is allowed — useful for pinning an older
        convention version to reproduce an archived result."""
        system = IonSystem(
            species_per_ion=(mg25_plus(),),
            convention_version="pinned-0.0.1-archive",
        )
        assert system.convention_version == "pinned-0.0.1-archive"
