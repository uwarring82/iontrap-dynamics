# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.species`.

Covers :class:`Transition` and :class:`IonSpecies` construction, validation,
immutability, keyword-only-construction, and the factory-function values
for the three shipped species.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from iontrap_dynamics.exceptions import ConventionError, IonTrapError
from iontrap_dynamics.species import (
    ATOMIC_MASS_UNIT_KG,
    IonSpecies,
    Transition,
    TransitionType,
    ca40_plus,
    ca43_plus,
    mg25_plus,
)

# ----------------------------------------------------------------------------
# Transition
# ----------------------------------------------------------------------------


class TestTransitionConstruction:
    def test_minimal_construction(self) -> None:
        t = Transition(
            label="test",
            wavelength_m=500e-9,
            transition_type=TransitionType.E1,
        )
        assert t.label == "test"
        assert t.wavelength_m == 500e-9
        assert t.transition_type is TransitionType.E1
        assert t.linewidth_rad_s is None

    def test_with_linewidth(self) -> None:
        t = Transition(
            label="clock",
            wavelength_m=729e-9,
            transition_type=TransitionType.E2,
            linewidth_rad_s=1.0,
        )
        assert t.linewidth_rad_s == 1.0


class TestTransitionValidation:
    def test_zero_wavelength_rejected(self) -> None:
        with pytest.raises(ConventionError, match="wavelength_m"):
            Transition(
                label="bad",
                wavelength_m=0.0,
                transition_type=TransitionType.E1,
            )

    def test_negative_wavelength_rejected(self) -> None:
        with pytest.raises(ConventionError, match="wavelength_m"):
            Transition(
                label="bad",
                wavelength_m=-1e-9,
                transition_type=TransitionType.E1,
            )

    def test_negative_linewidth_rejected(self) -> None:
        with pytest.raises(ConventionError, match="linewidth_rad_s"):
            Transition(
                label="bad",
                wavelength_m=500e-9,
                transition_type=TransitionType.E1,
                linewidth_rad_s=-1.0,
            )

    def test_zero_linewidth_accepted(self) -> None:
        """A zero natural linewidth is physically meaningful — an idealised
        closed transition with no spontaneous emission. Not to be rejected."""
        t = Transition(
            label="ideal",
            wavelength_m=500e-9,
            transition_type=TransitionType.E1,
            linewidth_rad_s=0.0,
        )
        assert t.linewidth_rad_s == 0.0


class TestTransitionImmutability:
    def test_attribute_assignment_raises(self) -> None:
        t = Transition(
            label="x",
            wavelength_m=500e-9,
            transition_type=TransitionType.E1,
        )
        with pytest.raises(FrozenInstanceError):
            t.wavelength_m = 400e-9  # type: ignore[misc]

    def test_positional_construction_forbidden(self) -> None:
        with pytest.raises(TypeError):
            Transition("x", 500e-9, TransitionType.E1)  # type: ignore[misc]


# ----------------------------------------------------------------------------
# IonSpecies
# ----------------------------------------------------------------------------


class TestIonSpeciesConstruction:
    def test_minimal_construction(self) -> None:
        s = IonSpecies(
            element="Mg",
            mass_number=25,
            mass_kg=25 * ATOMIC_MASS_UNIT_KG,
        )
        assert s.element == "Mg"
        assert s.mass_number == 25
        assert s.charge == +1  # default
        assert s.transitions == ()

    def test_construction_with_transitions(self) -> None:
        t = Transition(label="a", wavelength_m=1e-6, transition_type=TransitionType.E1)
        s = IonSpecies(
            element="Mg",
            mass_number=25,
            mass_kg=25 * ATOMIC_MASS_UNIT_KG,
            transitions=(t,),
        )
        assert s.transitions == (t,)

    def test_non_default_charge(self) -> None:
        s = IonSpecies(
            element="Ba",
            mass_number=138,
            mass_kg=138 * ATOMIC_MASS_UNIT_KG,
            charge=+2,
        )
        assert s.charge == +2


class TestIonSpeciesValidation:
    def test_empty_element_rejected(self) -> None:
        with pytest.raises(ConventionError, match="element"):
            IonSpecies(element="", mass_number=1, mass_kg=1e-27)

    def test_zero_mass_number_rejected(self) -> None:
        with pytest.raises(ConventionError, match="mass_number"):
            IonSpecies(element="Mg", mass_number=0, mass_kg=1e-27)

    def test_negative_mass_number_rejected(self) -> None:
        with pytest.raises(ConventionError, match="mass_number"):
            IonSpecies(element="Mg", mass_number=-1, mass_kg=1e-27)

    def test_zero_mass_kg_rejected(self) -> None:
        with pytest.raises(ConventionError, match="mass_kg"):
            IonSpecies(element="Mg", mass_number=25, mass_kg=0.0)

    def test_negative_mass_kg_rejected(self) -> None:
        with pytest.raises(ConventionError, match="mass_kg"):
            IonSpecies(element="Mg", mass_number=25, mass_kg=-1e-27)

    def test_duplicate_transition_labels_rejected(self) -> None:
        t1 = Transition(label="same", wavelength_m=1e-6, transition_type=TransitionType.E1)
        t2 = Transition(label="same", wavelength_m=2e-6, transition_type=TransitionType.E2)
        with pytest.raises(ConventionError, match="duplicate"):
            IonSpecies(
                element="Mg",
                mass_number=25,
                mass_kg=25 * ATOMIC_MASS_UNIT_KG,
                transitions=(t1, t2),
            )

    def test_validation_errors_subclass_iontraperror(self) -> None:
        """Blanket `except IonTrapError` must catch every species
        validation failure (see CONVENTIONS.md §15 on the exception
        family)."""
        with pytest.raises(IonTrapError):
            IonSpecies(element="Mg", mass_number=-1, mass_kg=1e-27)


class TestIonSpeciesImmutability:
    def test_attribute_assignment_raises(self) -> None:
        s = ca40_plus()
        with pytest.raises(FrozenInstanceError):
            s.charge = 2  # type: ignore[misc]

    def test_transitions_is_tuple_not_mutable(self) -> None:
        """Workflow: storing transitions as a tuple prevents a mutation via
        `species.transitions.append(...)` from silently succeeding."""
        s = ca40_plus()
        assert isinstance(s.transitions, tuple)
        with pytest.raises(AttributeError):
            s.transitions.append("something")  # type: ignore[attr-defined]


class TestTransitionLookup:
    def test_lookup_by_label(self) -> None:
        s = ca40_plus()
        clock = s.transition("4s²S₁/₂ → 3d²D₅/₂")
        assert clock.wavelength_m == pytest.approx(729.147e-9)
        assert clock.transition_type is TransitionType.E2

    def test_lookup_unknown_label_raises(self) -> None:
        s = ca40_plus()
        with pytest.raises(ConventionError, match="unknown transition"):
            s.transition("fictional_transition")


class TestSpeciesName:
    def test_singly_charged_cation(self) -> None:
        assert ca40_plus().name == "40Ca+"
        assert mg25_plus().name == "25Mg+"

    def test_doubly_charged_cation(self) -> None:
        s = IonSpecies(
            element="Ba",
            mass_number=138,
            mass_kg=138 * ATOMIC_MASS_UNIT_KG,
            charge=+2,
        )
        assert s.name == "138Ba2+"

    def test_neutral_atom(self) -> None:
        s = IonSpecies(
            element="Rb",
            mass_number=87,
            mass_kg=87 * ATOMIC_MASS_UNIT_KG,
            charge=0,
        )
        assert s.name == "87Rb"

    def test_singly_charged_anion(self) -> None:
        s = IonSpecies(
            element="H",
            mass_number=1,
            mass_kg=1 * ATOMIC_MASS_UNIT_KG,
            charge=-1,
        )
        assert s.name == "1H-"


# ----------------------------------------------------------------------------
# Factory functions
# ----------------------------------------------------------------------------


class TestFactoryFunctions:
    def test_mg25_plus_basic(self) -> None:
        s = mg25_plus()
        assert s.element == "Mg"
        assert s.mass_number == 25
        assert s.charge == +1
        # 25Mg atomic mass ≈ 24.986 u
        assert s.mass_kg == pytest.approx(24.98583696 * ATOMIC_MASS_UNIT_KG)
        # Two principal transitions
        assert len(s.transitions) == 2

    def test_mg25_plus_has_d1_transition(self) -> None:
        s = mg25_plus()
        d1 = s.transition("3s²S₁/₂ → 3p²P₁/₂")
        # 280 nm ± a few nm
        assert 270e-9 < d1.wavelength_m < 290e-9
        assert d1.transition_type is TransitionType.E1

    def test_ca40_plus_has_quadrupole_clock(self) -> None:
        s = ca40_plus()
        clock = s.transition("4s²S₁/₂ → 3d²D₅/₂")
        # 729 nm quadrupole clock transition
        assert clock.wavelength_m == pytest.approx(729.147e-9)
        assert clock.transition_type is TransitionType.E2

    def test_ca43_plus_has_different_isotope(self) -> None:
        ca40 = ca40_plus()
        ca43 = ca43_plus()
        assert ca43.mass_number == 43
        assert ca43.mass_kg > ca40.mass_kg  # 43 is heavier than 40
        assert ca43.element == "Ca"

    def test_factory_returns_fresh_instance(self) -> None:
        """Each call constructs a new IonSpecies so clients can safely
        mutate fields that happen to be mutable-but-contained (none in
        v0.1, but the contract should hold preemptively)."""
        a = mg25_plus()
        b = mg25_plus()
        assert a is not b
        assert a == b  # equal by value
