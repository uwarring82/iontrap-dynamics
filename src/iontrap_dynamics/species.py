# SPDX-License-Identifier: MIT
"""Ion-species configuration types.

Defines :class:`IonSpecies` (element, isotope, mass, transitions, charge)
and the :class:`Transition` records it carries, plus lightweight factories
for the species most commonly used in the lab (²⁵Mg⁺, ⁴⁰Ca⁺, ⁴³Ca⁺).
Every solver call in the library accepts species as configuration — no
atomic-physics value is allowed to live as a hidden default in solver
code (Design Principle 2, ``CONVENTIONS.md`` §1 and workplan §2).

Scope
-----

**In scope for v0.1:** mass, charge, named optical transitions (label,
wavelength, multipole order, optional natural linewidth). This is what
the builders in Phase 1 (carrier, red/blue sideband, Mølmer–Sørensen)
need to build their Hamiltonians.

**Deferred:** nuclear spin, hyperfine splittings, Zeeman sub-levels,
branching ratios, metastable-state lifetimes. These are needed for the
measurement/systematics layers (Phase 1+) and will extend :class:`Transition`
or add sibling types without breaking the v0.1 API.

Factory values
--------------

The factory functions (:func:`mg25_plus`, :func:`ca40_plus`, :func:`ca43_plus`)
return convenience presets with nominal atomic-physics values. Serious
experiments should supply lab-calibrated wavelengths and masses as
configuration — the factories exist to lower the barrier to a first
"hello world" simulation, not to be authoritative references for
spectroscopy-grade work. CONVENTIONS.md §1 units apply throughout: mass
in kg, wavelength in m, linewidth in rad·s⁻¹.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .exceptions import ConventionError

# ----------------------------------------------------------------------------
# Multipole order
# ----------------------------------------------------------------------------


class TransitionType(StrEnum):
    """Electromagnetic multipole order of an optical transition.

    * ``E1`` — electric dipole (the "strong" lines: D1, D2 in alkali-like
      ions; short natural lifetimes, typically hundreds of MHz).
    * ``E2`` — electric quadrupole (optical "clock" transitions in
      alkaline-earth-like ions; narrow natural linewidths, long lifetimes).
    * ``M1`` — magnetic dipole (e.g. hyperfine transitions when used
      optically; sub-Hz linewidths).
    """

    E1 = "E1"
    E2 = "E2"
    M1 = "M1"


# ----------------------------------------------------------------------------
# Transition record
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Transition:
    """One named optical transition of an ion species.

    Parameters
    ----------
    label
        Human-readable identifier (e.g. ``"3s²S₁/₂ → 3p²P₁/₂"``). Used as
        the lookup key in :meth:`IonSpecies.transition`.
    wavelength_m
        Transition wavelength in metres (CONVENTIONS.md §1).
    transition_type
        Multipole order — see :class:`TransitionType`.
    linewidth_rad_s
        Natural linewidth Γ, in rad·s⁻¹. Optional; set when the transition
        will drive dissipative dynamics in the Phase 1+ observation/apparatus
        layers.
    """

    label: str
    wavelength_m: float
    transition_type: TransitionType
    linewidth_rad_s: float | None = None

    def __post_init__(self) -> None:
        if self.wavelength_m <= 0.0:
            raise ConventionError(f"wavelength_m must be positive; got {self.wavelength_m!r}")
        if self.linewidth_rad_s is not None and self.linewidth_rad_s < 0.0:
            raise ConventionError(
                f"linewidth_rad_s must be non-negative; got {self.linewidth_rad_s!r}"
            )


# ----------------------------------------------------------------------------
# Ion species
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class IonSpecies:
    """Configuration record for one trapped-ion species.

    All fields are frozen and keyword-only (Design Principle 6).
    Transitions are stored as a tuple, so the set of named transitions
    cannot be mutated after construction. Lookup is via
    :meth:`transition` rather than dictionary access; that keeps
    uniqueness-validation on the construction path.

    Parameters
    ----------
    element
        Chemical symbol (``"Mg"``, ``"Ca"``, ``"Sr"``, ``"Yb"``, …).
    mass_number
        Isotope mass number (e.g. 25 for ²⁵Mg, 40 for ⁴⁰Ca).
    mass_kg
        Atomic mass in kilograms (CONVENTIONS.md §1).
    charge
        Ion charge state. Defaults to ``+1`` for singly-charged cations,
        which covers the vast majority of trapped-ion experiments.
    transitions
        Tuple of :class:`Transition` records. Labels must be unique
        within the tuple.
    """

    element: str
    mass_number: int
    mass_kg: float
    charge: int = +1
    transitions: tuple[Transition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.element:
            raise ConventionError("element must be a non-empty string")
        if self.mass_number <= 0:
            raise ConventionError(
                f"mass_number must be a positive integer; got {self.mass_number!r}"
            )
        if self.mass_kg <= 0.0:
            raise ConventionError(f"mass_kg must be positive; got {self.mass_kg!r}")
        labels = [t.label for t in self.transitions]
        if len(labels) != len(set(labels)):
            duplicates = sorted({lab for lab in labels if labels.count(lab) > 1})
            raise ConventionError(
                f"duplicate transition labels: {duplicates}. Each transition "
                "must carry a unique label within an IonSpecies."
            )

    # ------------------------------------------------------------------------
    # Lookups and derived properties
    # ------------------------------------------------------------------------

    def transition(self, label: str) -> Transition:
        """Return the transition with the given label or raise.

        Raises
        ------
        ConventionError
            If no transition in :attr:`transitions` has the given label.
        """
        for t in self.transitions:
            if t.label == label:
                return t
        available = [t.label for t in self.transitions]
        raise ConventionError(f"unknown transition label: {label!r}. Available: {available!r}")

    @property
    def name(self) -> str:
        """Conventional textual name, e.g. ``"25Mg+"``.

        Charge state encoding: ``+1`` → ``"+"``, ``+2`` → ``"2+"``,
        ``-1`` → ``"-"``, ``0`` → ``""`` (neutral atom).
        """
        if self.charge == +1:
            charge_str = "+"
        elif self.charge == -1:
            charge_str = "-"
        elif self.charge > 0:
            charge_str = f"{self.charge}+"
        elif self.charge < 0:
            charge_str = f"{-self.charge}-"
        else:
            charge_str = ""
        return f"{self.mass_number}{self.element}{charge_str}"


# ----------------------------------------------------------------------------
# Physical constants (CODATA, recapped for standalone use)
# ----------------------------------------------------------------------------

#: Atomic mass unit in kg (CODATA 2018). Used as the conversion factor in
#: the factory functions below; clients can also import this for their
#: own mass-specification code.
ATOMIC_MASS_UNIT_KG: float = 1.66053906660e-27


# ----------------------------------------------------------------------------
# Factory functions — nominal values for common species
# ----------------------------------------------------------------------------
#
# Nominal wavelengths are typical values from standard references (Drake,
# Lucas et al.) sufficient for a first-simulation "hello world". For
# spectroscopy-grade work, override the wavelengths at construction time.


def mg25_plus() -> IonSpecies:
    """Return a nominal ²⁵Mg⁺ species with its two principal dipole
    transitions (S₁/₂ ↔ P₁/₂, S₁/₂ ↔ P₃/₂)."""
    return IonSpecies(
        element="Mg",
        mass_number=25,
        charge=+1,
        mass_kg=24.98583696 * ATOMIC_MASS_UNIT_KG,
        transitions=(
            Transition(
                label="3s²S₁/₂ → 3p²P₁/₂",
                wavelength_m=280.353e-9,
                transition_type=TransitionType.E1,
            ),
            Transition(
                label="3s²S₁/₂ → 3p²P₃/₂",
                wavelength_m=279.630e-9,
                transition_type=TransitionType.E1,
            ),
        ),
    )


def ca40_plus() -> IonSpecies:
    """Return a nominal ⁴⁰Ca⁺ species with D1, D2, and the S→D₅/₂ clock
    transition."""
    return IonSpecies(
        element="Ca",
        mass_number=40,
        charge=+1,
        mass_kg=39.96259098 * ATOMIC_MASS_UNIT_KG,
        transitions=(
            Transition(
                label="4s²S₁/₂ → 4p²P₁/₂",
                wavelength_m=396.959e-9,
                transition_type=TransitionType.E1,
            ),
            Transition(
                label="4s²S₁/₂ → 4p²P₃/₂",
                wavelength_m=393.366e-9,
                transition_type=TransitionType.E1,
            ),
            Transition(
                label="4s²S₁/₂ → 3d²D₅/₂",
                wavelength_m=729.147e-9,
                transition_type=TransitionType.E2,
            ),
        ),
    )


def ca43_plus() -> IonSpecies:
    """Return a nominal ⁴³Ca⁺ species with D1, D2, and the S→D₅/₂ clock
    transition. ⁴³Ca has nuclear spin I = 7/2, giving it a hyperfine
    structure — the hyperfine levels themselves are not yet modelled
    at the :class:`IonSpecies` level in v0.1 (deferred to Phase 1+).
    """
    return IonSpecies(
        element="Ca",
        mass_number=43,
        charge=+1,
        mass_kg=42.9587666 * ATOMIC_MASS_UNIT_KG,
        transitions=(
            Transition(
                label="4s²S₁/₂ → 4p²P₁/₂",
                wavelength_m=396.959e-9,
                transition_type=TransitionType.E1,
            ),
            Transition(
                label="4s²S₁/₂ → 4p²P₃/₂",
                wavelength_m=393.366e-9,
                transition_type=TransitionType.E1,
            ),
            Transition(
                label="4s²S₁/₂ → 3d²D₅/₂",
                wavelength_m=729.147e-9,
                transition_type=TransitionType.E2,
            ),
        ),
    )


__all__ = [
    "ATOMIC_MASS_UNIT_KG",
    "IonSpecies",
    "Transition",
    "TransitionType",
    "ca40_plus",
    "ca43_plus",
    "mg25_plus",
]
