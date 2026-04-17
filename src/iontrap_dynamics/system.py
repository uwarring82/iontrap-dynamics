# SPDX-License-Identifier: MIT
"""Composition of species, drives, and motional modes into a full
trapped-ion configuration.

Defines :class:`IonSystem` — the central configuration record that Phase 1
Hamiltonian builders, state-prep routines, and observable definitions
accept as their physical-parameter input. Every hidden atomic-physics
default is forbidden by Design Principle 2; the input surface is this
dataclass.

Composition rules
-----------------

An :class:`IonSystem` holds:

- ``species_per_ion`` — one :class:`~iontrap_dynamics.species.IonSpecies`
  per ion in the crystal. Same species for every ion in the common
  single-species case; different species for mixed-species experiments
  (e.g. ²⁵Mg⁺ + ⁴⁰Ca⁺ sympathetic cooling).
- ``drives`` — tuple of :class:`~iontrap_dynamics.drives.DriveConfig`
  records describing the laser beams addressing the crystal.
- ``modes`` — tuple of :class:`~iontrap_dynamics.modes.ModeConfig` records
  describing the normal-mode spectrum. Each mode must have
  ``n_ions`` matching ``len(species_per_ion)``.

Cross-validation at construction (all via :class:`ConventionError`):

1. At least one ion present.
2. All modes' ``n_ions`` match the crystal size.
3. Mode labels are unique.
4. If a drive's ``transition_label`` is set, some species in the crystal
   carries that transition — the drive must reference a physically
   present transition.

Convention version snapshot
---------------------------

Every :class:`IonSystem` records the ``CONVENTION_VERSION`` at its
construction. Downstream :class:`~iontrap_dynamics.results.TrajectoryResult`
objects can copy this into their metadata so the convention version is
unambiguously tied to the physics that produced the result.

Trap frame note
---------------

CONVENTIONS.md §12 specifies right-handed Cartesian coordinates with the
z-axis along the trap symmetry axis for linear Paul traps. For v0.1,
this convention is assumed implicitly; non-linear geometries (zigzag,
2D crystals, surface traps) will require an explicit ``trap_frame``
field that declares the axis convention. Not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .conventions import CONVENTION_VERSION
from .drives import DriveConfig
from .exceptions import ConventionError
from .modes import ModeConfig
from .species import IonSpecies


@dataclass(frozen=True, slots=True, kw_only=True)
class IonSystem:
    """Composite configuration for a trapped-ion crystal.

    Parameters
    ----------
    species_per_ion
        One :class:`IonSpecies` per ion in the crystal, in ion-index order.
        Must contain at least one element.
    drives
        Tuple of :class:`DriveConfig` records. May be empty for pure-
        motion or dark-state simulations.
    modes
        Tuple of :class:`ModeConfig` records. May be empty for pure-spin
        simulations with no motional dynamics. When non-empty, every mode
        must share the crystal's ion count and carry a unique label.
    convention_version
        Defaults to :data:`~iontrap_dynamics.CONVENTION_VERSION` at
        construction; clients should almost never override. Preserved on
        results downstream so the convention version travels with the
        output.

    Raises
    ------
    ConventionError
        On empty crystal, mode ``n_ions`` mismatch, duplicate mode
        labels, or drive transition-label references to a transition no
        species in the crystal carries.
    """

    species_per_ion: tuple[IonSpecies, ...]
    drives: tuple[DriveConfig, ...] = ()
    modes: tuple[ModeConfig, ...] = ()
    convention_version: str = field(default=CONVENTION_VERSION)

    def __post_init__(self) -> None:
        if not self.species_per_ion:
            raise ConventionError("species_per_ion must contain at least one ion")

        n_ions = len(self.species_per_ion)

        # Mode n_ions consistency
        for mode in self.modes:
            if mode.n_ions != n_ions:
                raise ConventionError(
                    f"mode {mode.label!r} has n_ions={mode.n_ions} but the crystal "
                    f"has {n_ions} ions; every mode's eigenvector_per_ion must match."
                )

        # Mode-label uniqueness
        labels = [m.label for m in self.modes]
        if len(labels) != len(set(labels)):
            duplicates = sorted({lab for lab in labels if labels.count(lab) > 1})
            raise ConventionError(
                f"duplicate mode labels: {duplicates}. Each mode must carry a unique label."
            )

        # Drive transition-label references
        all_transition_labels: set[str] = set()
        for sp in self.species_per_ion:
            for t in sp.transitions:
                all_transition_labels.add(t.label)
        for i, drv in enumerate(self.drives):
            if (
                drv.transition_label is not None
                and drv.transition_label not in all_transition_labels
            ):
                raise ConventionError(
                    f"drives[{i}].transition_label = {drv.transition_label!r} is not "
                    f"carried by any species in the crystal. Available: "
                    f"{sorted(all_transition_labels)!r}"
                )

    # ------------------------------------------------------------------------
    # Simple accessors
    # ------------------------------------------------------------------------

    @property
    def n_ions(self) -> int:
        """Return the number of ions in the crystal."""
        return len(self.species_per_ion)

    @property
    def n_drives(self) -> int:
        """Return the number of laser drives."""
        return len(self.drives)

    @property
    def n_modes(self) -> int:
        """Return the number of motional modes."""
        return len(self.modes)

    def species(self, ion_index: int) -> IonSpecies:
        """Return the :class:`IonSpecies` at the given ion index.

        Raises
        ------
        IndexError
            If ``ion_index`` is outside ``[0, n_ions)``.
        """
        if not 0 <= ion_index < self.n_ions:
            raise IndexError(
                f"ion_index {ion_index} out of range for a crystal with {self.n_ions} ions"
            )
        return self.species_per_ion[ion_index]

    def mode(self, label: str) -> ModeConfig:
        """Return the mode with the given label or raise.

        Raises
        ------
        ConventionError
            If no mode has the given label.
        """
        for m in self.modes:
            if m.label == label:
                return m
        available = [m.label for m in self.modes]
        raise ConventionError(f"unknown mode label: {label!r}. Available: {available!r}")

    @property
    def is_homogeneous(self) -> bool:
        """Return True if every ion carries the same species (by equality).

        Useful for Phase 1 builders that take shortcuts in the common
        single-species case.
        """
        if self.n_ions <= 1:
            return True
        first = self.species_per_ion[0]
        return all(sp == first for sp in self.species_per_ion[1:])

    # ------------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------------

    @classmethod
    def homogeneous(
        cls,
        *,
        species: IonSpecies,
        n_ions: int,
        drives: tuple[DriveConfig, ...] = (),
        modes: tuple[ModeConfig, ...] = (),
    ) -> IonSystem:
        """Construct an :class:`IonSystem` with the same species on every ion.

        Short-hand for the common single-species case. Equivalent to
        ``IonSystem(species_per_ion=(species,) * n_ions, ...)``.

        Raises
        ------
        ConventionError
            If ``n_ions`` is non-positive, or via the same cross-validation
            rules as the main constructor.
        """
        if n_ions <= 0:
            raise ConventionError(
                f"n_ions must be positive; got {n_ions}. A homogeneous crystal needs at "
                "least one ion."
            )
        return cls(
            species_per_ion=(species,) * n_ions,
            drives=drives,
            modes=modes,
        )


__all__ = [
    "IonSystem",
]
