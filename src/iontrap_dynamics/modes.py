# SPDX-License-Identifier: MIT
"""Motional-mode configuration type.

Defines :class:`ModeConfig` — the configuration record for one normal
mode of the ion-crystal's motion. Phase 1 Hamiltonian builders combine
modes with :class:`~iontrap_dynamics.species.IonSpecies` and
:class:`~iontrap_dynamics.drives.DriveConfig` to compute Lamb–Dicke
parameters (CONVENTIONS.md §10) and build sideband couplings.

Unit conventions (CONVENTIONS.md §1, §11):

- Mode frequency in rad·s⁻¹
- Eigenvectors dimensionless, one 3-vector per ion
- Per-mode normalisation: Σ_i ||b_{i,m}||² = 1 across ions
  (CONVENTIONS.md §11)

Mode eigenvectors come from an external normal-mode solver (e.g.
``pylion``, ``trical``) or analytic diagonalisation of the
ion-crystal's trap + Coulomb potential. :class:`ModeConfig` treats them
as given configuration; it does not attempt to derive them from trap
parameters. This is deliberate — Design Principle 2 bars the library
from guessing mode structure from first principles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .exceptions import ConventionError

#: Tolerance for the per-mode normalisation check (§11). Matches the
#: bit-exact tier of the reproducibility ladder (CONVENTIONS.md §14)
#: with some slack for float round-off in externally-supplied mode
#: solvers.
_NORMALISATION_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True, kw_only=True)
class ModeConfig:
    """Configuration record for one motional normal mode.

    Parameters
    ----------
    label
        Human-readable identifier (e.g. ``"axial"``, ``"radial_x"``,
        ``"com"``). Used as the lookup key when multiple modes are
        gathered on an :class:`~iontrap_dynamics.system.IonSystem`.
    frequency_rad_s
        Mode angular frequency ω_m, rad·s⁻¹ (CONVENTIONS.md §1). Must be
        positive — motional modes don't have zero or negative secular
        frequencies in any supported trap geometry.
    eigenvector_per_ion
        Normal-mode eigenvector evaluated at each ion in the crystal.
        Shape ``(N_ions, 3)``, dimensionless. Element
        ``eigenvector_per_ion[i, :]`` is the 3-vector displacement
        amplitude of ion ``i`` in this mode. CONVENTIONS.md §11 mandates
        the per-mode normalisation

            Σ_i ||b_{i,m}||² = 1

        which is verified at construction time to the tolerance
        :data:`_NORMALISATION_TOLERANCE`.

    Raises
    ------
    ConventionError
        If ``frequency_rad_s`` is non-positive, if ``eigenvector_per_ion``
        has the wrong shape, or if the per-mode normalisation is not
        satisfied within tolerance.
    """

    label: str
    frequency_rad_s: float
    eigenvector_per_ion: NDArray[np.floating]

    def __post_init__(self) -> None:
        if not self.label:
            raise ConventionError("label must be a non-empty string")

        if self.frequency_rad_s <= 0.0:
            raise ConventionError(f"frequency_rad_s must be positive; got {self.frequency_rad_s!r}")

        ev = np.asarray(self.eigenvector_per_ion, dtype=np.float64)
        object.__setattr__(self, "eigenvector_per_ion", ev)

        if ev.ndim != 2 or ev.shape[1] != 3:
            raise ConventionError(
                f"eigenvector_per_ion must have shape (N_ions, 3); got {ev.shape}"
            )
        if ev.shape[0] == 0:
            raise ConventionError("eigenvector_per_ion must contain at least one ion")

        total_sq = float(np.sum(np.abs(ev) ** 2))
        if abs(total_sq - 1.0) > _NORMALISATION_TOLERANCE:
            raise ConventionError(
                f"mode eigenvector violates CONVENTIONS.md §11 normalisation: "
                f"Σᵢ ||b_{{i,m}}||² = {total_sq!r}, expected 1 within {_NORMALISATION_TOLERANCE}. "
                "Check the eigensolver output or apply a 1/√norm rescaling."
            )

    @property
    def n_ions(self) -> int:
        """Return the number of ions this mode is defined for."""
        return int(self.eigenvector_per_ion.shape[0])

    def eigenvector_at_ion(self, ion_index: int) -> NDArray[np.floating]:
        """Return the 3-vector mode eigenvector at the given ion index.

        Returns a fresh copy so downstream code cannot mutate the stored
        array through the returned reference.

        Raises
        ------
        IndexError
            If ``ion_index`` is outside ``[0, n_ions)``.
        """
        if not 0 <= ion_index < self.n_ions:
            raise IndexError(
                f"ion_index {ion_index} out of range for a mode with {self.n_ions} ions"
            )
        return self.eigenvector_per_ion[ion_index].copy()


__all__ = [
    "ModeConfig",
]
