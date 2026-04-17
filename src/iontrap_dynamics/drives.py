# SPDX-License-Identifier: MIT
"""Laser-drive configuration type.

Defines :class:`DriveConfig` — the configuration record carried by a
single laser beam addressing the ion. Phase 1 Hamiltonian builders
(carrier, sideband, Mølmer–Sørensen, stroboscopic AC) accept a tuple of
``DriveConfig`` instances as input; no drive parameter is allowed to live
as a hidden default in solver code (Design Principle 2).

Unit conventions follow ``CONVENTIONS.md`` §1:

- Wavevector in m⁻¹
- Angular frequencies (Rabi rate, detuning) in rad·s⁻¹
- Phase in radians
- Polarisation vectors are dimensionless

Sign convention for detuning follows ``CONVENTIONS.md`` §4: δ = ω_laser −
ω_atom, so positive δ means blue-detuned. The field stores δ directly —
builders resolve ω_laser at Hamiltonian-construction time by combining δ
with the transition frequency from the associated ``IonSpecies``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .exceptions import ConventionError

_VECTOR_SHAPE_TOLERANCE = 1e-10


def _as_3_vector(name: str, value: NDArray[np.floating] | tuple[float, ...] | list[float]) -> NDArray[np.floating]:
    """Coerce and validate a 3-vector input.

    Shared helper — used by both :class:`DriveConfig` (wavevector,
    polarisation) and the future tensor-product builders that accept
    arbitrary spatial vectors.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ConventionError(
            f"{name} must be a length-3 vector; got shape {array.shape}"
        )
    return array


@dataclass(frozen=True, slots=True, kw_only=True)
class DriveConfig:
    """Configuration record for one laser drive on an ion.

    Parameters
    ----------
    k_vector_m_inv
        Laser wavevector ``k``, 3-vector in m⁻¹. CONVENTIONS.md §10 — the
        sign of the dot product ``k · b_{i,m}`` is physical (encodes the
        drive–mode phase relationship); do not pre-normalise or take the
        magnitude at this level.
    carrier_rabi_frequency_rad_s
        On-resonance Rabi frequency Ω, rad·s⁻¹. Strictly positive; the
        overall sign of the coupling is absorbed into :attr:`phase_rad`.
    detuning_rad_s
        Laser–atom detuning δ = ω_laser − ω_atom, rad·s⁻¹ (CONVENTIONS.md §4).
        Positive = blue-detuned; negative = red-detuned; zero = on resonance.
        Default 0.0 (on resonance).
    phase_rad
        Initial phase φ, rad. Modular 2π; values outside [−π, π] are
        accepted and preserved verbatim — builders apply the phase via
        ``exp(i φ)`` so no wrapping is needed. Default 0.0.
    polarisation
        Optional 3-vector polarisation direction, dimensionless. Builders
        that care about selection rules (E1 σ±, π drives) read this
        field; builders that work at the Rabi-amplitude level ignore it.
        Unit-norm is encouraged but not enforced — users may supply a
        non-normalised vector to encode amplitude weighting. Default
        ``None`` = unspecified.
    transition_label
        Optional label identifying which transition in the associated
        :class:`~iontrap_dynamics.species.IonSpecies` this drive couples
        to. Builders needing the transition frequency look it up via
        ``species.transition(drive.transition_label)``. Default ``None``
        = unspecified; set when the drive is associated with a named
        optical transition.
    """

    k_vector_m_inv: NDArray[np.floating]
    carrier_rabi_frequency_rad_s: float
    detuning_rad_s: float = 0.0
    phase_rad: float = 0.0
    polarisation: NDArray[np.floating] | None = None
    transition_label: str | None = None

    def __post_init__(self) -> None:
        # Coerce list/tuple inputs to ndarray and validate shape.
        # Writing to a frozen dataclass requires object.__setattr__.
        k = _as_3_vector("k_vector_m_inv", self.k_vector_m_inv)
        object.__setattr__(self, "k_vector_m_inv", k)

        if self.polarisation is not None:
            p = _as_3_vector("polarisation", self.polarisation)
            object.__setattr__(self, "polarisation", p)

        if self.carrier_rabi_frequency_rad_s <= 0.0:
            raise ConventionError(
                f"carrier_rabi_frequency_rad_s must be positive; "
                f"got {self.carrier_rabi_frequency_rad_s!r}. Sign is absorbed into phase_rad."
            )

    @property
    def wavenumber_m_inv(self) -> float:
        """Return the scalar wavenumber |k| (magnitude of the wavevector)."""
        return float(np.linalg.norm(self.k_vector_m_inv))


__all__ = [
    "DriveConfig",
]
