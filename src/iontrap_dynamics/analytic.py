# SPDX-License-Identifier: MIT
"""Closed-form reference formulas for the analytic-regression tier.

This module provides canonical analytic expressions for well-known
trapped-ion observables and operators. The formulas serve two roles:

1. **Reference truth for the analytic-regression tier** (workplan §0.B).
   Builder output (Phase 1+) will be compared against these expressions to
   detect solver drift. Invariants are numerical anchors; analytic formulas
   are physical ones.

2. **Pedagogical reference.** A Clock-School student reading this module
   should find the same formulas they see in their textbook — with the
   same sign conventions, units, and variable names this library uses
   throughout (``CONVENTIONS.md``).

Unit conventions (SI internally — ``CONVENTIONS.md`` §1):

- Angular frequencies in rad·s⁻¹ (not 2π·MHz)
- Times in s (not μs)
- Masses in kg
- Wavevectors in m⁻¹
- Mode eigenvectors dimensionless

Sign conventions match ``CONVENTIONS.md``:

- σ_z basis: ⟨↓|σ_z|↓⟩ = −1, ⟨↑|σ_z|↑⟩ = +1 (atomic-physics convention, §3)
- Lamb–Dicke sign is physical and preserved by :func:`lamb_dicke_parameter`
  (§10)
- Detuning δ = ω_laser − ω_atom (§4) — positive δ = blue-detuned
"""

from __future__ import annotations

import numpy as np
import scipy.constants as const
from numpy.typing import ArrayLike, NDArray

#: Reduced Planck constant (CODATA). Re-exported here so the analytic module
#: is self-contained for readers.
HBAR: float = float(const.hbar)


# ----------------------------------------------------------------------------
# Carrier flopping (single ion, on-resonance, no motion)
# ----------------------------------------------------------------------------


def carrier_rabi_excited_population(
    omega_rabi: float,
    t: ArrayLike,
) -> NDArray[np.floating]:
    """Return ⟨P↑⟩(t) for on-resonance carrier flopping starting in |↓⟩.

    ``P↑(t) = sin²(Ω · t / 2)``

    Derived from ``H_carrier = (ℏΩ/2)(σ_+ + σ_−)`` (CONVENTIONS.md §5) acting
    on the initial state |↓⟩. Phase φ is absorbed into the definition of
    σ_± and does not affect the population.

    Parameters
    ----------
    omega_rabi
        Carrier Rabi frequency Ω, rad·s⁻¹.
    t
        Time or array of times, s.

    Returns
    -------
    np.ndarray
        Excited-state population, same shape as ``t``. Values in [0, 1].
    """
    time = np.asarray(t, dtype=np.float64)
    return np.sin(omega_rabi * time / 2.0) ** 2


def carrier_rabi_sigma_z(
    omega_rabi: float,
    t: ArrayLike,
) -> NDArray[np.floating]:
    """Return ⟨σ_z⟩(t) for on-resonance carrier flopping starting in |↓⟩.

    ``⟨σ_z⟩(t) = −cos(Ω · t)``

    Follows from the population formula with the atomic-physics Pauli
    convention (CONVENTIONS.md §3): ⟨σ_z⟩ = ⟨P↑⟩ − ⟨P↓⟩ =
    sin²(Ωt/2) − cos²(Ωt/2) = −cos(Ωt).

    At ``t = 0`` (all population in |↓⟩), ⟨σ_z⟩ = −1. After a π-pulse
    (``t = π/Ω``), ⟨σ_z⟩ = +1.

    Parameters
    ----------
    omega_rabi
        Carrier Rabi frequency Ω, rad·s⁻¹.
    t
        Time or array of times, s.

    Returns
    -------
    np.ndarray
        Expectation value of σ_z, same shape as ``t``. Values in [−1, +1].
    """
    time = np.asarray(t, dtype=np.float64)
    return -np.cos(omega_rabi * time)


# ----------------------------------------------------------------------------
# Sideband Rabi frequencies (Lamb–Dicke leading order)
# ----------------------------------------------------------------------------


def red_sideband_rabi_frequency(
    *,
    carrier_rabi_frequency: float,
    lamb_dicke_parameter: float,
    n_initial: int,
) -> float:
    """Return the leading-order red-sideband Rabi frequency.

    ``Ω_{n → n−1} = |η| · √n · Ω``    for ``n ≥ 1``, else ``0``.

    Couples |↓, n⟩ ↔ |↑, n − 1⟩. The vacuum state cannot be red-detuned off
    the motional ground, so ``n_initial = 0`` returns 0 exactly (this is
    the "vacuum-state red sideband" sanity check in workplan §0.B).

    |η| is used rather than η because the Rabi frequency is a magnitude;
    the sign of η appears in the phase of the coupling, not the rate
    (CONVENTIONS.md §10, "|η| is reserved for derived quantities").

    Parameters
    ----------
    carrier_rabi_frequency
        Carrier Rabi frequency Ω, rad·s⁻¹.
    lamb_dicke_parameter
        Lamb–Dicke parameter η, dimensionless. Sign is discarded here.
    n_initial
        Initial Fock level n. Non-negative integer.

    Returns
    -------
    float
        Red-sideband Rabi frequency, rad·s⁻¹. Zero when ``n_initial == 0``.

    Raises
    ------
    ValueError
        If ``n_initial < 0``.
    """
    if n_initial < 0:
        raise ValueError(f"n_initial must be non-negative; got {n_initial}")
    if n_initial == 0:
        return 0.0
    return float(abs(lamb_dicke_parameter) * np.sqrt(n_initial) * carrier_rabi_frequency)


def blue_sideband_rabi_frequency(
    *,
    carrier_rabi_frequency: float,
    lamb_dicke_parameter: float,
    n_initial: int,
) -> float:
    """Return the leading-order blue-sideband Rabi frequency.

    ``Ω_{n → n+1} = |η| · √(n + 1) · Ω``    for ``n ≥ 0``.

    Couples |↓, n⟩ ↔ |↑, n + 1⟩. Non-zero even from the vacuum state,
    since one phonon can always be created.

    Parameters
    ----------
    carrier_rabi_frequency
        Carrier Rabi frequency Ω, rad·s⁻¹.
    lamb_dicke_parameter
        Lamb–Dicke parameter η, dimensionless. Sign is discarded here.
    n_initial
        Initial Fock level n. Non-negative integer.

    Returns
    -------
    float
        Blue-sideband Rabi frequency, rad·s⁻¹. Non-zero for any ``n_initial``.

    Raises
    ------
    ValueError
        If ``n_initial < 0``.
    """
    if n_initial < 0:
        raise ValueError(f"n_initial must be non-negative; got {n_initial}")
    return float(abs(lamb_dicke_parameter) * np.sqrt(n_initial + 1) * carrier_rabi_frequency)


# ----------------------------------------------------------------------------
# Lamb–Dicke parameter (CONVENTIONS.md §10, full 3D)
# ----------------------------------------------------------------------------


def lamb_dicke_parameter(
    *,
    k_vec: ArrayLike,
    mode_eigenvector: ArrayLike,
    ion_mass: float,
    mode_frequency: float,
) -> float:
    """Return the Lamb–Dicke parameter of an ion with respect to a mode.

    ``η_{i,m} = (k⃗ · b⃗_{i,m}) · √(ℏ / (2 · m_i · ω_m))``

    Implements the full 3D projection mandated in CONVENTIONS.md §10. No 1D
    shortcut. The sign of the dot product is preserved — η can be negative,
    and the sign is physical (it encodes the relative phase of the drive
    with respect to the mode displacement).

    Three test geometries are exercised in ``tests/regression/analytic/``:

    - **k ∥ b**: η reduces to the 1D formula ``|k| · √(ℏ / 2mω)``.
    - **k ⊥ b**: η is identically zero.
    - **Oblique**: η = ``|k| · cos(θ) · √(ℏ / 2mω)`` for the angle θ
      between ``k`` and ``b``.

    Parameters
    ----------
    k_vec
        Laser wavevector ``k``, 3-vector in m⁻¹.
    mode_eigenvector
        Mode eigenvector ``b_{i,m}`` at the ion, 3-vector, dimensionless.
        Should satisfy ``Σ_i |b_{i,m}|² = 1`` across ions (CONVENTIONS.md §11);
        not checked here because this function operates on a single
        ion–mode pair.
    ion_mass
        Ion mass ``m_i``, kg.
    mode_frequency
        Mode angular frequency ``ω_m``, rad·s⁻¹.

    Returns
    -------
    float
        Lamb–Dicke parameter η, dimensionless. Sign preserved.

    Raises
    ------
    ValueError
        If either vector is not length-3, or if ``ion_mass`` or
        ``mode_frequency`` is non-positive.
    """
    k = np.asarray(k_vec, dtype=np.float64)
    b = np.asarray(mode_eigenvector, dtype=np.float64)
    if k.shape != (3,):
        raise ValueError(f"k_vec must be a 3-vector; got shape {k.shape}")
    if b.shape != (3,):
        raise ValueError(f"mode_eigenvector must be a 3-vector; got shape {b.shape}")
    if ion_mass <= 0.0:
        raise ValueError(f"ion_mass must be positive; got {ion_mass}")
    if mode_frequency <= 0.0:
        raise ValueError(f"mode_frequency must be positive; got {mode_frequency}")

    projection = float(np.dot(k, b))  # m⁻¹
    zero_point = np.sqrt(HBAR / (2.0 * ion_mass * mode_frequency))  # m
    return float(projection * zero_point)


# ----------------------------------------------------------------------------
# Coherent states
# ----------------------------------------------------------------------------


def coherent_state_mean_n(alpha: complex) -> float:
    """Return the mean phonon number of a coherent state ``|α⟩``.

    ``⟨n⟩_{|α⟩} = |α|²``

    This is the displacement-operator action check (workplan §0.B analytic
    list): displacing the vacuum by α produces a coherent state whose
    occupation is exactly |α|².

    Parameters
    ----------
    alpha
        Coherent-state amplitude ``α = |α|·exp(iφ)`` (CONVENTIONS.md §7),
        dimensionless.

    Returns
    -------
    float
        Mean phonon number ``⟨n⟩``. Non-negative.
    """
    return float(abs(alpha) ** 2)


__all__ = [
    "HBAR",
    "blue_sideband_rabi_frequency",
    "carrier_rabi_excited_population",
    "carrier_rabi_sigma_z",
    "coherent_state_mean_n",
    "lamb_dicke_parameter",
    "red_sideband_rabi_frequency",
]
