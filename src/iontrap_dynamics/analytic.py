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

from enum import StrEnum

import numpy as np
import scipy.constants as const
from numpy.typing import ArrayLike, NDArray
from scipy.special import eval_genlaguerre

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
# Detuned carrier (off-resonance Rabi flopping)
# ----------------------------------------------------------------------------


def generalized_rabi_frequency(
    *,
    carrier_rabi_frequency: float,
    detuning_rad_s: float,
) -> float:
    """Return the generalised Rabi frequency ``Ω_gen = √(Ω² + δ²)``.

    The off-resonance carrier oscillates at ``Ω_gen`` rather than
    ``Ω``; the amplitude of the population swing drops as the
    detuning grows:

    .. math::
        P_{\\uparrow,\\max} = \\bigl( \\Omega / \\Omega_\\text{gen} \\bigr)^2
        = \\frac{\\Omega^2}{\\Omega^2 + \\delta^2}.

    Parameters
    ----------
    carrier_rabi_frequency
        On-resonance Rabi frequency Ω, rad·s⁻¹.
    detuning_rad_s
        Detuning δ = ω_laser − ω_atom (CONVENTIONS.md §4),
        rad·s⁻¹. Sign does not affect ``Ω_gen``.

    Returns
    -------
    float
        Generalised Rabi frequency, rad·s⁻¹. Always non-negative.
    """
    return float(np.sqrt(carrier_rabi_frequency**2 + detuning_rad_s**2))


def detuned_rabi_sigma_z(
    *,
    carrier_rabi_frequency: float,
    detuning_rad_s: float,
    t: ArrayLike,
) -> NDArray[np.floating]:
    """Return ⟨σ_z⟩(t) for detuned-carrier flopping starting in |↓⟩.

    .. math::
        \\langle \\sigma_z \\rangle(t) = -1
        + 2 \\, (\\Omega / \\Omega_\\text{gen})^2
          \\sin^2(\\Omega_\\text{gen} t / 2)

    where ``Ω_gen = √(Ω² + δ²)``. The population is frame-invariant
    (the atomic-transition interaction-picture and the rotating-frame
    expressions differ by a σ_z rotation that leaves ⟨σ_z⟩ unchanged),
    so this formula applies to both the time-dependent interaction-
    picture Hamiltonian returned by
    :func:`iontrap_dynamics.hamiltonians.detuned_carrier_hamiltonian`
    and the rotating-frame form.

    At ``δ = 0`` this reduces to the on-resonance
    :func:`carrier_rabi_sigma_z`.

    Parameters
    ----------
    carrier_rabi_frequency
        On-resonance Rabi frequency Ω, rad·s⁻¹.
    detuning_rad_s
        Detuning δ, rad·s⁻¹.
    t
        Time or array of times, s.

    Returns
    -------
    np.ndarray
        ⟨σ_z⟩(t), same shape as ``t``. Values in ``[−1, +1]``.
        At ``t = 0`` evaluates to exactly ``−1`` (ground state).
    """
    time = np.asarray(t, dtype=np.float64)
    omega_gen = generalized_rabi_frequency(
        carrier_rabi_frequency=carrier_rabi_frequency,
        detuning_rad_s=detuning_rad_s,
    )
    amplitude_squared = (carrier_rabi_frequency / omega_gen) ** 2
    return -1.0 + 2.0 * amplitude_squared * np.sin(omega_gen * time / 2.0) ** 2


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
# Lamb–Dicke regime helpers (Debye–Waller, regime classifier, all-orders
# sideband Rabi) — Wineland et al. (1998), Leibfried et al. (2003)
# ----------------------------------------------------------------------------

#: Upper bound on the regime argument η²(2n̄+1) for the *deep* Lamb–Dicke
#: regime. At η²(2n̄+1) = 0.1 the Debye–Waller carrier suppression
#: e^{−η²(2n̄+1)/2} is 0.951 — under ~5 %, where the leading-order sideband
#: Rabi formulas are accurate to a few percent.
LAMB_DICKE_DEEP_MAX: float = 0.1

#: Upper bound on η²(2n̄+1) for the *intermediate* regime; at or above 1.0 the
#: carrier suppression is e^{−1/2} = 0.607 and the Laguerre series is no longer
#: perturbative (the *beyond* regime).
LAMB_DICKE_INTERMEDIATE_MAX: float = 1.0


class LambDickeRegime(StrEnum):
    """Qualitative Lamb–Dicke regime, classified from η²(2n̄+1).

    Mirrors the ``StrEnum`` idiom used elsewhere in the package
    (e.g. :class:`~iontrap_dynamics.results.StorageMode`). Members compare
    equal to their string values, so ``regime is LambDickeRegime.DEEP`` and
    ``regime == "deep"`` are both valid.
    """

    DEEP = "deep"
    INTERMEDIATE = "intermediate"
    BEYOND = "beyond"


def _validate_regime_inputs(lamb_dicke_parameter: float, mean_phonon_number: float) -> None:
    """Shared validation for the regime helpers: finite η, finite n̄ ≥ 0."""
    if not np.isfinite(lamb_dicke_parameter):
        raise ValueError(f"lamb_dicke_parameter must be finite; got {lamb_dicke_parameter}")
    if not np.isfinite(mean_phonon_number):
        raise ValueError(f"mean_phonon_number must be finite; got {mean_phonon_number}")
    if mean_phonon_number < 0.0:
        raise ValueError(f"mean_phonon_number must be non-negative; got {mean_phonon_number}")


def lamb_dicke_confinement(
    *,
    lamb_dicke_parameter: float,
    mean_phonon_number: float,
) -> float:
    """Return the Lamb–Dicke regime argument ``η²(2n̄+1)``.

    The squared root-mean-square ratio ``(k·Δx_rms)²`` of the thermal
    wavepacket extent to the drive wavelength: ``Δx_rms² = x_zpf²(2n̄+1)``, so
    ``η²(2n̄+1) = (k·Δx_rms)²``. The deep Lamb–Dicke regime is η²(2n̄+1) ≪ 1
    (Wineland et al. 1998; Leibfried et al. 2003). It is the natural classifier
    variable because the Debye–Waller carrier suppression is exactly
    ``e^{−η²(2n̄+1)/2}`` — see :func:`debye_waller_factor`.

    Parameters
    ----------
    lamb_dicke_parameter
        Lamb–Dicke parameter η, dimensionless. Sign is irrelevant — η² is
        used (CONVENTIONS.md §10: |η| reserved for derived quantities).
    mean_phonon_number
        Mean motional occupation n̄ ≥ 0.

    Returns
    -------
    float
        The regime argument η²(2n̄+1), dimensionless and non-negative.

    Raises
    ------
    ValueError
        If ``mean_phonon_number`` is negative, or either input is non-finite.
    """
    _validate_regime_inputs(lamb_dicke_parameter, mean_phonon_number)
    return float(lamb_dicke_parameter**2 * (2.0 * mean_phonon_number + 1.0))


def lamb_dicke_regime(
    *,
    lamb_dicke_parameter: float,
    mean_phonon_number: float,
) -> LambDickeRegime:
    """Classify the Lamb–Dicke regime from ``η²(2n̄+1)``.

    - :attr:`LambDickeRegime.DEEP` — ``η²(2n̄+1) < LAMB_DICKE_DEEP_MAX`` (0.1):
      carrier suppression < 5 %; the leading-order sideband Rabi formulas
      (:func:`red_sideband_rabi_frequency`, :func:`blue_sideband_rabi_frequency`)
      are trustworthy.
    - :attr:`LambDickeRegime.INTERMEDIATE` — ``LAMB_DICKE_DEEP_MAX ≤ η²(2n̄+1)
      < LAMB_DICKE_INTERMEDIATE_MAX`` (1.0): Debye–Waller suppression 5–40 %;
      the all-orders forms (:func:`blue_sideband_rabi_frequency_full_ld`,
      :func:`red_sideband_rabi_frequency_full_ld`) are needed for fidelity-grade
      accuracy.
    - :attr:`LambDickeRegime.BEYOND` — ``η²(2n̄+1) ≥ LAMB_DICKE_INTERMEDIATE_MAX``:
      the Laguerre series is non-perturbative (Rabi-rate nulls appear); only the
      all-orders closed form is valid.

    Thresholds are the module constants :data:`LAMB_DICKE_DEEP_MAX` and
    :data:`LAMB_DICKE_INTERMEDIATE_MAX`, tied to the Debye–Waller suppression
    e^{−η²(2n̄+1)/2} crossing 0.951 and 0.607.

    Parameters and Raises are as for :func:`lamb_dicke_confinement`.

    Returns
    -------
    LambDickeRegime
        ``DEEP``, ``INTERMEDIATE``, or ``BEYOND``.
    """
    confinement = lamb_dicke_confinement(
        lamb_dicke_parameter=lamb_dicke_parameter,
        mean_phonon_number=mean_phonon_number,
    )
    if confinement < LAMB_DICKE_DEEP_MAX:
        return LambDickeRegime.DEEP
    if confinement < LAMB_DICKE_INTERMEDIATE_MAX:
        return LambDickeRegime.INTERMEDIATE
    return LambDickeRegime.BEYOND


def debye_waller_factor(
    *,
    lamb_dicke_parameter: float,
    mean_phonon_number: float,
) -> float:
    """Return the Debye–Waller factor ``e^{−η²(2n̄+1)/2}``.

    The thermal suppression of the carrier (Δn = 0) Rabi amplitude for an ion
    in a thermal motional state of mean occupation n̄. Equivalently the thermal
    average of the per-Fock carrier matrix element,

        Σ_n p_n(n̄) · e^{−η²/2} L_n(η²) = e^{−η²(2n̄+1)/2},

    with ``p_n(n̄) = n̄ⁿ / (n̄+1)^{n+1}`` (proven via the displacement-operator
    characteristic function of a thermal state, or the Laguerre generating
    function). The regression tests validate the closed form against this
    truncated series and against the shipped full-Lamb–Dicke carrier operator.

    Note the exponent is ``η²(2n̄+1)/2 = η²(n̄ + 1/2)``: the vacuum (n̄ = 0)
    retains the zero-point suppression e^{−η²/2}, *not* unity.

    References: Wineland et al. (1998) §IV; Leibfried et al. (2003) §III.

    Parameters and Raises are as for :func:`lamb_dicke_confinement`.

    Returns
    -------
    float
        Debye–Waller factor in ``(0, 1]``.
    """
    confinement = lamb_dicke_confinement(
        lamb_dicke_parameter=lamb_dicke_parameter,
        mean_phonon_number=mean_phonon_number,
    )
    return float(np.exp(-confinement / 2.0))


def blue_sideband_rabi_frequency_full_ld(
    *,
    carrier_rabi_frequency: float,
    lamb_dicke_parameter: float,
    n_initial: int,
) -> float:
    """Return the all-orders (full Lamb–Dicke) blue-sideband Rabi frequency.

    ``Ω_{n→n+1} = |η| · e^{−η²/2} · |L_n^{(1)}(η²)| / √(n+1) · Ω``  (magnitude)

    The exact Wineland–Itano matrix element of ``e^{iη(a+a†)}`` for the first
    blue sideband |↓,n⟩ ↔ |↑,n+1⟩, of which :func:`blue_sideband_rabi_frequency`
    (``|η|√(n+1)·Ω``) is the η→0 leading-order limit. This is the
    "intermediate-order correction" that :func:`lamb_dicke_regime` flags as
    required past the deep regime; it is continuous *by identity* with the
    shipped ``full_lamb_dicke=True`` Hamiltonian builders.

    The Rabi *rate* is a magnitude, so the associated Laguerre polynomial —
    which changes sign past its first zero — enters as ``|L_n^{(1)}(η²)|``. The
    first explicit correction term is ``|η|√(n+1)·(1 − (n+1)η²/2)·Ω``, but that
    truncation turns unphysical (negative) once (n+1)η²/2 > 1, so the exact
    closed form is exposed instead.

    References: Wineland et al. (1998) Eq. (81); Leibfried et al. (2003) §III.

    Parameters
    ----------
    carrier_rabi_frequency
        Carrier Rabi frequency Ω, rad·s⁻¹.
    lamb_dicke_parameter
        Lamb–Dicke parameter η. Sign discarded (the rate is a magnitude).
    n_initial
        Initial Fock level n ≥ 0.

    Returns
    -------
    float
        Blue-sideband Rabi frequency, rad·s⁻¹. The "(magnitude)" applies to the
        matrix-element factor (``|η|·e^{−η²/2}·|L|/√(n+1)``); like the
        leading-order forms, the result carries the sign of
        ``carrier_rabi_frequency``.

    Raises
    ------
    ValueError
        If ``n_initial < 0`` or ``lamb_dicke_parameter`` is non-finite.
    """
    if n_initial < 0:
        raise ValueError(f"n_initial must be non-negative; got {n_initial}")
    if not np.isfinite(lamb_dicke_parameter):
        raise ValueError(f"lamb_dicke_parameter must be finite; got {lamb_dicke_parameter}")
    eta = abs(lamb_dicke_parameter)
    laguerre = abs(float(eval_genlaguerre(n_initial, 1, eta**2)))
    matrix_element = eta * np.exp(-(eta**2) / 2.0) * laguerre / np.sqrt(n_initial + 1)
    return float(matrix_element * carrier_rabi_frequency)


def red_sideband_rabi_frequency_full_ld(
    *,
    carrier_rabi_frequency: float,
    lamb_dicke_parameter: float,
    n_initial: int,
) -> float:
    """Return the all-orders (full Lamb–Dicke) red-sideband Rabi frequency.

    ``Ω_{n→n−1} = |η| · e^{−η²/2} · |L_{n−1}^{(1)}(η²)| / √n · Ω``  (magnitude),
    and exactly 0 for ``n_initial = 0`` (the vacuum cannot lose a phonon).

    The exact counterpart of the leading-order
    :func:`red_sideband_rabi_frequency` (``|η|√n·Ω``), which is its η→0 limit.
    See :func:`blue_sideband_rabi_frequency_full_ld` for the magnitude/sign and
    continuity notes; here the coupling is |↓,n⟩ ↔ |↑,n−1⟩.

    References: Wineland et al. (1998) Eq. (81); Leibfried et al. (2003) §III.

    Parameters
    ----------
    carrier_rabi_frequency
        Carrier Rabi frequency Ω, rad·s⁻¹.
    lamb_dicke_parameter
        Lamb–Dicke parameter η. Sign discarded (the rate is a magnitude).
    n_initial
        Initial Fock level n ≥ 0. Returns 0 exactly at ``n_initial = 0``.

    Returns
    -------
    float
        Red-sideband Rabi frequency, rad·s⁻¹. Zero when ``n_initial == 0``.

    Raises
    ------
    ValueError
        If ``n_initial < 0`` or ``lamb_dicke_parameter`` is non-finite.
    """
    if n_initial < 0:
        raise ValueError(f"n_initial must be non-negative; got {n_initial}")
    if not np.isfinite(lamb_dicke_parameter):
        raise ValueError(f"lamb_dicke_parameter must be finite; got {lamb_dicke_parameter}")
    if n_initial == 0:
        return 0.0
    eta = abs(lamb_dicke_parameter)
    laguerre = abs(float(eval_genlaguerre(n_initial - 1, 1, eta**2)))
    matrix_element = eta * np.exp(-(eta**2) / 2.0) * laguerre / np.sqrt(n_initial)
    return float(matrix_element * carrier_rabi_frequency)


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
# Mølmer–Sørensen gate (δ = 0, time-independent form)
# ----------------------------------------------------------------------------


def ms_gate_closing_detuning(
    *,
    carrier_rabi_frequency: float,
    lamb_dicke_parameter: float,
    loops: int = 1,
) -> float:
    """Return the MS detuning ``δ`` that closes the phase-space loop
    **and** produces a maximally entangling Bell-state rotation.

    .. math::
        \\delta = 2 \\, |\\Omega \\, \\eta| \\, \\sqrt{K}

    where ``K = loops`` is the integer number of loops the motional
    coherent state traces in phase space before returning to the
    origin at ``t_gate = 2π K / δ``.

    Derivation. In the interaction picture, the Magnus expansion of
    the detuned MS Hamiltonian
    (:func:`iontrap_dynamics.hamiltonians.detuned_ms_gate_hamiltonian`)
    has a vanishing first-order term at ``t_gate`` (loop closes) and
    a second-order spin–spin coupling

    .. math::
        U(t_\\text{gate}) = \\exp\\!\\Bigl[ -i \\, \\pi K \\,
          (\\Omega \\eta / \\delta)^2
          \\bigl( I + \\sigma_x^{(0)} \\sigma_x^{(1)} \\bigr) \\Bigr].

    Setting the argument of the ``σ_x σ_x`` exponential to ``π/4``
    (maximally entangling) gives ``π K (Ω η / δ)² = π / 4``, which
    rearranges to the formula above.

    Parameters
    ----------
    carrier_rabi_frequency
        Carrier Rabi frequency Ω, rad·s⁻¹.
    lamb_dicke_parameter
        Lamb–Dicke parameter η, dimensionless. Sign is discarded —
        only ``|Ωη|`` enters the closure condition.
    loops
        Integer number of phase-space loops ``K`` before the gate
        closes. Must be ≥ 1. Larger ``K`` gives slower, more lenient
        gates at the cost of more motional decoherence exposure.

    Returns
    -------
    float
        Gate-closing detuning ``δ`` in rad·s⁻¹. Always positive;
        physical sign is a convention choice (flipping δ flips the
        sign of the Bell-state relative phase but not the fidelity).

    Raises
    ------
    ValueError
        If ``loops < 1``.
    """
    if loops < 1:
        raise ValueError(f"loops must be >= 1; got {loops}")
    return float(2.0 * abs(carrier_rabi_frequency * lamb_dicke_parameter) * np.sqrt(loops))


def ms_gate_closing_time(
    *,
    carrier_rabi_frequency: float,
    lamb_dicke_parameter: float,
    loops: int = 1,
) -> float:
    """Return the MS gate time ``t_gate = 2π K / δ`` for the Bell-state
    closing condition (:func:`ms_gate_closing_detuning`).

    .. math::
        t_\\text{gate} = \\frac{\\pi \\, \\sqrt{K}}{|\\Omega \\, \\eta|}

    which is ``π / |Ωη|`` for the default single-loop gate (``K = 1``).

    Parameters mirror :func:`ms_gate_closing_detuning`. Returns
    seconds. Raises :exc:`ValueError` if ``loops < 1`` or if
    ``|Ωη| == 0`` (would give an infinite gate time).
    """
    if loops < 1:
        raise ValueError(f"loops must be >= 1; got {loops}")
    g = abs(carrier_rabi_frequency * lamb_dicke_parameter)
    if g == 0.0:
        raise ValueError("ms_gate_closing_time is undefined for |Ω η| == 0 (infinite gate time)")
    return float(np.pi * np.sqrt(loops) / g)


def ms_gate_phonon_number(
    *,
    carrier_rabi_frequency: float,
    lamb_dicke_parameters: tuple[float, float],
    spin_eigenvalues: tuple[int, int],
    t: ArrayLike,
) -> NDArray[np.floating]:
    """Return ⟨n̂⟩(t) for the symmetric δ = 0 MS gate starting from a
    σ_x eigenstate times motional vacuum.

    At φ = 0 the MS Hamiltonian of
    :func:`iontrap_dynamics.hamiltonians.ms_gate_hamiltonian` reduces to

    .. math::
        H / \\hbar = \\tfrac{\\Omega}{2}
        \\bigl( \\eta_0 \\sigma_x^{(0)} + \\eta_1 \\sigma_x^{(1)} \\bigr)
        \\otimes \\bigl( a + a^\\dagger \\bigr),

    which commutes with every individual σ_x — so starting from
    ``|ε_0, ε_1⟩ ⊗ |0⟩`` with ``ε_k ∈ {−1, +1}`` (a product σ_x
    eigenstate), the spin sector factors out and the motion evolves
    under the constant Hermitian generator
    ``κ (a + a†)`` where ``κ = (Ω/2)(η₀ ε₀ + η₁ ε₁)``.

    This is the textbook spin-dependent force: the motional state
    becomes a coherent state ``|α(t)⟩`` with
    ``α(t) = −i · κ · t`` and therefore

    .. math::
        \\langle n\\rangle(t) = |\\alpha(t)|^2
        = \\bigl(\\tfrac{\\Omega t}{2}\\bigr)^2
          \\bigl( \\eta_0 \\varepsilon_0 + \\eta_1 \\varepsilon_1 \\bigr)^2.

    ``⟨n̂⟩(t) = 0`` exactly when the two ions' forces cancel
    (e.g. ``ε_0 = +1, ε_1 = −1`` with ``η_0 = η_1`` — the "dark"
    σ_x eigenstate for a stretch-mode drive).

    Parameters
    ----------
    carrier_rabi_frequency
        Carrier Rabi frequency Ω, rad·s⁻¹.
    lamb_dicke_parameters
        ``(η_0, η_1)`` — one per ion in the gate. Signs are preserved
        (CONVENTIONS.md §10) and enter the displacement additively.
    spin_eigenvalues
        ``(ε_0, ε_1)`` — σ_x eigenvalues of the initial spin state.
        Must each be in ``{−1, +1}``.
    t
        Time or array of times, s.

    Returns
    -------
    np.ndarray
        Mean phonon number ``⟨n̂⟩(t)``, same shape as ``t``.
        Non-negative.

    Raises
    ------
    ValueError
        If either entry of ``spin_eigenvalues`` is not ±1.
    """
    eps_0, eps_1 = spin_eigenvalues
    if eps_0 not in (-1, 1) or eps_1 not in (-1, 1):
        raise ValueError(f"spin_eigenvalues must each be -1 or +1; got {spin_eigenvalues}")
    eta_0, eta_1 = lamb_dicke_parameters
    time = np.asarray(t, dtype=np.float64)
    kappa = 0.5 * carrier_rabi_frequency * (eta_0 * eps_0 + eta_1 * eps_1)
    return (kappa * time) ** 2


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


def adiabaticity_parameter(omega_rad_s: float, omega_dot_rad_s2: float) -> float:
    r"""Return the non-adiabaticity figure of merit ``ω̇ / ω²`` (dimensionless).

    CONVENTIONS.md §26.1 / Wittemer 2020. The dynamics is **adiabatic** when
    ``|ω̇/ω²| ≪ 1`` (the state tracks the instantaneous ground state) and
    **non-adiabatic** (squeezing-generating) when ``|ω̇/ω²| ≳ 1``; the 2020
    single-ion quench reaches ``ω̇/ω² ≈ 5``.

    Parameters
    ----------
    omega_rad_s
        Instantaneous mode frequency ``ω`` (rad·s⁻¹); must be non-zero.
    omega_dot_rad_s2
        Its time derivative ``ω̇`` (rad·s⁻²). For a
        :class:`~iontrap_dynamics.waveforms.FrequencyWaveform` at time ``t``,
        ``ω̇ = ω(t)·(d ln ω/dt)(t)``.

    Returns
    -------
    float
        ``ω̇ / ω²``.
    """
    if omega_rad_s == 0.0:
        raise ValueError("adiabaticity_parameter: omega_rad_s must be non-zero.")
    return float(omega_dot_rad_s2 / (omega_rad_s * omega_rad_s))


__all__ = [
    "HBAR",
    "LAMB_DICKE_DEEP_MAX",
    "LAMB_DICKE_INTERMEDIATE_MAX",
    "LambDickeRegime",
    "adiabaticity_parameter",
    "blue_sideband_rabi_frequency",
    "blue_sideband_rabi_frequency_full_ld",
    "carrier_rabi_excited_population",
    "carrier_rabi_sigma_z",
    "coherent_state_mean_n",
    "debye_waller_factor",
    "detuned_rabi_sigma_z",
    "generalized_rabi_frequency",
    "lamb_dicke_confinement",
    "lamb_dicke_parameter",
    "lamb_dicke_regime",
    "ms_gate_closing_detuning",
    "ms_gate_closing_time",
    "ms_gate_phonon_number",
    "red_sideband_rabi_frequency",
    "red_sideband_rabi_frequency_full_ld",
]
