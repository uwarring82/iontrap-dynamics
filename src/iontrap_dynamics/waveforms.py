# SPDX-License-Identifier: MIT
"""Time-dependent trap-frequency waveforms for the squeezing generator.

CONVENTIONS.md §26.1 / WP-05 R5. The non-adiabatic squeezing Hamiltonian
(:func:`iontrap_dynamics.hamiltonians.nonadiabatic_squeezing_hamiltonian`)
takes a :class:`FrequencyWaveform` exposing **both** ``ω(t)`` and the
**analytic** ``d ln ω/dt`` (plus optional JAX mirrors) — *not* a bare callable,
because the squeezing coefficient is the log-derivative and a black-box callable
cannot supply it safely across a sharp quench (**never** runtime numerical
differentiation). The object validates finite, strictly-positive ``ω`` and
finite ``d ln ω/dt`` at supplied sample times (build-time spot-checks);
paired-callable mutual consistency is the caller's responsibility.

Named shapes (:func:`gaussian_quench`, :func:`sinusoidal_modulation`,
:func:`smooth_ramp`) supply their derivatives **analytically**.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .exceptions import ConventionError

if TYPE_CHECKING:
    from jax import Array


def _maybe_jnp() -> Any:
    """Return :mod:`jax.numpy` if the ``[jax]`` extras are importable, else ``None``."""
    try:
        import jax.numpy as jnp
    except ImportError:
        return None
    return jnp


@dataclass(frozen=True)
class FrequencyWaveform:
    """A trap-frequency waveform ``ω(t)`` with its analytic log-derivative.

    Parameters
    ----------
    omega
        ``t -> float``: the instantaneous mode frequency ``ω(t)`` in rad·s⁻¹.
        Must be finite and **strictly positive** everywhere it is evaluated.
    d_ln_omega_dt
        ``t -> float``: the **analytic** log-derivative ``d ln ω/dt`` in s⁻¹.
        This is the squeezing-generator coefficient carrier (§26.1); it is
        supplied analytically, never numerically differentiated at run time.
    omega_jax, d_ln_omega_dt_jax
        Optional JAX-traceable mirrors (``jax.numpy`` primitives, no Python
        branching on the traced ``t``). Required only for ``backend="jax"``.
    """

    omega: Callable[[float], float]
    d_ln_omega_dt: Callable[[float], float]
    omega_jax: Callable[[float], Array] | None = None
    d_ln_omega_dt_jax: Callable[[float], Array] | None = None

    def validate_at(self, times: Iterable[float]) -> None:
        """Spot-check finite, strictly-positive ``ω`` and finite ``d ln ω/dt``.

        A build-time diagnostic over the supplied sample times (typically the
        solver's ``times`` grid or a subsample) — **not** a continuous guarantee
        and **not** numerical differentiation. Raises :class:`ConventionError`
        on the first violation.
        """
        for t in times:
            w = float(self.omega(t))
            if not math.isfinite(w) or w <= 0.0:
                raise ConventionError(
                    f"FrequencyWaveform: omega({t}) = {w} must be finite and strictly "
                    "positive (CONVENTIONS.md §26.1; a vanishing/negative trap frequency "
                    "is unphysical)."
                )
            d = float(self.d_ln_omega_dt(t))
            if not math.isfinite(d):
                raise ConventionError(
                    f"FrequencyWaveform: d_ln_omega_dt({t}) = {d} must be finite "
                    "(CONVENTIONS.md §26.1)."
                )


def sinusoidal_modulation(
    *, omega_ini: float, mod_amplitude: float, mod_frequency: float
) -> FrequencyWaveform:
    """Parametric modulation ``ω(t) = ω_ini + δω·sin(ω_mod·t)`` (§26.1, quench-ii).

    Resonant parametric amplification uses ``ω_mod = 2·ω_ini``. Requires
    ``|δω| < ω_ini`` so ``ω(t)`` stays strictly positive.
    """
    if abs(mod_amplitude) >= omega_ini:
        raise ConventionError(
            f"sinusoidal_modulation: |mod_amplitude| ({mod_amplitude}) must be < "
            f"omega_ini ({omega_ini}) to keep omega(t) strictly positive."
        )

    def omega(t: float) -> float:
        return omega_ini + mod_amplitude * math.sin(mod_frequency * t)

    def d_ln_omega_dt(t: float) -> float:
        w = omega_ini + mod_amplitude * math.sin(mod_frequency * t)
        return mod_amplitude * mod_frequency * math.cos(mod_frequency * t) / w

    jnp = _maybe_jnp()
    if jnp is None:
        return FrequencyWaveform(omega, d_ln_omega_dt)

    def omega_jax(t: float) -> Any:
        return omega_ini + mod_amplitude * jnp.sin(mod_frequency * t)

    def d_ln_jax(t: float) -> Any:
        w = omega_ini + mod_amplitude * jnp.sin(mod_frequency * t)
        return mod_amplitude * mod_frequency * jnp.cos(mod_frequency * t) / w

    return FrequencyWaveform(omega, d_ln_omega_dt, omega_jax, d_ln_jax)


def gaussian_quench(
    *, omega_ini: float, amplitude: float, width_s: float, center_s: float
) -> FrequencyWaveform:
    """Gaussian quench ``ω(t) = ω_ini·(1 + A·exp(−½((t−t₀)/δτ)²))`` (§26.1, quench-i).

    A fast, strong Gaussian excursion of the trap frequency (width ``δτ``,
    fractional amplitude ``A``, centred at ``t₀``). Requires ``A > −1`` so
    ``ω(t)`` stays strictly positive.
    """
    if amplitude <= -1.0:
        raise ConventionError(
            f"gaussian_quench: amplitude ({amplitude}) must be > -1 to keep omega(t) "
            "strictly positive."
        )
    if width_s <= 0.0:
        raise ConventionError(f"gaussian_quench: width_s ({width_s}) must be > 0.")

    def omega(t: float) -> float:
        return omega_ini * (1.0 + amplitude * math.exp(-0.5 * ((t - center_s) / width_s) ** 2))

    def d_ln_omega_dt(t: float) -> float:
        g = math.exp(-0.5 * ((t - center_s) / width_s) ** 2)
        gprime = -(t - center_s) / (width_s * width_s) * g
        return amplitude * gprime / (1.0 + amplitude * g)

    jnp = _maybe_jnp()
    if jnp is None:
        return FrequencyWaveform(omega, d_ln_omega_dt)

    def omega_jax(t: float) -> Any:
        return omega_ini * (1.0 + amplitude * jnp.exp(-0.5 * ((t - center_s) / width_s) ** 2))

    def d_ln_jax(t: float) -> Any:
        g = jnp.exp(-0.5 * ((t - center_s) / width_s) ** 2)
        gprime = -(t - center_s) / (width_s * width_s) * g
        return amplitude * gprime / (1.0 + amplitude * g)

    return FrequencyWaveform(omega, d_ln_omega_dt, omega_jax, d_ln_jax)


def smooth_ramp(
    *, omega_i: float, omega_f: float, center_s: float, width_s: float
) -> FrequencyWaveform:
    """Smooth log-linear ``tanh`` ramp from ``ω_i`` to ``ω_f`` (a squeeze kick).

    ``ln ω(t) = ½(ln ω_i + ln ω_f) + ½ ln(ω_f/ω_i)·tanh((t−t₀)/δτ)`` — a
    monotone crossover whose **total** log-swing is exactly ``ln(ω_f/ω_i)``
    regardless of the width. Narrowing ``δτ`` → the sudden (squeeze-kick) limit;
    widening ``δτ`` → the adiabatic limit (§26.1 conventions-test gate). Stays
    strictly positive by construction (it interpolates ``ln ω``).
    """
    if omega_i <= 0.0 or omega_f <= 0.0:
        raise ConventionError(f"smooth_ramp: omega_i ({omega_i}), omega_f ({omega_f}) must be > 0.")
    if width_s <= 0.0:
        raise ConventionError(f"smooth_ramp: width_s ({width_s}) must be > 0.")
    ln_mid = 0.5 * (math.log(omega_i) + math.log(omega_f))
    ln_half = 0.5 * math.log(omega_f / omega_i)

    def omega(t: float) -> float:
        return math.exp(ln_mid + ln_half * math.tanh((t - center_s) / width_s))

    def d_ln_omega_dt(t: float) -> float:
        sech2 = 1.0 / math.cosh((t - center_s) / width_s) ** 2
        return ln_half * sech2 / width_s

    jnp = _maybe_jnp()
    if jnp is None:
        return FrequencyWaveform(omega, d_ln_omega_dt)

    def omega_jax(t: float) -> Any:
        return jnp.exp(ln_mid + ln_half * jnp.tanh((t - center_s) / width_s))

    def d_ln_jax(t: float) -> Any:
        sech2 = 1.0 / jnp.cosh((t - center_s) / width_s) ** 2
        return ln_half * sech2 / width_s

    return FrequencyWaveform(omega, d_ln_omega_dt, omega_jax, d_ln_jax)


__all__ = [
    "FrequencyWaveform",
    "gaussian_quench",
    "sinusoidal_modulation",
    "smooth_ramp",
]
