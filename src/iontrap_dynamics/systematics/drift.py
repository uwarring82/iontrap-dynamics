# SPDX-License-Identifier: MIT
"""Static parameter drifts for the systematics layer.

Drift models *systematic* bias shifts — a parameter sits at ``p₀ + Δ``
for the full run, not ``p₀``. Physical sources: calibration errors
(AOM power miscalibration, laser-frequency setpoint offset), slow
drifts that haven't been corrected, intentional mis-set parameters
used in a perturbation study.

Distinct from jitter (stochastic, shot-to-shot, zero-mean) in three
ways:

1. **Deterministic.** A drift spec encodes a fixed offset; two runs
   with the same drift produce bit-identical trajectories (given the
   same solver parameters).
2. **Single-solve composition.** No ensemble over shots — just one
   perturbed :class:`DriveConfig`, one :func:`solve` call.
3. **Scan-friendly.** Drifts compose naturally in a Python ``for``
   loop over a range of offsets, producing a parameter scan. No
   dedicated scan helper is needed — the pattern is ``[apply_*(drive,
   Drift(delta=d)) for d in deltas]``.

Dispatch T ships three primitives parallel to the jitter set (§18.3):

- :class:`RabiDrift` — multiplicative relative offset on ``Ω``.
- :class:`DetuningDrift` — additive offset on ``δ`` in rad·s⁻¹.
- :class:`PhaseDrift` — additive offset on ``φ`` in rad.

Dispatch MCG adds the motional complement:

- :class:`ModeFrequencyDrift` — multiplicative relative offset on the mode
  frequency ``ω_m``. Acts on :class:`~iontrap_dynamics.modes.ModeConfig`
  (not the drive); the Lamb–Dicke parameter ``η ∝ ω_m^{−1/2}`` follows.

Each has a matching ``apply_*`` helper that returns a single perturbed
config via :func:`dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..drives import DriveConfig
from ..modes import ModeConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class RabiDrift:
    """Systematic multiplicative offset on carrier Rabi frequency.

    Applies ``Ω → Ω · (1 + delta)`` once, with no shot-to-shot
    variation. Use for calibration-error studies (e.g. "what if my
    π-pulse is tuned 3 % below optimum?") or for scanning a range of
    Rabi-amplitude miscalibrations.

    Parameters
    ----------
    delta
        Relative offset. Dimensionless. Can be negative
        (below-nominal calibration) or positive (above-nominal).
        No sign restriction; unlike :class:`RabiJitter`'s stochastic
        ``σ ≥ 0``, drift is a deterministic shift.
    label
        Identifier used by downstream aggregation code. Defaults to
        ``"rabi_drift"``.
    """

    delta: float
    label: str = "rabi_drift"


@dataclass(frozen=True, slots=True, kw_only=True)
class DetuningDrift:
    """Systematic additive offset on laser-atom detuning.

    Applies ``δ → δ + delta_rad_s`` once. Use for laser-frequency
    calibration-error studies or for computing a detuning scan (e.g.
    to visualise the Lorentzian profile of a sideband).

    Parameters
    ----------
    delta_rad_s
        Additive offset in rad·s⁻¹. Can be negative (red-shifted
        mis-calibration) or positive (blue-shifted).
    label
        Identifier used by downstream aggregation code. Defaults to
        ``"detuning_drift"``.
    """

    delta_rad_s: float
    label: str = "detuning_drift"


@dataclass(frozen=True, slots=True, kw_only=True)
class PhaseDrift:
    """Systematic additive offset on drive phase.

    Applies ``φ → φ + delta_rad`` once. Use for optical-path-length
    drift studies or for phase-scan protocols (e.g. parity scans
    sweeping the analysis-pulse phase).

    Parameters
    ----------
    delta_rad
        Additive offset in rad. Not wrapped — ``DriveConfig``
        accepts any real phase and builders apply ``exp(i φ)``.
    label
        Identifier used by downstream aggregation code. Defaults to
        ``"phase_drift"``.
    """

    delta_rad: float
    label: str = "phase_drift"


@dataclass(frozen=True, slots=True, kw_only=True)
class ModeFrequencyDrift:
    """Systematic multiplicative offset on a motional mode frequency.

    Applies ``ω_m → ω_m · (1 + delta)`` once to a
    :class:`~iontrap_dynamics.modes.ModeConfig`, with no shot-to-shot
    variation. Models slow trap-frequency drift (RF-amplitude or DC-voltage
    setpoint drift), which is conventionally a *fractional* quantity — hence
    the multiplicative, dimensionless ``delta``, parallel to :class:`RabiDrift`
    (``ω_m`` is a positive magnitude with no meaningful zero, like ``Ω``; it is
    not an offset-from-reference like a detuning).

    Because the Lamb–Dicke parameter scales as ``η ∝ ω_m^{−1/2}``
    (CONVENTIONS.md §10), a drift ``delta`` rescales η to
    ``η → η / √(1 + delta)`` — and only η's *magnitude*: its sign
    (``k⃗·b⃗``) is unchanged. Note the response is sub-linear: a fractional
    frequency drift ``delta`` produces a fractional η change of
    ``(1 + delta)^{−1/2} − 1 ≈ −delta/2``, **not** ``−delta`` — so ``delta``
    is the trap-frequency relative drift, not the η relative drift.

    Parameters
    ----------
    delta
        Relative offset on ``ω_m``. Dimensionless, signed (below- or
        above-nominal). ``delta ≤ −1`` would drive ``ω_m ≤ 0`` and is
        rejected by :class:`~iontrap_dynamics.modes.ModeConfig` with a
        :class:`~iontrap_dynamics.exceptions.ConventionError` — unlike
        :class:`RabiDrift`, a vanishing trap frequency is unphysical and has
        no phase-flip escape, so scans must stay strictly above ``−1``.
    label
        Identifier used by downstream aggregation code. Defaults to
        ``"mode_frequency_drift"``.
    """

    delta: float
    label: str = "mode_frequency_drift"


def apply_rabi_drift(drive: DriveConfig, drift: RabiDrift) -> DriveConfig:
    """Return a new :class:`DriveConfig` with ``Ω → Ω · (1 + delta)``.

    Raises :class:`iontrap_dynamics.exceptions.ConventionError`
    (via the :class:`DriveConfig` post-init) if the resulting Rabi
    frequency is non-positive — ``DriveConfig`` invariant requires
    strictly positive ``carrier_rabi_frequency_rad_s`` (§3: sign is
    absorbed into phase). Callers sweeping ``delta`` across sign
    flips must either keep ``delta > −1`` or flip ``phase_rad`` by
    ``π`` themselves.
    """
    return replace(
        drive,
        carrier_rabi_frequency_rad_s=float(
            drive.carrier_rabi_frequency_rad_s * (1.0 + drift.delta)
        ),
    )


def apply_detuning_drift(drive: DriveConfig, drift: DetuningDrift) -> DriveConfig:
    """Return a new :class:`DriveConfig` with ``δ → δ + delta_rad_s``."""
    return replace(
        drive,
        detuning_rad_s=float(drive.detuning_rad_s + drift.delta_rad_s),
    )


def apply_phase_drift(drive: DriveConfig, drift: PhaseDrift) -> DriveConfig:
    """Return a new :class:`DriveConfig` with ``φ → φ + delta_rad``.

    Phase is not wrapped — values outside ``[−π, π]`` are preserved
    verbatim. Builders apply ``exp(i φ)`` so any real phase is legal.
    """
    return replace(
        drive,
        phase_rad=float(drive.phase_rad + drift.delta_rad),
    )


def apply_mode_frequency_drift(mode: ModeConfig, drift: ModeFrequencyDrift) -> ModeConfig:
    """Return a new :class:`~iontrap_dynamics.modes.ModeConfig` with ``ω_m → ω_m · (1 + delta)``.

    Only ``frequency_rad_s`` changes; ``label`` and ``eigenvector_per_ion``
    pass through unchanged (the mode's spatial profile does not drift). The
    resulting ``ModeConfig`` is fully re-validated by its post-init, so
    ``delta ≤ −1`` (which would make ``ω_m ≤ 0``) raises
    :class:`~iontrap_dynamics.exceptions.ConventionError`. The Lamb–Dicke
    parameter ``η ∝ ω_m^{−1/2}`` rescales to ``η / √(1 + delta)`` downstream.
    """
    return replace(
        mode,
        frequency_rad_s=float(mode.frequency_rad_s * (1.0 + drift.delta)),
    )


__all__ = [
    "DetuningDrift",
    "ModeFrequencyDrift",
    "PhaseDrift",
    "RabiDrift",
    "apply_detuning_drift",
    "apply_mode_frequency_drift",
    "apply_phase_drift",
    "apply_rabi_drift",
]
