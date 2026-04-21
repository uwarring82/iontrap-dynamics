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

Each has a matching ``apply_*`` helper that returns a single perturbed
:class:`DriveConfig` via :func:`dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..drives import DriveConfig


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


__all__ = [
    "DetuningDrift",
    "PhaseDrift",
    "RabiDrift",
    "apply_detuning_drift",
    "apply_phase_drift",
    "apply_rabi_drift",
]
