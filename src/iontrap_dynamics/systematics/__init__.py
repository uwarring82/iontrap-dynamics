# SPDX-License-Identifier: MIT
"""Systematics layer (Phase 1, v0.2 track — staged, not frozen).

Models the experimental imperfections named in ``WORKPLAN_v0.3.md``
§5 Phase 1: drifts, jitter, and state-preparation errors. Unlike the
measurement layer (Dispatches H–P) which post-processes a trajectory,
the systematics layer **perturbs the dynamics inputs before the
solver runs** — each shot sees a slightly different Hamiltonian drawn
from a parameter distribution. The user composes by:

1. Constructing a systematics spec (e.g. :class:`RabiJitter`).
2. Calling ``spec.perturb(config, shots=N, seed=...)`` to materialise
   ``N`` perturbed :class:`DriveConfig` (or similar) instances.
3. Running ``solve(...)`` once per perturbed config.
4. Aggregating observables across the resulting list of
   :class:`TrajectoryResult` via normal NumPy reductions.

Dispatch R landed the first stochastic primitive — :class:`RabiJitter`
— plus a thin :func:`perturb_carrier_rabi` helper. Dispatch S adds
:class:`DetuningJitter` and :class:`PhaseJitter` with matching
``perturb_detuning`` and ``perturb_phase`` composition helpers.
Dispatch T adds static :class:`RabiDrift`, :class:`DetuningDrift`,
and :class:`PhaseDrift` — deterministic single-value offsets that
compose via ``apply_*`` helpers returning a single perturbed
:class:`DriveConfig` (no ensemble). Dispatch U closes the track
with state-preparation (SPAM) primitives :class:`SpinPreparationError`
and :class:`ThermalPreparationError`, plus the
:func:`imperfect_spin_ground` and :func:`imperfect_motional_ground`
helpers that produce per-subsystem density matrices for composition
via :func:`iontrap_dynamics.states.compose_density`. With jitter
(stochastic, ensemble), drift (systematic, single-solve), and SPAM
(state-prep) all in place, the ``CONVENTIONS.md`` §18 section is a
complete read-through and freezes at the v0.2 release.
"""

from __future__ import annotations

from .drift import (
    DetuningDrift,
    PhaseDrift,
    RabiDrift,
    apply_detuning_drift,
    apply_phase_drift,
    apply_rabi_drift,
)
from .jitter import (
    DetuningJitter,
    PhaseJitter,
    RabiJitter,
    perturb_carrier_rabi,
    perturb_detuning,
    perturb_phase,
)
from .spam import (
    SpinPreparationError,
    ThermalPreparationError,
    imperfect_motional_ground,
    imperfect_spin_ground,
)

__all__ = [
    "DetuningDrift",
    "DetuningJitter",
    "PhaseDrift",
    "PhaseJitter",
    "RabiDrift",
    "RabiJitter",
    "SpinPreparationError",
    "ThermalPreparationError",
    "apply_detuning_drift",
    "apply_phase_drift",
    "apply_rabi_drift",
    "imperfect_motional_ground",
    "imperfect_spin_ground",
    "perturb_carrier_rabi",
    "perturb_detuning",
    "perturb_phase",
]
