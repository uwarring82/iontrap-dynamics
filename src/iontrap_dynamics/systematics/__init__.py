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

Dispatch R lands the first stochastic primitive — :class:`RabiJitter`
— plus a thin :func:`perturb_carrier_rabi` helper that builds the
perturbed :class:`DriveConfig` ensemble. Detuning / phase jitter,
parameter drifts (offsets), and SPAM (state-prep) errors follow in
Dispatches S–U. The ``CONVENTIONS.md`` §18 section is opened with
Dispatch R as staged rules and will freeze at the close of the
systematics track.
"""

from __future__ import annotations

from .jitter import RabiJitter, perturb_carrier_rabi

__all__ = [
    "RabiJitter",
    "perturb_carrier_rabi",
]
