# SPDX-License-Identifier: MIT
"""Common-mode (shared-latent) phase channel for the systematics layer.

WI-4 of WP-01 (dispatch EDD). Application-agnostic: a correlated phase
perturbation shared, to a tunable degree, across several drives (subsystems).
Where :class:`~iontrap_dynamics.systematics.PhaseJitter` draws an *independent*
offset per drive, the common-mode channel draws, per shot, a shared latent plus
a per-subsystem part and mixes them by a correlation parameter:

    offset_i = √c · ξ_shared + √(1 − c) · ξ_i,   ξ_shared, ξ_i ~ N(0, σ_rad²)

with ``c = correlation ∈ [0, 1]``. At ``c = 0`` the offsets are independent (the
channel reduces to per-drive ``PhaseJitter``); at ``c = 1`` a single offset is
shared across all drives, so it cancels in any difference observable
(common-mode rejection). ``σ_rad`` is the marginal per-subsystem standard
deviation at *every* ``c`` (``Var = c·σ² + (1 − c)·σ² = σ²``).

It mirrors the systematics composition pattern (a frozen-dataclass spec plus a
``perturb_*`` helper that materialises perturbed :class:`DriveConfig`\\s via
``dataclasses.replace``); the departure is that it perturbs a *sequence* of
drives with one correlated draw per shot, returning a per-shot tuple of
perturbed drives.

Conventions: CONVENTIONS.md §22 (staged for the shared v0.3 Convention Freeze —
see ``WP/FREEZE-v0.3.md``). Distinct from the v0.2-frozen §18 systematics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..drives import DriveConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class CommonModePhase:
    """Correlated shared-latent phase channel across subsystems.

    Parameters
    ----------
    sigma_rad
        Marginal per-subsystem phase standard deviation in rad. Non-negative;
        ``0.0`` is a valid no-op. This is the marginal std at every
        ``correlation``.
    correlation
        Shared fraction ``c ∈ [0, 1]``. ``0`` → independent per subsystem
        (reduces to :class:`~iontrap_dynamics.systematics.PhaseJitter`); ``1`` →
        a single shared latent across all subsystems (full common mode).
        Default ``1.0``.
    label
        Identifier for downstream aggregation. Default ``"common_mode_phase"``.

    Raises
    ------
    ValueError
        On negative ``sigma_rad`` or ``correlation`` outside ``[0, 1]``.
    """

    sigma_rad: float
    correlation: float = 1.0
    label: str = "common_mode_phase"

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma_rad) or self.sigma_rad < 0.0:
            raise ValueError(
                f"CommonModePhase: sigma_rad must be a finite value >= 0; got {self.sigma_rad}"
            )
        if not np.isfinite(self.correlation) or not 0.0 <= self.correlation <= 1.0:
            raise ValueError(
                f"CommonModePhase: correlation must be a finite value in [0, 1]; "
                f"got {self.correlation}"
            )

    def sample_offsets(
        self,
        *,
        n_subsystems: int,
        shots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw per-(shot, subsystem) phase offsets, shape ``(shots, n_subsystems)``.

        Each shot mixes a shared latent and per-subsystem draws by the
        correlation: ``offset_i = √c · ξ_shared + √(1 − c) · ξ_i``. At ``c = 1``
        the per-subsystem offsets within a shot are identical; at ``c = 0`` they
        are independent.

        Raises
        ------
        ValueError
            If ``n_subsystems < 1`` or ``shots < 1``.
        """
        if n_subsystems < 1:
            raise ValueError(
                f"CommonModePhase.sample_offsets: n_subsystems must be >= 1; got {n_subsystems}"
            )
        if shots < 1:
            raise ValueError(f"CommonModePhase.sample_offsets: shots must be >= 1; got {shots}")
        shared = rng.normal(loc=0.0, scale=self.sigma_rad, size=shots)
        independent = rng.normal(loc=0.0, scale=self.sigma_rad, size=(shots, n_subsystems))
        c = self.correlation
        offsets = np.sqrt(c) * shared[:, None] + np.sqrt(1.0 - c) * independent
        return np.asarray(offsets, dtype=np.float64)


def perturb_common_mode(
    drives: Sequence[DriveConfig],
    spec: CommonModePhase,
    *,
    shots: int,
    seed: int | None = None,
) -> tuple[tuple[DriveConfig, ...], ...]:
    """Materialise ``shots`` correlated-perturbation copies of ``drives``.

    Returns a tuple of length ``shots``; each entry is a tuple of perturbed
    :class:`DriveConfig`\\s — one per input drive, in order — with
    ``phase_rad += offset_i`` for that shot's correlated draw. All other fields
    pass through. At ``spec.correlation = 1`` every drive in a shot shares the
    same offset (so it cancels in difference observables); at ``0`` the offsets
    are independent per drive.

    Bit-reproducibility: fully determined by ``(drives, spec, shots, seed)``.

    Raises
    ------
    ValueError
        If ``drives`` is empty or ``shots < 1``.
    """
    if not drives:
        raise ValueError("perturb_common_mode: drives must be non-empty")
    if shots < 1:
        raise ValueError(f"perturb_common_mode: shots must be >= 1; got {shots}")
    n_subsystems = len(drives)
    rng = np.random.default_rng(seed)
    offsets = spec.sample_offsets(n_subsystems=n_subsystems, shots=shots, rng=rng)
    return tuple(
        tuple(
            replace(drive, phase_rad=float(drive.phase_rad + offsets[shot, i]))
            for i, drive in enumerate(drives)
        )
        for shot in range(shots)
    )


__all__ = [
    "CommonModePhase",
    "perturb_common_mode",
]
