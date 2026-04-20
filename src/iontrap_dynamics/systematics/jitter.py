# SPDX-License-Identifier: MIT
"""Shot-to-shot parameter jitter for the systematics layer.

Jitter models *stochastic* shot-to-shot fluctuations — laser intensity
noise, acousto-optic modulator amplitude jitter, magnetic-field
instability in the readout window. Mathematically these are
*multiplicative* or *additive* perturbations on drive parameters,
independent from one shot to the next, drawn from a distribution
(typically Gaussian). Over many shots the ensemble mean of an
observable trajectory dephases from the noise-free signal — the
classic inhomogeneous-dephasing signature.

Dispatch R shipped :class:`RabiJitter` — multiplicative Gaussian noise
on carrier Rabi frequency, the baseline against which later primitives
are compared. Dispatch S adds :class:`DetuningJitter` (additive noise
on ``detuning_rad_s``) and :class:`PhaseJitter` (additive noise on
``phase_rad``). All three use the same composition pattern: sample
per-shot perturbations, materialise a tuple of perturbed
:class:`DriveConfig`\\s via ``perturb_*`` helpers, run solve() per
entry, aggregate with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..drives import DriveConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class RabiJitter:
    """Shot-to-shot multiplicative Gaussian jitter on carrier Rabi frequency.

    Each shot independently samples a perturbation ``ε ~ Normal(0, σ)``
    and scales the carrier Rabi frequency by ``(1 + ε)``. Negative
    realisations of ``(1 + ε)`` are legal — the drive can in principle
    flip sign at large jitter amplitudes — but the jitter ``σ`` should
    normally be ``≪ 1`` (e.g. ``σ ≈ 0.01–0.05`` for a well-stabilised
    laser), in which case the ``Normal``-sampled ``(1 + ε)`` stays
    positive with overwhelming probability.

    Parameters
    ----------
    sigma
        Relative jitter amplitude ``σ``. Dimensionless, non-negative.
        ``σ = 0`` is a valid no-op used for pipeline tests.
    label
        Identifier used by downstream aggregation code when a run
        carries multiple jitter sources. Defaults to ``"rabi_jitter"``.

    Raises
    ------
    ValueError
        On negative ``sigma``.
    """

    sigma: float
    label: str = "rabi_jitter"

    def __post_init__(self) -> None:
        if self.sigma < 0.0:
            raise ValueError(f"RabiJitter: sigma must be >= 0; got {self.sigma}")

    def sample_multipliers(
        self,
        *,
        shots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw ``shots`` independent ``(1 + ε)`` multipliers.

        Parameters
        ----------
        shots
            Number of per-shot multipliers to sample. ``>= 1``.
        rng
            NumPy random generator supplying the stochastic bits.

        Returns
        -------
        np.ndarray
            1-D ``float64`` array of length ``shots`` — the per-shot
            ``(1 + ε)`` multipliers. Mean → 1, std → ``sigma`` in
            the large-shot limit.

        Raises
        ------
        ValueError
            If ``shots < 1``.
        """
        if shots < 1:
            raise ValueError(f"RabiJitter.sample_multipliers: shots must be >= 1; got {shots}")
        return 1.0 + rng.normal(loc=0.0, scale=self.sigma, size=shots).astype(np.float64)


def perturb_carrier_rabi(
    drive: DriveConfig,
    jitter: RabiJitter,
    *,
    shots: int,
    seed: int | None = None,
) -> tuple[DriveConfig, ...]:
    """Materialise a tuple of ``shots`` jittered :class:`DriveConfig`\\s.

    Builds ``shots`` copies of ``drive`` with the carrier Rabi
    frequency scaled by independently-sampled ``(1 + ε)`` factors.
    All other :class:`DriveConfig` fields pass through untouched;
    user-facing composition is then: loop over the returned tuple,
    run ``solve()`` per perturbed drive, and stack the resulting
    observable trajectories.

    Parameters
    ----------
    drive
        Base :class:`DriveConfig` whose carrier Rabi frequency will be
        jittered. All other fields copy into every perturbed output.
    jitter
        :class:`RabiJitter` spec — controls the multiplier distribution.
    shots
        Number of perturbed drives to build. ``>= 1``.
    seed
        Optional integer seed for :func:`numpy.random.default_rng`.
        When supplied, the output tuple is bit-reproducible given
        ``(drive, jitter, shots, seed)``.

    Returns
    -------
    tuple[DriveConfig, ...]
        Length ``shots``; each entry is a :class:`DriveConfig` whose
        ``carrier_rabi_frequency_rad_s`` is ``drive_Ω · (1 + ε_i)``
        for independent ``ε_i ~ Normal(0, sigma)``.

    Notes
    -----
    The canonical aggregation pattern::

        drives = perturb_carrier_rabi(drive, jitter, shots=N, seed=0)
        results = [solve(..., drive=d, ...) for d in drives]
        expectations_stack = np.stack(
            [r.expectations["sigma_z_0"] for r in results]
        )
        ensemble_mean = expectations_stack.mean(axis=0)
        ensemble_std  = expectations_stack.std(axis=0)

    The shape is ``(shots, n_times)``; ``mean`` and ``std`` collapse
    the shot axis, giving the inhomogeneously-dephased signal and its
    finite-ensemble error bar.
    """
    if shots < 1:
        raise ValueError(f"perturb_carrier_rabi: shots must be >= 1; got {shots}")
    rng = np.random.default_rng(seed)
    multipliers = jitter.sample_multipliers(shots=shots, rng=rng)
    base_rabi = drive.carrier_rabi_frequency_rad_s
    return tuple(
        replace(drive, carrier_rabi_frequency_rad_s=float(base_rabi * m)) for m in multipliers
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class DetuningJitter:
    """Shot-to-shot additive Gaussian jitter on laser-atom detuning.

    Each shot independently samples an offset ``Δδ ~ Normal(0, σ)``
    and adds it to the drive's ``detuning_rad_s``. Physical sources:
    laser-frequency jitter over the detection window, magnetic-field
    drift translating into Zeeman-shift noise on the atomic
    resonance, or AOM / EOM frequency instability.

    Parameters
    ----------
    sigma_rad_s
        Detuning jitter amplitude in rad·s⁻¹. Non-negative;
        ``0.0`` is a valid no-op used for pipeline tests. Typical
        experimental values are ``σ / 2π ≈ 10 Hz – 1 kHz`` depending
        on the stabilisation.
    label
        Identifier used by downstream aggregation code when a run
        carries multiple jitter sources. Defaults to
        ``"detuning_jitter"``.

    Raises
    ------
    ValueError
        On negative ``sigma_rad_s``.
    """

    sigma_rad_s: float
    label: str = "detuning_jitter"

    def __post_init__(self) -> None:
        if self.sigma_rad_s < 0.0:
            raise ValueError(f"DetuningJitter: sigma_rad_s must be >= 0; got {self.sigma_rad_s}")

    def sample_offsets(
        self,
        *,
        shots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw ``shots`` independent additive offsets in rad·s⁻¹."""
        if shots < 1:
            raise ValueError(f"DetuningJitter.sample_offsets: shots must be >= 1; got {shots}")
        return rng.normal(loc=0.0, scale=self.sigma_rad_s, size=shots).astype(np.float64)


@dataclass(frozen=True, slots=True, kw_only=True)
class PhaseJitter:
    """Shot-to-shot additive Gaussian jitter on drive phase.

    Each shot independently samples an offset ``Δφ ~ Normal(0, σ)``
    and adds it to the drive's ``phase_rad``. Physical sources:
    optical-phase noise from the laser or its delivery fibre, AOM
    RF-phase jitter, vibration-induced path-length fluctuations.

    Over many shots, phase jitter decoheres any protocol that relies
    on a fixed phase reference between pulses (Ramsey-style
    sequences, MS gates with calibration pulses). Single-pulse
    Rabi flopping is insensitive to a constant phase offset, so
    :class:`PhaseJitter` only becomes visible on multi-pulse
    sequences or interferometric observables.

    Parameters
    ----------
    sigma_rad
        Phase jitter amplitude in rad. Non-negative. Values of
        ``σ ≈ 0.01 – 0.1`` correspond to state-of-the-art to modest
        phase stability on optical paths.
    label
        Identifier used by downstream aggregation code. Defaults to
        ``"phase_jitter"``.

    Raises
    ------
    ValueError
        On negative ``sigma_rad``.
    """

    sigma_rad: float
    label: str = "phase_jitter"

    def __post_init__(self) -> None:
        if self.sigma_rad < 0.0:
            raise ValueError(f"PhaseJitter: sigma_rad must be >= 0; got {self.sigma_rad}")

    def sample_offsets(
        self,
        *,
        shots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw ``shots`` independent additive phase offsets in rad."""
        if shots < 1:
            raise ValueError(f"PhaseJitter.sample_offsets: shots must be >= 1; got {shots}")
        return rng.normal(loc=0.0, scale=self.sigma_rad, size=shots).astype(np.float64)


def perturb_detuning(
    drive: DriveConfig,
    jitter: DetuningJitter,
    *,
    shots: int,
    seed: int | None = None,
) -> tuple[DriveConfig, ...]:
    """Return ``shots`` :class:`DriveConfig`\\s with jittered detuning.

    Each entry has ``detuning_rad_s = drive.detuning_rad_s + Δδ_i``
    for independent ``Δδ_i ~ Normal(0, jitter.sigma_rad_s)``. Other
    fields pass through.

    Bit-reproducibility: fully determined by
    ``(drive, jitter, shots, seed)``.
    """
    if shots < 1:
        raise ValueError(f"perturb_detuning: shots must be >= 1; got {shots}")
    rng = np.random.default_rng(seed)
    offsets = jitter.sample_offsets(shots=shots, rng=rng)
    base = drive.detuning_rad_s
    return tuple(replace(drive, detuning_rad_s=float(base + o)) for o in offsets)


def perturb_phase(
    drive: DriveConfig,
    jitter: PhaseJitter,
    *,
    shots: int,
    seed: int | None = None,
) -> tuple[DriveConfig, ...]:
    """Return ``shots`` :class:`DriveConfig`\\s with jittered phase.

    Each entry has ``phase_rad = drive.phase_rad + Δφ_i`` for
    independent ``Δφ_i ~ Normal(0, jitter.sigma_rad)``. Other fields
    pass through. Phase is not wrapped — builders apply ``exp(i φ)``
    so values outside ``[−π, π]`` are legal.

    Bit-reproducibility: fully determined by
    ``(drive, jitter, shots, seed)``.
    """
    if shots < 1:
        raise ValueError(f"perturb_phase: shots must be >= 1; got {shots}")
    rng = np.random.default_rng(seed)
    offsets = jitter.sample_offsets(shots=shots, rng=rng)
    base = drive.phase_rad
    return tuple(replace(drive, phase_rad=float(base + o)) for o in offsets)


__all__ = [
    "DetuningJitter",
    "PhaseJitter",
    "RabiJitter",
    "perturb_carrier_rabi",
    "perturb_detuning",
    "perturb_phase",
]
