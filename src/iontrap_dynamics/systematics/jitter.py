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

Dispatch R ships :class:`RabiJitter` — multiplicative Gaussian noise
on carrier Rabi frequency, the most common real-experiment imperfection
and the baseline against which later jitter primitives (detuning,
phase, trap-frequency) will be compared.
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


__all__ = [
    "RabiJitter",
    "perturb_carrier_rabi",
]
