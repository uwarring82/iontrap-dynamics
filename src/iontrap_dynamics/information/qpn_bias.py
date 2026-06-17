# SPDX-License-Identifier: MIT
"""Quantum-projection-noise (QPN) bias of the BLP non-Markovianity measure.

Phase A / WI-3 of the non-Markovianity + spectral-density task card
(``task cards/TC-non-markovianity-spectral-density.md``; WP-04 candidate).
**Application-agnostic** and **observable-only**: it simulates the standard
projective binomial shot model (CONVENTIONS §17 ``BinomialChannel`` semantics)
and *reports* an empirical bias — it defines no debiasing convention and adds no
new symbol (the MCF probe-QFI precedent).

Why a bias exists at all
------------------------

The BLP measure ``𝒩 = Σ [D(tᵢ) − D(tᵢ₋₁)]_{>0}`` sums only the **positive**
increments of the trace distance. Finite-shot spin-state tomography reconstructs
each ``ρ_S(t)`` from ``⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩`` estimated from ``shots`` projective
measurements per axis, so each estimate carries ``∝ 1/√shots`` quantum projection
noise. Because only positive increments are kept, that noise does **not** average
to zero — it *rectifies* into a systematic, positive bias. Wittemer et al.,
PRA 97, 020102(R) (2018), make this their central result: the bias decomposes as

    ℬ_total = 𝒩(γ, r) − 𝒩_true,
    ℬ_QPN     = 𝒩(γ, r) − 𝒩(γ, ∞)   (positive; → 0 as shots r → ∞ at fixed γ),
    ℬ_sampling = 𝒩(γ, ∞) − 𝒩_true     (≤ 0 for a coarse time grid; r-independent),

with ``𝒩_true = lim_{γ, r → ∞} 𝒩`` (zero noise **and** infinite sampling) and
``γ`` the sampling rate (here a time-grid stride).

For a single qubit the trace distance is the half-Euclidean Bloch distance
``D = ½‖r⃗₁ − r⃗₂‖``; the noise-free measures use the physical
:func:`~iontrap_dynamics.information.distinguishability.blp_non_markovianity`,
while the *sampled* estimator is accumulated guard-free — a finite-shot Bloch
vector can leave the unit ball, so a sampled ``D`` may exceed 1, and that
excursion is exactly the QPN inflation that produces ``ℬ_QPN``.

This Phase-A estimator models the dominant projective binomial QPN only; detector
infidelity (the §17 ``DetectorConfig`` TP/TN map) and a measurement-layer
``SpinTomographyReadout`` wrapper are deferred (task card §9). An analytic
sampling-covariance propagation is recorded as a deferred alternative in
``WP/LOGBOOK.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .distinguishability import blp_non_markovianity

_BLOCH_TOLERANCE: float = 1e-9
"""Slack on the physical ``|⟨σ_a⟩| ≤ 1`` bound for the input Bloch trajectories."""


@dataclass(frozen=True, slots=True, kw_only=True)
class QpnBiasResult:
    """Outcome of a QPN-bias estimate for the BLP measure.

    All measures are the accumulated BLP non-Markovianity ``𝒩`` (dimensionless).
    """

    shots: int
    repeats: int
    sampling_stride: int
    measure_true: float
    """``𝒩_true`` — noise-free, full time grid (``lim_{γ, r → ∞}``)."""
    measure_infinite_shots: float
    """``𝒩(γ, ∞)`` — noise-free at the working sampling stride ``γ``."""
    measure_sampled_mean: float
    """Mean of ``𝒩(γ, r)`` over ``repeats`` independent finite-shot realisations."""
    measure_sampled_std: float
    """Standard deviation of ``𝒩(γ, r)`` across realisations."""
    qpn_bias: float
    """``ℬ_QPN = 𝒩(γ, r) − 𝒩(γ, ∞)`` — positive; vanishes as ``shots → ∞``."""
    sampling_bias: float
    """``ℬ_sampling = 𝒩(γ, ∞) − 𝒩_true`` — ``≤ 0`` for a coarse grid; r-independent."""
    total_bias: float
    """``ℬ_total = 𝒩(γ, r) − 𝒩_true``."""
    total_bias_ci95: tuple[float, float]
    """95% percentile interval of ``𝒩(γ, r) − 𝒩_true`` across realisations."""


def _qubit_trace_distance(bloch_1: NDArray[np.float64], bloch_2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Half-Euclidean Bloch distance ``D = ½‖r⃗₁ − r⃗₂‖`` per time index."""
    return np.asarray(0.5 * np.linalg.norm(bloch_1 - bloch_2, axis=-1), dtype=np.float64)


def _positive_increment_sum(distances: NDArray[np.float64]) -> NDArray[np.float64]:
    """Guard-free ``Σ [Dᵢ − Dᵢ₋₁]_{>0}`` along the last axis (the noisy estimator)."""
    increments = np.diff(distances, axis=-1)
    return np.asarray(np.sum(np.where(increments > 0.0, increments, 0.0), axis=-1), dtype=np.float64)


def non_markovianity_qpn_bias(
    bloch_1: NDArray[np.float64],
    bloch_2: NDArray[np.float64],
    *,
    shots: int,
    repeats: int = 512,
    sampling_stride: int = 1,
    seed: int | None = None,
) -> QpnBiasResult:
    """Estimate the QPN bias of the BLP measure for two reduced-qubit trajectories.

    Parameters
    ----------
    bloch_1, bloch_2
        The two **noise-free** reduced-qubit Bloch trajectories, shape ``(T, 3)``
        — the rows are ``⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩`` at each of ``T`` times, for the two
        initial conditions whose distinguishability is tracked. Components must be
        physical (``|⟨σ_a⟩| ≤ 1``).
    shots
        Projective measurements **per Pauli axis per time** (``r`` in the paper).
    repeats
        Independent finite-shot realisations used to estimate the mean bias and CI.
    sampling_stride
        Time-grid stride modelling a finite sampling rate ``γ`` (``1`` = the full
        grid). A coarser stride misses fast features of ``D(t)`` and yields the
        negative ``sampling_bias``.
    seed
        Seed for the projective sampler (reproducible when set).

    Returns
    -------
    QpnBiasResult
        The noise-free measures, the sampled-measure statistics, and the
        QPN / sampling / total bias decomposition with a 95% interval.

    Raises
    ------
    ValueError
        If the trajectories are not equal-shape ``(T ≥ 2, 3)``, carry non-finite
        entries or a non-physical Bloch vector (``‖r⃗‖ > 1``), or ``shots`` /
        ``repeats`` are ``< 1`` or ``sampling_stride < 1`` (§15 ladder — surfaced,
        not masked).
    """
    b1 = np.asarray(bloch_1, dtype=np.float64)
    b2 = np.asarray(bloch_2, dtype=np.float64)
    if b1.shape != b2.shape:
        raise ValueError(f"non_markovianity_qpn_bias: trajectories differ in shape {b1.shape} vs {b2.shape}")
    if b1.ndim != 2 or b1.shape[1] != 3:
        raise ValueError(f"non_markovianity_qpn_bias: trajectories must be (T, 3); got {b1.shape}")
    if b1.shape[0] < 2:
        raise ValueError("non_markovianity_qpn_bias: need at least two time samples")
    if not (np.all(np.isfinite(b1)) and np.all(np.isfinite(b2))):
        raise ValueError("non_markovianity_qpn_bias: Bloch trajectories contain non-finite entries")
    norm_1 = float(np.max(np.linalg.norm(b1, axis=1)))
    norm_2 = float(np.max(np.linalg.norm(b2, axis=1)))
    if norm_1 > 1.0 + _BLOCH_TOLERANCE or norm_2 > 1.0 + _BLOCH_TOLERANCE:
        raise ValueError(
            "non_markovianity_qpn_bias: Bloch vectors must be physical (‖r⃗‖ ≤ 1); "
            f"got max norms {norm_1:.6g}, {norm_2:.6g} — a component-wise bound is not "
            "enough (e.g. ⟨σ⟩=(0.8,0.8,0) has ‖r⃗‖>1)"
        )
    if shots < 1:
        raise ValueError(f"non_markovianity_qpn_bias: shots must be >= 1; got {shots}")
    if repeats < 1:
        raise ValueError(f"non_markovianity_qpn_bias: repeats must be >= 1; got {repeats}")
    if sampling_stride < 1:
        raise ValueError(f"non_markovianity_qpn_bias: sampling_stride must be >= 1; got {sampling_stride}")

    # Noise-free references: physical trace distances → the guarded BLP measure.
    measure_true = blp_non_markovianity(_qubit_trace_distance(b1, b2))
    strided_1 = b1[::sampling_stride]
    strided_2 = b2[::sampling_stride]
    measure_infinite_shots = blp_non_markovianity(_qubit_trace_distance(strided_1, strided_2))

    # Finite-shot projective tomography: per axis, p = (1 + ⟨σ_a⟩) / 2, k ~ Binomial(shots, p),
    # ⟨σ_a⟩_hat = 2k/shots − 1. Vectorised over (repeats, T_strided, 3).
    rng = np.random.default_rng(seed)
    p1 = np.clip(0.5 * (1.0 + strided_1), 0.0, 1.0)
    p2 = np.clip(0.5 * (1.0 + strided_2), 0.0, 1.0)
    shape = (repeats, *strided_1.shape)
    est_1 = 2.0 * rng.binomial(shots, np.broadcast_to(p1, shape)) / shots - 1.0
    est_2 = 2.0 * rng.binomial(shots, np.broadcast_to(p2, shape)) / shots - 1.0
    sampled_distance = 0.5 * np.linalg.norm(est_1 - est_2, axis=-1)  # (repeats, T_strided); may exceed 1
    sampled_measures = _positive_increment_sum(sampled_distance)  # (repeats,)

    mean = float(np.mean(sampled_measures))
    std = float(np.std(sampled_measures))
    lo, hi = (float(x) for x in np.percentile(sampled_measures - measure_true, [2.5, 97.5]))

    return QpnBiasResult(
        shots=shots,
        repeats=repeats,
        sampling_stride=sampling_stride,
        measure_true=measure_true,
        measure_infinite_shots=measure_infinite_shots,
        measure_sampled_mean=mean,
        measure_sampled_std=std,
        qpn_bias=mean - measure_infinite_shots,
        sampling_bias=measure_infinite_shots - measure_true,
        total_bias=mean - measure_true,
        total_bias_ci95=(lo, hi),
    )


__all__ = [
    "QpnBiasResult",
    "non_markovianity_qpn_bias",
]
