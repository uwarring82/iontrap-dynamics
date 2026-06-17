# SPDX-License-Identifier: MIT
"""Analytic-regression oracle for the QPN-bias estimator (Phase A, WI-3).

Validates the projection-noise bias of the BLP measure (Wittemer et al.,
PRA 97, 020102(R) (2018)): the QPN bias is **positive** (rectified positive
increments) and shrinks as ``shots → ∞``; the finite-sampling (coarse-``γ``) bias
is **negative**; and the QPN + sampling parts **telescope** to the total. Bloch
trajectories use the antipodal revival signal ``D = |cos t|`` (so ``𝒩_true > 0``).
"""

from __future__ import annotations

import numpy as np
import pytest

from iontrap_dynamics.information import blp_non_markovianity, non_markovianity_qpn_bias

pytestmark = pytest.mark.regression_analytic


def _revival(n_t: int = 120, half_periods: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Antipodal equatorial Bloch trajectories with D(t) = |cos t| (revivals)."""
    t = np.linspace(0.0, half_periods * np.pi, n_t)
    f = np.cos(t)
    zeros = np.zeros_like(f)
    return np.stack([f, zeros, zeros], axis=1), np.stack([-f, zeros, zeros], axis=1)


def test_measure_true_matches_blp_on_bloch_distance() -> None:
    b1, b2 = _revival()
    res = non_markovianity_qpn_bias(b1, b2, shots=1000, repeats=2, seed=0)
    d_true = 0.5 * np.linalg.norm(b1 - b2, axis=1)
    assert res.measure_true == pytest.approx(blp_non_markovianity(d_true), abs=1e-12)
    assert res.measure_true > 1.0  # several revivals over the window


def test_qpn_bias_positive_and_vanishes_with_shots() -> None:
    b1, b2 = _revival()
    small = non_markovianity_qpn_bias(b1, b2, shots=50, repeats=300, seed=1)
    large = non_markovianity_qpn_bias(b1, b2, shots=200_000, repeats=300, seed=1)
    assert small.qpn_bias > 0.0  # QPN rectifies into a positive bias at low shots
    assert abs(large.qpn_bias) < 0.05  # ℬ_QPN → 0 as shots → ∞ (the paper's headline)
    assert large.qpn_bias < small.qpn_bias  # monotone shrink with shots (1/√r)
    assert abs(large.qpn_bias) < 0.1 * small.qpn_bias  # vanishes relative to the low-shot bias


def test_sampling_bias_negative_for_coarse_grid() -> None:
    b1, b2 = _revival(n_t=160)
    res = non_markovianity_qpn_bias(b1, b2, shots=10_000, repeats=50, sampling_stride=8, seed=2)
    # A coarse time grid (low γ) misses fast features of D(t) → underestimate, r-independent.
    assert res.sampling_bias < 0.0
    assert res.measure_infinite_shots < res.measure_true


def test_bias_decomposition_telescopes() -> None:
    b1, b2 = _revival()
    res = non_markovianity_qpn_bias(b1, b2, shots=400, repeats=100, sampling_stride=3, seed=3)
    # ℬ_total = ℬ_QPN + ℬ_sampling = (mean − 𝒩(γ,∞)) + (𝒩(γ,∞) − 𝒩_true).
    assert res.qpn_bias + res.sampling_bias == pytest.approx(res.total_bias, abs=1e-12)


def test_reproducible_with_seed() -> None:
    b1, b2 = _revival()
    a = non_markovianity_qpn_bias(b1, b2, shots=200, repeats=64, seed=7)
    b = non_markovianity_qpn_bias(b1, b2, shots=200, repeats=64, seed=7)
    assert a.measure_sampled_mean == b.measure_sampled_mean
    assert a.total_bias_ci95 == b.total_bias_ci95


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shots": 0}, "shots"),
        ({"shots": 100, "repeats": 0}, "repeats"),
        ({"shots": 100, "sampling_stride": 0}, "sampling_stride"),
    ],
)
def test_invalid_parameters_raise(kwargs: dict, match: str) -> None:
    b1, b2 = _revival(n_t=10)
    with pytest.raises(ValueError, match=match):
        non_markovianity_qpn_bias(b1, b2, **kwargs)


def test_nonphysical_or_misshaped_bloch_raise() -> None:
    b1, b2 = _revival(n_t=10)
    with pytest.raises(ValueError, match="T, 3"):
        non_markovianity_qpn_bias(b1[:, :2], b2[:, :2], shots=100)
    bad = b1.copy()
    bad[0, 0] = 1.5  # |⟨σ_x⟩| > 1 — non-physical Bloch component
    with pytest.raises(ValueError, match="physical"):
        non_markovianity_qpn_bias(bad, b2, shots=100)
