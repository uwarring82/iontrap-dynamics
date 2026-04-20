# SPDX-License-Identifier: MIT
"""Unit tests for the measurement-layer statistics surface (Dispatch P).

Covers :func:`wilson_interval`, :func:`clopper_pearson_interval`, and
:func:`binomial_summary` — their scalar + vector contracts, numerical
correctness at reference inputs, edge-case handling at
``k ∈ {0, n}``, confidence-level scaling, and statistical coverage
(Monte-Carlo against a known Bernoulli truth). Tests are
backend-agnostic and do not import QuTiP.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iontrap_dynamics import (
    BinomialSummary,
    binomial_summary,
    clopper_pearson_interval,
    wilson_interval,
)

# ----------------------------------------------------------------------------
# wilson_interval — correctness + contract
# ----------------------------------------------------------------------------


class TestWilsonIntervalScalar:
    def test_reference_k5_n10(self) -> None:
        """k=5, n=10 at 95 % — classic worked example."""
        lower, upper = wilson_interval(5, 10)
        # Reference from Wilson (1927) / any stats textbook:
        # [0.2366, 0.7634] — exact to within Φ⁻¹ precision.
        np.testing.assert_allclose(lower, 0.23659309, atol=1e-6)
        np.testing.assert_allclose(upper, 0.76340691, atol=1e-6)

    def test_midpoint_symmetric(self) -> None:
        """p̂ = 0.5: interval should be centred on 0.5."""
        lower, upper = wilson_interval(50, 100)
        np.testing.assert_allclose(0.5 * (lower + upper), 0.5, atol=1e-10)

    def test_k_zero_lower_is_zero(self) -> None:
        lower, upper = wilson_interval(0, 100)
        assert float(lower) == 0.0
        assert 0.0 < float(upper) < 0.1  # < 10 % at 95 % CI for n = 100

    def test_k_equals_n_upper_is_one(self) -> None:
        lower, upper = wilson_interval(100, 100)
        assert float(upper) == 1.0
        assert 0.9 < float(lower) < 1.0

    def test_bounds_in_unit_interval(self) -> None:
        """Wilson never escapes [0, 1] at any (k, n, confidence) combo."""
        for k in (0, 1, 5, 10, 50, 99, 100):
            lower, upper = wilson_interval(k, 100, confidence=0.999)
            assert 0.0 <= float(lower) <= float(upper) <= 1.0

    def test_higher_confidence_is_wider(self) -> None:
        lo_90, hi_90 = wilson_interval(5, 10, confidence=0.90)
        lo_99, hi_99 = wilson_interval(5, 10, confidence=0.99)
        assert float(hi_99) - float(lo_99) > float(hi_90) - float(lo_90)

    def test_larger_n_gives_tighter_interval(self) -> None:
        """Width should shrink like 1/√n for matched p̂."""
        _, hi_10 = wilson_interval(5, 10)
        lo_100, hi_100 = wilson_interval(50, 100)
        lo_1000, hi_1000 = wilson_interval(500, 1000)
        width_10 = float(hi_10) - float(_)
        width_100 = float(hi_100) - float(lo_100)
        width_1000 = float(hi_1000) - float(lo_1000)
        assert width_10 > width_100 > width_1000
        # Check rough 1/√n scaling (allow 30 % slack since Wilson has
        # small finite-n corrections).
        ratio_100_10 = width_100 / width_10
        assert 0.25 < ratio_100_10 < 0.45  # ≈ √(10/100) = 0.316


class TestWilsonIntervalVector:
    def test_elementwise_on_1d_array(self) -> None:
        k = np.array([0, 5, 10])
        lower, upper = wilson_interval(k, 10)
        assert lower.shape == (3,)
        assert upper.shape == (3,)
        assert float(lower[0]) == 0.0
        assert float(upper[-1]) == 1.0
        np.testing.assert_allclose(lower[1], 0.23659309, atol=1e-6)

    def test_broadcasting_trials_scalar(self) -> None:
        k = np.array([10, 20, 30])
        lower, upper = wilson_interval(k, 100)
        assert lower.shape == (3,)
        assert upper.shape == (3,)

    def test_broadcasting_both_arrays(self) -> None:
        k = np.array([5, 10])
        n = np.array([20, 40])
        lower, upper = wilson_interval(k, n)
        assert lower.shape == (2,)
        # Matched p̂ = 0.25 — both share the same point estimate but
        # widths should shrink for the larger n.
        width = upper - lower
        assert float(width[0]) > float(width[1])


# ----------------------------------------------------------------------------
# clopper_pearson_interval — correctness + contract
# ----------------------------------------------------------------------------


class TestClopperPearsonIntervalScalar:
    def test_reference_k5_n10(self) -> None:
        """k=5, n=10 at 95 % — classic worked example."""
        lower, upper = clopper_pearson_interval(5, 10)
        # From scipy or any stats textbook: [0.1871, 0.8129].
        np.testing.assert_allclose(lower, 0.18708, atol=5e-5)
        np.testing.assert_allclose(upper, 0.81292, atol=5e-5)

    def test_k_zero_lower_is_zero(self) -> None:
        lower, upper = clopper_pearson_interval(0, 100)
        assert float(lower) == 0.0
        assert 0.0 < float(upper) < 0.1

    def test_k_equals_n_upper_is_one(self) -> None:
        lower, upper = clopper_pearson_interval(100, 100)
        assert float(upper) == 1.0
        assert 0.9 < float(lower) < 1.0

    def test_bounds_in_unit_interval(self) -> None:
        for k in (0, 1, 5, 10, 50, 99, 100):
            lower, upper = clopper_pearson_interval(k, 100, confidence=0.999)
            assert 0.0 <= float(lower) <= float(upper) <= 1.0

    def test_both_intervals_valid_and_near_each_other(self) -> None:
        """Wilson and C-P give distinct but similar bounds — neither nests the other.

        A common misconception is that C-P strictly contains Wilson
        because it's the "conservative" method. That's not true — the
        two intervals cross at certain (k, n) values. Both *do* have
        nominal ≥ 95 % coverage, and their point estimates agree.
        Test here: both give bounds that bracket the point estimate and
        differ from Wilson by less than ~3 % at n = 100.
        """
        for k in (1, 3, 5, 10, 50, 95):
            wl, wu = wilson_interval(k, 100)
            cp_lo, cp_hi = clopper_pearson_interval(k, 100)
            p_hat = k / 100
            assert float(cp_lo) <= p_hat <= float(cp_hi)
            assert float(wl) <= p_hat <= float(wu)
            # Intervals agree to within ~3 % at n = 100.
            assert abs(float(cp_lo) - float(wl)) < 0.03
            assert abs(float(cp_hi) - float(wu)) < 0.03


class TestClopperPearsonIntervalVector:
    def test_elementwise_on_1d_array(self) -> None:
        k = np.array([0, 5, 10])
        lower, upper = clopper_pearson_interval(k, 10)
        assert lower.shape == (3,)
        assert upper.shape == (3,)
        assert float(lower[0]) == 0.0
        assert float(upper[-1]) == 1.0


# ----------------------------------------------------------------------------
# binomial_summary — dispatch + dataclass
# ----------------------------------------------------------------------------


class TestBinomialSummary:
    def test_wilson_default(self) -> None:
        summary = binomial_summary(5, 10)
        assert isinstance(summary, BinomialSummary)
        assert summary.method == "wilson"
        assert summary.confidence == 0.95
        np.testing.assert_allclose(summary.point_estimate, 0.5)
        np.testing.assert_allclose(summary.lower, 0.23659309, atol=1e-6)
        np.testing.assert_allclose(summary.upper, 0.76340691, atol=1e-6)

    def test_clopper_pearson_dispatch(self) -> None:
        summary = binomial_summary(5, 10, method="clopper-pearson")
        assert summary.method == "clopper-pearson"
        np.testing.assert_allclose(summary.lower, 0.18708, atol=5e-5)

    def test_custom_confidence(self) -> None:
        summary_90 = binomial_summary(5, 10, confidence=0.90)
        summary_99 = binomial_summary(5, 10, confidence=0.99)
        assert summary_90.confidence == 0.90
        assert summary_99.confidence == 0.99
        assert (summary_99.upper - summary_99.lower) > (summary_90.upper - summary_90.lower)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            binomial_summary(5, 10, method="bayesian")

    def test_vectorized_summary(self) -> None:
        k = np.array([0, 5, 10])
        summary = binomial_summary(k, 10)
        assert summary.point_estimate.shape == (3,)
        np.testing.assert_allclose(summary.point_estimate, [0.0, 0.5, 1.0])

    def test_frozen(self) -> None:
        summary = binomial_summary(5, 10)
        with pytest.raises(FrozenInstanceError):
            summary.method = "clopper-pearson"  # type: ignore[misc]

    def test_metadata_fields_preserved(self) -> None:
        k = np.array([3, 7])
        n = np.array([10, 20])
        summary = binomial_summary(k, n, confidence=0.68)
        np.testing.assert_array_equal(summary.successes, k)
        np.testing.assert_array_equal(summary.trials, n)
        assert summary.confidence == 0.68


# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------


class TestValidation:
    def test_zero_trials_raises(self) -> None:
        with pytest.raises(ValueError, match="trials must be >= 1"):
            wilson_interval(0, 0)

    def test_negative_successes_raises(self) -> None:
        with pytest.raises(ValueError, match="successes must be >= 0"):
            wilson_interval(-1, 10)

    def test_successes_exceeding_trials_raises(self) -> None:
        with pytest.raises(ValueError, match="successes must be <= trials"):
            wilson_interval(11, 10)

    def test_confidence_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=r"confidence must lie in \(0, 1\)"):
            wilson_interval(5, 10, confidence=0.0)

    def test_confidence_one_raises(self) -> None:
        with pytest.raises(ValueError, match=r"confidence must lie in \(0, 1\)"):
            wilson_interval(5, 10, confidence=1.0)

    def test_confidence_negative_raises(self) -> None:
        with pytest.raises(ValueError, match=r"confidence must lie in \(0, 1\)"):
            clopper_pearson_interval(5, 10, confidence=-0.1)

    def test_vector_validation_catches_single_bad_entry(self) -> None:
        k = np.array([3, 15])  # 15 > trials = 10
        with pytest.raises(ValueError, match="successes must be <= trials"):
            wilson_interval(k, 10)


# ----------------------------------------------------------------------------
# Monte-Carlo coverage tests
# ----------------------------------------------------------------------------


class TestMonteCarloCoverage:
    def test_wilson_coverage_at_p_midrange(self) -> None:
        """95 % Wilson should cover p=0.3 at >= 93 % empirically.

        The nominal 95 % coverage isn't exact for discrete distributions;
        the Wilson formula oscillates as a function of (p, n) and
        typically sits in 93–97 %. We test >= 93 % — a standard
        textbook guarantee at n = 100 — and allow the upper end to
        float, since over-coverage is acceptable.
        """
        rng = np.random.default_rng(2026)
        true_p = 0.3
        n = 100
        n_trials = 10_000
        k = rng.binomial(n, true_p, size=n_trials)
        lower, upper = wilson_interval(k, n, confidence=0.95)
        covered = (lower <= true_p) & (true_p <= upper)
        coverage = float(covered.mean())
        assert coverage >= 0.93

    def test_clopper_pearson_is_conservative(self) -> None:
        """95 % C-P actual coverage must exceed 95 % (never anti-conservative)."""
        rng = np.random.default_rng(2027)
        true_p = 0.3
        n = 100
        n_trials = 10_000
        k = rng.binomial(n, true_p, size=n_trials)
        lower, upper = clopper_pearson_interval(k, n, confidence=0.95)
        covered = (lower <= true_p) & (true_p <= upper)
        coverage = float(covered.mean())
        # C-P is conservative — nominal 0.95, actual typically 0.96–0.98.
        assert coverage >= 0.95

    def test_wilson_coverage_near_boundary(self) -> None:
        """p=0.05 is the regime where Wald fails and Wilson shines."""
        rng = np.random.default_rng(2028)
        true_p = 0.05
        n = 100
        n_trials = 10_000
        k = rng.binomial(n, true_p, size=n_trials)
        lower, upper = wilson_interval(k, n, confidence=0.95)
        covered = (lower <= true_p) & (true_p <= upper)
        coverage = float(covered.mean())
        # Wilson keeps near-nominal coverage even near p → 0.
        assert coverage >= 0.93
