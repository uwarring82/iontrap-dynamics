# SPDX-License-Identifier: MIT
"""Confidence-interval estimators for measurement-layer count data.

Closes the Dispatch P surface of the measurement track. Consumes the
binomial counts produced by the protocols (bright-fraction bits from
:class:`SpinReadout`, per-ion counts from :class:`ParityScan`,
sideband-branch bright fractions from :class:`SidebandInference`) and
returns Wilson-score or Clopper–Pearson confidence intervals as
:class:`BinomialSummary` records.

Two methods are supported and both are fully vectorised over
``(successes, trials)`` — scalar or 1-D (broadcasting is standard
NumPy rules):

- :func:`wilson_interval` — Wilson score interval. Well-behaved at
  ``p̂ = 0`` and ``p̂ = 1`` and close to the nominal coverage for
  modest ``n``. The recommended default.
- :func:`clopper_pearson_interval` — exact binomial-quantile interval
  via the Beta distribution. Conservative (actual coverage ≥ nominal)
  but guaranteed. Useful for reporting worst-case uncertainties on
  small shot budgets.

No Wald interval is shipped: its coverage collapses near the
``p̂ ∈ {0, 1}`` extremes that arise routinely in ion-trap readout
(ground-state RSB probes, high-fidelity detectors on ``|↓⟩`` shots),
so it would be a silent footgun.

Conventions for z / confidence level follow §17.12:
``z = Φ^{-1}((1 + confidence) / 2)`` via :func:`scipy.stats.norm.ppf`,
so ``confidence = 0.95`` gives ``z ≈ 1.959963984…``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, kw_only=True)
class BinomialSummary:
    """Binomial-proportion point estimate plus confidence interval.

    Shape-preserving: all four numerical fields (``successes``,
    ``trials``, ``point_estimate``, ``lower``, ``upper``) share the same
    shape, broadcasted from the inputs to the factory function.

    Parameters
    ----------
    successes
        Counts of '1' outcomes. Integer-valued, non-negative,
        element-wise ``<= trials``.
    trials
        Total number of Bernoulli trials per entry. Positive integer.
    point_estimate
        ``successes / trials``, computed as ``float64``.
    lower
        Lower bound of the two-sided CI at the stated ``confidence``.
        ``[0, 1]``-valued.
    upper
        Upper bound of the two-sided CI at the stated ``confidence``.
        ``[0, 1]``-valued, ``>= lower``.
    confidence
        Nominal coverage in ``(0, 1)`` — e.g. ``0.95`` for 95 % CI.
    method
        Which estimator was used — ``"wilson"`` or ``"clopper-pearson"``.
    """

    successes: NDArray[np.int64]
    trials: NDArray[np.int64]
    point_estimate: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    confidence: float
    method: str


def wilson_interval(
    successes: int | NDArray[np.integer],
    trials: int | NDArray[np.integer],
    *,
    confidence: float = 0.95,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wilson score confidence interval for binomial proportion.

    Parameters
    ----------
    successes
        Counts of '1' outcomes. Scalar or array.
    trials
        Bernoulli trial counts per entry. Scalar or array. Must
        broadcast against ``successes``.
    confidence
        Two-sided nominal coverage, in ``(0, 1)``. Default ``0.95``.

    Returns
    -------
    tuple
        ``(lower, upper)`` — same shape as the broadcast of
        ``(successes, trials)``, both arrays in ``[0, 1]``.

    Notes
    -----
    Formula::

        p̂ = k / n
        z = Φ⁻¹((1 + confidence) / 2)
        centre = (p̂ + z² / (2n)) / (1 + z² / n)
        half_width = (z / (1 + z² / n)) · √(p̂(1 − p̂) / n + z² / (4n²))
        (lower, upper) = (centre − half_width, centre + half_width)

    Handles ``p̂ ∈ {0, 1}`` cleanly — ``lower = 0`` when ``k = 0``
    and ``upper = 1`` when ``k = n`` by construction of the formula.
    Clipped to ``[0, 1]`` as a defence against pathological rounding.

    Raises
    ------
    ValueError
        If ``confidence`` is not in ``(0, 1)``, or if any
        ``successes < 0`` or ``successes > trials`` or ``trials < 1``.
    """
    k, n = _validate_counts(successes, trials)
    z = _z_score(confidence)
    z2 = z * z

    p_hat = k / n
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * np.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    lower = np.clip(centre - half_width, 0.0, 1.0)
    upper = np.clip(centre + half_width, 0.0, 1.0)
    # Snap k=0 and k=n to exact boundaries — closed-form arithmetic
    # leaves sub-epsilon noise there that otherwise surprises callers.
    lower = np.where(k == 0, 0.0, lower)
    upper = np.where(k == n, 1.0, upper)
    return lower, upper


def clopper_pearson_interval(
    successes: int | NDArray[np.integer],
    trials: int | NDArray[np.integer],
    *,
    confidence: float = 0.95,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Clopper–Pearson (exact) confidence interval for binomial proportion.

    Parameters
    ----------
    successes
        Counts of '1' outcomes. Scalar or array.
    trials
        Bernoulli trial counts per entry.
    confidence
        Two-sided nominal coverage, in ``(0, 1)``. Default ``0.95``.

    Returns
    -------
    tuple
        ``(lower, upper)`` — same shape as the broadcast of
        ``(successes, trials)``, both arrays in ``[0, 1]``.

    Notes
    -----
    Uses the Beta-distribution-inverse formulation
    (Brown–Cai–DasGupta, 2001):

        α = 1 − confidence
        lower = Beta⁻¹(α/2;     k,     n − k + 1)     (0 when k = 0)
        upper = Beta⁻¹(1 − α/2; k + 1, n − k)         (1 when k = n)

    The interval's actual coverage is guaranteed to meet or exceed the
    nominal ``confidence`` — *conservative*, never anti-conservative.
    The cost is wider intervals than Wilson for the same ``n``; callers
    reporting worst-case uncertainties (publication / regulatory)
    should prefer Clopper–Pearson.

    Raises
    ------
    ValueError
        If ``confidence`` is not in ``(0, 1)``, or if any
        ``successes < 0`` or ``successes > trials`` or ``trials < 1``.
    """
    from scipy.stats import beta  # local import — only touched by CP

    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must lie in (0, 1); got {confidence}")
    k, n = _validate_counts(successes, trials)
    alpha = 1.0 - confidence

    # Scalar / array-compatible. beta.ppf returns NaN when one of its
    # shape parameters is 0; handle the k=0 and k=n cases explicitly.
    lower = np.where(k == 0, 0.0, beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = np.where(k == n, 1.0, beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def binomial_summary(
    successes: int | NDArray[np.integer],
    trials: int | NDArray[np.integer],
    *,
    confidence: float = 0.95,
    method: str = "wilson",
) -> BinomialSummary:
    """Point estimate + CI as a :class:`BinomialSummary`.

    Parameters
    ----------
    successes
        Counts of '1' outcomes. Scalar or array.
    trials
        Bernoulli trial counts per entry.
    confidence
        Two-sided nominal coverage, in ``(0, 1)``. Default ``0.95``.
    method
        Which estimator to use. One of ``"wilson"`` (default) or
        ``"clopper-pearson"``.

    Returns
    -------
    BinomialSummary
        With ``successes``, ``trials``, ``point_estimate``, ``lower``,
        ``upper`` all as shape-aligned ``NDArray`` records, plus the
        ``confidence`` and ``method`` metadata fields.

    Raises
    ------
    ValueError
        On unknown ``method``, or any input-validation failure from
        the underlying interval function.
    """
    k, n = _validate_counts(successes, trials)
    point = k / n

    if method == "wilson":
        lower, upper = wilson_interval(k, n, confidence=confidence)
    elif method == "clopper-pearson":
        lower, upper = clopper_pearson_interval(k, n, confidence=confidence)
    else:
        raise ValueError(
            f"binomial_summary: unknown method {method!r}; expected 'wilson' or 'clopper-pearson'."
        )

    return BinomialSummary(
        successes=k,
        trials=n,
        point_estimate=point.astype(np.float64),
        lower=lower,
        upper=upper,
        confidence=confidence,
        method=method,
    )


def _validate_counts(
    successes: int | NDArray[np.integer],
    trials: int | NDArray[np.integer],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Broadcast + validate a ``(successes, trials)`` pair.

    Returns both as ``int64`` arrays broadcast to a common shape. The
    arithmetic in Wilson / CP is written to handle array inputs, so
    callers pass through these normalised arrays.
    """
    k = np.asarray(successes, dtype=np.int64)
    n = np.asarray(trials, dtype=np.int64)
    k, n = np.broadcast_arrays(k, n)

    if np.any(n < 1):
        raise ValueError(f"trials must be >= 1; got min={int(n.min())}")
    if np.any(k < 0):
        raise ValueError(f"successes must be >= 0; got min={int(k.min())}")
    if np.any(k > n):
        raise ValueError(
            f"successes must be <= trials element-wise; max excess = {int((k - n).max())}"
        )
    return k.astype(np.int64, copy=True), n.astype(np.int64, copy=True)


def _z_score(confidence: float) -> float:
    """Normal ppf for a two-sided confidence level.

    ``z = Φ⁻¹((1 + confidence) / 2)``. Validates ``confidence ∈ (0, 1)``.
    """
    from scipy.stats import norm

    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must lie in (0, 1); got {confidence}")
    return float(norm.ppf(0.5 * (1.0 + confidence)))


__all__ = [
    "BinomialSummary",
    "binomial_summary",
    "clopper_pearson_interval",
    "wilson_interval",
]
