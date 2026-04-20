# SPDX-License-Identifier: MIT
"""Detector-response models for the measurement layer.

Detectors compose with rate-consuming channels (Dispatch K's
:class:`PoissonChannel`). :class:`DetectorConfig` is a small frozen
container for the three parameters that matter for ion-fluorescence
readout — collection / quantum efficiency ``η``, stray-count rate
``γ_d``, and the bright/dark discrimination threshold ``N̂`` — plus
three methods that implement the rate transform, the per-shot
thresholding, and an analytic classification-fidelity computation.

Physical composition
--------------------

**Thinning.** A Poisson stream at rate ``λ`` passed through an ideal
binary detector with efficiency ``η`` is still Poisson, with rate
``η · λ``. This is exact — no approximation — because thinning a
Poisson process by an independent Bernoulli(η) preserves Poisson
statistics. (See Kingman, *Poisson Processes*, §3.)

**Additive background.** Stray light and detector dark counts arrive
as an independent Poisson process at rate ``γ_d`` per shot; sums of
independent Poissons are Poisson, so the total detected rate is

    λ_detected = η · λ_emitted + γ_d

with no cross terms.

**Thresholding.** A shot is classified ``bright`` if its count is at
least ``N̂``; otherwise ``dark``. The fidelity of that classification
against a known-bright or known-dark rate follows from the Poisson
CDF and is computed by :meth:`DetectorConfig.classification_fidelity`.

The orchestrator :func:`sample_outcome` is deliberately detector-
agnostic: callers transform the emitted rate with
:meth:`DetectorConfig.apply` *before* passing it to the channel, and
threshold the sampled counts with :meth:`DetectorConfig.discriminate`
*after*. This keeps the channel / detector boundary clean and lets
future non-Poisson detector models slot in without breaking the
orchestrator's signature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, kw_only=True)
class DetectorConfig:
    """Detector-response parameters for ion-fluorescence readout.

    Parameters
    ----------
    efficiency
        Collection + quantum efficiency ``η ∈ [0, 1]``. Thins the
        emitted Poisson rate multiplicatively.
    dark_count_rate
        Mean stray-light / dark-count rate ``γ_d ≥ 0`` in counts per
        shot. Adds an independent Poisson background.
    threshold
        Classification threshold ``N̂ ≥ 1``. Per-shot counts strictly
        below ``N̂`` are classified dark (0); counts at or above are
        classified bright (1). ``threshold = 1`` is the "any photon
        means bright" limit; experiments typically tune ``N̂`` to
        balance false-positive and false-negative rates.
    label
        Identifier used by downstream protocol code when a result
        carries multiple detector models (e.g. two ions on separate
        PMTs). Defaults to ``"detector"``.

    Raises
    ------
    ValueError
        At construction, if ``efficiency`` lies outside ``[0, 1]``,
        ``dark_count_rate`` is negative, or ``threshold < 1``. These
        are system-boundary input checks — detector parameters are
        physical data, not convention-level rules.
    """

    efficiency: float
    dark_count_rate: float
    threshold: int
    label: str = "detector"

    def __post_init__(self) -> None:
        if not (0.0 <= self.efficiency <= 1.0):
            raise ValueError(
                f"DetectorConfig: efficiency must lie in [0, 1]; got {self.efficiency}"
            )
        if self.dark_count_rate < 0.0:
            raise ValueError(
                f"DetectorConfig: dark_count_rate must be >= 0; got {self.dark_count_rate}"
            )
        if self.threshold < 1:
            raise ValueError(f"DetectorConfig: threshold must be >= 1; got {self.threshold}")

    def apply(self, emitted_rate: NDArray[np.floating]) -> NDArray[np.float64]:
        """Transform emitted Poisson rate into the detected rate.

        Returns ``η · λ_emitted + γ_d``, broadcasting element-wise.

        Parameters
        ----------
        emitted_rate
            Array of non-negative Poisson rates (mean counts per shot)
            *before* detector thinning and dark-count addition.

        Returns
        -------
        NDArray[np.float64]
            Same shape as the input, with entries
            ``η · emitted_rate + γ_d``. Always non-negative.

        Raises
        ------
        ValueError
            If any entry of ``emitted_rate`` is negative.
        """
        lam = np.asarray(emitted_rate, dtype=np.float64)
        if np.any(lam < 0.0):
            raise ValueError(
                f"DetectorConfig.apply: emitted_rate must be >= 0; got min={lam.min()}"
            )
        return self.efficiency * lam + self.dark_count_rate

    def discriminate(self, counts: NDArray[np.integer]) -> NDArray[np.int8]:
        """Threshold per-shot counts into bright/dark bits.

        Returns 1 (bright) where ``counts >= threshold``, 0 (dark)
        elsewhere. Shape of the input is preserved.

        Parameters
        ----------
        counts
            Integer-valued count array, typically the Poisson samples
            under :attr:`MeasurementResult.sampled_outcome`. Any
            non-negative integer shape is accepted.

        Returns
        -------
        NDArray[np.int8]
            Same shape as ``counts``, entries in ``{0, 1}``.

        Raises
        ------
        ValueError
            If any entry of ``counts`` is negative.
        """
        arr = np.asarray(counts)
        if np.any(arr < 0):
            raise ValueError(
                f"DetectorConfig.discriminate: counts must be >= 0; got min={int(arr.min())}"
            )
        return (arr >= self.threshold).astype(np.int8)

    def classification_fidelity(
        self,
        *,
        lambda_bright: float,
        lambda_dark: float,
    ) -> dict[str, float]:
        """Analytic bright/dark classification rates at fixed rates.

        Given *emitted* rates ``λ_bright`` and ``λ_dark``, returns the
        Poisson-CDF-derived true-positive rate (probability of
        correctly classifying a bright shot as bright) and true-
        negative rate (probability of correctly classifying a dark
        shot as dark), along with the overall fidelity and the
        effective per-shot rates post-thinning.

        Parameters
        ----------
        lambda_bright
            Emitted photon rate per shot when the qubit is bright.
            Must be ``>= lambda_dark``; negative values raise.
        lambda_dark
            Emitted photon rate per shot when the qubit is dark.
            Must be ``>= 0``.

        Returns
        -------
        dict[str, float]
            ``{"true_positive_rate": P(count ≥ N̂ | bright),
               "true_negative_rate": P(count < N̂ | dark),
               "fidelity": mean of the two,
               "effective_bright_rate": η·λ_bright + γ_d,
               "effective_dark_rate":   η·λ_dark   + γ_d}``

        Notes
        -----
        Uses :func:`scipy.stats.poisson.cdf`. The ``fidelity`` entry
        assumes the qubit is equally likely to be bright or dark —
        callers weighting by an a-priori probability should compute
        the weighted sum themselves.

        Raises
        ------
        ValueError
            If rates are negative or if ``lambda_dark > lambda_bright``.
        """
        from scipy.stats import poisson

        if lambda_bright < 0.0 or lambda_dark < 0.0:
            raise ValueError(
                "DetectorConfig.classification_fidelity: rates must be >= 0; "
                f"got lambda_bright={lambda_bright}, lambda_dark={lambda_dark}"
            )
        if lambda_dark > lambda_bright:
            raise ValueError(
                "DetectorConfig.classification_fidelity: lambda_dark must be "
                f"<= lambda_bright; got {lambda_dark} > {lambda_bright}"
            )

        eff_bright = self.efficiency * lambda_bright + self.dark_count_rate
        eff_dark = self.efficiency * lambda_dark + self.dark_count_rate

        # P(count >= N̂ | bright) = 1 − P(count <= N̂−1 | bright)
        true_positive = float(1.0 - poisson.cdf(self.threshold - 1, mu=eff_bright))
        # P(count <  N̂ | dark) = P(count <= N̂−1 | dark)
        true_negative = float(poisson.cdf(self.threshold - 1, mu=eff_dark))

        return {
            "true_positive_rate": true_positive,
            "true_negative_rate": true_negative,
            "fidelity": 0.5 * (true_positive + true_negative),
            "effective_bright_rate": eff_bright,
            "effective_dark_rate": eff_dark,
        }


__all__ = [
    "DetectorConfig",
]
