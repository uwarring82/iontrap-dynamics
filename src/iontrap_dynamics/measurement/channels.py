# SPDX-License-Identifier: MIT
"""Sampling channels for the measurement layer.

A *channel* maps noise-free inputs (probabilities or rates) to per-shot
or aggregated stochastic outcomes. Channels are intentionally thin:
they do not know about operators, states, or detector hardware.
Composition with detector models (Dispatch L) and with observable-to-
input reductions (Dispatches M–N) happens a layer up.

Dispatch H ships :class:`BernoulliChannel` — each input probability
``p ∈ [0, 1]`` produces ``shots`` independent bits in ``{0, 1}`` with
``P(1) = p``. The per-shot shape is kept (not pre-aggregated to counts)
so that downstream protocol code can build parity estimators and
shot-shot correlations without re-sampling.

Dispatch J adds :class:`BinomialChannel` — same probability contract,
but returns the aggregated count of '1' outcomes per input as a single
integer in ``[0, shots]``. Output shape drops the shot axis and becomes
``(n_inputs,)``. The two channels are distributionally equivalent
(Binomial ≡ sum of Bernoulli) but not bit-identical under a shared
seed: Binomial uses :meth:`numpy.random.Generator.binomial` directly
for efficiency, which consumes RNG bits differently than the Bernoulli
threshold path. Downstream statistics (Wilson CI, Clopper–Pearson, …)
in Dispatch P consume the Binomial counts directly.

Dispatch K adds :class:`PoissonChannel` — per-shot photon-counting
path. Inputs are **rates** (mean counts per shot, ``λ ≥ 0``), not
probabilities, so the orchestrator keyword is ``inputs`` rather than
``probabilities``; each channel names what it consumes through the
``ideal_label`` class attribute (``"probability"`` for
Bernoulli / Binomial, ``"rate"`` for Poisson). Output is
``(shots, n_inputs)`` int64 counts via
:meth:`numpy.random.Generator.poisson`.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ..results import (
    MeasurementResult,
    ResultMetadata,
    StorageMode,
    TrajectoryResult,
)

try:
    _LIBRARY_VERSION = _pkg_version("iontrap-dynamics")
except PackageNotFoundError:
    _LIBRARY_VERSION = "unknown"


@dataclass(frozen=True, slots=True, kw_only=True)
class BernoulliChannel:
    """Per-shot Bernoulli sampling channel.

    For each input probability ``p_j ∈ [0, 1]``, returns an independent
    sample in ``{0, 1}`` per shot. Output shape: ``(shots, n_inputs)``
    so shot-axis is always the leading axis.

    Parameters
    ----------
    label
        Identifier routed into :attr:`MeasurementResult.sampled_outcome`
        when the channel is dispatched through :func:`sample_outcome`.
        Defaults to ``"bernoulli"``; override when a result carries
        multiple Bernoulli channels on different sites or observables.
    """

    ideal_label: ClassVar[str] = "probability"
    label: str = "bernoulli"

    def sample(
        self,
        probabilities: NDArray[np.floating],
        *,
        shots: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int8]:
        """Draw ``shots`` independent Bernoulli bits per input probability.

        Parameters
        ----------
        probabilities
            1-D array of Bernoulli success probabilities. Every entry must
            lie in ``[0, 1]``; violations raise :class:`ValueError`
            (system-boundary input — not a convention error).
        shots
            Number of independent Bernoulli trials per probability entry.
            Must be ``>= 1``.
        rng
            NumPy random generator supplying the stochastic bits.

        Returns
        -------
        NDArray[np.int8]
            Array of shape ``(shots, probabilities.size)`` with entries in
            ``{0, 1}``. ``int8`` keeps the memory footprint tight for
            large shot counts; callers wanting counts reduce along
            ``axis=0``.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim != 1:
            raise ValueError(
                f"BernoulliChannel.sample: probabilities must be 1-D; got shape {probs.shape}"
            )
        if shots < 1:
            raise ValueError(f"BernoulliChannel.sample: shots must be >= 1; got {shots}")
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError(
                "BernoulliChannel.sample: probabilities must lie in [0, 1]; "
                f"min={probs.min()}, max={probs.max()}"
            )
        uniforms = rng.random(size=(shots, probs.size))
        return (uniforms < probs).astype(np.int8)


@dataclass(frozen=True, slots=True, kw_only=True)
class BinomialChannel:
    """Aggregated Bernoulli sampling channel.

    For each input probability ``p_j ∈ [0, 1]``, returns the count of
    '1' outcomes across ``shots`` independent trials — a single integer
    in ``[0, shots]``. Output shape: ``(n_inputs,)`` — the shot axis is
    absorbed into the count.

    Distributionally equivalent to summing
    :meth:`BernoulliChannel.sample` along ``axis=0``, but uses
    :meth:`numpy.random.Generator.binomial` directly for efficiency and
    therefore is **not** bit-identical under a matched seed. Choose
    Bernoulli when per-shot granularity matters (parity scans,
    correlations); choose Binomial when only the aggregate count is
    needed (population estimation, Wilson-CI bars, χ²-goodness-of-fit).

    Parameters
    ----------
    label
        Identifier routed into :attr:`MeasurementResult.sampled_outcome`
        when dispatched through :func:`sample_outcome`. Defaults to
        ``"binomial"``; override to distinguish multiple binomial
        channels on different sites or observables.
    """

    ideal_label: ClassVar[str] = "probability"
    label: str = "binomial"

    def sample(
        self,
        probabilities: NDArray[np.floating],
        *,
        shots: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Draw a binomial count per input probability.

        Parameters
        ----------
        probabilities
            1-D array of success probabilities in ``[0, 1]``. Violations
            raise :class:`ValueError` (system-boundary input).
        shots
            Number of Bernoulli trials aggregated per probability entry.
            Must be ``>= 1``.
        rng
            NumPy random generator supplying the stochastic bits.

        Returns
        -------
        NDArray[np.int64]
            Array of shape ``(probabilities.size,)`` with entries in
            ``[0, shots]``. ``int64`` accommodates shot counts up to the
            integer-overflow boundary without per-call dtype
            considerations.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim != 1:
            raise ValueError(
                f"BinomialChannel.sample: probabilities must be 1-D; got shape {probs.shape}"
            )
        if shots < 1:
            raise ValueError(f"BinomialChannel.sample: shots must be >= 1; got {shots}")
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError(
                "BinomialChannel.sample: probabilities must lie in [0, 1]; "
                f"min={probs.min()}, max={probs.max()}"
            )
        return rng.binomial(shots, probs).astype(np.int64)


@dataclass(frozen=True, slots=True, kw_only=True)
class PoissonChannel:
    """Per-shot Poisson-counting channel.

    Models photon-counting readout: for each input rate
    ``λ_j ≥ 0`` (mean counts per shot), returns an independent Poisson
    sample per shot. Output shape: ``(shots, n_inputs)`` — shot axis
    leading per §17.1, matching :class:`BernoulliChannel`.

    Physics note. In atomic-readout contexts the rate is typically
    ``λ(t) = λ_dark + (λ_bright − λ_dark) · p_↑(t)``, so the Poisson
    channel consumes a mixture of dark-count and fluorescence rates
    weighted by the qubit state. The channel itself is state-agnostic:
    it just samples Poisson(λ). Rate → probability conversion (if
    needed for bright/dark thresholding) happens a layer up, in the
    protocols module (Dispatches L–N).

    Parameters
    ----------
    label
        Identifier routed into :attr:`MeasurementResult.sampled_outcome`
        when dispatched through :func:`sample_outcome`. Defaults to
        ``"poisson"``; override to distinguish multiple Poisson channels
        on different sites or detection windows.
    """

    ideal_label: ClassVar[str] = "rate"
    label: str = "poisson"

    def sample(
        self,
        rates: NDArray[np.floating],
        *,
        shots: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Draw ``shots`` independent Poisson counts per input rate.

        Parameters
        ----------
        rates
            1-D array of non-negative Poisson rates (mean counts per
            shot). Violations raise :class:`ValueError`
            (system-boundary input).
        shots
            Number of independent Poisson draws per rate entry.
            Must be ``>= 1``.
        rng
            NumPy random generator supplying the stochastic bits.

        Returns
        -------
        NDArray[np.int64]
            Array of shape ``(shots, rates.size)`` with non-negative
            integer entries. ``int64`` accommodates large counts without
            per-call overflow considerations.
        """
        lam = np.asarray(rates, dtype=np.float64)
        if lam.ndim != 1:
            raise ValueError(f"PoissonChannel.sample: rates must be 1-D; got shape {lam.shape}")
        if shots < 1:
            raise ValueError(f"PoissonChannel.sample: shots must be >= 1; got {shots}")
        if np.any(lam < 0.0):
            raise ValueError(f"PoissonChannel.sample: rates must be >= 0; got min={lam.min()}")
        return rng.poisson(lam, size=(shots, lam.size)).astype(np.int64)


def sample_outcome(
    *,
    channel: BernoulliChannel | BinomialChannel | PoissonChannel,
    inputs: NDArray[np.floating],
    shots: int,
    seed: int | None = None,
    upstream: TrajectoryResult | None = None,
    provenance_tags: tuple[str, ...] = (),
) -> MeasurementResult:
    """Apply ``channel`` to ``inputs`` and wrap the output.

    The thin orchestrator leans on each channel's uniform
    ``.sample(inputs, *, shots, rng)`` signature; its return shape is
    channel-dependent (Bernoulli → ``(shots, n_inputs)``, Binomial →
    ``(n_inputs,)``, Poisson → ``(shots, n_inputs)``). Detector-composed
    channels (Dispatch L) slot in without changing this signature.

    The orchestrator is input-neutral: each channel's
    :attr:`ideal_label` class attribute names what ``inputs`` means to
    it (``"probability"`` for Bernoulli / Binomial, ``"rate"`` for
    Poisson), and that label becomes the key in
    :attr:`MeasurementResult.ideal_outcome`.

    Parameters
    ----------
    channel
        Sampling channel to apply. Its ``label`` names the entry in
        :attr:`MeasurementResult.sampled_outcome`; its ``ideal_label``
        names the entry in :attr:`MeasurementResult.ideal_outcome`.
    inputs
        1-D array of channel inputs — probabilities in ``[0, 1]`` for
        Bernoulli / Binomial, non-negative rates for Poisson. Typically
        derived from an expectation-value trajectory
        (e.g. ``p_up = (1 + sigma_z_traj) / 2``, or
        ``lam = lam_dark + (lam_bright − lam_dark) * p_up``).
    shots
        Number of independent shots per input entry.
    seed
        Optional seed for :func:`numpy.random.default_rng`. When
        supplied, the resulting :class:`MeasurementResult` is fully
        reproducible for a given ``(channel, seed, inputs, shots)``
        tuple. Bit-reproducibility across channel types is *not*
        guaranteed — different channels consume RNG bits differently
        for efficiency.
    upstream
        Optional :class:`TrajectoryResult` that produced ``probabilities``.
        When supplied, its metadata (convention version, backend, Fock
        truncations) is inherited into the measurement result, and
        ``trajectory_hash`` records the upstream ``request_hash`` so the
        provenance chain is preserved.
    provenance_tags
        Extra tags to concatenate onto the inherited provenance.

    Returns
    -------
    MeasurementResult
        With ``ideal_outcome = {channel.ideal_label: inputs}``,
        ``sampled_outcome = {channel.label: samples}`` (shape depends on
        channel), and ``storage_mode = OMITTED`` in its inherited
        metadata.
    """
    rng = np.random.default_rng(seed)
    samples = channel.sample(inputs, shots=shots, rng=rng)

    metadata = _build_metadata(upstream=upstream, provenance_tags=provenance_tags)
    trajectory_hash = upstream.metadata.request_hash if upstream is not None else None

    return MeasurementResult(
        metadata=metadata,
        shots=shots,
        rng_seed=seed,
        ideal_outcome={channel.ideal_label: np.asarray(inputs, dtype=np.float64)},
        sampled_outcome={channel.label: samples},
        trajectory_hash=trajectory_hash,
    )


def _build_metadata(
    *,
    upstream: TrajectoryResult | None,
    provenance_tags: tuple[str, ...],
) -> ResultMetadata:
    """Synthesize measurement-result metadata.

    When an upstream trajectory is supplied, the measurement inherits its
    convention/backend/truncation context and appends the measurement
    tag. Free-standing measurements get a minimal self-declared metadata
    tagged ``"measurement"`` so provenance is still searchable.
    """
    from ..conventions import CONVENTION_VERSION

    if upstream is not None:
        upstream_meta = upstream.metadata
        return ResultMetadata(
            convention_version=upstream_meta.convention_version,
            request_hash=upstream_meta.request_hash,
            backend_name=upstream_meta.backend_name,
            backend_version=upstream_meta.backend_version,
            storage_mode=StorageMode.OMITTED,
            fock_truncations=upstream_meta.fock_truncations,
            provenance_tags=(*upstream_meta.provenance_tags, "measurement", *provenance_tags),
        )
    return ResultMetadata(
        convention_version=CONVENTION_VERSION,
        request_hash="0" * 64,
        backend_name="iontrap-dynamics.measurement",
        backend_version=_LIBRARY_VERSION,
        storage_mode=StorageMode.OMITTED,
        provenance_tags=("measurement", *provenance_tags),
    )


__all__ = [
    "BernoulliChannel",
    "BinomialChannel",
    "PoissonChannel",
    "sample_outcome",
]
