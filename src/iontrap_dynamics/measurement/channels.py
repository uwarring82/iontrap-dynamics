# SPDX-License-Identifier: MIT
"""Sampling channels for the measurement layer.

A *channel* maps noise-free probability inputs to per-shot stochastic
outcomes. Channels are intentionally thin: they do not know about
operators, states, or detector hardware. Composition with detector
models (Dispatch K) and with observable-to-probability reductions
(Dispatches L–N) happens a layer up.

Dispatch H ships :class:`BernoulliChannel` — each input probability
``p ∈ [0, 1]`` produces ``shots`` independent bits in ``{0, 1}`` with
``P(1) = p``. The per-shot shape is kept (not pre-aggregated to counts)
so that downstream protocol code can build parity estimators and
shot-shot correlations without re-sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..results import (
    MeasurementResult,
    ResultMetadata,
    StorageMode,
    TrajectoryResult,
)


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


def sample_outcome(
    *,
    channel: BernoulliChannel,
    probabilities: NDArray[np.floating],
    shots: int,
    seed: int | None = None,
    upstream: TrajectoryResult | None = None,
    provenance_tags: tuple[str, ...] = (),
) -> MeasurementResult:
    """Apply ``channel`` to ``probabilities`` and wrap the output.

    The thin orchestrator that Dispatches I–N will grow into a registry-
    style dispatch on channel type. For Dispatch H it handles the single
    :class:`BernoulliChannel` case.

    Parameters
    ----------
    channel
        Sampling channel to apply. Its ``label`` names the entry in
        :attr:`MeasurementResult.sampled_outcome`.
    probabilities
        1-D array of input probabilities. Typically derived from an
        expectation-value trajectory (e.g.
        ``p_up = (1 + sigma_z_traj) / 2``).
    shots
        Number of independent shots per probability entry.
    seed
        Optional seed for :func:`numpy.random.default_rng`. When
        supplied, the resulting :class:`MeasurementResult` is fully
        reproducible: rerunning with the same ``(seed, probabilities,
        shots)`` reproduces the sampled bits exactly.
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
        With ``ideal_outcome = {"probability": probabilities}``,
        ``sampled_outcome = {channel.label: bits}``, and
        ``storage_mode = OMITTED`` in its inherited metadata.
    """
    rng = np.random.default_rng(seed)
    bits = channel.sample(probabilities, shots=shots, rng=rng)

    metadata = _build_metadata(upstream=upstream, provenance_tags=provenance_tags)
    trajectory_hash = upstream.metadata.request_hash if upstream is not None else None

    return MeasurementResult(
        metadata=metadata,
        shots=shots,
        rng_seed=seed,
        ideal_outcome={"probability": np.asarray(probabilities, dtype=np.float64)},
        sampled_outcome={channel.label: bits},
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
        backend_version="0.1.0.dev0",
        storage_mode=StorageMode.OMITTED,
        provenance_tags=("measurement", *provenance_tags),
    )


__all__ = [
    "BernoulliChannel",
    "sample_outcome",
]
