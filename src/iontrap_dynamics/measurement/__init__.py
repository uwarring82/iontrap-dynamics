# SPDX-License-Identifier: MIT
"""Measurement layer (Phase 1, v0.2 track — staged, not frozen).

This subpackage hosts the hard boundary between dynamics and observation
named in ``WORKPLAN_v0.3.md`` §3 (Architectural skeleton) and §5 Phase 1
(Core physics and measurement layer). The layer is ring-fenced into four
surfaces — channels, detectors, protocols, statistics — each shipping in
its own dispatch so the boundary can be exercised before it hardens.

Dispatch H lands :class:`BernoulliChannel` and the minimal
:func:`sample_outcome` orchestrator; Dispatch J adds
:class:`BinomialChannel`; Dispatch K adds :class:`PoissonChannel` and
renames the orchestrator keyword ``probabilities`` → ``inputs`` to
reflect that Poisson consumes rates, not probabilities; Dispatch L
adds :class:`DetectorConfig` — efficiency / dark-count rate /
threshold — composing with rate-consuming channels via explicit
:meth:`DetectorConfig.apply` (rate transform) and
:meth:`DetectorConfig.discriminate` (bright / dark thresholding)
calls either side of the channel; Dispatch M adds the first
protocol, :class:`SpinReadout`, which wraps projective state
projection + Poisson photon counting + thresholding into a named
``.run(trajectory, shots, seed)`` call; Dispatch N adds
:class:`ParityScan`, which reads two ions on the same shot and
reconstructs the joint readout distribution from
``⟨σ_z^i⟩``, ``⟨σ_z^j⟩``, and ``⟨σ_z^i σ_z^j⟩`` so entanglement-
bearing correlations (Bell-state verification, CHSH) survive.
Dispatch O adds :class:`SidebandInference`, which composes paired
red / blue sideband readouts through the short-time
``n̄ = r / (1 − r)`` ratio to report a motional-occupation estimate.
Dispatch P closes the track with confidence-interval statistics —
:func:`wilson_interval` and :func:`clopper_pearson_interval` on
binomial counts, plus the :func:`binomial_summary` /
:class:`BinomialSummary` convenience wrapper. The ``CONVENTIONS.md``
§17 section is opened here as staged rules and freezes at the close
of the track.
"""

from __future__ import annotations

from .channels import (
    BernoulliChannel,
    BinomialChannel,
    PoissonChannel,
    sample_outcome,
)
from .detectors import DetectorConfig
from .protocols import ParityScan, SidebandInference, SpinReadout
from .statistics import (
    BinomialSummary,
    binomial_summary,
    clopper_pearson_interval,
    wilson_interval,
)

__all__ = [
    "BernoulliChannel",
    "BinomialChannel",
    "BinomialSummary",
    "DetectorConfig",
    "ParityScan",
    "PoissonChannel",
    "SidebandInference",
    "SpinReadout",
    "binomial_summary",
    "clopper_pearson_interval",
    "sample_outcome",
    "wilson_interval",
]
