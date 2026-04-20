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
calls either side of the channel. Protocols and statistics follow in
Dispatches M–P. The ``CONVENTIONS.md`` §17 section is opened here as
staged rules and will freeze at the close of the track.
"""

from __future__ import annotations

from .channels import (
    BernoulliChannel,
    BinomialChannel,
    PoissonChannel,
    sample_outcome,
)
from .detectors import DetectorConfig

__all__ = [
    "BernoulliChannel",
    "BinomialChannel",
    "DetectorConfig",
    "PoissonChannel",
    "sample_outcome",
]
