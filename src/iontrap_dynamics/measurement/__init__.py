# SPDX-License-Identifier: MIT
"""Measurement layer (Phase 1, v0.2 track — staged, not frozen).

This subpackage hosts the hard boundary between dynamics and observation
named in ``WORKPLAN_v0.3.md`` §3 (Architectural skeleton) and §5 Phase 1
(Core physics and measurement layer). The layer is ring-fenced into four
surfaces — channels, detectors, protocols, statistics — each shipping in
its own dispatch so the boundary can be exercised before it hardens.

Dispatch H lands :class:`BernoulliChannel` and the minimal
:func:`sample_outcome` orchestrator; Dispatch J adds
:class:`BinomialChannel`. Poisson channels, detector models, protocols,
and statistics follow in Dispatches K–P. The ``CONVENTIONS.md`` §17
section is opened here as staged rules and will freeze at the close of
the track.
"""

from __future__ import annotations

from .channels import BernoulliChannel, BinomialChannel, sample_outcome

__all__ = [
    "BernoulliChannel",
    "BinomialChannel",
    "sample_outcome",
]
