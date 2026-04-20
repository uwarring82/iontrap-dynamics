# SPDX-License-Identifier: MIT
"""Canonical result schema for :mod:`iontrap_dynamics`.

Defines the output contract that every solver in the library produces and
every analysis consumes. Locked in Phase 0 per ``WORKPLAN_v0.3.md`` §0.E and
``CONVENTIONS.md`` §15; schema changes after v0.1 require a Convention Freeze.

Design
------

The module exposes an abstract :class:`Result` base and a concrete
:class:`TrajectoryResult` for deterministic (pure-state / Lindblad) evolution.
Phase 1 siblings — stochastic-trajectory results, measurement outcomes — are
expected to subclass :class:`Result` so that downstream code can ``except
Result`` as a blanket and narrow with ``isinstance`` only when needed. This
"result family" shape was agreed as decision D5 during Phase 0 planning.

All result dataclasses are frozen, slotted, and keyword-only: positional
construction is forbidden per Design Principle 6, attributes cannot be
reassigned after construction, and the memory footprint is tight.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .exceptions import ConventionError

# Quantum-state objects are backend-specific (QuTiP ``Qobj`` in Phase 0;
# backend-agnostic in Phase 2+). Declaring ``State`` as :data:`typing.Any`
# keeps the schema portable without pretending to type-check across backends.
State = Any


# ----------------------------------------------------------------------------
# Storage policy
# ----------------------------------------------------------------------------


class StorageMode(StrEnum):
    """How a result retains quantum states along its independent axis.

    The mode is declared at solver entry and frozen into
    :attr:`ResultMetadata.storage_mode`. Consumers MUST check this field
    before accessing :attr:`TrajectoryResult.states`.

    * ``EAGER`` — every state is retained; :attr:`TrajectoryResult.states`
      is a populated tuple of length ``len(times)``.
    * ``LAZY`` — states are fetched on demand via
      :attr:`TrajectoryResult.states_loader`; the ``states`` attribute is
      ``None``.
    * ``OMITTED`` — states are discarded after expectation computation;
      both ``states`` and ``states_loader`` are ``None``. Valid when the
      caller explicitly requests expectation-only output.
    """

    EAGER = "eager"
    LAZY = "lazy"
    OMITTED = "omitted"


# ----------------------------------------------------------------------------
# Warnings
# ----------------------------------------------------------------------------


class WarningSeverity(StrEnum):
    """Severity ladder matching :doc:`CONVENTIONS.md` §15.

    Only Level 1 (``CONVERGENCE``) and Level 2 (``QUALITY``) warnings are
    recorded here; Level 3 hard failures raise typed exceptions and never
    reach a result object.
    """

    CONVERGENCE = "convergence"
    QUALITY = "quality"


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultWarning:
    """Structured warning record attached to a :class:`Result`.

    Emitted to both the Python :mod:`warnings` channel and the result's
    :attr:`Result.warnings` tuple. Silent degradation is forbidden
    (CONVENTIONS.md §15): every anomaly must produce either a warning
    record here or a typed exception.
    """

    severity: WarningSeverity
    category: str
    message: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultMetadata:
    """Provenance and context recorded for every result.

    All fields are populated at solver entry. Downstream analysis code may
    rely on them being present; missing or placeholder values should be
    expressed as explicit empty values (e.g. ``()``, ``{}``), not ``None``.
    """

    # --- convention and reproducibility binding ---
    convention_version: str
    request_hash: str

    # --- backend identification ---
    backend_name: str
    backend_version: str

    # --- storage declaration (consumers check this before indexing states) ---
    storage_mode: StorageMode

    # --- truncation choices: mode-label → Fock cutoff used ---
    fock_truncations: Mapping[str, int] = field(default_factory=dict)

    # --- free-form provenance tags: e.g. ("notebook", "sweep", "ci") ---
    provenance_tags: tuple[str, ...] = ()


# ----------------------------------------------------------------------------
# Result base + TrajectoryResult
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Result:
    """Abstract base for result objects produced by :mod:`iontrap_dynamics`.

    Phase 0 concrete realisation: :class:`TrajectoryResult`.
    Phase 1+ planned siblings: ``StochasticTrajectoryResult``,
    ``MeasurementResult``.
    """

    metadata: ResultMetadata
    warnings: tuple[ResultWarning, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectoryResult(Result):
    """Deterministic trajectory of a quantum state along a time axis.

    Canonical output of every closed-system and Lindblad-master-equation
    solver in the library. For stochastic unravellings, use the (Phase 1+)
    ``StochasticTrajectoryResult``.

    Parameters
    ----------
    times
        1-D array of evaluation times, in seconds (internal SI; see
        CONVENTIONS.md §1). Monotonically non-decreasing; not validated at
        construction time (adaptive solvers may legitimately produce
        near-duplicates around stiff regions).
    expectations
        Mapping from observable label to a 1-D array of length ``len(times)``.
        Observable labels follow the registry in ``observables.py`` (Phase 1).
    states
        Tuple of quantum-state objects, one per time, when
        :attr:`ResultMetadata.storage_mode` is :attr:`StorageMode.EAGER`.
        ``None`` for :attr:`StorageMode.LAZY` and :attr:`StorageMode.OMITTED`.
    states_loader
        Callable ``loader(i) -> State`` that retrieves the state at index
        ``i`` on demand, when :attr:`StorageMode.LAZY`. ``None`` otherwise.

    Raises
    ------
    ConventionError
        At construction, if the combination of ``states`` and
        ``states_loader`` is inconsistent with
        :attr:`ResultMetadata.storage_mode`.
    """

    times: NDArray[np.floating]
    expectations: Mapping[str, NDArray[np.floating]] = field(default_factory=dict)
    states: tuple[State, ...] | None = None
    states_loader: Callable[[int], State] | None = None

    def __post_init__(self) -> None:
        """Enforce storage-policy consistency (CONVENTIONS.md §0.E)."""
        mode = self.metadata.storage_mode
        have_states = self.states is not None
        have_loader = self.states_loader is not None

        if mode is StorageMode.EAGER:
            if not have_states:
                raise ConventionError(
                    "storage_mode=EAGER requires a populated `states` tuple; got states=None."
                )
            if have_loader:
                raise ConventionError(
                    "storage_mode=EAGER forbids `states_loader`; "
                    "exactly one of (states, states_loader) must be set."
                )
        elif mode is StorageMode.LAZY:
            if have_states:
                raise ConventionError(
                    "storage_mode=LAZY forbids a materialised `states` tuple; "
                    "set states=None and supply `states_loader`."
                )
            if not have_loader:
                raise ConventionError("storage_mode=LAZY requires a `states_loader` callable.")
        else:  # OMITTED
            if have_states or have_loader:
                raise ConventionError(
                    "storage_mode=OMITTED requires both `states` and `states_loader` to be None."
                )


# ----------------------------------------------------------------------------
# MeasurementResult (Phase 1 sibling of TrajectoryResult)
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementResult(Result):
    """Outcome of applying a measurement channel to a state or trajectory.

    The dual-view shape mandated by ``WORKPLAN_v0.3.md`` §5 (Phase 1):
    :attr:`ideal_outcome` carries the noise-free expectation values the
    channel was applied to; :attr:`sampled_outcome` carries the per-shot
    stochastic result of the channel. Downstream statistics code consumes
    the sampled view; analytic-limit checks compare to the ideal view.

    Parameters
    ----------
    shots
        Number of independent measurement shots per time / setting entry.
        Must be ≥ 1.
    rng_seed
        Seed that was passed to :func:`numpy.random.default_rng` to
        produce the sampled outcome. ``None`` when the caller supplied a
        pre-seeded generator and the seed is therefore unknown.
    ideal_outcome
        Mapping from a channel-dependent label to the noise-free input
        (e.g. ``"probability"`` for Bernoulli; shape matches the upstream
        expectation array).
    sampled_outcome
        Mapping from a channel-dependent label to the stochastic output
        (e.g. ``"counts"`` for Bernoulli; shape depends on channel — per-
        shot bits give ``(shots, n_times)`` integer arrays).
    trajectory_hash
        Request-hash of the upstream :class:`TrajectoryResult` when one
        exists. ``None`` when the measurement was applied to a free-
        standing probability array without an owning trajectory.

    Raises
    ------
    ConventionError
        At construction, if ``shots < 1`` or if
        :attr:`ResultMetadata.storage_mode` is not
        :attr:`StorageMode.OMITTED`. Measurement results do not retain
        quantum states; the storage-mode field on inherited metadata
        must reflect that.
    """

    shots: int
    rng_seed: int | None
    ideal_outcome: Mapping[str, NDArray[np.floating]] = field(default_factory=dict)
    sampled_outcome: Mapping[str, NDArray[Any]] = field(default_factory=dict)
    trajectory_hash: str | None = None

    def __post_init__(self) -> None:
        if self.shots < 1:
            raise ConventionError(f"MeasurementResult requires shots >= 1; got shots={self.shots}.")
        if self.metadata.storage_mode is not StorageMode.OMITTED:
            raise ConventionError(
                "MeasurementResult requires metadata.storage_mode=OMITTED; "
                "measurement outcomes do not retain quantum states."
            )


__all__ = [
    "MeasurementResult",
    "Result",
    "ResultMetadata",
    "ResultWarning",
    "State",
    "StorageMode",
    "TrajectoryResult",
    "WarningSeverity",
]
