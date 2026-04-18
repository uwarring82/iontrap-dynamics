# SPDX-License-Identifier: MIT
"""Solver dispatcher — combines a Hamiltonian builder's output with a
state, a time grid, and an observable list, and returns a populated
:class:`TrajectoryResult`.

This is the public entry point users call for a full simulation. The
library's internals are stitched together here:

    configuration (species, drives, modes, system)
           |
           v
    HilbertSpace  →  operators  →  Hamiltonian
                                      |
                                      v
             initial_state  +  qutip.mesolve  →  states
                                                     |
                                                     v
                                       observables  →  expectations
                                                          |
                                                          v
                                                 TrajectoryResult

Design
------

v0.1 wraps QuTiP's :func:`qutip.mesolve` directly. The backend name and
version are recorded in the result's metadata so downstream caches can
detect backend drift. Time-dependent Hamiltonians (QuTiP list format)
are accepted by the signature but the first builders to use them
(detuned carrier, near-sideband) land in follow-on dispatches.

Storage modes (CONVENTIONS.md §0.E)
-----------------------------------

- ``StorageMode.OMITTED`` (default) — states are discarded after
  expectations are computed; only the expectation dict and metadata
  survive. Best for long trajectories where keeping N_steps × Qobj in
  memory is expensive.
- ``StorageMode.EAGER`` — all mesolve output states are packed into a
  tuple and attached to :attr:`TrajectoryResult.states`. Use when
  downstream code needs per-time density matrices (Wigner plots,
  entanglement-of-formation, partial traces).
- ``StorageMode.LAZY`` — not supported from :func:`solve` in v0.1
  because mesolve eagerly materialises all states anyway; there's no
  per-call wins for lazy access here. Callers who want a
  lazy-evaluated :class:`TrajectoryResult` can construct one manually
  by wrapping a callable around an already-materialised state list.

Warnings
--------

The Fock-truncation convergence check (CONVENTIONS.md §13) runs on
every ``solve()`` call. For each mode, the top-level population
``p_top = max_t ⟨N_Fock−1|ρ_m(t)|N_Fock−1⟩`` is classified against
the tolerance ε (``conventions.FOCK_CONVERGENCE_TOLERANCE`` by
default, overridable per-call via the ``fock_tolerance`` argument):

- ``p_top < ε/10``     → OK, silent.
- ``ε/10 ≤ p_top < ε`` → :class:`FockConvergenceWarning` (Level 1).
- ``ε ≤ p_top < 10·ε`` → :class:`FockQualityWarning` (Level 2).
- ``p_top ≥ 10·ε``     → :class:`ConvergenceError` (Level 3, raised).

Warnings are emitted to both the Python ``warnings`` channel and
recorded as :class:`ResultWarning` entries on the returned
:class:`TrajectoryResult` — so downstream code that filters or
silences library warnings still has the diagnostic record
available via ``result.warnings``.
"""

from __future__ import annotations

import warnings as _warnings
from collections.abc import Iterable, Sequence
from typing import TypeAlias

import numpy as np
import qutip

from .conventions import FOCK_CONVERGENCE_TOLERANCE
from .exceptions import (
    ConventionError,
    ConvergenceError,
    FockConvergenceWarning,
    FockQualityWarning,
)
from .hilbert import HilbertSpace
from .observables import Observable, expectations_over_time
from .results import (
    ResultMetadata,
    ResultWarning,
    StorageMode,
    TrajectoryResult,
    WarningSeverity,
)

# A Hamiltonian for mesolve can be a single Qobj (time-independent) or a
# list in QuTiP's time-dependent format. The list holds either Qobj's
# (constant pieces) or [Qobj, callable] / [Qobj, ndarray] pairs
# (time-dependent pieces) — we type it loosely to match mesolve's
# polymorphism without pretending to model each variant.
MesolveHamiltonian: TypeAlias = qutip.Qobj | list[object]


def _fock_saturation_warnings(
    hilbert: HilbertSpace,
    states: Iterable[qutip.Qobj],
    tolerance: float,
) -> tuple[ResultWarning, ...]:
    """Classify Fock-truncation saturation per mode and return the warning
    records (emitting Python warnings along the way).

    Raises :class:`ConvergenceError` on any Level 3 failure — per
    CONVENTIONS.md §13, a single mode exceeding ``10·ε`` is a hard
    failure for the whole run, not a per-mode warning.
    """
    if tolerance <= 0.0:
        raise ConventionError(
            f"fock_tolerance must be positive; got {tolerance!r}. "
            "Passing zero disables the convergence check entirely, which "
            "is a silent-degradation hazard (CONVENTIONS.md §15). "
            "Use a large positive tolerance (e.g. 1.0) if a truly "
            "suppressed check is required."
        )

    # Materialise once — we iterate through the trajectory twice below for
    # clarity, but want consistent views from both passes.
    states_list = list(states)

    records: list[ResultWarning] = []
    level3_failures: list[tuple[str, float]] = []

    for mode in hilbert.system.modes:
        fock_dim = hilbert.fock_truncations[mode.label]
        top_level_single = qutip.basis(fock_dim, fock_dim - 1).proj()
        top_level_embedded = hilbert.mode_op_for(top_level_single, mode.label)

        p_top_max = 0.0
        for state in states_list:
            # qutip.expect returns a complex for general operators; the
            # projector is Hermitian and positive so the real part is the
            # physical value. Clamp at ≥0 so tiny negative roundoff
            # (~1e-18) never flips the classification.
            p = float(qutip.expect(top_level_embedded, state).real)
            if p > p_top_max:
                p_top_max = p

        diagnostics = {
            "mode_label": mode.label,
            "fock_dim": fock_dim,
            "p_top_max": p_top_max,
            "tolerance_epsilon": tolerance,
        }

        if p_top_max < tolerance / 10.0:
            continue  # OK, silent
        if p_top_max >= 10.0 * tolerance:
            level3_failures.append((mode.label, p_top_max))
            continue
        if p_top_max >= tolerance:
            message = (
                f"mode {mode.label!r}: top-Fock population p_top = "
                f"{p_top_max:.3e} exceeds ε = {tolerance:.3e} "
                f"(N_Fock = {fock_dim}); quality degraded (CONVENTIONS.md §15 "
                "Level 2). Consult result.warnings before publication use."
            )
            _warnings.warn(message, FockQualityWarning, stacklevel=3)
            records.append(
                ResultWarning(
                    severity=WarningSeverity.QUALITY,
                    category="fock_truncation",
                    message=message,
                    diagnostics=diagnostics,
                )
            )
        else:
            message = (
                f"mode {mode.label!r}: top-Fock population p_top = "
                f"{p_top_max:.3e} approaches ε = {tolerance:.3e} "
                f"(N_Fock = {fock_dim}); solver converged but the "
                "truncation is close to its envelope (CONVENTIONS.md §15 "
                "Level 1). Consider tightening fock_truncations for "
                "publication-grade results."
            )
            _warnings.warn(message, FockConvergenceWarning, stacklevel=3)
            records.append(
                ResultWarning(
                    severity=WarningSeverity.CONVERGENCE,
                    category="fock_truncation",
                    message=message,
                    diagnostics=diagnostics,
                )
            )

    if level3_failures:
        summary = ", ".join(
            f"{label}: p_top = {p_top_max:.3e}" for label, p_top_max in level3_failures
        )
        raise ConvergenceError(
            "Fock-truncation failure (CONVENTIONS.md §13, §15 Level 3): "
            f"top-level populations meet or exceed 10·ε = {10.0 * tolerance:.3e} "
            f"for one or more modes [{summary}]. Increase fock_truncations "
            "for the affected mode(s) and re-run."
        )

    return tuple(records)


def solve(
    *,
    hilbert: HilbertSpace,
    hamiltonian: MesolveHamiltonian,
    initial_state: qutip.Qobj,
    times: np.ndarray,
    observables: Sequence[Observable] = (),
    request_hash: str = "",
    backend_name: str = "qutip-mesolve",
    storage_mode: StorageMode = StorageMode.OMITTED,
    provenance_tags: tuple[str, ...] = (),
    fock_tolerance: float | None = None,
) -> TrajectoryResult:
    """Run the Lindblad solver and wrap the output as a :class:`TrajectoryResult`.

    Parameters
    ----------
    hilbert
        The :class:`HilbertSpace` the simulation lives on. Its
        ``system.convention_version`` and ``fock_truncations`` are
        recorded on the result's metadata.
    hamiltonian
        Either a time-independent :class:`qutip.Qobj` or QuTiP's
        time-dependent list format. Dims must match
        :meth:`HilbertSpace.qutip_dims`.
    initial_state
        The initial ket or density matrix. Dims must match the
        Hamiltonian and the Hilbert space.
    times
        1-D array of evaluation times in SI seconds (CONVENTIONS.md §1).
        Must be monotonically non-decreasing; not validated here — QuTiP
        raises on pathological inputs.
    observables
        Sequence of :class:`Observable` records. Defaults to ``()``
        (expectation dict will be empty).
    request_hash
        Caller-supplied SHA-256 hex of the canonical parameter set —
        typically produced via
        :func:`iontrap_dynamics.cache.compute_request_hash`. Baked into
        the metadata so cached results can round-trip through
        :func:`iontrap_dynamics.cache.load_trajectory`.
    backend_name
        String tag for the solver. Default ``"qutip-mesolve"``; callers
        using a different backend should override.
    storage_mode
        :class:`StorageMode` for the returned result. ``OMITTED`` by
        default; ``EAGER`` attaches all states as a tuple;
        ``LAZY`` raises (see module docstring).
    provenance_tags
        Tuple of free-form tags for the metadata (e.g. ``("ci",
        "phase1_test")``). Not interpreted by the library.
    fock_tolerance
        Override ε for the CONVENTIONS.md §13 Fock-truncation
        convergence check. ``None`` (default) reads
        :data:`iontrap_dynamics.conventions.FOCK_CONVERGENCE_TOLERANCE`
        (``1e-4``). See the module docstring for the full status ladder.

    Returns
    -------
    TrajectoryResult
        Frozen result with ``times``, ``expectations``, ``metadata``,
        ``warnings`` (empty tuple when all modes are well converged, or
        a tuple of :class:`ResultWarning` records for Level 1/2
        anomalies), and ``states``/``states_loader`` per
        ``storage_mode``.

    Raises
    ------
    ConventionError
        If ``storage_mode`` is :attr:`StorageMode.LAZY` — mesolve
        eagerly materialises states so a lazy wrapper is not
        meaningful here. Also raised if ``fock_tolerance`` is
        non-positive.
    ConvergenceError
        If any mode's top-Fock population meets or exceeds ``10·ε``
        during the trajectory (CONVENTIONS.md §15 Level 3). Increase
        ``fock_truncations`` for the affected mode and re-run.
    """
    if storage_mode is StorageMode.LAZY:
        raise ConventionError(
            "sequences.solve does not support StorageMode.LAZY in v0.1: "
            "qutip.mesolve materialises all states eagerly, so a lazy loader "
            "buys nothing here. If you need lazy access for a specific "
            "workflow, construct TrajectoryResult manually with a callable "
            "wrapping the already-materialised state list."
        )

    time_array = np.asarray(times, dtype=np.float64)

    # e_ops kwarg is explicit to avoid QuTiP 5.3's upcoming keyword-only
    # enforcement (see solver_base.py:598 FutureWarning). We pass empty
    # e_ops because observables are computed downstream from stored states,
    # which lets the caller mix OMITTED/EAGER modes without re-solving.
    solver_result = qutip.mesolve(
        hamiltonian,
        initial_state,
        time_array,
        c_ops=[],
        e_ops=[],
    )

    expectations = expectations_over_time(solver_result.states, observables)

    # Fock-truncation convergence check (CONVENTIONS.md §13, §15). Raises
    # ConvergenceError on Level 3 failure; otherwise returns per-mode
    # Level 1/2 records (empty tuple when all modes are well converged).
    effective_tolerance = FOCK_CONVERGENCE_TOLERANCE if fock_tolerance is None else fock_tolerance
    warning_records = _fock_saturation_warnings(hilbert, solver_result.states, effective_tolerance)

    metadata = ResultMetadata(
        convention_version=hilbert.system.convention_version,
        request_hash=request_hash,
        backend_name=backend_name,
        backend_version=qutip.__version__,
        storage_mode=storage_mode,
        fock_truncations=dict(hilbert.fock_truncations),
        provenance_tags=provenance_tags,
    )

    states: tuple[qutip.Qobj, ...] | None = (
        tuple(solver_result.states) if storage_mode is StorageMode.EAGER else None
    )

    return TrajectoryResult(
        metadata=metadata,
        times=time_array,
        expectations=expectations,
        warnings=warning_records,
        states=states,
        states_loader=None,
    )


__all__ = [
    "solve",
]
