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

v0.1 returns ``warnings=()``. A follow-on dispatch will add the
CONVENTIONS.md §13 Fock-truncation check (monitor the top-level Fock
population against the tolerance ε and emit a
:class:`ConvergenceWarning` if it enters the marginal or degraded
regime). Until then, the warnings field is empty; callers who need
saturation awareness should inspect the final state's top-level
occupation directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import qutip

from .exceptions import ConventionError
from .hilbert import HilbertSpace
from .observables import Observable, expectations_over_time
from .results import ResultMetadata, StorageMode, TrajectoryResult

# A Hamiltonian for mesolve can be a single Qobj (time-independent) or a
# list in QuTiP's time-dependent format. The list holds either Qobj's
# (constant pieces) or [Qobj, callable] / [Qobj, ndarray] pairs
# (time-dependent pieces) — we type it loosely to match mesolve's
# polymorphism without pretending to model each variant.
MesolveHamiltonian: TypeAlias = qutip.Qobj | list[object]


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

    Returns
    -------
    TrajectoryResult
        Frozen result with ``times``, ``expectations``, ``metadata``,
        ``warnings=()``, and ``states``/``states_loader`` per
        ``storage_mode``.

    Raises
    ------
    ConventionError
        If ``storage_mode`` is :attr:`StorageMode.LAZY` — mesolve
        eagerly materialises states so a lazy wrapper is not
        meaningful here.
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
        warnings=(),
        states=states,
        states_loader=None,
    )


__all__ = [
    "solve",
]
