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
             initial_state  +  qutip.{se,me}solve  →  states
                                                         |
                                                         v
                                       observables  →  expectations
                                                          |
                                                          v
                                                 TrajectoryResult

Design
------

``solve()`` dispatches to :func:`qutip.sesolve` or :func:`qutip.mesolve`
based on the initial-state type. Kets take the Schrödinger-equation
fast path (``sesolve``); density matrices take the master-equation path
(``mesolve``). The chosen solver is recorded in the result's
``backend_name`` metadata as ``"qutip-sesolve"`` or ``"qutip-mesolve"``
so downstream caches and analysis can distinguish them. Callers can
override via ``solver="sesolve" | "mesolve" | "auto"`` — ``"sesolve"``
on a density matrix raises :class:`ConventionError` because the
Schrödinger equation only evolves pure states.

The sesolve dispatch was opened as Phase 2 / v0.3 Dispatch X. On
QuTiP 5.2 at the Hilbert-space sizes this library routinely uses
(dim ≤ 48), the two paths run at comparable speed — the sesolve
advantage that is folklore from QuTiP 4.x era has largely been
closed. Nonetheless, dispatching sesolve on pure kets is
*semantically* cleaner — the Schrödinger equation is the correct
dynamics for pure states — and leaves headroom for larger Hilbert
spaces (hundreds of dimensions, two-ion MS gates with large Fock
truncations) where the density-matrix lifting cost grows as N²
and the sesolve path genuinely pulls ahead. The baseline is
recorded by ``tools/run_benchmark_sesolve_speedup.py`` so Phase 2
follow-ons (sparse ops, JAX) measure against a fixed starting
point.

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
    backend_name: str | None = None,
    storage_mode: StorageMode = StorageMode.OMITTED,
    provenance_tags: tuple[str, ...] = (),
    fock_tolerance: float | None = None,
    solver: str = "auto",
    backend: str = "qutip",
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
        String tag for the solver recorded on result metadata. Default
        ``None`` auto-selects based on which QuTiP solver ran —
        ``"qutip-sesolve"`` for the Schrödinger-equation fast path (pure
        kets) or ``"qutip-mesolve"`` for the master-equation path
        (density matrices). Callers using a different backend or
        wanting a custom tag should override.
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
    solver
        Which QuTiP solver to dispatch to. ``"auto"`` (default) picks
        :func:`qutip.sesolve` when ``initial_state`` is a ket and
        :func:`qutip.mesolve` otherwise — the sesolve path is
        2–3× faster for pure-state dynamics and returns numerically
        identical expectations. ``"sesolve"`` forces the
        Schrödinger-equation path and raises
        :class:`ConventionError` when fed a density matrix.
        ``"mesolve"`` forces the master-equation path even on a ket
        (for v0.2 backwards compatibility or to share a code path
        with mixed-state trajectories); kets are internally promoted
        to density matrices by QuTiP. **QuTiP-specific**:
        ``"sesolve"``/``"mesolve"`` are QuTiP solver identifiers and
        are only valid with ``backend="qutip"`` (the default).
        Passing them with ``backend="jax"`` raises
        :class:`ConventionError`; the JAX backend infers
        Schrödinger-vs-Lindblad from the input dtype exactly like
        ``solver="auto"`` does on QuTiP.
    backend
        Which solver backend to use. ``"qutip"`` (default) dispatches
        to the QuTiP reference backend described above. ``"jax"``
        opts into the JAX / Dynamiqs backend (Phase 2 deliverable —
        Dispatch β.1 ships the skeleton + availability check; the
        Dynamiqs integrator wiring is scoped for Dispatch β.2 per
        ``docs/phase-2-jax-backend-design.md`` §7). Requires the
        ``[jax]`` optional dependencies
        (``pip install iontrap-dynamics[jax]``); if those are not
        installed, a :class:`BackendError` is raised with an
        install hint. Unknown backend strings raise
        :class:`ConventionError`.

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
    _validate_backend(backend, solver)

    if backend == "jax":
        from .backends.jax import solve_via_jax

        return solve_via_jax(
            hilbert=hilbert,
            hamiltonian=hamiltonian,
            initial_state=initial_state,
            times=times,
            observables=observables,
            request_hash=request_hash,
            backend_name=backend_name,
            storage_mode=storage_mode,
            provenance_tags=provenance_tags,
            fock_tolerance=fock_tolerance,
            solver=solver,
        )

    if storage_mode is StorageMode.LAZY:
        raise ConventionError(
            "sequences.solve does not support StorageMode.LAZY in v0.1: "
            "qutip.mesolve materialises all states eagerly, so a lazy loader "
            "buys nothing here. If you need lazy access for a specific "
            "workflow, construct TrajectoryResult manually with a callable "
            "wrapping the already-materialised state list."
        )

    # Pick the solver path. sesolve (Schrödinger) is 2–3× faster than
    # mesolve on pure kets because it avoids lifting to the density-matrix
    # representation. mesolve remains the fallback for density-matrix
    # inputs (SPAM-prep trajectories, thermally-mixed initial states).
    selected_solver = _choose_solver(solver, initial_state)

    time_array = np.asarray(times, dtype=np.float64)

    # e_ops kwarg is explicit to avoid QuTiP 5.3's upcoming keyword-only
    # enforcement (see solver_base.py:598 FutureWarning). We pass empty
    # e_ops because observables are computed downstream from stored states,
    # which lets the caller mix OMITTED/EAGER modes without re-solving.
    if selected_solver == "sesolve":
        solver_result = qutip.sesolve(
            hamiltonian,
            initial_state,
            time_array,
            e_ops=[],
        )
    else:
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
        backend_name=backend_name if backend_name is not None else f"qutip-{selected_solver}",
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


def solve_ensemble(
    *,
    hilbert: HilbertSpace,
    hamiltonians: Sequence[MesolveHamiltonian],
    initial_state: qutip.Qobj,
    times: np.ndarray,
    observables: Sequence[Observable] = (),
    request_hash: str = "",
    backend_name: str | None = None,
    storage_mode: StorageMode = StorageMode.OMITTED,
    provenance_tags: tuple[str, ...] = (),
    fock_tolerance: float | None = None,
    solver: str = "auto",
    backend: str = "qutip",
    n_jobs: int = 1,
    parallel_backend: str = "loky",
) -> tuple[TrajectoryResult, ...]:
    """Run ``solve()`` across a sequence of Hamiltonians (batch API, optional parallelism).

    The canonical aggregation pattern for jitter ensembles (§18.2):
    given ``N`` perturbed :class:`DriveConfig`\\s, build one Hamiltonian
    per config, then call this function to run all ``N`` trajectories
    and collect a tuple of :class:`TrajectoryResult`.

    Phase 2 Dispatch Y — workplan §5 "parallel sweeps via joblib". The
    function wraps :class:`joblib.Parallel` so callers can opt into
    process- or thread-based parallelism via ``n_jobs``.

    **Performance reality.** Measured crossover on QuTiP 5.2 +
    Python 3.13 (single-core laptop, see
    ``tools/run_benchmark_ensemble_parallel.py``):

    - single-solve < ~5 ms — serial wins; loky is ~20× slower
      (process-spawn + pickle overhead dominates).
    - single-solve ~5–10 ms — serial ≈ loky; noise-dominated.
    - single-solve > ~15 ms — loky pulls ahead; typical speedup
      2–3× at 20 trials on a 10-core machine.

    The default is therefore ``n_jobs=1`` (serial in the main
    process, zero joblib overhead). Callers hitting the large
    single-solve regime (MS gates with Fock > 24, two-ion full-LD
    builders, long-duration parameter scans) can flip to
    ``n_jobs=-1`` — benchmark before committing to a backend.

    Parameters
    ----------
    hilbert, initial_state, times, observables, request_hash,
    backend_name, storage_mode, provenance_tags, fock_tolerance,
    solver
        Shared across all trials — identical to :func:`solve`
        semantics. Metadata fields that would be per-trial (e.g. a
        varying seed recorded in provenance_tags) must be baked into
        the Hamiltonian or passed to ``solve()`` manually in a
        per-trial comprehension.
    hamiltonians
        One Hamiltonian per trial. The ensemble length is
        ``len(hamiltonians)``; zero-length is rejected.
    n_jobs
        Number of parallel workers. Default ``1`` runs serially in
        the main process with zero joblib overhead (recommended at
        the current library scales). ``-1`` uses all available CPUs;
        any positive integer pins to that many workers. Passed
        straight to :class:`joblib.Parallel`.
    parallel_backend
        Joblib backend — ``"loky"`` (default, process-based, works
        for CPU-bound numpy/QuTiP), ``"threading"`` (shares memory,
        GIL-limited for CPU-bound), or ``"sequential"`` (serial).
        See :mod:`joblib` docs for details.

    Returns
    -------
    tuple[TrajectoryResult, ...]
        One result per Hamiltonian, in the same order. Aggregation is
        the caller's responsibility — typically::

            results = solve_ensemble(hilbert=h, hamiltonians=hams, ...)
            stack = np.stack(
                [r.expectations["sigma_z_0"] for r in results], axis=0
            )
            ensemble_mean = stack.mean(axis=0)

    Raises
    ------
    ConventionError
        Propagated from any individual solve — e.g. invalid
        ``storage_mode``, ``solver``, or Level-3 Fock saturation on
        any trial. When the parallel backend is process-based, only
        the first failing trial's traceback is reported.

    Notes
    -----
    **Determinism.** Each :func:`solve` call is deterministic given
    its inputs, so the ensemble output is bit-reproducible regardless
    of worker scheduling — order of the output tuple matches the
    input ``hamiltonians`` order.

    **Serialization.** Process-based ``loky`` backend pickles
    everything (Hamiltonians, initial state, observable operators).
    QuTiP ``Qobj`` objects are pickleable. The pickling cost is
    non-trivial for large trajectories — for trivial solves (<10 ms)
    serial execution is typically faster overall.

    **Warnings.** Fock-saturation warnings emitted by worker
    processes do not propagate to the parent's ``warnings`` channel
    (joblib limitation), but they *are* recorded as
    :class:`ResultWarning` entries on each trial's
    :attr:`TrajectoryResult.warnings` tuple.
    """
    _validate_backend(backend, solver)

    from joblib import Parallel, delayed

    if len(hamiltonians) == 0:
        raise ConventionError(
            "solve_ensemble: hamiltonians must be non-empty; got an empty sequence."
        )

    solve_kwargs = {
        "hilbert": hilbert,
        "initial_state": initial_state,
        "times": times,
        "observables": tuple(observables),
        "request_hash": request_hash,
        "backend_name": backend_name,
        "storage_mode": storage_mode,
        "provenance_tags": provenance_tags,
        "fock_tolerance": fock_tolerance,
        "solver": solver,
        "backend": backend,
    }

    results = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
        delayed(solve)(hamiltonian=h, **solve_kwargs) for h in hamiltonians
    )
    return tuple(results)


_VALID_BACKENDS: frozenset[str] = frozenset({"qutip", "jax"})
_QUTIP_SOLVER_VALUES: frozenset[str] = frozenset({"auto", "sesolve", "mesolve"})


def _validate_backend(backend: str, solver: str) -> None:
    """Validate the ``backend=`` kwarg and its interaction with ``solver=``.

    Called at the top of :func:`solve` before any dispatch. Keeps the
    per-backend validation close to the dispatch so invalid kwarg
    combinations raise before any solver is touched.

    ``solver=`` has a stable QuTiP-specific vocabulary (CONVENTIONS.md
    §0.E and the ``solve`` docstring). On the JAX backend the QuTiP
    solver identifiers are semantically inapplicable — the JAX path
    infers Schrödinger-vs-Lindblad from the input dtype like
    ``solver="auto"`` does on QuTiP — so explicit ``"sesolve"`` /
    ``"mesolve"`` raises :class:`ConventionError`. Backend choice
    changes the implementation, not the public kwarg's semantics;
    see ``docs/phase-2-jax-backend-design.md`` §4.1.
    """
    if backend not in _VALID_BACKENDS:
        raise ConventionError(
            f"solve(backend={backend!r}): unknown backend; expected "
            f"one of {sorted(_VALID_BACKENDS)!r}."
        )
    if backend == "jax" and solver != "auto":
        raise ConventionError(
            f"solve(backend='jax', solver={solver!r}): the "
            f"solver= kwarg carries QuTiP-specific identifiers "
            f"('sesolve' / 'mesolve') that do not apply on the "
            f"JAX backend. Use solver='auto' (default) — the JAX "
            f"path infers Schrödinger-vs-Lindblad from the input "
            f"dtype. See docs/phase-2-jax-backend-design.md §4.1."
        )


def _choose_solver(solver: str, initial_state: qutip.Qobj) -> str:
    """Pick ``"sesolve"`` or ``"mesolve"`` from the ``solver`` kwarg.

    ``"auto"`` picks sesolve for kets, mesolve for density matrices.
    Explicit choices are honoured with a convention check:
    ``"sesolve"`` on a density matrix raises :class:`ConventionError`
    because the Schrödinger equation only evolves pure states.
    ``"mesolve"`` on a ket is legal — QuTiP promotes internally.
    """
    if solver not in {"auto", "sesolve", "mesolve"}:
        raise ConventionError(
            f"solve(): unknown solver {solver!r}; expected one of 'auto', 'sesolve', 'mesolve'."
        )
    if solver == "sesolve" and not initial_state.isket:
        raise ConventionError(
            "solve(solver='sesolve'): sesolve requires a ket initial state; "
            f"got a density matrix (isoper={initial_state.isoper}). Use "
            "solver='mesolve' or solver='auto' for mixed-state inputs."
        )
    if solver == "auto":
        return "sesolve" if initial_state.isket else "mesolve"
    return solver


__all__ = [
    "solve",
    "solve_ensemble",
]
