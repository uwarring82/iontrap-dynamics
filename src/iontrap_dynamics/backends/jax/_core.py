# SPDX-License-Identifier: MIT
"""JAX-backend dispatch entry point (Dynamiqs integrator).

This module is private. It is imported lazily from
:func:`iontrap_dynamics.sequences.solve` only when the caller selects
``backend="jax"``, so the library's top-level import does not require
the ``[jax]`` extras to be installed.

Dispatch β.2 replaced the β.1 :class:`NotImplementedError` stub
with a real Dynamiqs :func:`sesolve` / :func:`mesolve` integrator.
Dispatch β.3 lifted the :class:`StorageMode.LAZY` restriction by
wiring a JAX-backed ``states_loader`` that materialises one
``Qobj`` per index on demand. The module:

* Forces JAX x64 at solve entry so the integrator runs in complex128
  (the library's CONVENTIONS.md §1 unit commitment). This is a
  process-wide JAX config change; if a user has already configured
  JAX for float32 elsewhere in their session, ``solve(backend="jax")``
  will transparently upgrade it — that is the correct behaviour for
  physics numerics.
* Dispatches to :func:`dynamiqs.sesolve` for ket inputs and
  :func:`dynamiqs.mesolve` (with empty jump ops) for density-matrix
  inputs — mirroring the QuTiP path's ket-vs-DM dispatch.
* Passes QuTiP ``Qobj`` operators and states to Dynamiqs directly;
  Dynamiqs treats them as ``QArrayLike`` via duck typing and
  converts internally. No explicit ``Qobj → jnp.asarray`` step is
  needed at this layer.
* Honours :class:`StorageMode` per the design note §2:
  ``OMITTED`` → no state materialisation (expectations already
  computed JAX-side via ``exp_ops``); ``EAGER`` → convert each time
  slice back to :class:`qutip.Qobj` with the original dims;
  ``LAZY`` → ``states_loader`` closes over the JAX
  :class:`~jax.Array` returned by Dynamiqs and materialises one
  ``Qobj`` per ``loader(i)`` call via :func:`numpy.asarray` slicing.
  The full trajectory is never bulk-copied to NumPy; per-index
  cost is one dense slice.
* Runs the CONVENTIONS.md §13 Fock-truncation check by piggybacking
  per-mode top-Fock projectors onto the Dynamiqs ``exp_ops`` list,
  so no state materialisation is needed to classify. The shared
  classifier :func:`_classify_fock_saturation` (in
  :mod:`.sequences`) maps the resulting ``p_top`` values to the
  §15 warning ladder identically to the QuTiP path.

Time-dependent Hamiltonians (QuTiP list-format with
callables / piecewise arrays) are not yet supported on the JAX
backend — Dynamiqs uses its own :class:`TimeQArray` wrapping for
that, and the translation is β.4 scope (deferred from β.3 once
investigation revealed Dynamiqs's :func:`dynamiqs.modulated`
requires JAX-traceable coefficient callables, whereas the library's
current time-dep builders emit scipy-traced callables using
:mod:`numpy` and :mod:`math`; bridging the two is an architectural
decision, not a mechanical translation). A clear
:class:`NotImplementedError` fires on list-shaped Hamiltonian
inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import qutip

from ...conventions import FOCK_CONVERGENCE_TOLERANCE
from ...exceptions import BackendError
from ...hilbert import HilbertSpace
from ...observables import Observable
from ...results import (
    ResultMetadata,
    StorageMode,
    TrajectoryResult,
)

if TYPE_CHECKING:
    # Optional dep: only installed with the [jax] extras. Keeps the
    # runtime import of this module JAX-free so the availability
    # check at _require_jax() fires cleanly.
    from dynamiqs import TimeQArray


def _is_jax_available() -> bool:
    """Return whether the ``[jax]`` extras (JAX + Dynamiqs) are importable.

    Kept as a module-level function so tests can monkey-patch
    availability without requiring a real install / uninstall cycle.
    Imports are guarded and do not propagate — a missing dependency
    is reported through the return value, not an exception.
    """
    try:
        import dynamiqs  # noqa: F401
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


_INSTALL_HINT = (
    "solve(backend='jax') requires the [jax] optional dependencies "
    "(JAX + Dynamiqs). Install with:\n"
    "    pip install iontrap-dynamics[jax]\n"
    "or add the extras to your dependency manager's lockfile. The "
    "existing [jax] block in pyproject.toml declares the required "
    "versions; see docs/phase-2-jax-backend-design.md §4.5."
)


def _require_jax() -> None:
    """Precondition check: the ``[jax]`` extras must be importable.

    Raises :class:`~iontrap_dynamics.exceptions.BackendError` with
    the shared install hint if any of the required dependencies are
    missing. Call at the top of any code path that subsequently
    imports ``jax`` or ``dynamiqs`` — both :func:`solve_via_jax` and
    β.4's time-dependent Hamiltonian builders use this for a
    consistent failure mode (one hint message, one exception type).
    """
    if not _is_jax_available():
        raise BackendError(_INSTALL_HINT)


def solve_via_jax(
    *,
    hilbert: HilbertSpace,
    hamiltonian: qutip.Qobj | list[object] | TimeQArray,
    initial_state: qutip.Qobj,
    times: np.ndarray,
    observables: Sequence[Observable] = (),
    request_hash: str = "",
    backend_name: str | None = None,
    storage_mode: StorageMode = StorageMode.OMITTED,
    provenance_tags: tuple[str, ...] = (),
    fock_tolerance: float | None = None,
    solver: str = "auto",
) -> TrajectoryResult:
    """JAX-backend solve — Dynamiqs integrator.

    The keyword-only signature mirrors
    :func:`iontrap_dynamics.sequences.solve` exactly (minus the
    ``backend=`` discriminator, which resolves here by definition).
    ``solver`` is accepted and ignored: the design-note §4.1
    contract is that only ``solver="auto"`` reaches this entry
    (the QuTiP-specific ``"sesolve"``/``"mesolve"`` values are
    rejected upstream in
    :func:`iontrap_dynamics.sequences._validate_backend`).

    Returns
    -------
    TrajectoryResult
        Tagged with ``backend_name="jax-dynamiqs"`` unless the
        caller overrode via the ``backend_name`` kwarg; the
        ``backend_version`` metadata records both the Dynamiqs and
        JAX versions.

    Raises
    ------
    BackendError
        When the ``[jax]`` extras are not importable.
    NotImplementedError
        When ``hamiltonian`` is a list (QuTiP time-dependent format);
        Dynamiqs requires its own :class:`TimeQArray` wrapping that
        is β.3 scope. When ``storage_mode=StorageMode.LAZY``; β.3
        scope per module docstring.
    """
    del solver  # validated upstream; only "auto" reaches here
    _require_jax()

    if isinstance(hamiltonian, list):
        raise NotImplementedError(
            "solve(backend='jax') does not accept a QuTiP time-"
            "dependent Hamiltonian list directly — the list-format "
            "coefficient callables are scipy-traced (numpy / math) "
            "whereas Dynamiqs requires JAX-traceable (jnp) "
            "coefficients. Use the builder's backend='jax' kwarg "
            "instead: e.g. "
            "detuned_carrier_hamiltonian(..., backend='jax') "
            "returns a dynamiqs.TimeQArray that this solver accepts. "
            "See docs/phase-2-jax-time-dep-design.md §2 for the list "
            "of JAX-enabled builders. Builders still covered only "
            "by the QuTiP path emit a QuTiP list and remain "
            "β.4.2 through β.4.4 scope."
        )

    import dynamiqs as dq
    import jax

    # Force x64 — the library's unit contract is double precision. See
    # docs/phase-2-jax-backend-design.md §8 ("Float32/64 defaults trip
    # up convention-version test"). Module-docstring documents the
    # side effect on user-controlled JAX config.
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

    tolerance = fock_tolerance if fock_tolerance is not None else FOCK_CONVERGENCE_TOLERANCE
    # Defer tolerance validation to the classifier; it already raises
    # ConventionError on non-positive values.

    tsave = np.asarray(times, dtype=np.float64)

    # Build the exp_ops list: user-requested observables first, then
    # piggyback top-Fock projectors per mode. This lets the Fock-
    # saturation check run without materialising states under OMITTED.
    user_exp_ops = [obs.operator for obs in observables]
    fock_check_modes: list[str] = []
    fock_check_ops: list[qutip.Qobj] = []
    for mode in hilbert.system.modes:
        fock_dim = hilbert.fock_truncations[mode.label]
        top_level_single = qutip.basis(fock_dim, fock_dim - 1).proj()
        top_level_embedded = hilbert.mode_op_for(top_level_single, mode.label)
        fock_check_modes.append(mode.label)
        fock_check_ops.append(top_level_embedded)

    combined_exp_ops = user_exp_ops + fock_check_ops
    # Dynamiqs treats empty exp_ops as None; guard explicitly.
    dq_exp_ops = combined_exp_ops if combined_exp_ops else None
    # Suppress the progress bar — library calls are not interactive.
    dq_options = dq.Options(progress_meter=None)

    if initial_state.isket:
        dq_result = dq.sesolve(
            hamiltonian,
            initial_state,
            tsave,
            exp_ops=dq_exp_ops,
            options=dq_options,
        )
    else:
        dq_result = dq.mesolve(
            hamiltonian,
            [],  # no jump operators — unitary Lindblad evolution
            initial_state,
            tsave,
            exp_ops=dq_exp_ops,
            options=dq_options,
        )

    # Pull expectations out. Dynamiqs returns shape (N_ops, N_times)
    # complex; observables are Hermitian so the physical value is the
    # real part. Cast to float64 to match QuTiP-path contract.
    expectations: dict[str, np.ndarray] = {}
    p_top_by_mode: dict[str, float] = {}
    if combined_exp_ops:
        raw_expects = np.asarray(dq_result.expects)  # (N_ops, N_times)
        n_user = len(user_exp_ops)
        for i, obs in enumerate(observables):
            expectations[obs.label] = raw_expects[i, :].real.astype(np.float64)
        for i, mode_label in enumerate(fock_check_modes):
            # Clamp at 0 to match the QuTiP-path roundoff-safety.
            p_top_series = raw_expects[n_user + i, :].real
            p_top_max = float(max(0.0, np.max(p_top_series)))
            p_top_by_mode[mode_label] = p_top_max
    else:
        # No exp_ops at all — shouldn't happen because fock_check_ops
        # is always populated, but guard defensively.
        for mode_label in fock_check_modes:
            p_top_by_mode[mode_label] = 0.0

    # Classify Fock saturation via the shared classifier — same
    # warnings channel, same ResultWarning records, same
    # ConvergenceError on Level 3 as the QuTiP path.
    from ...sequences import _classify_fock_saturation

    warning_records = _classify_fock_saturation(hilbert, p_top_by_mode, tolerance)

    # State materialisation per storage_mode.
    states_tuple: tuple[qutip.Qobj, ...] | None = None
    states_loader: Callable[[int], qutip.Qobj] | None = None
    if storage_mode is StorageMode.EAGER:
        raw_states = np.asarray(dq_result.states)
        # For kets: shape (N_times, dim, 1); for DMs: (N_times, dim, dim).
        # Either way, pass the slice to Qobj with the original dims
        # (which encodes ket-vs-operator) preserved from initial_state.
        states_tuple = tuple(
            qutip.Qobj(raw_states[i], dims=initial_state.dims) for i in range(raw_states.shape[0])
        )
    elif storage_mode is StorageMode.LAZY:
        # Keep the JAX Array alive through a closure; each loader(i)
        # call materialises a single time-slice via np.asarray. No bulk
        # host copy is triggered until the caller actually fetches a
        # state — OMITTED's memory-win shape at EAGER's per-state API.
        #
        # JAX silently clamps out-of-bounds indexing to the nearest
        # valid index (unlike NumPy which raises IndexError). That
        # would be a silent-degradation hazard (CONVENTIONS.md §15),
        # so bounds-check explicitly here and raise IndexError to
        # match the tuple-indexing contract the caller expects from
        # a Sequence-like interface.
        jax_states = dq_result.states
        dims_snapshot = initial_state.dims
        n_steps = int(jax_states.shape[0])

        def _lazy_loader(i: int) -> qutip.Qobj:
            # Normalise negative indices before the bounds check, as
            # NumPy / tuple / list do. -1 → n_steps - 1, etc.
            normalised = i + n_steps if i < 0 else i
            if not 0 <= normalised < n_steps:
                raise IndexError(
                    f"states_loader index {i} out of range for a trajectory of length {n_steps}."
                )
            return qutip.Qobj(np.asarray(jax_states[normalised]), dims=dims_snapshot)

        states_loader = _lazy_loader
    # StorageMode.OMITTED: leave both None.

    resolved_backend_name = backend_name if backend_name is not None else "jax-dynamiqs"
    resolved_backend_version = f"dynamiqs-{dq.__version__}+jax-{jax.__version__}"

    metadata = ResultMetadata(
        # Mirror the QuTiP path (sequences.py): honour whatever
        # convention_version the caller pinned on the IonSystem.
        # Using the CONVENTION_VERSION module constant directly would
        # silently re-label archived / intentionally-pinned systems
        # under the library-current string, breaking the archival
        # contract. The system itself holds the authoritative value.
        convention_version=hilbert.system.convention_version,
        request_hash=request_hash,
        backend_name=resolved_backend_name,
        backend_version=resolved_backend_version,
        storage_mode=storage_mode,
        fock_truncations=dict(hilbert.fock_truncations),
        provenance_tags=provenance_tags,
    )

    return TrajectoryResult(
        metadata=metadata,
        times=tsave,
        expectations=expectations,
        warnings=warning_records,
        states=states_tuple,
        states_loader=states_loader,
    )
