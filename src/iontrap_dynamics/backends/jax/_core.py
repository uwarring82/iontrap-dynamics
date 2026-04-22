# SPDX-License-Identifier: MIT
"""JAX-backend dispatch entry point (skeleton).

This module is private. It is imported lazily from
:func:`iontrap_dynamics.sequences.solve` only when the caller selects
``backend="jax"``, so the library's top-level import does not require
the ``[jax]`` extras to be installed.

Dispatch β.1 ships the availability check + stub; β.2 replaces the
:class:`NotImplementedError` with a real Dynamiqs integrator call
per ``docs/phase-2-jax-backend-design.md`` §7.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import qutip

from ...exceptions import BackendError
from ...hilbert import HilbertSpace
from ...observables import Observable
from ...results import StorageMode, TrajectoryResult


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

_BETA2_STUB_MESSAGE = (
    "solve(backend='jax') skeleton landed in Dispatch β.1. The "
    "Dynamiqs integrator wiring is scoped for Dispatch β.2 per "
    "docs/phase-2-jax-backend-design.md §7 staging. Until β.2 lands, "
    "use backend='qutip' (the default). When β.2 ships, this entry "
    "will return a TrajectoryResult tagged with "
    "backend_name='jax-dynamiqs'."
)


def solve_via_jax(
    *,
    hilbert: HilbertSpace,
    hamiltonian: qutip.Qobj | list[object],
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
    """JAX-backend solve entry. Skeleton stub — see module docstring.

    The keyword-only signature mirrors
    :func:`iontrap_dynamics.sequences.solve` exactly (minus the
    ``backend=`` discriminator, which resolves here by definition).
    Locking the contract at β.1 means any future solve() → solve_via_jax
    kwarg mismatch surfaces as a :class:`TypeError` rather than
    silently landing in ``**kwargs``. β.2 will consume these parameters
    in the Dynamiqs integrator call; β.1 ignores them after the
    availability check.

    Annotated as returning :class:`TrajectoryResult` to match the
    post-β.2 contract; the β.1 body unconditionally raises, so the
    declared return type is satisfied vacuously until the integrator
    lands.

    Raises
    ------
    BackendError
        When the ``[jax]`` extras are not importable. Contains an
        actionable install hint.
    NotImplementedError
        When the extras are installed but the Dynamiqs integrator
        has not yet been wired (Dispatch β.2 work).
    """
    del (  # signature-contract binding; unused until β.2
        hilbert,
        hamiltonian,
        initial_state,
        times,
        observables,
        request_hash,
        backend_name,
        storage_mode,
        provenance_tags,
        fock_tolerance,
        solver,
    )
    if not _is_jax_available():
        raise BackendError(_INSTALL_HINT)
    raise NotImplementedError(_BETA2_STUB_MESSAGE)
