# SPDX-License-Identifier: MIT
"""JAX-traceable coefficient callables for time-dependent Hamiltonians.

β.4's Option-X plan (parallel JAX-native builders) requires
coefficient callables that JAX can trace through for
:func:`dynamiqs.modulated`. The library's QuTiP-facing builders
use :mod:`math` / :mod:`numpy` coefficients; those cannot be
traced by JAX. This module provides factory closures emitting
:mod:`jax.numpy`-based coefficients that every structured
detuning builder (β.4.1 through β.4.3) consumes.

Design
------

Each factory takes the scalar parameter(s) of the coefficient
(e.g. a detuning in rad / s) and returns a callable
``f(t) -> jax.Array`` suitable for
``dynamiqs.modulated(f, H_piece)``. The scalar is captured by
value in the returned closure — callers mutating their own
``delta`` variable after factory invocation do not affect the
already-built closure.

Import policy
-------------

This module is imported **lazily** from the ``backend="jax"``
branch of time-dependent Hamiltonian builders. Top-level library
import (``from iontrap_dynamics import ...``) does not pull this
module in, so the ``[jax]`` extras are not required for the QuTiP
path. A user importing this module directly without ``[jax]``
installed will hit an ``ImportError`` at module load — that is
the expected behaviour; callers should route the request through
a builder's ``backend="jax"`` branch, which calls
:func:`iontrap_dynamics.backends.jax._core._require_jax` first for
a clean :class:`~iontrap_dynamics.exceptions.BackendError` with
install hint.

Scope
-----

β.4.1 introduced the two factories:

- :func:`cos_detuning_jax` — returns ``t ↦ jnp.cos(δ · t)``.
- :func:`sin_detuning_jax` — returns ``t ↦ jnp.sin(δ · t)``.

β.4.2 adds the assembly helper
:func:`timeqarray_cos_sin(H_static, H_quadrature, δ)` — JAX-side
sibling of :func:`iontrap_dynamics.hamiltonians._list_format_cos_sin`.
Emits ``dq.modulated(cos_jax(δ), H_static) + dq.modulated(sin_jax(δ), H_quadrature)``
so the four structured detuning builders
(:func:`detuned_carrier_hamiltonian`,
:func:`detuned_red_sideband_hamiltonian`,
:func:`detuned_blue_sideband_hamiltonian`,
:func:`detuned_ms_gate_hamiltonian`) share a single JAX assembly
path. β.4.3 consumes the same helper for the MS gate's 2-piece
cos/sin form.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array


def cos_detuning_jax(delta: float) -> Callable[[float], Array]:
    """Return a JAX-traceable callable ``t ↦ jnp.cos(delta * t)``.

    Parameters
    ----------
    delta
        Detuning in SI rad / s. Captured by value in the returned
        closure; the closure is independent of the caller's
        subsequent mutations.

    Returns
    -------
    Callable[[float], jax.Array]
        Scalar-in, JAX-Array-out. Safe to pass as the first
        argument to :func:`dynamiqs.modulated`.
    """

    def coeff(t: float) -> Array:
        return jnp.cos(delta * t)

    return coeff


def sin_detuning_jax(delta: float) -> Callable[[float], Array]:
    """Return a JAX-traceable callable ``t ↦ jnp.sin(delta * t)``.

    Parameters
    ----------
    delta
        Detuning in SI rad / s. See :func:`cos_detuning_jax` for
        closure semantics.

    Returns
    -------
    Callable[[float], jax.Array]
        Scalar-in, JAX-Array-out.
    """

    def coeff(t: float) -> Array:
        return jnp.sin(delta * t)

    return coeff


def timeqarray_cos_sin(
    h_static: object,
    h_quadrature: object,
    delta: float,
) -> object:
    """Assemble the ``cos(δt)·H_static + sin(δt)·H_quadrature``
    time-dependent Hamiltonian as a Dynamiqs ``TimeQArray``.

    JAX-path sibling of
    :func:`iontrap_dynamics.hamiltonians._list_format_cos_sin`. Same
    structural form — two Hermitian operators modulated by cos /
    sin of a fixed detuning — but returns
    ``dq.modulated(cos_jax(δ), H_static) + dq.modulated(sin_jax(δ), H_quadrature)``
    so it can be consumed by :func:`dynamiqs.sesolve` /
    :func:`dynamiqs.mesolve` under :func:`iontrap_dynamics.sequences.solve`
    with ``backend="jax"``.

    Parameters
    ----------
    h_static, h_quadrature
        QuTiP ``Qobj`` or any ``QArrayLike`` Dynamiqs accepts.
        Dims must match and must be compatible with the evolution
        state's dims.
    delta
        Detuning in SI rad / s. Snapshotted in the closures returned
        by :func:`cos_detuning_jax` / :func:`sin_detuning_jax`.

    Returns
    -------
    object
        A Dynamiqs ``SummedTimeQArray`` (type lives in
        :mod:`dynamiqs.time_qarray`; exposed here as ``object`` to
        keep this module's signature free of runtime Dynamiqs
        imports at type-check time).

    Notes
    -----
    Dynamiqs must be importable to call this helper — it imports
    :mod:`dynamiqs` at call time. Callers should guard with
    :func:`iontrap_dynamics.backends.jax._core._require_jax` for
    a clean :class:`BackendError` when the ``[jax]`` extras are
    missing.
    """
    import dynamiqs as dq

    return dq.modulated(cos_detuning_jax(delta), h_static) + dq.modulated(
        sin_detuning_jax(delta), h_quadrature
    )


__all__ = ["cos_detuning_jax", "sin_detuning_jax", "timeqarray_cos_sin"]
