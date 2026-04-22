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

Scope (β.4.1)
-------------

Two factories:

- :func:`cos_detuning_jax` — returns ``t ↦ jnp.cos(δ · t)``.
- :func:`sin_detuning_jax` — returns ``t ↦ jnp.sin(δ · t)``.

Used today by :func:`iontrap_dynamics.hamiltonians.
detuned_carrier_hamiltonian` (β.4.1). Extensions to the other
structured detuning builders — :func:`detuned_red_sideband_hamiltonian`,
:func:`detuned_blue_sideband_hamiltonian`,
:func:`detuned_ms_gate_hamiltonian` — land in β.4.2 and β.4.3 by
consuming the same factories.
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


__all__ = ["cos_detuning_jax", "sin_detuning_jax"]
