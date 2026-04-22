# SPDX-License-Identifier: MIT
"""JAX / Dynamiqs backend for :mod:`iontrap_dynamics`.

Dispatches **β.1 through β.4** have landed on ``main``:

* β.1 — backend=kwarg dispatch plumbing, ``[jax]`` availability
  check, ``solver=``/``backend=`` compatibility validation.
* β.2 — real Dynamiqs :func:`dynamiqs.sesolve` / :func:`dynamiqs.mesolve`
  integrator; cross-backend numeric equivalence on carrier Rabi.
* β.3 — :attr:`StorageMode.LAZY` support via a JAX-Array-backed
  ``states_loader`` that materialises one ``Qobj`` per access.
* β.4 — time-dependent Hamiltonian support across all five
  builders (detuned carrier, detuned red / blue sideband, detuned
  MS gate, modulated carrier). Coefficient factories in
  :mod:`._coefficients`; assembly helper :func:`timeqarray_cos_sin`;
  user envelope ``envelope_jax`` kwarg on modulated carrier.

Cross-backend numeric equivalence is validated at library-default
integrator tolerances to better than ``1e-3`` on every builder
(empirical worst case ``1.4e-4``; see
:doc:`docs/benchmarks.md <../../../docs/benchmarks.md>`).

Phase 2 JAX-backend performance characterisation (β.4.5) returned
a null result at library scale — the JAX backend's value is
positioning, cross-backend consistency checking, and
forward-looking capability (autograd via a future γ track,
GPU / TPU dispatch), not raw CPU wall-clock at dim ≤ 200.

This subpackage is intentionally private-by-default: only
:func:`iontrap_dynamics.sequences.solve` and the time-dependent
:mod:`iontrap_dynamics.hamiltonians` builders import from it, and
they do so lazily so that the top-level package import does not
require the ``[jax]`` extras. Users select the JAX path through
the public ``backend="jax"`` kwarg on ``solve`` and on each
time-dependent builder.
"""

from ._core import solve_via_jax

__all__ = ["solve_via_jax"]
