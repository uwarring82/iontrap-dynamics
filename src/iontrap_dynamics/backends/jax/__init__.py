# SPDX-License-Identifier: MIT
"""JAX / Dynamiqs backend for :mod:`iontrap_dynamics`.

Dispatch β.1 (this dispatch) ships the **skeleton only**: the
availability check, the ``backend="jax"`` dispatch plumbing in
:func:`iontrap_dynamics.sequences.solve`, and a
:class:`NotImplementedError` stub at the solver entry point. The
actual Dynamiqs integrator wiring (cross-backend equivalence test
for carrier Rabi, ``backend_name="jax-dynamiqs"`` tagging, etc.)
is Dispatch β.2 per ``docs/phase-2-jax-backend-design.md`` §7.

This subpackage is intentionally private-by-default: only
:func:`iontrap_dynamics.sequences.solve` imports from it, and it
does so lazily (inside the function body) so that the top-level
package import does not require JAX to be installed.
"""

from ._core import solve_via_jax

__all__ = ["solve_via_jax"]
