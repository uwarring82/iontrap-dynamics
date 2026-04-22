# SPDX-License-Identifier: MIT
"""Backend dispatch surface for :mod:`iontrap_dynamics`.

Each subpackage implements a concrete solver backend. The QuTiP
reference backend lives at module level in :mod:`.sequences` and is
not re-homed here — backward compatibility with the v0.2 API is
non-negotiable. Subpackages under this namespace register alternative
backends that :func:`iontrap_dynamics.sequences.solve` dispatches to
when the ``backend=`` kwarg selects them.

Subpackages:

* :mod:`.jax` — Phase 2 JAX / Dynamiqs backend. See
  ``docs/phase-2-jax-backend-design.md`` for the design deliberation
  and ``docs/phase-1-architecture.md`` "Result family vs. backend
  variety (D5)" for the contract that all backends satisfy.

Per D5 and Design Principle 5, backends produce the canonical
:class:`~iontrap_dynamics.results.TrajectoryResult` schema and
identify themselves via ``ResultMetadata.backend_name``. Backend
code is invisible to downstream analyses; the public entry is
always :func:`iontrap_dynamics.sequences.solve`.
"""
