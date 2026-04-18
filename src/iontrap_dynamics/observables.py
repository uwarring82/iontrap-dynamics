# SPDX-License-Identifier: MIT
"""Named observable factories and expectation-value computation.

Each factory here returns an :class:`Observable` — a frozen record
holding the observable's label and the operator already embedded on
the full :class:`HilbertSpace` tensor-product structure. Downstream
code (``sequences.py`` in Phase 1; user-level analysis always) takes
a list of these and calls :func:`expectations_over_time` to get a
dict of ``label → np.ndarray`` across a trajectory.

Factory pattern, not a registry (v0.1)
--------------------------------------

Workplan §3 plans a Strategy-pattern registry eventually; the
observable surface is too small for that to pay off in v0.1. When the
measurement/apparatus layers (Phase 1.C / 1.D) add detector models,
readout infidelities, and derived observables like
``logarithmic_negativity`` and ``bell_fidelity``, the registry
becomes the natural home for discovery and composition. Until then,
explicit factory calls keep the API small and the imports honest.

Naming convention
-----------------

Default labels follow a uniform pattern:

- Spin observables on ion ``i``:   ``sigma_{x|y|z}_{i}``
- Mode occupation for label ``L``: ``n_{L}``

Callers can override via the keyword-only ``label`` argument when
computing multiple observables of the same kind (e.g. multiple σ_z on
the same ion at different times by different names).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import qutip
from numpy.typing import NDArray

from .hilbert import HilbertSpace
from .operators import sigma_x_ion, sigma_y_ion, sigma_z_ion


@dataclass(frozen=True, slots=True, kw_only=True)
class Observable:
    """A named observable ready for expectation-value computation.

    Parameters
    ----------
    label
        Human-readable identifier. Becomes the key in the dict
        returned by :func:`expectations_over_time`.
    operator
        QuTiP operator already embedded on the full tensor-product
        Hilbert space (dims match :meth:`HilbertSpace.qutip_dims`).
        Produced by one of the factory functions in this module.
    """

    label: str
    operator: qutip.Qobj


# ----------------------------------------------------------------------------
# Spin observables
# ----------------------------------------------------------------------------


def spin_x(
    hilbert: HilbertSpace,
    ion_index: int,
    *,
    label: str | None = None,
) -> Observable:
    """Return ``σ_x`` on the specified ion (CONVENTIONS.md §3)."""
    op = hilbert.spin_op_for_ion(sigma_x_ion(), ion_index)
    return Observable(label=label or f"sigma_x_{ion_index}", operator=op)


def spin_y(
    hilbert: HilbertSpace,
    ion_index: int,
    *,
    label: str | None = None,
) -> Observable:
    """Return ``σ_y`` on the specified ion (atomic-physics convention)."""
    op = hilbert.spin_op_for_ion(sigma_y_ion(), ion_index)
    return Observable(label=label or f"sigma_y_{ion_index}", operator=op)


def spin_z(
    hilbert: HilbertSpace,
    ion_index: int,
    *,
    label: str | None = None,
) -> Observable:
    """Return ``σ_z_ion`` on the specified ion.

    In the atomic-physics convention (§3): eigenvalue −1 on |↓⟩,
    +1 on |↑⟩.
    """
    op = hilbert.spin_op_for_ion(sigma_z_ion(), ion_index)
    return Observable(label=label or f"sigma_z_{ion_index}", operator=op)


# ----------------------------------------------------------------------------
# Mode observables
# ----------------------------------------------------------------------------


def number(
    hilbert: HilbertSpace,
    mode_label: str,
    *,
    label: str | None = None,
) -> Observable:
    """Return the number operator ``n̂ = a† a`` for the named mode.

    Expectation ``⟨n̂⟩`` gives the mean phonon occupation.
    """
    op = hilbert.number_for_mode(mode_label)
    return Observable(label=label or f"n_{mode_label}", operator=op)


# ----------------------------------------------------------------------------
# Expectation-value computation
# ----------------------------------------------------------------------------


def expectations_over_time(
    states: Sequence[qutip.Qobj],
    observables: Sequence[Observable],
) -> dict[str, NDArray[np.floating]]:
    """Compute ``⟨O_i⟩(t_j)`` for every observable × every state.

    Parameters
    ----------
    states
        Sequence of QuTiP states — kets or density matrices — all
        living on the same Hilbert space as the observables' embedded
        operators.
    observables
        Sequence of :class:`Observable` records. Order is preserved
        only for ease of debugging; the returned dict is keyed by
        ``label`` and callers should not rely on insertion order for
        correctness.

    Returns
    -------
    dict[str, np.ndarray]
        One entry per observable, keyed by its label. Each array has
        length ``len(states)`` and holds ``qutip.expect(op, state_j)``
        for ``j = 0 … len(states) − 1``.

    Notes
    -----
    Raises ``ValueError`` (via QuTiP) if any state's dims do not match
    any observable's operator dims. That's detected per-element rather
    than upfront; the first mismatch aborts the loop.
    """
    return {
        obs.label: np.asarray(
            [qutip.expect(obs.operator, state) for state in states],
            dtype=np.float64,
        )
        for obs in observables
    }


__all__ = [
    "Observable",
    "expectations_over_time",
    "number",
    "spin_x",
    "spin_y",
    "spin_z",
]
