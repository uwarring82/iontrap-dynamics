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
# Multi-ion spin observables
# ----------------------------------------------------------------------------


def parity(
    hilbert: HilbertSpace,
    ion_indices: Sequence[int],
    *,
    label: str | None = None,
) -> Observable:
    """Return the multi-ion parity operator ``Π_i σ_z^(i)``.

    For a pair ``(i, j)`` the operator is ``σ_z^(i) · σ_z^(j)``; its
    expectation is the joint two-body correlation ``⟨σ_z^(i) σ_z^(j)⟩``
    consumed by parity-scan protocols (Dispatch N) to reconstruct the
    joint readout distribution from single-ion marginals.

    For a general sequence of ion indices, returns the product
    ``∏_k σ_z^(i_k)``. Sign convention: eigenvalue ``(-1)^n_↑`` where
    ``n_↑`` is the count of ``|↑⟩`` among the named ions — consistent
    with the atomic-physics σ_z convention (§3: ⟨↑|σ_z|↑⟩ = +1).

    Default label: ``"parity_{i0}_{i1}_..."``.
    """
    indices = tuple(ion_indices)
    if len(indices) < 2:
        raise ValueError(f"parity: ion_indices must name at least two ions; got {indices}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"parity: ion_indices must be distinct; got {indices}")

    op = hilbert.spin_op_for_ion(sigma_z_ion(), indices[0])
    for i in indices[1:]:
        op = op * hilbert.spin_op_for_ion(sigma_z_ion(), i)
    default = "parity_" + "_".join(str(i) for i in indices)
    return Observable(label=label or default, operator=op)


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


# ----------------------------------------------------------------------------
# Interferometric fringe analysis (WP-02 WI-4 / F4, dispatch MCD)
# ----------------------------------------------------------------------------
#
# Application-agnostic post-processing of a readout signal over a phase /
# parameter scan: the fringe visibility V = (S_max − S_min)/(S_max + S_min) and
# the fitted phase shift φ of the cosine fringe. These are general primitives —
# the assembly of a full SU(1,1) interferometer sequence and any
# resource-constrained benchmark belong to the consuming programme, not here
# (WP/WP-02-two-mode-motional.md §1 boundary). Additive under CONVENTIONS v0.3:
# standard interferometric definitions, no new convention section.


@dataclass(frozen=True, slots=True, kw_only=True)
class FringeFit:
    """Least-squares fit of an interferometric fringe ``A + B·cos(θ − φ)``.

    Parameters
    ----------
    offset
        The fringe mean ``A``.
    amplitude
        The fringe amplitude ``B ≥ 0``.
    phase_rad
        The fitted phase shift ``φ`` in rad, in ``(−π, π]`` (the scan value at
        which the fringe is maximal).
    visibility
        The fitted interferometric contrast ``B / A`` (for ``A > 0``); ``inf``
        when the fitted offset is exactly zero.
    """

    offset: float
    amplitude: float
    phase_rad: float
    visibility: float


def fringe_visibility(signal: NDArray[np.floating] | Sequence[float]) -> float:
    """Model-free fringe visibility ``(S_max − S_min) / (S_max + S_min)``.

    For a **non-negative** readout signal (a bright-state probability or an
    intensity) sampled over a parameter scan. Model-free: it reads the extrema
    directly, with no fringe-shape assumption.

    Raises
    ------
    ValueError
        If ``signal`` is empty or non-finite, has a negative entry, or has
        ``S_max + S_min ≤ 0`` (the visibility is then undefined).
    """
    arr = np.asarray(signal, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("fringe_visibility: signal must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("fringe_visibility: signal must be finite")
    if np.any(arr < 0.0):
        raise ValueError(
            "fringe_visibility: signal must be non-negative (a probability / intensity)"
        )
    s_max = float(arr.max())
    s_min = float(arr.min())
    total = s_max + s_min
    if total <= 0.0:
        raise ValueError("fringe_visibility: S_max + S_min ≤ 0; visibility is undefined")
    return (s_max - s_min) / total


def fit_fringe(
    scan_rad: NDArray[np.floating] | Sequence[float],
    signal: NDArray[np.floating] | Sequence[float],
) -> FringeFit:
    """Least-squares fit of ``signal(θ) = A + B·cos(θ − φ)`` over a phase scan.

    Fits the linear-in-coefficients model ``c₀ + c₁ cos θ + c₂ sin θ`` (so
    ``A = c₀``, ``B = √(c₁² + c₂²)``, ``φ = atan2(c₂, c₁)``) and returns the
    fringe parameters as a :class:`FringeFit`. The visibility is the fitted
    contrast ``B / A``.

    Parameters
    ----------
    scan_rad
        Scan parameter θ in rad (the interferometer phase); at least 3 points.
    signal
        The readout signal at each θ; same shape as ``scan_rad``.

    Raises
    ------
    ValueError
        On non-1-D, mismatched-shape, fewer-than-3-point, non-finite, or
        **rank-deficient** input (a scan whose ``[1, cos θ, sin θ]`` design is
        rank < 3 — e.g. repeated or collinear phases — cannot determine the
        three parameters and is rejected rather than silently mis-fit).
    """
    theta = np.asarray(scan_rad, dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)
    if theta.ndim != 1:
        raise ValueError(f"fit_fringe: scan_rad must be 1-D; got ndim {theta.ndim}")
    if theta.shape != sig.shape:
        raise ValueError(
            f"fit_fringe: scan_rad {theta.shape} and signal {sig.shape} must have the same shape"
        )
    if theta.size < 3:
        raise ValueError("fit_fringe: need at least 3 scan points to fit A + B·cos(θ − φ)")
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(sig))):
        raise ValueError("fit_fringe: scan_rad and signal must be finite")
    design = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError(
            "fit_fringe: the scan is rank-deficient (repeated or collinear phases); "
            "A + B·cos(θ − φ) is underdetermined — use ≥3 distinct phases spanning the fringe"
        )
    coeffs = np.linalg.lstsq(design, sig, rcond=None)[0]
    c0, c1, c2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    amplitude = float(np.hypot(c1, c2))
    phase = float(np.arctan2(c2, c1))
    visibility = amplitude / c0 if c0 != 0.0 else float("inf")
    return FringeFit(offset=c0, amplitude=amplitude, phase_rad=phase, visibility=visibility)


__all__ = [
    "FringeFit",
    "Observable",
    "expectations_over_time",
    "fit_fringe",
    "fringe_visibility",
    "number",
    "parity",
    "spin_x",
    "spin_y",
    "spin_z",
]
