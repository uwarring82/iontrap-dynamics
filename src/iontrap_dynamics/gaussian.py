# SPDX-License-Identifier: MIT
"""Multimode Gaussian phase-space toolbox (the covariance / symplectic core).

CONVENTIONS.md §26.2 / §26.4 / §27. This module owns the **general** covariance /
symplectic core; WP-05 (SQ1/SQ2) created the single-mode (``N = 1``) case for the
non-adiabatic-squeezing readout, and ``phase_space.py`` holds the Wigner / readout
façades over it (WP-05 R4, no-fork rule). WP-07 GT1 lands the §27 multimode
primitives — the per-mode quadrature ordering, the symplectic form ``Ω = ⊕J``, the
``V + iΩ ≥ 0`` physicality guard, the Williamson symplectic spectrum, and the
partial transpose — as a **pure extension** of this module, never a refactor: the
``N = 1`` call and return contract of :func:`covariance_matrix` is unchanged.

Convention (§26.2). Dimensionless quadratures ``x̂ = â + â†``, ``p̂ = i(â† − â)``,
so the **vacuum quadrature variance is 1** and ``[x̂, p̂] = 2i``. The covariance
is ``V_ij = ½⟨{ΔR_i, ΔR_j}⟩`` with ``R = (x̂, p̂)``; the single-mode vacuum gives
``V = 𝟙₂``. All readout functionals here are **observable-only** (§26.4): they
compose §26.2, they are not new convention symbols.

.. warning::
    **Fock-truncation honesty (§13/§15).** ``V`` is computed from the truncated
    Fock representation, so an under-truncated squeezed state biases the readout —
    ``r`` reads low and ``ν`` reads > 1 (a pure state masquerading as thermal) —
    and the state norm stays 1 (QuTiP renormalises the truncated squeeze operator),
    so norm cannot reveal it. The parity-aware :func:`check_fock_truncation` guard
    (Dispatch **SQ4**) closes this: :func:`gaussian_readout` runs it by default,
    warning (Levels 1–2) or raising :class:`ConvergenceError` (Level 3) on an
    under-resolved squeezed state. The guard needs the **density matrix**, so it
    attaches to :func:`gaussian_readout`/:func:`check_fock_truncation` — the
    pure-``numpy`` functionals (:func:`squeezing_parameter`, …) that see only ``V``
    stay unguarded primitives. A squeezed state needs a **generous** cutoff (its
    even-``n`` pair-creation tail); ``r ≳ 1`` wants Fock ≳ 80.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

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
from .results import ResultWarning, WarningSeverity


def quadrature_operators(fock_dim: int) -> tuple[qutip.Qobj, qutip.Qobj]:
    """Return the single-mode Hermitian quadratures ``(x̂, p̂)`` on a Fock mode.

    ``x̂ = â + â†``, ``p̂ = i(â† − â)`` (CONVENTIONS.md §26.2, vacuum variance 1).
    """
    a = qutip.destroy(fock_dim)
    ad = a.dag()
    return a + ad, 1j * (ad - a)


def symplectic_form(n_modes: int) -> np.ndarray:
    """The multimode symplectic form ``Ω = ⊕_k J``, ``J = [[0, 1], [−1, 0]]`` (§27.2).

    Per-mode ``J``-blocks matching the §27.1 ordering ``R = (x̂₁, p̂₁, …, x̂_N, p̂_N)``,
    so ``[R_i, R_j] = 2i Ω_ij`` (the vacuum-variance-1 ``2i`` scaling). Shape ``(2N, 2N)``.
    """
    if n_modes < 1:
        raise ValueError(f"symplectic_form: n_modes ({n_modes}) must be ≥ 1.")
    j = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return np.kron(np.eye(n_modes), j)


def reduced_single_mode(state: qutip.Qobj, hilbert: HilbertSpace, mode_label: str) -> qutip.Qobj:
    """Partial-trace a full spin⊗mode ``state`` down to the named mode.

    Returns the reduced single-mode density matrix (dims ``[[N], [N]]``). Modes
    are appended after the ``n_ions`` spins in the §2 tensor order, so the mode's
    subsystem index is ``n_ions + position``.
    """
    labels = [m.label for m in hilbert.system.modes]
    if mode_label not in labels:
        raise ValueError(f"unknown mode label: {mode_label!r}. Available: {labels!r}")
    index = hilbert.n_ions + labels.index(mode_label)
    return state.ptrace(index)


def covariance_matrix(state: qutip.Qobj) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(V, d)`` for an ``N``-mode Gaussian state (ket or density matrix).

    ``V`` is the ``2N×2N`` covariance ``V_ij = ½⟨{ΔR_i, ΔR_j}⟩`` in the §27.1
    per-mode ordering ``R = (x̂₁, p̂₁, …, x̂_N, p̂_N)``, and ``d = ⟨R⟩`` the first
    moments; ``N`` is inferred from ``state.dims`` (mode ``k`` = subsystem ``k``).
    **Single-mode (``N = 1``) returns the 2×2 ``V`` and length-2 ``d`` unchanged**
    (CONVENTIONS.md §26.2 / §27.1); vacuum → ``V = 𝟙_{2N}``, ``d = 0``.
    """
    dims = state.dims[0]
    n_modes = len(dims)
    if n_modes < 1:
        raise ValueError(f"covariance_matrix: state has no modes (dims {state.dims!r}).")
    # Embedded Hermitian quadratures R = (x̂₁, p̂₁, …, x̂_N, p̂_N) in §27.1 per-mode order.
    ops: list[qutip.Qobj] = []
    for k in range(n_modes):
        x_k, p_k = quadrature_operators(dims[k])
        for local in (x_k, p_k):
            if n_modes == 1:
                ops.append(local)
            else:
                factors: list[qutip.Qobj] = [qutip.qeye(d) for d in dims]
                factors[k] = local
                ops.append(qutip.tensor(factors))
    dim2 = 2 * n_modes
    means = np.array([float(np.real(qutip.expect(op, state))) for op in ops], dtype=float)
    cov = np.empty((dim2, dim2), dtype=float)
    for i in range(dim2):
        for jx in range(i, dim2):
            anticomm = ops[i] * ops[jx] + ops[jx] * ops[i]
            sym = 0.5 * float(np.real(qutip.expect(anticomm, state)))
            cov[i, jx] = cov[jx, i] = sym - means[i] * means[jx]
    return cov, means


def symplectic_eigenvalue(cov: np.ndarray) -> float:
    """Single-mode symplectic eigenvalue ``ν = √(det V)`` (purity / thermal core).

    ``ν = 1`` for a pure state, ``ν = 2n̄_th + 1`` for a thermal state
    (CONVENTIONS.md §26.4). Squeezing-invariant (``det V`` is unchanged by the
    symplectic squeeze).
    """
    return float(np.sqrt(max(np.linalg.det(cov), 0.0)))


def _check_square_even(cov: np.ndarray, name: str) -> int:
    """Validate a ``2N×2N`` covariance array and return its mode count ``N``."""
    arr = np.asarray(cov)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] % 2 != 0:
        raise ValueError(
            f"{name}: expected a square 2N×2N covariance matrix; got shape {arr.shape}."
        )
    return int(arr.shape[0] // 2)


#: Float-noise tolerance for the structural covariance checks (realness, symmetry, the
#: ``V ⪰ 0`` positive-semidefiniteness that rejects an indefinite ``V``) and the ``ν ≈ 1``
#: pure-state clamp. Tight: covariance eigenvalues are variances, so a materially negative
#: one is a genuine violation, not noise.
_COVARIANCE_TOL = 1e-9

#: Tolerance for the squeezing-invariant physicality test ``ν_i ≥ 1``. Looser than
#: :data:`_COVARIANCE_TOL` because finite Fock truncation of a *displaced* state populates
#: the top level and pushes a symplectic eigenvalue below 1 by up to ~1e-6, while a genuine
#: uncertainty violation sits ≥ 1e-2 below 1 — a ~4-order gap this value splits. An absolute
#: tolerance on ``min eig(V + iΩ)`` is **not** squeezing-invariant and is defeatable (a
#: scale-asymmetric squeeze hides a ν = 0.5 violation inside 1e-9), so the physicality
#: tests use the invariant Williamson spectrum / an equilibrated ``V + iΩ`` instead.
_SYMPLECTIC_UNIT_TOL = 1e-4

#: Threshold for the **per-entry relative** symplecticity test in :func:`congruence`: each
#: ``|SΩSᵀ − Ω|_ij`` residual is measured against its rounding scale ``‖row_i(S)‖·‖row_j(S)‖``
#: (the Cauchy–Schwarz bound on ``|(SΩSᵀ)_ij|`` since Ω is orthogonal). A **genuinely** symplectic
#: map — including a strong ``r = 12`` squeeze or a deep 60-factor product — sits at ``≤ ~2e-15``
#: (a small multiple of machine ε, independent of ‖S‖), while a *near-symplectic* S tuned just
#: under a looser ``1e-9`` budget sits at ``~7e-10``. This value splits that ~5-order gap, ~5000×
#: above the genuine rounding floor and ~70× below the tightest attack. It is deliberately
#: **relative, not absolute**: a strong squeeze's absolute residual can reach ~1e-6 from
#: catastrophic ``e^{2r}`` cancellation while the map is exactly symplectic, so an absolute cap
#: false-rejects it; and the ν-damage a sub-threshold near-symplectic S can inflict on an
#: ill-conditioned V grows with ``√cond(V)`` (unbounded by any S-only absolute residual), so the
#: honest ``cond ≳ 1e12`` output guard (:data:`_MIN_RECIPROCAL_CONDITION`) is the amplification
#: backstop.
_SYMPLECTIC_RELATIVE_TOL = 1e-11

#: Reciprocal-condition-number floor below which the SPD-stable symplectic spectrum is
#: dominated by rounding and lies **outside the currently certified numerical range**.
#: ``cond(V) ≳ 1e12`` is *not physically unreachable* — an analytic pure squeeze
#: ``diag(1e6, 1e-6)`` is physical (``det V = 1``, ``ν = 1``) at ``cond = 1e12``, and
#: stronger squeezing goes higher — only outside what float64 certifies here; the
#: repository's finite-Fock construction stays far below it (even ``z = 5`` gives
#: ``cond ~ 5e8``). A caller supplying such an analytic ``V`` gets an explicit certification
#: error, not a wrong value or an "unphysical" verdict.
_MIN_RECIPROCAL_CONDITION = 1e-12


def symplectic_eigenvalues(cov: np.ndarray) -> np.ndarray:
    """The Williamson symplectic eigenvalues ``ν_i`` of a ``2N×2N`` ``V`` (§27.2).

    The moduli of the eigenvalues of ``iΩV`` — real ``±ν`` pairs — taken **one per pair
    with multiplicity preserved** (**not** the SVD of ``iΩV`` and **not** a tolerance-based
    deduplication that would merge degenerate modes). For a physical ``V`` every
    ``ν_i ≥ 1``. Returned ascending.

    **Numerically stable realisation.** For a positive-semidefinite ``V`` (every physical
    covariance) ``iΩV`` is *similar* to the Hermitian ``M = i V^{1/2} Ω V^{1/2}``, so its
    eigenvalues are ``eigvalsh(M)`` — real by construction (no imaginary parts silently
    discarded via ``.real``) and stable for the ill-conditioned ``V`` a squeezed state
    produces. A ``V`` **outside the currently certified numerical range** (``cond ≳ 1e12`` —
    not physically unreachable, but the spectrum would be rounding-dominated) raises
    :class:`ValueError` (§15 honesty; this spectrum feeds GT4 log-negativity). An indefinite
    ``V`` (not a bona-fide covariance, whose SPD square root does not exist) falls back to
    the direct ``|eig(iΩV)|`` definition.
    """
    n_modes = _check_square_even(cov, "symplectic_eigenvalues")
    arr = np.asarray(cov, dtype=float)
    omega = symplectic_form(n_modes)
    eigvals_v, basis = np.linalg.eigh(arr)
    if eigvals_v[0] < -_COVARIANCE_TOL:
        # Indefinite V: direct §27.2 definition |eig(iΩV)| (best effort; physical inputs
        # never reach here — the guard's PSD check rejects them first). The 2N moduli come
        # in equal adjacent pairs (|+ν_k| = |−ν_k|); ``[::2]`` takes one per pair keeping
        # multiplicity — NOT ``[n:]`` (the N largest), which would drop a distinct small ν.
        evals = np.linalg.eigvals(1j * omega @ arr)
        return np.sort(np.abs(evals))[::2]
    if eigvals_v[0] <= eigvals_v[-1] * _MIN_RECIPROCAL_CONDITION:
        raise ValueError(
            "symplectic_eigenvalues: covariance matrix is outside the certified numerical "
            f"range (condition number ≳ {1 / _MIN_RECIPROCAL_CONDITION:.0e}); the symplectic "
            "spectrum would be dominated by rounding."
        )
    v_half = (basis * np.sqrt(eigvals_v)) @ basis.T
    lam = np.linalg.eigvalsh(1j * v_half @ omega @ v_half)
    return np.sort(lam)[n_modes:]


def _require_physical_covariance(cov: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(V, ν)`` for a validated real physical covariance, or raise ``ValueError``.

    The GT2+ derived functionals (:func:`purity`, :func:`gaussian_entropy_bits`) are
    physically meaningful only on a bona-fide covariance: a **real, finite, symmetric**
    ``2N×2N`` matrix with ``V + iΩ ≥ 0``. Silently returning e.g. ``purity > 1`` (from
    ``0.5·𝟙``) or ``entropy = 0`` (the ``ν ≤ 1`` branch mistaking a gross violation for
    pure-state noise) would mask a malformed input — the §15 honesty rule, mirroring the
    ``information._common._von_neumann_entropy_bits`` non-finite guard.

    Physicality is tested in the **squeezing-invariant** Williamson form (§27.2: *"for a
    real, symmetric, positive-semidefinite covariance candidate, physicality is equivalent
    to ν_i ≥ 1"*): ``V`` must be PSD — rejecting an indefinite ``V`` such as
    ``diag(3,3,−3,−3)`` — **and** every ``ν_i ≥ 1``, rejecting a sub-uncertainty ``V`` such
    as ``0.5·𝟙`` or a scale-asymmetric squeeze that an absolute ``min eig(V + iΩ)`` tol
    would miss. Returns the symplectic spectrum so the caller need not recompute it. An
    extremely ill-conditioned ``V`` raises via :func:`symplectic_eigenvalues` (kept as the
    single source of that check, so GT4's PT-spectrum consumers inherit it). This
    Williamson guard is retained as **defence in depth** alongside :func:`is_physical`.
    """
    _check_square_even(cov, name)
    arr = np.asarray(cov)
    if np.iscomplexobj(arr):
        if np.any(np.abs(arr.imag) > _COVARIANCE_TOL):
            raise ValueError(f"{name}: covariance matrix must be real; got complex entries.")
        arr = arr.real
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: covariance matrix has non-finite entries (NaN/inf).")
    if not np.allclose(arr, arr.T, atol=_COVARIANCE_TOL):
        raise ValueError(f"{name}: covariance matrix must be symmetric.")
    if float(np.min(np.linalg.eigvalsh(arr))) < -_COVARIANCE_TOL:
        raise ValueError(f"{name}: unphysical covariance matrix (V is not positive semidefinite).")
    nus = symplectic_eigenvalues(arr)  # raises on an uncertifiably ill-conditioned V
    if float(np.min(nus)) < 1.0 - _SYMPLECTIC_UNIT_TOL:
        raise ValueError(
            f"{name}: unphysical covariance matrix (symplectic eigenvalue ν < 1, i.e. "
            "V + iΩ ≱ 0); purity/entropy are defined only for a bona-fide physical covariance."
        )
    return arr, nus


def purity(cov: np.ndarray) -> float:
    """Gaussian purity ``μ = Tr(ρ²) = ∏_i 1/ν_i = 1/√det V`` (Dispatch GT2; §27.4).

    The product of the reciprocal Williamson symplectic eigenvalues (equivalently
    ``1/√det V``): ``μ = 1`` for a pure Gaussian state (vacuum, coherent, squeezed —
    squeezing is symplectic and leaves ``μ`` unchanged), ``μ < 1`` for a mixed one.
    Raises ``ValueError`` on a non-real / non-finite / non-symmetric / unphysical ``V``
    (an unphysical ``V`` would give the nonsensical ``μ > 1``). Each ``ν_i`` is clamped to
    ``≥ 1`` within :data:`_SYMPLECTIC_UNIT_TOL` so a truncation artifact cannot yield
    ``μ > 1``. **Gaussianity precondition (§27.4):** this equals the true state purity only
    for a Gaussian state; for a non-Gaussian state it is the purity of the Gaussian state
    with the same second moments. Consumes :func:`symplectic_eigenvalues`.
    """
    _, nus = _require_physical_covariance(cov, "purity")
    return float(1.0 / np.prod(np.maximum(nus, 1.0)))


def _entropy_g_bits(nu: float) -> float:
    """Bosonic entropy term ``g(ν)`` in bits (§27.4); ``g(ν) = 0`` within tol of ``ν = 1``.

    ``g(ν) = (ν+1)/2·log₂((ν+1)/2) − (ν−1)/2·log₂((ν−1)/2)`` for ``ν > 1``. The
    ``ν ≤ 1 + tol`` branch folds in the exact pure-state value ``g(1) = 0``, avoiding
    ``log₂`` of a non-positive ``(ν−1)/2``. :func:`gaussian_entropy_bits` validates
    physicality and clamps ``ν`` to ``≥ 1`` **first**, so this only ever sees ``ν ≥ 1``.
    """
    if nu <= 1.0 + _COVARIANCE_TOL:
        return 0.0
    nu_p = (nu + 1.0) / 2.0
    nu_m = (nu - 1.0) / 2.0
    return float(nu_p * math.log2(nu_p) - nu_m * math.log2(nu_m))


def gaussian_entropy_bits(cov: np.ndarray) -> float:
    """Gaussian (bosonic von Neumann) entropy ``S = Σ_i g(ν_i)`` in bits (GT2; §27.4).

    ``g(ν) = (ν+1)/2·log₂((ν+1)/2) − (ν−1)/2·log₂((ν−1)/2)`` for ``ν ≥ 1``, ``g(1) = 0``;
    summed over the Williamson symplectic eigenvalues. Vacuum / any pure Gaussian state
    → 0; a thermal mode ``ν = 2n̄+1`` → ``(n̄+1)log₂(n̄+1) − n̄ log₂ n̄``, matching
    :func:`iontrap_dynamics.information._common._von_neumann_entropy_bits` on the
    truncated density matrix. Raises ``ValueError`` on a non-real / non-finite /
    non-symmetric / unphysical ``V`` (a gross violation must not be silently read as a
    pure state). **Gaussianity precondition (§27.4)** as for :func:`purity`. Consumes
    :func:`symplectic_eigenvalues`.
    """
    _, nus = _require_physical_covariance(cov, "gaussian_entropy_bits")
    return float(sum(_entropy_g_bits(float(nu)) for nu in np.maximum(nus, 1.0)))


def is_physical(cov: np.ndarray, *, tol: float = _SYMPLECTIC_UNIT_TOL) -> bool:
    """Bona-fide covariance test: ``V + iΩ ≥ 0`` (Robertson–Schrödinger; §27.2).

    Realised in the **squeezing-invariant** Williamson form that §27.2 states is equivalent
    for a covariance candidate: ``V`` positive-semidefinite **and** every symplectic
    eigenvalue ``ν_i ≥ 1``. This is scale- **and** correlation-invariant, where a direct
    ``min eig(V + iΩ) ≥ −tol`` is not — even after diagonal equilibration: a strongly
    correlated (or scale-asymmetric) ``V`` shrinks the ``V + iΩ`` violation below an
    absolute ``tol`` while the invariant ``ν`` stays ``0.5`` (e.g. ``[[c, d], [d, c]]`` with
    ``c² − d² = ¼``, or ``diag(1e9, 2.5e-10)``). The ``V ⪰ 0`` check rejects an indefinite
    ``V`` — the case §27.2 warns a *bare* ``ν ≥ 1`` misses — so this is §27.2's own
    "PSD candidate ⇒ ``ν_i ≥ 1``" equivalence, not a shortcut, and needs no convention
    change. ``tol`` absorbs finite-Fock truncation artifacts. Returns ``False`` for a
    non-real / non-finite / non-symmetric / unphysical ``V``. For a ``V`` outside the
    certified numerical range (``cond ≳ 1e12``) it **raises** (via
    :func:`symplectic_eigenvalues`) rather than returning ``False`` — a physical-but-
    uncertifiable ``V`` such as ``diag(1e6, 1e-6)`` must not be mislabelled unphysical.
    """
    _check_square_even(cov, "is_physical")
    arr = np.asarray(cov)
    if np.iscomplexobj(arr):
        if np.any(np.abs(arr.imag) > _COVARIANCE_TOL):
            return False
        arr = arr.real
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)) or not np.allclose(arr, arr.T, atol=_COVARIANCE_TOL):
        return False
    if float(np.min(np.linalg.eigvalsh(arr))) < -_COVARIANCE_TOL:
        return False  # indefinite V — the case a bare ν ≥ 1 would miss (§27.2)
    # symplectic_eigenvalues raises for an uncertifiably ill-conditioned V; let that
    # propagate rather than mislabel a physical-but-uncertifiable V as unphysical.
    return bool(np.min(symplectic_eigenvalues(arr)) >= 1.0 - tol)


def partial_transpose(cov: np.ndarray, mode_indices: Sequence[int]) -> np.ndarray:
    """Partial transpose ``Ṽ = T V T`` over the modes in ``mode_indices`` (§27.3).

    Time-reversal on subsystem ``B``: ``p̂_b → −p̂_b`` for each ``b`` in ``mode_indices``
    (``x̂`` and the untouched modes unchanged), i.e. ``T`` flips the momentum row/column
    of each named mode. Returns a new ``2N×2N`` array; ``mode_indices`` must be distinct
    and in ``[0, N)``.
    """
    n_modes = _check_square_even(cov, "partial_transpose")
    signs = np.ones(2 * n_modes)
    seen: set[int] = set()
    for raw in mode_indices:
        b = int(raw)
        if b != raw:
            raise ValueError(f"partial_transpose: mode index {raw!r} is not an integer.")
        if not 0 <= b < n_modes:
            raise ValueError(f"partial_transpose: mode index {b} out of range [0, {n_modes}).")
        if b in seen:
            raise ValueError(f"partial_transpose: duplicate mode index {b}.")
        seen.add(b)
        signs[2 * b + 1] = -1.0  # flip p̂_b (the momentum row/column of mode b)
    t = np.diag(signs)
    return np.asarray(t @ np.asarray(cov) @ t, dtype=float)


def congruence(symplectic: np.ndarray, cov: np.ndarray) -> np.ndarray:
    r"""Symplectic congruence ``V ↦ S V Sᵀ`` for a symplectic ``S`` (GT3a; §27).

    Applies a symplectic map ``S`` (``S Ω Sᵀ = Ω``) to a physical covariance ``V``. Because
    ``S`` is symplectic the Williamson spectrum ``ν_i`` — and hence physicality, purity, and
    entropy — is **preserved**: a squeeze / rotation / beamsplitter re-shapes ``V`` without
    adding or removing mixedness. ``S`` must be real, finite, and square ``2N×2N`` matching
    ``V``; the ``S Ω Sᵀ = Ω`` condition is checked with a **per-entry, scale-aware relative**
    tolerance — each residual entry against its rounding scale ``‖row_i(S)‖·‖row_j(S)‖``, so a
    large symplectic block cannot inflate the budget for a small non-symplectic one, and a strong
    squeeze (whose *absolute* residual balloons from ``e^{2r}`` cancellation while it stays exactly
    symplectic) is not false-rejected. Both the input ``V`` and the output ``S V Sᵀ`` are validated
    as bona-fide physical covariances; the output validation also carries the honest ``cond ≳ 1e12``
    certification guard, which backstops the residual ``√cond(V)`` ν-amplification a
    just-sub-threshold near-symplectic ``S`` could otherwise exert on a strongly-squeezed ``V``.

    The gate directly guarantees only **approximate preservation of** ``Ω``; the output validation
    adds certifiability and physicality, *not* closeness to the input spectrum. Empirically, over
    this WP's certified test domain, an accepted ``S`` preserves the Williamson spectrum *relatively*
    to ``|Δν_i| / ν_i ≲ 1e-11`` — the physically meaningful invariant, since purity, entropy, and
    physicality all turn on ``ν`` *relative* to the vacuum floor ``1``. This is an **observed,
    validated numerical bound, not a proven per-call guarantee.** The *absolute* shift ``|Δν_i|``
    scales with ``ν_i`` itself, so it stays ``≪ 1e-4`` throughout the validated trapped-ion operating
    range used by this WP (``ν ≲ 1e4``) and would reach ``1e-4`` only for ``ν ≳ 1e7`` (e.g. a thermal
    ``V = 1e7·𝟙`` — a perfectly valid covariance, merely **outside** that operating range, since this
    module is application-agnostic), where the *relative* preservation still holds. A caller needing a
    formal *absolute* per-call spectrum guarantee should compare :func:`symplectic_eigenvalues` before
    and after (optionally behind a separate relative tolerance).

    This is the **application-agnostic** congruence (**GT3a**). The ion-specific normal→local
    ``S`` — built from mode eigenvectors, ion masses, and local frequencies — is a separate
    adapter (**GT3b**) kept **out** of this module (near ``iontrap-structure``); none of that
    machinery lives here.
    """
    n_modes = _check_square_even(cov, "congruence")
    s = np.asarray(symplectic)
    if s.shape != (2 * n_modes, 2 * n_modes):
        raise ValueError(
            f"congruence: symplectic map S must be {2 * n_modes}×{2 * n_modes} to match V; "
            f"got {s.shape}."
        )
    if np.iscomplexobj(s):
        if np.any(np.abs(s.imag) > _COVARIANCE_TOL):
            raise ValueError("congruence: symplectic map S must be real; got complex entries.")
        s = s.real
    s = np.asarray(s, dtype=float)
    if not np.all(np.isfinite(s)):
        raise ValueError("congruence: symplectic map S has non-finite entries (NaN/inf).")
    omega = symplectic_form(n_modes)
    residual = np.abs(s @ omega @ s.T - omega)
    # PER-ENTRY RELATIVE symplecticity gate. Each residual entry is measured against its rounding
    # scale ‖row_i(S)‖·‖row_j(S)‖ (Cauchy–Schwarz bound on |(SΩSᵀ)_ij|, Ω orthogonal; floored at
    # 1 = Ω's unit scale). This is:
    #   • rounding-aware — a genuine symplectic map, however strongly squeezed, sits at ~n·eps
    #     (≤ ~2e-15), so it is never false-rejected (an *absolute* residual cap would, since a
    #     strong squeeze's absolute |SΩSᵀ−Ω| reaches ~1e-6 from e^{2r} cancellation while exact);
    #   • block-local — a large symplectic block cannot inflate the budget for a small
    #     non-symplectic block beside it (the multimode normal→local regime GT3b targets), which a
    #     single global ‖S‖² budget would; and
    #   • tight — the threshold (:data:`_SYMPLECTIC_RELATIVE_TOL`, ~5000× the genuine floor) rejects
    #     near-symplectic S tuned just under a looser 1e-9 budget. A sub-threshold S can still, in
    #     principle, drift ν on an ill-conditioned V (the damage grows ~√cond(V), unbounded by any
    #     S-only residual); the honest cond≳1e12 guard on the OUTPUT below is that backstop.
    row_norms = np.linalg.norm(s, axis=1)
    scale = np.maximum(np.outer(row_norms, row_norms), 1.0)
    relative = residual / scale
    worst_rel = float(np.max(relative))
    if worst_rel > _SYMPLECTIC_RELATIVE_TOL:
        i, j = (int(k) for k in np.unravel_index(int(np.argmax(relative)), relative.shape))
        raise ValueError(
            f"congruence: S is not symplectic — relative |SΩSᵀ − Ω| = {worst_rel:.2e} (at entry "
            f"[{i},{j}]) exceeds the scale-aware tolerance {_SYMPLECTIC_RELATIVE_TOL:.0e}; a "
            f"congruence must preserve Ω."
        )
    _require_physical_covariance(cov, "congruence")  # the input must be a bona-fide covariance
    transformed = s @ np.asarray(cov, dtype=float) @ s.T
    result = 0.5 * (transformed + transformed.T)  # symmetrise (S V Sᵀ is symmetric)
    _require_physical_covariance(result, "congruence output")  # physicality is preserved
    return np.asarray(result, dtype=float)


def _validate_cut(mode_indices: Sequence[int], n_modes: int, name: str) -> set[int]:
    """Validate ``mode_indices`` as a proper bipartition ``1 ≤ |cut| ≤ N − 1``; return the set.

    Rejects non-integral indices (e.g. ``0.5``) rather than silently coercing them.
    """
    cut: set[int] = set()
    for raw in mode_indices:
        b = int(raw)
        if b != raw:
            raise ValueError(f"{name}: mode index {raw!r} is not an integer.")
        cut.add(b)
    if not 0 < len(cut) < n_modes:
        raise ValueError(
            f"{name}: the cut must name a proper bipartition (1 ≤ |cut| ≤ {n_modes - 1}); "
            f"got {sorted(cut)} of {n_modes} modes."
        )
    return cut


def log_negativity(cov: np.ndarray, mode_indices: Sequence[int]) -> float:
    r"""Gaussian logarithmic negativity ``E_N = Σ_k max(0, −log₂ ν̃_k)`` (bits; GT4; §27.4).

    ``ν̃_k`` are the symplectic eigenvalues of the partial transpose ``Ṽ = T_B V T_B`` over
    the modes in ``mode_indices`` (subsystem ``B``). The **full sum** over the PT symplectic
    eigenvalues is taken — **not** the smallest-only form, which holds only for two-mode /
    ``1×N`` cuts. Faithful to the sealed §27.4 formula (no clamp). ``E_N > 0`` witnesses NPT
    entanglement across the cut; ``E_N = 0`` (PPT) certifies separability **only for a ``1×N``
    cut** (see :func:`is_separable`) — for an ``M×N`` cut with ``M, N ≥ 2`` PPT-bound-entangled
    Gaussian states exist, so it is only an NPT witness there. ``mode_indices`` must be a proper
    bipartition (``1 ≤ |cut| ≤ N − 1``) and ``V`` a physical Gaussian covariance (the value
    equals the true state's only for a Gaussian state, §27.4). Oracle: a two-mode squeezed
    vacuum with squeezing ``r`` gives ``E_N = 2r / ln 2``.

    .. note::
        A **well-truncated** separable state gives exactly ``E_N = 0`` (its PT eigenvalues are
        ``≥ 1``). An **under-truncated** *displaced* state has a small finite-Fock floor (a top-
        Fock-level artifact pushes a PT eigenvalue just below 1); that floor is a covariance-
        estimation error, indistinguishable from genuine weak entanglement of the same size, so
        it is **not** clamped away here — it is the honest value. :func:`is_separable` guards
        against reading such a floor as a certificate.
    """
    n_modes = _check_square_even(cov, "log_negativity")
    _validate_cut(mode_indices, n_modes, "log_negativity")
    _require_physical_covariance(cov, "log_negativity")  # the state must be a bona-fide covariance
    nu_tilde = symplectic_eigenvalues(partial_transpose(cov, mode_indices))
    return float(sum(max(0.0, -math.log2(nu)) for nu in nu_tilde))


def is_separable(cov: np.ndarray, mode_indices: Sequence[int], *, tol: float = 0.0) -> bool:
    r"""Certify Gaussian separability across a **``1×N``** cut via PPT (GT4; §27.4).

    A **strictly one-sided** certificate: returns ``True`` **only** when ``E_N ≤ tol``. With the
    default ``tol = 0.0`` this is ``E_N == 0`` ⟺ PPT ⟺ separability (for a ``1×N`` Gaussian cut,
    one side a single mode; Werner–Wolf / Simon) — a **genuine** certificate. Because
    entanglement is continuous, *any* ``tol > 0`` would falsely certify a state entangled below
    it (e.g. a TMSV with ``r < tol·ln 2 / 2``), so a positive ``tol`` yields only *"PPT within
    numerical tolerance"*, **not** certified separability — pass it only when that weaker
    statement is what you want. A well-truncated separable state has ``E_N = 0`` exactly, so it
    certifies at ``tol = 0``; float noise or coarse Fock truncation can push ``E_N`` slightly
    positive, giving the conservative ``False`` (**not certified** — the state may be entangled
    *or* merely under-resolved; do **not** read ``False`` as "entangled", use
    :func:`log_negativity` for the witness). Uncertainty is never converted to ``True``.
    **Raises** :class:`ValueError` for a non-finite / negative ``tol``, and for an ``M×N`` cut
    with ``M, N ≥ 2`` — PPT-bound-entangled Gaussian states exist there, so ``E_N = 0`` does not
    certify separability; use :func:`log_negativity` as an NPT witness.
    """
    n_modes = _check_square_even(cov, "is_separable")
    if not (math.isfinite(tol) and tol >= 0.0):
        raise ValueError(f"is_separable: tol must be finite and ≥ 0; got {tol!r}.")
    cut = _validate_cut(mode_indices, n_modes, "is_separable")
    if min(len(cut), n_modes - len(cut)) != 1:
        raise ValueError(
            f"is_separable: PPT certifies separability only for a 1×N Gaussian cut; this is a "
            f"{n_modes - len(cut)}×{len(cut)} cut. For M×N with M, N ≥ 2, PPT-bound-entangled "
            "Gaussian states exist, so E_N = 0 does not certify separability — use "
            "log_negativity as an NPT witness."
        )
    return log_negativity(cov, mode_indices) <= tol


def squeezing_parameter(cov: np.ndarray) -> float:
    """Squeezing ``r = ¼·ln(λ_max / λ_min)`` from the eigenvalues of ``V``.

    Uses the eigenvalue **ratio** (CONVENTIONS.md §26.4), **not** ``tr V`` —
    ``tr V`` cannot separate squeezing from thermal width. Vacuum / thermal →
    ``r = 0``; pure squeezed vacuum → ``r`` (principal variances ``e^{∓2r}``).
    """
    lam = np.linalg.eigvalsh(cov)
    lam_min, lam_max = float(lam[0]), float(lam[-1])
    if lam_min <= 0.0:
        raise ValueError(f"non-physical covariance: eigenvalues {lam} must be strictly positive.")
    return 0.25 * float(np.log(lam_max / lam_min))


def mean_squeezed_occupation(cov: np.ndarray) -> float:
    """Pure-squeezing occupation ``n̄_sq = sinh²r`` (CONVENTIONS.md §26.4).

    The pure-squeezing content, **not** the state's centred occupation when
    ``n̄_th > 0`` (that would also include the thermal core ``ν``).
    """
    return float(np.sinh(squeezing_parameter(cov)) ** 2)


def coherent_amplitude(mean: np.ndarray) -> complex:
    """Coherent displacement ``α = (⟨x̂⟩ + i⟨p̂⟩)/2`` from the first moments.

    ``|α| = ½√(⟨x̂⟩² + ⟨p̂⟩²)`` (frozen §7-consistent; CONVENTIONS.md §26.4).
    """
    return complex(0.5 * (mean[0] + 1j * mean[1]))


def mean_occupation(cov: np.ndarray, mean: np.ndarray) -> float:
    """Physical mean occupation ``n̄ = (tr V + dᵀd − 2)/4 = ⟨â†â⟩``.

    First-moment-aware (CONVENTIONS.md §26.4 caveat): includes the displacement
    energy. **Not** ``(ν − 1)/2``, which is the thermal-core occupation only.
    """
    return float((np.trace(cov) + float(mean @ mean) - 2.0) / 4.0)


#: Physical constants for the effective-temperature conversion (CODATA 2018, SI); the ``ℏ``
#: value matches ``clos2016._HBAR``.
_HBAR_J_S = 1.054571817e-34  # ℏ [J·s]
_K_B_J_PER_K = 1.380649e-23  # k_B [J/K]


def effective_temperature(nbar: float, omega_loc: float) -> float:
    r"""Effective temperature ``T_eff = ℏ ω_loc / (k_B ln(1 + 1/n̄))`` in kelvin (GT5; §27.4).

    Maps a mode's **first-moment-aware** mean occupation ``n̄`` — from :func:`mean_occupation`,
    which includes displacement/squeezing energy (**not** the thermal core ``(ν − 1)/2``) — to
    the temperature of the thermal state with the same ``⟨n̂⟩`` (**energy-equivalent**;
    ``n̄ → T_eff → n̄`` round-trips exactly). ``omega_loc`` is the mode's local **angular**
    frequency (rad/s); ``ℏ``, ``k_B`` are SI. ``T_eff = 0`` at ``n̄ = 0`` by continuity — the
    **vacuum / zero-occupation** limit is 0 K, **not** every pure state: a squeezed and/or
    displaced pure marginal has ``n̄ > 0`` and hence ``T_eff > 0`` (unlike the thermal-core
    reading). **Neutral** — carries no consuming-application framing
    (no ``T_H``/"Hawking"; that is owned by the application). Raises for a non-finite or
    negative ``n̄``, and for a non-finite or non-positive ``omega_loc``.
    """
    if not (math.isfinite(nbar) and nbar >= 0.0):  # rejects NaN and ±inf
        raise ValueError(
            f"effective_temperature: mean occupation n̄ must be finite and ≥ 0; got {nbar!r}."
        )
    if not (math.isfinite(omega_loc) and omega_loc > 0.0):  # rejects NaN and ±inf
        raise ValueError(
            f"effective_temperature: omega_loc must be finite and > 0 (rad/s); got {omega_loc!r}."
        )
    if nbar == 0.0:
        return 0.0  # continuity limit: ln(1 + 1/n̄) → ∞ as n̄ → 0⁺
    return float(_HBAR_J_S * omega_loc / (_K_B_J_PER_K * math.log1p(1.0 / nbar)))


def phonon_number_distribution(state: qutip.Qobj) -> np.ndarray:
    """Return the phonon-number diagonals ``Pₙ = ⟨n|ρ|n⟩`` of a single-mode state.

    The direct number distribution (a ket's ``|ψₙ|²`` or a density matrix's
    diagonal), real and non-negative, length ``fock_dim`` (CONVENTIONS.md §26.4,
    observable-only). For a **pure squeezed vacuum** only the even ``n`` are
    populated (:func:`pure_squeezed_vacuum_pn` is the analytic oracle);
    displacement or thermal content fills the odd ``n``.
    """
    if len(state.dims[0]) != 1:
        raise ValueError(
            f"phonon_number_distribution expects a single-mode state; got dims {state.dims}. "
            "Partial-trace to one mode first (see reduced_single_mode)."
        )
    dense = np.asarray(state.full())
    populations = np.abs(dense.ravel()) ** 2 if state.isket else np.real(np.diag(dense))
    return np.asarray(populations, dtype=float)


def pure_squeezed_vacuum_pn(r: float, n_max: int) -> np.ndarray:
    """Analytic ``Pₙ`` of a pure squeezed vacuum ``S(r)|0⟩`` (the even-only oracle).

    ``P_{2k} = [(2k)! / (2^k k!)²] · (tanh r)^{2k} / cosh r``, ``P_{2k+1} = 0``
    (CONVENTIONS.md §26.4). Depends on ``r = |z|`` only (phase-independent);
    ``⟨n̂⟩ = sinh²r`` and ``Σₙ Pₙ = 1``. The even-only phonon-**pair** signature
    holds **only** for the pure, undisplaced squeezed vacuum. Returns length
    ``n_max + 1``; ``n_max`` must be large enough to capture the tail
    (``≳ 6·sinh²r`` plus margin) for the normalisation to close.
    """
    r = abs(float(r))
    populations = np.zeros(n_max + 1, dtype=float)
    cosh_r = math.cosh(r)
    tanh_r = math.tanh(r)
    for k in range(n_max // 2 + 1):
        # log of the combinatorial prefactor (2k)!/(2^k k!)² — lgamma-stable at large k.
        log_coeff = math.lgamma(2 * k + 1) - 2.0 * k * math.log(2.0) - 2.0 * math.lgamma(k + 1)
        populations[2 * k] = math.exp(log_coeff) * tanh_r ** (2 * k) / cosh_r
    return populations


def check_fock_truncation(
    state: qutip.Qobj,
    *,
    tolerance: float = FOCK_CONVERGENCE_TOLERANCE,
    window: int = 2,
) -> tuple[ResultWarning, ...]:
    """Parity-aware Fock-truncation guard for a single-mode readout (§13/§15).

    Computes a **parity-aware edge-window** metric ``p_tail = Σ`` over the last
    ``window`` Fock levels (excluding the ground state) and classifies it against
    the §13 four-band ladder (ε = ``tolerance``, default
    :data:`~iontrap_dynamics.conventions.FOCK_CONVERGENCE_TOLERANCE`): ``< ε/10``
    OK (silent); ``[ε/10, ε)`` → :class:`FockConvergenceWarning` (Level 1);
    ``[ε, 10ε)`` → :class:`FockQualityWarning` (Level 2); ``≥ 10ε`` → raises
    :class:`ConvergenceError` (Level 3). Same ladder and classes as the solver's
    §13 guard — only the metric is parity-honest.

    ``window = 2`` (default) is the minimal parity-honest **edge** metric: the top
    two levels always straddle both parities, so the one populated-parity edge
    level enters the sum for either ``fock_dim`` parity. This closes the
    parity-blind hole in a single-level check — a squeezed vacuum in an
    even-dimensional Fock space has **zero** population in its topmost (odd) level,
    so a top-level-only test would report convergence even when the even tail is
    saturated. A *wide* window would instead over-flag well-truncated moderate-``N``
    states (it would sum bulk, not tail), so keep it at the edge; the ground state
    is always excluded.

    Returns the :class:`ResultWarning` records (empty tuple when converged);
    warns on the Python channel at Levels 1–2 and raises at Level 3.
    """
    if tolerance <= 0.0:
        raise ConventionError(
            f"fock tolerance must be positive; got {tolerance!r}. Zero disables the "
            "convergence check entirely — a silent-degradation hazard (CONVENTIONS.md §15)."
        )
    if window < 1:
        raise ConventionError(
            f"fock truncation window must be positive; got {window!r}. A zero-width "
            "window disables the edge-tail check entirely — a silent-degradation hazard "
            "(CONVENTIONS.md §15)."
        )
    populations = phonon_number_distribution(state)
    fock_dim = len(populations)
    width = max(0, min(window, fock_dim - 1))  # top `window` levels, never the ground state
    p_tail = float(populations[fock_dim - width :].sum()) if width > 0 else 0.0
    diagnostics = {
        "fock_dim": fock_dim,
        "p_tail": p_tail,
        "window_width": width,
        "tolerance_epsilon": tolerance,
    }
    if p_tail < tolerance / 10.0:
        return ()
    if p_tail >= 10.0 * tolerance:
        raise ConvergenceError(
            "Fock-truncation failure (CONVENTIONS.md §13, §15 Level 3): parity-aware tail "
            f"population p_tail = {p_tail:.3e} over the last {width} Fock levels meets or "
            f"exceeds 10·ε = {10.0 * tolerance:.3e} (N_Fock = {fock_dim}). The squeezing "
            "readout is truncation-biased; increase the Fock dimension and re-run."
        )
    if p_tail >= tolerance:
        message = (
            f"parity-aware tail population p_tail = {p_tail:.3e} over the last {width} Fock "
            f"levels exceeds ε = {tolerance:.3e} (N_Fock = {fock_dim}); squeezing-readout "
            "quality degraded (CONVENTIONS.md §15 Level 2). Increase the Fock dimension "
            "before publication use."
        )
        warnings.warn(message, FockQualityWarning, stacklevel=2)
        severity = WarningSeverity.QUALITY
    else:
        message = (
            f"parity-aware tail population p_tail = {p_tail:.3e} over the last {width} Fock "
            f"levels approaches ε = {tolerance:.3e} (N_Fock = {fock_dim}); the truncation is "
            "close to its envelope (CONVENTIONS.md §15 Level 1). Consider a larger Fock "
            "dimension for publication-grade squeezing readout."
        )
        warnings.warn(message, FockConvergenceWarning, stacklevel=2)
        severity = WarningSeverity.CONVERGENCE
    return (
        ResultWarning(
            severity=severity,
            category="fock_truncation",
            message=message,
            diagnostics=diagnostics,
        ),
    )


@dataclass(frozen=True)
class GaussianReadout:
    """Single-mode Gaussian readout bundle (CONVENTIONS.md §26.4, observable-only)."""

    covariance: np.ndarray
    mean: np.ndarray
    symplectic_eigenvalue: float
    squeezing_parameter: float
    mean_squeezed_occupation: float
    coherent_amplitude: complex
    mean_occupation: float


def gaussian_readout(
    state: qutip.Qobj,
    *,
    check_truncation: bool = True,
    tolerance: float = FOCK_CONVERGENCE_TOLERANCE,
    window: int = 2,
) -> GaussianReadout:
    """Full single-mode Gaussian readout of a state (ket or density matrix).

    Runs the parity-aware Fock-truncation guard (:func:`check_fock_truncation`)
    first by default (§13/§15) — warns or raises on an under-resolved squeezed
    state rather than returning a silently-biased ``r``/``ν``. Set
    ``check_truncation=False`` to skip the guard for a state known to be well
    within its cutoff. The scalar functionals below (:func:`covariance_matrix`,
    :func:`squeezing_parameter`, …) stay **unguarded** primitives — the guard
    needs the density matrix, which this composed readout has.
    """
    if check_truncation:
        check_fock_truncation(state, tolerance=tolerance, window=window)
    cov, mean = covariance_matrix(state)
    return GaussianReadout(
        covariance=cov,
        mean=mean,
        symplectic_eigenvalue=symplectic_eigenvalue(cov),
        squeezing_parameter=squeezing_parameter(cov),
        mean_squeezed_occupation=mean_squeezed_occupation(cov),
        coherent_amplitude=coherent_amplitude(mean),
        mean_occupation=mean_occupation(cov, mean),
    )


__all__ = [
    "GaussianReadout",
    "check_fock_truncation",
    "coherent_amplitude",
    "congruence",
    "covariance_matrix",
    "effective_temperature",
    "gaussian_entropy_bits",
    "gaussian_readout",
    "is_physical",
    "is_separable",
    "log_negativity",
    "mean_occupation",
    "mean_squeezed_occupation",
    "partial_transpose",
    "phonon_number_distribution",
    "pure_squeezed_vacuum_pn",
    "purity",
    "quadrature_operators",
    "reduced_single_mode",
    "squeezing_parameter",
    "symplectic_eigenvalue",
    "symplectic_eigenvalues",
    "symplectic_form",
]
