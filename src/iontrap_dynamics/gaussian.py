# SPDX-License-Identifier: MIT
"""Single-mode Gaussian phase-space readout (the ``N = 1`` covariance core).

CONVENTIONS.md §26.2 / §26.4. This module owns the **general** covariance /
symplectic core; WP-05 (SQ1/SQ2) creates only the single-mode (``N = 1``) case
it needs for the non-adiabatic-squeezing readout, and ``phase_space.py`` holds
the Wigner / readout façades over it (WP-05 R4, no-fork rule). The multimode
generalisation (quadrature ordering, the symplectic form ``Ω``, partial
transpose, Gaussian log-negativity, ``E_F``) is the Gaussian-toolbox card's
§27 extension of this module — a pure extension, never a refactor.

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
    """Return ``(V, d)`` for a **single-mode** state (ket or density matrix).

    ``V`` is the 2×2 covariance ``V_ij = ½⟨{ΔR_i, ΔR_j}⟩`` (incl. the ``x̂p̂``
    cross term) and ``d = (⟨x̂⟩, ⟨p̂⟩)`` the first moments (CONVENTIONS.md §26.2).
    Vacuum → ``V = 𝟙₂``, ``d = 0``.
    """
    if len(state.dims[0]) != 1:
        raise ValueError(
            f"covariance_matrix expects a single-mode state; got dims {state.dims}. "
            "Partial-trace to one mode first (see reduced_single_mode)."
        )
    fock_dim = state.shape[0]
    x, p = quadrature_operators(fock_dim)
    ex = float(np.real(qutip.expect(x, state)))
    ep = float(np.real(qutip.expect(p, state)))
    exx = float(np.real(qutip.expect(x * x, state)))
    epp = float(np.real(qutip.expect(p * p, state)))
    exp_sym = 0.5 * float(np.real(qutip.expect(x * p + p * x, state)))
    v_xx = exx - ex * ex
    v_pp = epp - ep * ep
    v_xp = exp_sym - ex * ep
    return np.array([[v_xx, v_xp], [v_xp, v_pp]], dtype=float), np.array([ex, ep], dtype=float)


def symplectic_eigenvalue(cov: np.ndarray) -> float:
    """Single-mode symplectic eigenvalue ``ν = √(det V)`` (purity / thermal core).

    ``ν = 1`` for a pure state, ``ν = 2n̄_th + 1`` for a thermal state
    (CONVENTIONS.md §26.4). Squeezing-invariant (``det V`` is unchanged by the
    symplectic squeeze).
    """
    return float(np.sqrt(max(np.linalg.det(cov), 0.0)))


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
    "covariance_matrix",
    "gaussian_readout",
    "mean_occupation",
    "mean_squeezed_occupation",
    "phonon_number_distribution",
    "pure_squeezed_vacuum_pn",
    "quadrature_operators",
    "reduced_single_mode",
    "squeezing_parameter",
    "symplectic_eigenvalue",
]
