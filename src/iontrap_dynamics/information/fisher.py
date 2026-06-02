# SPDX-License-Identifier: MIT
"""Classical and quantum Fisher information for parameter estimation.

WI-1 of WP-01 (estimation / Darwinism service surface; dispatch ``EDA``). Every
symbol here is **application-agnostic** — defined on generic inputs (a discrete
probability distribution, a quantum state, a generator, or a linear-Gaussian
model) with no reference to any consuming application.

The module provides:

- :func:`quantum_fisher_information_trajectory` — the symmetric-logarithmic-
  derivative (SLD) quantum Fisher information ``F_Q[ρ, G]`` for a unitary
  parameter encoding ``ρ(θ) = e^{-iθG} ρ e^{+iθG}``, evaluated for each state
  in a trajectory. It mirrors the *trajectory-evaluator* shape of
  :mod:`iontrap_dynamics.entanglement` (the QFI is nonlinear in ρ and so does
  not fit the ``operator → qutip.expect`` observable shape): run the solver
  with ``storage_mode=StorageMode.EAGER`` and apply this as post-processing.
- :func:`classical_fisher_information` — the classical Fisher information of a
  discrete model from its probabilities and their parameter derivative.
- :func:`cramer_rao_bound` — the single-parameter Cramér–Rao bound ``1 / F``.
- :func:`linear_gaussian_fisher` — the Fisher matrix ``F = AᵀΣ⁻¹A`` of a
  linear-Gaussian model.

Conventions
-----------

(CONVENTIONS.md §19, staged for the shared v0.3 Convention Freeze — see
``WP/FREEZE-v0.3.md``.) The SLD quantum Fisher information of ``ρ`` under a
Hermitian generator ``G`` is

    F_Q[ρ, G] = 2 Σ_{j,k : λ_j + λ_k > 0} (λ_j − λ_k)² / (λ_j + λ_k) · |⟨j|G|k⟩|²

over the eigendecomposition ``ρ = Σ_j λ_j |j⟩⟨j|``. For a pure state this
reduces to ``F_Q = 4 Var_ρ(G)``. Eigenpairs whose sum ``λ_j + λ_k`` falls below
:data:`_SLD_EIGENVALUE_CUTOFF` are dropped — the SLD is undefined on the kernel
of ``ρ`` — which is the documented cutoff for the mixed-state form.

For the canonical phase-estimation oracle with the collective generator
``J_z = ½ Σ_i σ_z^{(i)}``: ``F_Q(GHZ) = N²`` (the Heisenberg limit) and
``F_Q(product) = N`` (the standard quantum limit). The Braunstein–Caves
inequality ``CFI ≤ QFI`` holds for every measurement of the encoded parameter.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import qutip
from numpy.typing import NDArray

from ..hilbert import HilbertSpace
from ._common import _ensure_density

_SLD_EIGENVALUE_CUTOFF: float = 1e-12
"""Floor on ``λ_j + λ_k`` below which an SLD eigenpair is dropped (mixed-state QFI)."""

_PROBABILITY_TOLERANCE: float = 1e-9
"""Tolerance on ``Σ_x p_x = 1`` and on the non-negativity of probabilities."""


def quantum_fisher_information_trajectory(
    states: Sequence[qutip.Qobj],
    *,
    hilbert: HilbertSpace,
    generator: qutip.Qobj,
) -> NDArray[np.float64]:
    """SLD quantum Fisher information ``F_Q[ρ, G]`` along a trajectory.

    For each state in ``states`` the SLD-QFI under the Hermitian ``generator``
    ``G`` is computed (see the module conventions). The QFI quantifies the
    sensitivity of ``ρ(θ) = e^{-iθG} ρ e^{+iθG}`` to the encoded parameter
    ``θ`` at ``θ = 0``.

    Parameters
    ----------
    states
        Sequence of QuTiP states — kets or density matrices — all on the
        Hilbert space described by ``hilbert``.
    hilbert
        :class:`HilbertSpace` used as the ground-truth for the total dimension;
        the ``generator`` and every state are checked against it.
    generator
        Hermitian generator ``G`` (a ``qutip.Qobj`` on the same Hilbert space).
        Typically a collective spin operator such as ``J_z = ½ Σ_i σ_z^{(i)}``.

    Returns
    -------
    NDArray[np.float64]
        QFI trajectory, length ``len(states)``, entries ``≥ 0``.

    Raises
    ------
    ValueError
        If ``generator`` is not Hermitian, or if ``generator`` or any state
        does not match ``hilbert.total_dim``.

    Notes
    -----
    The QFI is nonlinear in ``ρ`` — the solver must run with
    ``storage_mode=StorageMode.EAGER`` so the states are retained, and this
    evaluator is applied afterwards (the same pattern as
    :mod:`iontrap_dynamics.entanglement`).
    """
    dim = hilbert.total_dim
    if generator.shape != (dim, dim):
        raise ValueError(
            f"quantum_fisher_information_trajectory: generator has shape "
            f"{generator.shape}; expected ({dim}, {dim}) for this Hilbert space"
        )
    generator_matrix = np.asarray(generator.full(), dtype=np.complex128)
    if not np.allclose(generator_matrix, generator_matrix.conj().T):
        raise ValueError("quantum_fisher_information_trajectory: generator must be Hermitian")
    values = np.empty(len(states), dtype=np.float64)
    for idx, state in enumerate(states):
        rho = _ensure_density(state)
        if rho.shape != (dim, dim):
            raise ValueError(
                f"quantum_fisher_information_trajectory: state {idx} has shape "
                f"{rho.shape}; expected ({dim}, {dim}) for this Hilbert space"
            )
        rho_matrix = np.asarray(rho.full(), dtype=np.complex128)
        values[idx] = _sld_quantum_fisher_information(rho_matrix, generator_matrix)
    return values


def classical_fisher_information(
    probabilities: NDArray[np.float64] | Sequence[float],
    *,
    parameter_derivative: NDArray[np.float64] | Sequence[float],
) -> float:
    """Classical Fisher information ``Σ_x (∂_θ p_x)² / p_x`` of a discrete model.

    Parameters
    ----------
    probabilities
        Outcome probabilities ``p_x(θ)`` at the working point. Must be
        non-negative and sum to 1 (within :data:`_PROBABILITY_TOLERANCE`).
    parameter_derivative
        The derivatives ``∂_θ p_x`` at the same point, in the same order /
        shape as ``probabilities``.

    Returns
    -------
    float
        The classical Fisher information ``≥ 0``. Outcomes with ``p_x = 0``
        contribute nothing; a non-zero derivative at a zero-probability outcome
        is rejected as ill-posed (see Raises) rather than silently dropped.

    Raises
    ------
    ValueError
        If the shapes differ, a probability is negative, the probabilities do
        not sum to 1, or a zero-probability outcome carries a non-zero
        derivative (the Fisher information would diverge).
    """
    p = np.asarray(probabilities, dtype=np.float64)
    dp = np.asarray(parameter_derivative, dtype=np.float64)
    if p.shape != dp.shape:
        raise ValueError(
            f"classical_fisher_information: probabilities {p.shape} and "
            f"parameter_derivative {dp.shape} must have the same shape"
        )
    if p.size == 0:
        raise ValueError("classical_fisher_information: probabilities must be non-empty")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(dp))):
        raise ValueError(
            "classical_fisher_information: probabilities and parameter_derivative must be finite"
        )
    if np.any(p < -_PROBABILITY_TOLERANCE):
        raise ValueError("classical_fisher_information: probabilities must be non-negative")
    total = float(np.sum(p))
    if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
        raise ValueError(f"classical_fisher_information: probabilities must sum to 1; got {total}")
    # A zero-probability outcome carrying a non-zero derivative makes the Fisher
    # information diverge — that is an ill-posed model, not a finite value to
    # truncate silently. Reject it rather than masking it away.
    zero_probability = p <= 0.0
    if np.any(zero_probability & (np.abs(dp) > _PROBABILITY_TOLERANCE)):
        raise ValueError(
            "classical_fisher_information: a zero-probability outcome has a non-zero "
            "derivative — the Fisher information diverges and the model is ill-posed"
        )
    support = p > 0.0
    contributions = (dp[support] ** 2) / p[support]
    return float(np.sum(contributions))


def cramer_rao_bound(fisher: float) -> float:
    """Single-parameter Cramér–Rao bound ``Var(θ̂) ≥ 1 / F``.

    Parameters
    ----------
    fisher
        A (classical or quantum) Fisher information; must be strictly positive.

    Returns
    -------
    float
        The bound ``1 / fisher`` on the variance of any unbiased estimator.

    Raises
    ------
    ValueError
        If ``fisher`` is not a finite, strictly-positive number.
    """
    if not np.isfinite(fisher) or fisher <= 0.0:
        raise ValueError(
            f"cramer_rao_bound: Fisher information must be a finite positive number; got {fisher}"
        )
    return 1.0 / fisher


def linear_gaussian_fisher(
    *,
    A: NDArray[np.float64] | Sequence[Sequence[float]],
    sigma: NDArray[np.float64] | Sequence[Sequence[float]],
) -> NDArray[np.float64]:
    """Fisher matrix ``F = AᵀΣ⁻¹A`` of a linear-Gaussian model.

    Model ``y = A θ + ε`` with ``ε ~ N(0, Σ)``: the Fisher information matrix
    is ``F = AᵀΣ⁻¹A`` and its inverse is the Cramér–Rao covariance bound.

    Parameters
    ----------
    A
        Design / Jacobian matrix of shape ``(n_obs, n_params)``.
    sigma
        Noise covariance ``Σ`` of shape ``(n_obs, n_obs)`` — symmetric and
        positive-definite.

    Returns
    -------
    NDArray[np.float64]
        The ``(n_params, n_params)`` Fisher matrix.

    Raises
    ------
    ValueError
        If ``A`` or ``sigma`` has non-finite entries, ``A`` is not 2-D,
        ``sigma`` does not match ``A``'s observation count, or ``sigma`` is not
        symmetric positive-definite.
    """
    a_mat = np.asarray(A, dtype=np.float64)
    sigma_mat = np.asarray(sigma, dtype=np.float64)
    if not (np.all(np.isfinite(a_mat)) and np.all(np.isfinite(sigma_mat))):
        raise ValueError("linear_gaussian_fisher: A and sigma must have finite entries")
    if a_mat.ndim != 2:
        raise ValueError(
            f"linear_gaussian_fisher: A must be 2-D (n_obs, n_params); got ndim {a_mat.ndim}"
        )
    n_obs = a_mat.shape[0]
    if sigma_mat.shape != (n_obs, n_obs):
        raise ValueError(
            f"linear_gaussian_fisher: sigma must be ({n_obs}, {n_obs}) to match A's "
            f"{n_obs} observations; got {sigma_mat.shape}"
        )
    if not np.allclose(sigma_mat, sigma_mat.T):
        raise ValueError("linear_gaussian_fisher: sigma must be symmetric")
    # Symmetry alone is not enough: an indefinite Σ would invert to a
    # nonsensical (non-PSD) Fisher matrix. Cholesky succeeds iff Σ is
    # positive-definite, so it is the cheap correctness gate here.
    try:
        np.linalg.cholesky(sigma_mat)
    except np.linalg.LinAlgError as exc:
        raise ValueError("linear_gaussian_fisher: sigma must be positive-definite") from exc
    sigma_inv = np.linalg.inv(sigma_mat)
    fisher_matrix = a_mat.T @ sigma_inv @ a_mat
    return np.asarray(fisher_matrix, dtype=np.float64)


def _sld_quantum_fisher_information(
    rho_matrix: NDArray[np.complex128],
    generator_matrix: NDArray[np.complex128],
) -> float:
    """SLD-QFI of a density matrix under a Hermitian generator (dense form)."""
    # Hermitise ρ before the eigendecomposition to absorb round-off.
    rho_h = 0.5 * (rho_matrix + rho_matrix.conj().T)
    eigvals, eigvecs = np.linalg.eigh(rho_h)
    # Clip tiny negative eigenvalues from numerical noise; they sit in ρ's kernel.
    lam = np.clip(eigvals.real, 0.0, None)
    # Generator in ρ's eigenbasis: g_jk = ⟨j|G|k⟩.
    g_eigenbasis = eigvecs.conj().T @ generator_matrix @ eigvecs
    lam_sum = lam[:, None] + lam[None, :]
    lam_diff = lam[:, None] - lam[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        coeff = np.where(
            lam_sum > _SLD_EIGENVALUE_CUTOFF,
            (lam_diff * lam_diff) / lam_sum,
            0.0,
        )
    return 2.0 * float(np.sum(coeff * np.abs(g_eigenbasis) ** 2))


__all__ = [
    "classical_fisher_information",
    "cramer_rao_bound",
    "linear_gaussian_fisher",
    "quantum_fisher_information_trajectory",
]
