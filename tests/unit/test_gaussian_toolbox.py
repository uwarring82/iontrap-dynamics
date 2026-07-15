# SPDX-License-Identifier: MIT
"""Unit tests for the application-agnostic Gaussian toolbox functionals (WP-07).

Dispatch GT2 lands ``purity`` and ``gaussian_entropy_bits`` over the sealed §27
symplectic spectrum. These are the physics/oracle checks that complement the §27.4
convention pins in ``tests/conventions/test_gaussian_conventions.py``: purity is
symplectic-invariant (squeezing and displacement do not change it), entropy is
extensive over independent modes, and both track the mixedness of a thermal marginal.
Later toolbox slices (GT4 log-negativity, GT5 effective temperature) extend this file.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pytest
import qutip

from iontrap_dynamics import gaussian
from iontrap_dynamics.states import (
    coherent_mode,
    squeezed_coherent_mode,
    squeezed_vacuum_mode,
    two_mode_squeezed_vacuum,
)

FOCK = 48  # generous for the r ≤ 0.7 squeezing used here (even-n pair-creation tail, §13)


def _two_mode(state_a: qutip.Qobj, state_b: qutip.Qobj) -> qutip.Qobj:
    return qutip.tensor(state_a, state_b)


# --- purity: symplectic-invariant, ≤ 1, mixedness-tracking ----------------------


def test_purity_is_one_for_pure_squeezed_and_displaced_states() -> None:
    # Squeezing (S) and displacement (D) are symplectic / phase-space translations:
    # neither adds mixedness, so μ = 1 for D(α)S(ξ)|0⟩ despite a non-identity V.
    state = squeezed_coherent_mode(FOCK, z=0.6, alpha=1.1)
    cov, _ = gaussian.covariance_matrix(state)
    assert not np.allclose(cov, np.eye(2)), "the marginal V is genuinely squeezed/displaced"
    assert gaussian.purity(cov) == pytest.approx(1.0, abs=1e-3), "pure Gaussian state → μ = 1"


def test_purity_decreases_with_thermal_occupation() -> None:
    mus = [
        gaussian.purity(gaussian.covariance_matrix(qutip.thermal_dm(FOCK, n))[0])
        for n in (0.0, 0.3, 1.0, 2.5)
    ]
    assert mus[0] == pytest.approx(1.0, abs=1e-6)
    assert all(later < earlier for earlier, later in itertools.pairwise(mus)), (
        "purity is strictly decreasing in n̄ (more thermal ⇒ more mixed)"
    )
    assert all(0.0 < m <= 1.0 + 1e-9 for m in mus), "0 < μ ≤ 1"


def test_purity_of_reduced_tmsv_marginal_matches_thermal() -> None:
    # Each arm of a TMSV is a thermal marginal with n̄ = sinh²r, so its purity is
    # 1/(2 sinh²r + 1). This is the reduced-state (partial-trace) path, not a product.
    r = 0.7
    tmsv = two_mode_squeezed_vacuum(FOCK, z=r)
    arm = tmsv.ptrace(0)
    cov, _ = gaussian.covariance_matrix(arm)
    nbar = np.sinh(r) ** 2
    assert gaussian.purity(cov) == pytest.approx(1.0 / (2 * nbar + 1), rel=1e-2)


# --- entropy: pure → 0, extensive, monotone --------------------------------------


def test_entropy_is_zero_for_pure_states() -> None:
    for state in (
        qutip.basis(FOCK, 0),
        squeezed_coherent_mode(FOCK, z=0.5, alpha=0.8),
    ):
        cov, _ = gaussian.covariance_matrix(state)
        assert gaussian.gaussian_entropy_bits(cov) == pytest.approx(0.0, abs=1e-3)


def test_entropy_is_extensive_over_independent_modes() -> None:
    # S(ρ_A ⊗ ρ_B) = S(ρ_A) + S(ρ_B): a product of two thermals sums the per-mode entropy.
    na, nb = 0.4, 1.3
    cov_a, _ = gaussian.covariance_matrix(qutip.thermal_dm(FOCK, na))
    cov_b, _ = gaussian.covariance_matrix(qutip.thermal_dm(FOCK, nb))
    cov_ab, _ = gaussian.covariance_matrix(
        _two_mode(qutip.thermal_dm(FOCK, na), qutip.thermal_dm(FOCK, nb))
    )
    s_sum = gaussian.gaussian_entropy_bits(cov_a) + gaussian.gaussian_entropy_bits(cov_b)
    assert gaussian.gaussian_entropy_bits(cov_ab) == pytest.approx(s_sum, rel=1e-4)


def test_entropy_increases_with_thermal_occupation() -> None:
    entropies = [
        gaussian.gaussian_entropy_bits(gaussian.covariance_matrix(qutip.thermal_dm(FOCK, n))[0])
        for n in (0.0, 0.3, 1.0, 2.5)
    ]
    assert entropies[0] == pytest.approx(0.0, abs=1e-3)
    assert all(later > earlier for earlier, later in itertools.pairwise(entropies)), (
        "Gaussian entropy is strictly increasing in n̄"
    )


def test_pure_global_tmsv_has_zero_entropy_but_mixed_marginals() -> None:
    # A TMSV is globally pure (S = 0) yet each arm is thermal (S > 0) — the hallmark of
    # entanglement, and a check that entropy is computed on the given V, not assumed.
    r = 0.7
    tmsv = two_mode_squeezed_vacuum(FOCK, z=r)
    cov_global, _ = gaussian.covariance_matrix(tmsv)
    cov_arm, _ = gaussian.covariance_matrix(tmsv.ptrace(0))
    assert gaussian.gaussian_entropy_bits(cov_global) == pytest.approx(0.0, abs=1e-2)
    assert gaussian.gaussian_entropy_bits(cov_arm) > 0.1


# --- guards: only bona-fide physical covariance matrices are accepted ------------

_Functional = Callable[[np.ndarray], float]
_FUNCTIONALS: list[_Functional] = [gaussian.purity, gaussian.gaussian_entropy_bits]


@pytest.mark.parametrize("func", _FUNCTIONALS)
def test_functionals_reject_non_2n_covariance(func: _Functional) -> None:
    with pytest.raises(ValueError, match="square 2N×2N"):
        func(np.eye(3))


@pytest.mark.parametrize("func", _FUNCTIONALS)
def test_functionals_reject_unphysical_covariance(func: _Functional) -> None:
    # 0.5·𝟙 sits below vacuum (ν = 0.5 < 1) — purity would return the nonsensical 2, and
    # the entropy ν≤1 branch would silently read it as pure (0). The sealed indefinite
    # counterexample diag(3,3,−3,−3) (|eig(iΩV)| ≥ 1 yet V+iΩ ≱ 0) must also be rejected.
    for bad in (0.5 * np.eye(2), np.diag([3.0, 3.0, -3.0, -3.0])):
        assert not gaussian.is_physical(bad)  # sanity: these are genuinely unphysical
        with pytest.raises(ValueError, match="unphysical"):
            func(bad)


@pytest.mark.parametrize("func", _FUNCTIONALS)
def test_functionals_reject_sub_uncertainty_covariance(func: _Functional) -> None:
    # Scale-asymmetric but well-conditioned covariances that violate the uncertainty bound
    # (ν < 1) must be rejected by the squeezing-invariant Williamson (ν ≥ 1) test — an
    # absolute min-eig(V+iΩ) tol is not squeezing-invariant and would miss them. The
    # invariant ν is the tell; is_physical (equilibrated) rejects them too.
    for v_bad, nu in ((np.diag([4.0, 0.0625]), 0.5), (np.diag([2.7, 0.3]), 0.9)):
        assert not gaussian.is_physical(v_bad)
        assert gaussian.symplectic_eigenvalues(v_bad)[0] == pytest.approx(nu, rel=1e-6)
        with pytest.raises(ValueError, match="ν < 1"):
            func(v_bad)


@pytest.mark.parametrize("func", _FUNCTIONALS)
def test_functionals_reject_ill_conditioned_covariance(func: _Functional) -> None:
    # cond(V) ≳ 1e12 is unreachable by any physical state, so the symplectic spectrum would
    # be rounding-dominated and symplectic_eigenvalues refuses to certify (§15). Both an
    # extremely scale-asymmetric diagonal (cond 4e18, ν nominally 0.5) and a near-singular
    # V ≈ 2.5e10·[[1,−1],[−1,1]] (cond ≈ 1.5e16, where the naive eig(iΩV) returned a spurious
    # ν ≈ 152) must raise rather than return a silently-wrong value.
    v_singular = np.array(
        [
            [24999999999.999996, -24999999999.999992],
            [-24999999999.999992, 24999999999.99999],
        ]
    )
    for bad in (np.diag([1e9, 2.5e-10]), v_singular):
        with pytest.raises(ValueError, match="ill-conditioned"):
            gaussian.symplectic_eigenvalues(bad)
        with pytest.raises(ValueError, match="ill-conditioned"):
            func(bad)


def test_is_physical_is_scale_and_correlation_invariant() -> None:
    # is_physical realises §27.2's "PSD candidate ⇒ ν_i ≥ 1" equivalence (Williamson form),
    # which is scale- AND correlation-invariant. Physical states → True; an indefinite V and
    # every ν = 0.5 violation → False, including the two forms that defeat a direct
    # min-eig(V+iΩ) test: extreme diagonal scale, and strong off-diagonal correlation (whose
    # V+iΩ violation shrinks below an absolute tol even after diagonal equilibration).
    assert gaussian.is_physical(np.eye(4))
    assert gaussian.is_physical(gaussian.covariance_matrix(coherent_mode(20, 2.0))[0])
    assert not gaussian.is_physical(np.diag([3.0, 3.0, -3.0, -3.0]))  # indefinite
    assert not gaussian.is_physical(np.diag([1e9, 2.5e-10]))  # ν = 0.5, extreme scale
    assert not gaussian.is_physical(np.diag([4.0, 0.0625]))  # ν = 0.5, moderate scale
    d = 100.0  # strongly correlated [[c, d], [d, c]] with c² − d² = ¼ ⇒ ν = 0.5
    correlated = np.array([[np.sqrt(d * d + 0.25), d], [d, np.sqrt(d * d + 0.25)]])
    assert gaussian.symplectic_eigenvalues(correlated)[0] == pytest.approx(0.5, rel=1e-4)
    assert not gaussian.is_physical(correlated)


def test_symplectic_eigenvalues_stable_under_strong_squeezing() -> None:
    # The SPD-stable Williamson realisation stays accurate for the ill-conditioned V a
    # squeezed state produces: it agrees with the exact single-mode ν = √det V, where the
    # naive eig(iΩV).real drifts. cond(V) here is ~e^{4r}, still far below the 1e12 floor.
    for z in (0.5, 1.0, 1.5):
        cov, _ = gaussian.covariance_matrix(squeezed_vacuum_mode(120, z=z))
        assert gaussian.symplectic_eigenvalues(cov)[0] == pytest.approx(
            gaussian.symplectic_eigenvalue(cov), rel=1e-6
        )


@pytest.mark.parametrize("func", _FUNCTIONALS)
def test_functionals_accept_generously_truncated_pure_states(func: _Functional) -> None:
    # The false-reject regression: displacement populates the top Fock level, breaking the
    # truncated commutator and driving min-eig(V+iΩ) to O(−1e-7) — past a 1e-9 tol — even
    # though the state is exactly pure. A coherent state is the most classical Gaussian
    # state; the guard's ν-tolerance (1e-4) must absorb the artifact, not raise.
    expected = 0.0 if func is gaussian.gaussian_entropy_bits else 1.0
    for state in (coherent_mode(12, 1.0), coherent_mode(20, 2.0), coherent_mode(30, 3.0)):
        cov, _ = gaussian.covariance_matrix(state)
        assert func(cov) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("func", _FUNCTIONALS)
def test_functionals_reject_nonsymmetric_covariance(func: _Functional) -> None:
    # eigvalsh silently reads one triangle, so a non-symmetric V could otherwise slip
    # through the physicality check with a wrong answer — reject it explicitly.
    with pytest.raises(ValueError, match="symmetric"):
        func(np.array([[1.0, 0.5], [0.0, 1.0]]))


@pytest.mark.parametrize("func", _FUNCTIONALS)
def test_functionals_reject_nonfinite_covariance(func: _Functional) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        func(np.array([[np.nan, 0.0], [0.0, 1.0]]))


@pytest.mark.parametrize("func", _FUNCTIONALS)
def test_functionals_reject_complex_covariance(func: _Functional) -> None:
    with pytest.raises(ValueError, match="real"):
        func(np.array([[1.0 + 0.5j, 0.0], [0.0, 1.0]]))
