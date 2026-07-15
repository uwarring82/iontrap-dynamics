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
    # invariant ν is the tell; is_physical (Williamson) rejects them too.
    for v_bad, nu in ((np.diag([4.0, 0.0625]), 0.5), (np.diag([2.7, 0.3]), 0.9)):
        assert not gaussian.is_physical(v_bad)
        assert gaussian.symplectic_eigenvalues(v_bad)[0] == pytest.approx(nu, rel=1e-6)
        with pytest.raises(ValueError, match="ν < 1"):
            func(v_bad)


@pytest.mark.parametrize("func", _FUNCTIONALS)
def test_functionals_reject_ill_conditioned_covariance(func: _Functional) -> None:
    # cond(V) ≳ 1e12 is outside the certified numerical range (the spectrum would be
    # rounding-dominated), so symplectic_eigenvalues refuses (§15). An extremely
    # scale-asymmetric diagonal (cond 4e18) and a near-singular V ≈ 2.5e10·[[1,−1],[−1,1]]
    # (cond ≈ 1.5e16, where the naive eig(iΩV) returned a spurious ν ≈ 152) must raise.
    v_singular = np.array(
        [
            [24999999999.999996, -24999999999.999992],
            [-24999999999.999992, 24999999999.99999],
        ]
    )
    for bad in (np.diag([1e9, 2.5e-10]), v_singular):
        with pytest.raises(ValueError, match="condition number"):
            gaussian.symplectic_eigenvalues(bad)
        with pytest.raises(ValueError, match="condition number"):
            func(bad)


def test_physical_but_uncertifiable_raises_not_unphysical() -> None:
    # An analytic PURE squeeze diag(1e6, 1e-6) is physical (det V = 1, ν = 1) yet has
    # cond = 1e12 — outside the certified numerical range. It must raise a certification
    # error, NOT be mislabelled unphysical (is_physical False / a "ν < 1" verdict).
    v_phys_uncert = np.diag([1e6, 1e-6])
    assert np.linalg.det(v_phys_uncert) == pytest.approx(1.0)  # genuinely physical, ν = 1
    with pytest.raises(ValueError, match="condition number"):
        gaussian.symplectic_eigenvalues(v_phys_uncert)
    with pytest.raises(ValueError, match="condition number"):
        gaussian.is_physical(v_phys_uncert)  # raises, not False
    with pytest.raises(ValueError, match="condition number"):
        gaussian.purity(v_phys_uncert)


def test_symplectic_eigenvalues_indefinite_keeps_distinct_pairs() -> None:
    # The indefinite fallback must return one ν per ±pair, keeping DISTINCT values — not the
    # N largest moduli. diag(3,3,−4,−4) has |eig(iΩV)| = [3,3,4,4]; the answer is [3, 4]
    # (the earlier [n:] slice would drop the ν = 3 and return [4, 4]).
    nus = gaussian.symplectic_eigenvalues(np.diag([3.0, 3.0, -4.0, -4.0]))
    assert np.allclose(np.sort(nus), [3.0, 4.0])
    # A degenerate indefinite block keeps multiplicity: diag(3,3,−3,−3) → [3, 3].
    assert np.allclose(gaussian.symplectic_eigenvalues(np.diag([3.0, 3.0, -3.0, -3.0])), [3.0, 3.0])


def test_is_physical_is_scale_and_correlation_invariant() -> None:
    # is_physical realises §27.2's "PSD candidate ⇒ ν_i ≥ 1" equivalence (Williamson form),
    # which is scale- AND correlation-invariant. Physical states → True; an indefinite V and
    # every well-conditioned ν = 0.5 violation → False, including the two forms that defeat a
    # direct min-eig(V+iΩ) test: diagonal scale asymmetry, and strong off-diagonal correlation
    # (whose V+iΩ violation shrinks below an absolute tol even after diagonal equilibration).
    assert gaussian.is_physical(np.eye(4))
    assert gaussian.is_physical(gaussian.covariance_matrix(coherent_mode(20, 2.0))[0])
    assert not gaussian.is_physical(np.diag([3.0, 3.0, -3.0, -3.0]))  # indefinite
    assert not gaussian.is_physical(np.diag([4.0, 0.0625]))  # ν = 0.5, scale asymmetry
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


# --- GT4: arbitrary-cut logarithmic negativity + separability scoping -------------


def test_log_negativity_tmsv_oracle_over_r() -> None:
    # E_N(TMSV, r) = 2r/ln2 [bits], the full-sum formula (not smallest-only).
    for r in (0.3, 0.6, 1.0):
        cov, _ = gaussian.covariance_matrix(two_mode_squeezed_vacuum(FOCK, z=r))
        assert gaussian.log_negativity(cov, [1]) == pytest.approx(2 * r / np.log(2), rel=1e-2)


@pytest.mark.filterwarnings("ignore:Matrix is ill-conditioned")  # qutip.negativity internal sqrtm
def test_log_negativity_matches_fock_negativity() -> None:
    # Independent oracle: the covariance E_N must equal qutip.negativity(logarithmic=True)
    # (= log₂‖ρ^{T_A}‖₁) computed on the truncated density matrix itself.
    r = 0.6
    state = two_mode_squeezed_vacuum(FOCK, z=r)
    cov, _ = gaussian.covariance_matrix(state)
    fock_en = float(qutip.negativity(qutip.ket2dm(state), 0, logarithmic=True))
    assert gaussian.log_negativity(cov, [1]) == pytest.approx(fock_en, rel=1e-3)


def test_log_negativity_symmetric_in_bipartition() -> None:
    cov, _ = gaussian.covariance_matrix(two_mode_squeezed_vacuum(FOCK, z=0.7))
    assert gaussian.log_negativity(cov, [0]) == pytest.approx(gaussian.log_negativity(cov, [1]))


def test_log_negativity_multimode_1x2_cut() -> None:
    # TMSV on modes {0,1} ⊗ vacuum mode 2 (analytic, exact — no truncation). Cutting off the
    # separable mode 2 → 0 (certified separable); cutting a member of the entangled pair
    # (mode 0 vs {1,2}) → 2r/ln2. Uses the FULL PT spectrum sum.
    r = 0.6
    cov = np.zeros((6, 6))
    cov[:4, :4], cov[4:, 4:] = _tmsv_cov(r), np.eye(2)
    assert gaussian.log_negativity(cov, [2]) == pytest.approx(0.0, abs=1e-12)
    assert gaussian.log_negativity(cov, [0]) == pytest.approx(2 * r / np.log(2), rel=1e-9)
    assert gaussian.is_separable(cov, [2])  # 1×2 cut: mode 2 separable from the entangled pair


def test_is_separable_scoping_and_guards() -> None:
    cov_ent, _ = gaussian.covariance_matrix(two_mode_squeezed_vacuum(FOCK, z=0.6))
    cov_sep, _ = gaussian.covariance_matrix(
        _two_mode(qutip.coherent_dm(FOCK, 1.0), qutip.thermal_dm(FOCK, 0.5))
    )
    assert not gaussian.is_separable(cov_ent, [1])
    assert gaussian.is_separable(cov_sep, [1])
    # M×N (M, N ≥ 2) → raises (PPT-bound-entangled caveat surfaced).
    with pytest.raises(ValueError, match="1×N"):
        gaussian.is_separable(np.eye(8), [0, 1])  # 2×2


def _tmsv_cov(r: float) -> np.ndarray:
    # Analytic two-mode-squeezed-vacuum covariance (vacuum-variance-1, per-mode ordering).
    c, s = np.cosh(2 * r), np.sinh(2 * r)
    return np.array([[c, 0, s, 0], [0, c, 0, -s], [s, 0, c, 0], [0, -s, 0, c]])


def test_is_separable_is_a_one_sided_certificate() -> None:
    # NEVER a false certificate: a weakly-entangled TMSV (E_N ~ 2.9e-5) must return False —
    # log_negativity stays faithful (no clamp) and is_separable's tight tol keeps True a real
    # certificate. (A 1e-4 clamp once wrongly reported this state separable.)
    cov_weak, _ = gaussian.covariance_matrix(two_mode_squeezed_vacuum(FOCK, z=1e-5))
    assert gaussian.log_negativity(cov_weak, [1]) == pytest.approx(2e-5 / np.log(2), rel=1e-2)
    assert not gaussian.is_separable(cov_weak, [1])  # entangled → not certified separable
    # A well-truncated separable product gives E_N = 0 exactly → certified separable.
    cov_ok, _ = gaussian.covariance_matrix(
        _two_mode(coherent_mode(60, 3.0), coherent_mode(60, 3.0))
    )
    assert gaussian.log_negativity(cov_ok, [1]) == 0.0
    assert gaussian.is_separable(cov_ok, [1])
    # An UNDER-truncated displaced separable has a finite-Fock floor (~1e-6) indistinguishable
    # from weak entanglement; is_separable honestly returns False (not certified) — never a
    # false True — and log_negativity reports the honest non-zero floor (not clamped to 0).
    cov_coarse, _ = gaussian.covariance_matrix(
        _two_mode(coherent_mode(30, 3.0), coherent_mode(30, 3.0))
    )
    assert gaussian.log_negativity(cov_coarse, [1]) > 1e-9
    assert not gaussian.is_separable(cov_coarse, [1])


def test_log_negativity_full_sum_two_entangled_blocks() -> None:
    # Two independent TMSV pairs straddling one cut → TWO PT eigenvalues < 1. The FULL sum
    # gives E_N = (2r1 + 2r2)/ln2; a smallest-only form would report only the larger block.
    r1, r2 = 0.5, 0.8
    cov = np.zeros((8, 8))  # modes (a1,b1,a2,b2); analytic, exact (no truncation)
    cov[:4, :4], cov[4:, 4:] = _tmsv_cov(r1), _tmsv_cov(r2)
    assert gaussian.log_negativity(cov, [1, 3]) == pytest.approx(
        2 * (r1 + r2) / np.log(2), rel=1e-9
    )
    nu_tilde = gaussian.symplectic_eigenvalues(gaussian.partial_transpose(cov, [1, 3]))
    assert int(np.sum(nu_tilde < 1.0)) == 2  # genuinely two sub-unity PT eigenvalues


def test_gt4_rejects_non_integral_mode_index() -> None:
    cov = _tmsv_cov(0.5)
    for func in (gaussian.log_negativity, gaussian.is_separable, gaussian.partial_transpose):
        with pytest.raises(ValueError, match="not an integer"):
            func(cov, [0.5])  # type: ignore[list-item]


def test_log_negativity_rejects_improper_cut_and_unphysical() -> None:
    cov, _ = gaussian.covariance_matrix(two_mode_squeezed_vacuum(FOCK, z=0.5))
    for bad_cut in ([], [0, 1]):  # empty and full — not a proper bipartition
        with pytest.raises(ValueError, match="proper bipartition"):
            gaussian.log_negativity(cov, bad_cut)
    with pytest.raises(ValueError, match="unphysical"):
        gaussian.log_negativity(np.diag([4.0, 0.0625, 1.0, 1.0]), [1])  # mode-0 ν = 0.5
