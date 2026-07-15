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
    # mode 2's E_N is zero to float precision (~2e-15 from the eigensolver); certify the 1×2
    # separability within a small opt-in tolerance (the strict tol=0 conservatively rejects it).
    assert gaussian.is_separable(cov, [2], tol=1e-12)


def test_is_separable_scoping_and_guards() -> None:
    cov_ent, _ = gaussian.covariance_matrix(two_mode_squeezed_vacuum(FOCK, z=0.6))
    # A thermal product has a diagonal covariance with ν well above 1, so E_N = 0 exactly
    # (robust at the strict tol = 0, unlike a coherent block whose ν = 1 carries float noise).
    cov_sep, _ = gaussian.covariance_matrix(
        _two_mode(qutip.thermal_dm(FOCK, 0.4), qutip.thermal_dm(FOCK, 0.9))
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


def test_is_separable_is_a_strict_one_sided_certificate() -> None:
    # STRICT one-sided certificate at the default tol = 0.0: True ⟺ E_N == 0 ⟺ separable.
    # Entanglement is continuous, so *any* tol > 0 would false-certify a state entangled below
    # it — even a TMSV as weak as r = 1e-10 (E_N ≈ 2.9e-10 > 0, which the earlier tol = 1e-9
    # wrongly certified) must return False. Never a false True.
    for r in (1e-5, 1e-10):
        cov = _tmsv_cov(r)  # analytic, exact; E_N = 2r/ln2 > 0 for every r > 0
        assert gaussian.log_negativity(cov, [1]) == pytest.approx(2 * r / np.log(2), rel=1e-9)
        assert not gaussian.is_separable(cov, [1])
    # Exactly separable covariances (vacuum; thermal n̄=1 ⊗ n̄=2) → E_N = 0 → certified.
    for cov_sep in (np.eye(4), np.diag([3.0, 3.0, 5.0, 5.0])):
        assert gaussian.log_negativity(cov_sep, [1]) == 0.0
        assert gaussian.is_separable(cov_sep, [1])
    # An under-truncated displaced separable has a finite-Fock floor > 0 → not certified
    # (False), never a false True; log_negativity reports the honest floor (no clamp to 0).
    cov_coarse, _ = gaussian.covariance_matrix(
        _two_mode(coherent_mode(30, 3.0), coherent_mode(30, 3.0))
    )
    assert gaussian.log_negativity(cov_coarse, [1]) > 0.0
    assert not gaussian.is_separable(cov_coarse, [1])


def test_is_separable_positive_tol_is_opt_in_not_a_certificate() -> None:
    # A positive tol is the weaker "PPT within numerical tolerance", not a strict certificate:
    # a TMSV with r < tol·ln2/2 passes it, whereas the default tol = 0 does not. tol must be
    # finite and ≥ 0.
    cov = _tmsv_cov(1e-10)  # E_N ≈ 2.9e-10
    assert not gaussian.is_separable(cov, [1])  # strict default → not certified
    assert gaussian.is_separable(cov, [1], tol=1e-9)  # opt-in "PPT within 1e-9" — NOT a certificate
    for bad_tol in (-1e-9, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="tol must be finite"):
            gaussian.is_separable(cov, [1], tol=bad_tol)


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


# --- GT5: effective temperature --------------------------------------------------

_HBAR, _K_B = 1.054571817e-34, 1.380649e-23  # SI (CODATA 2018)


def _bose(temp: float, omega: float) -> float:
    return float(1.0 / np.expm1(_HBAR * omega / (_K_B * temp)))


def test_effective_temperature_round_trip_and_monotonic() -> None:
    omega = 2 * np.pi * 1.5e6
    temps = [gaussian.effective_temperature(n, omega) for n in (0.05, 0.5, 2.0, 10.0)]
    for nbar, t in zip((0.05, 0.5, 2.0, 10.0), temps, strict=True):
        assert _bose(t, omega) == pytest.approx(nbar, rel=1e-9)  # energy-equivalent round-trip
    assert all(later > earlier for earlier, later in itertools.pairwise(temps))  # increases in n̄


def test_effective_temperature_thermal_dm_round_trip() -> None:
    # Full toolbox round-trip through an actual thermal state: n̄ = mean_occupation(thermal_dm),
    # then T_eff, then the Bose occupation at T_eff recovers n̄.
    omega = 2 * np.pi * 1e6
    for n_set in (0.3, 1.0, 2.5):
        cov, d = gaussian.covariance_matrix(qutip.thermal_dm(FOCK, n_set))
        nbar = gaussian.mean_occupation(cov, d)
        assert nbar == pytest.approx(n_set, rel=1e-3)
        assert _bose(gaussian.effective_temperature(nbar, omega), omega) == pytest.approx(
            nbar, rel=1e-3
        )


def test_effective_temperature_continuity_and_quantum_scale() -> None:
    omega = 2 * np.pi * 1e6
    assert gaussian.effective_temperature(0.0, omega) == 0.0  # vacuum / zero occupation → 0 K
    # n̄ = 1 → T = (ℏω_loc / k_B) / ln 2 (the mode's quantum-temperature scale over ln 2).
    assert gaussian.effective_temperature(1.0, omega) == pytest.approx(
        _HBAR * omega / _K_B / np.log(2), rel=1e-9
    )


def test_effective_temperature_first_moment_aware_marginal() -> None:
    # A pure squeezed + displaced marginal: thermal-core (ν−1)/2 ≈ 0 but n̄ > 0 ⇒ T_eff > 0.
    cov, d = gaussian.covariance_matrix(squeezed_coherent_mode(FOCK, z=0.5, alpha=1.2))
    assert gaussian.effective_temperature(gaussian.mean_occupation(cov, d), 2 * np.pi * 1e6) > 0.0


def test_effective_temperature_guards() -> None:
    omega = 2 * np.pi * 1e6
    # non-finite n̄ (NaN, +inf) and negative n̄ are rejected (not an opaque ZeroDivisionError).
    for bad_n in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="n̄ must be"):
            gaussian.effective_temperature(bad_n, omega)
    for bad_w in (0.0, -omega, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="omega_loc must be"):
            gaussian.effective_temperature(0.5, bad_w)


# --- GT3a: generic symplectic congruence V ↦ S V Sᵀ ------------------------------


def _squeeze(r: float) -> np.ndarray:
    return np.diag([np.exp(r), np.exp(-r)])  # single-mode squeeze (symplectic)


def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])  # single-mode phase rotation (symplectic)


def test_congruence_identity_and_squeeze() -> None:
    v = np.diag([2.0, 3.0])  # a physical V (ν = √6)
    assert np.allclose(gaussian.congruence(np.eye(2), v), v)  # identity leaves V unchanged
    out = gaussian.congruence(_squeeze(0.4), np.eye(2))  # squeeze the vacuum
    assert np.allclose(out, np.diag([np.exp(0.8), np.exp(-0.8)]))  # V ↦ diag(e^{2r}, e^{-2r})
    assert gaussian.symplectic_eigenvalue(out) == pytest.approx(1.0)  # still pure (ν = 1)


def test_congruence_preserves_symplectic_spectrum_and_physicality() -> None:
    # A symplectic map preserves the Williamson spectrum ν_i (hence purity/entropy) and
    # physicality — squeeze, rotation, and a two-mode beamsplitter.
    cov, _ = gaussian.covariance_matrix(qutip.thermal_dm(FOCK, 0.8))  # ν = 2.6
    for s in (_squeeze(0.6), _rotation(0.7)):
        out = gaussian.congruence(s, cov)
        assert gaussian.symplectic_eigenvalues(out) == pytest.approx(
            gaussian.symplectic_eigenvalues(cov), rel=1e-6
        )
        assert gaussian.is_physical(out)
    theta = np.pi / 4  # a 50:50 beamsplitter mixes the two modes (symplectic)
    c, s = np.cos(theta), np.sin(theta)
    bs = np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])
    cov2, _ = gaussian.covariance_matrix(
        _two_mode(qutip.thermal_dm(24, 0.3), qutip.thermal_dm(24, 1.0))
    )
    assert np.allclose(
        np.sort(gaussian.symplectic_eigenvalues(gaussian.congruence(bs, cov2))),
        np.sort(gaussian.symplectic_eigenvalues(cov2)),
    )


def test_congruence_inverse_round_trip() -> None:
    # S symplectic ⇒ S⁻¹ symplectic; congruence(S⁻¹, congruence(S, V)) = V.
    s = _squeeze(0.5)
    v = np.diag([2.0, 3.0])
    assert np.allclose(gaussian.congruence(np.linalg.inv(s), gaussian.congruence(s, v)), v)


def test_congruence_scale_aware_gate_is_block_local() -> None:
    # Regression (adversarial): a large but genuinely symplectic squeeze block A = diag(λ, 1/λ)
    # must NOT inflate the symplectic-tolerance budget for a small, genuinely NON-symplectic
    # block B alongside it. A global 1e-9·‖S‖² tolerance false-accepts B (its 2e-4 Ω-violation
    # fits inside 1e-9·λ²); the per-entry ‖row_i‖·‖row_j‖ scale rejects it block-locally.
    lam = 500.0
    c = np.sqrt(1.0 + 2e-4)  # B Ω Bᵀ = 1.0002·Ω — a real 2e-4 symplecticity violation
    a = np.diag([lam, 1.0 / lam])  # exactly symplectic, large ‖·‖
    b = np.diag([c, c])  # non-symplectic
    s_bad = np.block([[a, np.zeros((2, 2))], [np.zeros((2, 2)), b]])
    with pytest.raises(ValueError, match="not symplectic"):
        gaussian.congruence(s_bad, np.eye(4))
    # …while the SAME large squeeze paired with a genuinely symplectic rotation is accepted and
    # preserves ν (the fix must not false-reject legitimate large-norm block-diagonal maps).
    theta = 0.6
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    s_ok = np.block([[a, np.zeros((2, 2))], [np.zeros((2, 2)), rot]])
    cov = np.diag([2.0, 2.0, 3.0, 3.0])  # a physical (thermal) V, ν = [2, 3]
    out = gaussian.congruence(s_ok, cov)
    assert np.allclose(
        np.sort(gaussian.symplectic_eigenvalues(out)),
        np.sort(gaussian.symplectic_eigenvalues(cov)),
    )


def _worst_relative_omega_residual(s: np.ndarray, n_modes: int) -> float:
    omega = gaussian.symplectic_form(n_modes)
    residual = np.abs(s @ omega @ s.T - omega)
    row_norms = np.linalg.norm(s, axis=1)
    return float(np.max(residual / np.maximum(np.outer(row_norms, row_norms), 1.0)))


def test_congruence_relative_gate_rejects_tuned_near_symplectic() -> None:
    # Regression (adversarial rounds 2–3): a near-symplectic S whose per-entry relative residual
    # is tuned to sit just under a *looser* 1e-9 budget (but well above the ~2e-15 genuine
    # rounding floor) is caught by the tightened 1e-11 gate. Two constructions, both silently
    # corrupting the Williamson spectrum if accepted — an entangling strong two-mode squeezer with
    # one corrupted entry, and a near-symplectic single-mode S applied to a strongly-squeezed
    # (ill-conditioned) V where the ν-damage is amplified by ~√cond(V). No *absolute* Ω-residual
    # bound distinguishes these from genuine strong squeezes; the rounding-referenced relative gate
    # does, independent of input conditioning.
    ch, sh = np.cosh(6.0), np.sinh(6.0)
    s_entangling = np.array([[ch, 0, sh, 0], [0, ch, 0, -sh], [sh, 0, ch, 0], [0, -sh, 0, ch]])
    s_entangling[2, 0] += 3.63e-7  # non-symplectic (det ≠ 1); relative residual ≈ 9e-10
    assert 1e-11 < _worst_relative_omega_residual(s_entangling, 2) < 1e-9
    with pytest.raises(ValueError, match="not symplectic"):
        gaussian.congruence(s_entangling, np.eye(4))

    sv = 3.0
    s_near = np.diag([np.exp(sv) + 5e-10 * np.exp(sv), np.exp(-sv)])  # relative residual ≈ 5e-10
    assert 1e-11 < _worst_relative_omega_residual(s_near, 1) < 1e-9
    v_squeezed = np.diag([2500.0, 1.0 / 2500.0])  # pure squeezed vacuum, ν = 1, cond 6.25e6
    assert gaussian.is_physical(v_squeezed)
    with pytest.raises(ValueError, match="not symplectic"):
        gaussian.congruence(s_near, v_squeezed)  # rejected regardless of input conditioning


def test_congruence_accepts_strong_squeeze_no_absolute_false_reject() -> None:
    # Regression (adversarial round 3): the gate must be *relative*, not absolute. A genuine strong
    # squeeze's absolute |SΩSᵀ − Ω| balloons to ~1e-6 from catastrophic e^{2r} cancellation while
    # the map stays exactly symplectic (relative residual ~1e-16), and its output is still
    # certifiable (cond < 1e12). An absolute cap would false-reject it; the relative gate accepts.
    rot = np.array(
        [[np.cos(np.pi / 4), np.sin(np.pi / 4)], [-np.sin(np.pi / 4), np.cos(np.pi / 4)]]
    )
    s = rot @ np.diag([np.exp(12.0), np.exp(-12.0)]) @ rot.T  # exact symplectic; abs residual ~2e-6
    assert _worst_relative_omega_residual(s, 1) < 1e-13  # genuinely symplectic (rounding only)
    v = np.linalg.inv(s)  # a physical pure state (ν = 1), cond ≈ 2.6e10 < 1e12 (certifiable)
    out = gaussian.congruence(s, v)  # accepted, not false-rejected by any absolute cap
    assert gaussian.symplectic_eigenvalues(out) == pytest.approx(np.ones(1), abs=1e-4)
    # …and a clean strong two-mode squeezer (r = 6) is likewise accepted, preserving ν.
    ch, sh = np.cosh(6.0), np.sinh(6.0)
    s_tms = np.array([[ch, 0, sh, 0], [0, ch, 0, -sh], [sh, 0, ch, 0], [0, -sh, 0, ch]])
    out2 = gaussian.congruence(s_tms, np.eye(4))
    assert gaussian.symplectic_eigenvalues(out2) == pytest.approx(np.ones(2), abs=1e-5)


def test_congruence_relative_nu_preservation_is_safe_in_physical_regime() -> None:
    # Regression (adversarial round 4): a relative symplecticity gate guarantees RELATIVE
    # ν-preservation (the physically meaningful invariant — purity/entropy/physicality turn on ν
    # relative to the vacuum floor 1). A near-symplectic uniform dilation S = s·𝟙 with det = 1+9e-12
    # is within the 1e-11 relative gate and shifts ν by only a RELATIVE ~9e-12. The *absolute*
    # shift = (rel)·ν scales with ν, so it stays ≪ the 1e-4 physicality tolerance throughout the
    # trapped-ion regime (ν ≲ 1e4); only a physically-unreachable ν ≳ 1e7 makes it reach 1e-4, and
    # even then relative preservation and physicality (ν ≥ 1) still hold.
    s = np.sqrt(1.0 + 9e-12)  # det(s·𝟙) = 1 + 9e-12: inside the relative gate
    for nu_in in (1.0, 100.0, 1.0e4):  # vacuum … hot thermal, spanning the physical range
        out = gaussian.congruence(s * np.eye(2), nu_in * np.eye(2))
        nu_out = gaussian.symplectic_eigenvalues(out)[0]
        assert abs(nu_out - nu_in) / nu_in < 1e-10  # RELATIVE ν preserved (the real guarantee)
        assert abs(nu_out - nu_in) < 1e-4  # ABSOLUTE ν preserved in the physical regime
        assert nu_out >= 1.0  # output stays physical


def test_congruence_rejects_nonsymplectic_and_malformed() -> None:
    v = np.eye(2)
    with pytest.raises(ValueError, match="not symplectic"):
        gaussian.congruence(np.diag([2.0, 1.0]), v)  # SΩSᵀ ≠ Ω (a non-symplectic scaling)
    with pytest.raises(ValueError, match="to match V"):
        gaussian.congruence(np.eye(4), v)  # shape mismatch
    with pytest.raises(ValueError, match="must be real"):
        gaussian.congruence(np.array([[1.0 + 1j, 0.0], [0.0, 1.0]]), v)  # complex S
    with pytest.raises(ValueError, match="non-finite"):
        gaussian.congruence(np.array([[np.nan, 0.0], [0.0, 1.0]]), v)  # NaN S
