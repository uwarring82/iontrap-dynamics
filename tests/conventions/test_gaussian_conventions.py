# SPDX-License-Identifier: MIT
"""Conventions test for CONVENTIONS.md §27 — the multimode Gaussian toolbox.

Implementation-facing (WP-07 dispatch GT1, conventions-before-code): this pins §27's
behavioural anchors against the GT1 multimode surface in ``gaussian.py`` —
``symplectic_form``, the multimode ``covariance_matrix``, ``symplectic_eigenvalues``
(plural, via ``|eig(iΩV)|``), ``is_physical`` (the ``V + iΩ ≥ 0`` PSD guard), and
``partial_transpose``. It **fails until GT1 lands** (those symbols do not yet exist)
and goes green with the covariance-core implementation; the §27 seal follows the green.

It never reads the ``CONVENTIONS.md`` markdown, so it stays green regardless of wording.
Anchors (per §27):
- 27.1 per-mode ordering ``R = (x̂₁,p̂₁,…,x̂_N,p̂_N)``; vacuum ``V = 𝟙_{2N}``.
- 27.2 ``Ω = ⊕J``, ``[R_i,R_j] = 2iΩ``; physicality is the Hermitian PSD test on
  ``V + iΩ`` (NOT a bare ``ν_i ≥ 1`` check); Williamson ``ν_i = |eig(iΩV)|`` (NOT SVD).
- 27.3 partial transpose flips ``p̂_B → −p̂_B``; TMSV → smallest PT eigenvalue ``e^{−2r}``.
- 27.4 first-moment-aware occupation ``n̄ = (tr V + dᵀd − 2)/4`` (NOT ``(ν−1)/2``);
  purity ``μ = ∏ 1/ν_i = 1/√det V``; Gaussian entropy ``S = Σ g(ν_i)`` in bits
  (matching ``_von_neumann_entropy_bits`` on the truncated density matrix) — GT2.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics import gaussian
from iontrap_dynamics.information._common import _von_neumann_entropy_bits
from iontrap_dynamics.states import squeezed_coherent_mode, two_mode_squeezed_vacuum

FOCK = 24


def _two_mode(state_a: qutip.Qobj, state_b: qutip.Qobj) -> qutip.Qobj:
    """A product motional state on two modes (no spin), each a ket or density matrix."""
    return qutip.tensor(state_a, state_b)


# --- 27.1 ordering + vacuum ------------------------------------------------------


def test_vacuum_covariance_is_identity() -> None:
    vac = _two_mode(qutip.basis(FOCK, 0), qutip.basis(FOCK, 0))
    cov, d = gaussian.covariance_matrix(vac)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, np.eye(4), atol=1e-6), "two-mode vacuum must give V = 𝟙₄ (§27.1)"
    assert np.allclose(d, 0.0, atol=1e-6)


def test_per_mode_block_ordering() -> None:
    # Distinct thermals: V must be block-diagonal per mode (R = (x̂₁,p̂₁,x̂₂,p̂₂)),
    # each block 𝟙₂·(2n̄+1). The Simon (x₁,x₂,p₁,p₂) layout would NOT be block-diagonal here.
    cov, _ = gaussian.covariance_matrix(
        _two_mode(qutip.thermal_dm(FOCK, 0.3), qutip.thermal_dm(FOCK, 1.0))
    )
    assert np.allclose(cov[:2, :2], (2 * 0.3 + 1) * np.eye(2), atol=1e-3)
    assert np.allclose(cov[2:, 2:], (2 * 1.0 + 1) * np.eye(2), atol=1e-3)
    assert np.allclose(cov[:2, 2:], 0.0, atol=1e-6), (
        "distinct product modes ⇒ zero inter-mode block"
    )


# --- 27.2 symplectic form, physicality guard, Williamson spectrum ----------------


def test_symplectic_form_is_block_j() -> None:
    omega = gaussian.symplectic_form(3)
    j = np.array([[0.0, 1.0], [-1.0, 0.0]])
    expected = np.block([[j if i == k else np.zeros((2, 2)) for k in range(3)] for i in range(3)])
    assert np.array_equal(omega, expected), "Ω = ⊕J with per-mode J-blocks (§27.2)"


def test_physicality_guard_is_psd_not_nu() -> None:
    # Vacuum is physical.
    cov_vac, _ = gaussian.covariance_matrix(_two_mode(qutip.basis(FOCK, 0), qutip.basis(FOCK, 0)))
    assert gaussian.is_physical(cov_vac)
    # The correction-2 anchor: an INDEFINITE candidate can have |eig(iΩV)| ≥ 1 yet fail
    # V + iΩ ≥ 0. Physicality is the PSD test, not a ν ≥ 1 check.
    bad = np.diag([3.0, 3.0, -3.0, -3.0])  # |eig(iΩ·bad)| = 3 ≥ 1, but bad + iΩ is not PSD
    assert np.min(np.abs(gaussian.symplectic_eigenvalues(bad))) >= 1.0
    assert not gaussian.is_physical(bad), "indefinite V with |eig(iΩV)|≥1 must still be rejected"


def test_thermal_symplectic_eigenvalues() -> None:
    nbar = 0.7
    cov, _ = gaussian.covariance_matrix(
        _two_mode(qutip.thermal_dm(FOCK, nbar), qutip.thermal_dm(FOCK, nbar))
    )
    nus = np.sort(gaussian.symplectic_eigenvalues(cov))
    assert nus.shape == (2,), "one ν per ±pair, counted once per mode (§27.2)"
    assert np.allclose(nus, 2 * nbar + 1, atol=1e-3), "thermal ν = 2n̄+1"


def test_symplectic_eigenvalues_are_not_singular_values() -> None:
    # For a squeezed marginal the symplectic eigenvalues (|eig(iΩV)|) differ from the
    # singular values of iΩV — the ⚠ correction the card pins.
    r = 0.8
    tmsv = two_mode_squeezed_vacuum(FOCK, z=r)
    cov, _ = gaussian.covariance_matrix(tmsv)
    omega = gaussian.symplectic_form(2)
    nus = np.sort(gaussian.symplectic_eigenvalues(cov))
    svd = np.sort(np.linalg.svd(1j * omega @ cov, compute_uv=False))[::2]  # ±pairs → once
    assert not np.allclose(nus, svd, atol=1e-2), "ν = |eig(iΩV)| must not equal the SVD of iΩV"
    assert np.allclose(nus, 1.0, atol=1e-3), (
        "a pure two-mode state is globally symplectic-pure (ν=1)"
    )


# --- 27.3 partial transpose ------------------------------------------------------


def test_tmsv_partial_transpose_smallest_eigenvalue() -> None:
    r = 0.6
    cov, _ = gaussian.covariance_matrix(two_mode_squeezed_vacuum(FOCK, z=r))
    cov_pt = gaussian.partial_transpose(cov, mode_indices=[1])  # flip p̂ of mode B
    nu_tilde = np.min(gaussian.symplectic_eigenvalues(cov_pt))
    assert nu_tilde == pytest.approx(np.exp(-2.0 * r), rel=1e-2), "TMSV PT: ν̃₋ = e^{−2r} (§27.3)"


def test_partial_transpose_flips_only_momentum_of_b() -> None:
    cov, _ = gaussian.covariance_matrix(
        _two_mode(qutip.thermal_dm(FOCK, 0.4), qutip.thermal_dm(FOCK, 0.9))
    )
    cov_pt = gaussian.partial_transpose(cov, mode_indices=[1])
    t = np.diag(
        [1.0, 1.0, 1.0, -1.0]
    )  # p̂ of mode 1 (index 3) flips; x̂ of mode 1 (index 2) does not
    assert np.allclose(cov_pt, t @ cov @ t, atol=1e-9)


# --- 27.4 first-moment-aware occupation -----------------------------------------


def test_occupation_is_first_moment_aware_not_nu() -> None:
    # A squeezed + displaced single-mode marginal: n̄ = (tr V + dᵀd − 2)/4 must include
    # both the squeezing and the displacement energy; (ν−1)/2 misses both.
    r, alpha = 0.5, 1.2
    state = squeezed_coherent_mode(FOCK, z=r, alpha=alpha)
    cov, d = gaussian.covariance_matrix(state)
    nbar = gaussian.mean_occupation(cov, d)
    nu_core = (gaussian.symplectic_eigenvalue(cov) - 1.0) / 2.0
    expected = float(qutip.expect(qutip.num(FOCK), state))
    assert nbar == pytest.approx(expected, rel=1e-3), "n̄ = (tr V + dᵀd − 2)/4 = ⟨â†â⟩"
    assert nbar > nu_core + 0.1, (
        "the thermal-core (ν−1)/2 undercounts a squeezed+displaced marginal"
    )


# --- 27.4 purity + Gaussian entropy (GT2) ---------------------------------------


def test_purity_is_product_of_reciprocal_symplectic_eigenvalues() -> None:
    # μ = ∏ 1/ν_i = 1/√det V. Vacuum → 1; a two-mode thermal product → 1/((2n̄+1)²).
    vac, _ = gaussian.covariance_matrix(_two_mode(qutip.basis(FOCK, 0), qutip.basis(FOCK, 0)))
    assert gaussian.purity(vac) == pytest.approx(1.0, abs=1e-6), "pure state → μ = 1 (§27.4)"

    nbar = 0.7
    cov, _ = gaussian.covariance_matrix(
        _two_mode(qutip.thermal_dm(FOCK, nbar), qutip.thermal_dm(FOCK, nbar))
    )
    mu = gaussian.purity(cov)
    assert mu == pytest.approx(1.0 / (2 * nbar + 1) ** 2, rel=1e-3), "μ = ∏ 1/ν_i"
    assert mu == pytest.approx(1.0 / np.sqrt(np.linalg.det(cov)), rel=1e-6), "μ = 1/√det V"


def test_gaussian_entropy_bits_matches_von_neumann() -> None:
    # S = Σ g(ν_i) in bits equals the exact −Σλ log₂λ of the truncated thermal ρ.
    vac, _ = gaussian.covariance_matrix(_two_mode(qutip.basis(FOCK, 0), qutip.basis(FOCK, 0)))
    assert gaussian.gaussian_entropy_bits(vac) == pytest.approx(0.0, abs=1e-9), "pure → S = 0"

    nbar = 0.7
    rho = qutip.thermal_dm(FOCK, nbar)
    cov, _ = gaussian.covariance_matrix(rho)
    assert gaussian.gaussian_entropy_bits(cov) == pytest.approx(
        _von_neumann_entropy_bits(rho), rel=1e-3
    ), "S = Σ g(ν_i) [bits] matches the von Neumann entropy of the thermal ρ (§27.4)"


# --- 27.4 logarithmic negativity + separability scoping (GT4) --------------------


def test_log_negativity_is_full_sum_tmsv_oracle() -> None:
    # E_N = Σ_k max(0, −log₂ ν̃_k) [bits]; a two-mode squeezed vacuum (squeezing r) → 2r/ln2,
    # and it is symmetric in the bipartition choice.
    r = 0.6
    cov, _ = gaussian.covariance_matrix(two_mode_squeezed_vacuum(FOCK, z=r))
    assert gaussian.log_negativity(cov, [1]) == pytest.approx(2 * r / np.log(2), rel=1e-2)
    assert gaussian.log_negativity(cov, [0]) == pytest.approx(gaussian.log_negativity(cov, [1]))


def test_log_negativity_separable_is_zero() -> None:
    cov, _ = gaussian.covariance_matrix(
        _two_mode(qutip.thermal_dm(FOCK, 0.4), qutip.thermal_dm(FOCK, 0.9))
    )
    assert gaussian.log_negativity(cov, [1]) == pytest.approx(0.0, abs=1e-6)


def test_separability_certified_only_for_1xn() -> None:
    # 1×N: PPT ⟺ separable, so is_separable certifies. TMSV → entangled (False); thermal
    # product → separable (True).
    cov_ent, _ = gaussian.covariance_matrix(two_mode_squeezed_vacuum(FOCK, z=0.6))
    cov_sep, _ = gaussian.covariance_matrix(
        _two_mode(qutip.thermal_dm(FOCK, 0.4), qutip.thermal_dm(FOCK, 0.9))
    )
    assert not gaussian.is_separable(cov_ent, [1])
    assert gaussian.is_separable(cov_sep, [1])
    # M×N (M, N ≥ 2): E_N = 0 does NOT certify separability (PPT-bound-entangled) → raises.
    with pytest.raises(ValueError, match="1×N"):
        gaussian.is_separable(np.eye(8), [0, 1])  # a 2×2 cut


# --- 27.4 effective temperature (GT5) -------------------------------------------

_HBAR, _K_B = 1.054571817e-34, 1.380649e-23  # SI (CODATA 2018)


def test_effective_temperature_round_trips_through_bose() -> None:
    # T_eff = ℏω_loc/(k_B ln(1 + 1/n̄)) is energy-equivalent: the thermal Bose occupation
    # 1/(e^{ℏω/k_B T} − 1) at T_eff recovers n̄ exactly.
    omega = 2 * np.pi * 1e6  # a 1-MHz mode (rad/s)
    for nbar in (0.1, 1.0, 5.0):
        t = gaussian.effective_temperature(nbar, omega)
        assert 1.0 / np.expm1(_HBAR * omega / (_K_B * t)) == pytest.approx(nbar, rel=1e-9)


def test_effective_temperature_zero_at_pure_and_rejects_negative() -> None:
    assert gaussian.effective_temperature(0.0, 2 * np.pi * 1e6) == 0.0  # pure mode → 0 K
    with pytest.raises(ValueError, match="n̄ must be"):
        gaussian.effective_temperature(-0.1, 2 * np.pi * 1e6)


def test_effective_temperature_is_first_moment_aware() -> None:
    # A pure squeezed + displaced marginal has thermal-core (ν−1)/2 ≈ 0 but n̄ > 0, so its
    # effective temperature is > 0 — driven by the first-moment-aware occupation, not the core.
    cov, d = gaussian.covariance_matrix(squeezed_coherent_mode(FOCK, z=0.5, alpha=1.2))
    nbar = gaussian.mean_occupation(cov, d)
    core = (gaussian.symplectic_eigenvalue(cov) - 1.0) / 2.0
    assert core == pytest.approx(0.0, abs=1e-6)  # pure marginal: thermal core ≈ 0
    assert nbar > 0.1
    assert gaussian.effective_temperature(nbar, 2 * np.pi * 1e6) > 0.0
