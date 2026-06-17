# SPDX-License-Identifier: MIT
"""Analytic-regression oracle for the BLP non-Markovianity primitives (Phase A, WI-1).

Validates :func:`trace_distance` + :func:`blp_non_markovianity` against the
closed forms of Breuer–Laine–Piilo, PRL 103, 210401 (2009):

* the central-spin **pure-dephasing** trace distance ``D = √(a² + f²|b|²)``
  (Eq. 14), where ``a`` is the population difference, ``b`` the coherence
  difference of the two initial states, and ``f`` the (common) dephasing factor.
  For the Hermitian ``Δ = ρ₁ − ρ₂ = [[a, f b], [f b*, −a]]`` the eigenvalues are
  ``±√(a² + f²|b|²)``, so ``D = ½Σ|λ| = √(a² + f²|b|²)`` for *any* qubit pair
  under common dephasing.
* the divisibility property ``𝒩 = 0`` for a monotone non-increasing ``D``
  (Markovian), and ``𝒩 > 0`` when a non-monotone coherence factor ``f(t)``
  (revivals) drives information back-flow — the defining signature of
  non-Markovianity.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics.information import blp_non_markovianity, trace_distance

pytestmark = pytest.mark.regression_analytic

ATOL = 1e-12


def _qubit(rho00: float, rho01: complex) -> qutip.Qobj:
    """A single-qubit density matrix from a population ``rho00`` and coherence ``rho01``."""
    matrix = np.array([[rho00, rho01], [np.conj(rho01), 1.0 - rho00]], dtype=complex)
    return qutip.Qobj(matrix, dims=[[2], [2]])


def _dephase(rho: qutip.Qobj, f: float) -> qutip.Qobj:
    """Pure dephasing: scale the off-diagonal coherence by ``f`` (populations fixed)."""
    matrix = np.asarray(rho.full(), dtype=complex)
    matrix[0, 1] *= f
    matrix[1, 0] *= f
    return qutip.Qobj(matrix, dims=rho.dims)


@pytest.mark.parametrize("f", [1.0, 0.8, 0.5, 0.2, 0.0])
def test_dephasing_trace_distance_matches_blp_eq14(f: float) -> None:
    rho1 = _qubit(0.7, 0.3 + 0.1j)
    rho2 = _qubit(0.4, -0.2 + 0.0j)
    a = 0.7 - 0.4  # population difference Δρ₀₀
    b = (0.3 + 0.1j) - (-0.2 + 0.0j)  # coherence difference Δρ₀₁
    expected = float(np.sqrt(a**2 + (f**2) * abs(b) ** 2))  # BLP Eq. 14
    assert trace_distance(_dephase(rho1, f), _dephase(rho2, f)) == pytest.approx(expected, abs=ATOL)


@pytest.mark.parametrize("f", [1.0, 0.6, 0.3, 0.0, -0.4, -1.0])
def test_antipodal_pair_distance_is_abs_f(f: float) -> None:
    # σ_x eigenstates |±⟩: a = 0, |b| = 1 → D = |f| (central-spin special case).
    plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    minus = (qutip.basis(2, 0) - qutip.basis(2, 1)).unit()
    rho_p, rho_m = plus * plus.dag(), minus * minus.dag()
    assert trace_distance(_dephase(rho_p, f), _dephase(rho_m, f)) == pytest.approx(abs(f), abs=ATOL)


def test_markovian_monotone_dephasing_gives_zero_measure() -> None:
    # f(t) = e^{−γt} monotone ⇒ D monotone non-increasing ⇒ 𝒩 = 0 (divisible / Markovian).
    rho1, rho2 = _qubit(0.5, 0.5), _qubit(0.5, -0.5)  # |+⟩, |−⟩
    f = np.exp(-0.7 * np.linspace(0.0, 5.0, 200))
    d = np.array([trace_distance(_dephase(rho1, fi), _dephase(rho2, fi)) for fi in f])
    assert blp_non_markovianity(d) == pytest.approx(0.0, abs=1e-9)


def test_non_markovian_revivals_give_positive_measure() -> None:
    # f(t) = cos(2t) revives ⇒ D = |cos 2t| has back-flow ⇒ 𝒩 > 0, and equals the
    # analytic sum of positive increments of |cos 2t| (D = |f| for the antipodal pair).
    rho1, rho2 = _qubit(0.5, 0.5), _qubit(0.5, -0.5)
    f = np.cos(2.0 * np.linspace(0.0, 3.0 * np.pi, 600))
    d = np.array([trace_distance(_dephase(rho1, fi), _dephase(rho2, fi)) for fi in f])
    measure = blp_non_markovianity(d)
    assert measure > 0.5  # several revivals over the window
    assert measure == pytest.approx(blp_non_markovianity(np.abs(f)), abs=ATOL)
