# SPDX-License-Identifier: MIT
"""Analytic-regression oracle for the GHZ / cat factory benchmark.

Binding closed-form anchors for ``tools/run_benchmark_ghz_cat.py`` (WP-01 §7
row 5, dispatch EDE):

(a) The N-ion GHZ state, phase-rotated by ``U(φ) = exp(−i φ J_z)`` with the
    collective generator ``J_z = ½ Σ_i σ_z``, has spin parity
    ``P = ∏_i σ_x`` expectation ``⟨P⟩(φ) = cos(N φ)`` — the Heisenberg-limited
    fringe at N times the single-spin frequency, for N = 2 and N = 3.

(b) ``cat_mode(parity="even")`` is a +1 eigenstate of the Fock parity
    ``Π = diag((−1)^n)`` and ``parity="odd"`` is a −1 eigenstate; the GHZ
    N = 2 Wootters concurrence equals 1 (maximally entangled).

These are independent of the benchmark tool (which reports
``max_numerical_vs_analytic_error``); the tolerance is a named symbolic
constant. Lives in the regression_analytic tier alongside the other compute
oracles.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from _helpers import _collective_jz, _spin_hilbert
from iontrap_dynamics.entanglement import concurrence_trajectory
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.operators import sigma_x_ion
from iontrap_dynamics.states import cat_mode, ghz_state

pytestmark = pytest.mark.regression_analytic

ATOL_GHZ_CAT = 1e-9

CAT_FOCK_DIM = 24
CAT_ALPHA = 1.3


def _spin_x_parity(hilbert: HilbertSpace) -> qutip.Qobj:
    """The spin parity ``P = ∏_i σ_x`` on the spins of ``hilbert``."""
    ops = [hilbert.spin_op_for_ion(sigma_x_ion(), i) for i in range(hilbert.n_ions)]
    total = ops[0]
    for op in ops[1:]:
        total = total * op
    return total


@pytest.mark.parametrize("n_ions", [2, 3])
def test_ghz_parity_fringe_is_cos_n_phi(n_ions: int) -> None:
    """⟨∏_i σ_x⟩(φ) = cos(N φ) for the phase-rotated GHZ state."""
    h = _spin_hilbert(n_ions)
    jz = _collective_jz(h)
    parity_op = _spin_x_parity(h)
    ghz = ghz_state(h)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, 41)
    for phi in phi_grid:
        evolved = (-1j * phi * jz).expm() * ghz
        value = float(qutip.expect(parity_op, evolved))
        assert value == pytest.approx(np.cos(n_ions * phi), abs=ATOL_GHZ_CAT)


def test_cat_even_parity_is_plus_one() -> None:
    """An even cat is a +1 eigenstate of the Fock parity diag((-1)^n)."""
    fock_parity = qutip.Qobj(np.diag([(-1.0) ** n for n in range(CAT_FOCK_DIM)]))
    psi = cat_mode(CAT_FOCK_DIM, CAT_ALPHA, parity="even")
    assert float(qutip.expect(fock_parity, psi)) == pytest.approx(1.0, abs=ATOL_GHZ_CAT)


def test_cat_odd_parity_is_minus_one() -> None:
    """An odd cat is a -1 eigenstate of the Fock parity diag((-1)^n)."""
    fock_parity = qutip.Qobj(np.diag([(-1.0) ** n for n in range(CAT_FOCK_DIM)]))
    psi = cat_mode(CAT_FOCK_DIM, CAT_ALPHA, parity="odd")
    assert float(qutip.expect(fock_parity, psi)) == pytest.approx(-1.0, abs=ATOL_GHZ_CAT)


def test_ghz_n2_concurrence_is_one() -> None:
    """The two-ion GHZ (Bell) state is maximally entangled: concurrence = 1."""
    h = _spin_hilbert(2)
    c = concurrence_trajectory([ghz_state(h)], hilbert=h, ion_indices=(0, 1))
    assert c[0] == pytest.approx(1.0, abs=ATOL_GHZ_CAT)
