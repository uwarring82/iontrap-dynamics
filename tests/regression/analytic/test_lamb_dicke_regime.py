# SPDX-License-Identifier: MIT
"""Analytic-regression oracles for the Lamb–Dicke regime helpers (WP-02 WI-5, MCE).

Three physics anchors for the ``analytic.py`` additions:

* **Debye–Waller matches the series.** The closed form ``e^{−η²(2n̄+1)/2}`` equals
  the truncated thermal average ``Σ_n p_n(n̄)·e^{−η²/2}L_n(η²)`` of the per-Fock
  carrier matrix element — proven two ways in the docstring, checked here to
  machine precision.
* **Continuity with the shipped full-LD numerics.** The same Debye–Waller factor
  reproduces the thermal average of the *shipped* ``_full_ld_carrier_single_mode``
  diagonal, and the all-orders sideband helpers reproduce the magnitudes of the
  *shipped* ``_full_ld_{raising,lowering}_single_mode`` off-diagonals — continuous
  by identity with the ``full_lamb_dicke=True`` Hamiltonian builders.
* **Leading-order limit.** As η→0 the all-orders sideband rates collapse onto the
  leading-order ``|η|√(n+1)·Ω`` / ``|η|√n·Ω`` already shipped.

Tolerances are named symbolic constants. The shipped operators carry Fock-
truncation noise at high n, so the continuity oracle uses a generous Fock
dimension and stays at low n (pitfall: truncation noise ≠ formula error).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import eval_genlaguerre

from iontrap_dynamics import analytic
from iontrap_dynamics.hamiltonians import (
    _full_ld_carrier_single_mode,
    _full_ld_lowering_single_mode,
    _full_ld_raising_single_mode,
)

pytestmark = pytest.mark.regression_analytic

ATOL_DEBYE_WALLER = 1e-12
ATOL_SIDEBAND_LD = 1e-9
RTOL_LEADING_ORDER = 1e-6


def _thermal_weights(n_bar: float, n_levels: int) -> np.ndarray:
    """Thermal occupation ``p_n(n̄) = n̄ⁿ/(n̄+1)^{n+1}`` over ``[0, n_levels)``."""
    n = np.arange(n_levels)
    return n_bar**n / (n_bar + 1.0) ** (n + 1)


@pytest.mark.parametrize(
    ("eta", "n_bar"),
    [(0.1, 2.0), (0.3, 2.0), (0.3, 5.0), (0.5, 2.0), (0.5, 5.0)],
)
def test_debye_waller_matches_thermal_series(eta: float, n_bar: float) -> None:
    # closed form e^{−η²(2n̄+1)/2}  ==  Σ_n p_n(n̄) · e^{−η²/2} · L_n(η²)
    n_levels = int(np.ceil(50 + 30 * n_bar))  # geometric tail negligible by here
    weights = _thermal_weights(n_bar, n_levels)
    laguerre = eval_genlaguerre(np.arange(n_levels), 0, eta**2)
    series = float(np.sum(weights * np.exp(-(eta**2) / 2.0) * laguerre))
    closed = analytic.debye_waller_factor(lamb_dicke_parameter=eta, mean_phonon_number=n_bar)
    assert closed == pytest.approx(series, abs=ATOL_DEBYE_WALLER)


@pytest.mark.parametrize(("eta", "n_bar"), [(0.2, 1.0), (0.3, 3.0), (0.4, 2.0)])
def test_debye_waller_matches_shipped_full_ld_carrier(eta: float, n_bar: float) -> None:
    # Continuity: thermal average of the SHIPPED carrier diagonal ⟨n|M̂_0|n⟩
    # equals the analytic Debye–Waller factor.
    n_levels = int(np.ceil(50 + 30 * n_bar))
    carrier_diag = np.real(np.diag(_full_ld_carrier_single_mode(n_levels, eta).full()))
    weights = _thermal_weights(n_bar, n_levels)
    shipped_thermal = float(np.sum(weights * carrier_diag))
    closed = analytic.debye_waller_factor(lamb_dicke_parameter=eta, mean_phonon_number=n_bar)
    assert closed == pytest.approx(shipped_thermal, abs=ATOL_DEBYE_WALLER)


def test_debye_waller_vacuum_is_zero_point_suppression() -> None:
    # n̄ = 0 retains the zero-point suppression e^{−η²/2}, NOT unity.
    eta = 0.4
    factor = analytic.debye_waller_factor(lamb_dicke_parameter=eta, mean_phonon_number=0.0)
    assert factor == pytest.approx(np.exp(-(eta**2) / 2.0), abs=ATOL_DEBYE_WALLER)


@pytest.mark.parametrize("eta", [0.1, 0.3, 0.5])
def test_full_ld_sideband_matches_shipped_operators(eta: float) -> None:
    # Headline continuity: the all-orders analytic sideband rate equals the
    # magnitude of the shipped full-LD operator's off-diagonal element.
    omega = 2.0
    fock_dim = 24  # generous: keeps truncation noise far below ATOL at n ≤ 5
    raising = _full_ld_raising_single_mode(fock_dim, eta).full()
    lowering = _full_ld_lowering_single_mode(fock_dim, eta).full()
    for n in range(6):
        blue = analytic.blue_sideband_rabi_frequency_full_ld(
            carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
        )
        assert blue == pytest.approx(abs(raising[n + 1, n]) * omega, abs=ATOL_SIDEBAND_LD)
        red = analytic.red_sideband_rabi_frequency_full_ld(
            carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
        )
        expected_red = abs(lowering[n - 1, n]) * omega if n >= 1 else 0.0
        assert red == pytest.approx(expected_red, abs=ATOL_SIDEBAND_LD)


@pytest.mark.parametrize(
    ("branch", "eta", "n"),
    [
        ("blue", 0.9, 4),  # L_4^{(1)}(0.81) < 0
        ("blue", 0.9, 5),  # L_5^{(1)}(0.81) < 0
        ("blue", 1.0, 3),  # L_3^{(1)}(1.0)  < 0
        ("red", 0.9, 5),  # red(5) uses L_4^{(1)}(0.81) < 0
        ("red", 1.0, 5),  # red(5) uses L_4^{(1)}(1.0)  < 0
    ],
)
def test_full_ld_sideband_past_laguerre_null_returns_positive_magnitude(
    branch: str, eta: float, n: int
) -> None:
    # Past the deep regime the associated Laguerre L^{(1)}(η²) changes sign; the
    # Rabi *rate* is its magnitude. This pins the abs() in both branches that the
    # deep / lower-intermediate cases (η ≤ 0.5, where L^{(1)} stays positive)
    # never reach — dropping it would return an unphysical *negative* rate that no
    # longer matches the shipped full-LD operator. This is the Rabi-null physics
    # the all-orders forms exist for.
    laguerre_index = n if branch == "blue" else n - 1
    assert eval_genlaguerre(laguerre_index, 1, eta**2) < 0.0  # precondition
    omega, fock_dim = 1.0, 40  # generous: high η spreads the displacement
    if branch == "blue":
        rate = analytic.blue_sideband_rabi_frequency_full_ld(
            carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
        )
        shipped = abs(_full_ld_raising_single_mode(fock_dim, eta).full()[n + 1, n])
    else:
        rate = analytic.red_sideband_rabi_frequency_full_ld(
            carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
        )
        shipped = abs(_full_ld_lowering_single_mode(fock_dim, eta).full()[n - 1, n])
    assert rate > 0.0  # magnitude: the rate stays positive past the Laguerre null
    assert rate == pytest.approx(shipped * omega, abs=ATOL_SIDEBAND_LD)


@pytest.mark.parametrize("n", [0, 1, 3, 5])
def test_full_ld_sideband_reduces_to_leading_order(n: int) -> None:
    # η→0: the all-orders forms collapse onto the shipped leading-order rates.
    eta, omega = 1e-5, 3.0
    blue_full = analytic.blue_sideband_rabi_frequency_full_ld(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
    )
    blue_lead = analytic.blue_sideband_rabi_frequency(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
    )
    assert blue_full == pytest.approx(blue_lead, rel=RTOL_LEADING_ORDER)
    red_full = analytic.red_sideband_rabi_frequency_full_ld(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
    )
    red_lead = analytic.red_sideband_rabi_frequency(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
    )
    assert red_full == pytest.approx(red_lead, rel=RTOL_LEADING_ORDER, abs=1e-18)


def test_red_full_ld_vacuum_is_zero() -> None:
    # The vacuum cannot lose a phonon — red sideband is exactly 0 at n = 0.
    assert (
        analytic.red_sideband_rabi_frequency_full_ld(
            carrier_rabi_frequency=5.0, lamb_dicke_parameter=0.3, n_initial=0
        )
        == 0.0
    )


@pytest.mark.parametrize("n", [1, 2, 4])
def test_red_and_blue_full_ld_share_the_same_transition(n: int) -> None:
    # red(n) and blue(n−1) are the SAME |n−1⟩ ↔ |n⟩ coupling — a correctness
    # signature independent of the shipped operators.
    omega, eta = 4.0, 0.35
    red = analytic.red_sideband_rabi_frequency_full_ld(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
    )
    blue = analytic.blue_sideband_rabi_frequency_full_ld(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n - 1
    )
    assert red == pytest.approx(blue, abs=ATOL_SIDEBAND_LD)
