# SPDX-License-Identifier: MIT
"""Unit tests for the Lamb–Dicke regime helpers (WP-02 WI-5, MCE).

Classifier behaviour (the deep/intermediate/beyond thresholds on η²(2n̄+1)),
the ``StrEnum`` return contract, the confinement value, and input validation.
The physics oracles (Debye–Waller series, full-LD continuity) live in
``tests/regression/analytic/test_lamb_dicke_regime.py``.
"""

from __future__ import annotations

import pytest

from iontrap_dynamics import analytic
from iontrap_dynamics.analytic import LambDickeRegime


def test_lamb_dicke_confinement_value() -> None:
    # x = η²(2n̄+1)
    assert analytic.lamb_dicke_confinement(
        lamb_dicke_parameter=0.3, mean_phonon_number=2.0
    ) == pytest.approx(0.09 * 5.0)
    # vacuum: x = η²
    assert analytic.lamb_dicke_confinement(
        lamb_dicke_parameter=0.5, mean_phonon_number=0.0
    ) == pytest.approx(0.25)
    # sign of η is irrelevant (η² used)
    assert analytic.lamb_dicke_confinement(
        lamb_dicke_parameter=-0.3, mean_phonon_number=2.0
    ) == pytest.approx(
        analytic.lamb_dicke_confinement(lamb_dicke_parameter=0.3, mean_phonon_number=2.0)
    )


@pytest.mark.parametrize(
    ("eta", "n_bar", "expected"),
    [
        (0.1, 0.0, LambDickeRegime.DEEP),  # x = 0.01
        (0.3, 0.0, LambDickeRegime.DEEP),  # x = 0.09  (just below 0.1)
        (0.3, 2.0, LambDickeRegime.INTERMEDIATE),  # x = 0.45
        (0.3, 5.0, LambDickeRegime.INTERMEDIATE),  # x = 0.99 (just below 1.0)
        (0.5, 5.0, LambDickeRegime.BEYOND),  # x = 2.75
        (1.0, 0.0, LambDickeRegime.BEYOND),  # x = 1.0 exactly -> BEYOND (≥, not <)
    ],
)
def test_regime_classifier_bins(eta: float, n_bar: float, expected: LambDickeRegime) -> None:
    assert (
        analytic.lamb_dicke_regime(lamb_dicke_parameter=eta, mean_phonon_number=n_bar) is expected
    )


def test_regime_boundaries_are_half_open_below() -> None:
    # The thresholds are exclusive lower bounds for the next bin: x = 1.0 is
    # BEYOND, not INTERMEDIATE; the deep/intermediate cut is the documented 0.1.
    assert analytic.LAMB_DICKE_DEEP_MAX == 0.1
    assert analytic.LAMB_DICKE_INTERMEDIATE_MAX == 1.0
    # x = 1.0 exactly (η=1, n̄=0): not < 1.0 -> BEYOND
    assert (
        analytic.lamb_dicke_regime(lamb_dicke_parameter=1.0, mean_phonon_number=0.0)
        is LambDickeRegime.BEYOND
    )
    # x = 0.09 -> DEEP; x = 0.16 -> INTERMEDIATE (straddling 0.1)
    assert (
        analytic.lamb_dicke_regime(lamb_dicke_parameter=0.3, mean_phonon_number=0.0)
        is LambDickeRegime.DEEP
    )
    assert (
        analytic.lamb_dicke_regime(lamb_dicke_parameter=0.4, mean_phonon_number=0.0)
        is LambDickeRegime.INTERMEDIATE
    )


def test_regime_is_strenum_and_compares_to_string() -> None:
    regime = analytic.lamb_dicke_regime(lamb_dicke_parameter=0.05, mean_phonon_number=0.0)
    assert isinstance(regime, LambDickeRegime)
    assert regime == "deep"  # StrEnum: member equals its string value
    assert regime is LambDickeRegime.DEEP
    assert {r.value for r in LambDickeRegime} == {"deep", "intermediate", "beyond"}


def test_debye_waller_in_unit_interval_and_monotone() -> None:
    # DW ∈ (0, 1], decreasing in both η and n̄.
    f00 = analytic.debye_waller_factor(lamb_dicke_parameter=0.0, mean_phonon_number=0.0)
    assert f00 == pytest.approx(1.0)
    f1 = analytic.debye_waller_factor(lamb_dicke_parameter=0.3, mean_phonon_number=1.0)
    f2 = analytic.debye_waller_factor(lamb_dicke_parameter=0.3, mean_phonon_number=4.0)
    f3 = analytic.debye_waller_factor(lamb_dicke_parameter=0.6, mean_phonon_number=1.0)
    assert 0.0 < f2 < f1 < 1.0  # more phonons -> more suppression
    assert f3 < f1  # larger η -> more suppression


@pytest.mark.parametrize(
    "call",
    [
        lambda **kw: analytic.lamb_dicke_confinement(**kw),
        lambda **kw: analytic.lamb_dicke_regime(**kw),
        lambda **kw: analytic.debye_waller_factor(**kw),
    ],
)
def test_regime_helpers_reject_bad_inputs(call) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        call(lamb_dicke_parameter=0.3, mean_phonon_number=-1.0)
    with pytest.raises(ValueError, match="finite"):
        call(lamb_dicke_parameter=0.3, mean_phonon_number=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        call(lamb_dicke_parameter=float("nan"), mean_phonon_number=1.0)


@pytest.mark.parametrize(
    "call",
    [
        analytic.blue_sideband_rabi_frequency_full_ld,
        analytic.red_sideband_rabi_frequency_full_ld,
    ],
)
def test_full_ld_sideband_rejects_bad_inputs(call) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        call(carrier_rabi_frequency=2.0, lamb_dicke_parameter=0.3, n_initial=-1)
    with pytest.raises(ValueError, match="finite"):
        call(carrier_rabi_frequency=2.0, lamb_dicke_parameter=float("inf"), n_initial=2)
