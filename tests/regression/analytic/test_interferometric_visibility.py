# SPDX-License-Identifier: MIT
"""Analytic-regression oracle for interferometric fringe analysis (WP-02 WI-4, MCD).

A phase-scanned single-spin interferometer built from real quantum states + the
§3 spin operators (not hand-typed cosines): ``|ψ(θ)⟩ = e^{−iθ σ_z/2}|+⟩`` read out
in the X basis gives ``P(θ) = (1 + cos θ)/2`` — a **unit-visibility** fringe at
phase 0. A contrast loss (what a dephasing channel does to the fringe — it scales
the amplitude about the mean) reduces the fitted visibility to the closed-form
value ``e^{−Γ}``. ``fit_fringe`` recovers both. The tolerance is a named symbolic
constant.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics.observables import fit_fringe, fringe_visibility

pytestmark = pytest.mark.regression_analytic

ATOL_FRINGE = 1e-6


def _x_readout_fringe(scan_rad: np.ndarray, *, contrast: float = 1.0) -> np.ndarray:
    """``P(+|X)(θ)`` for ``e^{−iθ σ_z/2}|+⟩``, optionally contrast-scaled.

    ``contrast = 1`` is the ideal fringe ``(1 + cos θ)/2``; ``contrast < 1``
    scales the fringe amplitude about its mean ``0.5`` — the effect a dephasing
    channel has on the readout fringe.
    """
    plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    projector = plus * plus.dag()
    sigma_z = qutip.sigmaz()
    ideal = np.array(
        [
            float(qutip.expect(projector, (-1j * theta * sigma_z / 2).expm() * plus))
            for theta in scan_rad
        ]
    )
    return 0.5 + contrast * (ideal - 0.5)


def test_ideal_fringe_has_unit_visibility() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    signal = _x_readout_fringe(theta)
    assert fringe_visibility(signal) == pytest.approx(1.0, abs=1e-6)
    fit = fit_fringe(theta, signal)
    assert fit.visibility == pytest.approx(1.0, abs=ATOL_FRINGE)
    assert fit.phase_rad == pytest.approx(0.0, abs=ATOL_FRINGE)


def test_contrast_loss_reduces_visibility_to_the_closed_form() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    gamma = 0.7
    signal = _x_readout_fringe(theta, contrast=float(np.exp(-gamma)))
    fit = fit_fringe(theta, signal)
    # P = 0.5 + (e^{−Γ}/2) cos θ  ->  visibility = e^{−Γ}
    assert fit.visibility == pytest.approx(np.exp(-gamma), abs=ATOL_FRINGE)
    assert fit.phase_rad == pytest.approx(0.0, abs=ATOL_FRINGE)
