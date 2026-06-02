# SPDX-License-Identifier: MIT
"""Analytic-regression oracle for interferometric fringe analysis (WP-02 WI-4, MCD).

A phase-scanned single-spin interferometer built from real quantum states + the
§3 ion spin operators (:func:`sigma_z_ion`/:func:`sigma_y_ion`, **not** hand-typed
cosines and **not** ``qutip.sigmaz``, whose sign is banned in library code):

* X-basis readout of ``|ψ(θ)⟩ = e^{−iθ σ_z/2}|+⟩`` gives ``P(θ) = (1 + cos θ)/2`` —
  a **unit-visibility** fringe at phase 0. A contrast loss (what a dephasing channel
  does — it scales the amplitude about the mean) drops the fitted visibility to the
  closed form ``e^{−Γ}``.
* Y-quadrature readout of the same state gives ``⟨σ_y⟩(θ) = +sin θ = cos(θ − π/2)`` —
  a fringe with a **non-zero, sign-definite phase** ``+π/2``. This is the convention
  guard: ``qutip.sigmaz`` would flip the rotation sense and yield ``−π/2``, so a
  silent sign slip in §3 fails the test rather than passing unnoticed.

``fit_fringe`` recovers all three. The tolerance is a named symbolic constant.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics.observables import fit_fringe, fringe_visibility
from iontrap_dynamics.operators import sigma_y_ion, sigma_z_ion, spin_down, spin_up

pytestmark = pytest.mark.regression_analytic

ATOL_FRINGE = 1e-6


def _plus_state() -> qutip.Qobj:
    """The X eigenstate ``|+⟩ = (|↑⟩ + |↓⟩)/√2`` from the §3 basis kets."""
    return (spin_up() + spin_down()).unit()


def _evolved(theta: float) -> qutip.Qobj:
    """``e^{−iθ σ_z_ion/2}|+⟩`` — the phase-scanned interferometer state."""
    return (-1j * theta * sigma_z_ion() / 2).expm() * _plus_state()


def _x_readout_fringe(scan_rad: np.ndarray, *, contrast: float = 1.0) -> np.ndarray:
    """``P(+|X)(θ)`` for ``e^{−iθ σ_z/2}|+⟩``, optionally contrast-scaled.

    ``contrast = 1`` is the ideal fringe ``(1 + cos θ)/2``; ``contrast < 1``
    scales the fringe amplitude about its mean ``0.5`` — the effect a dephasing
    channel has on the readout fringe.
    """
    projector = _plus_state() * _plus_state().dag()
    ideal = np.array([float(qutip.expect(projector, _evolved(theta))) for theta in scan_rad])
    return 0.5 + contrast * (ideal - 0.5)


def _y_quadrature_fringe(scan_rad: np.ndarray) -> np.ndarray:
    """``⟨σ_y_ion⟩(θ) = +sin θ`` for ``e^{−iθ σ_z/2}|+⟩`` — a phase-``+π/2`` fringe."""
    sigma_y = sigma_y_ion()
    return np.array([float(qutip.expect(sigma_y, _evolved(theta))) for theta in scan_rad])


def test_ideal_fringe_has_unit_visibility() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    signal = _x_readout_fringe(theta)
    assert fringe_visibility(signal) == pytest.approx(1.0, abs=ATOL_FRINGE)
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


def test_y_quadrature_fringe_phase_pins_the_section_3_sign() -> None:
    # ⟨σ_y_ion⟩(θ) = +sin θ = cos(θ − π/2): a non-zero, sign-definite phase.
    # qutip.sigmaz (the banned sign) would give −sin θ → phase −π/2, so this
    # oracle fails loudly on a §3 convention slip.
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    signal = _y_quadrature_fringe(theta)
    fit = fit_fringe(theta, signal)
    assert fit.offset == pytest.approx(0.0, abs=ATOL_FRINGE)
    assert fit.amplitude == pytest.approx(1.0, abs=ATOL_FRINGE)
    assert fit.phase_rad == pytest.approx(np.pi / 2, abs=ATOL_FRINGE)
