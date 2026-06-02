# SPDX-License-Identifier: MIT
"""Unit tests for the interferometric fringe analysis (WP-02 WI-4 / F4, dispatch MCD).

``fringe_visibility`` (model-free contrast) and ``fit_fringe`` (least-squares
``A + B·cos(θ − φ)`` fit) in ``observables.py`` — exact recovery on synthetic
fringes plus input validation. The physics-fringe oracle lives in
``tests/regression/analytic/test_interferometric_visibility.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from iontrap_dynamics.observables import FringeFit, fit_fringe, fringe_visibility


def test_fringe_visibility_model_free():
    assert fringe_visibility([0.1, 0.9, 0.5]) == pytest.approx(0.8)  # (0.9−0.1)/(0.9+0.1)
    assert fringe_visibility([0.5, 0.5, 0.5]) == pytest.approx(0.0)  # flat → 0
    assert fringe_visibility([0.0, 1.0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad",
    [[], [0.1, -0.2], [0.1, float("nan")], [0.0, 0.0]],
)
def test_fringe_visibility_invalid_raises(bad):
    with pytest.raises(ValueError):
        fringe_visibility(bad)


def test_fit_fringe_recovers_synthetic_fringe():
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    offset, amplitude, phase = 0.4, 0.25, 0.6
    signal = offset + amplitude * np.cos(theta - phase)
    fit = fit_fringe(theta, signal)
    assert fit.offset == pytest.approx(offset, abs=1e-9)
    assert fit.amplitude == pytest.approx(amplitude, abs=1e-9)
    assert fit.phase_rad == pytest.approx(phase, abs=1e-9)
    assert fit.visibility == pytest.approx(amplitude / offset, abs=1e-9)


def test_fit_fringe_phase_is_the_maximum_location():
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    signal = 0.5 + 0.5 * np.cos(theta - 1.2)
    fit = fit_fringe(theta, signal)
    # the fringe peaks at θ = φ; the grid-nearest peak matches the fitted phase
    assert theta[int(np.argmax(signal))] == pytest.approx(
        fit.phase_rad, abs=2.0 * np.pi / 128 + 1e-9
    )


def test_fit_fringe_validation():
    theta = np.linspace(0.0, np.pi, 5)
    with pytest.raises(ValueError, match="same shape"):
        fit_fringe(theta, theta[:-1])
    with pytest.raises(ValueError, match="at least 3"):
        fit_fringe([0.0, 1.0], [0.5, 0.6])
    with pytest.raises(ValueError, match="finite"):
        fit_fringe([0.0, 1.0, float("inf")], [0.5, 0.6, 0.7])


def test_fringe_fit_is_frozen():
    fit = FringeFit(offset=0.5, amplitude=0.5, phase_rad=0.0, visibility=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fit.visibility = 0.0  # type: ignore[misc]
