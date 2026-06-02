# SPDX-License-Identifier: MIT
"""Analytic-regression oracle for the recoverability-channel benchmark.

Binding closed-form anchor for ``tools/run_benchmark_recoverability.py`` (WP-01
§7 row 4, dispatch EDE): the recoverability measure (clamped coherent
information ``max(0, S(ρ_A) − S(ρ_{S∪A}))`` in bits) on the two-qubit Werner
dephasing channel ``ρ(p) = p |Φ⁺⟩⟨Φ⁺| + (1 − p) I/4`` reproduces the textbook
endpoints exactly — ``H_S = 1`` bit at perfect recovery (``p = 1``) and ``0`` at
full decoherence (``p = 0``) — and is monotone non-decreasing in ``p`` between.
The benchmark tool records max_numerical_vs_analytic_error; this is the
independent binding assertion, with the tolerance as a named symbolic constant.

This file is kept separate from test_analytic.py: the latter is deliberately
QuTiP-free, whereas this oracle constructs quantum states. Both live in the
regression_analytic tier.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from _helpers import _spin_hilbert
from iontrap_dynamics.information import recoverability
from iontrap_dynamics.states import ghz_state

pytestmark = pytest.mark.regression_analytic

ATOL_RECOVERABILITY_CHANNEL = 1e-9


def _werner_recoverability(p_grid: np.ndarray) -> np.ndarray:
    """Recoverability over the two-qubit Werner family at each ``p`` in the grid."""
    h = _spin_hilbert(2)
    bell = ghz_state(h)
    bell_dm = bell * bell.dag()
    maximally_mixed = qutip.tensor(qutip.qeye(2), qutip.qeye(2)) / 4.0
    return np.array(
        [
            recoverability(
                float(p) * bell_dm + (1.0 - float(p)) * maximally_mixed,
                hilbert=h,
                system_indices=[0],
                accessible_indices=[1],
            )
            for p in p_grid
        ]
    )


def test_perfect_recovery_equals_system_entropy() -> None:
    """p = 1: the Bell pair is pure, so recoverability == H_S == 1 bit."""
    vals = _werner_recoverability(np.array([1.0]))
    assert vals[0] == pytest.approx(1.0, abs=ATOL_RECOVERABILITY_CHANNEL)


def test_full_decoherence_is_zero() -> None:
    """p = 0: the accessible qubit is uncorrelated with the system, so 0."""
    vals = _werner_recoverability(np.array([0.0]))
    assert vals[0] == pytest.approx(0.0, abs=ATOL_RECOVERABILITY_CHANNEL)


def test_recoverability_monotone_nondecreasing_in_p() -> None:
    """The channel is monotone non-decreasing as the Bell weight ``p`` rises."""
    p_grid = np.linspace(0.0, 1.0, 21)
    vals = _werner_recoverability(p_grid)
    assert vals[0] == pytest.approx(0.0, abs=ATOL_RECOVERABILITY_CHANNEL)
    assert vals[-1] == pytest.approx(1.0, abs=ATOL_RECOVERABILITY_CHANNEL)
    assert np.all(np.diff(vals) >= -ATOL_RECOVERABILITY_CHANNEL)
