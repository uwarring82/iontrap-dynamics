# SPDX-License-Identifier: MIT
"""Analytic-regression oracle for the Darwinism-redundancy benchmark.

Binding closed-form anchor for ``tools/run_benchmark_darwinism_redundancy.py``
(WP-01 §7 row 3, dispatch EDE): the GHZ cascade — one system qubit perfectly
copied onto N environment qubits in a single GHZ state — exhibits the
partial-information plateau ``I(S:F) = H_S = 1`` bit for every non-empty proper
fragment (jumping to 2 bits only at the full environment), and redundancy
``R_δ = N`` at deficit ``δ = 0.1``. The benchmark tool writes
max_numerical_vs_analytic_error; this is the independent binding assertion,
with the tolerance as a named symbolic constant.

This file is kept separate from test_analytic.py: the latter is deliberately
QuTiP-free (it validates closed forms in iontrap_dynamics.analytic without a
backend), whereas the Darwinism oracle constructs quantum states. Both live in
the regression_analytic tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import _spin_hilbert
from iontrap_dynamics.information import partial_information_plot, redundancy
from iontrap_dynamics.states import ghz_state

pytestmark = pytest.mark.regression_analytic

ATOL_DARWINISM_REDUNDANCY = 1e-9
DELTA = 0.1


@pytest.mark.parametrize("n_env", [3, 4, 5, 6])
def test_ghz_cascade_partial_information_plateau(n_env: int) -> None:
    """PIP is 0, then a flat 1-bit plateau over proper fragments, then 2."""
    h = _spin_hilbert(n_env + 1)
    state = ghz_state(h)
    pip = partial_information_plot(
        state,
        hilbert=h,
        system_indices=[0],
        environment_indices=list(range(1, n_env + 1)),
    )
    assert pip[0] == pytest.approx(0.0, abs=ATOL_DARWINISM_REDUNDANCY)
    # Every non-empty proper fragment carries exactly the system bit H_S = 1.
    for f in range(1, n_env):
        assert pip[f] == pytest.approx(1.0, abs=ATOL_DARWINISM_REDUNDANCY)
    # Only the full environment lifts the curve to 2 bits.
    assert pip[n_env] == pytest.approx(2.0, abs=ATOL_DARWINISM_REDUNDANCY)


@pytest.mark.parametrize("n_env", [3, 4, 5, 6])
def test_ghz_cascade_redundancy_is_n(n_env: int) -> None:
    """Each single environment qubit already carries the bit, so R_δ = N."""
    h = _spin_hilbert(n_env + 1)
    r = redundancy(
        ghz_state(h),
        hilbert=h,
        system_indices=[0],
        environment_indices=list(range(1, n_env + 1)),
        delta=DELTA,
    )
    assert r == pytest.approx(float(n_env), abs=ATOL_DARWINISM_REDUNDANCY)


def test_ghz_cascade_plateau_equals_system_entropy() -> None:
    """The plateau height equals H_S = 1 bit independent of environment size."""
    for n_env in (3, 4, 5, 6):
        h = _spin_hilbert(n_env + 1)
        pip = partial_information_plot(
            ghz_state(h),
            hilbert=h,
            system_indices=[0],
            environment_indices=list(range(1, n_env + 1)),
        )
        plateau = np.asarray(pip[1:n_env], dtype=np.float64)
        assert np.allclose(plateau, 1.0, atol=ATOL_DARWINISM_REDUNDANCY)
