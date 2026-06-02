# SPDX-License-Identifier: MIT
"""Analytic-regression oracle for the keystone QFI-scaling benchmark.

Binding closed-form anchor for ``tools/run_benchmark_qfi_scaling.py`` (WP-01
WI-1 / WI-3, dispatch EDA): the SLD quantum Fisher information of a GHZ probe
scales as N^2 (the Heisenberg limit) and of a product probe as N (the standard
quantum limit), under the collective generator J_z = 0.5 * sum_i sigma_z. The
benchmark tool records max_numerical_vs_analytic_error; this is the independent
binding assertion, with the tolerance as a named symbolic constant.

This file is kept separate from test_analytic.py: the latter is deliberately
QuTiP-free (it validates closed forms in iontrap_dynamics.analytic without a
backend), whereas the QFI oracle constructs quantum states. Both live in the
regression_analytic tier.
"""

from __future__ import annotations

import pytest
import qutip

from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.information import quantum_fisher_information_trajectory
from iontrap_dynamics.operators import sigma_z_ion, spin_down, spin_up
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import ghz_state
from iontrap_dynamics.system import IonSystem

pytestmark = pytest.mark.regression_analytic

ATOL_QFI_SCALING = 1e-9


def _spin_hilbert(n_ions: int) -> HilbertSpace:
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})


def _collective_jz(hilbert: HilbertSpace) -> qutip.Qobj:
    ops = [hilbert.spin_op_for_ion(sigma_z_ion(), i) for i in range(hilbert.n_ions)]
    total = ops[0]
    for op in ops[1:]:
        total = total + op
    return 0.5 * total


def _product_plus(n_ions: int) -> qutip.Qobj:
    plus = (spin_up() + spin_down()).unit()
    return qutip.tensor([plus] * n_ions)


@pytest.mark.parametrize("n_ions", [1, 2, 3, 4, 5])
def test_ghz_qfi_is_heisenberg_n_squared(n_ions: int) -> None:
    h = _spin_hilbert(n_ions)
    qfi = quantum_fisher_information_trajectory(
        [ghz_state(h)], hilbert=h, generator=_collective_jz(h)
    )
    assert qfi[0] == pytest.approx(float(n_ions**2), abs=ATOL_QFI_SCALING)


@pytest.mark.parametrize("n_ions", [1, 2, 3, 4, 5])
def test_product_qfi_is_standard_quantum_limit_n(n_ions: int) -> None:
    h = _spin_hilbert(n_ions)
    qfi = quantum_fisher_information_trajectory(
        [_product_plus(n_ions)], hilbert=h, generator=_collective_jz(h)
    )
    assert qfi[0] == pytest.approx(float(n_ions), abs=ATOL_QFI_SCALING)


def test_ghz_qfi_is_n_times_product_qfi() -> None:
    """The Heisenberg/SQL gap: F_Q(GHZ) = N * F_Q(product) for N >= 2."""
    for n_ions in (2, 3, 4, 5):
        h = _spin_hilbert(n_ions)
        jz = _collective_jz(h)
        ghz = quantum_fisher_information_trajectory([ghz_state(h)], hilbert=h, generator=jz)[0]
        product = quantum_fisher_information_trajectory(
            [_product_plus(n_ions)], hilbert=h, generator=jz
        )[0]
        assert ghz == pytest.approx(n_ions * product, abs=ATOL_QFI_SCALING)
