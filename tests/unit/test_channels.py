# SPDX-License-Identifier: MIT
"""Unit tests for the typed motional CPTP channels (WP-02 WI-3a, dispatch MCC).

Covers the channel records (validation, frozen-ness, collapse-operator
construction) and the ``sequences.solve(channels=…)`` integration (forces the
master-equation path, leaves the no-channel path byte-for-byte unchanged, and
guards the backend / solver conflicts). The closed-form decay oracles live in
``tests/regression/analytic/test_motional_channels.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import qutip

from _helpers import _single_mode_hilbert
from iontrap_dynamics import AmplitudeDamping, Dephasing, Heating
from iontrap_dynamics.channels import build_collapse_operators
from iontrap_dynamics.exceptions import ConventionError
from iontrap_dynamics.observables import Observable
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve

FOCK_DIM = 12


def _ground_ket(hilbert):
    """``|down> ⊗ |0>`` on the one-ion, one-mode space."""
    return qutip.tensor(qutip.basis(2, 0), qutip.basis(FOCK_DIM, 0))


def _fock_ket(hilbert, n: int):
    """``|down> ⊗ |n>``."""
    return qutip.tensor(qutip.basis(2, 0), qutip.basis(FOCK_DIM, n))


# --------------------------------------------------------------------------
# Channel records: validation + frozen-ness
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda r: AmplitudeDamping(mode="b", rate=r),
        lambda r: Dephasing(mode="b", rate=r),
        lambda r: Heating(mode="b", rate=r, n_bar_bath=0.0),
    ],
)
def test_negative_rate_raises(factory):
    with pytest.raises(ValueError, match="non-negative"):
        factory(-1.0)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_heating_bad_n_bar_raises(bad):
    with pytest.raises(ValueError):
        Heating(mode="b", rate=1.0, n_bar_bath=bad)


def test_nonfinite_rate_raises():
    with pytest.raises(ValueError, match="finite"):
        AmplitudeDamping(mode="b", rate=float("nan"))


def test_channels_are_frozen():
    ch = AmplitudeDamping(mode="b", rate=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ch.rate = 2.0  # type: ignore[misc]
    # dataclasses.replace gives a new instance (no mutation).
    ch2 = dataclasses.replace(ch, rate=5.0)
    assert ch.rate == 1.0
    assert ch2.rate == 5.0


# --------------------------------------------------------------------------
# Collapse-operator construction
# --------------------------------------------------------------------------


def test_amplitude_damping_collapse_operator():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    ops = AmplitudeDamping(mode="b", rate=4.0).collapse_operators(hilbert)
    assert len(ops) == 1
    # L = sqrt(rate) * a  ->  L == 2 * a
    assert ops[0] == 2.0 * hilbert.annihilation_for_mode("b")


def test_dephasing_collapse_operator():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    ops = Dephasing(mode="b", rate=9.0).collapse_operators(hilbert)
    assert len(ops) == 1
    assert ops[0] == 3.0 * hilbert.number_for_mode("b")


def test_heating_two_operators_when_nbar_positive():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    ops = Heating(mode="b", rate=1.0, n_bar_bath=2.0).collapse_operators(hilbert)
    assert len(ops) == 2  # down (a) and up (a-dagger)


def test_heating_one_operator_at_zero_temperature():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    ops = Heating(mode="b", rate=1.0, n_bar_bath=0.0).collapse_operators(hilbert)
    assert len(ops) == 1  # reduces to amplitude damping


def test_zero_rate_yields_no_operators():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    assert AmplitudeDamping(mode="b", rate=0.0).collapse_operators(hilbert) == []
    assert Dephasing(mode="b", rate=0.0).collapse_operators(hilbert) == []
    assert Heating(mode="b", rate=0.0, n_bar_bath=3.0).collapse_operators(hilbert) == []


def test_build_collapse_operators_concatenates():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    channels = [
        AmplitudeDamping(mode="b", rate=1.0),
        Heating(mode="b", rate=1.0, n_bar_bath=2.0),
        Dephasing(mode="b", rate=1.0),
    ]
    c_ops = build_collapse_operators(channels, hilbert)
    assert len(c_ops) == 1 + 2 + 1
    assert build_collapse_operators([], hilbert) == []


def test_unknown_mode_raises_convention_error():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    with pytest.raises(ConventionError, match="unknown mode"):
        AmplitudeDamping(mode="does_not_exist", rate=1.0).collapse_operators(hilbert)


# --------------------------------------------------------------------------
# solve(channels=…) integration
# --------------------------------------------------------------------------


def test_channels_force_mesolve_even_on_a_ket():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    n_op = hilbert.number_for_mode("b")
    res = solve(
        hilbert=hilbert,
        hamiltonian=0.0 * hilbert.identity(),
        initial_state=_fock_ket(hilbert, 3),
        times=np.linspace(0.0, 1e-4, 5),
        observables=(Observable(label="n", operator=n_op),),
        channels=[AmplitudeDamping(mode="b", rate=1000.0)],
    )
    assert res.metadata.backend_name == "qutip-mesolve"


def test_no_channels_leaves_sesolve_path_unchanged():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    res = solve(
        hilbert=hilbert,
        hamiltonian=0.0 * hilbert.identity(),
        initial_state=_fock_ket(hilbert, 3),
        times=np.linspace(0.0, 1e-4, 5),
    )
    # A pure ket with no dissipation still takes the fast Schrödinger path.
    assert res.metadata.backend_name == "qutip-sesolve"


def test_all_zero_rate_channels_do_not_force_mesolve():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    res = solve(
        hilbert=hilbert,
        hamiltonian=0.0 * hilbert.identity(),
        initial_state=_fock_ket(hilbert, 3),
        times=np.linspace(0.0, 1e-4, 5),
        channels=[AmplitudeDamping(mode="b", rate=0.0)],
    )
    assert res.metadata.backend_name == "qutip-sesolve"


def test_channels_with_jax_backend_raise():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    with pytest.raises(ConventionError, match="backend='qutip'"):
        solve(
            hilbert=hilbert,
            hamiltonian=0.0 * hilbert.identity(),
            initial_state=_fock_ket(hilbert, 3),
            times=np.linspace(0.0, 1e-4, 5),
            channels=[AmplitudeDamping(mode="b", rate=1.0)],
            backend="jax",
        )


def test_forcing_sesolve_with_channels_raises():
    hilbert = _single_mode_hilbert(FOCK_DIM)
    with pytest.raises(ConventionError, match="master-equation"):
        solve(
            hilbert=hilbert,
            hamiltonian=0.0 * hilbert.identity(),
            initial_state=_fock_ket(hilbert, 3),
            times=np.linspace(0.0, 1e-4, 5),
            channels=[AmplitudeDamping(mode="b", rate=1.0)],
            solver="sesolve",
        )


def test_cptp_trace_preserved():
    # Amplitude damping + dephasing from |5> relax downward, so the top Fock
    # level never saturates (no §13 convergence trip); both are trace-preserving.
    hilbert = _single_mode_hilbert(20)
    psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(20, 5))
    res = solve(
        hilbert=hilbert,
        hamiltonian=0.0 * hilbert.identity(),
        initial_state=psi0,
        times=np.linspace(0.0, 5e-4, 11),
        channels=[
            AmplitudeDamping(mode="b", rate=3000.0),
            Dephasing(mode="b", rate=1500.0),
        ],
        storage_mode=StorageMode.EAGER,
    )
    assert res.states is not None
    for state in res.states:
        assert state.tr() == pytest.approx(1.0, abs=1e-9)
