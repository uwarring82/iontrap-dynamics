# SPDX-License-Identifier: MIT
"""Unit tests for recoverability (WI-2 of WP-01, dispatch EDB).

The recoverability measure is the clamped coherent information
max(0, S(rho_A) - S(rho_{S union A})) (Schumacher-Nielsen). Endpoints: perfect
recovery (the system maximally entangled with the accessible part) gives H_S;
full decoherence (the accessible part uncorrelated with the system) gives 0;
monotone in between, checked on a Werner-state dephasing family.
"""

from __future__ import annotations

import pytest
import qutip

from _helpers import _spin_hilbert
from iontrap_dynamics import ghz_state, recoverability
from iontrap_dynamics.operators import spin_down, spin_up


def test_perfect_recovery_equals_system_entropy() -> None:
    # Bell pair (GHZ over 2 qubits): system maximally entangled with accessible.
    h = _spin_hilbert(2)
    bell = ghz_state(h)
    r = recoverability(bell, hilbert=h, system_indices=[0], accessible_indices=[1])
    assert r == pytest.approx(1.0, abs=1e-9)  # H_S = 1 bit


def test_full_decoherence_is_zero() -> None:
    # 3 qubits: system (0) entangled only with an inaccessible reference (2);
    # the accessible qubit (1) sits in a product state, uncorrelated with S.
    h = _spin_hilbert(3)
    s_ddd = qutip.tensor(spin_down(), spin_down(), spin_down())
    s_udu = qutip.tensor(spin_up(), spin_down(), spin_up())
    state = (s_ddd + s_udu).unit()
    r = recoverability(state, hilbert=h, system_indices=[0], accessible_indices=[1])
    assert r == pytest.approx(0.0, abs=1e-12)  # coherent information < 0, clamped to 0


def test_recoverability_monotone_in_werner_fidelity() -> None:
    h = _spin_hilbert(2)
    bell = ghz_state(h)
    bell_dm = bell * bell.dag()
    maximally_mixed = qutip.tensor(qutip.qeye(2), qutip.qeye(2)) / 4.0

    def werner(p: float) -> qutip.Qobj:
        return p * bell_dm + (1.0 - p) * maximally_mixed

    ps = [0.0, 0.85, 0.92, 1.0]
    vals = [
        recoverability(werner(p), hilbert=h, system_indices=[0], accessible_indices=[1]) for p in ps
    ]
    assert vals[0] == pytest.approx(0.0, abs=1e-9)  # fully mixed: nothing recoverable
    assert vals[-1] == pytest.approx(1.0, abs=1e-9)  # perfect Bell: full bit
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))  # non-decreasing
    assert 0.0 < vals[1] < vals[2] < 1.0  # a genuine intermediate rise


def test_dimension_mismatch_is_rejected() -> None:
    h2 = _spin_hilbert(2)
    state3 = ghz_state(_spin_hilbert(3))  # 3-qubit state against a 2-qubit Hilbert space
    with pytest.raises(ValueError):
        recoverability(state3, hilbert=h2, system_indices=[0], accessible_indices=[1])


def test_recoverability_validation() -> None:
    h = _spin_hilbert(2)
    bell = ghz_state(h)
    with pytest.raises(ValueError):  # overlapping system / accessible
        recoverability(bell, hilbert=h, system_indices=[0], accessible_indices=[0])
    with pytest.raises(ValueError):  # out-of-range index
        recoverability(bell, hilbert=h, system_indices=[0], accessible_indices=[5])
    with pytest.raises(ValueError):  # empty set
        recoverability(bell, hilbert=h, system_indices=[], accessible_indices=[1])
