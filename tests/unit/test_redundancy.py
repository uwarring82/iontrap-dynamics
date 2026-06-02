# SPDX-License-Identifier: MIT
"""Unit tests for quantum-Darwinism redundancy (WI-2 of WP-01, dispatch EDB).

Covers fragment_mutual_information, partial_information_plot, and redundancy on
the canonical GHZ-cascade decoherence model: one system qubit perfectly copied
onto N environment qubits (the GHZ state across 1 + N qubits). The partial-
information plot then shows the textbook plateau I(S:F) = H_S = 1 bit for every
non-empty proper fragment, jumping to 2 only at the full environment, and the
redundancy is R_delta = N (each single qubit already carries the whole bit).
"""

from __future__ import annotations

import pytest
import qutip

from _helpers import _spin_hilbert
from iontrap_dynamics import (
    fragment_mutual_information,
    ghz_state,
    partial_information_plot,
    redundancy,
)
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.operators import spin_down


def _ghz_cascade(n_env: int) -> tuple[HilbertSpace, qutip.Qobj, list[int], list[int]]:
    """1 system qubit + n_env environment qubits, all in one GHZ state."""
    h = _spin_hilbert(n_env + 1)
    return h, ghz_state(h), [0], list(range(1, n_env + 1))


@pytest.mark.parametrize("n_env", [3, 5])
def test_partial_information_plot_shows_darwinism_plateau(n_env: int) -> None:
    h, state, sys, env = _ghz_cascade(n_env)
    pip = partial_information_plot(state, hilbert=h, system_indices=sys, environment_indices=env)
    assert pip.shape == (n_env + 1,)
    assert pip[0] == pytest.approx(0.0, abs=1e-9)
    # Plateau at H_S = 1 bit for every non-empty proper fragment.
    for f in range(1, n_env):
        assert pip[f] == pytest.approx(1.0, abs=1e-9)
    # The full environment purifies the system: I(S:E) = 2 H_S.
    assert pip[n_env] == pytest.approx(2.0, abs=1e-9)


def test_fragment_mutual_information_single_qubit_carries_full_bit() -> None:
    h, state, sys, env = _ghz_cascade(4)
    info = fragment_mutual_information(
        state, hilbert=h, system_indices=sys, fragment_indices=[env[0]]
    )
    assert info == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n_env", [3, 5])
def test_redundancy_is_maximal_for_ghz_cascade(n_env: int) -> None:
    h, state, sys, env = _ghz_cascade(n_env)
    # Each single qubit already carries (1 - delta) H_S, so R_delta = N.
    r = redundancy(state, hilbert=h, system_indices=sys, environment_indices=env, delta=0.1)
    assert r == pytest.approx(float(n_env), abs=1e-9)


def test_redundancy_zero_when_system_has_no_information() -> None:
    # System qubit pure and uncorrelated: H_S = 0, nothing to encode redundantly.
    h = _spin_hilbert(3)
    product = qutip.tensor([spin_down()] * 3)
    r = redundancy(product, hilbert=h, system_indices=[0], environment_indices=[1, 2], delta=0.1)
    assert r == pytest.approx(0.0, abs=1e-12)


def test_dimension_mismatch_is_rejected() -> None:
    h2 = _spin_hilbert(2)
    state3 = ghz_state(_spin_hilbert(3))  # 3-qubit state against a 2-qubit Hilbert space
    with pytest.raises(ValueError):
        fragment_mutual_information(state3, hilbert=h2, system_indices=[0], fragment_indices=[1])
    with pytest.raises(ValueError):
        redundancy(state3, hilbert=h2, system_indices=[0], environment_indices=[1])


def test_mutual_information_validation() -> None:
    h, state, _, _ = _ghz_cascade(3)
    with pytest.raises(ValueError):  # overlapping system / fragment
        fragment_mutual_information(state, hilbert=h, system_indices=[0], fragment_indices=[0])
    with pytest.raises(ValueError):  # out-of-range index
        fragment_mutual_information(state, hilbert=h, system_indices=[0], fragment_indices=[99])
    with pytest.raises(ValueError):  # empty set
        fragment_mutual_information(state, hilbert=h, system_indices=[], fragment_indices=[1])


def test_redundancy_delta_validation() -> None:
    h, state, sys, env = _ghz_cascade(3)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            redundancy(state, hilbert=h, system_indices=sys, environment_indices=env, delta=bad)
