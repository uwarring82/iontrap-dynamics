# SPDX-License-Identifier: MIT
"""Analytic-regression oracles for the two-mode SU(1,1) surface (WP-02 WI-1/WI-2).

Binding closed-form anchors (CONVENTIONS.md §23, staged for the v0.3 freeze) for:

- the **factory** ``states.two_mode_squeezed_vacuum`` — per-mode mean occupation
  ``⟨n̂_a⟩ = ⟨n̂_b⟩ = sinh²|z|``;
- the **Hamiltonian** ``hamiltonians.two_mode_squeezing_hamiltonian`` — evolving
  the two-mode vacuum for a time ``τ`` gives ``⟨n̂⟩ = sinh²(gτ)`` with the su(1,1)
  difference number ``n̂_a − n̂_b`` conserved (the Casimir label);
- their **mutual consistency** — the Hamiltonian-evolved vacuum is the factory
  state with ``|z| = gτ`` (high state fidelity);
- the **beamsplitter** ``hamiltonians.beamsplitter_hamiltonian`` — total
  occupation ``n̂_a + n̂_b`` conserved (SU(2)).

Tolerances are named symbolic constants here, not read from an artefact. Kept
separate from the QuTiP-free ``test_analytic.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from _helpers import _two_mode_hilbert
from iontrap_dynamics.hamiltonians import (
    beamsplitter_hamiltonian,
    two_mode_squeezing_hamiltonian,
)
from iontrap_dynamics.observables import Observable
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.states import two_mode_squeezed_vacuum

pytestmark = pytest.mark.regression_analytic

ATOL_OCCUPATION = 1e-4
ATOL_CASIMIR = 1e-9
MIN_CONSISTENCY_FIDELITY = 1.0 - 1e-4

FOCK_DIM = 24


def _motional_vacuum(hilbert: object) -> qutip.Qobj:
    """``|↓⟩ ⊗ |0⟩_a ⊗ |0⟩_b`` on the one-ion, two-mode space."""
    return qutip.tensor(qutip.basis(2, 0), qutip.basis(FOCK_DIM, 0), qutip.basis(FOCK_DIM, 0))


@pytest.mark.parametrize("z", [0.3, 0.6, 0.4 + 0.2j, 0.5j])
def test_factory_per_mode_occupation_is_sinh_squared(z) -> None:
    psi = two_mode_squeezed_vacuum(FOCK_DIM, z)
    n_a = qutip.tensor(qutip.num(FOCK_DIM), qutip.qeye(FOCK_DIM))
    n_b = qutip.tensor(qutip.qeye(FOCK_DIM), qutip.num(FOCK_DIM))
    expected = np.sinh(abs(z)) ** 2
    assert qutip.expect(n_a, psi) == pytest.approx(expected, abs=ATOL_OCCUPATION)
    assert qutip.expect(n_b, psi) == pytest.approx(expected, abs=ATOL_OCCUPATION)


def test_hamiltonian_evolution_generates_sinh_squared_and_conserves_casimir() -> None:
    hilbert = _two_mode_hilbert(FOCK_DIM)
    g = 3000.0
    tau = 1.5e-4  # r = g·tau = 0.45
    hamiltonian = two_mode_squeezing_hamiltonian(hilbert, g=g, phase=0.0)
    n_a = hilbert.number_for_mode("a")
    n_b = hilbert.number_for_mode("b")
    times = np.linspace(0.0, tau, 9)
    res = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=_motional_vacuum(hilbert),
        times=times,
        observables=(Observable(label="na", operator=n_a), Observable(label="nb", operator=n_b)),
    )
    na_t = np.asarray(res.expectations["na"], dtype=np.float64)
    nb_t = np.asarray(res.expectations["nb"], dtype=np.float64)
    # per-mode occupation follows sinh^2(g t) at every time
    expected = np.sinh(g * times) ** 2
    assert np.max(np.abs(na_t - expected)) < ATOL_OCCUPATION
    assert np.max(np.abs(nb_t - expected)) < ATOL_OCCUPATION
    # su(1,1) Casimir: n_a - n_b stays zero throughout
    assert np.max(np.abs(na_t - nb_t)) < ATOL_CASIMIR


def test_hamiltonian_evolution_matches_the_factory_state() -> None:
    hilbert = _two_mode_hilbert(FOCK_DIM)
    g = 3000.0
    tau = 1.5e-4
    hamiltonian = two_mode_squeezing_hamiltonian(hilbert, g=g, phase=0.0)
    res = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=_motional_vacuum(hilbert),
        times=np.linspace(0.0, tau, 2),
        storage_mode=StorageMode.EAGER,
    )
    assert res.states is not None
    final = res.states[-1]
    # Trace out the (untouched) spin; compare the motional state to the factory.
    motional = final.ptrace([1, 2])
    expected = two_mode_squeezed_vacuum(FOCK_DIM, -g * tau)  # phase 0 -> z = -g·tau
    assert qutip.fidelity(motional, expected) > MIN_CONSISTENCY_FIDELITY


def test_hamiltonian_matches_factory_with_nonzero_phase() -> None:
    # §23 is explicitly phase/sign/ordering: check the signed complex map
    # z = −gτ·e^{iφ} at a non-trivial phase, not just |z| = gτ.
    hilbert = _two_mode_hilbert(FOCK_DIM)
    g = 3000.0
    tau = 1.5e-4
    phi = 0.7
    hamiltonian = two_mode_squeezing_hamiltonian(hilbert, g=g, phase=phi)
    res = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=_motional_vacuum(hilbert),
        times=np.linspace(0.0, tau, 2),
        storage_mode=StorageMode.EAGER,
    )
    assert res.states is not None
    motional = res.states[-1].ptrace([1, 2])
    expected = two_mode_squeezed_vacuum(FOCK_DIM, -g * tau * np.exp(1j * phi))
    assert qutip.fidelity(motional, expected) > MIN_CONSISTENCY_FIDELITY


def test_beamsplitter_conserves_total_occupation() -> None:
    hilbert = _two_mode_hilbert(FOCK_DIM)
    hamiltonian = beamsplitter_hamiltonian(hilbert, coupling=4000.0, phase=0.0)
    n_total = hilbert.number_for_mode("a") + hilbert.number_for_mode("b")
    # seed one excitation in mode a; the beamsplitter only rotates it to b and back
    initial = qutip.tensor(qutip.basis(2, 0), qutip.basis(FOCK_DIM, 1), qutip.basis(FOCK_DIM, 0))
    res = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=initial,
        times=np.linspace(0.0, 5e-4, 21),
        observables=(Observable(label="ntot", operator=n_total),),
    )
    ntot_t = np.asarray(res.expectations["ntot"], dtype=np.float64)
    assert np.max(np.abs(ntot_t - 1.0)) < ATOL_OCCUPATION
