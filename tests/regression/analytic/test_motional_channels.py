# SPDX-License-Identifier: MIT
"""Analytic-regression oracles for the motional CPTP channels (WP-02 WI-3a, MCC).

Binding closed-form anchors for :mod:`iontrap_dynamics.channels` routed through
:func:`iontrap_dynamics.sequences.solve`, each reproduced by the QuTiP
master-equation path:

- **amplitude damping** — ``⟨n̂(t)⟩ = n₀ · e^{−κt}`` (relaxes to the ground state);
- **heating (bath n̄)** — ``⟨n̂(t)⟩ = n̄ (1 − e^{−κt}) → n̄`` (relaxes to the bath);
- **pure dephasing** — ``⟨n̂⟩`` constant; the coherence quadrature
  ``⟨X̂(t)⟩ = ⟨X̂(0)⟩ · e^{−γt/2}`` decays.

These are the convention staged for CONVENTIONS.md §24. Compute-only and
deterministic; the tolerances are named symbolic constants here, never read from
an artefact. The sequence-aware (time-windowed) composition and the R8
non-commuting-order test are WI-3b.

Kept separate from ``test_analytic.py`` (which is QuTiP-free); this oracle solves
dissipative trajectories. Both live in the ``regression_analytic`` tier.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from _helpers import _single_mode_hilbert
from iontrap_dynamics import AmplitudeDamping, Dephasing, Heating
from iontrap_dynamics.observables import Observable
from iontrap_dynamics.sequences import solve

pytestmark = pytest.mark.regression_analytic

ATOL_DAMPING_DECAY = 1e-4
ATOL_HEATING_RELAXATION = 1e-3
ATOL_DEPHASING_COHERENCE = 1e-4


def _zero_hamiltonian(hilbert: qutip.Qobj) -> qutip.Qobj:
    """A zero Hamiltonian on the full space — pure dissipation, no coherent drive."""
    return 0.0 * hilbert.identity()


def test_amplitude_damping_decay() -> None:
    """``⟨n̂(t)⟩ = n₀ · e^{−κt}`` for zero-temperature amplitude damping."""
    hilbert = _single_mode_hilbert(16)
    n0 = 5
    kappa = 4000.0
    psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(16, n0))
    times = np.linspace(0.0, 6e-4, 61)
    res = solve(
        hilbert=hilbert,
        hamiltonian=_zero_hamiltonian(hilbert),
        initial_state=psi0,
        times=times,
        observables=(Observable(label="n", operator=hilbert.number_for_mode("b")),),
        channels=[AmplitudeDamping(mode="b", rate=kappa)],
    )
    n_t = np.asarray(res.expectations["n"], dtype=np.float64)
    analytic = n0 * np.exp(-kappa * times)
    assert np.max(np.abs(n_t - analytic)) < ATOL_DAMPING_DECAY


def test_heating_relaxes_to_bath_occupation() -> None:
    """``⟨n̂(t)⟩ = n̄ (1 − e^{−κt})`` from the ground state, with steady state n̄."""
    hilbert = _single_mode_hilbert(40)  # room for the thermal tail at n̄ = 2
    kappa = 5000.0
    n_bar = 2.0
    psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(40, 0))
    times = np.linspace(0.0, 2e-3, 101)
    res = solve(
        hilbert=hilbert,
        hamiltonian=_zero_hamiltonian(hilbert),
        initial_state=psi0,
        times=times,
        observables=(Observable(label="n", operator=hilbert.number_for_mode("b")),),
        channels=[Heating(mode="b", rate=kappa, n_bar_bath=n_bar)],
    )
    n_t = np.asarray(res.expectations["n"], dtype=np.float64)
    analytic = n_bar * (1.0 - np.exp(-kappa * times))
    assert np.max(np.abs(n_t - analytic)) < ATOL_HEATING_RELAXATION
    assert n_t[-1] == pytest.approx(n_bar, abs=ATOL_HEATING_RELAXATION)


def test_dephasing_preserves_occupation_decoheres_coherence() -> None:
    """Pure dephasing: ``⟨n̂⟩`` constant; ``⟨X̂(t)⟩ = ⟨X̂(0)⟩ e^{−γt/2}``."""
    hilbert = _single_mode_hilbert(8)
    gamma = 6000.0
    # Mode in (|0⟩ + |1⟩)/√2: ⟨n̂⟩ = 0.5, and the quadrature ⟨X̂⟩ = 1/√2.
    mode_superposition = (qutip.basis(8, 0) + qutip.basis(8, 1)).unit()
    psi0 = qutip.tensor(qutip.basis(2, 0), mode_superposition)
    a = hilbert.annihilation_for_mode("b")
    quadrature = (a + a.dag()) / np.sqrt(2.0)
    times = np.linspace(0.0, 5e-4, 51)
    res = solve(
        hilbert=hilbert,
        hamiltonian=_zero_hamiltonian(hilbert),
        initial_state=psi0,
        times=times,
        observables=(
            Observable(label="n", operator=hilbert.number_for_mode("b")),
            Observable(label="X", operator=quadrature),
        ),
        channels=[Dephasing(mode="b", rate=gamma)],
    )
    n_t = np.asarray(res.expectations["n"], dtype=np.float64)
    x_t = np.asarray(res.expectations["X"], dtype=np.float64)
    assert np.max(np.abs(n_t - 0.5)) < ATOL_DEPHASING_COHERENCE
    analytic_x = (1.0 / np.sqrt(2.0)) * np.exp(-0.5 * gamma * times)
    assert np.max(np.abs(x_t - analytic_x)) < ATOL_DEPHASING_COHERENCE
