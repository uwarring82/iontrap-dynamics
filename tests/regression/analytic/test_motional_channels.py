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

import dataclasses

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
ATOL_WINDOWED = 3e-3
# The two channel orderings must differ by at least this much in ⟨n̂⟩ to count as
# a genuine R8 (non-commuting) demonstration — the actual gap here is ≈ 1.6.
R8_MIN_OCCUPATION_SEPARATION = 1.0


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


# --------------------------------------------------------------------------
# WI-3b: time-windowed (sequence-aware) channels + the R8 non-commuting test
# --------------------------------------------------------------------------


def test_windowed_damping_acts_only_inside_its_window() -> None:
    """Damping windowed to ``[0, T/2)``: ``⟨n̂⟩`` decays inside, then stays flat."""
    hilbert = _single_mode_hilbert(16)
    n0 = 4
    kappa = 4000.0
    total = 1.0e-3
    psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(16, n0))
    times = np.linspace(0.0, total, 101)  # T/2 = 5e-4 lands on the grid (index 50)
    res = solve(
        hilbert=hilbert,
        hamiltonian=_zero_hamiltonian(hilbert),
        initial_state=psi0,
        times=times,
        observables=(Observable(label="n", operator=hilbert.number_for_mode("b")),),
        channels=[AmplitudeDamping(mode="b", rate=kappa, window=(0.0, 0.5 * total))],
    )
    n_t = np.asarray(res.expectations["n"], dtype=np.float64)
    n_at_window_close = n0 * np.exp(-kappa * 0.5 * total)
    # Inside the window the decay follows e^{-κt}; afterwards the value is held.
    assert n_t[25] == pytest.approx(n0 * np.exp(-kappa * times[25]), abs=ATOL_WINDOWED)
    assert n_t[-1] == pytest.approx(n_at_window_close, abs=ATOL_WINDOWED)
    assert np.max(np.abs(n_t[55:] - n_t[-1])) < ATOL_WINDOWED  # flat after the window


def test_r8_non_commuting_channels_are_order_dependent() -> None:
    """R8: heating-then-damping ≠ damping-then-heating (card F3 hard requirement).

    Two disjoint ordered windows over ``[0, T]``. Heating drives ``⟨n̂⟩`` up to
    ``n̄(1 − e^{−κ_h T/2})``; amplitude damping then scales it by ``e^{−κ_d T/2}``.
    Damping the ground state first is a no-op, so swapping the order leaves the
    heated value un-damped — a materially different final state. The library must
    **not** assume the two dissipators commute.
    """
    hilbert = _single_mode_hilbert(40)  # headroom for the heated occupation
    kappa_h = 4000.0
    n_bar = 2.0
    kappa_d = 5000.0
    total = 1.0e-3
    half = 0.5 * total
    psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(40, 0))
    times = np.linspace(0.0, total, 101)
    obs = (Observable(label="n", operator=hilbert.number_for_mode("b")),)

    heating = Heating(mode="b", rate=kappa_h, n_bar_bath=n_bar)
    damping = AmplitudeDamping(mode="b", rate=kappa_d)

    res_heat_then_damp = solve(
        hilbert=hilbert,
        hamiltonian=_zero_hamiltonian(hilbert),
        initial_state=psi0,
        times=times,
        observables=obs,
        channels=[
            dataclasses.replace(heating, window=(0.0, half)),
            dataclasses.replace(damping, window=(half, total)),
        ],
    )
    res_damp_then_heat = solve(
        hilbert=hilbert,
        hamiltonian=_zero_hamiltonian(hilbert),
        initial_state=psi0,
        times=times,
        observables=obs,
        channels=[
            dataclasses.replace(heating, window=(half, total)),
            dataclasses.replace(damping, window=(0.0, half)),
        ],
    )
    n_heat_then_damp = float(res_heat_then_damp.expectations["n"][-1])
    n_damp_then_heat = float(res_damp_then_heat.expectations["n"][-1])

    n_after_heating = n_bar * (1.0 - np.exp(-kappa_h * half))
    analytic_heat_then_damp = n_after_heating * np.exp(-kappa_d * half)
    analytic_damp_then_heat = n_after_heating  # damping the ground state is a no-op

    assert n_heat_then_damp == pytest.approx(analytic_heat_then_damp, abs=ATOL_WINDOWED)
    assert n_damp_then_heat == pytest.approx(analytic_damp_then_heat, abs=ATOL_WINDOWED)
    # The two orderings give a materially different final occupation.
    assert abs(n_heat_then_damp - n_damp_then_heat) > R8_MIN_OCCUPATION_SEPARATION


def test_short_window_between_output_points_is_not_skipped() -> None:
    """Regression: a short window lying entirely between two output points fires.

    Heating on ``[0.53T, 0.54T)`` with a coarse output spacing of ``0.1T`` — the
    whole window falls between grid points. A plain ``min(diff(times))`` step cap
    would step over it (final ``⟨n̂⟩ = 0``); capping ``max_step`` at the
    union-of-times-and-endpoints gap captures it.
    """
    hilbert = _single_mode_hilbert(40)
    kappa_h = 4000.0
    n_bar = 2.0
    total = 1.0e-3
    t0, t1 = 0.53 * total, 0.54 * total
    psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(40, 0))
    times = np.linspace(0.0, total, 11)  # spacing 0.1T; the window is between points
    res = solve(
        hilbert=hilbert,
        hamiltonian=_zero_hamiltonian(hilbert),
        initial_state=psi0,
        times=times,
        observables=(Observable(label="n", operator=hilbert.number_for_mode("b")),),
        channels=[Heating(mode="b", rate=kappa_h, n_bar_bath=n_bar, window=(t0, t1))],
    )
    n_final = float(res.expectations["n"][-1])
    analytic = n_bar * (1.0 - np.exp(-kappa_h * (t1 - t0)))
    assert analytic > 0.05  # the window genuinely does something
    assert n_final == pytest.approx(analytic, abs=ATOL_WINDOWED)
