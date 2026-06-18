# SPDX-License-Identifier: MIT
"""Oracle for the Wittemer-2018 single-mode benchmark (Phase A, WI-4 / ND4).

Binding qualitative anchor for ``tools/run_wittemer_2018_benchmark.py``. It
re-computes small single-mode spin-boson trajectories (independently of the tool)
and asserts the integration reproduces the paper's qualitative features:

* the reduced-spin trace distance starts at 1 (orthogonal initial pair) and
  **revives** off resonance (information back-flow) → ``𝒩 > 0`` (Fig. 2);
* ``𝒩`` has **resonance structure** in the spin–mode detuning (Fig. 4a);
* the QPN bias is positive and **vanishes as shots → ∞** (Fig. 3 — via the ND3
  estimator on the benchmark's Bloch trajectories);
* **added spin decoherence suppresses** ``𝒩`` (the deliberate extension beyond
  the paper, which keeps ``Γ_dec`` negligible).

A tight numeric match to the paper needs the (APS-gated) Supplemental; this is
the qualitative + ND3-consistency acceptance gate (WP-04 §3, ND4).
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics.clos2016 import clos2016_initial_state, clos2016_spin_boson_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.information import blp_non_markovianity, non_markovianity_qpn_bias
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_x, spin_y, spin_z
from iontrap_dynamics.operators import sigma_x_ion, sigma_y_ion, sigma_z_ion
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

pytestmark = pytest.mark.regression_analytic

TWO_PI = 2.0 * np.pi
OMEGA_E = TWO_PI * 1.920e6
OMEGA = TWO_PI * 0.100e6
ION_MASS_KG = 25.0 * 1.66053906660e-27
MAX_PHONONS = 8  # fast oracle with enough Fock headroom at n̄=0.1
N_BAR = 0.1  # near the paper's Fig. 4(a) regime — sharp resonance, minimal Fock occupation
TAU = TWO_PI / OMEGA
TIMES = np.linspace(0.0, 9.0 * TAU, 300)  # fine sampling: 𝒩 is sampling-rate (γ) sensitive


def _hilbert() -> HilbertSpace:
    mode = ModeConfig(
        label="b", frequency_rad_s=OMEGA_E, eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]])
    )
    return HilbertSpace(
        system=IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,)),
        fock_truncations={"b": MAX_PHONONS + 1},
    )


def _hamiltonian(detuning_rad_s: float) -> qutip.Qobj:
    return clos2016_spin_boson_hamiltonian(
        max_phonons=MAX_PHONONS,
        axial_frequency_rad_s=OMEGA_E,
        dimensionless_mode_frequencies=[1.0],
        center_mode_weights=[1.0],
        carrier_rabi_frequency_rad_s=OMEGA,
        detuning_rad_s=detuning_rad_s,
        ion_mass_kg=ION_MASS_KG,
    )


def _bloch(hilbert: HilbertSpace, hamiltonian: qutip.Qobj, theta: float) -> np.ndarray:
    psi0 = clos2016_initial_state(
        max_phonons=MAX_PHONONS, mean_occupations=[N_BAR], theta_rad=theta
    )
    e = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=psi0,
        times=TIMES,
        observables=(spin_x(hilbert, 0), spin_y(hilbert, 0), spin_z(hilbert, 0)),
    ).expectations
    return np.stack([e["sigma_x_0"], e["sigma_y_0"], e["sigma_z_0"]], axis=1)


def _bloch_dephased(
    hilbert: HilbertSpace, hamiltonian: qutip.Qobj, theta: float, gamma: float
) -> np.ndarray:
    """Bloch trajectory with a benchmark-local spin-dephasing c_op (WP-04 R5)."""
    psi0 = clos2016_initial_state(
        max_phonons=MAX_PHONONS, mean_occupations=[N_BAR], theta_rad=theta
    )
    e_ops = [hilbert.spin_op_for_ion(o, 0) for o in (sigma_x_ion(), sigma_y_ion(), sigma_z_ion())]
    sz = hilbert.spin_op_for_ion(sigma_z_ion(), 0)
    # √(Γ_dec/2)·σ_z ⇒ coherence ∝ exp(−Γ_dec·t): gamma is the physical pure-dephasing rate.
    c_ops = [np.sqrt(gamma / 2.0) * sz] if gamma > 0.0 else []
    res = qutip.mesolve(hamiltonian, psi0, TIMES, c_ops=c_ops, e_ops=e_ops)
    return np.stack([res.expect[0], res.expect[1], res.expect[2]], axis=1)


def _distance(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(b1 - b2, axis=1)


def _distance_at(hilbert: HilbertSpace, detuning: float) -> np.ndarray:
    ham = _hamiltonian(detuning)
    return _distance(_bloch(hilbert, ham, 0.0), _bloch(hilbert, ham, np.pi))


def test_trace_distance_starts_at_one_and_revives() -> None:
    h = _hilbert()
    d = _distance_at(h, 0.0)  # on resonance: deepest dip, largest back-flow
    assert d[0] == pytest.approx(1.0, abs=1e-6)  # orthogonal initial pair
    assert d.min() < 0.9  # a substantial dip (information flows to the mode)
    assert blp_non_markovianity(d) > 0.03  # revivals → non-Markovian (𝒩 > 0)


def test_non_markovianity_resonance_structure() -> None:
    # Local-probing signature: the spin-mode coupling and the information back-flow
    # are strongest on resonance, so D dips deepest and 𝒩 is largest at δω_z = 0,
    # weakening as |δω_z| grows.
    h = _hilbert()
    d_on, d_off = _distance_at(h, 0.0), _distance_at(h, 2.0 * OMEGA)
    assert d_on.min() < d_off.min() - 0.05  # deeper dip on resonance
    assert blp_non_markovianity(d_on) > blp_non_markovianity(d_off)  # more 𝒩 on resonance


def test_qpn_bias_positive_then_vanishes_on_benchmark_trajectories() -> None:
    h = _hilbert()
    ham = _hamiltonian(0.0)
    b_up, b_dn = _bloch(h, ham, 0.0), _bloch(h, ham, np.pi)
    small = non_markovianity_qpn_bias(b_up, b_dn, shots=50, repeats=300, seed=2018)
    large = non_markovianity_qpn_bias(b_up, b_dn, shots=200_000, repeats=300, seed=2018)
    assert small.qpn_bias > 0.0
    assert abs(large.qpn_bias) < 0.1 * small.qpn_bias  # ℬ_QPN → 0 as r → ∞


def test_added_decoherence_suppresses_non_markovianity() -> None:
    h = _hilbert()
    ham = _hamiltonian(0.0)
    n_clean = blp_non_markovianity(
        _distance(_bloch_dephased(h, ham, 0.0, 0.0), _bloch_dephased(h, ham, np.pi, 0.0))
    )
    n_dephased = blp_non_markovianity(
        _distance(_bloch_dephased(h, ham, 0.0, OMEGA), _bloch_dephased(h, ham, np.pi, OMEGA))
    )
    assert n_clean > 0.0
    assert n_dephased < 0.5 * n_clean  # strong spin dephasing kills the revivals
