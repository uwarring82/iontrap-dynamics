# SPDX-License-Identifier: MIT
"""MS-gate entanglement demo — concurrence + spin-motion log-negativity.

Runs the gate-closing Mølmer–Sørensen Hamiltonian on a two-ion system
(same scenario as ``run_demo_parity_scan.py``) and evaluates the
registered entanglement observables at every step. The MS gate's
textbook trajectory takes ``|↓↓, 0⟩`` through an intermediate
spin-motion-entangled path and lands on the Bell state
``(|↓↓⟩ − i|↑↑⟩)/√2`` at closing time ``t_gate``; the two registered
observables tell complementary stories:

- **Two-ion concurrence** ``C(ρ_{ij})`` ∈ [0, 1]: pure-spin
  entanglement after tracing the motion. Starts at 0 (product
  state), grows toward 1 at ``t_gate``.
- **Spin-motion log-negativity** ``E_N`` ≥ 0: bipartite entanglement
  across the spin ↔ mode cut. Starts at 0 (product of two
  uncoupled subsystems), grows during the loop, **closes back to 0
  at t_gate** (motion returns to vacuum, disentangled from spin).

The simultaneous trajectories make the gate's defining property
visible at a glance: spin entanglement grows monotonically while
spin-motion entanglement traces out a loop and returns to zero.

Usage::

    python tools/run_demo_bell_entanglement.py

Requires matplotlib. Falls back to "data only, no plot" if
matplotlib is absent.

Output::

    benchmarks/data/bell_entanglement_demo/
      arrays.npz        — times_s + ideal expectations (sigma_z, parity)
      entanglement.npz  — concurrence(t), log_neg_spin_motion(t), EoF(t)
      demo_report.json  — parameters, environment, gate-closing values
      plot.png          — concurrence + log-negativity overlay

Note: no canonical ``manifest.json`` emitted — cache v1 rejects
``storage_mode=EAGER`` which the nonlinear evaluators require.
Phase 1+ backend-annotated persistence will close that gap.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import qutip

from iontrap_dynamics import (
    concurrence_trajectory,
    entanglement_of_formation_trajectory,
    log_negativity_trajectory,
)
from iontrap_dynamics.analytic import (
    lamb_dicke_parameter,
    ms_gate_closing_detuning,
    ms_gate_closing_time,
)
from iontrap_dynamics.cache import compute_request_hash
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import detuned_ms_gate_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import parity, spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "bell_entanglement_demo"

N_FOCK = 12
N_STEPS = 200
LOOPS = 1

RABI_OVER_2PI_MHZ = 0.1
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)


def _build_scenario() -> tuple[HilbertSpace, list[object], qutip.Qobj, float, float]:
    mode = ModeConfig(
        label="com",
        frequency_rad_s=MODE_FREQ_RAD_S,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]) / np.sqrt(2.0),
    )
    system = IonSystem(species_per_ion=(mg25_plus(), mg25_plus()), modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"com": N_FOCK})

    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, WAVENUMBER_M_INV],
        carrier_rabi_frequency_rad_s=RABI_RAD_S,
        phase_rad=0.0,
    )
    eta = lamb_dicke_parameter(
        k_vec=drive.k_vector_m_inv,
        mode_eigenvector=mode.eigenvector_at_ion(0),
        ion_mass=mg25_plus().mass_kg,
        mode_frequency=MODE_FREQ_RAD_S,
    )
    delta = ms_gate_closing_detuning(
        carrier_rabi_frequency=RABI_RAD_S,
        lamb_dicke_parameter=eta,
        loops=LOOPS,
    )
    t_gate = ms_gate_closing_time(
        carrier_rabi_frequency=RABI_RAD_S,
        lamb_dicke_parameter=eta,
        loops=LOOPS,
    )
    hamiltonian = detuned_ms_gate_hamiltonian(
        hilbert, drive, "com", ion_indices=(0, 1), detuning_rad_s=delta
    )
    psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(N_FOCK, 0))
    return hilbert, hamiltonian, psi_0, t_gate, eta


def _environment() -> dict[str, str]:
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "qutip": qutip.__version__,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hilbert, hamiltonian, psi_0, t_gate, eta = _build_scenario()
    tlist = np.linspace(0.0, t_gate, N_STEPS)

    print(">>> running MS-gate entanglement demo")
    print(f"    LD parameter η = {eta:.4f}, t_gate = {t_gate * 1e6:.3f} μs")
    print(f"    N_Fock = {N_FOCK}, steps = {N_STEPS}")

    parameters = {
        "scenario": "bell_entanglement_demo",
        "N_fock": N_FOCK,
        "n_steps": N_STEPS,
        "loops": LOOPS,
        "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
        "phase_rad": 0.0,
        "initial_state": "|↓↓, 0⟩",
    }
    request_hash = compute_request_hash(parameters)

    t0 = time.perf_counter()
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=psi_0,
        times=tlist,
        observables=[
            spin_z(hilbert, 0),
            spin_z(hilbert, 1),
            parity(hilbert, (0, 1)),
        ],
        request_hash=request_hash,
        storage_mode=StorageMode.EAGER,  # EAGER so nonlinear evaluators see full states
        provenance_tags=("demo", "bell_entanglement"),
    )
    elapsed_solve = time.perf_counter() - t0
    print(f"    solve elapsed:        {elapsed_solve:.3f} s")

    assert result.states is not None, "EAGER storage mode should retain states"

    t1 = time.perf_counter()
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        concurrence = concurrence_trajectory(result.states, hilbert=hilbert, ion_indices=(0, 1))
        eof = entanglement_of_formation_trajectory(
            result.states, hilbert=hilbert, ion_indices=(0, 1)
        )
        log_neg = log_negativity_trajectory(result.states, hilbert=hilbert, partition="spins")
    elapsed_ent = time.perf_counter() - t1
    print(f"    entanglement elapsed: {elapsed_ent:.3f} s")
    print(
        f"    concurrence: start = {concurrence[0]:.3e}, "
        f"max = {concurrence.max():.3f}, end = {concurrence[-1]:.3f}"
    )
    print(
        f"    log-neg:     start = {log_neg[0]:.3e}, "
        f"max = {log_neg.max():.3f}, end = {log_neg[-1]:.3e}"
    )
    print(f"    EoF:         start = {eof[0]:.3e}, max = {eof.max():.3f}, end = {eof[-1]:.3f}")

    # Cache v1 rejects storage_mode=EAGER (states aren't serialisable
    # yet — Phase 1+ backend-annotated persistence will unlock this).
    # We output the expectation arrays + entanglement measures
    # directly; this demo isn't wired into migration-tier regression.
    np.savez(
        OUTPUT_DIR / "arrays.npz",
        times_s=tlist,
        sigma_z_0=result.expectations["sigma_z_0"],
        sigma_z_1=result.expectations["sigma_z_1"],
        parity_0_1=result.expectations["parity_0_1"],
    )
    np.savez(
        OUTPUT_DIR / "entanglement.npz",
        times_s=tlist,
        concurrence=concurrence,
        log_negativity_spin_motion=log_neg,
        entanglement_of_formation=eof,
    )

    demo_report = {
        "scenario": "bell_entanglement_demo",
        "purpose": (
            "registered entanglement observables on the MS-gate "
            "Bell-formation trajectory — concurrence grows "
            "monotonically, spin-motion log-negativity loops "
            "and closes back to 0 at t_gate."
        ),
        "workplan_reference": (
            "WORKPLAN_v0.3.md §5 Phase 1 — log-neg / concurrence / "
            "EoF as registered observables (Dispatch Q)"
        ),
        "convention_references": [
            "§2 Tensor ordering (spins before modes)",
            "§3 Spin basis",
            "§9 Bell state convention",
        ],
        "solve_elapsed_seconds": elapsed_solve,
        "entanglement_elapsed_seconds": elapsed_ent,
        "concurrence_start": float(concurrence[0]),
        "concurrence_max": float(concurrence.max()),
        "concurrence_end": float(concurrence[-1]),
        "log_negativity_start": float(log_neg[0]),
        "log_negativity_max": float(log_neg.max()),
        "log_negativity_end": float(log_neg[-1]),
        "eof_end": float(eof[-1]),
        "lamb_dicke_parameter": float(eta),
        "t_gate_us": float(t_gate * 1e6),
        "parameters": {
            **parameters,
            "duration_us": float(tlist[-1] * 1e6),
        },
        "canonical_request_hash": request_hash,
        "environment": _environment(),
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": 2,
    }
    (OUTPUT_DIR / "demo_report.json").write_text(
        json.dumps(demo_report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/"
        "{arrays.npz, entanglement.npz, demo_report.json}"
    )

    times_us = tlist * 1e6

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not installed; skipping plot")
        return 0

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(
        times_us,
        concurrence,
        color="#1f77b4",
        linewidth=1.8,
        label=r"concurrence $C(\rho_{01})$ — spin-spin entanglement",
    )
    ax.plot(
        times_us,
        log_neg,
        color="#d62728",
        linewidth=1.8,
        label=r"log-negativity $E_N$ — spin ↔ motion entanglement",
    )
    ax.plot(
        times_us,
        eof,
        color="#2ca02c",
        linewidth=1.2,
        linestyle="--",
        label=r"entanglement of formation $E_F(\rho_{01})$",
    )
    ax.axvline(t_gate * 1e6, color="grey", linewidth=0.4, linestyle=":", label=r"$t_\mathrm{gate}$")
    ax.axhline(1.0, color="grey", linewidth=0.3)
    ax.set_xlabel(r"time (μs)")
    ax.set_ylabel("entanglement measure")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(
        f"MS-gate entanglement trajectory — η={eta:.3f}, "
        f"$t_\\mathrm{{gate}}$={t_gate * 1e6:.2f} μs, loops={LOOPS}"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
