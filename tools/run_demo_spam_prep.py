# SPDX-License-Identifier: MIT
"""SPAM preparation-error demo — imperfect initial state on carrier Rabi.

Closes the systematics track (Dispatch U). Builds an imperfect
initial state from :func:`imperfect_spin_ground` +
:func:`imperfect_motional_ground` via
:func:`iontrap_dynamics.states.compose_density`, runs the carrier-
Rabi scenario, and contrasts three trajectories:

1. **Ideal** — pure ``|↓, 0⟩`` ket. ``⟨σ_z⟩(t) = −cos(Ω t)``
   swings full amplitude.
2. **Spin-prep error only** — ``p_↑_prep = 0.03`` (3 % pumping
   leakage). The (1 − 2p) factor on the classical mixture damps
   the Rabi amplitude; the trajectory still oscillates cleanly
   but with a reduced swing.
3. **Combined SPAM** — spin-prep error + thermal motion
   ``n̄_prep = 0.1``. Adds very slight additional shift; on the
   carrier (spin-only) transition the thermal motion mostly
   commutes with the Rabi drive and has a small effect. On a
   sideband transition the effect would be much larger.

The demo verifies two invariants:

- Amplitude of the spin-prep-only trajectory matches the analytic
  ``(1 − 2·p_↑_prep)`` prediction to solver-tolerance precision.
- All initial states have trace 1 (probability conservation).

Usage::

    python tools/run_demo_spam_prep.py

Output::

    benchmarks/data/spam_prep_demo/
      arrays.npz        — times_s + three ⟨σ_z⟩ trajectories
      demo_report.json  — parameters, analytic amplitudes, initial traces
      plot.png          — overlay of ideal vs SPAM-affected curves
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
    SpinPreparationError,
    ThermalPreparationError,
    imperfect_motional_ground,
    imperfect_spin_ground,
)
from iontrap_dynamics.cache import compute_request_hash
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_z
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import compose_density, ground_state
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "spam_prep_demo"

N_FOCK = 8  # comfortable margin above 4·n̄ + 4 for n̄ = 0.1 — no convergence warning
N_STEPS = 300

P_UP_PREP = 0.03  # 3 % spin-pumping error
N_BAR_PREP = 0.10  # residual thermal motion after cooling

RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)


def _build_scenario() -> tuple[HilbertSpace, qutip.Qobj]:
    mode = ModeConfig(
        label="axial",
        frequency_rad_s=MODE_FREQ_RAD_S,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"axial": N_FOCK})
    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, WAVENUMBER_M_INV],
        carrier_rabi_frequency_rad_s=RABI_RAD_S,
        phase_rad=0.0,
    )
    ham = carrier_hamiltonian(hilbert, drive, ion_index=0)
    return hilbert, ham


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

    hilbert, hamiltonian = _build_scenario()
    rabi_period = 2 * np.pi / RABI_RAD_S
    tlist = np.linspace(0.0, 2 * rabi_period, N_STEPS)

    # Three initial states: ideal pure ket, spin-prep error only, full SPAM.
    fock_zero = qutip.basis(N_FOCK, 0)
    rho_ideal_mode = fock_zero * fock_zero.dag()

    psi_ideal = ground_state(hilbert)  # pure ket
    rho_spin_only = compose_density(
        hilbert,
        spin_states_per_ion=[imperfect_spin_ground(SpinPreparationError(p_up_prep=P_UP_PREP))],
        mode_states_by_label={"axial": rho_ideal_mode},
    )
    rho_full_spam = compose_density(
        hilbert,
        spin_states_per_ion=[imperfect_spin_ground(SpinPreparationError(p_up_prep=P_UP_PREP))],
        mode_states_by_label={
            "axial": imperfect_motional_ground(
                ThermalPreparationError(n_bar_prep=N_BAR_PREP),
                fock_dim=N_FOCK,
            )
        },
    )

    print(">>> running SPAM-prep demo (imperfect initial state)")
    print(f"    p_up_prep = {P_UP_PREP}, n_bar_prep = {N_BAR_PREP}")
    print(f"    Ω / 2π = {RABI_OVER_2PI_MHZ} MHz, N_fock = {N_FOCK}")
    print(
        f"    initial traces: ideal={1.0:.6f}, "
        f"spin-only={float(rho_spin_only.tr()):.6f}, "
        f"full-SPAM={float(rho_full_spam.tr()):.6f}"
    )

    def _run(initial_state: qutip.Qobj, tag: str) -> np.ndarray:
        parameters = {
            "scenario": "spam_prep_demo",
            "tag": tag,
            "N_fock": N_FOCK,
            "n_steps": N_STEPS,
            "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
            "p_up_prep": P_UP_PREP if tag != "ideal" else 0.0,
            "n_bar_prep": N_BAR_PREP if tag == "full_spam" else 0.0,
        }
        res = solve(
            hilbert=hilbert,
            hamiltonian=hamiltonian,
            initial_state=initial_state,
            times=tlist,
            observables=[spin_z(hilbert, 0)],
            request_hash=compute_request_hash(parameters),
            storage_mode=StorageMode.OMITTED,
            provenance_tags=("demo", "spam_prep", tag),
        )
        return res.expectations["sigma_z_0"]

    t0 = time.perf_counter()
    sz_ideal = _run(psi_ideal, "ideal")
    sz_spin_only = _run(rho_spin_only, "spin_only")
    sz_full_spam = _run(rho_full_spam, "full_spam")
    elapsed = time.perf_counter() - t0
    print(f"    three solves elapsed: {elapsed:.3f} s")

    # Analytic check: with spin-prep error p, the ensemble average
    # ⟨σ_z⟩(t) = (1 − 2p) · (−cos(Ω t)). Amplitude = (1 − 2p).
    expected_amplitude = 1.0 - 2.0 * P_UP_PREP
    measured_amplitude = float(np.ptp(sz_spin_only) / 2.0)
    print(
        f"    spin-only amplitude: expected {expected_amplitude:.4f}, "
        f"measured {measured_amplitude:.4f}  "
        f"(error {abs(measured_amplitude - expected_amplitude):.2e})"
    )

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        times_s=tlist,
        sigma_z_ideal=sz_ideal,
        sigma_z_spin_only=sz_spin_only,
        sigma_z_full_spam=sz_full_spam,
    )

    demo_report = {
        "scenario": "spam_prep_demo",
        "purpose": (
            "closing demo for the systematics track — preparation-side "
            "SPAM shifts the carrier-Rabi trajectory in an analytically "
            "tractable way; the (1 − 2p) amplitude factor is the "
            "classical-mixture signature of incomplete spin-state "
            "preparation."
        ),
        "workplan_reference": ("WORKPLAN_v0.3.md §5 Phase 1 systematics layer (Dispatch U)"),
        "convention_references": [
            "§3 Spin basis",
            "§18.1 Noise taxonomy (SPAM — preparation side)",
            "§18.5 State-preparation errors",
        ],
        "solves_elapsed_seconds": elapsed,
        "p_up_prep": P_UP_PREP,
        "n_bar_prep": N_BAR_PREP,
        "expected_spin_only_amplitude": expected_amplitude,
        "measured_spin_only_amplitude": measured_amplitude,
        "amplitude_error": abs(measured_amplitude - expected_amplitude),
        "initial_trace_ideal": 1.0,
        "initial_trace_spin_only": float(rho_spin_only.tr()),
        "initial_trace_full_spam": float(rho_full_spam.tr()),
        "parameters": {
            "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
            "N_fock": N_FOCK,
            "n_steps": N_STEPS,
            "duration_us": float(tlist[-1] * 1e6),
            "initial_state_ideal": "|↓, 0⟩ (pure ket)",
            "initial_state_spin_only": (f"(1 − {P_UP_PREP}) |↓⟩⟨↓| + {P_UP_PREP} |↑⟩⟨↑| ⊗ |0⟩⟨0|"),
            "initial_state_full_spam": (
                f"spin-prep ({P_UP_PREP}) ⊗ thermal(n̄={N_BAR_PREP}) "
                f"truncated to {N_FOCK} Fock states"
            ),
        },
        "environment": _environment(),
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": 2,
    }
    (OUTPUT_DIR / "demo_report.json").write_text(
        json.dumps(demo_report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/{{arrays.npz, demo_report.json}}")

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
        sz_ideal,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label=r"ideal $|\downarrow, 0\rangle$",
    )
    ax.plot(
        times_us,
        sz_spin_only,
        color="#1f77b4",
        linewidth=1.8,
        label=f"spin-prep error ($p_↑ = {P_UP_PREP}$)",
    )
    ax.plot(
        times_us,
        sz_full_spam,
        color="#d62728",
        linewidth=1.8,
        linestyle=":",
        label=(f"full SPAM ($p_↑ = {P_UP_PREP}$, $\\bar n = {N_BAR_PREP}$)"),
    )
    ax.axhline(expected_amplitude, color="#1f77b4", linewidth=0.3, linestyle=":")
    ax.axhline(-expected_amplitude, color="#1f77b4", linewidth=0.3, linestyle=":")
    ax.axhline(0.0, color="grey", linewidth=0.3)
    ax.set_xlabel(r"time (μs)")
    ax.set_ylabel(r"$\langle\sigma_z\rangle$")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(
        "SPAM preparation-error carrier Rabi — "
        f"amplitude reduction $(1 − 2p) = {expected_amplitude:.3f}$"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
