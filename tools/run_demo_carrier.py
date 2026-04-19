# SPDX-License-Identifier: MIT
"""Carrier Rabi flopping — illustrative demo, parallel to run_benchmark_sideband.py.

Unlike the sideband tool (which is a Phase 0.F timing benchmark), this
captures the carrier Hamiltonian's dynamics for pedagogical display.
Single ion on resonance, Bloch vector rotating in the y–z plane at the
carrier Rabi rate Ω.

Analytic prediction (CONVENTIONS.md §3 and analytic.carrier_rabi_sigma_z):
starting in |↓⟩ with φ = 0,

    ⟨σ_z⟩(t) = −cos(Ω t)
    ⟨σ_y⟩(t) = +sin(Ω t)
    ⟨σ_x⟩(t) = 0  (Bloch vector stays in the y–z plane)

The plot overlays the three numerical trajectories against the analytic
curves for visual cross-validation. Any visible discrepancy would
indicate a bug in the builder, operators, or state-prep.

Usage::

    python tools/run_demo_carrier.py

Requires matplotlib (``pip install -e ".[plot]"``). Falls back to
"data only, no plot" if matplotlib is absent.

Output::

    benchmarks/data/carrier_rabi_demo/
      arrays.npz     — times_us, sigma_x, sigma_y, sigma_z
      metadata.json  — parameters, environment, analytic formulas used
      plot.png       — three-panel figure with numerical vs analytic overlay
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

from iontrap_dynamics.cache import compute_request_hash, save_trajectory
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_x, spin_y, spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "carrier_rabi_demo"

# Physical parameters — on-resonance, zero phase, visible timescales
N_FOCK = 3  # tiny — carrier doesn't couple motion; Fock doesn't evolve
N_STEPS = 200

RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5  # only used so HilbertSpace has a mode to attach to
LASER_WAVELENGTH_NM = 280.0

RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)


def _build_scenario() -> tuple[HilbertSpace, qutip.Qobj, qutip.Qobj]:
    """Return (hilbert, H_carrier, initial |↓, 0⟩ ket)."""
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
        phase_rad=0.0,  # H / ℏ = (Ω/2) σ_x — drives rotation in y–z plane
    )
    hamiltonian = carrier_hamiltonian(hilbert, drive, ion_index=0)
    psi_0 = qutip.tensor(spin_down(), qutip.basis(N_FOCK, 0))
    return hilbert, hamiltonian, psi_0


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

    hilbert, hamiltonian, psi_0 = _build_scenario()
    rabi_period = 2 * np.pi / RABI_RAD_S
    tlist = np.linspace(0.0, 2 * rabi_period, N_STEPS)

    print(">>> running carrier Rabi flopping demo")
    print(f"    N_Fock = {N_FOCK} (trivial — carrier doesn't couple motion)")
    print(f"    steps = {N_STEPS}, duration = {tlist[-1] * 1e6:.3f} μs = 2 Rabi periods")
    print(f"    Ω/2π = {RABI_OVER_2PI_MHZ} MHz")

    # Parameters — frozen inputs, canonical-cache hash binding
    parameters = {
        "scenario": "carrier_rabi_demo",
        "N_fock": N_FOCK,
        "n_steps": N_STEPS,
        "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
        "phase_rad": 0.0,
        "initial_state": "|↓, 0⟩",
    }
    request_hash = compute_request_hash(parameters)

    t0 = time.perf_counter()
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=psi_0,
        times=tlist,
        observables=[spin_x(hilbert, 0), spin_y(hilbert, 0), spin_z(hilbert, 0)],
        request_hash=request_hash,
        storage_mode=StorageMode.OMITTED,
        provenance_tags=("demo", "carrier_rabi"),
    )
    elapsed = time.perf_counter() - t0
    print(f"    elapsed: {elapsed:.3f} s")

    sx_traj = result.expectations["sigma_x_0"]
    sy_traj = result.expectations["sigma_y_0"]
    sz_traj = result.expectations["sigma_z_0"]

    # Analytic comparison — starting in |↓⟩ with φ = 0:
    #   ⟨σ_x⟩ = 0,  ⟨σ_y⟩ = sin(Ω t),  ⟨σ_z⟩ = −cos(Ω t)
    sx_analytic = np.zeros_like(tlist)
    sy_analytic = np.sin(RABI_RAD_S * tlist)
    sz_analytic = -np.cos(RABI_RAD_S * tlist)

    max_error = max(
        float(np.max(np.abs(sx_traj - sx_analytic))),
        float(np.max(np.abs(sy_traj - sy_analytic))),
        float(np.max(np.abs(sz_traj - sz_analytic))),
    )
    print(f"    max |numerical − analytic| = {max_error:.3e}")

    # ------------------------------------------------------------------------
    # Canonical cache (manifest.json + arrays.npz, hash-verified)
    # ------------------------------------------------------------------------
    save_trajectory(result, OUTPUT_DIR, overwrite=True)

    # ------------------------------------------------------------------------
    # Demo-specific narrative (analytic comparison arrays, elapsed, env)
    # ------------------------------------------------------------------------
    np.savez(
        OUTPUT_DIR / "analytic_overlay.npz",
        sigma_x_analytic=sx_analytic,
        sigma_y_analytic=sy_analytic,
        sigma_z_analytic=sz_analytic,
    )

    demo_report = {
        "scenario": "carrier_rabi_demo",
        "purpose": "illustrative — overlays numerical solve() output against analytic formulas",
        "workplan_reference": "not a Phase 0.F benchmark; companion to run_benchmark_sideband.py",
        "analytic_formulas": {
            "sigma_x_0": "0",
            "sigma_y_0": "sin(Ω t)",
            "sigma_z_0": "-cos(Ω t)",
        },
        "elapsed_seconds": elapsed,
        "max_numerical_vs_analytic_error": max_error,
        "parameters": {
            **parameters,
            "duration_us": float(tlist[-1] * 1e6),
        },
        "arrays_schema_note": (
            "arrays.npz follows the canonical cache schema: 'times' in SI "
            "seconds, observables under 'expectation__<label>'. The analytic "
            "overlay arrays live separately in analytic_overlay.npz."
        ),
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
        "{manifest.json, arrays.npz, analytic_overlay.npz, demo_report.json}"
    )

    times_us = tlist * 1e6

    # ------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not installed; skipping plot")
        return 0

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.0, 6.0))

    for ax, label, num, ana, color in (
        (axes[0], r"$\langle \sigma_x \rangle$", sx_traj, sx_analytic, "#2ca02c"),
        (axes[1], r"$\langle \sigma_y \rangle$", sy_traj, sy_analytic, "#ff7f0e"),
        (axes[2], r"$\langle \sigma_z \rangle$", sz_traj, sz_analytic, "#1f77b4"),
    ):
        ax.plot(times_us, num, color=color, linewidth=2.0, label="mesolve")
        ax.plot(
            times_us,
            ana,
            color="black",
            linewidth=0.8,
            linestyle="--",
            label="analytic",
        )
        ax.axhline(0.0, color="grey", linewidth=0.3)
        ax.set_ylabel(label)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel(r"time (μs)")
    axes[0].set_title(
        f"Single-ion carrier Rabi flopping — Ω/2π = {RABI_OVER_2PI_MHZ} MHz, "
        f"max |Δ| = {max_error:.1e}"
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
