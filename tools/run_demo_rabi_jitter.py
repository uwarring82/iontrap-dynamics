# SPDX-License-Identifier: MIT
"""Rabi-jitter demo — inhomogeneous dephasing of carrier Rabi flopping.

Opens the systematics track (Dispatch R). Runs the standard carrier
Rabi scenario from ``run_demo_carrier.py`` but with
:class:`RabiJitter` adding a 3 % multiplicative Gaussian noise on Ω
shot-to-shot. The ensemble of 200 shots produces the classic
inhomogeneous-dephasing signature:

- **Each individual shot** oscillates cleanly at its own
  slightly-shifted Rabi frequency.
- **The ensemble mean** ``⟨σ_z⟩(t)`` dephases — amplitude decays on
  a timescale ``T₂* ≈ 1 / (σ_Ω Δt)`` characteristic of
  shot-to-shot frequency spread.

The plot overlays three curves:

1. Ideal noise-free ``⟨σ_z⟩(t) = −cos(Ω₀ t)`` trajectory.
2. Ensemble-mean over 200 jittered shots — visibly damped.
3. ±1σ shot-to-shot spread band — widens over time as phases
   accumulate at different rates.

Usage::

    python tools/run_demo_rabi_jitter.py

Requires matplotlib. Falls back to "data only, no plot" if
matplotlib is absent.

Output::

    benchmarks/data/rabi_jitter_demo/
      arrays.npz        — times_s + ensemble sigma_z trajectories
      demo_report.json  — parameters, jitter stats, T₂* estimate
      plot.png          — ideal vs ensemble-mean vs ±1σ band
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

from iontrap_dynamics import RabiJitter, perturb_carrier_rabi
from iontrap_dynamics.cache import compute_request_hash
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "rabi_jitter_demo"

N_FOCK = 3
N_STEPS = 300
SHOTS = 200
SEED = 20260420
JITTER_SIGMA = 0.03  # 3 % Rabi jitter — modest but visible dephasing

RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)


def _build_scenario() -> tuple[HilbertSpace, DriveConfig, qutip.Qobj]:
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
    psi_0 = qutip.tensor(spin_down(), qutip.basis(N_FOCK, 0))
    return hilbert, drive, psi_0


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

    hilbert, base_drive, psi_0 = _build_scenario()
    rabi_period = 2 * np.pi / RABI_RAD_S
    # Go long enough to see the dephasing envelope build up — ~5 Rabi
    # periods, which at σ = 3 % gives ~0.5 rad of phase spread at the
    # end (visibly damped mean but still-oscillating individual shots).
    tlist = np.linspace(0.0, 5 * rabi_period, N_STEPS)

    jitter = RabiJitter(sigma=JITTER_SIGMA)
    perturbed_drives = perturb_carrier_rabi(base_drive, jitter, shots=SHOTS, seed=SEED)

    print(">>> running Rabi-jitter demo (inhomogeneous dephasing)")
    print(f"    shots = {SHOTS}, seed = {SEED}, σ_Ω = {JITTER_SIGMA * 100:.1f} %")
    print(f"    Ω₀/2π = {RABI_OVER_2PI_MHZ} MHz, steps = {N_STEPS}, duration = 5·T_Ω")

    parameters = {
        "scenario": "rabi_jitter_demo",
        "N_fock": N_FOCK,
        "n_steps": N_STEPS,
        "shots": SHOTS,
        "seed": SEED,
        "jitter_sigma": JITTER_SIGMA,
        "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
        "phase_rad": 0.0,
        "initial_state": "|↓, 0⟩",
    }
    request_hash = compute_request_hash(parameters)

    # Ideal (noise-free) trajectory for the overlay.
    hamiltonian_ideal = carrier_hamiltonian(hilbert, base_drive, ion_index=0)
    t0 = time.perf_counter()
    ideal_result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian_ideal,
        initial_state=psi_0,
        times=tlist,
        observables=[spin_z(hilbert, 0)],
        request_hash=request_hash,
        storage_mode=StorageMode.OMITTED,
        provenance_tags=("demo", "rabi_jitter", "ideal"),
    )
    sz_ideal = ideal_result.expectations["sigma_z_0"]
    elapsed_ideal = time.perf_counter() - t0
    print(f"    ideal trajectory elapsed: {elapsed_ideal:.3f} s")

    # Ensemble of jittered trajectories.
    t1 = time.perf_counter()
    sz_stack = np.empty((SHOTS, N_STEPS), dtype=np.float64)
    for idx, drive in enumerate(perturbed_drives):
        ham = carrier_hamiltonian(hilbert, drive, ion_index=0)
        res = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=tlist,
            observables=[spin_z(hilbert, 0)],
            request_hash=request_hash,
            storage_mode=StorageMode.OMITTED,
            provenance_tags=("demo", "rabi_jitter", "perturbed"),
        )
        sz_stack[idx] = res.expectations["sigma_z_0"]
    elapsed_ensemble = time.perf_counter() - t1
    print(f"    ensemble elapsed:          {elapsed_ensemble:.3f} s")

    sz_mean = sz_stack.mean(axis=0)
    sz_std = sz_stack.std(axis=0, ddof=1)

    # T₂* estimator — time at which the envelope decays to 1/e of its
    # initial amplitude. The noise-free amplitude is 1 (σ_z swings from
    # −1 to +1); jittered mean amplitude ≈ exp(−(σ_Ω Ω₀ t)² / 2) · cos(Ω₀ t).
    # Extract the envelope from |peak over each period|.
    analytic_t2_star = 1.0 / (JITTER_SIGMA * RABI_RAD_S)
    print(
        f"    analytic T₂* ≈ 1 / (σ·Ω₀) = {analytic_t2_star * 1e6:.3f} μs  "
        f"(Gaussian-envelope 1/e time)"
    )
    max_mean_deviation = float(np.max(np.abs(sz_mean - sz_ideal)))
    print(f"    max |ensemble mean − ideal| = {max_mean_deviation:.3f}  (dephasing-induced gap)")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        times_s=tlist,
        sigma_z_ideal=sz_ideal,
        sigma_z_ensemble=sz_stack,
        sigma_z_mean=sz_mean,
        sigma_z_std=sz_std,
    )

    demo_report = {
        "scenario": "rabi_jitter_demo",
        "purpose": (
            "first systematics-track dispatch — shows the inhomogeneous-"
            "dephasing signature of shot-to-shot Rabi-frequency jitter "
            "on the carrier Rabi flopping."
        ),
        "workplan_reference": ("WORKPLAN_v0.3.md §5 Phase 1 systematics layer (Dispatch R)"),
        "convention_references": [
            "§3 Spin basis",
            "§18.1 Noise taxonomy",
            "§18.2 Jitter composition pattern",
            "§18.3 RabiJitter semantics",
        ],
        "ideal_elapsed_seconds": elapsed_ideal,
        "ensemble_elapsed_seconds": elapsed_ensemble,
        "max_mean_vs_ideal_deviation": max_mean_deviation,
        "analytic_T2_star_us": analytic_t2_star * 1e6,
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
    # Individual shot trajectories as a faint cloud.
    ax.plot(
        times_us,
        sz_stack.T,
        color="#aaaaaa",
        linewidth=0.3,
        alpha=0.15,
    )
    ax.fill_between(
        times_us,
        sz_mean - sz_std,
        sz_mean + sz_std,
        color="#1f77b4",
        alpha=0.2,
        label=r"$\pm 1\sigma$ shot-to-shot spread",
    )
    ax.plot(
        times_us,
        sz_ideal,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label=r"ideal $\langle\sigma_z\rangle$ (no jitter)",
    )
    ax.plot(
        times_us,
        sz_mean,
        color="#1f77b4",
        linewidth=1.8,
        label=f"ensemble mean ({SHOTS} shots)",
    )
    # Reference T₂* line as a visual anchor.
    if analytic_t2_star * 1e6 < times_us[-1]:
        ax.axvline(
            analytic_t2_star * 1e6,
            color="#d62728",
            linewidth=0.6,
            linestyle=":",
            label=f"$T_2^\\ast \\approx {analytic_t2_star * 1e6:.2f}$ μs",
        )
    ax.axhline(0.0, color="grey", linewidth=0.3)
    ax.set_xlabel(r"time (μs)")
    ax.set_ylabel(r"$\langle \sigma_z \rangle$")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(
        "Inhomogeneous dephasing of carrier Rabi — "
        f"$\\sigma_\\Omega / \\Omega_0 = {JITTER_SIGMA * 100:.1f}\\%$, "
        f"{SHOTS} shots"
    )
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
