# SPDX-License-Identifier: MIT
"""Detuning-jitter demo — off-resonance Rabi mixing.

Parallel companion to ``run_demo_rabi_jitter.py``. Runs the carrier
Rabi scenario with :class:`DetuningJitter` adding a Gaussian
distribution of detuning offsets across shots. Each shot now
oscillates at its own generalised Rabi frequency

    Ω_gen(δ) = √(Ω₀² + δ²)

and — crucially — with its own *amplitude* reduction

    A(δ) = Ω₀² / (Ω₀² + δ²) = Ω₀² / Ω_gen²

because off-resonance drive only partially flips the spin. The
ensemble signature is therefore richer than plain Rabi-amplitude
jitter: the mean trajectory both dephases (frequency spread) and
damps toward ``+1 / 2`` (amplitude reduction at large |δ|).

The plot overlays:

1. Ideal noise-free ``⟨σ_z⟩(t) = −cos(Ω₀ t)``.
2. Ensemble mean over 200 shots with ``σ_δ / 2π = 300 kHz``.
3. Individual-shot cloud (faint) showing the frequency spread.

Usage::

    python tools/run_demo_detuning_jitter.py

Output::

    benchmarks/data/detuning_jitter_demo/
      arrays.npz        — times_s + ensemble sigma_z trajectories
      demo_report.json  — parameters, jitter stats
      plot.png          — ideal vs ensemble-mean vs shot cloud
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

from iontrap_dynamics import DetuningJitter, perturb_detuning
from iontrap_dynamics.cache import compute_request_hash
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import (
    carrier_hamiltonian,
    detuned_carrier_hamiltonian,
)
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "detuning_jitter_demo"

N_FOCK = 3
N_STEPS = 300
SHOTS = 200
SEED = 20260420
DETUNING_SIGMA_OVER_2PI_KHZ = 300.0  # σ_δ / 2π

RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)
DETUNING_SIGMA_RAD_S = 2 * np.pi * DETUNING_SIGMA_OVER_2PI_KHZ * 1e3


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
        detuning_rad_s=0.0,
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
    tlist = np.linspace(0.0, 5 * rabi_period, N_STEPS)

    jitter = DetuningJitter(sigma_rad_s=DETUNING_SIGMA_RAD_S)
    perturbed_drives = perturb_detuning(base_drive, jitter, shots=SHOTS, seed=SEED)

    print(">>> running detuning-jitter demo (off-resonance Rabi mixing)")
    print(f"    shots = {SHOTS}, seed = {SEED}, σ_δ / 2π = {DETUNING_SIGMA_OVER_2PI_KHZ} kHz")
    print(f"    Ω₀ / 2π = {RABI_OVER_2PI_MHZ} MHz, duration = 5·T_Ω")
    print(f"    σ_δ / Ω₀ = {DETUNING_SIGMA_RAD_S / RABI_RAD_S:.3f}")

    parameters = {
        "scenario": "detuning_jitter_demo",
        "N_fock": N_FOCK,
        "n_steps": N_STEPS,
        "shots": SHOTS,
        "seed": SEED,
        "detuning_sigma_over_2pi_kHz": DETUNING_SIGMA_OVER_2PI_KHZ,
        "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
        "phase_rad": 0.0,
        "initial_state": "|↓, 0⟩",
    }
    request_hash = compute_request_hash(parameters)

    # Ideal (noise-free) on-resonance trajectory.
    # Use the time-independent carrier_hamiltonian at δ=0 —
    # detuned_carrier_hamiltonian rejects δ=0 by design.
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
        provenance_tags=("demo", "detuning_jitter", "ideal"),
    )
    sz_ideal = ideal_result.expectations["sigma_z_0"]
    elapsed_ideal = time.perf_counter() - t0
    print(f"    ideal trajectory elapsed: {elapsed_ideal:.3f} s")

    # Ensemble of detuned trajectories.
    t1 = time.perf_counter()
    sz_stack = np.empty((SHOTS, N_STEPS), dtype=np.float64)
    for idx, drive in enumerate(perturbed_drives):
        ham = detuned_carrier_hamiltonian(hilbert, drive, ion_index=0)
        res = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=tlist,
            observables=[spin_z(hilbert, 0)],
            request_hash=request_hash,
            storage_mode=StorageMode.OMITTED,
            provenance_tags=("demo", "detuning_jitter", "perturbed"),
        )
        sz_stack[idx] = res.expectations["sigma_z_0"]
    elapsed_ensemble = time.perf_counter() - t1
    print(f"    ensemble elapsed:          {elapsed_ensemble:.3f} s")

    sz_mean = sz_stack.mean(axis=0)
    sz_std = sz_stack.std(axis=0, ddof=1)

    max_deviation = float(np.max(np.abs(sz_mean - sz_ideal)))
    terminal_damping = float(sz_mean[-SHOTS // 10 :].mean() + 1.0)
    print(f"    max |ensemble mean − ideal| = {max_deviation:.3f}")
    print(
        f"    ensemble-mean offset from −1 at end = {terminal_damping:.3f}  "
        "(off-resonance amplitude reduction signature)"
    )

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        times_s=tlist,
        sigma_z_ideal=sz_ideal,
        sigma_z_ensemble=sz_stack,
        sigma_z_mean=sz_mean,
        sigma_z_std=sz_std,
        detunings_rad_s=np.array([d.detuning_rad_s for d in perturbed_drives]),
    )

    demo_report = {
        "scenario": "detuning_jitter_demo",
        "purpose": (
            "companion to run_demo_rabi_jitter — shows off-resonance "
            "Rabi mixing under Gaussian detuning noise: both dephasing "
            "(frequency spread) and amplitude reduction "
            "(Ω² / (Ω² + δ²) factor) are visible in the ensemble mean."
        ),
        "workplan_reference": ("WORKPLAN_v0.3.md §5 Phase 1 systematics layer (Dispatch S)"),
        "convention_references": [
            "§3 Spin basis",
            "§4 Detuning sign",
            "§18.3 Jitter primitives",
        ],
        "ideal_elapsed_seconds": elapsed_ideal,
        "ensemble_elapsed_seconds": elapsed_ensemble,
        "max_mean_vs_ideal_deviation": max_deviation,
        "terminal_damping_from_minus_one": terminal_damping,
        "detuning_over_rabi_ratio": DETUNING_SIGMA_RAD_S / RABI_RAD_S,
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
        color="#ff7f0e",
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
        color="#ff7f0e",
        linewidth=1.8,
        label=f"ensemble mean ({SHOTS} shots)",
    )
    ax.axhline(0.0, color="grey", linewidth=0.3)
    ax.set_xlabel(r"time (μs)")
    ax.set_ylabel(r"$\langle \sigma_z \rangle$")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(
        "Detuning-jitter dephasing of carrier Rabi — "
        f"$\\sigma_\\delta / 2\\pi = {DETUNING_SIGMA_OVER_2PI_KHZ:.0f}$ kHz, "
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
