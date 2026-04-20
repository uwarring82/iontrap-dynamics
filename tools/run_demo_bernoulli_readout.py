# SPDX-License-Identifier: MIT
"""Bernoulli-readout demo — carrier Rabi flopping with shot noise.

Companion to ``run_demo_carrier.py``: takes the ideal ⟨σ_z⟩ trajectory,
maps it to the spin-up projection probability

    p_↑(t) = (1 + ⟨σ_z⟩(t)) / 2

in the atomic-physics convention (CONVENTIONS.md §3: ⟨σ_z⟩ = +1 on |↑⟩,
−1 on |↓⟩), applies :class:`BernoulliChannel` at a finite shot budget,
and plots the ideal curve against the shot-averaged estimator
``bits.mean(axis=0)``. The visible jitter is Bernoulli shot noise —
standard error σ_p̂ = sqrt(p(1−p)/N).

This is the first end-to-end exercise of the measurement boundary:

    state trajectory  →  observable expectation  →  probability  →  channel  →  counts

Usage::

    python tools/run_demo_bernoulli_readout.py

Requires matplotlib (``pip install -e ".[plot]"``). Falls back to
"data only, no plot" if matplotlib is absent.

Output::

    benchmarks/data/bernoulli_readout_demo/
      manifest.json     — canonical trajectory manifest
      arrays.npz        — times_s + ideal expectations
      measurement.npz   — probabilities, bits, shot-averaged estimate
      demo_report.json  — parameters, shot budget, seed, environment
      plot.png          — ideal p_↑(t) overlaid with shot-noisy estimate
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
    BernoulliChannel,
    sample_outcome,
)
from iontrap_dynamics.cache import compute_request_hash, save_trajectory
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
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "bernoulli_readout_demo"

N_FOCK = 3
N_STEPS = 200
SHOTS = 500
SEED = 20260420  # deterministic for archival reproducibility

RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)


def _build_scenario() -> tuple[HilbertSpace, qutip.Qobj, qutip.Qobj]:
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

    print(">>> running Bernoulli-readout demo (carrier Rabi with shot noise)")
    print(f"    shots = {SHOTS}, seed = {SEED}")
    print(f"    Ω/2π = {RABI_OVER_2PI_MHZ} MHz, steps = {N_STEPS}")

    parameters = {
        "scenario": "bernoulli_readout_demo",
        "N_fock": N_FOCK,
        "n_steps": N_STEPS,
        "shots": SHOTS,
        "seed": SEED,
        "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
        "phase_rad": 0.0,
        "initial_state": "|↓, 0⟩",
    }
    request_hash = compute_request_hash(parameters)

    t0 = time.perf_counter()
    trajectory = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=psi_0,
        times=tlist,
        observables=[spin_z(hilbert, 0)],
        request_hash=request_hash,
        storage_mode=StorageMode.OMITTED,
        provenance_tags=("demo", "bernoulli_readout"),
    )
    elapsed = time.perf_counter() - t0
    print(f"    trajectory elapsed: {elapsed:.3f} s")

    sigma_z = trajectory.expectations["sigma_z_0"]
    p_up = (1.0 + sigma_z) / 2.0  # CONVENTIONS.md §3

    measurement = sample_outcome(
        channel=BernoulliChannel(label="spin_up"),
        probabilities=p_up,
        shots=SHOTS,
        seed=SEED,
        upstream=trajectory,
        provenance_tags=("carrier_rabi",),
    )
    bits = measurement.sampled_outcome["spin_up"]
    estimate = bits.mean(axis=0)
    shot_noise_std = np.sqrt(p_up * (1.0 - p_up) / SHOTS)
    max_error = float(np.max(np.abs(estimate - p_up)))
    # Expected max of |Z| over N iid Normal(0, σ²) samples scales like
    # σ · sqrt(2 log N) — the Gumbel extreme-value bound, not 3σ.
    expected_max = float(shot_noise_std.max() * np.sqrt(2.0 * np.log(N_STEPS)))
    print(
        f"    max |estimate − p_↑| = {max_error:.3e}  "
        f"(extreme-value band σ·√(2 log N) = {expected_max:.3e})"
    )

    # Canonical trajectory cache for reproducibility chain
    save_trajectory(trajectory, OUTPUT_DIR, overwrite=True)

    # Measurement-specific arrays (separate from canonical schema)
    np.savez(
        OUTPUT_DIR / "measurement.npz",
        probability=p_up,
        counts=bits.sum(axis=0),
        estimate=estimate,
        shot_noise_std=shot_noise_std,
    )

    demo_report = {
        "scenario": "bernoulli_readout_demo",
        "purpose": (
            "first end-to-end exercise of the measurement boundary — "
            "ideal p_↑ vs. finite-shot Bernoulli estimator."
        ),
        "workplan_reference": "WORKPLAN_v0.3.md §5 Phase 1 measurement layer (Dispatch H)",
        "convention_references": [
            "§3 Spin basis (p_↑ = (1 + ⟨σ_z⟩)/2)",
            "§17 Measurement layer (staged)",
        ],
        "elapsed_seconds": elapsed,
        "max_estimator_error": max_error,
        "parameters": {
            **parameters,
            "duration_us": float(tlist[-1] * 1e6),
        },
        "trajectory_hash": measurement.trajectory_hash,
        "measurement_result_provenance_tags": list(measurement.metadata.provenance_tags),
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
        "{manifest.json, arrays.npz, measurement.npz, demo_report.json}"
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
    ax.plot(times_us, p_up, color="black", linewidth=1.5, label=r"ideal $p_\uparrow(t)$")
    ax.fill_between(
        times_us,
        p_up - shot_noise_std,
        p_up + shot_noise_std,
        color="#1f77b4",
        alpha=0.2,
        label=r"$\pm 1\sigma$ shot-noise band",
    )
    ax.plot(
        times_us,
        estimate,
        color="#d62728",
        linewidth=0.0,
        marker=".",
        markersize=3,
        label=f"estimate ({SHOTS} shots)",
    )
    ax.set_xlabel(r"time (μs)")
    ax.set_ylabel(r"$p_\uparrow$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"Bernoulli readout of carrier Rabi — Ω/2π = {RABI_OVER_2PI_MHZ} MHz, shots = {SHOTS}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
