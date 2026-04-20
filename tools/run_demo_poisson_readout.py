# SPDX-License-Identifier: MIT
"""Poisson-readout demo — photon-counting model on carrier Rabi.

Takes the same carrier Rabi trajectory as the Bernoulli / Binomial
demos and runs it through a Poisson-counting detector model. The
instantaneous scattering rate is built as

    λ(t) = λ_dark + (λ_bright − λ_dark) · p_↑(t)

with ``λ_bright = 10`` counts/shot and ``λ_dark = 0.5`` counts/shot —
figures loosely matched to typical atomic-fluorescence readout windows.
:class:`PoissonChannel` samples ``(shots, n_times)`` photon counts, and
the plot overlays the shot-averaged mean (with ``±1σ`` Poisson band
``sqrt(λ/N)``) against the ideal rate.

Exercises the rate-side of the measurement boundary:

    state trajectory  →  observable expectation  →  probability  →  rate  →  channel  →  counts

Threshold-based bright/dark discrimination (turning per-shot counts
into a binary up / down classification) is a protocol-layer concern
and arrives in Dispatch L.

Usage::

    python tools/run_demo_poisson_readout.py

Requires matplotlib (``pip install -e ".[plot]"``). Falls back to
"data only, no plot" if matplotlib is absent.

Output::

    benchmarks/data/poisson_readout_demo/
      manifest.json     — canonical trajectory manifest
      arrays.npz        — times_s + ideal expectations
      measurement.npz   — rate, counts, mean, 1σ Poisson band
      demo_report.json  — parameters, shot budget, seed, environment
      plot.png          — ideal λ(t) overlaid with shot-averaged counts
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
    PoissonChannel,
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
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "poisson_readout_demo"

N_FOCK = 3
N_STEPS = 200
SHOTS = 500
SEED = 20260420

RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

# Photon-counting rates per shot (atomic-fluorescence-window typical).
LAMBDA_BRIGHT = 10.0
LAMBDA_DARK = 0.5

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

    print(">>> running Poisson-readout demo (carrier Rabi, photon counts)")
    print(f"    shots = {SHOTS}, seed = {SEED}")
    print(f"    λ_bright = {LAMBDA_BRIGHT}, λ_dark = {LAMBDA_DARK} counts/shot")
    print(f"    Ω/2π = {RABI_OVER_2PI_MHZ} MHz, steps = {N_STEPS}")

    parameters = {
        "scenario": "poisson_readout_demo",
        "N_fock": N_FOCK,
        "n_steps": N_STEPS,
        "shots": SHOTS,
        "seed": SEED,
        "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
        "phase_rad": 0.0,
        "lambda_bright": LAMBDA_BRIGHT,
        "lambda_dark": LAMBDA_DARK,
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
        provenance_tags=("demo", "poisson_readout"),
    )
    elapsed = time.perf_counter() - t0
    print(f"    trajectory elapsed: {elapsed:.3f} s")

    sigma_z = trajectory.expectations["sigma_z_0"]
    p_up = (1.0 + sigma_z) / 2.0
    rate = LAMBDA_DARK + (LAMBDA_BRIGHT - LAMBDA_DARK) * p_up

    measurement = sample_outcome(
        channel=PoissonChannel(label="photon_counts"),
        inputs=rate,
        shots=SHOTS,
        seed=SEED,
        upstream=trajectory,
        provenance_tags=("carrier_rabi",),
    )
    counts = measurement.sampled_outcome["photon_counts"]
    mean_count = counts.mean(axis=0)
    shot_noise_std = np.sqrt(rate / SHOTS)
    max_error = float(np.max(np.abs(mean_count - rate)))
    expected_max = float(shot_noise_std.max() * np.sqrt(2.0 * np.log(N_STEPS)))
    print(
        f"    max |mean − λ| = {max_error:.3e}  "
        f"(extreme-value band σ·√(2 log N) = {expected_max:.3e})"
    )

    save_trajectory(trajectory, OUTPUT_DIR, overwrite=True)

    np.savez(
        OUTPUT_DIR / "measurement.npz",
        rate=rate,
        counts=counts,
        mean_count=mean_count,
        shot_noise_std=shot_noise_std,
    )

    demo_report = {
        "scenario": "poisson_readout_demo",
        "purpose": (
            "rate-side counterpart to the Bernoulli / Binomial demos — "
            "photon-counting Poisson channel on the carrier Rabi trajectory."
        ),
        "workplan_reference": "WORKPLAN_v0.3.md §5 Phase 1 measurement layer (Dispatch K)",
        "convention_references": [
            "§3 Spin basis (p_↑ = (1 + ⟨σ_z⟩)/2)",
            "§17 Measurement layer (staged) — §17.6 rate semantics",
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
    ax.plot(times_us, rate, color="black", linewidth=1.5, label=r"ideal $\lambda(t)$")
    ax.fill_between(
        times_us,
        rate - shot_noise_std,
        rate + shot_noise_std,
        color="#ff7f0e",
        alpha=0.2,
        label=r"$\pm 1\sigma$ mean-of-Poisson band",
    )
    ax.plot(
        times_us,
        mean_count,
        color="#ff7f0e",
        linewidth=0.0,
        marker=".",
        markersize=3,
        label=f"mean ({SHOTS} shots)",
    )
    ax.axhline(LAMBDA_BRIGHT, color="grey", linewidth=0.3, linestyle="--")
    ax.axhline(LAMBDA_DARK, color="grey", linewidth=0.3, linestyle="--")
    ax.set_xlabel(r"time (μs)")
    ax.set_ylabel(r"photon counts per shot")
    ax.set_title(
        f"Poisson readout of carrier Rabi — "
        f"$\\lambda_\\mathrm{{bright}}={LAMBDA_BRIGHT}$, "
        f"$\\lambda_\\mathrm{{dark}}={LAMBDA_DARK}$, shots = {SHOTS}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
