# SPDX-License-Identifier: MIT
"""Binomial-readout demo — carrier Rabi with aggregated shot counts.

Companion to ``run_demo_bernoulli_readout.py`` — same physics, same
dynamics core, but uses :class:`BinomialChannel` to collapse the
``(shots, n_times)`` bit matrix into ``(n_times,)`` aggregate counts.
Estimator is ``counts / shots``; the overlaid band is the Wald
(normal-approximation) 1σ CI, which for ``p̂ = n/N`` has width
``sqrt(p̂(1−p̂)/N)``.

Exercises the same measurement boundary as the Bernoulli demo:

    state trajectory  →  observable expectation  →  probability  →  channel  →  counts

…but now returning counts directly, which is the natural input to
Dispatch P's Wilson- and Clopper–Pearson-CI estimators.

Usage::

    python tools/run_demo_binomial_readout.py

Requires matplotlib (``pip install -e ".[plot]"``). Falls back to
"data only, no plot" if matplotlib is absent.

Output::

    benchmarks/data/binomial_readout_demo/
      manifest.json     — canonical trajectory manifest
      arrays.npz        — times_s + ideal expectations
      measurement.npz   — probabilities, counts, estimate, 1σ CI
      demo_report.json  — parameters, shot budget, seed, environment
      plot.png          — ideal p_↑(t) overlaid with aggregated estimate
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
    BinomialChannel,
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
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "binomial_readout_demo"

N_FOCK = 3
N_STEPS = 200
SHOTS = 500
SEED = 20260420

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

    print(">>> running Binomial-readout demo (carrier Rabi, aggregated counts)")
    print(f"    shots = {SHOTS}, seed = {SEED}")
    print(f"    Ω/2π = {RABI_OVER_2PI_MHZ} MHz, steps = {N_STEPS}")

    parameters = {
        "scenario": "binomial_readout_demo",
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
        provenance_tags=("demo", "binomial_readout"),
    )
    elapsed = time.perf_counter() - t0
    print(f"    trajectory elapsed: {elapsed:.3f} s")

    sigma_z = trajectory.expectations["sigma_z_0"]
    p_up = (1.0 + sigma_z) / 2.0

    measurement = sample_outcome(
        channel=BinomialChannel(label="spin_up"),
        inputs=p_up,
        shots=SHOTS,
        seed=SEED,
        upstream=trajectory,
        provenance_tags=("carrier_rabi",),
    )
    counts = measurement.sampled_outcome["spin_up"]
    estimate = counts.astype(np.float64) / SHOTS
    # Wald 1σ CI — width sqrt(p̂(1−p̂)/N); breaks down at p̂ = 0, 1 and
    # for small N. Wilson / Clopper–Pearson arrive with Dispatch P.
    wald_sigma = np.sqrt(estimate * (1.0 - estimate) / SHOTS)
    max_error = float(np.max(np.abs(estimate - p_up)))
    expected_max = float(
        np.sqrt(p_up * (1.0 - p_up) / SHOTS).max() * np.sqrt(2.0 * np.log(N_STEPS))
    )
    print(
        f"    max |estimate − p_↑| = {max_error:.3e}  "
        f"(extreme-value band σ·√(2 log N) = {expected_max:.3e})"
    )

    save_trajectory(trajectory, OUTPUT_DIR, overwrite=True)

    np.savez(
        OUTPUT_DIR / "measurement.npz",
        probability=p_up,
        counts=counts,
        estimate=estimate,
        wald_sigma=wald_sigma,
    )

    demo_report = {
        "scenario": "binomial_readout_demo",
        "purpose": (
            "aggregated-shot counterpart to the Bernoulli demo — "
            "ideal p_↑ vs. finite-shot Binomial estimator."
        ),
        "workplan_reference": "WORKPLAN_v0.3.md §5 Phase 1 measurement layer (Dispatch J)",
        "convention_references": [
            "§3 Spin basis (p_↑ = (1 + ⟨σ_z⟩)/2)",
            "§17 Measurement layer (staged) — §17.7 shape classes",
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
        estimate - wald_sigma,
        estimate + wald_sigma,
        color="#9467bd",
        alpha=0.25,
        label=r"estimate $\pm$ Wald $1\sigma$",
    )
    ax.plot(
        times_us,
        estimate,
        color="#9467bd",
        linewidth=0.0,
        marker=".",
        markersize=3,
        label=f"estimate (counts / {SHOTS})",
    )
    ax.set_xlabel(r"time (μs)")
    ax.set_ylabel(r"$p_\uparrow$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"Binomial readout of carrier Rabi — Ω/2π = {RABI_OVER_2PI_MHZ} MHz, shots = {SHOTS}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
