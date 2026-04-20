# SPDX-License-Identifier: MIT
"""Parity-scan protocol demo — Bell-state formation through the MS gate.

Runs the gate-closing Mølmer–Sørensen Hamiltonian on a two-ion system
and reads out joint parity at every step via :class:`ParityScan`. The
MS gate's textbook trajectory takes ``|↓↓, 0⟩`` through intermediate
entangled states and back to a Bell state ``(|↓↓⟩ − i|↑↑⟩)/√2`` at the
closing time ``t_gate``; the parity observable ``⟨σ_z^{(0)} σ_z^{(1)}⟩``
oscillates between ``+1`` (fully correlated, ``|↓↓⟩`` or Bell endpoint)
and nonzero intermediate values characteristic of the two-loop motional
excursion.

The demo contrasts three curves:

1. Ideal ``⟨σ_z^{(0)} σ_z^{(1)}⟩`` from the dynamics — ground truth.
2. The projective-shot envelope under a finite-fidelity detector
   (``(TP + TN − 1)² · ⟨σ_z σ_z⟩ + (TP − TN)²`` at zero marginals,
   more generally the full 4-term confusion-weighted sum — §17.10).
3. Shot-averaged parity estimate from :meth:`ParityScan.run` — the
   finite-shot Monte-Carlo reconstruction of (2).

Usage::

    python tools/run_demo_parity_scan.py

Requires matplotlib + scipy. Falls back to "data only, no plot" if
matplotlib is absent.

Output::

    benchmarks/data/parity_scan_demo/
      manifest.json     — canonical trajectory manifest
      arrays.npz        — times_s + ideal expectations (including parity)
      measurement.npz   — joint probabilities, parity envelope,
                          per-ion counts + bits, per-shot parity,
                          shot-averaged parity estimate
      demo_report.json  — parameters, detector, fidelities, environment
      plot.png          — ideal parity, projective envelope, estimate
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
    DetectorConfig,
    ParityScan,
)
from iontrap_dynamics.analytic import (
    lamb_dicke_parameter,
    ms_gate_closing_detuning,
    ms_gate_closing_time,
)
from iontrap_dynamics.cache import compute_request_hash, save_trajectory
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
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "parity_scan_demo"

N_FOCK = 12
N_STEPS = 400
SHOTS = 500
SEED = 20260420
LOOPS = 1

RABI_OVER_2PI_MHZ = 0.1
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

LAMBDA_BRIGHT_EMIT = 25.0
LAMBDA_DARK_EMIT = 0.0

EFFICIENCY = 0.4
DARK_COUNT_RATE = 0.3
THRESHOLD = 4

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

    detector = DetectorConfig(
        efficiency=EFFICIENCY,
        dark_count_rate=DARK_COUNT_RATE,
        threshold=THRESHOLD,
        label="pmt_shared",
    )
    protocol = ParityScan(
        ion_indices=(0, 1),
        detector=detector,
        lambda_bright=LAMBDA_BRIGHT_EMIT,
        lambda_dark=LAMBDA_DARK_EMIT,
        label="bell_parity",
    )
    analytic = detector.classification_fidelity(
        lambda_bright=LAMBDA_BRIGHT_EMIT,
        lambda_dark=LAMBDA_DARK_EMIT,
    )

    print(">>> running parity-scan protocol demo (MS-gate Bell-state formation)")
    print(f"    shots = {SHOTS}, seed = {SEED}, LD parameter η = {eta:.4f}")
    print(f"    t_gate = {t_gate * 1e6:.3f} μs  (closing condition)")
    print(f"    detector: η_det = {EFFICIENCY}, γ_d = {DARK_COUNT_RATE}, N̂ = {THRESHOLD}")
    print(
        f"    fidelity: TP = {analytic['true_positive_rate']:.4f}, "
        f"TN = {analytic['true_negative_rate']:.4f}, "
        f"F = {analytic['fidelity']:.4f}"
    )

    parameters = {
        "scenario": "parity_scan_demo",
        "N_fock": N_FOCK,
        "n_steps": N_STEPS,
        "shots": SHOTS,
        "seed": SEED,
        "loops": LOOPS,
        "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
        "phase_rad": 0.0,
        "lambda_bright_emit": LAMBDA_BRIGHT_EMIT,
        "lambda_dark_emit": LAMBDA_DARK_EMIT,
        "efficiency": EFFICIENCY,
        "dark_count_rate": DARK_COUNT_RATE,
        "threshold": THRESHOLD,
        "initial_state": "|↓↓, 0⟩",
    }
    request_hash = compute_request_hash(parameters)

    t0 = time.perf_counter()
    trajectory = solve(
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
        storage_mode=StorageMode.OMITTED,
        provenance_tags=("demo", "parity_scan"),
    )
    elapsed_traj = time.perf_counter() - t0
    print(f"    trajectory elapsed: {elapsed_traj:.3f} s")

    t1 = time.perf_counter()
    result = protocol.run(trajectory, shots=SHOTS, seed=SEED, provenance_tags=("ms_gate",))
    elapsed_readout = time.perf_counter() - t1
    print(f"    readout elapsed:    {elapsed_readout:.3f} s")

    ideal_parity = result.ideal_outcome["parity"]
    projective_envelope = result.ideal_outcome["parity_envelope"]
    joint_probabilities = result.ideal_outcome["joint_probabilities"]
    parity_estimate = result.sampled_outcome["bell_parity_parity_estimate"]

    shot_noise_std = np.sqrt((1.0 - projective_envelope**2).clip(min=0.0) / SHOTS)
    max_error_vs_envelope = float(np.max(np.abs(parity_estimate - projective_envelope)))
    expected_max = float(shot_noise_std.max() * np.sqrt(2.0 * np.log(N_STEPS)))
    print(
        f"    max |parity_estimate − projective envelope| = "
        f"{max_error_vs_envelope:.3e}  "
        f"(extreme-value band σ·√(2 log N) = {expected_max:.3e})"
    )
    print(
        f"    max |projective envelope − ideal parity|    = "
        f"{float(np.max(np.abs(projective_envelope - ideal_parity))):.3e}  "
        "(detector-fidelity shrinkage)"
    )

    save_trajectory(trajectory, OUTPUT_DIR, overwrite=True)

    np.savez(
        OUTPUT_DIR / "measurement.npz",
        p_up_0=result.ideal_outcome["p_up_0"],
        p_up_1=result.ideal_outcome["p_up_1"],
        ideal_parity=ideal_parity,
        projective_envelope=projective_envelope,
        joint_probabilities=joint_probabilities,
        counts_0=result.sampled_outcome["bell_parity_counts_0"],
        counts_1=result.sampled_outcome["bell_parity_counts_1"],
        bits_0=result.sampled_outcome["bell_parity_bits_0"],
        bits_1=result.sampled_outcome["bell_parity_bits_1"],
        parity_shots=result.sampled_outcome["bell_parity_parity"],
        parity_estimate=parity_estimate,
        shot_noise_std=shot_noise_std,
    )

    demo_report = {
        "scenario": "parity_scan_demo",
        "purpose": (
            "joint two-ion readout through a Bell-state-forming MS gate — "
            "shows the projective parity envelope shrunk by the detector's "
            "(TP + TN − 1)² contrast factor."
        ),
        "workplan_reference": ("WORKPLAN_v0.3.md §5 Phase 1 measurement layer (Dispatch N)"),
        "convention_references": [
            "§3 Spin basis",
            "§17.8 Detector response",
            "§17.9 Projective-shot readout",
            "§17.10 Multi-ion joint readout",
        ],
        "trajectory_elapsed_seconds": elapsed_traj,
        "readout_elapsed_seconds": elapsed_readout,
        "max_parity_estimate_vs_envelope": max_error_vs_envelope,
        "max_envelope_vs_ideal": float(np.max(np.abs(projective_envelope - ideal_parity))),
        "extreme_value_band": expected_max,
        "analytic_detector_fidelity": analytic,
        "lamb_dicke_parameter": float(eta),
        "t_gate_us": float(t_gate * 1e6),
        "parameters": {
            **parameters,
            "duration_us": float(tlist[-1] * 1e6),
        },
        "trajectory_hash": result.trajectory_hash,
        "measurement_result_provenance_tags": list(result.metadata.provenance_tags),
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
    ax.plot(
        times_us,
        ideal_parity,
        color="black",
        linewidth=1.5,
        label=r"ideal $\langle\sigma_z^{(0)}\sigma_z^{(1)}\rangle(t)$",
    )
    ax.plot(
        times_us,
        projective_envelope,
        color="#1f77b4",
        linewidth=1.2,
        linestyle="--",
        label=(f"projective envelope ($F={analytic['fidelity']:.3f}$)"),
    )
    ax.plot(
        times_us,
        parity_estimate,
        color="#d62728",
        linewidth=0.0,
        marker=".",
        markersize=3,
        label=f"parity estimate ({SHOTS} shots)",
    )
    ax.axhline(0.0, color="grey", linewidth=0.3)
    ax.axvline(t_gate * 1e6, color="grey", linewidth=0.3, linestyle=":")
    ax.set_xlabel(r"time (μs)")
    ax.set_ylabel(r"parity $\langle\sigma_z\sigma_z\rangle$")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(
        "Parity scan through MS-gate Bell-state formation — "
        f"$\\eta={EFFICIENCY}$, $\\gamma_d={DARK_COUNT_RATE}$, "
        f"$\\hat N={THRESHOLD}$, shots={SHOTS}"
    )
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
