# SPDX-License-Identifier: MIT
"""Sideband-inference protocol demo — motional thermometry on |↓, 0⟩.

Runs the red- and blue-sideband Rabi Hamiltonians on the ground state
``|↓, 0⟩`` for a short-time scan, then pipes both trajectories into
:meth:`SidebandInference.run` to recover the mean motional occupation
``n̄``. For the ground state ``|n=0⟩``:

- The red sideband can't remove a phonon — the RSB trajectory stays
  pinned at ``|↓, 0⟩`` and ``P↑_RSB(t) = 0`` for all ``t``.
- The blue sideband drives the standard single-Fock Rabi flop at
  rate ``2Ωη`` and ``P↑_BSB(t) > 0`` once ``t`` leaves ``0``.

The ratio ``P↑_RSB / P↑_BSB → 0`` in the short-time limit, so
``n̄ = r / (1 − r) → 0`` — inference correctly reports a cold ion.
The plot overlays four curves:

1. Ideal ``P↑_RSB(t)`` and ``P↑_BSB(t)`` from the dynamics.
2. Fidelity-corrected bright-fraction estimates from finite shots.
3. Inferred ``n̄(t)`` — expected ≈ 0 within shot noise.

Usage::

    python tools/run_demo_sideband_inference.py

Requires matplotlib + scipy. Falls back to "data only, no plot" if
matplotlib is absent.

Output::

    benchmarks/data/sideband_inference_demo/
      manifest.json     — canonical RSB trajectory manifest
      arrays.npz        — RSB times_s + ⟨σ_z⟩
      bsb_arrays.npz    — BSB times_s + ⟨σ_z⟩
      measurement.npz   — per-sideband bright fractions, fidelity-
                          corrected probabilities, n̄ trajectory
      demo_report.json  — parameters, detector, fidelities, environment
      plot.png          — RSB/BSB dynamics + inferred n̄ trajectory
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
    SidebandInference,
)
from iontrap_dynamics.cache import compute_request_hash, save_trajectory
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import (
    blue_sideband_hamiltonian,
    red_sideband_hamiltonian,
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
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "sideband_inference_demo"

N_FOCK = 5
N_STEPS = 200
SHOTS = 500
SEED = 20260420

RABI_OVER_2PI_MHZ = 0.1
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

LAMBDA_BRIGHT_EMIT = 12.0
LAMBDA_DARK_EMIT = 0.0

# Detector tuned so that rare dark-count false positives make the RSB
# bright fraction non-zero, exercising shot noise on both sidebands.
EFFICIENCY = 0.6
DARK_COUNT_RATE = 0.4
THRESHOLD = 2

RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)


def _build_scenario() -> tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, qutip.Qobj]:
    """Return (hilbert, H_rsb, H_bsb, psi_0) — same ion / mode / drive."""
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
    h_rsb = red_sideband_hamiltonian(hilbert, drive, "axial", ion_index=0)
    h_bsb = blue_sideband_hamiltonian(hilbert, drive, "axial", ion_index=0)
    psi_0 = qutip.tensor(spin_down(), qutip.basis(N_FOCK, 0))
    return hilbert, h_rsb, h_bsb, psi_0


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

    hilbert, h_rsb, h_bsb, psi_0 = _build_scenario()

    # Short-time scan — keep (2Ωη t)² ≪ 1 for the ratio formula to apply.
    # Ωη ~ 0.1 MHz · 0.18 ~ 20 kHz → pick duration ~ 10 μs.
    tlist = np.linspace(0.0, 10.0e-6, N_STEPS)

    detector = DetectorConfig(
        efficiency=EFFICIENCY,
        dark_count_rate=DARK_COUNT_RATE,
        threshold=THRESHOLD,
        label="pmt_main",
    )
    protocol = SidebandInference(
        ion_index=0,
        detector=detector,
        lambda_bright=LAMBDA_BRIGHT_EMIT,
        lambda_dark=LAMBDA_DARK_EMIT,
        label="ground_state_thermometry",
    )
    analytic = detector.classification_fidelity(
        lambda_bright=LAMBDA_BRIGHT_EMIT,
        lambda_dark=LAMBDA_DARK_EMIT,
    )

    print(">>> running sideband-inference protocol demo (ground-state thermometry)")
    print(f"    shots = {SHOTS}, seed = {SEED}")
    print(f"    detector: η = {EFFICIENCY}, γ_d = {DARK_COUNT_RATE}, N̂ = {THRESHOLD}")
    print(f"    fidelity: F = {analytic['fidelity']:.4f}")

    rsb_parameters = {
        "scenario": "sideband_inference_demo",
        "sideband": "red",
        "N_fock": N_FOCK,
        "n_steps": N_STEPS,
        "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
        "phase_rad": 0.0,
        "initial_state": "|↓, 0⟩",
    }
    bsb_parameters = {**rsb_parameters, "sideband": "blue"}
    rsb_hash = compute_request_hash(rsb_parameters)
    bsb_hash = compute_request_hash(bsb_parameters)

    t0 = time.perf_counter()
    rsb_trajectory = solve(
        hilbert=hilbert,
        hamiltonian=h_rsb,
        initial_state=psi_0,
        times=tlist,
        observables=[spin_z(hilbert, 0)],
        request_hash=rsb_hash,
        storage_mode=StorageMode.OMITTED,
        provenance_tags=("demo", "sideband_inference", "rsb"),
    )
    bsb_trajectory = solve(
        hilbert=hilbert,
        hamiltonian=h_bsb,
        initial_state=psi_0,
        times=tlist,
        observables=[spin_z(hilbert, 0)],
        request_hash=bsb_hash,
        storage_mode=StorageMode.OMITTED,
        provenance_tags=("demo", "sideband_inference", "bsb"),
    )
    elapsed_traj = time.perf_counter() - t0
    print(f"    trajectories elapsed: {elapsed_traj:.3f} s")

    t1 = time.perf_counter()
    result = protocol.run(
        rsb_trajectory=rsb_trajectory,
        bsb_trajectory=bsb_trajectory,
        shots=SHOTS,
        seed=SEED,
        provenance_tags=("ground_state",),
    )
    elapsed_inference = time.perf_counter() - t1
    print(f"    inference elapsed:    {elapsed_inference:.3f} s")

    p_up_rsb = result.ideal_outcome["p_up_rsb"]
    p_up_bsb = result.ideal_outcome["p_up_bsb"]
    nbar_ideal = result.ideal_outcome["nbar_from_ideal_ratio"]
    nbar_estimate = result.sampled_outcome["ground_state_thermometry_nbar_estimate"]
    p_up_rsb_hat = result.sampled_outcome["ground_state_thermometry_p_up_rsb_hat"]
    p_up_bsb_hat = result.sampled_outcome["ground_state_thermometry_p_up_bsb_hat"]

    # Restrict summary stats to the reliable region — skip t=0 and
    # anywhere p_up_bsb is below a few % (ratio dominated by shot noise).
    valid = p_up_bsb > 0.05
    nbar_valid = nbar_estimate[valid]
    nbar_median = float(np.nanmedian(nbar_valid))
    nbar_p95 = float(np.nanpercentile(np.abs(nbar_valid), 95))
    print(f"    median inferred n̄ (reliable region) = {nbar_median:.3e}  (truth = 0)")
    print(f"    95th-percentile |n̄| (reliable region) = {nbar_p95:.3e}  (shot-noise spread)")

    save_trajectory(rsb_trajectory, OUTPUT_DIR, overwrite=True)
    np.savez(
        OUTPUT_DIR / "bsb_arrays.npz",
        times_s=tlist,
        sigma_z_0=bsb_trajectory.expectations["sigma_z_0"],
    )

    np.savez(
        OUTPUT_DIR / "measurement.npz",
        p_up_rsb=p_up_rsb,
        p_up_bsb=p_up_bsb,
        p_up_rsb_hat=p_up_rsb_hat,
        p_up_bsb_hat=p_up_bsb_hat,
        nbar_ideal=nbar_ideal,
        nbar_estimate=nbar_estimate,
        nbar_from_raw_ratio=result.sampled_outcome["ground_state_thermometry_nbar_from_raw_ratio"],
    )

    demo_report = {
        "scenario": "sideband_inference_demo",
        "purpose": (
            "end-to-end motional thermometry on |↓, 0⟩ — RSB is dark, "
            "BSB flops; the ratio method correctly reports n̄ ≈ 0."
        ),
        "workplan_reference": ("WORKPLAN_v0.3.md §5 Phase 1 measurement layer (Dispatch O)"),
        "convention_references": [
            "§3 Spin basis",
            "§17.8 Detector response",
            "§17.9 Projective-shot readout",
            "§17.11 Sideband inference",
        ],
        "trajectories_elapsed_seconds": elapsed_traj,
        "inference_elapsed_seconds": elapsed_inference,
        "median_inferred_nbar": nbar_median,
        "p95_abs_inferred_nbar": nbar_p95,
        "analytic_detector_fidelity": analytic,
        "parameters": {
            "rsb": {**rsb_parameters, "duration_us": float(tlist[-1] * 1e6)},
            "bsb": {**bsb_parameters, "duration_us": float(tlist[-1] * 1e6)},
            "shots": SHOTS,
            "seed": SEED,
            "efficiency": EFFICIENCY,
            "dark_count_rate": DARK_COUNT_RATE,
            "threshold": THRESHOLD,
            "lambda_bright_emit": LAMBDA_BRIGHT_EMIT,
            "lambda_dark_emit": LAMBDA_DARK_EMIT,
        },
        "rsb_trajectory_hash": result.trajectory_hash,
        "bsb_trajectory_hash": str(result.ideal_outcome["bsb_trajectory_hash"]),
        "measurement_result_provenance_tags": list(result.metadata.provenance_tags),
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
        "{manifest.json, arrays.npz, bsb_arrays.npz, measurement.npz, demo_report.json}"
    )

    times_us = tlist * 1e6

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not installed; skipping plot")
        return 0

    fig, (ax_p, ax_n) = plt.subplots(2, 1, sharex=True, figsize=(8.0, 6.0))

    ax_p.plot(
        times_us, p_up_rsb, color="#1f77b4", linewidth=1.5, label=r"ideal $P_\uparrow^\mathrm{RSB}$"
    )
    ax_p.plot(
        times_us, p_up_bsb, color="#d62728", linewidth=1.5, label=r"ideal $P_\uparrow^\mathrm{BSB}$"
    )
    ax_p.plot(
        times_us,
        p_up_rsb_hat,
        color="#1f77b4",
        linewidth=0.0,
        marker=".",
        markersize=3,
        alpha=0.7,
        label=r"corrected $\hat p_\uparrow^\mathrm{RSB}$",
    )
    ax_p.plot(
        times_us,
        p_up_bsb_hat,
        color="#d62728",
        linewidth=0.0,
        marker=".",
        markersize=3,
        alpha=0.7,
        label=r"corrected $\hat p_\uparrow^\mathrm{BSB}$",
    )
    ax_p.set_ylabel(r"$P_\uparrow$")
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.legend(loc="upper left", fontsize=7)
    ax_p.set_title(
        f"Ground-state sideband thermometry — $F={analytic['fidelity']:.3f}$, shots={SHOTS}"
    )

    ax_n.plot(
        times_us[valid],
        nbar_estimate[valid],
        color="#2ca02c",
        linewidth=0.0,
        marker=".",
        markersize=3,
        label=r"$\hat n$",
    )
    ax_n.axhline(0.0, color="grey", linewidth=0.3)
    ax_n.set_xlabel(r"time (μs)")
    ax_n.set_ylabel(r"$\hat n$")
    ax_n.set_ylim(-0.3, 0.3)
    ax_n.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
