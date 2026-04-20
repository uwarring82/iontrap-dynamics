# SPDX-License-Identifier: MIT
"""Spin-readout protocol demo — projective shot model end-to-end.

Runs :meth:`SpinReadout.run` against the carrier Rabi trajectory with
the same detector parameters as ``run_demo_detected_readout.py``, so
the two demos can be read side-by-side. The key difference is the
sampling model:

- ``run_demo_detected_readout.py`` (Dispatch L) uses the *rate-averaged*
  pipeline (``PoissonChannel`` applied to ``λ_dark + (λ_bright −
  λ_dark) · p_↑``) and its infinite-shots envelope is
  ``P(Poisson(λ_det(t)) ≥ N̂)`` — non-linear in ``p_↑``.
- ``SpinReadout`` (this demo, Dispatch M) uses the *projective-shot*
  model (each shot projects to bright/dark, then Poisson at the
  state-conditional rate) and its infinite-shots envelope is
  ``TP · p_↑ + (1 − TN) · (1 − p_↑)`` — linear in ``p_↑``.

The projective model is the correct one for real ion-trap readout —
see CONVENTIONS.md §17.9 for the rationale. This demo visualises
both envelopes alongside the ideal ``p_↑`` to show how dramatically
the two sampling models can differ at finite detector fidelity.

Usage::

    python tools/run_demo_spin_readout.py

Requires matplotlib + scipy. Falls back to "data only, no plot" if
matplotlib is absent.

Output::

    benchmarks/data/spin_readout_demo/
      manifest.json     — canonical trajectory manifest
      arrays.npz        — times_s + ideal expectations
      measurement.npz   — p_↑, projective + rate-averaged envelopes,
                          per-shot counts, bits, shot-averaged estimate
      demo_report.json  — parameters, detector, analytic fidelities, env
      plot.png          — ideal p_↑, projective envelope, rate-averaged
                          envelope, shot-averaged estimate
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
    SpinReadout,
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
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "spin_readout_demo"

N_FOCK = 3
N_STEPS = 200
SHOTS = 500
SEED = 20260420

RABI_OVER_2PI_MHZ = 1.0
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

    detector = DetectorConfig(
        efficiency=EFFICIENCY,
        dark_count_rate=DARK_COUNT_RATE,
        threshold=THRESHOLD,
        label="pmt_main",
    )
    protocol = SpinReadout(
        ion_index=0,
        detector=detector,
        lambda_bright=LAMBDA_BRIGHT_EMIT,
        lambda_dark=LAMBDA_DARK_EMIT,
        label="ion_0_readout",
    )
    analytic = detector.classification_fidelity(
        lambda_bright=LAMBDA_BRIGHT_EMIT,
        lambda_dark=LAMBDA_DARK_EMIT,
    )

    print(">>> running spin-readout protocol demo (projective-shot model)")
    print(f"    shots = {SHOTS}, seed = {SEED}")
    print(f"    detector: η = {EFFICIENCY}, γ_d = {DARK_COUNT_RATE}, N̂ = {THRESHOLD}")
    print(
        f"    fidelity: TP = {analytic['true_positive_rate']:.4f}, "
        f"TN = {analytic['true_negative_rate']:.4f}, "
        f"F = {analytic['fidelity']:.4f}"
    )

    parameters = {
        "scenario": "spin_readout_demo",
        "N_fock": N_FOCK,
        "n_steps": N_STEPS,
        "shots": SHOTS,
        "seed": SEED,
        "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
        "phase_rad": 0.0,
        "lambda_bright_emit": LAMBDA_BRIGHT_EMIT,
        "lambda_dark_emit": LAMBDA_DARK_EMIT,
        "efficiency": EFFICIENCY,
        "dark_count_rate": DARK_COUNT_RATE,
        "threshold": THRESHOLD,
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
        provenance_tags=("demo", "spin_readout"),
    )
    elapsed_traj = time.perf_counter() - t0
    print(f"    trajectory elapsed: {elapsed_traj:.3f} s")

    t1 = time.perf_counter()
    result = protocol.run(trajectory, shots=SHOTS, seed=SEED, provenance_tags=("carrier_rabi",))
    elapsed_readout = time.perf_counter() - t1
    print(f"    readout elapsed:    {elapsed_readout:.3f} s")

    p_up = result.ideal_outcome["p_up"]
    projective_envelope = result.ideal_outcome["bright_fraction_envelope"]
    bright_fraction = result.sampled_outcome["ion_0_readout_bright_fraction"]

    # Rate-averaged envelope (Dispatch L model) — shown only for contrast.
    from scipy.stats import poisson as _poisson

    emitted_rate = LAMBDA_BRIGHT_EMIT * p_up + LAMBDA_DARK_EMIT * (1.0 - p_up)
    detected_rate = detector.apply(emitted_rate)
    rate_averaged_envelope = 1.0 - _poisson.cdf(THRESHOLD - 1, mu=detected_rate)

    shot_noise_std = np.sqrt(projective_envelope * (1.0 - projective_envelope) / SHOTS)
    max_error_vs_envelope = float(np.max(np.abs(bright_fraction - projective_envelope)))
    expected_max = float(shot_noise_std.max() * np.sqrt(2.0 * np.log(N_STEPS)))
    print(
        f"    max |bright_fraction − projective envelope| = "
        f"{max_error_vs_envelope:.3e}  "
        f"(extreme-value band σ·√(2 log N) = {expected_max:.3e})"
    )
    print(
        f"    max |projective − rate-averaged envelope|   = "
        f"{float(np.max(np.abs(projective_envelope - rate_averaged_envelope))):.3e}  "
        "(model divergence at fixed detector parameters)"
    )

    save_trajectory(trajectory, OUTPUT_DIR, overwrite=True)

    np.savez(
        OUTPUT_DIR / "measurement.npz",
        p_up=p_up,
        projective_envelope=projective_envelope,
        rate_averaged_envelope=rate_averaged_envelope,
        counts=result.sampled_outcome["ion_0_readout_counts"],
        bits=result.sampled_outcome["ion_0_readout_bits"],
        bright_fraction=bright_fraction,
        shot_noise_std=shot_noise_std,
    )

    demo_report = {
        "scenario": "spin_readout_demo",
        "purpose": (
            "first protocol-layer dispatch — shows the projective-shot "
            "readout envelope and contrasts it against the rate-averaged "
            "Dispatch L envelope at identical detector parameters."
        ),
        "workplan_reference": ("WORKPLAN_v0.3.md §5 Phase 1 measurement layer (Dispatch M)"),
        "convention_references": [
            "§3 Spin basis (p_↑ = (1 + ⟨σ_z⟩)/2)",
            "§17.8 Detector response",
            "§17.9 Projective-shot readout",
        ],
        "trajectory_elapsed_seconds": elapsed_traj,
        "readout_elapsed_seconds": elapsed_readout,
        "max_bright_fraction_vs_envelope": max_error_vs_envelope,
        "max_projective_vs_rate_averaged": float(
            np.max(np.abs(projective_envelope - rate_averaged_envelope))
        ),
        "extreme_value_band": expected_max,
        "analytic_detector_fidelity": analytic,
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
        p_up,
        color="black",
        linewidth=1.5,
        label=r"ideal $p_\uparrow(t)$",
    )
    ax.plot(
        times_us,
        projective_envelope,
        color="#1f77b4",
        linewidth=1.2,
        label=(
            "projective envelope "
            r"($\mathrm{TP}\cdot p_\uparrow + (1-\mathrm{TN})(1-p_\uparrow)$, "
            f"$F={analytic['fidelity']:.3f}$)"
        ),
    )
    ax.plot(
        times_us,
        rate_averaged_envelope,
        color="#2ca02c",
        linewidth=1.0,
        linestyle=":",
        label=(
            "rate-averaged envelope "
            r"($P[n \geq \hat N | \lambda_\mathrm{det}]$, Dispatch L model)"
        ),
    )
    ax.plot(
        times_us,
        bright_fraction,
        color="#d62728",
        linewidth=0.0,
        marker=".",
        markersize=3,
        label=f"bright fraction ({SHOTS} shots)",
    )
    ax.set_xlabel(r"time (μs)")
    ax.set_ylabel("probability / bright fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Projective spin readout of carrier Rabi — "
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
