# SPDX-License-Identifier: MIT
"""Detected-readout demo — Poisson channel composed with DetectorConfig.

Runs the same carrier Rabi trajectory as the Bernoulli / Binomial /
Poisson demos, but now exercises the full *apply → Poisson →
discriminate* pipeline:

    λ_emit(t) = λ_bright_0 · p_↑(t) + λ_dark_0 · p_↓(t)   (emitted)
    λ_det(t)  = η · λ_emit(t) + γ_d                        (detector.apply)
    counts    = PoissonChannel.sample(λ_det, shots=N)       (per-shot)
    bits      = detector.discriminate(counts)                (threshold)

with a non-ideal detector (``η = 0.4``, ``γ_d = 0.3``) and a
threshold tuned to the effective bright / dark rates. This is the
*rate-averaged* readout model — valid when the qubit emits at a
coherent state-dependent rate during the detection window. Its
infinite-shots limit is

    envelope(t) = P(Poisson(λ_det(t)) ≥ N̂)
                = 1 − CDF(N̂ − 1; λ_det(t))

which, as a function of ``p_↑``, is markedly *non-linear* — quite
different from what a projective-shot model would give. The plot
shows:

1. ``⟨bits⟩`` across shots — the physical "bright-fraction" estimate
   extractable from a real experiment.
2. The ideal ``p_↑(t)`` curve — ground truth from the dynamics.
3. The Poisson-tail envelope above — what the estimator converges to
   at infinite shots for this detector.

The gap between (1) and (3) is shot noise; the gap between (2) and
(3) is detector-induced distortion (threshold, efficiency, dark-
count bias) that protocol-layer code in Dispatches M–N will need to
invert. The classification fidelity ``F`` from
:meth:`DetectorConfig.classification_fidelity` is reported as an
auxiliary quality figure for the projective-shot model; it does not
directly give the envelope for a general ``p_↑`` trajectory.

Usage::

    python tools/run_demo_detected_readout.py

Requires matplotlib + scipy. Falls back to "data only, no plot" if
matplotlib is absent.

Output::

    benchmarks/data/detected_readout_demo/
      manifest.json     — canonical trajectory manifest
      arrays.npz        — times_s + ideal expectations
      measurement.npz   — emitted & detected rate, counts,
                          shot-averaged bright fraction,
                          Poisson-tail envelope + 1σ shot-noise std
      demo_report.json  — parameters, detector, analytic fidelities, env
      plot.png          — ideal p_↑, Poisson-tail envelope,
                          shot-averaged estimate
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
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "detected_readout_demo"

N_FOCK = 3
N_STEPS = 200
SHOTS = 500
SEED = 20260420

RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

# Emitted rates (counts per shot) — the physical scattering rates at
# the ion in a matched detection window, *before* imperfect collection.
LAMBDA_BRIGHT_EMIT = 25.0
LAMBDA_DARK_EMIT = 0.0

# Detector — deliberately non-ideal so the fidelity envelope is
# visible to the eye. η = 0.4, γ_d = 0.3, threshold tuned to balance.
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
    analytic = detector.classification_fidelity(
        lambda_bright=LAMBDA_BRIGHT_EMIT,
        lambda_dark=LAMBDA_DARK_EMIT,
    )

    print(">>> running detected-readout demo (Poisson + DetectorConfig)")
    print(f"    shots = {SHOTS}, seed = {SEED}")
    print(f"    emitted: λ_bright = {LAMBDA_BRIGHT_EMIT}, λ_dark = {LAMBDA_DARK_EMIT} counts/shot")
    print(f"    detector: η = {EFFICIENCY}, γ_d = {DARK_COUNT_RATE}, threshold = {THRESHOLD}")
    print(
        f"    effective: λ_bright_eff = {analytic['effective_bright_rate']:.3f}, "
        f"λ_dark_eff = {analytic['effective_dark_rate']:.3f}"
    )
    print(
        f"    analytic fidelity: TP = {analytic['true_positive_rate']:.4f}, "
        f"TN = {analytic['true_negative_rate']:.4f}, "
        f"F = {analytic['fidelity']:.4f}"
    )

    parameters = {
        "scenario": "detected_readout_demo",
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
        provenance_tags=("demo", "detected_readout"),
    )
    elapsed = time.perf_counter() - t0
    print(f"    trajectory elapsed: {elapsed:.3f} s")

    sigma_z = trajectory.expectations["sigma_z_0"]
    p_up = (1.0 + sigma_z) / 2.0
    emitted_rate = LAMBDA_BRIGHT_EMIT * p_up + LAMBDA_DARK_EMIT * (1.0 - p_up)
    detected_rate = detector.apply(emitted_rate)

    measurement = sample_outcome(
        channel=PoissonChannel(label="photon_counts"),
        inputs=detected_rate,
        shots=SHOTS,
        seed=SEED,
        upstream=trajectory,
        provenance_tags=("carrier_rabi",),
    )
    counts = measurement.sampled_outcome["photon_counts"]
    bits = detector.discriminate(counts)
    bright_fraction = bits.mean(axis=0)

    # Infinite-shots envelope for the rate-averaged Poisson pipeline.
    # Each shot's bit is Bernoulli with probability
    # P(count ≥ threshold | detected_rate) = 1 − Poisson-CDF(N̂−1; λ_det).
    from scipy.stats import poisson as _poisson

    envelope = 1.0 - _poisson.cdf(THRESHOLD - 1, mu=detected_rate)
    shot_noise_std = np.sqrt(envelope * (1.0 - envelope) / SHOTS)
    max_error_vs_envelope = float(np.max(np.abs(bright_fraction - envelope)))
    expected_max = float(shot_noise_std.max() * np.sqrt(2.0 * np.log(N_STEPS)))
    print(
        f"    max |bright_fraction − Poisson-tail envelope| = "
        f"{max_error_vs_envelope:.3e}  "
        f"(extreme-value band σ·√(2 log N) = {expected_max:.3e})"
    )

    save_trajectory(trajectory, OUTPUT_DIR, overwrite=True)

    np.savez(
        OUTPUT_DIR / "measurement.npz",
        emitted_rate=emitted_rate,
        detected_rate=detected_rate,
        counts=counts,
        bright_fraction=bright_fraction,
        poisson_tail_envelope=envelope,
        shot_noise_std=shot_noise_std,
    )

    demo_report = {
        "scenario": "detected_readout_demo",
        "purpose": (
            "first composition of detector + Poisson channel — shows the "
            "Poisson-tail envelope that separates ideal p_↑ from what a "
            "finite-efficiency, thresholded detector actually measures."
        ),
        "workplan_reference": ("WORKPLAN_v0.3.md §5 Phase 1 measurement layer (Dispatch L)"),
        "convention_references": [
            "§3 Spin basis (p_↑ = (1 + ⟨σ_z⟩)/2)",
            "§17 Measurement layer (staged) — §17.8 detector response",
        ],
        "elapsed_seconds": elapsed,
        "max_bright_fraction_vs_envelope": max_error_vs_envelope,
        "extreme_value_band": expected_max,
        "parameters": {
            **parameters,
            "duration_us": float(tlist[-1] * 1e6),
        },
        "analytic_detector_fidelity": analytic,
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
    ax.plot(
        times_us,
        p_up,
        color="black",
        linewidth=1.5,
        label=r"ideal $p_\uparrow(t)$",
    )
    ax.plot(
        times_us,
        envelope,
        color="#2ca02c",
        linewidth=1.2,
        linestyle="--",
        label=(
            "Poisson-tail envelope "
            r"($P[n \geq \hat N]$; "
            f"projective-$F={analytic['fidelity']:.3f}$)"
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
        "Detected readout of carrier Rabi — "
        f"$\\eta={EFFICIENCY}$, $\\gamma_d={DARK_COUNT_RATE}$, "
        f"$\\hat N={THRESHOLD}$, shots={SHOTS}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
