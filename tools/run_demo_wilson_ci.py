# SPDX-License-Identifier: MIT
"""Wilson-CI demo — confidence intervals on carrier-Rabi SpinReadout.

Capstone tool for the Dispatch P statistics surface. Runs the standard
carrier-Rabi :class:`SpinReadout` pipeline at a deliberately modest
shot budget, then draws Wilson and Clopper–Pearson 95 % confidence
intervals on the bright-fraction estimator at every time step. The
plot overlays four curves:

1. Ideal ``p_↑(t)`` from the dynamics — ground truth.
2. The fidelity-limited envelope
   ``TP · p_↑ + (1 − TN) · (1 − p_↑)`` — what the estimator converges
   to at infinite shots.
3. The shot-averaged ``bright_fraction`` point estimate — the
   finite-shot sample.
4. Wilson 95 % CI band — shaded region the true envelope should fall
   inside ~95 % of the time.

The empirical coverage over the full time series is computed and
reported against the nominal 95 % — a single-run realisation, so
sub-nominal coverage on one run is statistically normal; the
expected-coverage interpretation is over many replications at
different seeds.

Usage::

    python tools/run_demo_wilson_ci.py

Requires matplotlib + scipy. Falls back to "data only, no plot" if
matplotlib is absent.

Output::

    benchmarks/data/wilson_ci_demo/
      manifest.json     — canonical trajectory manifest
      arrays.npz        — times_s + ideal expectations
      measurement.npz   — point estimate, Wilson CI, C-P CI,
                          shot counts, envelope
      demo_report.json  — parameters, detector, coverage stats, env
      plot.png          — envelope with Wilson CI shading
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
    binomial_summary,
    clopper_pearson_interval,
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
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "wilson_ci_demo"

N_FOCK = 3
N_STEPS = 100
SHOTS = 80  # Deliberately small so CIs are visually wide.
SEED = 20260420
CONFIDENCE = 0.95

RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

LAMBDA_BRIGHT_EMIT = 20.0
LAMBDA_DARK_EMIT = 0.0

EFFICIENCY = 0.5
DARK_COUNT_RATE = 0.3
THRESHOLD = 3

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
        label="rabi_readout",
    )
    analytic = detector.classification_fidelity(
        lambda_bright=LAMBDA_BRIGHT_EMIT,
        lambda_dark=LAMBDA_DARK_EMIT,
    )

    print(">>> running Wilson-CI demo (carrier Rabi with finite-shot error bars)")
    print(f"    shots = {SHOTS}, seed = {SEED}, confidence = {CONFIDENCE}")
    print(
        f"    detector: η = {EFFICIENCY}, γ_d = {DARK_COUNT_RATE}, "
        f"N̂ = {THRESHOLD}, F = {analytic['fidelity']:.4f}"
    )

    parameters = {
        "scenario": "wilson_ci_demo",
        "N_fock": N_FOCK,
        "n_steps": N_STEPS,
        "shots": SHOTS,
        "seed": SEED,
        "confidence": CONFIDENCE,
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
        provenance_tags=("demo", "wilson_ci"),
    )
    elapsed_traj = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = protocol.run(trajectory, shots=SHOTS, seed=SEED, provenance_tags=("carrier_rabi",))
    elapsed_readout = time.perf_counter() - t1
    print(f"    trajectory elapsed: {elapsed_traj:.3f} s")
    print(f"    readout elapsed:    {elapsed_readout:.3f} s")

    # Per-time-bin Wilson and Clopper–Pearson CIs on the bit counts.
    bits = result.sampled_outcome["rabi_readout_bits"]
    successes = bits.sum(axis=0)  # (n_times,) int
    wilson = binomial_summary(successes, SHOTS, confidence=CONFIDENCE, method="wilson")
    cp_lower, cp_upper = clopper_pearson_interval(successes, SHOTS, confidence=CONFIDENCE)

    p_up = result.ideal_outcome["p_up"]
    envelope = result.ideal_outcome["bright_fraction_envelope"]
    point = wilson.point_estimate
    lo_w = wilson.lower
    hi_w = wilson.upper

    # Empirical coverage at this seed — the 95 % CI should contain the
    # envelope at ~95 % of time bins over many replications.
    covered = (lo_w <= envelope) & (envelope <= hi_w)
    coverage = float(covered.mean())
    mean_halfwidth = float((hi_w - lo_w).mean() / 2.0)
    print(
        f"    Wilson coverage (single seed) = {coverage:.3f}  "
        f"(nominal {CONFIDENCE:.2f}; many-seed expectation)"
    )
    print(f"    mean half-width = {mean_halfwidth:.3f}")

    save_trajectory(trajectory, OUTPUT_DIR, overwrite=True)

    np.savez(
        OUTPUT_DIR / "measurement.npz",
        p_up=p_up,
        envelope=envelope,
        successes=successes,
        point_estimate=point,
        wilson_lower=lo_w,
        wilson_upper=hi_w,
        cp_lower=cp_lower,
        cp_upper=cp_upper,
    )

    demo_report = {
        "scenario": "wilson_ci_demo",
        "purpose": (
            "capstone demo for the measurement track — Wilson and "
            "Clopper–Pearson CIs on the bright-fraction estimator "
            "over a carrier Rabi trajectory at a modest shot budget."
        ),
        "workplan_reference": ("WORKPLAN_v0.3.md §5 Phase 1 measurement layer (Dispatch P)"),
        "convention_references": [
            "§3 Spin basis",
            "§17.8 Detector response",
            "§17.9 Projective-shot readout",
            "§17.12 Binomial confidence intervals",
        ],
        "trajectory_elapsed_seconds": elapsed_traj,
        "readout_elapsed_seconds": elapsed_readout,
        "wilson_single_seed_coverage": coverage,
        "wilson_mean_halfwidth": mean_halfwidth,
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
        envelope,
        color="#1f77b4",
        linewidth=1.0,
        linestyle="--",
        label=(r"projective envelope $(F=" f"{analytic['fidelity']:.3f})$"),
    )
    ax.fill_between(
        times_us,
        lo_w,
        hi_w,
        color="#d62728",
        alpha=0.25,
        label=f"Wilson {int(CONFIDENCE * 100)} % CI",
    )
    ax.plot(
        times_us,
        point,
        color="#d62728",
        linewidth=0.0,
        marker=".",
        markersize=4,
        label=f"point estimate (k / {SHOTS})",
    )
    ax.set_xlabel(r"time (μs)")
    ax.set_ylabel(r"$p_\uparrow$ / bright fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"Wilson CI on carrier-Rabi SpinReadout — shots={SHOTS}, coverage (seed)={coverage:.2f}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
