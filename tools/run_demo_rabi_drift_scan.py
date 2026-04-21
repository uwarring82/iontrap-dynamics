# SPDX-License-Identifier: MIT
"""Rabi-drift scan demo — π-pulse miscalibration sensitivity.

Deterministic parameter-drift companion to the stochastic jitter demos
(Dispatches R / S). Fixes the pulse duration at the nominal π-time
``t_π = π / Ω₀`` and sweeps a :class:`RabiDrift` amount from −20 % to
+20 %. At the ideal setpoint (`delta = 0`) the final ``⟨σ_z⟩`` lands at
``+1`` (complete spin flip). At non-zero drift the actual rotation
angle is ``(1 + delta) · π``, so ``⟨σ_z⟩_final`` follows a
``cos((1 + delta) · π) = −cos(delta · π)`` curve:

    delta = 0   → ⟨σ_z⟩ = +1   (exact π-flip)
    delta = ±0.5 → ⟨σ_z⟩ = 0   (π/2 away from target)
    delta = ±1   → ⟨σ_z⟩ = −1  (2π, back to start)

The plot shows the scan curve and annotates the ``1 %``-error band
around the setpoint — the tolerance window within which a calibration
must land for high-fidelity single-qubit gates.

Usage::

    python tools/run_demo_rabi_drift_scan.py

Output::

    benchmarks/data/rabi_drift_scan_demo/
      arrays.npz        — deltas + final sigma_z values
      demo_report.json  — parameters, setpoint tolerance figures
      plot.png          — scan curve with ±1% band
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

from iontrap_dynamics import RabiDrift, apply_rabi_drift
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
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "rabi_drift_scan_demo"

N_FOCK = 3
N_DELTAS = 61
DELTA_MAX = 0.20  # sweep ±20 %

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
    t_pi = np.pi / RABI_RAD_S  # nominal π-pulse duration
    tlist = np.array([0.0, t_pi])  # endpoint-only — we only need the final state

    print(">>> running Rabi-drift scan demo (π-pulse miscalibration)")
    print(f"    Ω₀ / 2π = {RABI_OVER_2PI_MHZ} MHz, t_π = {t_pi * 1e6:.3f} μs")
    print(f"    drift scan: delta ∈ [−{DELTA_MAX}, +{DELTA_MAX}], {N_DELTAS} points")

    deltas = np.linspace(-DELTA_MAX, DELTA_MAX, N_DELTAS)
    final_sigma_z = np.empty(N_DELTAS, dtype=np.float64)

    t0 = time.perf_counter()
    for idx, delta in enumerate(deltas):
        drive = apply_rabi_drift(base_drive, RabiDrift(delta=float(delta)))
        ham = carrier_hamiltonian(hilbert, drive, ion_index=0)
        parameters = {
            "scenario": "rabi_drift_scan_demo",
            "delta": float(delta),
            "N_fock": N_FOCK,
        }
        request_hash = compute_request_hash(parameters)
        res = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=tlist,
            observables=[spin_z(hilbert, 0)],
            request_hash=request_hash,
            storage_mode=StorageMode.OMITTED,
            provenance_tags=("demo", "rabi_drift_scan"),
        )
        final_sigma_z[idx] = float(res.expectations["sigma_z_0"][-1])
    elapsed = time.perf_counter() - t0
    print(f"    scan elapsed: {elapsed:.3f} s  ({elapsed / N_DELTAS * 1e3:.1f} ms/point)")

    # Analytic reference — sz_final = −cos((1 + delta) π) for δ=0.
    analytic = -np.cos((1.0 + deltas) * np.pi)
    max_error = float(np.max(np.abs(final_sigma_z - analytic)))
    print(f"    max |numerical − analytic| = {max_error:.3e}  (solver fidelity)")

    # Fidelity = (1 + sz_final) / 2 is the |↑⟩ population; a "good"
    # calibration needs fidelity ≥ 0.99, i.e. sz_final ≥ 0.98.
    at_threshold = np.abs(final_sigma_z - 1.0) < 0.02  # F ≥ 99%
    if at_threshold.any():
        delta_at_99pct = float(np.ptp(deltas[at_threshold]) / 2.0)
        print(
            f"    |delta| ≤ {delta_at_99pct:.3f} keeps π-pulse fidelity ≥ 99 % "
            "(calibration tolerance)"
        )
    else:
        delta_at_99pct = float("nan")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        deltas=deltas,
        final_sigma_z=final_sigma_z,
        analytic_final_sigma_z=analytic,
    )

    demo_report = {
        "scenario": "rabi_drift_scan_demo",
        "purpose": (
            "deterministic Rabi-amplitude scan — shows how a systematic "
            "calibration error rotates the spin past or short of the π "
            "target, and quantifies the calibration tolerance for "
            "99 % π-pulse fidelity."
        ),
        "workplan_reference": ("WORKPLAN_v0.3.md §5 Phase 1 systematics layer (Dispatch T)"),
        "convention_references": [
            "§3 Spin basis",
            "§18.1 Noise taxonomy (drifts are systematic, not stochastic)",
            "§18.4 Drift primitives",
        ],
        "scan_elapsed_seconds": elapsed,
        "n_deltas": N_DELTAS,
        "delta_max": DELTA_MAX,
        "max_numerical_vs_analytic_error": max_error,
        "delta_tolerance_at_99pct_fidelity": delta_at_99pct,
        "t_pi_us": float(t_pi * 1e6),
        "parameters": {
            "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
            "initial_state": "|↓, 0⟩",
            "N_fock": N_FOCK,
        },
        "environment": _environment(),
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": 2,
    }
    (OUTPUT_DIR / "demo_report.json").write_text(
        json.dumps(demo_report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/{{arrays.npz, demo_report.json}}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not installed; skipping plot")
        return 0

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(
        deltas * 100,
        final_sigma_z,
        color="#d62728",
        linewidth=1.8,
        label=r"$\langle\sigma_z\rangle(t_\pi)$ (numerical)",
    )
    ax.plot(
        deltas * 100,
        analytic,
        color="black",
        linewidth=0.6,
        linestyle="--",
        label=r"$-\cos((1+\delta)\pi)$ (analytic)",
    )
    # 99 % fidelity band on sz (corresponds to |δ| < ~0.09).
    ax.axhspan(0.98, 1.0, color="#2ca02c", alpha=0.15, label="F ≥ 99 % band")
    ax.axhline(1.0, color="grey", linewidth=0.3)
    ax.axhline(0.0, color="grey", linewidth=0.3)
    ax.axvline(0.0, color="grey", linewidth=0.3, linestyle=":")
    ax.set_xlabel(r"Rabi drift $\delta$ (%)")
    ax.set_ylabel(r"$\langle\sigma_z\rangle$ at $t=t_\pi$")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(
        "π-pulse miscalibration sensitivity — "
        f"$\\Omega_0/2\\pi = {RABI_OVER_2PI_MHZ}$ MHz, "
        f"$t_\\pi = {t_pi * 1e6:.2f}$ μs"
    )
    ax.legend(loc="lower center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
