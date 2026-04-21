# SPDX-License-Identifier: MIT
"""Phase 2 baseline — sesolve-vs-mesolve speedup across canonical scenarios.

Opens the Phase 2 performance track (Dispatch X). For every public
Hamiltonian class on a single-ion Hilbert space, runs the trajectory
twice — once forcing ``solver="sesolve"`` (the Schrödinger fast path)
and once forcing ``solver="mesolve"`` (the master-equation fallback)
— and reports the wall-clock ratio. Both paths evolve the same pure
ket, so expectations agree to ODE tolerance and the difference is
purely solver overhead.

**Empirical outcome on QuTiP 5.2 (single-core laptop).** At the
Hilbert-space sizes this library uses (dim ≤ 48), the two paths
run at **comparable speed** — mean wall-clock ratio ~1.0×, not the
2–3× advantage sesolve had in QuTiP 4.x. The dispatch still opts
into sesolve on ket inputs for **semantic** reasons (Schrödinger
is the correct dynamics for pure states) and leaves headroom for
larger Hilbert spaces where density-matrix lifting cost grows
quadratically in dim(H); this benchmark records the baseline so
Phase 2 follow-ons (sparse ops, JAX backend) can be measured
against a fixed starting point.

Usage::

    python tools/run_benchmark_sesolve_speedup.py

Requires matplotlib; prints a summary table and writes a JSON report.

Output::

    benchmarks/data/sesolve_speedup/
      report.json  — per-scenario wall-clock + speedup ratio
      plot.png     — bar chart of speedup ratios
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

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import (
    blue_sideband_hamiltonian,
    carrier_hamiltonian,
    red_sideband_hamiltonian,
)
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import number, spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "sesolve_speedup"

N_STEPS = 400
N_REPEATS = 3  # average over a few runs to damp jitter
RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)

# Three Hilbert-space sizes — exposes how the density-matrix lifting
# cost grows with dim(H) while the Schrödinger path scales linearly.
HILBERT_VARIANTS = [
    ("carrier_fock04", 4),
    ("carrier_fock12", 12),
    ("carrier_fock24", 24),
]


def _build_ion(fock: int) -> tuple[HilbertSpace, qutip.Qobj, qutip.Qobj]:
    mode = ModeConfig(
        label="axial",
        frequency_rad_s=MODE_FREQ_RAD_S,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"axial": fock})
    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, WAVENUMBER_M_INV],
        carrier_rabi_frequency_rad_s=RABI_RAD_S,
        phase_rad=0.0,
    )
    return hilbert, drive, qutip.tensor(spin_down(), qutip.basis(fock, 0))


def _time_solve(
    *,
    hamiltonian,
    hilbert: HilbertSpace,
    psi_0: qutip.Qobj,
    tlist: np.ndarray,
    solver: str,
) -> tuple[float, dict[str, np.ndarray]]:
    """Run ``solve`` N times and return (min_elapsed, expectations)."""
    obs = [spin_z(hilbert, 0), number(hilbert, "axial")]
    elapsed = float("inf")
    last_exps: dict[str, np.ndarray] = {}
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        res = solve(
            hilbert=hilbert,
            hamiltonian=hamiltonian,
            initial_state=psi_0,
            times=tlist,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            solver=solver,
        )
        elapsed = min(elapsed, time.perf_counter() - t0)
        last_exps = dict(res.expectations)
    return elapsed, last_exps


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
    rabi_period = 2 * np.pi / RABI_RAD_S
    tlist = np.linspace(0.0, 4 * rabi_period, N_STEPS)

    print(">>> running sesolve-vs-mesolve speedup benchmark")
    print(f"    N_STEPS = {N_STEPS}, N_REPEATS = {N_REPEATS} (min of repeats)")
    print(f"    Ω / 2π = {RABI_OVER_2PI_MHZ} MHz, duration = 4·T_Ω")
    print()

    results: list[dict[str, float | str]] = []
    print(f"{'scenario':<30} {'mesolve [ms]':>14} {'sesolve [ms]':>14} {'speedup':>10}")
    print("-" * 72)
    for label, fock in HILBERT_VARIANTS:
        hilbert, drive, psi_0 = _build_ion(fock)
        hamiltonian = carrier_hamiltonian(hilbert, drive, ion_index=0)
        t_me, exps_me = _time_solve(
            hamiltonian=hamiltonian,
            hilbert=hilbert,
            psi_0=psi_0,
            tlist=tlist,
            solver="mesolve",
        )
        t_se, exps_se = _time_solve(
            hamiltonian=hamiltonian,
            hilbert=hilbert,
            psi_0=psi_0,
            tlist=tlist,
            solver="sesolve",
        )
        # Verify the two paths agree.
        max_err = 0.0
        for key in exps_me:
            max_err = max(max_err, float(np.max(np.abs(exps_me[key] - exps_se[key]))))
        speedup = t_me / t_se if t_se > 0 else float("inf")
        results.append(
            {
                "scenario": label,
                "fock_dim": fock,
                "hilbert_dim": 2 * fock,
                "mesolve_seconds": t_me,
                "sesolve_seconds": t_se,
                "speedup": speedup,
                "max_expectation_error": max_err,
            }
        )
        print(f"{label:<30} {t_me * 1e3:>14.2f} {t_se * 1e3:>14.2f} {speedup:>10.2f}×")

    # Also run a sideband-Hamiltonian scenario — lower-symmetry case.
    hilbert, drive, psi_0 = _build_ion(8)
    for name, builder_label, builder in [
        ("rsb_fock08", "red_sideband", red_sideband_hamiltonian),
        ("bsb_fock08", "blue_sideband", blue_sideband_hamiltonian),
    ]:
        psi_0 = qutip.tensor(spin_down(), qutip.basis(8, 1))
        ham = builder(hilbert, drive, "axial", ion_index=0)
        t_me, exps_me = _time_solve(
            hamiltonian=ham,
            hilbert=hilbert,
            psi_0=psi_0,
            tlist=tlist,
            solver="mesolve",
        )
        t_se, exps_se = _time_solve(
            hamiltonian=ham,
            hilbert=hilbert,
            psi_0=psi_0,
            tlist=tlist,
            solver="sesolve",
        )
        max_err = max(float(np.max(np.abs(exps_me[k] - exps_se[k]))) for k in exps_me)
        speedup = t_me / t_se if t_se > 0 else float("inf")
        results.append(
            {
                "scenario": name,
                "builder": builder_label,
                "fock_dim": 8,
                "hilbert_dim": 16,
                "mesolve_seconds": t_me,
                "sesolve_seconds": t_se,
                "speedup": speedup,
                "max_expectation_error": max_err,
            }
        )
        print(f"{name:<30} {t_me * 1e3:>14.2f} {t_se * 1e3:>14.2f} {speedup:>10.2f}×")

    mean_speedup = float(np.mean([r["speedup"] for r in results]))
    print("-" * 72)
    print(f"mean speedup: {mean_speedup:.2f}×")

    report = {
        "scenario": "sesolve_speedup",
        "purpose": (
            "Phase 2 / v0.3 Dispatch X baseline — measures sesolve fast-"
            "path speedup on pure-ket trajectories across several "
            "Hilbert-space sizes and Hamiltonian builders."
        ),
        "workplan_reference": (
            "WORKPLAN_v0.3.md §5 Phase 2 — performance (Dispatch X: sesolve dispatch)"
        ),
        "n_steps": N_STEPS,
        "n_repeats": N_REPEATS,
        "results": results,
        "mean_speedup": mean_speedup,
        "environment": _environment(),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/report.json")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0

    labels = [r["scenario"] for r in results]
    speedups = [r["speedup"] for r in results]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(labels, speedups, color="#2ca02c")
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_ylabel("mesolve / sesolve wall-clock ratio")
    ax.set_title(f"sesolve fast-path speedup — {qutip.__version__}, mean = {mean_speedup:.2f}×")
    for i, s in enumerate(speedups):
        ax.text(i, s + 0.05, f"{s:.2f}×", ha="center", fontsize=8)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
