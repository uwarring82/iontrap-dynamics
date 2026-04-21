# SPDX-License-Identifier: MIT
"""Phase 2 baseline — solve_ensemble parallelism crossover.

Measures ``solve_ensemble`` wall-clock under three joblib backends
(``sequential``, ``threading``, ``loky``) across three single-solve
cost regimes (small / medium / large). Establishes the crossover
point above which process-based parallelism begins to pay off.

**Headline findings on QuTiP 5.2 + Python 3.13 (single-core laptop).**

At single-solve cost well below process-spawn + pickle cost, loky
loses badly; at single-solve cost above ~15 ms with 2000+ steps,
loky wins cleanly with 2–3× speedup. Concrete crossover from the
three regimes here:

- **small** (fock=3, n_steps=200, single-solve ~2.7 ms):
  serial 88 ms, loky ~2 s (~22× slower). Threading 72 ms — tiny
  edge over serial because the 200-step loop has enough work to
  keep multiple threads busy inside BLAS while dodging per-step
  Python overhead.
- **medium** (fock=12, n_steps=500, single-solve ~6 ms): serial
  85 ms, loky 91 ms — essentially tied. Crossover regime.
- **large** (fock=24, n_steps=2000, single-solve ~16 ms): serial
  499 ms, loky 186 ms (2.68× speedup), threading 617 ms
  (Python overhead hurts at longer step counts).

``solve_ensemble`` therefore defaults to ``n_jobs=1``. Scenarios at
the large-regime end can flip to ``n_jobs=-1`` with
``parallel_backend="loky"`` for real wins.

Usage::

    python tools/run_benchmark_ensemble_parallel.py

Output::

    benchmarks/data/ensemble_parallel/
      report.json  — per-regime wall-clock for each backend
      plot.png     — bar chart of wall-clock ratios
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
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import number, spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve, solve_ensemble
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "ensemble_parallel"

RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0
RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)

# Three scale regimes — progressively bigger Hilbert × more steps.
REGIMES = [
    ("small  fock=3  n=200", 3, 200),
    ("medium fock=12 n=500", 12, 500),
    ("large  fock=24 n=2000", 24, 2000),
]
ENSEMBLE_SIZE = 20
BACKENDS = [
    ("n_jobs=1 (serial)", 1, "loky"),
    ("n_jobs=-1 loky", -1, "loky"),
    ("n_jobs=-1 threading", -1, "threading"),
]


def _build_scenario(fock: int) -> tuple[HilbertSpace, qutip.Qobj, qutip.Qobj]:
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
    ham = carrier_hamiltonian(hilbert, drive, ion_index=0)
    psi_0 = qutip.tensor(spin_down(), qutip.basis(fock, 0))
    return hilbert, ham, psi_0


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

    print(">>> running solve_ensemble parallelism crossover benchmark")
    print(f"    ensemble size = {ENSEMBLE_SIZE}")
    print()
    print(f"{'regime':<25} {'single-solve':>13} " + " ".join(f"{b[0]:>22}" for b in BACKENDS))
    print("-" * 120)

    results: list[dict[str, object]] = []
    for regime_label, fock, n_steps in REGIMES:
        hilbert, ham, psi_0 = _build_scenario(fock)
        tlist = np.linspace(0.0, 4 * rabi_period, n_steps)
        obs = [spin_z(hilbert, 0), number(hilbert, "axial")]

        # Single-solve cost as baseline.
        t0 = time.perf_counter()
        _ = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=tlist,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
        )
        t_single = time.perf_counter() - t0

        row: dict[str, object] = {
            "regime": regime_label,
            "hilbert_dim": hilbert.total_dim,
            "n_steps": n_steps,
            "ensemble_size": ENSEMBLE_SIZE,
            "single_solve_ms": t_single * 1e3,
            "backends": {},
        }
        backends_row: dict[str, float] = {}
        for backend_label, n_jobs, parallel_backend in BACKENDS:
            t0 = time.perf_counter()
            _ = solve_ensemble(
                hilbert=hilbert,
                hamiltonians=[ham] * ENSEMBLE_SIZE,
                initial_state=psi_0,
                times=tlist,
                observables=obs,
                storage_mode=StorageMode.OMITTED,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
            )
            elapsed = time.perf_counter() - t0
            backends_row[backend_label] = elapsed
        row["backends"] = backends_row
        results.append(row)

        backends_fmt = " ".join(f"{backends_row[b[0]] * 1e3:>18.0f} ms" for b in BACKENDS)
        print(f"{regime_label:<25} {t_single * 1e3:>10.1f} ms " + backends_fmt)

    print("-" * 120)
    # Highlight the winner per regime.
    for row in results:
        backends_row = row["backends"]  # type: ignore[index]
        winner = min(backends_row, key=backends_row.get)  # type: ignore[arg-type]
        speedup_vs_serial = backends_row["n_jobs=1 (serial)"] / backends_row[winner]
        row["winner_backend"] = winner
        row["winner_speedup_vs_serial"] = speedup_vs_serial
        print(f"  {row['regime']:<25} fastest: {winner:<22} (×{speedup_vs_serial:.2f} vs serial)")

    report = {
        "scenario": "ensemble_parallel_crossover",
        "purpose": (
            "Phase 2 Dispatch Y baseline — identifies where joblib "
            "parallelism begins to pay off. At the library's current "
            "Hilbert-space scales (dim ≤ 48), process-based parallelism "
            "is *not* faster than serial due to pickle + process-spawn "
            "overhead; solve_ensemble default is n_jobs=1. Callers "
            "hitting longer single-solve cost (>100–500 ms) can flip "
            "to n_jobs=-1 after measuring here."
        ),
        "workplan_reference": (
            "WORKPLAN_v0.3.md §5 Phase 2 — parallel sweeps via joblib (Dispatch Y)"
        ),
        "regimes": results,
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

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    width = 0.25
    x = np.arange(len(REGIMES))
    for i, (backend_label, _, _) in enumerate(BACKENDS):
        heights = [r["backends"][backend_label] * 1e3 for r in results]  # type: ignore[index]
        ax.bar(x + (i - 1) * width, heights, width, label=backend_label)
    ax.set_xticks(x)
    ax.set_xticklabels([r["regime"] for r in results], fontsize=8)
    ax.set_ylabel("ensemble wall-clock [ms]")
    ax.set_yscale("log")
    ax.set_title(
        f"solve_ensemble parallelism crossover — {ENSEMBLE_SIZE} trials, QuTiP {qutip.__version__}"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
