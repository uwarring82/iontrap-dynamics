# SPDX-License-Identifier: MIT
"""AAH — exact-diagonalization envelope benchmark.

Characterises the dense ``scipy.linalg.eigh`` path that
:func:`iontrap_dynamics.solve_spectrum` drives, across a grid of
``(N, n_c)`` configurations using the same non-RWA Clos 2016 spin-boson
Hamiltonian the regression tests consume. Answers the user-facing
question "how large a system can I exactly diagonalise on my laptop?"
and supplies the evidence AAG's dispatch gate depends on.

Each ``(N, n_c)`` point runs in an isolated subprocess so that peak
RSS reflects that single run rather than accumulated history. The
subprocess builds the Hamiltonian, calls ``solve_spectrum``, prints a
single JSON line, and exits. The parent collects results and writes
the bundle plus a log-log wall-clock-vs-dim plot.

Usage::

    python tools/run_benchmark_spectrum_envelope.py

Writes::

    benchmarks/data/spectrum_envelope/
      report.json  -- per-point dim, matrix size, wall-clock, peak RSS
      plot.png     -- wall-clock + peak RSS vs Hilbert dim
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "spectrum_envelope"

# Fixed physical configuration (matches the Clos 2016 N=1 theo_dim surface
# parameters, scaled up for the multi-ion points). Axial frequency and Rabi
# are both 0.71 legacy-MHz; detuning pinned at 0.5 (off-resonance, dense
# spectrum). n_bar = 1.0 for the thermal initial state.
AXIAL_MHZ = 0.71
RABI_MHZ = 0.71
DETUNING_MHZ = 0.5
N_BAR = 1.0

# (N, n_c) grid. Dim = 2 * (n_c + 1)^N. Points ordered so the sweep climbs
# dimension steadily. The "core" sweep below dim ~5 000 fits in ~2.3 GB peak
# RSS and runs in under 2 min total on Apple-Silicon-class hardware. The
# tail (last three points) deliberately pushes a 16 GB system into measurable
# swap territory to exercise the extrapolation against measured data — that
# block can take 15–25 min on the reference hardware and is gated by the
# `--include-large` CLI flag so casual reruns do not hammer the laptop.
GRID_CORE: list[tuple[int, int]] = [
    (1, 5), (1, 10), (1, 20), (1, 50), (1, 100),
    (2, 5), (2, 10), (2, 15), (2, 20), (2, 25),
    (3, 4), (3, 6), (3, 8), (3, 10), (3, 12),
    (4, 3), (4, 4), (4, 5), (4, 6),
    (5, 3),
]
GRID_LARGE: list[tuple[int, int]] = [
    (5, 4),  # dim 6 250  -- predicted peak RSS ~3.4 GB, ~3 min
    (4, 7),  # dim 8 192  -- predicted peak RSS ~5.5 GB, ~5 min
    (3, 14), # dim 10 368 -- predicted peak RSS ~11 GB,  ~10 min on swap
]

# Child-process script. Lives inline so the benchmark is self-contained;
# the parent launches it via ``python -c ``.
CHILD_SCRIPT = r"""
import json, math, resource, sys, time

import numpy as np

from iontrap_dynamics import (
    CLOS2016_LEGACY_WAVELENGTH_M,
    clos2016_initial_state,
    clos2016_spin_boson_hamiltonian,
    solve_spectrum,
)
from iontrap_dynamics.species import mg25_plus


def cchain(n):
    '''Reproduce the legacy cchain normal-mode solver inline (beta=4).'''
    if n == 1:
        return np.array([1.0]), np.array([1.0])
    from scipy.optimize import minimize
    def potential(x, beta=4.0):
        return float(np.sum(x**2)) + (beta / 2) * sum(
            1.0 / abs(x[i] - x[j]) for i in range(len(x)) for j in range(i)
        )
    def hessian(x, beta=4.0):
        m = len(x)
        H = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i == j:
                    s = sum(1.0 / abs(x[i] - x[k])**3 for k in range(m) if k != i)
                    H[i, i] = beta * s + 2.0
                else:
                    H[i, j] = -beta / abs(x[i] - x[j])**3
        return H
    x0 = np.linspace(-1.0, 1.0, n) * (n / 2.0)
    res = minimize(potential, x0, tol=1e-14)
    eigvals, eigvecs = np.linalg.eigh(0.5 * hessian(res.x))
    freqs = np.sqrt(np.maximum(eigvals, 0.0))
    return freqs / freqs[0], eigvecs[0, :].copy()


def main():
    n_ions = int(sys.argv[1])
    cutoff = int(sys.argv[2])
    axial_hz = float(sys.argv[3])
    rabi_hz = float(sys.argv[4])
    detuning_hz = float(sys.argv[5])
    n_bar = float(sys.argv[6])

    dimensionless_freqs, first_ion_weights = cchain(n_ions)

    init = clos2016_initial_state(
        max_phonons=cutoff,
        mean_occupations=[n_bar] * n_ions,
        theta_rad=0.0, phi_rad=0.0,
    )

    # Time just the Hamiltonian build plus solve_spectrum. Excluded:
    # module import, cchain, initial-state build.
    t0 = time.perf_counter()
    H = clos2016_spin_boson_hamiltonian(
        max_phonons=cutoff,
        axial_frequency_rad_s=axial_hz * 2 * math.pi,
        dimensionless_mode_frequencies=dimensionless_freqs.tolist(),
        center_mode_weights=first_ion_weights.tolist(),
        carrier_rabi_frequency_rad_s=rabi_hz * 2 * math.pi,
        detuning_rad_s=detuning_hz * 2 * math.pi,
        ion_mass_kg=mg25_plus().mass_kg,
        laser_wavelength_m=CLOS2016_LEGACY_WAVELENGTH_M,
    )
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    spec = solve_spectrum(H, initial_state=init)
    t_solve = time.perf_counter() - t0

    dim = spec.eigenvalues.shape[0]
    # ru_maxrss: macOS reports bytes, Linux reports KB. Normalise to bytes.
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss_raw if sys.platform == 'darwin' else rss_raw * 1024

    print(json.dumps({
        'n_ions': n_ions, 'cutoff': cutoff, 'dim': int(dim),
        'matrix_bytes': int(dim * dim * 16),
        'build_seconds': t_build, 'solve_seconds': t_solve,
        'peak_rss_bytes': int(rss_bytes),
    }))


if __name__ == '__main__':
    main()
"""


def _environment() -> dict[str, str]:
    import numpy  # noqa: PLC0415
    import qutip  # noqa: PLC0415
    import scipy  # noqa: PLC0415

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "qutip": qutip.__version__,
    }


def _fmt_bytes(b: int) -> str:
    if b < 1024**2:
        return f"{b / 1024:.1f} KB"
    if b < 1024**3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


def _run_point(n_ions: int, cutoff: int, *, timeout_s: int = 600) -> dict[str, float | int] | None:
    print(f"  running N={n_ions} n_c={cutoff} ... ", end="", flush=True)
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [
                sys.executable, "-c", CHILD_SCRIPT,
                str(n_ions), str(cutoff),
                str(AXIAL_MHZ * 1e6), str(RABI_MHZ * 1e6),
                str(DETUNING_MHZ * 1e6), str(N_BAR),
            ],
            capture_output=True, text=True, check=True, timeout=timeout_s,
        )
    except subprocess.CalledProcessError as exc:
        print(f"FAILED ({exc.returncode})")
        print(exc.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT (>{timeout_s} s)")
        return None

    wall = time.perf_counter() - t0
    last_line = result.stdout.strip().splitlines()[-1]
    payload = json.loads(last_line)
    payload["wall_seconds_including_subprocess_launch"] = wall
    print(
        f"dim={payload['dim']:>6}  "
        f"solve={payload['solve_seconds']:6.2f} s  "
        f"peak_rss={_fmt_bytes(payload['peak_rss_bytes'])}  "
        f"matrix={_fmt_bytes(payload['matrix_bytes'])}"
    )
    return payload


def main() -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-large", action="store_true",
        help="Run the three large-dim points (dim 6 250 / 8 192 / 10 368). "
             "These push a 16 GB system into measurable swap; total runtime "
             "balloons to ~25 min on the reference hardware.",
    )
    args = parser.parse_args()

    grid = GRID_CORE + (GRID_LARGE if args.include_large else [])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> exact-diagonalization envelope benchmark (AAH)")
    print(f"    grid: {len(grid)} points; isolated subprocess per point")
    if args.include_large:
        print(f"    INCLUDING {len(GRID_LARGE)} large-dim points (16 GB push)")
    print(f"    axial = Rabi = {AXIAL_MHZ} MHz, detuning = {DETUNING_MHZ} MHz, n_bar = {N_BAR}")
    print()

    results: list[dict[str, float | int]] = []
    for n_ions, cutoff in grid:
        # Large-dim points get a generous timeout; swap-induced slowdown can
        # easily 2-3x the no-swap wall-clock prediction.
        is_large = (n_ions, cutoff) in GRID_LARGE
        timeout_s = 1800 if is_large else 600
        point = _run_point(n_ions, cutoff, timeout_s=timeout_s)
        if point is not None:
            results.append(point)

    report = {
        "scenario": "spectrum_envelope",
        "purpose": (
            "AAH -- measures dense scipy.linalg.eigh wall-clock and peak RSS "
            "for the Clos 2016 non-RWA spin-boson Hamiltonian across an "
            "(N, n_c) grid. Answers the user-facing 'how big can I "
            "exact-diagonalise on my laptop?' question and supplies the "
            "evidence the AAG dispatch gate depends on."
        ),
        "workplan_reference": (
            "docs/workplan-clos-2016-integration.md §4.3 PC exact-diag envelope; §5 AAH"
        ),
        "configuration": {
            "axial_frequency_MHz": AXIAL_MHZ,
            "rabi_frequency_MHz": RABI_MHZ,
            "detuning_MHz": DETUNING_MHZ,
            "n_bar": N_BAR,
            "species": "25Mg+",
            "wavelength_label": "CLOS2016_LEGACY_WAVELENGTH_M (Raman two-photon)",
        },
        "results": results,
        "environment": _environment(),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/report.json")

    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("matplotlib not available -- skipping plot")
        return 0

    by_n: dict[int, list[dict[str, float | int]]] = {}
    for r in results:
        by_n.setdefault(int(r["n_ions"]), []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    colours = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728", 5: "#9467bd"}

    for n_ions in sorted(by_n):
        pts = sorted(by_n[n_ions], key=lambda r: r["dim"])
        dims = [r["dim"] for r in pts]
        solves = [r["solve_seconds"] for r in pts]
        rsss = [r["peak_rss_bytes"] / 1024**2 for r in pts]
        axes[0].loglog(dims, solves, "o-", color=colours[n_ions], label=f"N={n_ions}")
        axes[1].loglog(dims, rsss, "o-", color=colours[n_ions], label=f"N={n_ions}")

    axes[0].set_xlabel("Hilbert dimension")
    axes[0].set_ylabel("solve_spectrum wall-clock [s]")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(title="N ions")
    axes[0].set_title("Dense eigh wall-clock")

    axes[1].set_xlabel("Hilbert dimension")
    axes[1].set_ylabel("peak process RSS [MiB]")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(title="N ions")
    axes[1].set_title("Peak RSS (subprocess-isolated)")

    fig.suptitle(
        f"Exact-diag envelope -- scipy {report['environment']['scipy']} on "
        f"{report['environment']['platform']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
