# SPDX-License-Identifier: MIT
"""JAX counterpart to the AAH exact-diagonalization envelope benchmark.

Mirrors ``run_benchmark_spectrum_envelope.py`` scenario-for-scenario
but dispatches dense eigh through ``jax.numpy.linalg.eigh`` (XLA-
backed LAPACK) instead of ``scipy.linalg.eigh``. Same
subprocess-isolated-per-point design, same Clos 2016 non-RWA
Hamiltonian, same physical parameters; swapping the solver backend
keeps everything else fixed so the comparison lands on the solver
alone.

A warm-up eigh call on the same Hamiltonian precedes the timed call
to isolate **steady-state** wall-clock from XLA's first-call trace
and JIT compile. Peak RSS is captured after the timed call — XLA on
CPU keeps arrays in host memory, so ``ru_maxrss`` is directly
comparable to the scipy run.

Usage::

    python tools/run_benchmark_spectrum_envelope_jax.py [--include-large]

Writes::

    benchmarks/data/spectrum_envelope/
      report_jax.json  -- JAX counterpart of report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "spectrum_envelope"

AXIAL_MHZ = 0.71
RABI_MHZ = 0.71
DETUNING_MHZ = 0.5
N_BAR = 1.0

GRID_CORE: list[tuple[int, int]] = [
    (1, 5),
    (1, 10),
    (1, 20),
    (1, 50),
    (1, 100),
    (2, 5),
    (2, 10),
    (2, 15),
    (2, 20),
    (2, 25),
    (3, 4),
    (3, 6),
    (3, 8),
    (3, 10),
    (3, 12),
    (4, 3),
    (4, 4),
    (4, 5),
    (4, 6),
    (5, 3),
]
GRID_LARGE: list[tuple[int, int]] = [
    (5, 4),
    (4, 7),
    (3, 14),
]

CHILD_SCRIPT = r"""
import json, math, os, resource, sys, time

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import numpy as np
import jax
import jax.numpy as jnp

from iontrap_dynamics import (
    CLOS2016_LEGACY_WAVELENGTH_M,
    clos2016_initial_state,
    clos2016_spin_boson_hamiltonian,
)
from iontrap_dynamics.species import mg25_plus


def cchain(n):
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
    _init = clos2016_initial_state(
        max_phonons=cutoff,
        mean_occupations=[n_bar] * n_ions,
        theta_rad=0.0, phi_rad=0.0,
    )

    # Build the Hamiltonian (scipy/qutip-side; same as the scipy benchmark).
    t0 = time.perf_counter()
    H_qutip = clos2016_spin_boson_hamiltonian(
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

    # Hand the dense matrix to JAX. device_put ensures it's materialised on
    # the default device; without this the first eigh would include the
    # host->device transfer cost and report a distorted wall-clock.
    H_np = np.asarray(H_qutip.full(), dtype=np.complex128)
    t0 = time.perf_counter()
    H_jax = jax.device_put(jnp.asarray(H_np))
    t_transfer = time.perf_counter() - t0

    # Warm-up call: trace + compile + run once, block so no async tail.
    t0 = time.perf_counter()
    ev_warm, _ = jax.numpy.linalg.eigh(H_jax)
    ev_warm.block_until_ready()
    t_warm = time.perf_counter() - t0

    # Timed steady-state call.
    t0 = time.perf_counter()
    ev, vv = jax.numpy.linalg.eigh(H_jax)
    ev.block_until_ready()
    vv.block_until_ready()
    t_solve = time.perf_counter() - t0

    dim = int(ev.shape[0])
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss_raw if sys.platform == "darwin" else rss_raw * 1024

    print(json.dumps({
        "n_ions": n_ions, "cutoff": cutoff, "dim": dim,
        "matrix_bytes": int(dim * dim * 16),
        "build_seconds": t_build, "transfer_seconds": t_transfer,
        "warmup_seconds": t_warm, "solve_seconds": t_solve,
        "peak_rss_bytes": int(rss_bytes),
    }))


if __name__ == "__main__":
    main()
"""


def _environment() -> dict[str, str]:
    import jax
    import jaxlib
    import numpy
    import qutip
    import scipy

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "qutip": qutip.__version__,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax_backend": jax.default_backend(),
    }


def _fmt_bytes(b: int) -> str:
    if b < 1024**2:
        return f"{b / 1024:.1f} KB"
    if b < 1024**3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


def _run_point(n_ions: int, cutoff: int, *, timeout_s: int = 1800) -> dict[str, float | int] | None:
    print(f"  running N={n_ions} n_c={cutoff} ... ", end="", flush=True)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD_SCRIPT,
                str(n_ions),
                str(cutoff),
                str(AXIAL_MHZ * 1e6),
                str(RABI_MHZ * 1e6),
                str(DETUNING_MHZ * 1e6),
                str(N_BAR),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_s,
        )
    except subprocess.CalledProcessError as exc:
        print(f"FAILED ({exc.returncode})")
        print(exc.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT (>{timeout_s} s)")
        return None

    last_line = result.stdout.strip().splitlines()[-1]
    payload = json.loads(last_line)
    print(
        f"dim={payload['dim']:>6}  "
        f"warm={payload['warmup_seconds']:6.2f} s  "
        f"solve={payload['solve_seconds']:6.2f} s  "
        f"peak_rss={_fmt_bytes(payload['peak_rss_bytes'])}  "
        f"matrix={_fmt_bytes(payload['matrix_bytes'])}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Run the three large-dim points (dim 6 250 / 6 750 / 8 192).",
    )
    args = parser.parse_args()

    grid = GRID_CORE + (GRID_LARGE if args.include_large else [])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> exact-diagonalization envelope benchmark (JAX counterpart)")
    print(f"    grid: {len(grid)} points; isolated subprocess per point")
    print(f"    axial = Rabi = {AXIAL_MHZ} MHz, detuning = {DETUNING_MHZ} MHz, n_bar = {N_BAR}")
    print()

    results: list[dict[str, float | int]] = []
    for n_ions, cutoff in grid:
        is_large = (n_ions, cutoff) in GRID_LARGE
        timeout_s = 1800 if is_large else 600
        point = _run_point(n_ions, cutoff, timeout_s=timeout_s)
        if point is not None:
            results.append(point)

    report = {
        "scenario": "spectrum_envelope_jax",
        "purpose": (
            "JAX counterpart to AAH -- measures jax.numpy.linalg.eigh "
            "wall-clock and peak RSS on the same Clos 2016 non-RWA "
            "spin-boson Hamiltonian. Warm-up call precedes the timed "
            "call so steady-state cost is reported."
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
    (OUTPUT_DIR / "report_jax.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {(OUTPUT_DIR / 'report_jax.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
