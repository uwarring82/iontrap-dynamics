# SPDX-License-Identifier: MIT
"""Benchmark — quantum-Darwinism redundancy of the GHZ cascade.

WP-01 §7 row 3 (dispatch EDE). A compute-only proof of the Darwinism
*plateau* and the *redundancy* count, exercising the partial-information
primitives (:func:`iontrap_dynamics.partial_information_plot` and
:func:`iontrap_dynamics.redundancy`) together with the GHZ factory
(:func:`iontrap_dynamics.ghz_state`) — with **zero application framing**
(the oracle is textbook quantum Darwinism only).

Model — the GHZ cascade. One *system* qubit (subsystem 0) is perfectly copied
onto ``N`` *environment* qubits (subsystems ``1 .. N``) by placing all ``N + 1``
spins in a single GHZ state ``(|↑…↑⟩ + |↓…↓⟩) / √2``. This is the canonical
decoherence model of quantum Darwinism: the system bit is imprinted redundantly
across the environment.

For each ``N`` the tool builds a spin-only ``HilbertSpace`` of ``N + 1`` ions
inline, prepares the GHZ state, and reads off two textbook quantities:

- the **partial-information plot** ``I(S:F)`` over nested fragments — every
  non-empty *proper* fragment already carries the full system entropy
  ``H_S = 1`` bit (the Darwinism plateau), and the curve only jumps to ``2``
  bits when the *whole* environment is read out;
- the **redundancy** ``R_δ = N`` at deficit ``δ = 0.1`` — each single
  environment qubit already suffices to learn the system bit, so the bit is
  imprinted ``N`` times over.

Compute-only and deterministic — no solver trajectory, no randomness — so this
is a repo-precedent compute benchmark (``report.json`` + ``arrays.npz`` +
``plot.png``), not a solve-based cache artefact. The binding oracle assertion
lives in ``tests/regression/analytic/test_darwinism_redundancy.py``.

Usage::

    python tools/run_benchmark_darwinism_redundancy.py

Output::

    benchmarks/data/darwinism_redundancy/
      report.json  — per-N plateau height, redundancy, the analytic oracle,
                      max error, provenance
      arrays.npz   — n_env, plateau_height, redundancy, pip_largest_n
      plot.png     — the partial-information plot for a representative N
                     (0 → plateau at 1 bit → 2) and R_δ versus N
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import qutip

from iontrap_dynamics.conventions import CONVENTION_VERSION
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.information import partial_information_plot, redundancy
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import ghz_state
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "darwinism_redundancy"

# Number of environment qubits N; the Hilbert space carries N + 1 ions.
N_VALUES = (3, 4, 5, 6)
DELTA = 0.1

PLOT_ALT_TEXT = (
    "Two-panel figure for the GHZ-cascade quantum-Darwinism benchmark. Left "
    "panel: the partial-information plot, mutual information I(S:F) in bits "
    "versus environment-fragment size, for a representative environment of N "
    "qubits. The curve rises from zero to a flat plateau at one bit for every "
    "non-empty proper fragment, then jumps to two bits only when the entire "
    "environment is included. Right panel: the redundancy R_delta versus N, a "
    "straight line R_delta = N, showing the system bit is imprinted N times "
    "over. Numerical points lie exactly on the analytic references."
)


def _spin_hilbert(n_ions: int) -> HilbertSpace:
    """A spin-only Hilbert space of ``n_ions`` qubits (no motional modes)."""
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})


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

    n_arr = np.array(N_VALUES, dtype=np.int64)
    plateau_height = np.empty(len(N_VALUES), dtype=np.float64)
    redundancy_arr = np.empty(len(N_VALUES), dtype=np.float64)
    pip_largest: np.ndarray = np.empty(0, dtype=np.float64)
    for idx, n in enumerate(N_VALUES):
        hilbert = _spin_hilbert(n + 1)
        state = ghz_state(hilbert)
        env_indices = list(range(1, n + 1))
        pip = partial_information_plot(
            state, hilbert=hilbert, system_indices=[0], environment_indices=env_indices
        )
        # The Darwinism plateau: the value at every non-empty proper fragment.
        # I(S:F_1) (a single-qubit fragment) reads off the plateau height.
        plateau_height[idx] = float(pip[1])
        redundancy_arr[idx] = redundancy(
            state,
            hilbert=hilbert,
            system_indices=[0],
            environment_indices=env_indices,
            delta=DELTA,
        )
        if n == N_VALUES[-1]:
            pip_largest = np.asarray(pip, dtype=np.float64)

    analytic_plateau = np.ones(len(N_VALUES), dtype=np.float64)
    analytic_redundancy = n_arr.astype(np.float64)
    max_error = float(
        max(
            np.max(np.abs(plateau_height - analytic_plateau)),
            np.max(np.abs(redundancy_arr - analytic_redundancy)),
        )
    )

    print(">>> Darwinism redundancy benchmark — GHZ cascade (plateau = 1 bit, R_delta = N)")
    print(f"{'N':>3} {'plateau':>10} {'H_S':>5} {'R_delta':>10} {'N':>4}")
    print("-" * 38)
    for i, n in enumerate(N_VALUES):
        print(f"{n:>3} {plateau_height[i]:>10.6f} {1:>5d} {redundancy_arr[i]:>10.6f} {n:>4d}")
    print("-" * 38)
    print(f"PIP (largest N = {N_VALUES[-1]}): {np.array2string(pip_largest, precision=6)}")
    print(f"max |numerical - analytic| = {max_error:.2e}")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        n_env=n_arr,
        plateau_height=plateau_height,
        redundancy=redundancy_arr,
        analytic_plateau=analytic_plateau,
        analytic_redundancy=analytic_redundancy,
        pip_largest_n=pip_largest,
    )

    report = {
        "scenario": "darwinism_redundancy",
        "purpose": (
            "Quantum-Darwinism proof of WP-01 §7 row 3: the GHZ cascade (one "
            "system qubit perfectly copied onto N environment qubits in a "
            "single GHZ state) exhibits the partial-information plateau "
            "I(S:F) = H_S = 1 bit for every non-empty proper fragment, jumping "
            "to 2 bits only at the full environment, and redundancy "
            "R_delta = N at deficit delta = 0.1. Exercises the "
            "partial_information_plot and redundancy primitives with the GHZ "
            "factory. Application-agnostic: textbook oracle only, no "
            "application framing."
        ),
        "workplan_reference": "WP/WP-01-estimation-darwinism.md (§7 row 3, dispatch EDE)",
        "schema_version": 2,
        # Compute-only benchmark: no solve() trajectory, so no canonical cache
        # request hash. Carried as null for schema parity with solve-based
        # demo_report.json artefacts.
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": "qutip",
        "backend_version": qutip.__version__,
        "model": (
            "GHZ cascade: 1 system qubit (subsystem 0) + N environment qubits "
            "(subsystems 1..N) in one GHZ state; system_indices=[0], "
            "environment_indices=[1..N]"
        ),
        "delta": DELTA,
        "n_values": list(N_VALUES),
        "results": [
            {
                "n_env": int(n),
                "plateau_height": float(plateau_height[i]),
                "analytic_plateau": float(analytic_plateau[i]),
                "redundancy": float(redundancy_arr[i]),
                "analytic_redundancy": float(analytic_redundancy[i]),
            }
            for i, n in enumerate(N_VALUES)
        ],
        "pip_largest_n": {
            "n_env": int(N_VALUES[-1]),
            "values": [float(v) for v in pip_largest],
        },
        "analytic_formulas": {
            "plateau": "I(S:F) = H_S = 1 bit",
            "redundancy": "R_delta = N",
        },
        "max_numerical_vs_analytic_error": max_error,
        "tolerance": 1e-9,
        "plot_alt_text": PLOT_ALT_TEXT,
        "environment": _environment(),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/report.json + arrays.npz")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0

    n_float = n_arr.astype(np.float64)
    fig, (ax_pip, ax_red) = plt.subplots(1, 2, figsize=(10.0, 4.5))

    # Left: the partial-information plot for the largest N.
    n_largest = N_VALUES[-1]
    frag_sizes = np.arange(len(pip_largest), dtype=np.float64)
    ax_pip.axhline(
        1.0, color="#999999", linewidth=0.8, linestyle="--", label="plateau $H_S = 1$ bit"
    )
    ax_pip.plot(frag_sizes, pip_largest, color="#1f77b4", linewidth=1.0, marker="o", zorder=3)
    ax_pip.set_xlabel("environment-fragment size $|F|$")
    ax_pip.set_ylabel(r"mutual information $I(S{:}F)$ (bits)")
    ax_pip.set_title(f"Partial-information plot ($N = {n_largest}$)")
    ax_pip.set_ylim(-0.1, 2.2)
    ax_pip.legend(frameon=False, fontsize=8)

    # Right: redundancy versus N.
    ax_red.plot(
        n_float, analytic_redundancy, color="#d62728", linewidth=1.0, label=r"$R_\delta = N$"
    )
    ax_red.scatter(
        n_float, redundancy_arr, color="#d62728", marker="s", zorder=3, label="numerical"
    )
    ax_red.set_xlabel("environment qubits $N$")
    ax_red.set_ylabel(rf"redundancy $R_\delta$ ($\delta = {DELTA}$)")
    ax_red.set_title("Redundancy of the GHZ cascade")
    ax_red.legend(frameon=False, fontsize=8)

    fig.suptitle("Quantum Darwinism of the GHZ cascade")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
