# SPDX-License-Identifier: MIT
"""Keystone benchmark — QFI scaling: GHZ (Heisenberg N²) vs product (SQL N).

WI-1 / WI-3 of WP-01 (dispatch EDA). The single decoupling proof of the
estimation surface: it exercises the SLD-QFI primitive
(:func:`iontrap_dynamics.information.quantum_fisher_information_trajectory`),
the GHZ factory (:func:`iontrap_dynamics.states.ghz_state`), and the
SQL/Heisenberg distinction in one figure — with **zero application framing**
(no TMC content; the oracle is textbook only).

For each qubit number N it builds the GHZ probe and the product probe
``|+⟩^⊗N`` and computes their SLD quantum Fisher information under the
collective generator ``J_z = ½ Σ_i σ_z^{(i)}``. The closed-form oracle is
``F_Q(GHZ) = N²`` (the Heisenberg limit) and ``F_Q(product) = N`` (the standard
quantum limit). Compute-only and deterministic — no solver trajectory, no
randomness — so this is a repo-precedent compute benchmark
(``report.json`` + ``arrays.npz`` + ``plot.png``), not a solve-based cache
artefact. The binding oracle assertion lives in
``tests/regression/analytic/test_qfi_scaling.py``.

Usage::

    python tools/run_benchmark_qfi_scaling.py

Output::

    benchmarks/data/qfi_scaling/
      report.json  — per-N QFI, the analytic oracle, max error, provenance
      arrays.npz   — n_ions, qfi_ghz, qfi_product, analytic_ghz, analytic_product
      plot.png     — QFI vs N (log–log): GHZ N² above product N
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
from iontrap_dynamics.information import quantum_fisher_information_trajectory
from iontrap_dynamics.operators import sigma_z_ion, spin_down, spin_up
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import ghz_state
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "qfi_scaling"

N_VALUES = (1, 2, 3, 4, 5, 6)

PLOT_ALT_TEXT = (
    "Log-log plot of quantum Fisher information versus qubit number N for two "
    "probe states under the collective generator J_z = one half sum of sigma_z. "
    "The GHZ curve follows Heisenberg scaling N squared (slope 2); the "
    "product-state curve follows the standard quantum limit N (slope 1). The "
    "numerical points lie exactly on the analytic reference lines."
)


def _spin_hilbert(n_ions: int) -> HilbertSpace:
    """A spin-only Hilbert space of ``n_ions`` qubits (no motional modes)."""
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})


def _collective_jz(hilbert: HilbertSpace) -> qutip.Qobj:
    """``J_z = ½ Σ_i σ_z`` on the spins of ``hilbert``."""
    ops = [hilbert.spin_op_for_ion(sigma_z_ion(), i) for i in range(hilbert.n_ions)]
    total = ops[0]
    for op in ops[1:]:
        total = total + op
    return 0.5 * total


def _product_plus(n_ions: int) -> qutip.Qobj:
    """The product probe ``|+⟩^⊗N`` with ``|+⟩ = (|↑⟩ + |↓⟩) / √2``."""
    plus = (spin_up() + spin_down()).unit()
    return qutip.tensor([plus] * n_ions)


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
    qfi_ghz = np.empty(len(N_VALUES), dtype=np.float64)
    qfi_product = np.empty(len(N_VALUES), dtype=np.float64)
    for idx, n in enumerate(N_VALUES):
        hilbert = _spin_hilbert(n)
        jz = _collective_jz(hilbert)
        qfi_ghz[idx] = quantum_fisher_information_trajectory(
            [ghz_state(hilbert)], hilbert=hilbert, generator=jz
        )[0]
        qfi_product[idx] = quantum_fisher_information_trajectory(
            [_product_plus(n)], hilbert=hilbert, generator=jz
        )[0]

    analytic_ghz = n_arr.astype(np.float64) ** 2
    analytic_product = n_arr.astype(np.float64)
    max_error = float(
        max(
            np.max(np.abs(qfi_ghz - analytic_ghz)),
            np.max(np.abs(qfi_product - analytic_product)),
        )
    )

    print(">>> QFI scaling benchmark — GHZ (N^2, Heisenberg) vs product (N, SQL)")
    print(f"{'N':>3} {'QFI_GHZ':>14} {'N^2':>6} {'QFI_product':>14} {'N':>4}")
    print("-" * 48)
    for i, n in enumerate(N_VALUES):
        print(
            f"{n:>3} {qfi_ghz[i]:>14.8f} {int(analytic_ghz[i]):>6d} {qfi_product[i]:>14.8f} {n:>4d}"
        )
    print("-" * 48)
    print(f"max |numerical - analytic| = {max_error:.2e}")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        n_ions=n_arr,
        qfi_ghz=qfi_ghz,
        qfi_product=qfi_product,
        analytic_ghz=analytic_ghz,
        analytic_product=analytic_product,
    )

    report = {
        "scenario": "qfi_scaling",
        "purpose": (
            "Keystone decoupling proof of WP-01: QFI scaling of a GHZ probe "
            "(Heisenberg limit N^2) versus a product probe (standard quantum "
            "limit N) under the collective generator J_z = 0.5 * sum_i sigma_z. "
            "Exercises the SLD-QFI primitive and the GHZ factory together. "
            "Application-agnostic: textbook oracle only, no application framing."
        ),
        "workplan_reference": "WP/WP-01-estimation-darwinism.md (WI-1/WI-3, dispatch EDA)",
        "schema_version": 2,
        # Compute-only benchmark: no solve() trajectory, so no canonical cache
        # request hash. Carried as null for schema parity with solve-based
        # demo_report.json artefacts.
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": "qutip",
        "backend_version": qutip.__version__,
        "generator": "J_z = 0.5 * sum_i sigma_z_ion(i)",
        "n_values": list(N_VALUES),
        "results": [
            {
                "n_ions": int(n),
                "qfi_ghz": float(qfi_ghz[i]),
                "analytic_ghz": float(analytic_ghz[i]),
                "qfi_product": float(qfi_product[i]),
                "analytic_product": float(analytic_product[i]),
            }
            for i, n in enumerate(N_VALUES)
        ],
        "analytic_formulas": {
            "qfi_ghz": "N**2 (Heisenberg limit)",
            "qfi_product": "N (standard quantum limit)",
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
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(n_float, analytic_ghz, color="#1f77b4", linewidth=1.0, label="GHZ: $N^2$ (Heisenberg)")
    ax.plot(n_float, analytic_product, color="#d62728", linewidth=1.0, label="product: $N$ (SQL)")
    ax.scatter(n_float, qfi_ghz, color="#1f77b4", marker="o", zorder=3, label="GHZ (numerical)")
    ax.scatter(
        n_float, qfi_product, color="#d62728", marker="s", zorder=3, label="product (numerical)"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("qubit number $N$")
    ax.set_ylabel(r"quantum Fisher information $F_Q[\rho,\,J_z]$")
    ax.set_title("QFI scaling: GHZ (Heisenberg) vs product (SQL)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
