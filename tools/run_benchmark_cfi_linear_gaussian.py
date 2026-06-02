# SPDX-License-Identifier: MIT
"""Benchmark — classical Fisher information: linear-Gaussian matrix + Braunstein–Caves bound.

WP-01 §7 row 2 (dispatch EDE). Two exact, textbook oracles in one compute-only
benchmark — **zero application framing**, closed forms only:

(a) **Linear-Gaussian Fisher matrix.** For a family of design matrices ``A``
    with diagonal noise covariance ``Σ``, the Fisher information matrix of the
    model ``y = A θ + ε``, ``ε ~ N(0, Σ)``, is ``F = AᵀΣ⁻¹A``. The benchmark
    confirms :func:`iontrap_dynamics.information.linear_gaussian_fisher` equals
    the hand-computed ``AᵀΣ⁻¹A`` to machine precision across a sweep of variance
    values, and that on a 1-parameter case the single-parameter Cramér–Rao bound
    :func:`iontrap_dynamics.information.cramer_rao_bound` returns ``1 / F`` —
    i.e. the CRB is the inverse Fisher information.

(b) **Braunstein–Caves saturation: CFI ≤ QFI.** A single-qubit phase model with
    probe ``|+⟩ = (|↑⟩ + |↓⟩)/√2`` and generator ``J_z = ½ σ_z`` has quantum
    Fisher information ``F_Q = 1`` (computed with
    :func:`iontrap_dynamics.information.quantum_fisher_information_trajectory`).
    The optimal σ_x-basis measurement has outcome probabilities ``[½, ½]`` with
    derivatives ``[-½, +½]``, giving classical Fisher information ``F_C = 1`` —
    the bound is **saturated** (``F_C = F_Q``). A phase-insensitive measurement
    whose probabilities do not move (derivative ``[0, 0]``) gives ``F_C = 0``,
    strictly below ``F_Q`` — the bound holds with slack.

Compute-only and deterministic — no solver trajectory, no randomness — so this
is a repo-precedent compute benchmark (``report.json`` + ``arrays.npz`` +
``plot.png``), not a solve-based cache artefact. The binding oracle assertions
live in ``tests/regression/analytic/test_cfi_linear_gaussian.py``.

Usage::

    python tools/run_benchmark_cfi_linear_gaussian.py

Output::

    benchmarks/data/cfi_linear_gaussian/
      report.json  — Fisher matrices, CFI/QFI saturation, max error, provenance
      arrays.npz   — variances, library F values, analytic F values
      plot.png     — Fisher-matrix entry F[0,0] vs noise variance (library vs analytic)
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
from iontrap_dynamics.information import (
    classical_fisher_information,
    cramer_rao_bound,
    linear_gaussian_fisher,
    quantum_fisher_information_trajectory,
)
from iontrap_dynamics.operators import sigma_z_ion, spin_down, spin_up
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "cfi_linear_gaussian"

# Fixed two-observation, two-parameter design matrix for the linear-Gaussian
# oracle. The variance sweep scales an isotropic diagonal noise covariance.
DESIGN_MATRIX = np.array([[1.0, 0.0], [0.5, 2.0]], dtype=np.float64)
VARIANCE_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0)

PLOT_ALT_TEXT = (
    "Log-log plot of the leading Fisher-matrix entry F[0,0] versus the noise "
    "variance sigma squared for the linear-Gaussian model y = A theta + epsilon. "
    "The library values from linear_gaussian_fisher lie exactly on the analytic "
    "reference curve A-transpose Sigma-inverse A, which falls as one over the "
    "variance (slope minus one)."
)


def _spin_hilbert(n_ions: int) -> HilbertSpace:
    """A spin-only Hilbert space of ``n_ions`` qubits (no motional modes)."""
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})


def _plus_state() -> qutip.Qobj:
    """The single-qubit probe ``|+⟩ = (|↑⟩ + |↓⟩) / √2``."""
    return (spin_up() + spin_down()).unit()


def _single_qubit_jz(hilbert: HilbertSpace) -> qutip.Qobj:
    """The phase generator ``J_z = ½ σ_z`` on the one spin of ``hilbert``."""
    return 0.5 * hilbert.spin_op_for_ion(sigma_z_ion(), 0)


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

    # --- (a) Linear-Gaussian Fisher matrix: F = AᵀΣ⁻¹A -------------------
    variances = np.array(VARIANCE_VALUES, dtype=np.float64)
    n_obs = DESIGN_MATRIX.shape[0]
    n_params = DESIGN_MATRIX.shape[1]
    library_f = np.empty((len(VARIANCE_VALUES), n_params, n_params), dtype=np.float64)
    analytic_f = np.empty_like(library_f)
    lg_errors: list[float] = []
    lg_results: list[dict[str, object]] = []
    for idx, var in enumerate(VARIANCE_VALUES):
        sigma = var * np.eye(n_obs, dtype=np.float64)
        f_lib = linear_gaussian_fisher(A=DESIGN_MATRIX, sigma=sigma)
        f_hand = DESIGN_MATRIX.T @ np.linalg.inv(sigma) @ DESIGN_MATRIX
        library_f[idx] = f_lib
        analytic_f[idx] = f_hand
        err = float(np.max(np.abs(f_lib - f_hand)))
        lg_errors.append(err)
        lg_results.append(
            {
                "variance": float(var),
                "library_fisher_matrix": f_lib.tolist(),
                "analytic_fisher_matrix": f_hand.tolist(),
                "max_abs_error": err,
            }
        )

    # 1-parameter Cramér–Rao check: CRB == 1 / F[0,0] on the unit-variance case.
    f00 = float(library_f[VARIANCE_VALUES.index(1.0)][0, 0])
    crb = cramer_rao_bound(f00)
    crb_analytic = 1.0 / f00
    crb_error = abs(crb - crb_analytic)

    # --- (b) Braunstein–Caves saturation: CFI ≤ QFI ----------------------
    hilbert = _spin_hilbert(1)
    jz = _single_qubit_jz(hilbert)
    qfi = float(
        quantum_fisher_information_trajectory([_plus_state()], hilbert=hilbert, generator=jz)[0]
    )
    # Optimal σ_x-basis measurement: probabilities [½, ½], derivatives [-½, +½].
    cfi_optimal = classical_fisher_information([0.5, 0.5], parameter_derivative=[-0.5, 0.5])
    # Phase-insensitive measurement: probabilities do not move, derivative [0, 0].
    cfi_insensitive = classical_fisher_information([0.5, 0.5], parameter_derivative=[0.0, 0.0])

    qfi_analytic = 1.0
    cfi_optimal_analytic = 1.0
    cfi_insensitive_analytic = 0.0
    bc_errors = [
        abs(qfi - qfi_analytic),
        abs(cfi_optimal - cfi_optimal_analytic),
        abs(cfi_optimal - qfi),  # saturation: CFI == QFI
        abs(cfi_insensitive - cfi_insensitive_analytic),
    ]

    max_error = float(max(max(lg_errors), crb_error, max(bc_errors)))

    print(">>> CFI benchmark — (a) linear-Gaussian Fisher matrix  (b) Braunstein–Caves CFI <= QFI")
    print("(a) F = A^T Sigma^-1 A, diagonal Sigma = sigma^2 * I:")
    print(f"{'sigma^2':>9} {'F[0,0] (lib)':>14} {'F[0,0] (analytic)':>18} {'max|err|':>12}")
    print("-" * 56)
    for idx, var in enumerate(VARIANCE_VALUES):
        print(
            f"{var:>9.4f} {library_f[idx][0, 0]:>14.8f} "
            f"{analytic_f[idx][0, 0]:>18.8f} {lg_errors[idx]:>12.2e}"
        )
    print("-" * 56)
    print(f"    Cramer–Rao (1-param, sigma^2=1): CRB={crb:.8f}, 1/F={crb_analytic:.8f}")
    print("(b) single-qubit phase model, |+>, generator J_z = 0.5 * sigma_z:")
    print(f"    QFI                          = {qfi:.8f}  (analytic 1)")
    print(f"    CFI (optimal sigma_x basis)  = {cfi_optimal:.8f}  (analytic 1 -> saturates QFI)")
    print(f"    CFI (phase-insensitive)      = {cfi_insensitive:.8f}  (analytic 0 < QFI)")
    print("-" * 56)
    print(f"max |numerical - analytic| = {max_error:.2e}")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        variances=variances,
        library_fisher=library_f,
        analytic_fisher=analytic_f,
    )

    report = {
        "scenario": "cfi_linear_gaussian",
        "purpose": (
            "Classical Fisher information benchmark of WP-01 §7 row 2: (a) the "
            "linear-Gaussian Fisher matrix F = A^T Sigma^-1 A from "
            "linear_gaussian_fisher matches the hand-computed value and the "
            "single-parameter Cramer–Rao bound equals 1/F; (b) the "
            "Braunstein–Caves inequality CFI <= QFI for a single-qubit phase "
            "model — saturated (CFI = QFI = 1) by the optimal sigma_x-basis "
            "measurement and strict (CFI = 0 < QFI) for a phase-insensitive one. "
            "Application-agnostic: textbook oracle only, no application framing."
        ),
        "workplan_reference": "WP/WP-01-estimation-darwinism.md (§7 row 2, dispatch EDE)",
        "schema_version": 2,
        # Compute-only benchmark: no solve() trajectory, so no canonical cache
        # request hash. Carried as null for schema parity with solve-based
        # demo_report.json artefacts.
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": "qutip",
        "backend_version": qutip.__version__,
        "design_matrix": DESIGN_MATRIX.tolist(),
        "variance_values": list(VARIANCE_VALUES),
        "results": lg_results,
        "cramer_rao": {
            "variance": 1.0,
            "fisher_00": f00,
            "cramer_rao_bound": crb,
            "analytic_inverse_fisher": crb_analytic,
            "max_abs_error": crb_error,
        },
        "braunstein_caves": {
            "generator": "J_z = 0.5 * sigma_z",
            "probe": "|+> = (|up> + |down>)/sqrt(2)",
            "qfi": qfi,
            "qfi_analytic": qfi_analytic,
            "cfi_optimal": cfi_optimal,
            "cfi_optimal_analytic": cfi_optimal_analytic,
            "cfi_phase_insensitive": cfi_insensitive,
            "cfi_phase_insensitive_analytic": cfi_insensitive_analytic,
            "saturates_qfi": bool(abs(cfi_optimal - qfi) <= 1e-9),
        },
        "analytic_formulas": {
            "fisher_matrix": "A^T Sigma^-1 A",
            "saturation": "CFI = QFI at the optimal measurement",
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

    f00_lib = library_f[:, 0, 0]
    f00_analytic = analytic_f[:, 0, 0]
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(
        variances,
        f00_analytic,
        color="#1f77b4",
        linewidth=1.0,
        label=r"analytic $A^\top\Sigma^{-1}A$ entry $F_{00}$",
    )
    ax.scatter(
        variances,
        f00_lib,
        color="#1f77b4",
        marker="o",
        zorder=3,
        label="library (numerical)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"noise variance $\sigma^2$")
    ax.set_ylabel(r"Fisher-matrix entry $F_{00}$")
    ax.set_title("Linear-Gaussian Fisher matrix: library vs analytic")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
