# SPDX-License-Identifier: MIT
"""Benchmark — recoverability of a dephased system+accessible qubit pair.

WP-01 §7 row 4 (dispatch EDE). Compute-only proof that the recoverability
primitive (:func:`iontrap_dynamics.information.recoverability`, the clamped
coherent information ``max(0, S(ρ_A) − S(ρ_{S∪A}))`` in bits) reproduces the
textbook endpoints of a known-strength dephasing channel — with **zero
application framing** (the oracle is textbook only).

The channel is realised as the two-qubit Werner family

    ρ(p) = p · |Φ⁺⟩⟨Φ⁺| + (1 − p) · I/4 ,

a maximally-entangled Bell pair |Φ⁺⟩ (built via
:func:`iontrap_dynamics.states.ghz_state` on a two-ion spin-only
``HilbertSpace``) mixed with the maximally-mixed state ``I/4``. Qubit 0 is the
system ``S``; qubit 1 is the accessible part ``A``. Sweeping the mixing
parameter ``p ∈ [0, 1]`` interpolates from full decoherence (``p = 0``, the
accessible qubit uncorrelated with the system) to perfect recovery (``p = 1``,
the system maximally entangled with the accessible qubit).

The closed-form oracle at the endpoints is exact:

- **perfect recovery** ``p = 1`` — recoverability ``= H_S = 1`` bit;
- **full decoherence** ``p = 0`` — recoverability ``= 0``;

and the measure is monotone non-decreasing in ``p`` between them. Compute-only
and deterministic — no solver trajectory, no randomness — so this is a
repo-precedent compute benchmark (``report.json`` + ``arrays.npz`` +
``plot.png``), not a solve-based cache artefact. The binding oracle assertion
lives in ``tests/regression/analytic/test_recoverability_channel.py``.

Usage::

    python tools/run_benchmark_recoverability.py

Output::

    benchmarks/data/recoverability/
      report.json  — per-p recoverability, the analytic endpoints, max error, provenance
      arrays.npz   — p (mixing grid), recoverability (bits)
      plot.png     — recoverability vs p (monotone, endpoints 0 and 1)
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
from iontrap_dynamics.information import recoverability
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import ghz_state
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "recoverability"

N_POINTS = 21  # uniform grid on [0, 1] including both endpoints

PLOT_ALT_TEXT = (
    "Plot of recoverability in bits versus the Werner mixing parameter p for a "
    "two-qubit dephasing channel rho(p) = p times the Bell state plus one minus "
    "p times the maximally mixed state I over four, with qubit zero the system "
    "and qubit one the accessible part. The curve is monotone non-decreasing: it "
    "is exactly zero at p = 0 (full decoherence), stays clamped at zero for small "
    "p, then rises smoothly to exactly one bit at p = 1 (perfect recovery), the "
    "full system entropy H_S."
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

    hilbert = _spin_hilbert(2)
    bell = ghz_state(hilbert)  # |Phi+> = (|up,up> + |down,down>)/sqrt(2)
    bell_dm = bell * bell.dag()
    maximally_mixed = qutip.tensor(qutip.qeye(2), qutip.qeye(2)) / 4.0

    p_arr = np.linspace(0.0, 1.0, N_POINTS, dtype=np.float64)
    recov = np.empty(N_POINTS, dtype=np.float64)
    for idx, p in enumerate(p_arr):
        rho = float(p) * bell_dm + (1.0 - float(p)) * maximally_mixed
        recov[idx] = recoverability(
            rho, hilbert=hilbert, system_indices=[0], accessible_indices=[1]
        )

    analytic_perfect = 1.0  # p = 1: recoverability = H_S = 1 bit
    analytic_decohered = 0.0  # p = 0: recoverability = 0
    err_perfect = abs(float(recov[-1]) - analytic_perfect)
    err_decohered = abs(float(recov[0]) - analytic_decohered)
    max_error = float(max(err_perfect, err_decohered))

    # Monotonicity of the channel between the endpoints.
    monotone_nondecreasing = bool(np.all(np.diff(recov) >= -1e-12))

    print(">>> Recoverability benchmark — Werner dephasing channel rho(p)")
    print(f"{'p':>6} {'recoverability':>16}")
    print("-" * 24)
    for i in range(N_POINTS):
        print(f"{p_arr[i]:>6.3f} {recov[i]:>16.10f}")
    print("-" * 24)
    print(f"oracle p=0 (decohered): recoverability = {recov[0]:.10f}  (analytic 0)")
    print(f"oracle p=1 (perfect)  : recoverability = {recov[-1]:.10f}  (analytic H_S = 1)")
    print(f"monotone non-decreasing: {monotone_nondecreasing}")
    print(f"max |numerical - analytic| (endpoints) = {max_error:.2e}")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        p=p_arr,
        recoverability=recov,
    )

    report = {
        "scenario": "recoverability",
        "purpose": (
            "Compute-only proof of WP-01 §7 row 4: the recoverability primitive "
            "(clamped coherent information max(0, S(rho_A) - S(rho_{S union A})) "
            "in bits) reproduces the textbook endpoints of a known-strength "
            "dephasing channel, realised as the two-qubit Werner family "
            "rho(p) = p |Phi+><Phi+| + (1 - p) I/4 with system_indices=[0] and "
            "accessible_indices=[1]. Perfect recovery at p=1 gives H_S = 1 bit; "
            "full decoherence at p=0 gives 0; monotone non-decreasing between. "
            "Application-agnostic: textbook oracle only, no application framing."
        ),
        "workplan_reference": "WP/WP-01-estimation-darwinism.md (§7 row 4, dispatch EDE)",
        "schema_version": 2,
        # Compute-only benchmark: no solve() trajectory, so no canonical cache
        # request hash. Carried as null for schema parity with solve-based
        # demo_report.json artefacts.
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": "qutip",
        "backend_version": qutip.__version__,
        "channel": "Werner family rho(p) = p |Phi+><Phi+| + (1 - p) I/4 on 2 qubits",
        "system_indices": [0],
        "accessible_indices": [1],
        "n_points": N_POINTS,
        "monotone_nondecreasing": monotone_nondecreasing,
        "results": [
            {
                "p": float(p_arr[i]),
                "recoverability": float(recov[i]),
            }
            for i in range(N_POINTS)
        ],
        "analytic_formulas": {
            "perfect": "recoverability = H_S",
            "decohered": "recoverability = 0",
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

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(p_arr, recov, color="#1f77b4", linewidth=1.0, zorder=2)
    ax.scatter(p_arr, recov, color="#1f77b4", marker="o", s=20, zorder=3, label="numerical")
    ax.scatter(
        [0.0, 1.0],
        [analytic_decohered, analytic_perfect],
        color="#d62728",
        marker="x",
        s=60,
        zorder=4,
        label="analytic endpoints",
    )
    ax.set_xlabel("Werner mixing parameter $p$")
    ax.set_ylabel(r"recoverability $\max(0,\,S(\rho_A) - S(\rho_{S\cup A}))$ [bits]")
    ax.set_title("Recoverability vs dephasing: 0 at decoherence, $H_S=1$ at recovery")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
