# SPDX-License-Identifier: MIT
"""Benchmark — GHZ / cat factory properties: parity fringes and parity eigenvalues.

WP-01 §7 row 5 (dispatch EDE). A compute-only, deterministic benchmark of two
textbook factory properties — no solver trajectory, no randomness — with **zero
application framing** (the oracle is closed-form only).

(a) GHZ parity oscillation (Heisenberg-limited fringe).
    For the N-ion GHZ state ``(|↑⟩^⊗N + |↓⟩^⊗N) / √2``
    (:func:`iontrap_dynamics.states.ghz_state` on a spin-only Hilbert space),
    apply the phase rotation ``U(φ) = exp(−i φ J_z)`` with the collective
    generator ``J_z = ½ Σ_i σ_z^{(i)}`` and measure the spin parity
    ``P = ∏_i σ_x^{(i)}`` (each ``σ_x`` embedded per ion via
    ``hilbert.spin_op_for_ion``). The two GHZ branches accumulate opposite
    phases ``e^{∓ i N φ / 2}`` and ``P`` swaps them, so

        ⟨P⟩(φ) = cos(N φ)

    — a fringe oscillating at N times the single-spin frequency (the
    Heisenberg-limited phase sensitivity behind the N² QFI of the GHZ probe).
    The benchmark confirms the numerical ⟨P⟩(φ) over a φ grid matches
    ``cos(N φ)`` for N = 2 and N = 3 to tolerance.

(b) Cat / GHZ parity eigenvalues.
    :func:`iontrap_dynamics.states.cat_mode` with ``parity="even"`` is a +1
    eigenstate of the Fock parity operator ``Π = diag((−1)^n)`` and
    ``parity="odd"`` is a −1 eigenstate. Reported alongside the GHZ
    entanglement witness ``concurrence_trajectory`` (N = 2 → 1), tying the
    factory's two faces (spin GHZ, motional cat) into one artefact.

Compute-only and deterministic — a repo-precedent compute benchmark
(``report.json`` + ``arrays.npz`` + ``plot.png``), not a solve-based cache
artefact. The binding oracle assertions live in
``tests/regression/analytic/test_ghz_cat_properties.py``.

Usage::

    python tools/run_benchmark_ghz_cat.py

Output::

    benchmarks/data/ghz_cat/
      report.json  — fringe samples, parity eigenvalues, the analytic oracle,
                     max error, provenance
      arrays.npz   — phi, parity_n2, parity_n3, cos_2phi, cos_3phi
      plot.png     — ⟨P⟩(φ) vs φ for N = 2, 3 with the cos(N φ) overlay
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
from iontrap_dynamics.entanglement import concurrence_trajectory
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.operators import sigma_x_ion, sigma_z_ion
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import cat_mode, ghz_state
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "ghz_cat"

N_VALUES = (2, 3)
N_PHI = 73  # samples on [0, 2π], inclusive endpoints
CAT_FOCK_DIM = 24
CAT_ALPHA = 1.3

PLOT_ALT_TEXT = (
    "Plot of the GHZ spin parity expectation value of the product of sigma_x over "
    "all ions versus the phase phi for two ion numbers N. The N equals 2 curve "
    "oscillates as cosine of two phi and the N equals 3 curve as cosine of three "
    "phi, an N-fold faster Heisenberg-limited fringe. The numerical points lie "
    "exactly on the analytic cosine of N phi reference lines."
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


def _spin_x_parity(hilbert: HilbertSpace) -> qutip.Qobj:
    """The spin parity ``P = ∏_i σ_x`` on the spins of ``hilbert``."""
    ops = [hilbert.spin_op_for_ion(sigma_x_ion(), i) for i in range(hilbert.n_ions)]
    total = ops[0]
    for op in ops[1:]:
        total = total * op
    return total


def _fock_parity(fock_dim: int) -> qutip.Qobj:
    """The Fock parity operator ``Π = diag((−1)^n)`` on a mode of size ``fock_dim``."""
    return qutip.Qobj(np.diag([(-1.0) ** n for n in range(fock_dim)]))


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

    phi = np.linspace(0.0, 2.0 * np.pi, N_PHI, dtype=np.float64)

    # (a) GHZ parity fringe ⟨P⟩(φ) = cos(N φ) for N = 2, 3.
    parity_curves: dict[int, np.ndarray] = {}
    analytic_curves: dict[int, np.ndarray] = {}
    for n in N_VALUES:
        hilbert = _spin_hilbert(n)
        jz = _collective_jz(hilbert)
        parity_op = _spin_x_parity(hilbert)
        ghz = ghz_state(hilbert)
        curve = np.empty(N_PHI, dtype=np.float64)
        for k, p in enumerate(phi):
            evolved = (-1j * p * jz).expm() * ghz
            curve[k] = float(qutip.expect(parity_op, evolved))
        parity_curves[n] = curve
        analytic_curves[n] = np.cos(n * phi)

    fringe_error = max(
        float(np.max(np.abs(parity_curves[n] - analytic_curves[n]))) for n in N_VALUES
    )

    # (b) Cat parity eigenvalues: even → +1, odd → −1.
    fock_parity = _fock_parity(CAT_FOCK_DIM)
    cat_even = float(qutip.expect(fock_parity, cat_mode(CAT_FOCK_DIM, CAT_ALPHA, parity="even")))
    cat_odd = float(qutip.expect(fock_parity, cat_mode(CAT_FOCK_DIM, CAT_ALPHA, parity="odd")))
    cat_error = max(abs(cat_even - 1.0), abs(cat_odd - (-1.0)))

    # GHZ entanglement witness for N = 2: Wootters concurrence → 1.
    h2 = _spin_hilbert(2)
    ghz_concurrence_n2 = float(
        concurrence_trajectory([ghz_state(h2)], hilbert=h2, ion_indices=(0, 1))[0]
    )
    concurrence_error = abs(ghz_concurrence_n2 - 1.0)

    max_error = max(fringe_error, cat_error, concurrence_error)

    print(">>> GHZ / cat factory benchmark — parity fringe <P>(phi) = cos(N phi)")
    print(f"{'phi':>10} {'<P> N=2':>12} {'cos(2phi)':>12} {'<P> N=3':>12} {'cos(3phi)':>12}")
    print("-" * 62)
    for k in range(0, N_PHI, max(1, N_PHI // 8)):
        print(
            f"{phi[k]:>10.5f} {parity_curves[2][k]:>12.8f} {analytic_curves[2][k]:>12.8f} "
            f"{parity_curves[3][k]:>12.8f} {analytic_curves[3][k]:>12.8f}"
        )
    print("-" * 62)
    print(f"cat parity:  even = {cat_even:+.8f} (oracle +1)   odd = {cat_odd:+.8f} (oracle -1)")
    print(f"GHZ N=2 concurrence = {ghz_concurrence_n2:.8f} (oracle 1)")
    print(f"max |numerical - analytic| = {max_error:.2e}")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        phi=phi,
        parity_n2=parity_curves[2],
        parity_n3=parity_curves[3],
        cos_2phi=analytic_curves[2],
        cos_3phi=analytic_curves[3],
    )

    report = {
        "scenario": "ghz_cat",
        "purpose": (
            "GHZ / cat factory properties (WP-01 §7 row 5, dispatch EDE). "
            "(a) The N-ion GHZ parity fringe <P>(phi) = cos(N phi) with "
            "P = prod_i sigma_x and U(phi) = exp(-i phi J_z), "
            "J_z = 0.5 * sum_i sigma_z, for N=2 and N=3 (Heisenberg-limited, "
            "N-fold faster than a single spin). (b) cat_mode parity eigenvalues "
            "(even -> +1, odd -> -1) under the Fock parity diag((-1)^n), plus the "
            "GHZ N=2 Wootters concurrence (-> 1). Application-agnostic: textbook "
            "oracle only, no application framing."
        ),
        "workplan_reference": "WP/WP-01-estimation-darwinism.md (section 7 row 5, dispatch EDE)",
        "schema_version": 2,
        # Compute-only benchmark: no solve() trajectory, so no canonical cache
        # request hash. Carried as null for schema parity with solve-based
        # demo_report.json artefacts.
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": "qutip",
        "backend_version": qutip.__version__,
        "generator": "U(phi) = exp(-i phi J_z), J_z = 0.5 * sum_i sigma_z_ion(i)",
        "parity_observable": "P = prod_i sigma_x_ion(i)",
        "n_values": list(N_VALUES),
        "n_phi": N_PHI,
        "cat_fock_dim": CAT_FOCK_DIM,
        "cat_alpha": CAT_ALPHA,
        "results": [
            {
                "phi": float(phi[k]),
                "parity_n2": float(parity_curves[2][k]),
                "cos_2phi": float(analytic_curves[2][k]),
                "parity_n3": float(parity_curves[3][k]),
                "cos_3phi": float(analytic_curves[3][k]),
            }
            for k in range(N_PHI)
        ],
        "cat_parity_even": cat_even,
        "cat_parity_odd": cat_odd,
        "ghz_concurrence_n2": ghz_concurrence_n2,
        "analytic_formulas": {
            "ghz_parity": "<X^N>(phi) = cos(N phi)",
            "cat_parity": "+1 (even) / -1 (odd)",
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
    ax.plot(phi, analytic_curves[2], color="#1f77b4", linewidth=1.0, label=r"$\cos(2\phi)$")
    ax.plot(phi, analytic_curves[3], color="#d62728", linewidth=1.0, label=r"$\cos(3\phi)$")
    ax.scatter(
        phi, parity_curves[2], color="#1f77b4", marker="o", s=14, zorder=3, label="N=2 (numerical)"
    )
    ax.scatter(
        phi, parity_curves[3], color="#d62728", marker="s", s=14, zorder=3, label="N=3 (numerical)"
    )
    ax.set_xlabel(r"phase $\phi$")
    ax.set_ylabel(r"GHZ parity $\langle P\rangle = \langle\prod_i \sigma_x^{(i)}\rangle$")
    ax.set_title(r"GHZ parity fringe: $\langle P\rangle(\phi) = \cos(N\phi)$")
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(-1.15, 1.15)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
