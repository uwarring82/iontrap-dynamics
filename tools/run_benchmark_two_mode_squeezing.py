# SPDX-License-Identifier: MIT
"""Benchmark — two-mode SU(1,1) squeezing: parametric gain n̄ = sinh²(gτ).

Tutorial-companion benchmark for the two-mode surface (WP-02 WI-1/WI-2, dispatches
MCA/MCB). Evolves the two-mode vacuum under the SU(1,1) squeezing Hamiltonian
``H/ℏ = i g (e^{iφ} â†b̂† − e^{−iφ} âb̂)`` and tracks the per-mode occupation, which
grows as the textbook closed form ``⟨n̂_a⟩ = ⟨n̂_b⟩ = sinh²(gτ)`` while the difference
number ``⟨n̂_a − n̂_b⟩`` is pinned at zero (the conserved su(1,1) Casimir / the
photon-number correlation that defines two-mode squeezing). It also cross-checks the
static ``two_mode_squeezed_vacuum`` factory: per-mode ``⟨n̂⟩ = sinh²|z|`` (CONVENTIONS
§23 — explicitly NOT qutip.squeezing's ½ convention). Application-agnostic: textbook
oracle only, no application framing.

Usage::

    python tools/run_benchmark_two_mode_squeezing.py

Output::

    benchmarks/data/two_mode_squeezing/
      report.json  — per-time ⟨n̂⟩, the sinh² oracle, the difference-number, max error
      arrays.npz   — r, n_a, n_b, n_diff, analytic_sinh2, z, n_factory + oracles
      plot.png     — ⟨n̂⟩ vs r = gτ (dynamics + factory) on sinh²(r); difference ≡ 0
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
from iontrap_dynamics.hamiltonians import two_mode_squeezing_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import Observable
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import two_mode_squeezed_vacuum
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "two_mode_squeezing"

FOCK_DIM = 50  # generous: n̄ = sinh²(r) reaches ≈ 2.3 at r = 1.2; keep top-Fock pop tiny
G_RATE = 2.0 * np.pi * 2.0e3  # parametric coupling g, rad·s⁻¹
T_MAX = 1.2 / G_RATE  # so r = g·T_MAX = 1.2
N_TIMES = 40
Z_FACTORY = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)  # squeezing magnitudes for the static factory

PLOT_ALT_TEXT = (
    "Two panels. Left: per-mode mean occupation of the two squeezed modes versus the "
    "dimensionless squeezing r = g times tau, for both the time-evolved dynamics and the "
    "static two-mode squeezed-vacuum factory, overlaid on the analytic curve sinh squared "
    "of r; all numerical points lie on the curve, showing exponential parametric gain. "
    "Right: the difference between the two mode occupations stays flat at zero across the "
    "whole evolution, the conserved su(1,1) Casimir that marks the photon-number correlation."
)


def _two_mode_hilbert(fock_dim: int) -> HilbertSpace:
    """A one-ion, two-mode Hilbert space (the spin is an inert spectator)."""
    eigenvector = np.array([[0.0, 0.0, 1.0]])
    modes = (
        ModeConfig(label="a", frequency_rad_s=2.0 * np.pi * 1.0e6, eigenvector_per_ion=eigenvector),
        ModeConfig(label="b", frequency_rad_s=2.0 * np.pi * 1.1e6, eigenvector_per_ion=eigenvector),
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=modes)
    return HilbertSpace(system=system, fock_truncations={"a": fock_dim, "b": fock_dim})


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

    # --- dynamics: evolve the two-mode vacuum under H_TMS, watch n̄ = sinh²(gτ) ---
    hilbert = _two_mode_hilbert(FOCK_DIM)
    hamiltonian = two_mode_squeezing_hamiltonian(hilbert, g=G_RATE)
    vacuum = qutip.tensor(qutip.basis(2, 0), qutip.basis(FOCK_DIM, 0), qutip.basis(FOCK_DIM, 0))
    times = np.linspace(0.0, T_MAX, N_TIMES)
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=vacuum,
        times=times,
        observables=(
            Observable(label="n_a", operator=hilbert.number_for_mode("a")),
            Observable(label="n_b", operator=hilbert.number_for_mode("b")),
        ),
    )
    n_a = np.asarray(result.expectations["n_a"], dtype=np.float64)
    n_b = np.asarray(result.expectations["n_b"], dtype=np.float64)
    r_grid = G_RATE * times
    analytic_sinh2 = np.sinh(r_grid) ** 2
    n_diff = n_a - n_b

    # --- static factory: two_mode_squeezed_vacuum per-mode ⟨n̂⟩ = sinh²|z| ---
    z_arr = np.array(Z_FACTORY, dtype=np.float64)
    number_a = qutip.tensor(qutip.num(FOCK_DIM), qutip.qeye(FOCK_DIM))
    n_factory = np.array(
        [float(qutip.expect(number_a, two_mode_squeezed_vacuum(FOCK_DIM, z))) for z in Z_FACTORY],
        dtype=np.float64,
    )
    analytic_factory = np.sinh(z_arr) ** 2

    max_error = float(
        max(
            np.max(np.abs(n_a - analytic_sinh2)),
            np.max(np.abs(n_b - analytic_sinh2)),
            np.max(np.abs(n_diff)),
            np.max(np.abs(n_factory - analytic_factory)),
        )
    )

    print(">>> two-mode SU(1,1) squeezing — parametric gain n̄ = sinh²(gτ)")
    print(f"{'r=gτ':>6} {'⟨n_a⟩':>12} {'sinh²(r)':>12} {'⟨n_a−n_b⟩':>12}")
    print("-" * 46)
    for i in range(0, N_TIMES, max(1, N_TIMES // 8)):
        print(f"{r_grid[i]:>6.3f} {n_a[i]:>12.6f} {analytic_sinh2[i]:>12.6f} {n_diff[i]:>12.2e}")
    print("-" * 46)
    print(f"max |numerical - analytic| = {max_error:.2e}")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        r=r_grid,
        n_a=n_a,
        n_b=n_b,
        n_diff=n_diff,
        analytic_sinh2=analytic_sinh2,
        z=z_arr,
        n_factory=n_factory,
        analytic_factory=analytic_factory,
    )

    report = {
        "scenario": "two_mode_squeezing",
        "purpose": (
            "Two-mode SU(1,1) squeezing: evolving the two-mode vacuum under "
            "H/hbar = i g (a-dag b-dag - a b) grows the per-mode occupation as the "
            "closed form sinh^2(g tau) while the difference number a-dag a - b-dag b is "
            "conserved at zero (the su(1,1) Casimir / photon-number correlation). The "
            "static two_mode_squeezed_vacuum factory reproduces per-mode n-bar = sinh^2|z| "
            "(CONVENTIONS v0.3 convention, not qutip.squeezing's 1/2). Application-agnostic: "
            "textbook oracle only, no application framing."
        ),
        "workplan_reference": "WP/WP-02-two-mode-motional.md (WI-1/WI-2, dispatches MCA/MCB)",
        "schema_version": 2,
        # Deterministic solve validation (no randomness); not a cached-result artefact.
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": "qutip",
        "backend_version": qutip.__version__,
        "hamiltonian": "H/hbar = i g (a-dag b-dag - a b) (two_mode_squeezing_hamiltonian)",
        "g_rate_rad_s": G_RATE,
        "fock_dim": FOCK_DIM,
        "dynamics": [
            {
                "r": float(r_grid[i]),
                "n_a": float(n_a[i]),
                "n_b": float(n_b[i]),
                "analytic_sinh2": float(analytic_sinh2[i]),
                "n_diff": float(n_diff[i]),
            }
            for i in range(N_TIMES)
        ],
        "factory": [
            {
                "z": float(z),
                "n_factory": float(n_factory[i]),
                "analytic": float(analytic_factory[i]),
            }
            for i, z in enumerate(Z_FACTORY)
        ],
        "analytic_formulas": {
            "per_mode_occupation": "sinh(g*tau)^2  (dynamics) ;  sinh(|z|)^2  (factory)",
            "difference_number": "0  (conserved su(1,1) Casimir)",
        },
        "max_numerical_vs_analytic_error": max_error,
        "tolerance": 1e-4,
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

    fig, (ax_n, ax_d) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    r_fine = np.linspace(0.0, float(r_grid[-1]), 200)
    ax_n.plot(r_fine, np.sinh(r_fine) ** 2, color="#444444", linewidth=1.0, label=r"$\sinh^2 r$")
    ax_n.scatter(
        r_grid,
        n_a,
        color="#1f77b4",
        marker="o",
        s=18,
        zorder=3,
        label=r"dynamics $\langle n_a\rangle$",
    )
    ax_n.scatter(
        z_arr,
        n_factory,
        color="#d62728",
        marker="s",
        s=22,
        zorder=3,
        label=r"factory $\langle n\rangle$",
    )
    ax_n.set_xlabel(r"squeezing $r = g\,\tau$ (or $|z|$)")
    ax_n.set_ylabel(r"per-mode occupation $\langle \hat n\rangle$")
    ax_n.set_title("Parametric gain: two-mode SU(1,1)")
    ax_n.legend(frameon=False, fontsize=8)

    ax_d.axhline(0.0, color="#444444", linewidth=1.0)
    ax_d.scatter(r_grid, n_diff, color="#2ca02c", marker="o", s=18, zorder=3)
    ax_d.set_xlabel(r"squeezing $r = g\,\tau$")
    ax_d.set_ylabel(r"difference number $\langle \hat n_a - \hat n_b\rangle$")
    ax_d.set_title("Conserved su(1,1) Casimir")
    ax_d.set_ylim(-1.0, 1.0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
