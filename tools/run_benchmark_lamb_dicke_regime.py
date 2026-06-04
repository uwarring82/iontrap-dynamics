# SPDX-License-Identifier: MIT
"""Benchmark — Lamb–Dicke regime: Debye–Waller, the classifier, and full-LD sidebands.

Tutorial-companion benchmark for the Lamb–Dicke regime helpers (WP-02 WI-5, dispatch
MCE). Two closed-form panels (compute-only, no solver):

* **Debye–Waller + regime classifier.** Sweeping the regime parameter
  ``x = η²(2n̄+1)``, the thermal carrier suppression ``debye_waller_factor`` equals the
  closed form ``e^{−x/2}`` and the truncated thermal series
  ``Σ_n p_n(n̄)·e^{−η²/2}L_n(η²)``; ``lamb_dicke_regime`` partitions the axis into
  ``deep`` / ``intermediate`` / ``beyond`` at ``x = 0.1`` and ``x = 1.0``.
* **Full Lamb–Dicke vs leading order.** The all-orders blue-sideband Rabi frequency
  ``blue_sideband_rabi_frequency_full_ld`` bends away from the leading-order
  ``|η|√(n+1)·Ω`` as the Fock level climbs — the linearised sideband breaking down.

Application-agnostic: textbook Wineland–Itano oracles only, no application framing.

Usage::

    python tools/run_benchmark_lamb_dicke_regime.py

Output::

    benchmarks/data/lamb_dicke_regime/
      report.json  — Debye–Waller closed-form vs series, regime thresholds, sideband ratios
      arrays.npz   — x, dw_closed, dw_series, n_fock, rabi_leading, rabi_full + oracles
      plot.png     — Debye–Waller vs regime parameter (banded) + sideband breakdown
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.special import eval_genlaguerre

from iontrap_dynamics.analytic import (
    LAMB_DICKE_DEEP_MAX,
    LAMB_DICKE_INTERMEDIATE_MAX,
    blue_sideband_rabi_frequency,
    blue_sideband_rabi_frequency_full_ld,
    debye_waller_factor,
    lamb_dicke_parameter,
    lamb_dicke_regime,
)
from iontrap_dynamics.conventions import CONVENTION_VERSION
from iontrap_dynamics.species import mg25_plus

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "lamb_dicke_regime"

ETA_DW = 0.3  # representative Lamb–Dicke parameter for the Debye–Waller sweep
ETA_SIDEBAND = 0.5  # larger η to make the full-LD vs leading-order breakdown visible
N_BAR_GRID = np.linspace(0.0, 55.0, 120)  # sweeps x = η²(2n̄+1) up to ≈ 10 at η = 0.3
N_FOCK = np.arange(0, 13)
OMEGA0 = 2.0 * np.pi * 1.0e5  # carrier Rabi frequency, rad·s⁻¹

PLOT_ALT_TEXT = (
    "Two panels. Left: the Debye–Waller thermal carrier-suppression factor versus the "
    "regime parameter eta squared times (2 n-bar + 1) on a log x-axis; the library values "
    "and the truncated thermal series both lie on the closed form e to the minus x over "
    "two. Vertical lines at x = 0.1 and x = 1.0 split the axis into the deep, intermediate, "
    "and beyond Lamb–Dicke regimes from the classifier. Right: the blue-sideband Rabi "
    "frequency versus initial Fock level; the leading-order square-root-of-(n+1) line "
    "(dashed) climbs without bound while the exact all-orders curve (solid) bends over and "
    "turns down — the linearised sideband breaking down as the Fock level grows."
)


def _thermal_series_dw(eta: float, n_bar: float) -> float:
    """Σ_n p_n(n̄)·e^{−η²/2}·L_n(η²) — the thermal average of the carrier matrix element."""
    n_levels = int(np.ceil(50.0 + 30.0 * n_bar))
    n = np.arange(n_levels)
    # Overflow-safe geometric weights p_n = (n̄/(n̄+1))ⁿ / (n̄+1) (the ratio stays < 1).
    ratio = n_bar / (n_bar + 1.0) if n_bar > 0.0 else 0.0
    weights = ratio**n / (n_bar + 1.0)
    laguerre = eval_genlaguerre(n, 0, eta**2)
    return float(np.sum(weights * np.exp(-(eta**2) / 2.0) * laguerre))


def _environment() -> dict[str, str]:
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # A physical η anchor, for the report (the sweep itself uses ETA_DW directly).
    eta_phys = lamb_dicke_parameter(
        k_vec=np.array([0.0, 0.0, 2.0 * np.pi / 280e-9]),
        mode_eigenvector=np.array([0.0, 0.0, 1.0]),
        ion_mass=mg25_plus().mass_kg,
        mode_frequency=2.0 * np.pi * 1.0e6,
    )

    # --- Debye–Waller sweep over x = η²(2n̄+1) (vary n̄ at fixed η) ---
    x_grid = ETA_DW**2 * (2.0 * N_BAR_GRID + 1.0)
    dw_lib = np.array(
        [
            debye_waller_factor(lamb_dicke_parameter=ETA_DW, mean_phonon_number=nb)
            for nb in N_BAR_GRID
        ]
    )
    dw_closed = np.exp(-x_grid / 2.0)
    dw_series = np.array([_thermal_series_dw(ETA_DW, nb) for nb in N_BAR_GRID])
    regimes = [
        str(lamb_dicke_regime(lamb_dicke_parameter=ETA_DW, mean_phonon_number=nb))
        for nb in N_BAR_GRID
    ]

    # --- full-LD vs leading-order blue sideband Rabi vs Fock level ---
    rabi_leading = np.array(
        [
            blue_sideband_rabi_frequency(
                carrier_rabi_frequency=OMEGA0, lamb_dicke_parameter=ETA_SIDEBAND, n_initial=int(n)
            )
            for n in N_FOCK
        ]
    )
    rabi_full = np.array(
        [
            blue_sideband_rabi_frequency_full_ld(
                carrier_rabi_frequency=OMEGA0, lamb_dicke_parameter=ETA_SIDEBAND, n_initial=int(n)
            )
            for n in N_FOCK
        ]
    )

    max_error = float(max(np.max(np.abs(dw_lib - dw_closed)), np.max(np.abs(dw_lib - dw_series))))

    print(">>> Lamb–Dicke regime — Debye–Waller, classifier, full-LD vs leading-order")
    print(f"physical η (25Mg+, 1 MHz axial, 280 nm) = {eta_phys:.4f}")
    print(f"thresholds: deep < {LAMB_DICKE_DEEP_MAX}, intermediate < {LAMB_DICKE_INTERMEDIATE_MAX}")
    print(f"{'x=η²(2n̄+1)':>12} {'DW(lib)':>10} {'e^(-x/2)':>10} {'regime':>14}")
    print("-" * 50)
    for i in range(0, len(N_BAR_GRID), max(1, len(N_BAR_GRID) // 8)):
        print(f"{x_grid[i]:>12.4f} {dw_lib[i]:>10.6f} {dw_closed[i]:>10.6f} {regimes[i]:>14}")
    print("-" * 50)
    print(f"max |DW(lib) - closed/series| = {max_error:.2e}")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        x=x_grid,
        n_bar=N_BAR_GRID,
        dw_lib=dw_lib,
        dw_closed=dw_closed,
        dw_series=dw_series,
        n_fock=N_FOCK,
        rabi_leading=rabi_leading,
        rabi_full=rabi_full,
    )

    report = {
        "scenario": "lamb_dicke_regime",
        "purpose": (
            "Lamb-Dicke regime helpers: the Debye-Waller carrier suppression "
            "debye_waller_factor equals e^{-eta^2(2 n-bar+1)/2} and the truncated thermal "
            "series; lamb_dicke_regime partitions eta^2(2 n-bar+1) into deep/intermediate/"
            "beyond at 0.1 and 1.0; the all-orders blue-sideband Rabi frequency bends away "
            "from the leading-order |eta| sqrt(n+1) Omega as the Fock level climbs. "
            "Application-agnostic: textbook Wineland-Itano oracles only."
        ),
        "workplan_reference": "WP/WP-02-two-mode-motional.md (WI-5, dispatch MCE)",
        "schema_version": 2,
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": "numpy",
        "backend_version": np.__version__,
        "eta_debye_waller": ETA_DW,
        "eta_sideband": ETA_SIDEBAND,
        "physical_eta_25mg_1mhz_280nm": float(eta_phys),
        "regime_thresholds": {
            "deep_max": LAMB_DICKE_DEEP_MAX,
            "intermediate_max": LAMB_DICKE_INTERMEDIATE_MAX,
        },
        "debye_waller_samples": [
            {
                "x": float(x_grid[i]),
                "dw_lib": float(dw_lib[i]),
                "dw_closed": float(dw_closed[i]),
                "dw_series": float(dw_series[i]),
                "regime": regimes[i],
            }
            for i in range(0, len(N_BAR_GRID), max(1, len(N_BAR_GRID) // 16))
        ],
        "sideband_samples": [
            {
                "n": int(N_FOCK[i]),
                "rabi_leading_over_omega0": float(rabi_leading[i] / OMEGA0),
                "rabi_full_over_omega0": float(rabi_full[i] / OMEGA0),
            }
            for i in range(len(N_FOCK))
        ],
        "analytic_formulas": {
            "debye_waller": "exp(-eta^2 (2 n-bar + 1) / 2)",
            "leading_sideband": "|eta| sqrt(n+1) Omega",
            "full_ld_sideband": "|eta| e^{-eta^2/2} |L_n^(1)(eta^2)| / sqrt(n+1) Omega",
        },
        "max_numerical_vs_analytic_error": max_error,
        "tolerance": 1e-9,
        "plot_alt_text": PLOT_ALT_TEXT,
        "environment": _environment(),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/report.json + arrays.npz")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0

    fig, (ax_dw, ax_sb) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax_dw.axvspan(x_grid.min(), LAMB_DICKE_DEEP_MAX, color="#2ca02c", alpha=0.08)
    ax_dw.axvspan(LAMB_DICKE_DEEP_MAX, LAMB_DICKE_INTERMEDIATE_MAX, color="#ff7f0e", alpha=0.08)
    ax_dw.axvspan(LAMB_DICKE_INTERMEDIATE_MAX, x_grid.max(), color="#d62728", alpha=0.08)
    ax_dw.axvline(LAMB_DICKE_DEEP_MAX, color="#888888", linestyle="--", linewidth=0.8)
    ax_dw.axvline(LAMB_DICKE_INTERMEDIATE_MAX, color="#888888", linestyle="--", linewidth=0.8)
    ax_dw.plot(x_grid, dw_closed, color="#444444", linewidth=1.0, label=r"$e^{-x/2}$")
    ax_dw.scatter(
        x_grid[::6], dw_lib[::6], color="#1f77b4", s=14, zorder=3, label="debye_waller_factor"
    )
    ax_dw.scatter(
        x_grid[::6],
        dw_series[::6],
        color="#d62728",
        marker="x",
        s=16,
        zorder=3,
        label="thermal series",
    )
    ax_dw.set_xscale("log")
    ax_dw.set_xlabel(r"regime parameter $x = \eta^2(2\bar n + 1)$")
    ax_dw.set_ylabel("Debye–Waller factor")
    ax_dw.set_title("deep · intermediate · beyond")
    ax_dw.legend(frameon=False, fontsize=8)

    ax_sb.plot(
        N_FOCK,
        rabi_leading / OMEGA0,
        color="#1f77b4",
        linestyle="--",
        marker="o",
        markersize=4,
        label=r"leading $|\eta|\sqrt{n+1}$",
    )
    ax_sb.plot(
        N_FOCK,
        rabi_full / OMEGA0,
        color="#d62728",
        marker="s",
        markersize=4,
        label="full Lamb–Dicke",
    )
    ax_sb.set_xlabel(r"initial Fock level $n$")
    ax_sb.set_ylabel(r"blue-sideband $\Omega_{n}/\Omega_0$")
    ax_sb.set_title(rf"Sideband breakdown ($\eta = {ETA_SIDEBAND}$)")
    ax_sb.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
