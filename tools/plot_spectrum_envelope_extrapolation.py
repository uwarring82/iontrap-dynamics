# SPDX-License-Identifier: MIT
"""Fit scaling laws to the AAH envelope data and overlay RAM-limit bands.

Reads ``benchmarks/data/spectrum_envelope/report.json`` (produced by
``run_benchmark_spectrum_envelope.py``), fits the two expected scaling
laws

    t_solve(d) = alpha * d ** 3
    peak_rss(d) = rss_baseline + kappa * d ** 2

to the measured points, and renders a two-panel log-log figure that
extends each curve past the measurement range with the fitted model,
overlays horizontal bands at canonical wall-clock thresholds
(1 s / 10 s / 1 min / 10 min / 1 hr) and consumer-hardware RAM
tiers (8 / 16 / 32 / 64 / 128 GB), and annotates the crossover
dimension at which dense ``eigh`` saturates each resource.

Also prints a user-facing table:
"on this hardware / RAM tier, you can exact-diagonalize up to this
system size before peak RSS exceeds the budget."

Usage::

    python tools/plot_spectrum_envelope_extrapolation.py

Writes ``benchmarks/data/spectrum_envelope/plot_extrapolation.png``
and ``benchmarks/data/spectrum_envelope/envelope_table.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "benchmarks" / "data" / "spectrum_envelope"
REPORT_PATH = DATA_DIR / "report.json"
PLOT_PATH = DATA_DIR / "plot_extrapolation.png"
TABLE_PATH = DATA_DIR / "envelope_table.json"

WALL_THRESHOLDS_S = {
    "1 s": 1.0,
    "10 s": 10.0,
    "1 min": 60.0,
    "10 min": 600.0,
    "1 hr": 3600.0,
}

RAM_TIERS_GB = {
    "8 GB (budget laptop)": 8,
    "16 GB (standard laptop)": 16,
    "32 GB (workstation)": 32,
    "64 GB (high-end WS)": 64,
    "128 GB (lab server)": 128,
}

N_IONS_COLOURS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728", 5: "#9467bd"}


def _fit_power_law(dims: np.ndarray, values: np.ndarray, exponent: float) -> float:
    """Return the prefactor alpha of ``values = alpha * d**exponent`` via a
    log-space least-squares fit on ``log(alpha) = log(value) - exponent * log(d)``.
    """
    logs = np.log(values) - exponent * np.log(dims)
    return float(np.exp(np.mean(logs)))


def _fit_rss_quadratic(dims: np.ndarray, rss_bytes: np.ndarray) -> tuple[float, float]:
    """Fit ``rss = baseline + kappa * d**2`` via ordinary least squares on
    the linear model in ``(1, d**2)``. Returns ``(baseline_bytes, kappa)``.
    """
    x = dims.astype(np.float64) ** 2
    y = rss_bytes.astype(np.float64)
    A = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _invert_rss_for_dim(baseline_bytes: float, kappa: float, rss_budget_bytes: int) -> float:
    """Return the largest dim d such that fitted RSS <= budget."""
    if rss_budget_bytes <= baseline_bytes:
        return 0.0
    return math.sqrt((rss_budget_bytes - baseline_bytes) / kappa)


def _invert_wall_for_dim(alpha: float, wall_budget_s: float) -> float:
    return (wall_budget_s / alpha) ** (1.0 / 3.0)


def _fmt_dim(d: float) -> str:
    return f"{int(round(d)):,}"


def main() -> int:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    results = report["results"]

    dims = np.asarray([r["dim"] for r in results], dtype=np.int64)
    walls = np.asarray([r["solve_seconds"] for r in results], dtype=np.float64)
    rsss = np.asarray([r["peak_rss_bytes"] for r in results], dtype=np.float64)
    n_ions_arr = np.asarray([r["n_ions"] for r in results], dtype=np.int64)

    # Wall-clock fit: clip to the scaling regime (dim >= 500); below that the
    # baseline import / assemble cost dominates and the cubic model bends.
    wall_mask = dims >= 500
    alpha = _fit_power_law(dims[wall_mask], walls[wall_mask], exponent=3.0)

    # RSS fit uses all points — the baseline parameter absorbs the import
    # footprint directly.
    rss_baseline, kappa = _fit_rss_quadratic(dims, rsss)

    print("scaling-law fits")
    print(f"  wall-clock  t = alpha * d^3     with alpha = {alpha:.3e} s")
    print(f"  peak RSS    m = m0 + kappa*d^2  with m0 = {rss_baseline / 1024**2:.1f} MB, "
          f"kappa = {kappa:.2f} B/entry (≈ {kappa / 16:.2f}× the matrix dtype)")
    print()

    envelope: dict[str, dict[str, object]] = {}
    print(f"  {'RAM tier':<26} {'max dim (RSS)':>16} {'wall at that dim':>18}")
    for label, gb in RAM_TIERS_GB.items():
        budget_bytes = gb * 1024**3
        d_max = _invert_rss_for_dim(rss_baseline, kappa, budget_bytes)
        wall_at = alpha * d_max**3
        # Translate the RSS-limited dim back into the Clos 2016 reproduction
        # parameterisation: dim = 2 * (n_c + 1)**N, so the achievable n_c at
        # fixed N is n_c <= (d_max / 2) ** (1/N) - 1.
        per_n_max_cutoff: dict[int, int] = {}
        for n_ions in range(1, 6):
            nc_float = (d_max / 2.0) ** (1.0 / n_ions) - 1.0
            per_n_max_cutoff[n_ions] = max(0, int(math.floor(nc_float)))
        envelope[label] = {
            "ram_gb": gb,
            "max_dim": int(round(d_max)),
            "wall_at_max_dim_s": wall_at,
            "max_n_c_per_n_ions": per_n_max_cutoff,
        }
        print(f"  {label:<26} {_fmt_dim(d_max):>16} {wall_at:>14.1f} s")

    print()
    header = (
        f"  {'RAM tier':<26} "
        + " ".join(f"{'N=' + str(n):>6}" for n in range(1, 6))
    )
    print(header)
    for label, entry in envelope.items():
        nc = entry["max_n_c_per_n_ions"]  # type: ignore[index]
        row = f"  {label:<26} " + " ".join(f"{nc[n]:>6}" for n in range(1, 6))  # type: ignore[index]
        print(row)

    # Also report the wall-clock-limited dim at each threshold.
    print()
    print(f"  {'wall threshold':<14} {'max dim':>12}")
    wall_table: dict[str, dict[str, float | int]] = {}
    for label, seconds in WALL_THRESHOLDS_S.items():
        d_wall = _invert_wall_for_dim(alpha, seconds)
        wall_table[label] = {
            "budget_s": seconds,
            "max_dim": int(round(d_wall)),
        }
        print(f"  {label:<14} {_fmt_dim(d_wall):>12}")

    TABLE_PATH.write_text(
        json.dumps(
            {
                "fit": {
                    "wall_clock_alpha_s_per_d3": alpha,
                    "rss_baseline_bytes": rss_baseline,
                    "rss_kappa_bytes_per_d2": kappa,
                    "rss_workspace_multiplier_vs_matrix": kappa / 16.0,
                    "wall_fit_cutoff_dim": 500,
                },
                "ram_tiers": envelope,
                "wall_thresholds": wall_table,
                "environment": report["environment"],
            },
            indent=2, sort_keys=True, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {TABLE_PATH.relative_to(REPO_ROOT)}")

    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("matplotlib not available -- skipping plot")
        return 0

    # Extrapolation grid out to where even a 128 GB server saturates RSS
    d_model = np.logspace(1.0, math.log10(envelope["128 GB (lab server)"]["max_dim"] * 1.4), 200)
    wall_model = alpha * d_model**3
    rss_model = rss_baseline + kappa * d_model**2

    fig, (ax_wall, ax_rss) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    # Measured points, coloured by N
    for n_ions, colour in N_IONS_COLOURS.items():
        mask = n_ions_arr == n_ions
        if not mask.any():
            continue
        ax_wall.loglog(dims[mask], walls[mask], "o", color=colour, label=f"N={n_ions}", zorder=3)
        ax_rss.loglog(dims[mask], rsss[mask] / 1024**2, "o", color=colour, label=f"N={n_ions}", zorder=3)

    # Fitted extrapolation
    ax_wall.loglog(
        d_model, wall_model, "-", color="black", linewidth=1.4,
        label=r"fit $t = \alpha\,d^{3}$", zorder=2,
    )
    ax_rss.loglog(
        d_model, rss_model / 1024**2, "-", color="black", linewidth=1.4,
        label=r"fit $m = m_{0} + \kappa\,d^{2}$", zorder=2,
    )

    # Wall-clock threshold bands
    for label, seconds in WALL_THRESHOLDS_S.items():
        ax_wall.axhline(seconds, color="grey", linewidth=0.6, linestyle=":", alpha=0.8)
        ax_wall.text(
            d_model[-1] * 0.95, seconds, f"  {label}",
            ha="right", va="bottom", fontsize=8, color="grey",
        )
        # Mark the crossover dim
        d_cross = _invert_wall_for_dim(alpha, seconds)
        if d_model[0] < d_cross < d_model[-1]:
            ax_wall.axvline(d_cross, color="grey", linewidth=0.4, linestyle=":", alpha=0.6)

    # RAM-tier bands
    for label, gb in RAM_TIERS_GB.items():
        ax_rss.axhline(gb * 1024, color="grey", linewidth=0.6, linestyle=":", alpha=0.8)
        ax_rss.text(
            d_model[-1] * 0.95, gb * 1024, f"  {label}",
            ha="right", va="bottom", fontsize=8, color="grey",
        )
        d_cross = _invert_rss_for_dim(rss_baseline, kappa, gb * 1024**3)
        if d_model[0] < d_cross < d_model[-1]:
            ax_rss.axvline(d_cross, color="grey", linewidth=0.4, linestyle=":", alpha=0.6)

    # Highlight the measured-range boundary
    d_measured_max = float(dims.max())
    for ax in (ax_wall, ax_rss):
        ax.axvspan(d_model[0], d_measured_max, facecolor="#f0f0f0", alpha=0.55, zorder=0)
        ax.text(
            math.sqrt(d_model[0] * d_measured_max),
            ax.get_ylim()[1] * 0.5 if ax is ax_wall else 1e4,
            "measured", ha="center", va="top", fontsize=9, color="#555555", zorder=1,
        )

    ax_wall.set_xlabel("Hilbert dimension $d$")
    ax_wall.set_ylabel("solve_spectrum wall-clock [s]")
    ax_wall.set_title(f"Wall-clock: α = {alpha:.2e} s (cubic)")
    ax_wall.grid(True, which="both", alpha=0.25)
    ax_wall.legend(title="N ions", loc="lower right", fontsize=8)

    ax_rss.set_xlabel("Hilbert dimension $d$")
    ax_rss.set_ylabel("peak process RSS [MiB]")
    ax_rss.set_title(
        f"Peak RSS: m$_0$ = {rss_baseline / 1024**2:.0f} MB, κ ≈ "
        f"{kappa / 16:.1f}× matrix dtype"
    )
    ax_rss.grid(True, which="both", alpha=0.25)
    ax_rss.legend(title="N ions", loc="lower right", fontsize=8)

    fig.suptitle(
        "Exact-diag envelope — measured points + fitted extrapolation "
        f"({report['environment']['platform']})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOT_PATH.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
