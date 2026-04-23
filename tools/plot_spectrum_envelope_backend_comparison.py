# SPDX-License-Identifier: MIT
"""Overlay the scipy and JAX envelope benchmarks in one figure.

Reads ``benchmarks/data/spectrum_envelope/report.json`` (scipy) and
``.../report_jax.json`` (JAX), fits the same ``t = alpha * d^3`` and
``m = m0 + kappa * d^2`` scaling laws to each backend independently,
and renders a three-panel log-log figure:

1. wall-clock vs dim, both backends with separate fitted curves
2. peak RSS vs dim, both backends
3. speedup ratio ``t_scipy / t_jax`` vs dim — > 1 means JAX wins

Prints a pairwise table of per-point ratios so the headline
"scipy is Nx faster than JAX at dim D" story is immediately visible.

Usage::

    python tools/plot_spectrum_envelope_backend_comparison.py

Writes ``benchmarks/data/spectrum_envelope/plot_backend_comparison.png``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "benchmarks" / "data" / "spectrum_envelope"
SCIPY_REPORT = DATA_DIR / "report.json"
JAX_REPORT = DATA_DIR / "report_jax.json"
PLOT_PATH = DATA_DIR / "plot_backend_comparison.png"


def _fit_power_law(dims: np.ndarray, values: np.ndarray, exponent: float) -> float:
    logs = np.log(values) - exponent * np.log(dims)
    return float(np.exp(np.mean(logs)))


def _fit_rss_quadratic(dims: np.ndarray, rss_bytes: np.ndarray) -> tuple[float, float]:
    x = dims.astype(np.float64) ** 2
    y = rss_bytes.astype(np.float64)
    A = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    report = json.loads(path.read_text(encoding="utf-8"))
    results = report["results"]
    dims = np.asarray([r["dim"] for r in results], dtype=np.int64)
    walls = np.asarray([r["solve_seconds"] for r in results], dtype=np.float64)
    rsss = np.asarray([r["peak_rss_bytes"] for r in results], dtype=np.float64)
    order = np.argsort(dims)
    return dims[order], walls[order], rsss[order]


def main() -> int:
    if not JAX_REPORT.exists():
        print(f"error: {JAX_REPORT.relative_to(REPO_ROOT)} not found; "
              f"run tools/run_benchmark_spectrum_envelope_jax.py first.")
        return 1

    sci_d, sci_wall, sci_rss = _load(SCIPY_REPORT)
    jax_d, jax_wall, jax_rss = _load(JAX_REPORT)

    # Fits: use d >= 500 for wall-clock (same reason as the scipy-only tool)
    # and all points for RSS.
    sci_alpha = _fit_power_law(sci_d[sci_d >= 500], sci_wall[sci_d >= 500], exponent=3.0)
    jax_alpha = _fit_power_law(jax_d[jax_d >= 500], jax_wall[jax_d >= 500], exponent=3.0)
    sci_m0, sci_kappa = _fit_rss_quadratic(sci_d, sci_rss)
    jax_m0, jax_kappa = _fit_rss_quadratic(jax_d, jax_rss)

    print("scipy  : alpha = {:.3e} s/d^3,   m0 = {:.1f} MB,  kappa ≈ {:.2f}x dtype".format(
        sci_alpha, sci_m0 / 1024**2, sci_kappa / 16.0))
    print("jax    : alpha = {:.3e} s/d^3,   m0 = {:.1f} MB,  kappa ≈ {:.2f}x dtype".format(
        jax_alpha, jax_m0 / 1024**2, jax_kappa / 16.0))
    print()

    # Pairwise ratio table
    common = sorted(set(sci_d) & set(jax_d))
    print(f"  {'dim':>8} {'scipy wall':>14} {'jax wall':>14} {'scipy/jax':>12}  "
          f"{'scipy rss':>12} {'jax rss':>12} {'rss ratio':>12}")
    for d in common:
        si = int(np.flatnonzero(sci_d == d)[0])
        ji = int(np.flatnonzero(jax_d == d)[0])
        sw, jw = sci_wall[si], jax_wall[ji]
        sr, jr = sci_rss[si], jax_rss[ji]
        print(f"  {d:>8} {sw:>13.3f}s {jw:>13.3f}s {sw / jw:>11.2f}x  "
              f"{sr / 1024**2:>10.0f} MB {jr / 1024**2:>10.0f} MB {sr / jr:>11.2f}x")

    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("matplotlib not available -- skipping plot")
        return 0

    # Extrapolation grid
    d_max_plot = max(int(sci_d.max()), int(jax_d.max())) * 2
    d_model = np.logspace(1.0, math.log10(d_max_plot), 120)
    sci_wall_model = sci_alpha * d_model**3
    jax_wall_model = jax_alpha * d_model**3
    sci_rss_model = sci_m0 + sci_kappa * d_model**2
    jax_rss_model = jax_m0 + jax_kappa * d_model**2

    fig, (ax_wall, ax_rss, ax_ratio) = plt.subplots(1, 3, figsize=(14.5, 4.4))

    sci_colour = "#1f77b4"  # blue
    jax_colour = "#d62728"  # red

    ax_wall.loglog(sci_d, sci_wall, "o", color=sci_colour, markersize=7, label="scipy (measured)", zorder=3)
    ax_wall.loglog(jax_d, jax_wall, "s", color=jax_colour, markersize=7, label="jax (measured)", zorder=3)
    ax_wall.loglog(d_model, sci_wall_model, "-", color=sci_colour, linewidth=1.0, alpha=0.8,
                   label=f"scipy fit: α={sci_alpha:.2e}", zorder=2)
    ax_wall.loglog(d_model, jax_wall_model, "-", color=jax_colour, linewidth=1.0, alpha=0.8,
                   label=f"jax fit: α={jax_alpha:.2e}", zorder=2)
    for s, lbl in [(1.0, "1 s"), (60.0, "1 min"), (3600.0, "1 hr")]:
        ax_wall.axhline(s, color="grey", linewidth=0.5, linestyle=":", alpha=0.6)
        ax_wall.text(d_model[-1] * 0.95, s, f"  {lbl}", ha="right", va="bottom",
                     fontsize=8, color="grey")
    ax_wall.set_xlabel("Hilbert dimension $d$")
    ax_wall.set_ylabel("solve wall-clock [s]")
    ax_wall.set_title("Wall-clock")
    ax_wall.grid(True, which="both", alpha=0.25)
    ax_wall.legend(loc="lower right", fontsize=8)

    ax_rss.loglog(sci_d, sci_rss / 1024**2, "o", color=sci_colour, markersize=7,
                  label="scipy (measured)", zorder=3)
    ax_rss.loglog(jax_d, jax_rss / 1024**2, "s", color=jax_colour, markersize=7,
                  label="jax (measured)", zorder=3)
    ax_rss.loglog(d_model, sci_rss_model / 1024**2, "-", color=sci_colour, linewidth=1.0, alpha=0.8,
                  label=f"scipy: κ/dtype ≈ {sci_kappa / 16:.1f}", zorder=2)
    ax_rss.loglog(d_model, jax_rss_model / 1024**2, "-", color=jax_colour, linewidth=1.0, alpha=0.8,
                  label=f"jax: κ/dtype ≈ {jax_kappa / 16:.1f}", zorder=2)
    for gb in (8, 16, 32, 64, 128):
        ax_rss.axhline(gb * 1024, color="grey", linewidth=0.5, linestyle=":", alpha=0.6)
        ax_rss.text(d_model[-1] * 0.95, gb * 1024, f"  {gb} GB", ha="right", va="bottom",
                    fontsize=8, color="grey")
    ax_rss.set_xlabel("Hilbert dimension $d$")
    ax_rss.set_ylabel("peak process RSS [MiB]")
    ax_rss.set_title("Peak RSS")
    ax_rss.grid(True, which="both", alpha=0.25)
    ax_rss.legend(loc="lower right", fontsize=8)

    # Ratio panel. Exclude dim < 100: at those sizes both calls finish in
    # << 1 ms and the ratio is dominated by import-side-effect noise
    # (scipy's first eigh pays a 7 ms first-dispatch cost that masks the
    # underlying LAPACK call). The asymptotic story is at dim >= 500.
    ratio_mask = np.asarray([d >= 100 for d in common])
    ratio_d = np.asarray(common, dtype=np.int64)[ratio_mask]
    ratio_wall = np.asarray([
        sci_wall[int(np.flatnonzero(sci_d == d)[0])]
        / jax_wall[int(np.flatnonzero(jax_d == d)[0])]
        for d in common
    ])[ratio_mask]
    ratio_rss = np.asarray([
        sci_rss[int(np.flatnonzero(sci_d == d)[0])]
        / jax_rss[int(np.flatnonzero(jax_d == d)[0])]
        for d in common
    ])[ratio_mask]
    ax_ratio.axhline(1.0, color="black", linewidth=0.8)
    ax_ratio.semilogx(ratio_d, ratio_wall, "o-", color="#2ca02c", label="wall-clock ratio")
    ax_ratio.semilogx(ratio_d, ratio_rss, "s-", color="#9467bd", label="peak-RSS ratio")
    # Asymptotic ratio from the fits (scipy α / jax α, scipy κ / jax κ)
    asym_wall = sci_alpha / jax_alpha
    asym_rss = sci_kappa / jax_kappa
    ax_ratio.axhline(asym_wall, color="#2ca02c", linewidth=0.6, linestyle="--", alpha=0.6)
    ax_ratio.axhline(asym_rss, color="#9467bd", linewidth=0.6, linestyle="--", alpha=0.6)
    ax_ratio.text(
        ratio_d[-1] * 1.1, asym_wall,
        f"  α ratio = {asym_wall:.2f}", va="center", fontsize=8, color="#2ca02c",
    )
    ax_ratio.text(
        ratio_d[-1] * 1.1, asym_rss,
        f"  κ ratio = {asym_rss:.2f}", va="center", fontsize=8, color="#9467bd",
    )
    ax_ratio.set_ylim(0.3, 1.8)
    ax_ratio.set_xlabel("Hilbert dimension $d$")
    ax_ratio.set_ylabel("scipy / jax  (>1 favours JAX)")
    ax_ratio.set_title("Backend ratio")
    ax_ratio.grid(True, which="both", alpha=0.25)
    ax_ratio.legend(fontsize=8, loc="upper right")

    # Read backend metadata for subtitle
    sci_env = json.loads(SCIPY_REPORT.read_text())["environment"]
    jax_env = json.loads(JAX_REPORT.read_text())["environment"]
    fig.suptitle(
        f"Backend comparison — scipy {sci_env['scipy']} vs "
        f"jax {jax_env['jax']} ({jax_env['jax_backend']}) on "
        f"{sci_env['platform']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {PLOT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
