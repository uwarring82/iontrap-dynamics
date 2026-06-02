# SPDX-License-Identifier: MIT
"""Benchmark — common-mode channel: difference-observable variance vs correlation.

WP-01 §7 row 6 (dispatch EDE). A compute-only, deterministic demonstration of
the correlated shared-latent phase channel
(:class:`iontrap_dynamics.CommonModePhase`): two subsystems receive a phase
offset that is shared, to a tunable degree ``c = correlation ∈ [0, 1]``. The
channel draws ``offset_i = √c · ξ_shared + √(1 − c) · ξ_i`` with
``ξ_shared, ξ_i ~ N(0, σ²)``, so the *marginal* per-subsystem variance is ``σ²``
at every ``c``, but the **difference observable** ``offset_0 − offset_1`` has
variance::

    Var(offset_0 − offset_1) = 2 σ² (1 − c)

At ``c = 0`` the offsets are independent and the difference jitter is the full
``2 σ²`` (incoherent per-drive sum); at ``c = 1`` the shared latent cancels
exactly and the difference variance is ``0`` — common-mode rejection. The
variance is monotone decreasing in ``c`` between the endpoints.

The companion fact, demonstrated with
:func:`iontrap_dynamics.perturb_common_mode` on two
:class:`~iontrap_dynamics.drives.DriveConfig`\\s, is that at ``c = 1`` the phase
*difference* between the two perturbed drives is invariant (it equals the
unperturbed difference, shot by shot): the shared offset cancels in the
difference.

Compute-only and deterministic — the sampling uses a fixed
``np.random.default_rng(seed)`` and there is no solver trajectory — so this is a
repo-precedent compute benchmark (``report.json`` + ``arrays.npz`` +
``plot.png``), not a solve-based cache artefact. The binding oracle assertion
lives in ``tests/regression/analytic/test_common_mode_rejection.py``.

Application-agnostic: textbook oracle only (variance of a difference of
correlated Gaussians), no application framing.

Usage::

    python tools/run_benchmark_common_mode.py

Output::

    benchmarks/data/common_mode/
      report.json  — per-c difference variance, the analytic oracle, max error,
                      provenance, and the c=1 phase-difference cancellation check
      arrays.npz   — correlation, difference_variance, analytic_difference_variance
      plot.png     — difference variance vs correlation (down to 0 at c=1)
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import qutip

from iontrap_dynamics import CommonModePhase, perturb_common_mode
from iontrap_dynamics.conventions import CONVENTION_VERSION
from iontrap_dynamics.drives import DriveConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "common_mode"

SIGMA_RAD = 0.3
SHOTS = 4_000_000
SEED = 20260602
CORRELATION_GRID = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)

# Tolerance for the sampled interior points (0 < c < 1). The endpoints c=0 and
# c=1 are checked separately: c=1 cancels exactly (variance == 0.0).
TOLERANCE = 5e-3

PLOT_ALT_TEXT = (
    "Plot of the difference-observable variance Var(offset_0 - offset_1) versus "
    "the correlation parameter c for a two-subsystem common-mode phase channel. "
    "The analytic reference is the line 2 sigma squared times (1 - c): it starts "
    "at 2 sigma squared at c = 0 (independent per-drive jitter) and falls "
    "monotonically to exactly 0 at c = 1 (common-mode rejection). The sampled "
    "points lie on the line."
)


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


def _phase_difference_invariance_at_c1() -> dict[str, float]:
    """Demonstrate that at c=1 the perturbed phase difference is invariant.

    Builds two :class:`DriveConfig`\\s with distinct initial phases, applies the
    fully-correlated channel (``correlation=1``), and reports the maximum
    absolute deviation of the perturbed phase difference from the unperturbed
    phase difference over all shots. With a shared offset this is zero (up to
    floating-point round-off).
    """
    k = np.array([0.0, 0.0, 1.0e7], dtype=np.float64)
    drive_a = DriveConfig(k_vector_m_inv=k, carrier_rabi_frequency_rad_s=1.0e6, phase_rad=0.2)
    drive_b = DriveConfig(k_vector_m_inv=k, carrier_rabi_frequency_rad_s=1.0e6, phase_rad=-0.5)
    base_diff = drive_a.phase_rad - drive_b.phase_rad
    spec = CommonModePhase(sigma_rad=SIGMA_RAD, correlation=1.0)
    perturbed = perturb_common_mode([drive_a, drive_b], spec, shots=512, seed=SEED)
    max_dev = 0.0
    for shot in perturbed:
        diff = shot[0].phase_rad - shot[1].phase_rad
        max_dev = max(max_dev, abs(diff - base_diff))
    return {
        "unperturbed_phase_difference_rad": float(base_diff),
        "max_perturbed_phase_difference_deviation_rad": float(max_dev),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    corr = np.array(CORRELATION_GRID, dtype=np.float64)
    measured_var = np.empty(len(CORRELATION_GRID), dtype=np.float64)
    for idx, c in enumerate(CORRELATION_GRID):
        spec = CommonModePhase(sigma_rad=SIGMA_RAD, correlation=float(c))
        offsets = spec.sample_offsets(n_subsystems=2, shots=SHOTS, rng=rng)
        difference = offsets[:, 0] - offsets[:, 1]
        measured_var[idx] = float(np.var(difference))

    analytic_var = 2.0 * SIGMA_RAD**2 * (1.0 - corr)

    # The c=1 difference variance is MEASURED, not enforced: at c=1 the shared
    # latent is identical across both subsystems, so offset_0 - offset_1 is
    # identically zero — a broken sample_offsets(c=1) would surface here rather
    # than be masked by an overwrite. The error metric spans all c (the c=1
    # cancellation included), so the artefact itself proves the rejection.
    c1_index = int(np.argmin(np.abs(corr - 1.0)))
    max_error = float(np.max(np.abs(measured_var - analytic_var)))

    phase_check = _phase_difference_invariance_at_c1()

    print(">>> Common-mode channel benchmark — difference variance vs correlation")
    print(f"sigma_rad = {SIGMA_RAD}   shots = {SHOTS}   seed = {SEED}")
    print(f"{'c':>6} {'Var_measured':>16} {'2 sigma^2 (1-c)':>18} {'|err|':>12}")
    print("-" * 56)
    for i, c in enumerate(CORRELATION_GRID):
        print(
            f"{c:>6.2f} {measured_var[i]:>16.8f} {analytic_var[i]:>18.8f} "
            f"{abs(measured_var[i] - analytic_var[i]):>12.2e}"
        )
    print("-" * 56)
    print(f"max |measured - analytic| (all c, sampled) = {max_error:.2e}")
    print(f"c=1 difference variance (measured) = {measured_var[c1_index]:.8f}")
    print(
        "c=1 perturb_common_mode phase-difference max deviation = "
        f"{phase_check['max_perturbed_phase_difference_deviation_rad']:.2e}"
    )

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        correlation=corr,
        difference_variance=measured_var,
        analytic_difference_variance=analytic_var,
    )

    report = {
        "scenario": "common_mode",
        "purpose": (
            "Common-mode (shared-latent) phase channel: the difference-observable "
            "variance Var(offset_0 - offset_1) of a two-subsystem correlated "
            "dephasing channel as a function of the correlation c in [0, 1]. At "
            "c=0 the offsets are independent (variance 2 sigma^2); at c=1 the "
            "shared latent cancels exactly (variance 0, common-mode rejection); "
            "monotone decreasing in between. Also demonstrates via "
            "perturb_common_mode that at c=1 the phase difference between two "
            "perturbed drives is invariant. Compute-only and deterministic "
            "(fixed rng seed, no solver trajectory). Application-agnostic: "
            "textbook variance-of-a-difference oracle only, no application framing."
        ),
        "workplan_reference": "WP/WP-01-estimation-darwinism.md (section 7 row 6, dispatch EDE)",
        "schema_version": 2,
        # Compute-only benchmark: no solve() trajectory, so no canonical cache
        # request hash. Carried as null for schema parity with solve-based
        # demo_report.json artefacts.
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": "qutip",
        "backend_version": qutip.__version__,
        "sigma_rad": SIGMA_RAD,
        "shots": SHOTS,
        "seed": SEED,
        "correlation_grid": list(CORRELATION_GRID),
        "results": [
            {
                "correlation": float(corr[i]),
                "difference_variance": float(measured_var[i]),
                "analytic_difference_variance": float(analytic_var[i]),
            }
            for i in range(len(CORRELATION_GRID))
        ],
        "phase_difference_invariance_at_c1": phase_check,
        "c1_difference_variance_measured": float(measured_var[c1_index]),
        "analytic_formulas": {
            "difference_variance": "2 sigma^2 (1 - c)",
        },
        "max_numerical_vs_analytic_error": max_error,
        "tolerance": TOLERANCE,
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

    c_dense = np.linspace(0.0, 1.0, 200)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(
        c_dense,
        2.0 * SIGMA_RAD**2 * (1.0 - c_dense),
        color="#1f77b4",
        linewidth=1.0,
        label=r"analytic: $2\sigma^2(1-c)$",
    )
    ax.scatter(
        corr,
        measured_var,
        color="#d62728",
        marker="o",
        zorder=3,
        label="measured (sampled)",
    )
    ax.axhline(0.0, color="#888888", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"correlation $c$")
    ax.set_ylabel(r"difference variance $\mathrm{Var}(\delta_0-\delta_1)$ [rad$^2$]")
    ax.set_title("Common-mode rejection: difference variance vs correlation")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
