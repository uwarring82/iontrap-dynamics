# SPDX-License-Identifier: MIT
"""Phase 2 β.4.5 — cross-backend benchmark for time-dependent
Hamiltonians at scale.

Dispatch YY. Closes the β.4 track per
``docs/phase-2-jax-time-dep-design.md`` §5. For each of the five
time-dependent Hamiltonian builders on the JAX backend (the four
structured detuning builders + modulated_carrier with a Gaussian
envelope), runs the QuTiP and JAX backends side-by-side at

- Hilbert-space dim ≥ 100
- trajectory length ≥ 5000 steps

and reports (a) cross-backend expectation-array agreement (max
absolute delta over all observables + all time indices) and
(b) wall-clock per backend (minimum of N_REPEATS to damp OS
jitter).

The 5000-step / dim ≥ 100 regime is deliberately outside the
library-typical scale probed by Dispatches X / Y / OO (which
found QuTiP 5 near the floor at dim ≤ 60). β.4.5's headline
question is whether the Phase 2 JAX commitment produces
*measurable* wall-clock wins at the scale where users would
actually reach for it, or whether the Phase 2 value is
positioning / capability / future autograd (per the design note
§10 branch analysis).

Usage::

    python tools/run_benchmark_jax_timedep.py

Requires :mod:`matplotlib`, :mod:`jax`, :mod:`dynamiqs`; prints
a summary table and writes a JSON report.

Output::

    benchmarks/data/jax_timedep/
      report.json  — per-scenario: qutip wall-clock, jax wall-
                     clock, speedup ratio, max cross-backend delta,
                     environment metadata
      plot.png     — two-panel bar chart: wall-clock (left panel),
                     speedup ratio (right panel)
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import qutip

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import (
    detuned_blue_sideband_hamiltonian,
    detuned_carrier_hamiltonian,
    detuned_ms_gate_hamiltonian,
    detuned_red_sideband_hamiltonian,
    modulated_carrier_hamiltonian,
)
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import number, spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "jax_timedep"

N_STEPS = 5000
N_REPEATS = 3
RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)


def _build_single_ion(fock: int) -> tuple[HilbertSpace, DriveConfig]:
    mode = ModeConfig(
        label="axial",
        frequency_rad_s=MODE_FREQ_RAD_S,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"axial": fock})
    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, WAVENUMBER_M_INV],
        carrier_rabi_frequency_rad_s=RABI_RAD_S,
        phase_rad=0.0,
    )
    return hilbert, drive


def _build_two_ion(fock: int) -> tuple[HilbertSpace, DriveConfig]:
    com_mode = ModeConfig(
        label="com",
        frequency_rad_s=MODE_FREQ_RAD_S,
        eigenvector_per_ion=np.array(
            [
                [0.0, 0.0, 1.0 / np.sqrt(2.0)],
                [0.0, 0.0, 1.0 / np.sqrt(2.0)],
            ]
        ),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=2, modes=(com_mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"com": fock})
    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, WAVENUMBER_M_INV],
        carrier_rabi_frequency_rad_s=RABI_RAD_S,
        phase_rad=0.0,
    )
    return hilbert, drive


def _time_solve(
    *,
    hamiltonian: object,
    hilbert: HilbertSpace,
    psi_0: qutip.Qobj,
    tlist: np.ndarray,
    observables: list,
    backend: str,
) -> tuple[float, dict[str, np.ndarray]]:
    """Run ``solve`` N_REPEATS times and return (min_elapsed, expectations)."""
    elapsed = float("inf")
    last_exps: dict[str, np.ndarray] = {}
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        res = solve(
            hilbert=hilbert,
            hamiltonian=hamiltonian,
            initial_state=psi_0,
            times=tlist,
            observables=observables,
            storage_mode=StorageMode.OMITTED,
            backend=backend,
        )
        elapsed = min(elapsed, time.perf_counter() - t0)
        last_exps = dict(res.expectations)
    return elapsed, last_exps


def _environment() -> dict[str, str]:
    import scipy

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "qutip": qutip.__version__,
    }
    try:
        import dynamiqs
        import jax

        env["jax"] = jax.__version__
        env["dynamiqs"] = dynamiqs.__version__
    except ImportError:
        env["jax"] = "missing"
        env["dynamiqs"] = "missing"
    return env


def _max_expect_delta(
    exps_a: dict[str, np.ndarray],
    exps_b: dict[str, np.ndarray],
) -> float:
    """Return the maximum absolute expectation-array delta across all
    observables and all time indices."""
    return max(float(np.max(np.abs(exps_a[key] - exps_b[key]))) for key in exps_a)


def _run_scenario(
    *,
    label: str,
    builder_name: str,
    hilbert: HilbertSpace,
    h_qutip: object,
    h_jax: object,
    psi_0: qutip.Qobj,
    tlist: np.ndarray,
    observables: list,
) -> dict[str, object]:
    hilbert_dim = int(hilbert.total_dim)

    t_qutip, exps_qutip = _time_solve(
        hamiltonian=h_qutip,
        hilbert=hilbert,
        psi_0=psi_0,
        tlist=tlist,
        observables=observables,
        backend="qutip",
    )
    t_jax, exps_jax = _time_solve(
        hamiltonian=h_jax,
        hilbert=hilbert,
        psi_0=psi_0,
        tlist=tlist,
        observables=observables,
        backend="jax",
    )
    delta = _max_expect_delta(exps_qutip, exps_jax)
    ratio = t_qutip / t_jax if t_jax > 0 else float("inf")
    return {
        "scenario": label,
        "builder": builder_name,
        "hilbert_dim": hilbert_dim,
        "qutip_seconds": t_qutip,
        "jax_seconds": t_jax,
        "qutip_over_jax": ratio,
        "max_expectation_delta": delta,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Duration covers ~2 generalised-Rabi periods at Ω = 1 MHz to hit
    # enough dynamics; 5000 steps → ~2 ns resolution which is generous
    # for both backends' adaptive steppers.
    rabi_period = 2 * np.pi / RABI_RAD_S
    tlist = np.linspace(0.0, 8 * rabi_period, N_STEPS)

    print(">>> running JAX-vs-QuTiP time-dependent benchmark")
    print(f"    N_STEPS = {N_STEPS}, N_REPEATS = {N_REPEATS} (min of repeats)")
    print(f"    Ω / 2π = {RABI_OVER_2PI_MHZ} MHz, duration = 8·T_Ω")
    print()
    print(
        f"{'scenario':<30} {'dim':>5} {'qutip [s]':>11} "
        f"{'jax [s]':>11} {'qutip/jax':>10} {'delta':>10}"
    )
    print("-" * 80)

    results: list[dict[str, object]] = []

    # -- Scenario 1: detuned carrier (single-ion, Fock=50 → dim 100).
    fock = 50
    hilbert, drive = _build_single_ion(fock)
    drive_det = DriveConfig(
        k_vector_m_inv=drive.k_vector_m_inv,
        carrier_rabi_frequency_rad_s=drive.carrier_rabi_frequency_rad_s,
        detuning_rad_s=2 * np.pi * 0.5e6,
        phase_rad=drive.phase_rad,
    )
    psi_0 = qutip.tensor(spin_down(), qutip.basis(fock, 0))
    obs = [spin_z(hilbert, 0)]
    h_q = detuned_carrier_hamiltonian(hilbert, drive_det, ion_index=0, backend="qutip")
    h_j = detuned_carrier_hamiltonian(hilbert, drive_det, ion_index=0, backend="jax")
    r = _run_scenario(
        label=f"detuned_carrier_fock{fock}",
        builder_name="detuned_carrier_hamiltonian",
        hilbert=hilbert,
        h_qutip=h_q,
        h_jax=h_j,
        psi_0=psi_0,
        tlist=tlist,
        observables=obs,
    )
    results.append(r)
    print(
        f"{r['scenario']:<30} {r['hilbert_dim']:>5d} "
        f"{r['qutip_seconds']:>11.3f} {r['jax_seconds']:>11.3f} "
        f"{r['qutip_over_jax']:>9.2f}× {r['max_expectation_delta']:>9.1e}"
    )

    # -- Scenarios 2 / 3: detuned RSB / BSB (single-ion, Fock=50 → dim 100).
    for name, builder in [
        ("detuned_rsb", detuned_red_sideband_hamiltonian),
        ("detuned_bsb", detuned_blue_sideband_hamiltonian),
    ]:
        hilbert, drive = _build_single_ion(fock)
        psi_0 = qutip.tensor(spin_down(), qutip.basis(fock, 1))
        obs = [spin_z(hilbert, 0), number(hilbert, "axial")]
        delta = 2 * np.pi * 0.3e6
        h_q = builder(
            hilbert,
            drive,
            "axial",
            ion_index=0,
            detuning_rad_s=delta,
            backend="qutip",
        )
        h_j = builder(
            hilbert,
            drive,
            "axial",
            ion_index=0,
            detuning_rad_s=delta,
            backend="jax",
        )
        r = _run_scenario(
            label=f"{name}_fock{fock}",
            builder_name=builder.__name__,
            hilbert=hilbert,
            h_qutip=h_q,
            h_jax=h_j,
            psi_0=psi_0,
            tlist=tlist,
            observables=obs,
        )
        results.append(r)
        print(
            f"{r['scenario']:<30} {r['hilbert_dim']:>5d} "
            f"{r['qutip_seconds']:>11.3f} {r['jax_seconds']:>11.3f} "
            f"{r['qutip_over_jax']:>9.2f}× {r['max_expectation_delta']:>9.1e}"
        )

    # -- Scenario 4: detuned MS gate (two-ion, Fock=25 → dim 100).
    # Uses weaker Ω/2π = 100 kHz (matches the MS integration test
    # fixture) so the coherent displacement over 8 carrier-Rabi periods
    # stays within the Fock-25 truncation — at the reference Ω/2π of
    # 1 MHz with δ/2π = 30 kHz the MS mode saturates quickly. This
    # scenario therefore measures the solver cost at a physically
    # meaningful MS drive, not at the carrier-scenario drive strength.
    fock_ms = 25
    hilbert, _ = _build_two_ion(fock_ms)
    ms_drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, WAVENUMBER_M_INV],
        carrier_rabi_frequency_rad_s=2 * np.pi * 100e3,
        phase_rad=0.0,
    )
    psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(fock_ms, 0))
    obs = [spin_z(hilbert, 0), spin_z(hilbert, 1), number(hilbert, "com")]
    delta_ms = 2 * np.pi * 30e3
    h_q = detuned_ms_gate_hamiltonian(
        hilbert,
        ms_drive,
        "com",
        ion_indices=(0, 1),
        detuning_rad_s=delta_ms,
        backend="qutip",
    )
    h_j = detuned_ms_gate_hamiltonian(
        hilbert,
        ms_drive,
        "com",
        ion_indices=(0, 1),
        detuning_rad_s=delta_ms,
        backend="jax",
    )
    r = _run_scenario(
        label=f"detuned_ms_fock{fock_ms}",
        builder_name="detuned_ms_gate_hamiltonian",
        hilbert=hilbert,
        h_qutip=h_q,
        h_jax=h_j,
        psi_0=psi_0,
        tlist=tlist,
        observables=obs,
    )
    results.append(r)
    print(
        f"{r['scenario']:<30} {r['hilbert_dim']:>5d} "
        f"{r['qutip_seconds']:>11.3f} {r['jax_seconds']:>11.3f} "
        f"{r['qutip_over_jax']:>9.2f}× {r['max_expectation_delta']:>9.1e}"
    )

    # -- Scenario 5: modulated carrier with Gaussian envelope (single-ion, dim 100).
    hilbert, drive = _build_single_ion(fock)
    # On-resonance for modulated carrier.
    drive_zero = DriveConfig(
        k_vector_m_inv=drive.k_vector_m_inv,
        carrier_rabi_frequency_rad_s=drive.carrier_rabi_frequency_rad_s,
        detuning_rad_s=0.0,
        phase_rad=drive.phase_rad,
    )
    psi_0 = qutip.tensor(spin_down(), qutip.basis(fock, 0))
    obs = [spin_z(hilbert, 0)]
    t0 = 4 * rabi_period  # centre of trajectory
    sigma = 0.5 * rabi_period

    def env_np(t: float) -> float:
        return math.exp(-0.5 * ((t - t0) / sigma) ** 2)

    import jax.numpy as jnp

    def env_jax(t: float) -> object:
        return jnp.exp(-0.5 * ((t - t0) / sigma) ** 2)

    h_q = modulated_carrier_hamiltonian(
        hilbert,
        drive_zero,
        ion_index=0,
        envelope=env_np,
        backend="qutip",
    )
    h_j = modulated_carrier_hamiltonian(
        hilbert,
        drive_zero,
        ion_index=0,
        envelope=env_np,
        envelope_jax=env_jax,
        backend="jax",
    )
    r = _run_scenario(
        label=f"modulated_carrier_fock{fock}",
        builder_name="modulated_carrier_hamiltonian",
        hilbert=hilbert,
        h_qutip=h_q,
        h_jax=h_j,
        psi_0=psi_0,
        tlist=tlist,
        observables=obs,
    )
    results.append(r)
    print(
        f"{r['scenario']:<30} {r['hilbert_dim']:>5d} "
        f"{r['qutip_seconds']:>11.3f} {r['jax_seconds']:>11.3f} "
        f"{r['qutip_over_jax']:>9.2f}× {r['max_expectation_delta']:>9.1e}"
    )

    mean_ratio = float(np.mean([float(r["qutip_over_jax"]) for r in results]))
    max_delta = float(np.max([float(r["max_expectation_delta"]) for r in results]))
    print("-" * 80)
    print(f"mean speedup (qutip/jax): {mean_ratio:.2f}×")
    print(f"max cross-backend delta:   {max_delta:.1e}")

    report = {
        "scenario": "jax_timedep_cross_backend",
        "purpose": (
            "Phase 2 β.4.5 cross-backend benchmark — time-dependent "
            "Hamiltonians across all five β.4 builders, at Hilbert "
            "dim ≥ 100 and trajectory length ≥ 5000 steps. Establishes "
            "where (if anywhere) the JAX backend measurably outperforms "
            "QuTiP 5 at scale, and confirms cross-backend numeric "
            "equivalence at the 1e-3 design-target tolerance. Closes "
            "the β.4 track per docs/phase-2-jax-time-dep-design.md §5."
        ),
        "workplan_reference": (
            "WORKPLAN_v0.3.md §5.3 (β.4 as v0.3.x follow-up); "
            "phase-2-jax-time-dep-design.md §5 staging"
        ),
        "n_steps": N_STEPS,
        "n_repeats": N_REPEATS,
        "results": results,
        "mean_qutip_over_jax": mean_ratio,
        "max_cross_backend_delta": max_delta,
        "environment": _environment(),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/report.json")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0

    labels = [str(r["scenario"]) for r in results]
    qutip_times = [float(r["qutip_seconds"]) for r in results]
    jax_times = [float(r["jax_seconds"]) for r in results]
    ratios = [float(r["qutip_over_jax"]) for r in results]

    fig, (ax_time, ax_ratio) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    x = np.arange(len(labels))
    width = 0.35
    ax_time.bar(x - width / 2, qutip_times, width, color="#1f77b4", label="QuTiP")
    ax_time.bar(x + width / 2, jax_times, width, color="#2ca02c", label="JAX")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_time.set_ylabel("wall-clock [s], min of 3 repeats")
    ax_time.set_title(f"β.4.5 — {N_STEPS} steps, dim ≥ 100")
    ax_time.legend()

    colours = ["#2ca02c" if r >= 1.0 else "#d62728" for r in ratios]
    ax_ratio.bar(x, ratios, color=colours)
    ax_ratio.axhline(1.0, color="grey", linewidth=0.5, linestyle=":")
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_ratio.set_ylabel("qutip / jax wall-clock ratio")
    ax_ratio.set_title(f"speedup — mean {mean_ratio:.2f}×")
    for i, ratio in enumerate(ratios):
        ax_ratio.text(i, ratio + 0.02, f"{ratio:.2f}×", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
