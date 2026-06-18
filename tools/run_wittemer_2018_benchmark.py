# SPDX-License-Identifier: MIT
"""Benchmark — Wittemer 2018 single-mode non-Markovianity (dispatch ND4).

A compute-only reproduction of the *qualitative* features of Wittemer, Clos,
Breuer, Warring, Schaetz, PRA 97, 020102(R) (2018), built from the Phase-A
primitives (ND1 trace distance + BLP `𝒩`; ND3 QPN-bias estimator) on the
single-mode spin-boson Hamiltonian (`clos2016_spin_boson_hamiltonian` with one
mode). Application-agnostic: it consumes the library primitives and adds no new
public symbol (the MCF probe-QFI precedent). The binding oracle lives in
`tests/regression/analytic/test_wittemer_2018_benchmark.py`.

System (paper Eq. 2): a spin coupled to one motional mode of frequency `ω_E` via
the full-exponential spin-boson coupling at rate `Ω`; detuning `δω_z = ω_z − ω_E`
enters the builder as `detuning_rad_s`. Two orthogonal initial spin states
(`|↑⟩`, `|↓⟩`) ⊗ a thermal mode state (`n̄`) are evolved; the reduced-spin trace
distance `D(t) = ½‖r⃗↑ − r⃗↓‖` is read from `⟨σ_{x,y,z}⟩`.

Panels reproduced:

* **D(t)** — revivals of the trace distance (information back-flow), Fig. 2.
* **𝒩 vs δω_z** — the local-probing resonance in non-Markovianity, Fig. 4(a).
* **ℬ(r)** — the QPN bias decomposition `ℬ_QPN`/`ℬ_sampling`/`ℬ_total`, Fig. 3
  (`ℬ_QPN → 0` as shots `r → ∞`).
* **𝒩 vs Γ_dec** — non-Markovianity *suppressed by added spin decoherence*
  (the deliberate extension beyond the paper, which keeps `Γ_dec` negligible).
  `Γ_dec` is injected as a **benchmark-local** spin-dephasing collapse operator
  via `qutip.mesolve` (WP-04 R5: no public spin channel; `solve(channels=…)` is
  motional-only).

A tight numeric match to the paper needs the (APS-gated) Supplemental; this
benchmark anchors on the qualitative features and the ND3 internal consistency.

Usage::

    python tools/run_wittemer_2018_benchmark.py

Output::

    benchmarks/data/wittemer_2018/
      report.json  — 𝒩(δ), ℬ(r) decomposition, 𝒩(Γ_dec), provenance
      arrays.npz   — times, D(t), detunings, 𝒩(δ), shots, ℬ components, Γ_dec, 𝒩(Γ_dec)
      plot.png     — the four panels above
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import qutip
from numpy.typing import NDArray

from iontrap_dynamics.clos2016 import clos2016_initial_state, clos2016_spin_boson_hamiltonian
from iontrap_dynamics.conventions import CONVENTION_VERSION
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.information import blp_non_markovianity, non_markovianity_qpn_bias
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_x, spin_y, spin_z
from iontrap_dynamics.operators import sigma_x_ion, sigma_y_ion, sigma_z_ion
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "wittemer_2018"

MODE = "b"
TWO_PI = 2.0 * np.pi
OMEGA_E = TWO_PI * 1.920e6  # motional-mode frequency ω_E (rad/s), paper value
OMEGA = TWO_PI * 0.100e6  # spin-boson coupling rate Ω (rad/s), paper value
ION_MASS_KG = 25.0 * 1.66053906660e-27  # ²⁵Mg⁺
MAX_PHONONS = 12
N_BAR = 0.1  # thermal occupation near the paper's Fig. 4(a) n̄≈0.09 — small n̄ gives a sharp resonance and minimal Fock occupation
TAU = TWO_PI / OMEGA
N_TIMES = 360
T_MAX = 9.0 * TAU
DETUNINGS = np.linspace(-2.0, 2.0, 17) * OMEGA  # δω_z sweep for the resonance panel
SHOTS = (50, 200, 1000, 5000, 50_000)  # QPN repetitions r
GAMMA_DEC = np.array([0.0, 0.1, 0.25, 0.5, 1.0]) * OMEGA  # added spin dephasing
QPN_REPEATS = 400
QPN_DETUNING = (
    0.0  # on resonance: deepest D-dip and largest 𝒩 — clearest for the D(t) and bias panels
)

PLOT_ALT_TEXT = (
    "Four panels reproducing qualitative features of Wittemer 2018. Top-left: the "
    "trace distance D(t) revives, signalling information back-flow. Top-right: the "
    "non-Markovianity N versus spin-mode detuning shows a resonance structure. "
    "Bottom-left: the QPN bias decomposition versus shots r, with the QPN part "
    "vanishing as r grows. Bottom-right: N decreases as added spin decoherence "
    "Gamma_dec increases."
)


def _hilbert() -> HilbertSpace:
    mode = ModeConfig(
        label=MODE, frequency_rad_s=OMEGA_E, eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]])
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={MODE: MAX_PHONONS + 1})


def _hamiltonian(detuning_rad_s: float) -> qutip.Qobj:
    return clos2016_spin_boson_hamiltonian(
        max_phonons=MAX_PHONONS,
        axial_frequency_rad_s=OMEGA_E,
        dimensionless_mode_frequencies=[1.0],
        center_mode_weights=[1.0],
        carrier_rabi_frequency_rad_s=OMEGA,
        detuning_rad_s=detuning_rad_s,
        ion_mass_kg=ION_MASS_KG,
    )


def _initial_state(theta_rad: float) -> qutip.Qobj:
    return clos2016_initial_state(
        max_phonons=MAX_PHONONS, mean_occupations=[N_BAR], theta_rad=theta_rad
    )


def _bloch_via_solve(
    hilbert: HilbertSpace, hamiltonian: qutip.Qobj, theta: float, times: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Reduced-spin Bloch trajectory ``(T, 3)`` via the package solver (no decoherence)."""
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=_initial_state(theta),
        times=times,
        observables=(spin_x(hilbert, 0), spin_y(hilbert, 0), spin_z(hilbert, 0)),
    )
    e = result.expectations
    return np.stack([e["sigma_x_0"], e["sigma_y_0"], e["sigma_z_0"]], axis=1)


def _bloch_via_mesolve(
    hilbert: HilbertSpace,
    hamiltonian: qutip.Qobj,
    theta: float,
    times: NDArray[np.float64],
    gamma_dec: float,
) -> NDArray[np.float64]:
    """Reduced-spin Bloch trajectory with a benchmark-local spin-dephasing c_op (WP-04 R5)."""
    sx = hilbert.spin_op_for_ion(sigma_x_ion(), 0)
    sy = hilbert.spin_op_for_ion(sigma_y_ion(), 0)
    sz = hilbert.spin_op_for_ion(sigma_z_ion(), 0)
    # √(Γ_dec/2)·σ_z so the spin coherence decays as exp(−Γ_dec·t): Γ_dec is the
    # physical pure-dephasing rate (Γ_dec = 1/T2), not the bare Lindblad coefficient.
    c_ops = [np.sqrt(gamma_dec / 2.0) * sz] if gamma_dec > 0.0 else []
    result = qutip.mesolve(
        hamiltonian, _initial_state(theta), times, c_ops=c_ops, e_ops=[sx, sy, sz]
    )
    return np.stack([result.expect[0], result.expect[1], result.expect[2]], axis=1)


def _trace_distance(
    bloch_up: NDArray[np.float64], bloch_down: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Qubit trace distance ``D = ½‖r⃗↑ − r⃗↓‖`` per time index."""
    return np.asarray(0.5 * np.linalg.norm(bloch_up - bloch_down, axis=1), dtype=np.float64)


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
    hilbert = _hilbert()
    times = np.linspace(0.0, T_MAX, N_TIMES)

    # --- Panel 1: D(t) at the working detuning (revivals → back-flow) ---
    h_work = _hamiltonian(QPN_DETUNING)
    bloch_up = _bloch_via_solve(hilbert, h_work, 0.0, times)
    bloch_down = _bloch_via_solve(hilbert, h_work, np.pi, times)
    distance = _trace_distance(bloch_up, bloch_down)
    measure_work = blp_non_markovianity(distance)

    # --- Panel 2: 𝒩 vs δω_z (local-probing resonance) ---
    measure_vs_detuning = np.empty(len(DETUNINGS), dtype=np.float64)
    for i, delta in enumerate(DETUNINGS):
        h = _hamiltonian(float(delta))
        up = _bloch_via_solve(hilbert, h, 0.0, times)
        dn = _bloch_via_solve(hilbert, h, np.pi, times)
        measure_vs_detuning[i] = blp_non_markovianity(_trace_distance(up, dn))

    # --- Panel 3: QPN bias decomposition vs shots r (uses the working-point Bloch) ---
    shots_arr = np.array(SHOTS, dtype=np.int64)
    qpn_bias = np.empty(len(SHOTS), dtype=np.float64)
    sampling_bias = np.empty(len(SHOTS), dtype=np.float64)
    total_bias = np.empty(len(SHOTS), dtype=np.float64)
    for i, r in enumerate(SHOTS):
        res = non_markovianity_qpn_bias(
            bloch_up, bloch_down, shots=int(r), repeats=QPN_REPEATS, seed=20180205
        )
        qpn_bias[i] = res.qpn_bias
        sampling_bias[i] = res.sampling_bias
        total_bias[i] = res.total_bias

    # --- Panel 4: 𝒩 vs added spin decoherence Γ_dec (suppression) ---
    measure_vs_gamma = np.empty(len(GAMMA_DEC), dtype=np.float64)
    for i, gamma in enumerate(GAMMA_DEC):
        up = _bloch_via_mesolve(hilbert, h_work, 0.0, times, float(gamma))
        dn = _bloch_via_mesolve(hilbert, h_work, np.pi, times, float(gamma))
        measure_vs_gamma[i] = blp_non_markovianity(_trace_distance(up, dn))

    print(">>> Wittemer-2018 single-mode non-Markovianity benchmark (ND4)")
    print(
        f"D(0) = {distance[0]:.4f}  D_min = {distance.min():.4f}  𝒩(δ={QPN_DETUNING / OMEGA:.1f}Ω) = {measure_work:.4f}"
    )
    print(f"𝒩 vs δ: min {measure_vs_detuning.min():.3f}  max {measure_vs_detuning.max():.3f}")
    print(f"{'r':>8} {'B_QPN':>10} {'B_sampling':>12} {'B_total':>10}")
    for i, r in enumerate(SHOTS):
        print(f"{r:>8} {qpn_bias[i]:>10.4f} {sampling_bias[i]:>12.4f} {total_bias[i]:>10.4f}")
    print(
        f"𝒩 vs Γ_dec/Ω {list(np.round(GAMMA_DEC / OMEGA, 2))}: {list(np.round(measure_vs_gamma, 3))}"
    )

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        times=times,
        distance=distance,
        detunings_over_omega=DETUNINGS / OMEGA,
        measure_vs_detuning=measure_vs_detuning,
        shots=shots_arr,
        qpn_bias=qpn_bias,
        sampling_bias=sampling_bias,
        total_bias=total_bias,
        gamma_dec_over_omega=GAMMA_DEC / OMEGA,
        measure_vs_gamma=measure_vs_gamma,
    )

    report = {
        "scenario": "wittemer_2018",
        "purpose": (
            "Qualitative single-mode reproduction of Wittemer et al. PRA 97 020102 (2018) "
            "from the Phase-A non-Markovianity primitives (ND1 trace distance + BLP measure, "
            "ND3 QPN-bias estimator) on the clos2016 spin-boson Hamiltonian: trace-distance "
            "revivals, the non-Markovianity resonance vs detuning, the QPN bias decomposition "
            "vs shots, and the suppression of non-Markovianity by added spin decoherence. "
            "Application-agnostic; the binding oracle is the regression test."
        ),
        "workplan_reference": "WP/WP-04-non-markovianity-spectral-density.md (WI-4, dispatch ND4)",
        "schema_version": 2,
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": "qutip",
        "backend_version": qutip.__version__,
        "parameters": {
            "omega_E_rad_s": OMEGA_E,
            "Omega_rad_s": OMEGA,
            "ion_mass_kg": ION_MASS_KG,
            "max_phonons": MAX_PHONONS,
            "n_bar": N_BAR,
            "t_max_over_tau": T_MAX / TAU,
            "n_times": N_TIMES,
            "qpn_detuning_over_omega": QPN_DETUNING / OMEGA,
            "qpn_repeats": QPN_REPEATS,
        },
        "working_point": {
            "detuning_over_omega": QPN_DETUNING / OMEGA,
            "trace_distance_t0": float(distance[0]),
            "trace_distance_min": float(distance.min()),
            "measure": float(measure_work),
        },
        "measure_vs_detuning": [
            {"detuning_over_omega": float(d / OMEGA), "measure": float(m)}
            for d, m in zip(DETUNINGS, measure_vs_detuning, strict=True)
        ],
        "qpn_bias_vs_shots": [
            {
                "shots": int(r),
                "qpn_bias": float(qpn_bias[i]),
                "sampling_bias": float(sampling_bias[i]),
                "total_bias": float(total_bias[i]),
            }
            for i, r in enumerate(SHOTS)
        ],
        "measure_vs_gamma_dec": [
            {"gamma_dec_over_omega": float(g / OMEGA), "measure": float(m)}
            for g, m in zip(GAMMA_DEC, measure_vs_gamma, strict=True)
        ],
        "notes": {
            "supplemental": "PRA 97 020102 Supplemental Material is APS-gated; qualitative oracle only.",
            "gamma_dec": (
                "benchmark-local spin-dephasing c_op via qutip.mesolve (WP-04 R5). "
                "Collapse operator √(Γ_dec/2)·σ_z, so Γ_dec is the physical pure-dephasing "
                "rate (coherence ∝ exp(−Γ_dec·t), i.e. Γ_dec = 1/T2), not the bare Lindblad "
                "coefficient; the gamma_dec_over_omega axis is therefore Γ_dec/Ω in rate units."
            ),
            "eta": "Lamb-Dicke η is derived by the clos2016 builder from wavelength/mass/ω_E.",
            "generator": "clos2016 e^{η(a−a†)} ≡ paper e^{iη(a†+a)} via a→ia (same reduced D(t)/𝒩).",
            "calibration": (
                "𝒩 is sensitive to the time-sampling rate γ (the paper's own effect): "
                "undersampling (e.g. 150 pts / 6τ) collapses 𝒩 toward 0, so this benchmark "
                "uses fine sampling over t_max=9τ. It is also sensitive to n̄: large n̄≈1 "
                "washes out the detuning resonance and raises Fock occupation (a §15 "
                "Fock-quality warning at maxn=12, n̄=1), so the resonance uses small n̄≈0.1 "
                "(the Fig. 4a regime). The robust resonance signature is the trace-distance "
                "dip depth D_min(δω_z), deepest on resonance."
            ),
        },
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

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5))
    axes[0, 0].plot(times / TAU, distance, color="#1f77b4", linewidth=1.0)
    axes[0, 0].set_xlabel(r"interaction time $t/\tau$")
    axes[0, 0].set_ylabel(r"trace distance $D(t)$")
    axes[0, 0].set_title(rf"$D(t)$ revivals ($\delta\omega_z={QPN_DETUNING / OMEGA:.1f}\,\Omega$)")
    axes[0, 1].plot(
        DETUNINGS / OMEGA, measure_vs_detuning, color="#2ca02c", marker="o", markersize=3
    )
    axes[0, 1].set_xlabel(r"detuning $\delta\omega_z/\Omega$")
    axes[0, 1].set_ylabel(r"non-Markovianity $\mathcal{N}$")
    axes[0, 1].set_title(r"$\mathcal{N}$ vs detuning (local-probing resonance)")
    axes[1, 0].axhline(0.0, color="#999999", linewidth=0.7)
    axes[1, 0].plot(
        shots_arr, qpn_bias, marker="o", label=r"$\mathcal{B}_{\rm QPN}$", color="#d62728"
    )
    axes[1, 0].plot(
        shots_arr, sampling_bias, marker="s", label=r"$\mathcal{B}_{\rm sampling}$", color="#9467bd"
    )
    axes[1, 0].plot(
        shots_arr, total_bias, marker="^", label=r"$\mathcal{B}_{\rm total}$", color="#1f77b4"
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel(r"repetitions $r$")
    axes[1, 0].set_ylabel(r"bias $\mathcal{B}$")
    axes[1, 0].set_title(r"QPN bias: $\mathcal{B}_{\rm QPN}\to 0$ as $r\to\infty$")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(GAMMA_DEC / OMEGA, measure_vs_gamma, color="#ff7f0e", marker="o")
    axes[1, 1].set_xlabel(r"added spin decoherence $\Gamma_{\rm dec}/\Omega$")
    axes[1, 1].set_ylabel(r"non-Markovianity $\mathcal{N}$")
    axes[1, 1].set_title(r"$\mathcal{N}$ suppressed by decoherence")
    fig.suptitle("Wittemer 2018 single-mode non-Markovianity (ND4, qualitative)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
