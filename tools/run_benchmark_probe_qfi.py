# SPDX-License-Identifier: MIT
"""Benchmark — probe QFI: coherent shot-noise vs squeezed sub-shot-noise.

WI-6 of WP-02 (dispatch MCF). A compute-only **consumer** benchmark of WP-01's
SLD-QFI primitive (dispatch EDA): it adds no new public symbol (§8). For coherent
and squeezed-vacuum probes it computes the SLD quantum Fisher information under
unitary encoding ρ(θ)=e^{−iθG}ρe^{+iθG} (a probe is a pure state, so F_Q = 4·Var(G),
CONVENTIONS §19) and checks it against textbook closed forms — with **zero
application framing**: the oracle is textbook closed forms only (the WP-02 §1
boundary keeps any programme-specific probe comparison outside the library).

Cases (generators on a single mode; the spin is an inert spectator |↓⟩):

* coherent ``|α⟩`` phase (G = n̂):           ``F_Q = 4|α|²``  (shot-noise / SQL)
* squeezed ``|ξ=r⟩`` displacement (G = x):   ``F_Q = 2e^{−2r}``  (sub-shot-noise)
* squeezed ``|ξ=r⟩`` displacement (G = p):   ``F_Q = 2e^{+2r}``  (anti-squeezed)
* squeezed ``|ξ=r⟩`` phase (G = n̂):          ``F_Q = 2 sinh²(2r)``
* qubit ``|+⟩`` phase (G = J_z = ½σ_z):      ``F_Q = 1``; the σ_y measurement
  (the SLD eigenbasis) saturates the Cramér–Rao bound (CFI = QFI).

``squeezed_vacuum_mode(r)`` squeezes the ``x = (a+a†)/√2`` quadrature (verified by
direct computation), so the *x*-generator carries the sub-shot-noise gain.
Compute-only and deterministic — no solver trajectory, no randomness. The binding
oracle assertion lives in ``tests/regression/analytic/test_probe_qfi.py``.

Usage::

    python tools/run_benchmark_probe_qfi.py

Output::

    benchmarks/data/probe_qfi/
      report.json  — per-point F_Q, the analytic oracle, max error, provenance
      arrays.npz   — alpha, qfi_coherent_{phase,x,p}, r, qfi_squeezed_{x,p,phase} + oracles
      plot.png     — F_Q vs r: squeezed x below / p above the shot-noise floor 2
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
    quantum_fisher_information_trajectory,
)
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.operators import sigma_y_ion, sigma_z_ion, spin_down, spin_up
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import coherent_mode, compose_density, squeezed_vacuum_mode
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "probe_qfi"

MODE = "b"
FOCK_DIM_COHERENT = 60
FOCK_DIM_SQUEEZED = 100
ALPHA_VALUES = (0.5, 1.0, 1.5, 2.0)
R_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SHOT_NOISE_FLOOR = 2.0

PLOT_ALT_TEXT = (
    "Plot of quantum Fisher information versus squeezing parameter r for a "
    "squeezed-vacuum probe under displacement sensing, on a logarithmic y-axis. "
    "The squeezed-quadrature generator x follows two times e to the minus two r, "
    "dipping below the horizontal shot-noise floor at two; the conjugate "
    "generator p follows two times e to the plus two r, rising above it. The "
    "numerical points lie exactly on the analytic reference curves."
)


def _single_mode_hilbert(fock_dim: int) -> HilbertSpace:
    """A one-ion, one-mode Hilbert space (spin ⊗ Fock); the spin is a spectator."""
    mode = ModeConfig(
        label=MODE,
        frequency_rad_s=2.0 * np.pi * 1.0e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={MODE: fock_dim})


def _embed(hilbert: HilbertSpace, mode_ket: qutip.Qobj) -> qutip.Qobj:
    return compose_density(
        hilbert, spin_states_per_ion=[spin_down()], mode_states_by_label={MODE: mode_ket}
    )


def _quad_x(hilbert: HilbertSpace) -> qutip.Qobj:
    a = hilbert.annihilation_for_mode(MODE)
    return (a + a.dag()) / np.sqrt(2.0)


def _quad_p(hilbert: HilbertSpace) -> qutip.Qobj:
    a = hilbert.annihilation_for_mode(MODE)
    return (a - a.dag()) / (1j * np.sqrt(2.0))


def _qfi(hilbert: HilbertSpace, rho: qutip.Qobj, generator: qutip.Qobj) -> float:
    return float(
        quantum_fisher_information_trajectory([rho], hilbert=hilbert, generator=generator)[0]
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


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- coherent probe: phase QFI = 4|α|² (SQL) and displacement QFI = 2 ---
    alpha_arr = np.array(ALPHA_VALUES, dtype=np.float64)
    hc = _single_mode_hilbert(FOCK_DIM_COHERENT)
    n_c, x_c, p_c = hc.number_for_mode(MODE), _quad_x(hc), _quad_p(hc)
    rho_coh = [_embed(hc, coherent_mode(FOCK_DIM_COHERENT, a)) for a in ALPHA_VALUES]
    qfi_coh_phase = np.array([_qfi(hc, rho, n_c) for rho in rho_coh], dtype=np.float64)
    qfi_coh_x = np.array([_qfi(hc, rho, x_c) for rho in rho_coh], dtype=np.float64)
    qfi_coh_p = np.array([_qfi(hc, rho, p_c) for rho in rho_coh], dtype=np.float64)
    analytic_coh_phase = 4.0 * alpha_arr**2
    analytic_coh_disp = np.full_like(alpha_arr, SHOT_NOISE_FLOOR)  # F_Q = 2, α-independent

    # --- squeezed probe: displacement (x / p) and phase QFI ---
    r_arr = np.array(R_VALUES, dtype=np.float64)
    hs = _single_mode_hilbert(FOCK_DIM_SQUEEZED)
    n_s, x_s, p_s = hs.number_for_mode(MODE), _quad_x(hs), _quad_p(hs)
    rho_sq = [_embed(hs, squeezed_vacuum_mode(FOCK_DIM_SQUEEZED, r)) for r in R_VALUES]
    qfi_sq_x = np.array([_qfi(hs, rho, x_s) for rho in rho_sq], dtype=np.float64)
    qfi_sq_p = np.array([_qfi(hs, rho, p_s) for rho in rho_sq], dtype=np.float64)
    qfi_sq_phase = np.array([_qfi(hs, rho, n_s) for rho in rho_sq], dtype=np.float64)
    analytic_sq_x = 2.0 * np.exp(-2.0 * r_arr)
    analytic_sq_p = 2.0 * np.exp(+2.0 * r_arr)
    analytic_sq_phase = 2.0 * np.sinh(2.0 * r_arr) ** 2

    # --- CFI saturation on the |+⟩ / J_z qubit case (σ_y is the SLD eigenbasis) ---
    hq = IonSystem(species_per_ion=(mg25_plus(),))
    h_qubit = HilbertSpace(system=hq, fock_truncations={})
    jz = 0.5 * h_qubit.spin_op_for_ion(sigma_z_ion(), 0)
    plus = (spin_up() + spin_down()).unit()
    qfi_qubit = _qfi(h_qubit, plus, jz)
    # Outcome probabilities and θ-derivatives for the σ_y measurement, derived
    # from the probe: P_± = (I ± σ_y)/2, ∂_θp_±|₀ = i⟨ψ|[J_z, P_±]|ψ⟩.
    sigma_y = sigma_y_ion()
    identity = qutip.qeye(sigma_y.dims[0])
    proj_y = [(identity + sigma_y) / 2, (identity - sigma_y) / 2]
    p_y = [float(qutip.expect(pr, plus).real) for pr in proj_y]
    dp_y = [float((1j * qutip.expect(jz * pr - pr * jz, plus)).real) for pr in proj_y]
    cfi_optimal = classical_fisher_information(p_y, parameter_derivative=dp_y)

    max_error = float(
        max(
            np.max(np.abs(qfi_coh_phase - analytic_coh_phase)),
            np.max(np.abs(qfi_coh_x - analytic_coh_disp)),
            np.max(np.abs(qfi_coh_p - analytic_coh_disp)),
            np.max(np.abs(qfi_sq_x - analytic_sq_x)),
            np.max(np.abs(qfi_sq_p - analytic_sq_p)),
            np.max(np.abs(qfi_sq_phase - analytic_sq_phase)),
            abs(qfi_qubit - 1.0),
            abs(cfi_optimal - qfi_qubit),
        )
    )

    print(">>> probe-QFI benchmark — coherent SQL vs squeezed sub-shot-noise")
    print(f"{'r':>5} {'F_Q(x)':>12} {'2e^-2r':>10} {'F_Q(p)':>12} {'2e^+2r':>10}")
    print("-" * 54)
    for i, r in enumerate(R_VALUES):
        print(
            f"{r:>5.2f} {qfi_sq_x[i]:>12.6f} {analytic_sq_x[i]:>10.6f} "
            f"{qfi_sq_p[i]:>12.6f} {analytic_sq_p[i]:>10.4f}"
        )
    print("-" * 54)
    print(f"qubit |+⟩/J_z: QFI={qfi_qubit:.8f}  CFI_optimal={cfi_optimal:.8f}  (saturation)")
    print(f"max |numerical - analytic| = {max_error:.2e}")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        alpha=alpha_arr,
        qfi_coherent_phase=qfi_coh_phase,
        analytic_coherent_phase=analytic_coh_phase,
        qfi_coherent_x=qfi_coh_x,
        qfi_coherent_p=qfi_coh_p,
        analytic_coherent_displacement=analytic_coh_disp,
        r=r_arr,
        qfi_squeezed_x=qfi_sq_x,
        analytic_squeezed_x=analytic_sq_x,
        qfi_squeezed_p=qfi_sq_p,
        analytic_squeezed_p=analytic_sq_p,
        qfi_squeezed_phase=qfi_sq_phase,
        analytic_squeezed_phase=analytic_sq_phase,
    )

    report = {
        "scenario": "probe_qfi",
        "purpose": (
            "Probe QFI closed-form validation: a coherent probe's phase QFI "
            "follows the standard quantum limit 4|alpha|^2, while a squeezed-vacuum "
            "probe's displacement QFI drops below the shot-noise floor (2 e^-2r on "
            "the squeezed quadrature) and the optimal measurement saturates the "
            "Cramer-Rao bound (CFI = QFI). Consumes the WP-01 SLD-QFI primitive "
            "with no new symbol. Application-agnostic: textbook oracle only, no "
            "application framing."
        ),
        "workplan_reference": "WP/WP-02-two-mode-motional.md (WI-6, dispatch MCF)",
        "schema_version": 2,
        # Compute-only benchmark: no solve() trajectory, so no canonical cache hash.
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": "qutip",
        "backend_version": qutip.__version__,
        "generators": {
            "phase": "n_hat = a^dag a",
            "displacement_x": "x = (a + a^dag) / sqrt(2)",
            "displacement_p": "p = (a - a^dag) / (i sqrt(2))",
            "qubit_phase": "J_z = 0.5 * sigma_z_ion",
        },
        "fock_dim_coherent": FOCK_DIM_COHERENT,
        "fock_dim_squeezed": FOCK_DIM_SQUEEZED,
        "coherent": [
            {
                "alpha": float(a),
                "qfi_phase": float(qfi_coh_phase[i]),
                "analytic_phase": float(analytic_coh_phase[i]),
                "qfi_displacement_x": float(qfi_coh_x[i]),
                "qfi_displacement_p": float(qfi_coh_p[i]),
                "analytic_displacement": float(analytic_coh_disp[i]),
            }
            for i, a in enumerate(ALPHA_VALUES)
        ],
        "squeezed": [
            {
                "r": float(r),
                "qfi_x": float(qfi_sq_x[i]),
                "analytic_x": float(analytic_sq_x[i]),
                "qfi_p": float(qfi_sq_p[i]),
                "analytic_p": float(analytic_sq_p[i]),
                "qfi_phase": float(qfi_sq_phase[i]),
                "analytic_phase": float(analytic_sq_phase[i]),
            }
            for i, r in enumerate(R_VALUES)
        ],
        "cfi_saturation": {
            "qfi": float(qfi_qubit),
            "cfi_optimal": float(cfi_optimal),
            "cramer_rao_bound": float(cramer_rao_bound(qfi_qubit)),
            "note": "|+> probe, J_z generator, sigma_y measurement (SLD eigenbasis): CFI = QFI = 1",
        },
        "analytic_formulas": {
            "coherent_phase": "4 * |alpha|^2 (standard quantum limit)",
            "squeezed_displacement_x": "2 * exp(-2r) (sub-shot-noise)",
            "squeezed_displacement_p": "2 * exp(+2r) (anti-squeezed)",
            "squeezed_phase": "2 * sinh(2r)^2",
            "shot_noise_floor": SHOT_NOISE_FLOOR,
        },
        "max_numerical_vs_analytic_error": max_error,
        "tolerance": 1e-6,
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
    ax.plot(r_arr, analytic_sq_x, color="#1f77b4", linewidth=1.0, label=r"squeezed $x$: $2e^{-2r}$")
    ax.plot(
        r_arr, analytic_sq_p, color="#d62728", linewidth=1.0, label=r"conjugate $p$: $2e^{+2r}$"
    )
    ax.axhline(
        SHOT_NOISE_FLOOR, color="#555555", linestyle="--", linewidth=0.8, label="shot-noise floor"
    )
    ax.scatter(
        r_arr, qfi_sq_x, color="#1f77b4", marker="o", zorder=3, label=r"$F_Q(x)$ (numerical)"
    )
    ax.scatter(
        r_arr, qfi_sq_p, color="#d62728", marker="s", zorder=3, label=r"$F_Q(p)$ (numerical)"
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"squeezing parameter $r$")
    ax.set_ylabel(r"quantum Fisher information $F_Q[\rho,\,G]$")
    ax.set_title("Probe QFI: squeezed sub-shot-noise vs shot-noise floor")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
