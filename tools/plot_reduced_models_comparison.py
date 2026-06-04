# SPDX-License-Identifier: MIT
"""Benchmark — reduced light–matter models vs full trapped-ion dynamics (Cases A–D).

Tutorial-18 companion (WP-03 WI-6, dispatch RLF). The reduced JC / AJC / QRM models
(physics layer, ``reduced_models.py``) are compared against the apparatus sideband
realisations (``hamiltonians.py``) along the four falsifiable cases of the model
hierarchy:

* Case A — JC ↔ AJC are the *same* model up to a label: the LOCK-3 identity
  ``H_AJC(ω0) = σ_x H_JC(−ω0) σ_x`` makes the two spectra coincide.
* Case B — the label becomes a physical knob on the ion: from |↓,0⟩ the **red**
  sideband (→ JC) is dark (``a|0⟩ = 0``) while the **blue** sideband (→ AJC) flops
  at the blue-sideband Rabi rate.
* Case C — the label is false under the rotating-wave breakdown: the non-RWA QRM
  ground state acquires virtual phonons ``⟨a†a⟩ > 0`` and its ground energy departs
  from JC as ``g/ω0`` grows from weak to ultra-strong coupling.
* Case D — the full Lamb–Dicke (all-orders Debye–Waller × Laguerre) sideband rate
  tends to the leading-order ``2g√(n±1)`` limit as ``η²(2n+1) → 0`` and departs
  through the deep → intermediate → beyond regimes.

Application-agnostic: textbook/oracle comparison only, no consuming-application
framing. CONVENTIONS v0.4 (§25 reduced light–matter models).

Usage::

    python tools/plot_reduced_models_comparison.py

Output::

    benchmarks/data/reduced_models_comparison/
      report.json  — per-case control points, analytic oracles, max error
      arrays.npz   — the raw per-case arrays + analytic_* oracles
      plot.png     — four-panel A–D comparison (reduced model vs realisation)
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import qutip

from iontrap_dynamics.analytic import (
    blue_sideband_rabi_frequency,
    lamb_dicke_confinement,
    lamb_dicke_parameter,
    lamb_dicke_regime,
    red_sideband_rabi_frequency,
    red_sideband_rabi_frequency_full_ld,
)
from iontrap_dynamics.conventions import CONVENTION_VERSION
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import blue_sideband_hamiltonian, red_sideband_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import Observable
from iontrap_dynamics.operators import spin_down, spin_up
from iontrap_dynamics.reduced_models import (
    anti_jaynes_cummings_hamiltonian,
    jaynes_cummings_hamiltonian,
    quantum_rabi_hamiltonian,
)
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.spectrum import solve_spectrum
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "reduced_models_comparison"

# --- deterministic parameters (UPPER_SNAKE, no RNG) ---
OMEGA = 2.0 * np.pi * 1.0e6  # ω0 = ω_f, the resonant model scale (rad·s⁻¹)
MODE_FREQ = 2.0 * np.pi * 1.0e6  # apparatus mode frequency ν (rad·s⁻¹)
FOCK_SPECTRUM = 30  # Cases A/C: dense diagonalisation, generous truncation
FOCK_DYNAMICS = 12  # Case B: low-phonon sideband flop from |↓,0⟩
# Case A: a representative coupling for the spectrum-coincidence check.
G_CASE_A = 0.4 * OMEGA
# Case B: apparatus drive — k along the (axial, z) mode axis gives η ≈ 0.11.
K_VECTOR = np.array([0.0, 0.0, 8.0e6])
RABI = 2.0 * np.pi * 50.0e3  # carrier Rabi Ω for the sideband drive (rad·s⁻¹)
# Case C: weak → ultra-strong coupling sweep (dimensionless g/ω0).
G_OVER_W0 = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5])
# Case D: η sweep at n = 1 so η²(2n+1) = 3η² spans deep → beyond.
ETA_GRID = np.array([0.02, 0.06, 0.12, 0.2, 0.35, 0.6, 1.0])
CASE_D_N = 1

TOLERANCE = 1e-3

PLOT_ALT_TEXT = (
    "Four panels, one per case of the reduced-model hierarchy. "
    "Panel A: the lowest eigenvalues of the Jaynes-Cummings model at minus omega-zero "
    "and the anti-Jaynes-Cummings model at plus omega-zero lie exactly on top of each "
    "other, confirming the LOCK-3 identity that the two are the same model up to a "
    "label. Panel B: starting from spin-down with zero phonons, the spin-up population "
    "stays flat at zero under the red sideband (dark, the Jaynes-Cummings image) but "
    "oscillates fully between zero and one under the blue sideband (bright, the "
    "anti-Jaynes-Cummings image). Panel C: as the dimensionless coupling g over "
    "omega-zero grows from weak to ultra-strong, the ground-state energy gap between "
    "the Jaynes-Cummings and quantum Rabi models rises from near zero, while the "
    "quantum Rabi ground-state phonon number, overlaid, grows from zero, showing the "
    "rotating-wave approximation breaking down. Panel D: the relative deviation between "
    "the all-orders full-Lamb-Dicke red-sideband rate and the leading-order rate rises "
    "with the confinement parameter eta squared times two n plus one, crossing the "
    "deep, intermediate, and beyond regime bands."
)

# House colours.
_BLUE, _RED, _GREEN, _PURPLE, _GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#444444"


def _hilbert(fock_dim: int, label: str = "m") -> HilbertSpace:
    """A one-ion, one-mode Hilbert space (the spin is the qubit, ``m`` the mode)."""
    eigenvector = np.array([[0.0, 0.0, 1.0]])
    mode = ModeConfig(label=label, frequency_rad_s=MODE_FREQ, eigenvector_per_ion=eigenvector)
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={label: fock_dim})


def _ground_phonons(hamiltonian: qutip.Qobj, number_op: qutip.Qobj) -> float:
    """⟨a†a⟩ of the ground state via ``solve_spectrum``."""
    result = solve_spectrum(hamiltonian)
    ground = result.eigenvectors[:, 0]
    return float(np.real(ground.conj() @ (number_op.full() @ ground)))


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
    hilbert = _hilbert(FOCK_SPECTRUM)

    # --- Case A: spec(H_AJC(+ω0)) == spec(H_JC(−ω0)) (LOCK-3 → coincident spectra) ---
    ajc = anti_jaynes_cummings_hamiltonian(
        hilbert, "m", ion_index=0, omega_0=OMEGA, omega_f=OMEGA, g=G_CASE_A
    )
    jc_neg = jaynes_cummings_hamiltonian(
        hilbert, "m", ion_index=0, omega_0=-OMEGA, omega_f=OMEGA, g=G_CASE_A
    )
    eig_ajc = solve_spectrum(ajc).eigenvalues[:24] / OMEGA  # dimensionless (R4 presentation)
    eig_jc = solve_spectrum(jc_neg).eigenvalues[:24] / OMEGA
    case_a_error = float(np.max(np.abs(eig_ajc - eig_jc)))

    # --- Case B: |↓,0⟩ dark under the red sideband, bright under the blue sideband ---
    hb = _hilbert(FOCK_DYNAMICS, label="m")
    drive = DriveConfig(k_vector_m_inv=K_VECTOR, carrier_rabi_frequency_rad_s=RABI, phase_rad=0.0)
    eta = lamb_dicke_parameter(
        k_vec=K_VECTOR,
        mode_eigenvector=np.array([0.0, 0.0, 1.0]),
        ion_mass=mg25_plus().mass_kg,
        mode_frequency=MODE_FREQ,
    )
    h_red = red_sideband_hamiltonian(hb, drive, "m", ion_index=0)
    h_blue = blue_sideband_hamiltonian(hb, drive, "m", ion_index=0)
    psi0 = qutip.tensor(spin_down(), qutip.basis(FOCK_DYNAMICS, 0))
    p_up = Observable(label="p_up", operator=hb.spin_op_for_ion(spin_up() * spin_up().dag(), 0))
    blue_rate = blue_sideband_rabi_frequency(
        carrier_rabi_frequency=RABI, lamb_dicke_parameter=eta, n_initial=0
    )
    t_blue = 2.0 * np.pi / blue_rate  # one full blue flop period
    times_b = np.linspace(0.0, t_blue, 80)
    pop_red = np.asarray(
        solve(
            hilbert=hb, hamiltonian=h_red, initial_state=psi0, times=times_b, observables=(p_up,)
        ).expectations["p_up"],
        dtype=np.float64,
    )
    pop_blue = np.asarray(
        solve(
            hilbert=hb, hamiltonian=h_blue, initial_state=psi0, times=times_b, observables=(p_up,)
        ).expectations["p_up"],
        dtype=np.float64,
    )
    analytic_blue = np.sin(blue_rate * times_b / 2.0) ** 2  # two-level flop, Ω_R = blue_rate
    case_b_error = float(max(np.max(pop_red), np.max(np.abs(pop_blue - analytic_blue))))

    # --- Case C: JC vs QRM ground-energy deviation + QRM virtual phonons vs g/ω0 ---
    number_op = hilbert.number_for_mode("m")
    e0_jc = np.empty(G_OVER_W0.size, dtype=np.float64)
    e0_qrm = np.empty(G_OVER_W0.size, dtype=np.float64)
    n_qrm = np.empty(G_OVER_W0.size, dtype=np.float64)
    for i, ratio in enumerate(G_OVER_W0):
        g = ratio * OMEGA
        jc = jaynes_cummings_hamiltonian(
            hilbert, "m", ion_index=0, omega_0=OMEGA, omega_f=OMEGA, g=g
        )
        qrm = quantum_rabi_hamiltonian(hilbert, "m", ion_index=0, omega_0=OMEGA, omega_f=OMEGA, g=g)
        e0_jc[i] = solve_spectrum(jc).eigenvalues[0] / OMEGA
        e0_qrm[i] = solve_spectrum(qrm).eigenvalues[0] / OMEGA
        n_qrm[i] = _ground_phonons(qrm, number_op)
    case_c_deviation = np.abs(e0_jc - e0_qrm)
    # Weak-coupling oracle: ⟨a†a⟩ ≈ (g/(ω0+ω_f))² = (ratio/2)² (2nd-order PT).
    analytic_c_phonons = (G_OVER_W0 / 2.0) ** 2
    weak = G_OVER_W0 <= 0.1
    case_c_error = float(np.max(np.abs(n_qrm[weak] - analytic_c_phonons[weak])))

    # --- Case D: full-LD vs leading-order red-sideband rate vs η²(2n+1) ---
    confinement = np.array(
        [
            lamb_dicke_confinement(lamb_dicke_parameter=e, mean_phonon_number=CASE_D_N)
            for e in ETA_GRID
        ]
    )
    leading_d = np.array(
        [
            red_sideband_rabi_frequency(
                carrier_rabi_frequency=RABI, lamb_dicke_parameter=e, n_initial=CASE_D_N
            )
            for e in ETA_GRID
        ]
    )
    full_d = np.array(
        [
            red_sideband_rabi_frequency_full_ld(
                carrier_rabi_frequency=RABI, lamb_dicke_parameter=e, n_initial=CASE_D_N
            )
            for e in ETA_GRID
        ]
    )
    case_d_deviation = np.abs(full_d - leading_d) / leading_d
    regimes = [
        lamb_dicke_regime(lamb_dicke_parameter=e, mean_phonon_number=CASE_D_N) for e in ETA_GRID
    ]
    # Oracle (deep end): deviation → the Debye–Waller zero-point factor 1 − e^{−η²/2}.
    analytic_d_deep = 1.0 - np.exp(-(ETA_GRID[0] ** 2) / 2.0)
    case_d_error = float(abs(case_d_deviation[0] - analytic_d_deep))

    max_error = max(case_a_error, case_b_error, case_c_error, case_d_error)

    print(">>> reduced models vs full dynamics — Cases A–D")
    print(f"  Case A  spec coincidence (LOCK-3)         max|Δ| = {case_a_error:.2e}")
    print(f"  Case B  red dark / blue bright            max|Δ| = {case_b_error:.2e}")
    print(f"  Case C  QRM weak-coupling ⟨a†a⟩ vs PT      max|Δ| = {case_c_error:.2e}")
    print(f"  Case D  full-LD deep-limit vs Debye–Waller max|Δ| = {case_d_error:.2e}")
    print(f"max |numerical - analytic| = {max_error:.2e}  (tolerance {TOLERANCE:.0e})")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        case_a_eig_jc=eig_jc,
        case_a_eig_ajc=eig_ajc,
        case_b_t=times_b,
        case_b_pop_red=pop_red,
        case_b_pop_blue=pop_blue,
        case_b_analytic_blue=analytic_blue,
        case_c_g_over_w0=G_OVER_W0,
        case_c_deviation=case_c_deviation,
        case_c_n_qrm=n_qrm,
        case_c_analytic_phonons=analytic_c_phonons,
        case_d_confinement=confinement,
        case_d_deviation=case_d_deviation,
        case_d_eta=ETA_GRID,
    )

    report = {
        "scenario": "reduced_models_comparison",
        "purpose": (
            "Reduced light-matter models (JC/AJC/QRM, physics layer) vs the apparatus "
            "sideband realisations, along the four hierarchy cases. Case A: the LOCK-3 "
            "identity H_AJC(w0)=sigma_x H_JC(-w0) sigma_x makes spec(H_AJC(+w0)) and "
            "spec(H_JC(-w0)) coincide. Case B: from |down,0> the red sideband (JC image) "
            "is dark while the blue sideband (AJC image) flops at the blue-sideband Rabi "
            "rate. Case C: the non-RWA QRM ground state acquires virtual phonons "
            "<a-dag a> ~= (g/(w0+w_f))^2 at weak coupling and its ground energy departs "
            "from JC as g/w0 grows. Case D: the full-Lamb-Dicke red-sideband rate tends "
            "to the leading-order 2g sqrt(n) limit as eta^2 (2n+1) -> 0. "
            "Application-agnostic: textbook/oracle comparison only, no application framing."
        ),
        "workplan_reference": "WP/WP-03-reduced-models.md (WI-6, dispatch RLF)",
        "schema_version": 2,
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": "qutip",
        "backend_version": qutip.__version__,
        "fock_spectrum": FOCK_SPECTRUM,
        "fock_dynamics": FOCK_DYNAMICS,
        "lamb_dicke_parameter": float(eta),
        "blue_sideband_rate_rad_s": float(blue_rate),
        "case_c": [
            {
                "g_over_w0": float(G_OVER_W0[i]),
                "deviation": float(case_c_deviation[i]),
                "n_qrm": float(n_qrm[i]),
                "analytic_phonons_pt": float(analytic_c_phonons[i]),
            }
            for i in range(G_OVER_W0.size)
        ],
        "case_d": [
            {
                "eta": float(ETA_GRID[i]),
                "confinement_eta2_2n1": float(confinement[i]),
                "regime": regimes[i],
                "deviation": float(case_d_deviation[i]),
            }
            for i in range(ETA_GRID.size)
        ],
        "analytic_formulas": {
            "case_A": "spec(H_AJC(+w0)) = spec(H_JC(-w0))  (LOCK-3)",
            "case_B": "red P_up = 0 (dark); blue P_up = sin^2(Omega_blue t / 2)",
            "case_C": "QRM ground <a-dag a> ~= (g/(w0+w_f))^2  (2nd-order PT, weak)",
            "case_D": "leading red rate = Omega |eta| sqrt(n) = 2 g sqrt(n); deep dev = 1 - exp(-eta^2/2)",
        },
        "case_errors": {
            "A": case_a_error,
            "B": case_b_error,
            "C": case_c_error,
            "D": case_d_error,
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

    fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(1, 4, figsize=(15.0, 4.2))

    # Panel A — spectrum coincidence.
    idx = np.arange(eig_jc.size)
    ax_a.plot(idx, eig_jc, color=_GREY, linewidth=1.0, label=r"JC$(-\omega_0)$")
    ax_a.scatter(idx, eig_ajc, color=_RED, marker="o", s=16, zorder=3, label=r"AJC$(+\omega_0)$")
    ax_a.set_xlabel("eigenvalue index")
    ax_a.set_ylabel(r"$E / \omega_0$")
    ax_a.set_title("A · LOCK-3: spectra coincide")
    ax_a.legend(frameon=False, fontsize=8)

    # Panel B — red dark / blue bright.
    tau_b = times_b * blue_rate / (2.0 * np.pi)
    ax_b.plot(tau_b, analytic_blue, color=_GREY, linewidth=1.0, label=r"$\sin^2(\Omega_b t/2)$")
    ax_b.scatter(
        tau_b, pop_blue, color=_BLUE, marker="o", s=12, zorder=3, label="blue → AJC (bright)"
    )
    ax_b.scatter(tau_b, pop_red, color=_RED, marker="s", s=12, zorder=3, label="red → JC (dark)")
    ax_b.set_xlabel(r"blue flop phase $\Omega_b t / 2\pi$")
    ax_b.set_ylabel(r"spin-up population $\langle P_\uparrow\rangle$")
    ax_b.set_title(r"B · $|{\downarrow},0\rangle$ dark vs bright")
    ax_b.legend(frameon=False, fontsize=8)

    # Panel C — JC↔QRM deviation + QRM phonons.
    ax_c.plot(
        G_OVER_W0,
        case_c_deviation,
        color=_PURPLE,
        marker="o",
        markersize=4,
        label=r"$|E_0^{JC}-E_0^{QRM}|$",
    )
    ax_c.plot(
        G_OVER_W0,
        n_qrm,
        color=_GREEN,
        marker="s",
        markersize=4,
        label=r"QRM $\langle a^\dagger a\rangle$",
    )
    ax_c.set_xscale("log")
    ax_c.set_xlabel(r"$g / \omega_0$")
    ax_c.set_ylabel(r"deviation, $\langle a^\dagger a\rangle$ ($/\omega_0$)")
    ax_c.set_title("C · RWA breakdown")
    ax_c.legend(frameon=False, fontsize=8)

    # Panel D — full-LD vs leading deviation with regime bands.
    ax_d.axvspan(1e-4, 0.1, color=_GREEN, alpha=0.08)
    ax_d.axvspan(0.1, 1.0, color=_BLUE, alpha=0.08)
    ax_d.axvspan(1.0, 1e2, color=_RED, alpha=0.08)
    ax_d.plot(confinement, case_d_deviation, color=_GREY, marker="o", markersize=4)
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel(r"$\eta^2(2n+1)$")
    ax_d.set_ylabel("full-LD vs leading rel. dev.")
    ax_d.set_title("D · deep → intermediate → beyond")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
