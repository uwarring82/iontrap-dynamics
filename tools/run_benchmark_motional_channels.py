# SPDX-License-Identifier: MIT
"""Benchmark — typed motional CPTP channels through ``solve(channels=…)``.

Tutorial-companion benchmark for the motional open-system surface (WP-02 WI-3,
dispatch MCC). Drives a single mode with each typed channel and checks the closed-form
decay law (CONVENTIONS §24):

* **AmplitudeDamping** (``L = √κ â``): a Fock ``|5⟩`` relaxes to the ground state,
  ``⟨n̂⟩(t) = n₀ e^{−κt}``.
* **Heating** (``L₋ = √(κ(n̄+1)) â``, ``L₊ = √(κ n̄) â†``): the ground state relaxes
  *up* to the bath, ``⟨n̂⟩(t) = n̄_bath (1 − e^{−κt})``.
* **Dephasing** (``L = √γ n̂``): a coherent state's coherence quadrature decays,
  ``⟨x̂⟩(t) = ⟨x̂⟩(0) e^{−γt/2}``, while ``⟨n̂⟩`` is preserved (pure phase damping).
* **Windowed Heating** (``window = (0, T/2)``): the same heating, active only on the
  first half — ``⟨n̂⟩`` rises then flattens when the window closes, the sequence-aware
  noise model.

Any dissipative channel forces the master-equation (mesolve) path. Application-agnostic:
textbook decay laws only, no application framing.

Usage::

    python tools/run_benchmark_motional_channels.py

Output::

    benchmarks/data/motional_channels/
      report.json  — per-time ⟨n̂⟩ / ⟨x̂⟩, the closed-form oracles, max error
      arrays.npz   — t, n_heat, n_damp, n_heat_windowed, x_dephase, n_dephase + oracles
      plot.png     — occupation relaxation (up/down/windowed) + coherence decay
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import qutip

from iontrap_dynamics import AmplitudeDamping, Dephasing, Heating
from iontrap_dynamics.conventions import CONVENTION_VERSION
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import Observable
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import coherent_mode, compose_density
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "motional_channels"

MODE = "b"
FOCK_DIM = 30
T_MAX = 1.0e-3
N_TIMES = 80
KAPPA = 3.0e3  # amplitude-damping / heating rate, s⁻¹
N_BATH = 2.0  # heating bath occupation
GAMMA_PHI = 4.0e3  # dephasing rate, s⁻¹
N_INITIAL_DAMP = 5  # initial Fock level for the amplitude-damping run
ALPHA_DEPHASE = 2.0  # coherent amplitude for the dephasing run

PLOT_ALT_TEXT = (
    "Two panels. Left: motional occupation versus time under three channels — heating "
    "rising from the ground state toward the bath occupation of two, amplitude damping "
    "decaying a Fock-five state toward zero, and a windowed heating run that rises then "
    "flattens when its window closes at the half-way point; the closed-form exponential "
    "laws (dashed) overlie the numerics. Right: under pure dephasing the coherence "
    "quadrature of a coherent state decays as exp of minus gamma t over two while the "
    "normalised mean occupation trace stays flat at one — phase damping without energy loss."
)


def _single_mode_hilbert(fock_dim: int) -> HilbertSpace:
    mode = ModeConfig(
        label=MODE,
        frequency_rad_s=2.0 * np.pi * 1.0e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={MODE: fock_dim})


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

    hilbert = _single_mode_hilbert(FOCK_DIM)
    number = hilbert.number_for_mode(MODE)
    a = hilbert.annihilation_for_mode(MODE)
    quad_x = (a + a.dag()) / np.sqrt(2.0)
    zero_h = 0.0 * qutip.tensor(qutip.qeye(2), qutip.qeye(FOCK_DIM))  # pure dissipation, no drift
    times = np.linspace(0.0, T_MAX, N_TIMES)

    ground = qutip.tensor(qutip.basis(2, 0), qutip.basis(FOCK_DIM, 0))
    fock5 = qutip.tensor(qutip.basis(2, 0), qutip.basis(FOCK_DIM, N_INITIAL_DAMP))
    coherent = compose_density(
        hilbert,
        spin_states_per_ion=[spin_down()],
        mode_states_by_label={MODE: coherent_mode(FOCK_DIM, ALPHA_DEPHASE)},
    )

    def _run(initial, obs, channels):
        return solve(
            hilbert=hilbert,
            hamiltonian=zero_h,
            initial_state=initial,
            times=times,
            observables=obs,
            channels=channels,
        )

    obs_n = (Observable(label="n", operator=number),)
    n_heat = np.asarray(
        _run(ground, obs_n, [Heating(mode=MODE, rate=KAPPA, n_bar_bath=N_BATH)]).expectations["n"]
    )
    n_damp = np.asarray(
        _run(fock5, obs_n, [AmplitudeDamping(mode=MODE, rate=KAPPA)]).expectations["n"]
    )
    n_heat_win = np.asarray(
        _run(
            ground,
            obs_n,
            [Heating(mode=MODE, rate=KAPPA, n_bar_bath=N_BATH, window=(0.0, T_MAX / 2.0))],
        ).expectations["n"]
    )
    res_d = _run(
        coherent,
        (Observable(label="x", operator=quad_x), Observable(label="n", operator=number)),
        [Dephasing(mode=MODE, rate=GAMMA_PHI)],
    )
    x_dephase = np.asarray(res_d.expectations["x"], dtype=np.float64)
    n_dephase = np.asarray(res_d.expectations["n"], dtype=np.float64)

    analytic_heat = N_BATH * (1.0 - np.exp(-KAPPA * times))
    analytic_damp = N_INITIAL_DAMP * np.exp(-KAPPA * times)
    analytic_x = x_dephase[0] * np.exp(-GAMMA_PHI * times / 2.0)
    # Windowed heating is active only on [0, T/2): ⟨n̂⟩ rises, then freezes at its T/2 value.
    analytic_heat_windowed = N_BATH * (1.0 - np.exp(-KAPPA * np.minimum(times, T_MAX / 2.0)))

    max_error = float(
        max(
            np.max(np.abs(n_heat - analytic_heat)),
            np.max(np.abs(n_damp - analytic_damp)),
            np.max(np.abs(n_heat_win - analytic_heat_windowed)),
            np.max(np.abs(x_dephase - analytic_x)),
            np.max(np.abs(n_dephase - n_dephase[0])),  # occupation preserved under dephasing
        )
    )
    backend = res_d.metadata.backend_name

    print(">>> motional CPTP channels — closed-form decay laws via solve(channels=...)")
    print(f"backend forced by channels: {backend}")
    print(f"{'t/ms':>6} {'n_heat':>10} {'n_damp':>10} {'x_dephase':>10}")
    print("-" * 40)
    for i in range(0, N_TIMES, max(1, N_TIMES // 8)):
        print(f"{times[i] * 1e3:>6.3f} {n_heat[i]:>10.5f} {n_damp[i]:>10.5f} {x_dephase[i]:>10.5f}")
    print("-" * 40)
    print(f"max |numerical - analytic| = {max_error:.2e}")

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        t=times,
        n_heat=n_heat,
        n_damp=n_damp,
        n_heat_windowed=n_heat_win,
        x_dephase=x_dephase,
        n_dephase=n_dephase,
        analytic_heat=analytic_heat,
        analytic_damp=analytic_damp,
        analytic_heat_windowed=analytic_heat_windowed,
        analytic_x=analytic_x,
    )

    report = {
        "scenario": "motional_channels",
        "purpose": (
            "Typed motional CPTP channels through solve(channels=...): AmplitudeDamping "
            "decays a Fock state to ground as n0 e^{-kt}; Heating relaxes the ground state "
            "up to the bath as n_bath (1 - e^{-kt}); Dephasing decays a coherent state's "
            "coherence quadrature as e^{-g t/2} while preserving the occupation; a windowed "
            "Heating run shows the noise switching off mid-sequence. Any dissipative channel "
            "forces the master-equation path. Application-agnostic: textbook decay laws only."
        ),
        "workplan_reference": "WP/WP-02-two-mode-motional.md (WI-3, dispatch MCC)",
        "schema_version": 2,
        "canonical_request_hash": None,
        "convention_version": CONVENTION_VERSION,
        "backend_name": backend,
        "backend_version": qutip.__version__,
        "rates": {"kappa_s^-1": KAPPA, "n_bar_bath": N_BATH, "gamma_phi_s^-1": GAMMA_PHI},
        "fock_dim": FOCK_DIM,
        "samples": [
            {
                "t_s": float(times[i]),
                "n_heat": float(n_heat[i]),
                "analytic_heat": float(analytic_heat[i]),
                "n_damp": float(n_damp[i]),
                "analytic_damp": float(analytic_damp[i]),
                "n_heat_windowed": float(n_heat_win[i]),
                "analytic_heat_windowed": float(analytic_heat_windowed[i]),
                "x_dephase": float(x_dephase[i]),
                "analytic_x": float(analytic_x[i]),
                "n_dephase": float(n_dephase[i]),
            }
            for i in range(0, N_TIMES, max(1, N_TIMES // 16))
        ],
        "analytic_formulas": {
            "amplitude_damping": "n0 * exp(-kappa t)",
            "heating": "n_bath * (1 - exp(-kappa t))",
            "heating_windowed": "n_bath * (1 - exp(-kappa * min(t, T/2)))  (active on [0, T/2))",
            "dephasing_coherence": "x(0) * exp(-gamma t / 2)  (occupation preserved)",
        },
        "max_numerical_vs_analytic_error": max_error,
        "tolerance": 5e-3,
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

    t_ms = times * 1e3
    fig, (ax_n, ax_c) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax_n.plot(t_ms, analytic_heat, color="#d62728", linestyle="--", linewidth=0.9)
    ax_n.plot(t_ms, analytic_damp, color="#1f77b4", linestyle="--", linewidth=0.9)
    ax_n.scatter(t_ms[::3], n_heat[::3], color="#d62728", s=12, zorder=3, label="heating → bath")
    ax_n.scatter(
        t_ms[::3], n_damp[::3], color="#1f77b4", s=12, zorder=3, label="amplitude damping → 0"
    )
    ax_n.plot(t_ms, n_heat_win, color="#9467bd", linewidth=1.2, label="heating, window (0, T/2)")
    ax_n.axvline(t_ms[-1] / 2.0, color="#9467bd", linestyle=":", linewidth=0.8)
    ax_n.set_xlabel("time (ms)")
    ax_n.set_ylabel(r"occupation $\langle \hat n\rangle$")
    ax_n.set_title("Occupation relaxation")
    ax_n.legend(frameon=False, fontsize=8)

    ax_c.plot(
        t_ms,
        analytic_x / x_dephase[0],
        color="#444444",
        linestyle="--",
        linewidth=0.9,
        label=r"$e^{-\gamma t/2}$",
    )
    ax_c.scatter(
        t_ms[::3],
        x_dephase[::3] / x_dephase[0],
        color="#2ca02c",
        s=12,
        zorder=3,
        label=r"coherence $\langle x\rangle$",
    )
    ax_c.plot(
        t_ms,
        n_dephase / n_dephase[0],
        color="#ff7f0e",
        linewidth=1.2,
        label=r"occupation $\langle n\rangle$ (flat)",
    )
    ax_c.set_xlabel("time (ms)")
    ax_c.set_ylabel("normalised to t = 0")
    ax_c.set_title("Dephasing: coherence decays, energy does not")
    ax_c.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
