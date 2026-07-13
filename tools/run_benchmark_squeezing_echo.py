# SPDX-License-Identifier: MIT
"""Benchmark — forced displacement, single-pulse optimisation & the purifying echo (SQ6).

WP-05 / Dispatch SQ6 (the optional A3 tail). Reproduces the *forced-displacement* physics
of Wittemer et al., *Phil. Trans. R. Soc. A* 378, 20190230 (2020) / *PRL* 123, 180502
(2019), that the centred squeezing generator (§26.1, parity-preserving, ⟨â⟩=0) deliberately
leaves out. Three arms:

* **Single-pulse optimisation** (paper: "iteratively adjust δτ and Δω to find maximal |r|").
  A down/up trap-frequency pulse (`waveforms.down_up_pulse`) squeezes by an amount that
  **oscillates with the hold time** (the WKB phase ∫ω dt'): the down- and up-ramp
  contributions add constructively at the right hold and cancel at others (hold → 0 gives
  the adiabatic-cyclic r → 0). Scanning the hold finds the constructive optimum — which
  roughly **doubles** the squeezing of a one-way ramp of the same depth.
* **Forced displacement.** A linear force `displacement_force_hamiltonian` (H/ℏ = f·x̂) seeds
  a coherent displacement α (dα/dt = −i f; §26.4/§7): |α| scales **linearly** with the force,
  independent of the squeezing (the off-RF-null offset of the paper).
* **Purifying echo.** Two quench pulses, each with its parasitic displacement, separated by a
  free evolution t_free. The displacement rotates at ω and the squeezing ellipse at 2ω, so at
  the optimal t_free (≈ half a trap period, shifted by the finite pulse width) the second
  pulse's displacement **cancels** the first's while the squeezing **adds** — the parasitic
  displacement is suppressed by δp = n_dsp(1 pulse)/n_dsp(2 pulses) ≫ 1 (paper Fig. 2b).

Application-agnostic: the library carries no "Wittemer" framing; this tool is the only place
the target appears. Additive — no new convention symbol / no CONVENTION_VERSION bump (the
force composes the sealed §26.2 quadrature x̂ with a real coefficient; §26.4/MCF·ND precedent).

Usage::

    python tools/run_benchmark_squeezing_echo.py

Output::

    benchmarks/data/squeezing_echo/
      report.json  — single-pulse r(hold) + optimum, |α|(F) linearity, echo δp(t_free) + null
      arrays.npz   — hold/r; F/|α|; t_free/δp/|α|/r
      plot.png     — three panels: r vs hold; |α| vs F; δp vs t_free
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import qutip

from iontrap_dynamics import gaussian, waveforms
from iontrap_dynamics.conventions import CONVENTION_VERSION
from iontrap_dynamics.hamiltonians import (
    displacement_force_hamiltonian,
    nonadiabatic_squeezing_hamiltonian,
)
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "squeezing_echo"

TWOPI = 2.0 * np.pi
OMEGA_INI = TWOPI * 2.0e6  # ω_ini/2π = 2 MHz → trap period T = 0.5 µs
PERIOD = 1.0 / 2.0e6
FOCK = 50


def _single_mode() -> HilbertSpace:
    mode = ModeConfig(
        label="m", frequency_rad_s=OMEGA_INI, eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]])
    )
    return HilbertSpace(
        system=IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,)),
        fock_truncations={"m": FOCK},
    )


def _readout(hilbert: HilbertSpace, hamiltonian: object, tmax: float, n_times: int):
    psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(FOCK, 0))
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=psi0,
        times=np.linspace(0.0, tmax, n_times),
        storage_mode=StorageMode.EAGER,
    )
    return gaussian.gaussian_readout(gaussian.reduced_single_mode(result.states[-1], hilbert, "m"))


def _single_pulse_arm(hilbert: HilbertSpace) -> dict[str, object]:
    ramp = 0.02 * PERIOD
    holds = np.linspace(0.0, 1.0 * PERIOD, 21)
    r_of_hold = np.zeros_like(holds)
    for i, hold in enumerate(holds):
        center = 10.0 * ramp + 0.5 * hold
        pulse = waveforms.down_up_pulse(
            omega_ini=OMEGA_INI,
            omega_min=0.5 * OMEGA_INI,
            ramp_width_s=ramp,
            hold_s=float(hold),
            center_s=center,
        )
        tmax = center + 0.5 * hold + 15.0 * ramp
        r_of_hold[i] = _readout(
            hilbert,
            nonadiabatic_squeezing_hamiltonian(hilbert, "m", pulse, validate_at=(0.0, tmax)),
            tmax,
            1500,
        ).squeezing_parameter
    best = int(np.argmax(r_of_hold))
    return {
        "hold": holds,
        "r": r_of_hold,
        "hold_opt_periods": float(holds[best] / PERIOD),
        "r_opt": float(r_of_hold[best]),
        "r_naive_single_ramp": 0.5
        * abs(np.log(0.5)),  # ½|ln(ω_f/ω_i)| for one ramp of the same depth
    }


def _forced_displacement_arm(hilbert: HilbertSpace) -> dict[str, object]:
    const = waveforms.FrequencyWaveform(omega=lambda t: OMEGA_INI, d_ln_omega_dt=lambda t: 0.0)
    forces = np.array([2.5e6, 5.0e6, 1.0e7, 2.0e7])
    abs_alpha = np.zeros_like(forces)
    tmax = 6.0 * 0.02 * PERIOD
    for i, force in enumerate(forces):
        hamiltonian = nonadiabatic_squeezing_hamiltonian(
            hilbert, "m", const, validate_at=(0.0, tmax)
        ) + displacement_force_hamiltonian(hilbert, "m", lambda t, force=force: force)
        abs_alpha[i] = abs(_readout(hilbert, hamiltonian, tmax, 600).coherent_amplitude)
    # linearity: |α| / F should be constant
    ratios = abs_alpha / forces
    return {
        "force": forces,
        "abs_alpha": abs_alpha,
        "linearity_rel_spread": float((ratios.max() - ratios.min()) / ratios.mean()),
    }


def _echo_arm(hilbert: HilbertSpace) -> dict[str, object]:
    amp = -0.4  # a down-quench (Δω/ω_ini) — squeezes
    width = 0.02 * PERIOD
    force_amp = 2.0e7
    c1 = 20.0 * width

    def _pulse(centers: list[float], tmax: float):
        def bump(t: float) -> float:
            return sum(np.exp(-0.5 * ((t - c) / width) ** 2) for c in centers)

        def bump_prime(t: float) -> float:
            return sum(-(t - c) / width**2 * np.exp(-0.5 * ((t - c) / width) ** 2) for c in centers)

        wave = waveforms.FrequencyWaveform(
            omega=lambda t: OMEGA_INI * (1.0 + amp * bump(t)),
            d_ln_omega_dt=lambda t: amp * bump_prime(t) / (1.0 + amp * bump(t)),
        )

        def force(t: float) -> float:
            return force_amp * sum(np.exp(-0.5 * ((t - c) / width) ** 2) for c in centers)

        hamiltonian = nonadiabatic_squeezing_hamiltonian(
            hilbert, "m", wave, validate_at=(0.0, tmax)
        ) + displacement_force_hamiltonian(hilbert, "m", force)
        return _readout(hilbert, hamiltonian, tmax, 2500)

    one = _pulse([c1], c1 + 20.0 * width)
    n_dsp_one = abs(one.coherent_amplitude) ** 2
    t_free = np.linspace(0.40 * PERIOD, 0.60 * PERIOD, 21)
    delta_p = np.zeros_like(t_free)
    abs_alpha = np.zeros_like(t_free)
    r_echo = np.zeros_like(t_free)
    for i, tf in enumerate(t_free):
        two = _pulse([c1, c1 + tf], c1 + tf + 20.0 * width)
        n_dsp_two = abs(two.coherent_amplitude) ** 2
        delta_p[i] = n_dsp_one / n_dsp_two
        abs_alpha[i] = abs(two.coherent_amplitude)
        r_echo[i] = two.squeezing_parameter
    best = int(np.argmax(delta_p))
    return {
        "t_free": t_free,
        "delta_p": delta_p,
        "abs_alpha": abs_alpha,
        "r_echo": r_echo,
        "r_one_pulse": float(one.squeezing_parameter),
        "abs_alpha_one_pulse": float(abs(one.coherent_amplitude)),
        "t_free_opt_periods": float(t_free[best] / PERIOD),
        "delta_p_max": float(delta_p[best]),
        "r_echo_at_opt": float(r_echo[best]),
    }


def _environment() -> dict[str, str]:
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "qutip": qutip.__version__,
        "convention_version": CONVENTION_VERSION,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hilbert = _single_mode()
    print(">>> forced-displacement / single-pulse-optimisation / echo benchmark (SQ6)")

    single = _single_pulse_arm(hilbert)
    print(
        f"  single-pulse: r_opt = {single['r_opt']:.4f} at hold = {single['hold_opt_periods']:.2f}·T "
        f"(vs one-way ramp {single['r_naive_single_ramp']:.4f} → "
        f"{single['r_opt'] / single['r_naive_single_ramp']:.2f}×)"
    )
    forced = _forced_displacement_arm(hilbert)
    print(
        f"  forced displacement: |α| linear in F (rel spread of |α|/F = "
        f"{forced['linearity_rel_spread']:.2e})"
    )
    echo = _echo_arm(hilbert)
    print(
        f"  echo: δp_max = {echo['delta_p_max']:.0f} at t_free = {echo['t_free_opt_periods']:.2f}·T; "
        f"r {echo['r_one_pulse']:.3f} (1 pulse) → {echo['r_echo_at_opt']:.3f} (echo); "
        f"|α| {echo['abs_alpha_one_pulse']:.3f} → {echo['abs_alpha'][int(np.argmax(echo['delta_p']))]:.3f}"
    )

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        single_hold=single["hold"],
        single_r=single["r"],
        forced_force=forced["force"],
        forced_abs_alpha=forced["abs_alpha"],
        echo_t_free=echo["t_free"],
        echo_delta_p=echo["delta_p"],
        echo_abs_alpha=echo["abs_alpha"],
        echo_r=echo["r_echo"],
    )

    def _jsonable(d: dict[str, object]) -> dict[str, object]:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    report = {
        "benchmark": "squeezing_echo",
        "target": "Wittemer 2020 (Phil Trans R Soc A 378 20190230) / 2019 (PRL 123 180502), forced displacement + echo",
        "convention": "CONVENTIONS.md §26 (additive — no new symbol / no version bump); WP-05 dispatch SQ6",
        "omega_ini_over_2pi_hz": OMEGA_INI / TWOPI,
        "single_pulse_optimisation": _jsonable(single),
        "forced_displacement": _jsonable(forced),
        "echo": _jsonable(echo),
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

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14.0, 4.2))
    ax0.plot(single["hold"] / PERIOD, single["r"], "o-", color="#1f77b4")
    ax0.axhline(single["r_naive_single_ramp"], color="#444444", ls=":", label="one-way ramp")
    ax0.set_xlabel("hold time / trap period")
    ax0.set_ylabel("squeezing r")
    ax0.set_title("single-pulse optimisation")
    ax0.legend()

    ax1.plot(forced["force"] / 1e6, forced["abs_alpha"], "s-", color="#2ca02c")
    ax1.set_xlabel("force f [rad/µs]")
    ax1.set_ylabel(r"displacement $|\alpha|$")
    ax1.set_title("forced displacement (linear)")

    ax2.semilogy(echo["t_free"] / PERIOD, echo["delta_p"], "o-", color="#d62728")
    ax2.axhline(1.0, color="#444444", ls=":", label="no suppression")
    ax2.set_xlabel("free evolution / trap period")
    ax2.set_ylabel(r"$\delta_p = n_{dsp}^{(1)}/n_{dsp}^{(2)}$")
    ax2.set_title("purifying echo")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
