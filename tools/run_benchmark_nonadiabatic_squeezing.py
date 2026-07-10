# SPDX-License-Identifier: MIT
"""Benchmark — non-adiabatic squeezing (Wittemer 2020, single ion, compute-only).

WP-05 / Dispatch SQ5. Reproduces the two extreme-regime signatures of Wittemer et
al., *Phil. Trans. R. Soc. A* **378**, 20190230 (2020), for a single trapped-ion
motional mode with a time-dependent trap frequency ω(t) (CONVENTIONS §26):

* **Parametric arm** (paper Fig. 3). Sinusoidal modulation ω(t) = ω_ini + δω·sin(ω_mod t)
  at the resonance ω_mod = 2 ω_ini. The squeezing grows **linearly** in the modulation
  duration, ``r = ½ δω · T_mod`` (leading-order RWA), so ``n̄_sq = sinh²(2π g T_mod)``
  with parametric coupling ``g = δω / (4π)``. Displacement-free (``n_dsp ≡ 0``) — the
  centred quadratic generator preserves parity.
* **Quench arm** (paper Fig. 2). A fast Gaussian quench of ω(t) (fractional amplitude
  ``A = Δω/ω_ini``, width δτ ≪ 2π/ω) drives non-adiabatic squeezing that grows with the
  quench amplitude; also displacement-free for the centred generator (the paper's
  parasitic displacement is a *forced* effect, the optional A3 tail — out of scope here).
* **Analytic oracle**. The §26.1 limits: a narrow squeeze-kick ramp ω_i → ω_f gives
  ``r = ½|ln(ω_f/ω_i)|``; a slow cyclic down/up ramp returning to ω_ini gives ``r → 0``.

Application-agnostic: the library symbols carry no "Wittemer"/"cosmology" framing; this
tool + its oracle are the only place the programme target appears. Readout via the §26.4
covariance eigenvalue ``r`` (not ``tr V``); the SQ4 parity-aware guard runs on every
readout.

Usage::

    python tools/run_benchmark_nonadiabatic_squeezing.py

Output::

    benchmarks/data/nonadiabatic_squeezing/
      report.json  — parametric g_fit vs g_theory (+ paper's 4.64 kHz), quench sweep,
                     the two analytic-limit checks, environment metadata
      arrays.npz   — T_mod, r/n̄_sq (parametric) + oracle; A, r/n̄_sq (quench); limits
      plot.png     — two panels: n̄_sq vs T_mod (parametric + sinh² oracle) and vs A (quench)
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
from iontrap_dynamics.hamiltonians import nonadiabatic_squeezing_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "nonadiabatic_squeezing"

TWOPI = 2.0 * np.pi
OMEGA_INI = TWOPI * 2.8e6  # single-ion mode frequency ω_ini/2π = 2.8 MHz (paper)
DELTA_OMEGA_MOD = TWOPI * 8.0e3  # modulation amplitude δω/2π = 8 kHz (paper)
OMEGA_MOD = 2.0 * OMEGA_INI  # parametric resonance ω_mod = 2 ω_ini
G_THEORY = DELTA_OMEGA_MOD / (4.0 * np.pi)  # RWA parametric coupling (Hz) → 4.0 kHz
G_PAPER_HZ = 4.64e3  # Wittemer 2020 measured value (carries apparatus offset)

PARAM_FOCK = 40
PARAM_PERIODS = (30, 60, 90, 120, 150, 180, 210)  # T_mod in units of 2π/ω_mod

QUENCH_FOCK = 50
QUENCH_AMPLITUDES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)  # fractional A = Δω/ω_ini


def _single_mode(fock: int, freq_rad_s: float = OMEGA_INI) -> HilbertSpace:
    mode = ModeConfig(
        label="m",
        frequency_rad_s=freq_rad_s,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={"m": fock})


def _evolve_readout(
    hilbert: HilbertSpace, wave: waveforms.FrequencyWaveform, tmax: float, n_times: int
) -> gaussian.GaussianReadout:
    fock = hilbert.fock_truncations["m"]
    psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(fock, 0))
    times = np.linspace(0.0, tmax, n_times)
    hamiltonian = nonadiabatic_squeezing_hamiltonian(hilbert, "m", wave, validate_at=(0.0, tmax))
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=psi0,
        times=times,
        storage_mode=StorageMode.EAGER,
    )
    return gaussian.gaussian_readout(gaussian.reduced_single_mode(result.states[-1], hilbert, "m"))


def _parametric_arm() -> dict[str, object]:
    hilbert = _single_mode(PARAM_FOCK)
    period = TWOPI / OMEGA_MOD
    t_mod = np.array([n * period for n in PARAM_PERIODS])
    r_sim = np.zeros_like(t_mod)
    n_sq = np.zeros_like(t_mod)
    n_dsp = np.zeros_like(t_mod)
    for i, tmax in enumerate(t_mod):
        wave = waveforms.sinusoidal_modulation(
            omega_ini=OMEGA_INI, mod_amplitude=DELTA_OMEGA_MOD, mod_frequency=OMEGA_MOD
        )
        readout = _evolve_readout(hilbert, wave, float(tmax), max(2000, int(PARAM_PERIODS[i]) * 15))
        r_sim[i] = readout.squeezing_parameter
        n_sq[i] = readout.mean_squeezed_occupation
        n_dsp[i] = abs(readout.coherent_amplitude) ** 2
    r_oracle = 0.5 * DELTA_OMEGA_MOD * t_mod  # r = ½ δω T_mod
    # slope of r vs T_mod = ½ δω = 2π g  →  g_fit = slope / 2π
    slope = float(np.polyfit(t_mod, r_sim, 1)[0])
    g_fit = slope / TWOPI
    return {
        "t_mod": t_mod,
        "r_sim": r_sim,
        "r_oracle": r_oracle,
        "n_sq": n_sq,
        "n_sq_oracle": np.sinh(r_oracle) ** 2,
        "n_dsp": n_dsp,
        "g_fit_hz": g_fit,
        "g_theory_hz": float(G_THEORY),
        "g_paper_hz": G_PAPER_HZ,
        "max_r_abs_error": float(np.max(np.abs(r_sim - r_oracle))),
        "max_n_dsp": float(np.max(n_dsp)),
    }


def _quench_arm() -> dict[str, object]:
    hilbert = _single_mode(QUENCH_FOCK)
    period = TWOPI / OMEGA_INI
    width = 0.02 * period  # fast (non-adiabatic) quench
    center = 25.0 * width
    tmax = 55.0 * width
    amps = np.array(QUENCH_AMPLITUDES)
    n_sq = np.zeros_like(amps)
    n_dsp = np.zeros_like(amps)
    for i, amp in enumerate(amps):
        wave = waveforms.gaussian_quench(
            omega_ini=OMEGA_INI, amplitude=float(amp), width_s=width, center_s=center
        )
        readout = _evolve_readout(hilbert, wave, tmax, 1500)
        n_sq[i] = readout.mean_squeezed_occupation
        n_dsp[i] = abs(readout.coherent_amplitude) ** 2
    return {
        "amplitude": amps,
        "n_sq": n_sq,
        "n_dsp": n_dsp,
        "monotonic": bool(np.all(np.diff(n_sq) > 0.0)),
        "max_n_dsp": float(np.max(n_dsp)),
    }


def _analytic_limits() -> dict[str, float]:
    # Sudden squeeze-kick: narrow ramp ω_i → ω_f = ½ ω_i → r = ½|ln(ω_f/ω_i)| = ½ ln 2.
    hilbert = _single_mode(40)
    period = TWOPI / OMEGA_INI
    width = 0.01 * period
    kick = waveforms.smooth_ramp(
        omega_i=OMEGA_INI, omega_f=0.5 * OMEGA_INI, center_s=25.0 * width, width_s=width
    )
    r_sudden = _evolve_readout(hilbert, kick, 55.0 * width, 1500).squeezing_parameter
    r_sudden_oracle = 0.5 * abs(np.log(0.5))
    # Cyclic adiabatic: slow down/up ramp returning to ω_ini → r → 0.
    wide = 2.0 * period
    c1, c2 = 5.0 * wide, 15.0 * wide
    ln_half = 0.5 * np.log(0.5)

    def omega(t: float) -> float:
        return OMEGA_INI * float(
            np.exp(ln_half * (np.tanh((t - c1) / wide) - np.tanh((t - c2) / wide)))
        )

    def d_ln_omega_dt(t: float) -> float:
        s1 = 1.0 / np.cosh((t - c1) / wide) ** 2
        s2 = 1.0 / np.cosh((t - c2) / wide) ** 2
        return float(ln_half * (s1 - s2) / wide)

    cyclic = waveforms.FrequencyWaveform(omega=omega, d_ln_omega_dt=d_ln_omega_dt)
    r_cyclic = _evolve_readout(hilbert, cyclic, 20.0 * wide, 2500).squeezing_parameter
    return {
        "r_sudden": float(r_sudden),
        "r_sudden_oracle": float(r_sudden_oracle),
        "r_cyclic_adiabatic": float(r_cyclic),
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
    print(">>> non-adiabatic squeezing benchmark (Wittemer 2020, single ion)")

    parametric = _parametric_arm()
    print(
        f"  parametric: g_fit = {parametric['g_fit_hz'] / 1e3:.3f} kHz "
        f"(theory {parametric['g_theory_hz'] / 1e3:.3f}, paper 4.64); "
        f"max|r−oracle| = {parametric['max_r_abs_error']:.2e}; "
        f"max n_dsp = {parametric['max_n_dsp']:.1e}"
    )
    quench = _quench_arm()
    print(
        f"  quench: n̄_sq {quench['n_sq'][0]:.3f} → {quench['n_sq'][-1]:.3f} over A "
        f"{QUENCH_AMPLITUDES[0]}–{QUENCH_AMPLITUDES[-1]}; monotonic = {quench['monotonic']}; "
        f"max n_dsp = {quench['max_n_dsp']:.1e}"
    )
    limits = _analytic_limits()
    print(
        f"  limits: sudden r = {limits['r_sudden']:.4f} (oracle {limits['r_sudden_oracle']:.4f}); "
        f"cyclic-adiabatic r = {limits['r_cyclic_adiabatic']:.4f} (→ 0)"
    )

    np.savez(
        OUTPUT_DIR / "arrays.npz",
        param_t_mod=parametric["t_mod"],
        param_r_sim=parametric["r_sim"],
        param_r_oracle=parametric["r_oracle"],
        param_n_sq=parametric["n_sq"],
        param_n_sq_oracle=parametric["n_sq_oracle"],
        param_n_dsp=parametric["n_dsp"],
        quench_amplitude=quench["amplitude"],
        quench_n_sq=quench["n_sq"],
        quench_n_dsp=quench["n_dsp"],
    )

    report = {
        "benchmark": "nonadiabatic_squeezing",
        "target": "Wittemer et al., Phil. Trans. R. Soc. A 378, 20190230 (2020), single ion",
        "convention": "CONVENTIONS.md §26 (WP-05 / dispatch SQ5)",
        "omega_ini_over_2pi_hz": OMEGA_INI / TWOPI,
        "delta_omega_mod_over_2pi_hz": DELTA_OMEGA_MOD / TWOPI,
        "parametric": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in parametric.items()
        },
        "quench": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in quench.items()},
        "analytic_limits": limits,
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

    fig, (ax_p, ax_q) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    t_us = parametric["t_mod"] * 1e6
    ax_p.plot(
        t_us,
        parametric["n_sq_oracle"],
        "-",
        color="#1f77b4",
        label=r"$\sinh^2(\delta\omega\,T_\mathrm{mod}/2)$",
    )
    ax_p.plot(t_us, parametric["n_sq"], "o", color="#d62728", label="simulation")
    ax_p.set_xlabel(r"$T_\mathrm{mod}$ [µs]")
    ax_p.set_ylabel(r"$\bar n_\mathrm{sq}$")
    ax_p.set_title(f"parametric (g = {parametric['g_fit_hz'] / 1e3:.2f} kHz)")
    ax_p.legend()

    ax_q.plot(quench["amplitude"], quench["n_sq"], "s-", color="#2ca02c")
    ax_q.set_xlabel(r"quench amplitude $A = \Delta\omega/\omega_\mathrm{ini}$")
    ax_q.set_ylabel(r"$\bar n_\mathrm{sq}$")
    ax_q.set_title("quench (displacement-free)")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
