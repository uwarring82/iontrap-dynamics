# SPDX-License-Identifier: MIT
"""Gaussian-shaped π-pulse end-to-end demo through sequences.solve().

Companion to ``run_demo_carrier.py``. Where that demo shows a static
carrier drive, this one exercises the **time-dependent list-format
path** landed in ``modulated_carrier_hamiltonian`` — a smooth
Gaussian envelope whose pulse area

    ∫₀^T Ω · f(t) dt = π

delivers a clean |↓⟩ → |↑⟩ π-rotation. The expected trajectories
follow the integrated rotation angle θ(t):

    θ(t) = ∫₀^t Ω · f(t') dt'
    ⟨σ_x⟩(t) = 0
    ⟨σ_y⟩(t) = sin(θ(t))
    ⟨σ_z⟩(t) = −cos(θ(t))

so the state smoothly traces a meridian on the Bloch sphere in the
y–z plane, ending at the north pole when the pulse area reaches π.

Why this demo exists
--------------------

It is the first tool that drives the **full Phase 1 public API**
through a single entry point:

- configuration via :class:`IonSystem` + :class:`DriveConfig` +
  :class:`ModeConfig`
- Hilbert construction via :class:`HilbertSpace`
- initial state via :func:`ground_state`
- time-dependent Hamiltonian via
  :func:`modulated_carrier_hamiltonian`
- named observables via :func:`spin_x` / :func:`spin_y` /
  :func:`spin_z`
- solver dispatch via :func:`solve`

Anything that breaks the Phase 1 public surface will trip this
demo visibly: the comparison to the analytic θ(t) curves amplifies
small bugs (a mis-embedded operator, a wrong sign in σ_y, a
list-format regression) into a visible error on the plot.

Usage::

    python tools/run_demo_gaussian_pulse.py

Requires matplotlib (``pip install -e ".[plot]"``). Falls back to
"data only, no plot" if matplotlib is absent.

Output::

    benchmarks/data/gaussian_pi_pulse_demo/
      arrays.npz     — times_us, envelope, sigma_x/y/z (+ analytic)
      metadata.json  — parameters, environment, analytic formulas used
      plot.png       — four-panel figure: envelope + Bloch components
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

from iontrap_dynamics.cache import compute_request_hash, save_trajectory
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import modulated_carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_x, spin_y, spin_z
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import ground_state
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "gaussian_pi_pulse_demo"

# Physical parameters — on-resonance, zero phase, Gaussian envelope
N_FOCK = 3  # small; the carrier does not couple motion in v0.1
N_STEPS = 400

RABI_OVER_2PI_MHZ = 1.0
MODE_FREQ_OVER_2PI_MHZ = 1.5  # only used so HilbertSpace has a mode to attach to
LASER_WAVELENGTH_NM = 280.0

PULSE_DURATION_US = 5.0
PULSE_CENTRE_US = PULSE_DURATION_US / 2.0
PULSE_SIGMA_US = PULSE_DURATION_US / 10.0  # narrow enough that truncation error < 1e-30

RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)

PULSE_DURATION_S = PULSE_DURATION_US * 1e-6
PULSE_CENTRE_S = PULSE_CENTRE_US * 1e-6
PULSE_SIGMA_S = PULSE_SIGMA_US * 1e-6

# Envelope amplitude normalised so the pulse area ∫Ω·f(t) dt = π.
# For a centred Gaussian well inside [0, T]:
#   area ≈ Ω · A · σ · √(2π)
# (the erf correction from clipping at the window endpoints is <1e-30
# for σ = T/10 centred at T/2, so we ignore it.)
ENVELOPE_AMPLITUDE = np.pi / (RABI_RAD_S * PULSE_SIGMA_S * np.sqrt(2 * np.pi))


def gaussian_envelope(t: float) -> float:
    return float(
        ENVELOPE_AMPLITUDE * math.exp(-((t - PULSE_CENTRE_S) ** 2) / (2 * PULSE_SIGMA_S**2))
    )


def _rotation_angle(tlist: np.ndarray) -> np.ndarray:
    """Return θ(t) = ∫₀^t Ω · f(t') dt' via cumulative trapezoidal rule.

    Uses a finer internal grid than ``tlist`` so the integrated analytic
    curve is not limited by the sampling density of the solver's output.
    """
    fine = np.linspace(tlist[0], tlist[-1], 20 * len(tlist))
    f_fine = np.array([gaussian_envelope(t) for t in fine])
    omega_f_fine = RABI_RAD_S * f_fine
    integral = np.concatenate(
        ([0.0], np.cumsum(0.5 * (omega_f_fine[:-1] + omega_f_fine[1:])) * np.diff(fine)[0])
    )
    return np.interp(tlist, fine, integral)


def _build_scenario() -> tuple[HilbertSpace, list[object], qutip.Qobj]:
    """Return (hilbert, time-dependent list Hamiltonian, initial |↓, 0⟩ ket)."""
    mode = ModeConfig(
        label="axial",
        frequency_rad_s=MODE_FREQ_RAD_S,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"axial": N_FOCK})

    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, WAVENUMBER_M_INV],
        carrier_rabi_frequency_rad_s=RABI_RAD_S,
        phase_rad=0.0,
    )
    hamiltonian = modulated_carrier_hamiltonian(
        hilbert, drive, ion_index=0, envelope=gaussian_envelope
    )
    psi_0 = ground_state(hilbert)
    return hilbert, hamiltonian, psi_0


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

    hilbert, hamiltonian, psi_0 = _build_scenario()
    tlist = np.linspace(0.0, PULSE_DURATION_S, N_STEPS)

    print(">>> running Gaussian π-pulse demo (sequences.solve + modulated carrier)")
    print(f"    N_Fock = {N_FOCK}, steps = {N_STEPS}, duration = {PULSE_DURATION_US:.1f} μs")
    print(f"    Ω/2π = {RABI_OVER_2PI_MHZ} MHz, pulse σ = {PULSE_SIGMA_US:.2f} μs")
    print(f"    pulse peak amplitude f_max = {ENVELOPE_AMPLITUDE:.3f}")

    # Parameters — frozen inputs, canonical-cache hash binding
    parameters = {
        "scenario": "gaussian_pi_pulse_demo",
        "N_fock": N_FOCK,
        "n_steps": N_STEPS,
        "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
        "phase_rad": 0.0,
        "pulse_duration_us": PULSE_DURATION_US,
        "pulse_centre_us": PULSE_CENTRE_US,
        "pulse_sigma_us": PULSE_SIGMA_US,
        "envelope_amplitude": ENVELOPE_AMPLITUDE,
        "initial_state": "|↓, 0⟩",
    }
    request_hash = compute_request_hash(parameters)

    t0 = time.perf_counter()
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=psi_0,
        times=tlist,
        observables=[spin_x(hilbert, 0), spin_y(hilbert, 0), spin_z(hilbert, 0)],
        request_hash=request_hash,
        storage_mode=StorageMode.OMITTED,
        provenance_tags=("demo", "gaussian_pi_pulse"),
    )
    elapsed = time.perf_counter() - t0
    print(f"    elapsed: {elapsed:.3f} s")

    sx_traj = result.expectations["sigma_x_0"]
    sy_traj = result.expectations["sigma_y_0"]
    sz_traj = result.expectations["sigma_z_0"]

    # Analytic comparison — integrated rotation angle θ(t) = ∫Ω·f(t') dt'
    # gives ⟨σ_y⟩(t) = sin(θ), ⟨σ_z⟩(t) = −cos(θ), ⟨σ_x⟩(t) = 0.
    theta = _rotation_angle(tlist)
    sx_analytic = np.zeros_like(tlist)
    sy_analytic = np.sin(theta)
    sz_analytic = -np.cos(theta)
    envelope_samples = np.array([gaussian_envelope(t) for t in tlist])

    max_error = max(
        float(np.max(np.abs(sx_traj - sx_analytic))),
        float(np.max(np.abs(sy_traj - sy_analytic))),
        float(np.max(np.abs(sz_traj - sz_analytic))),
    )
    final_theta = float(theta[-1])
    print(f"    ∫Ω·f dt (pulse area) = {final_theta:.6f} rad (target π = {np.pi:.6f})")
    print(f"    max |numerical − analytic| = {max_error:.3e}")
    print(f"    final ⟨σ_z⟩ = {sz_traj[-1]:+.6f} (target +1.0 for a π-pulse)")

    # ------------------------------------------------------------------------
    # Canonical cache (manifest.json + arrays.npz, hash-verified)
    # ------------------------------------------------------------------------
    save_trajectory(result, OUTPUT_DIR, overwrite=True)

    # ------------------------------------------------------------------------
    # Demo-specific arrays (envelope shape, integrated θ, analytic overlays).
    # Kept separate from arrays.npz so the cache layout stays canonical.
    # ------------------------------------------------------------------------
    np.savez(
        OUTPUT_DIR / "analytic_overlay.npz",
        envelope=envelope_samples,
        rotation_angle=theta,
        sigma_x_analytic=sx_analytic,
        sigma_y_analytic=sy_analytic,
        sigma_z_analytic=sz_analytic,
    )

    demo_report = {
        "scenario": "gaussian_pi_pulse_demo",
        "purpose": (
            "end-to-end Phase 1 public API exercise — sequences.solve + "
            "modulated_carrier_hamiltonian + named observables"
        ),
        "workplan_reference": (
            "not a Phase 0.F benchmark; complements run_benchmark_sideband.py "
            "(static) and run_demo_carrier.py (static carrier) with a "
            "time-dependent list-format end-to-end trajectory"
        ),
        "analytic_formulas": {
            "rotation_angle": "θ(t) = ∫₀^t Ω · f(t') dt'",
            "sigma_x_0": "0",
            "sigma_y_0": "sin(θ(t))",
            "sigma_z_0": "-cos(θ(t))",
        },
        "elapsed_seconds": elapsed,
        "final_pulse_area_rad": final_theta,
        "final_sigma_z": float(sz_traj[-1]),
        "max_numerical_vs_analytic_error": max_error,
        "parameters": parameters,
        "arrays_schema_note": (
            "arrays.npz follows the canonical cache schema: 'times' in SI "
            "seconds, observables under 'expectation__<label>'. The Gaussian "
            "envelope, integrated rotation angle θ(t), and analytic "
            "overlays live separately in analytic_overlay.npz."
        ),
        "canonical_request_hash": request_hash,
        "environment": _environment(),
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": 2,
    }
    (OUTPUT_DIR / "demo_report.json").write_text(
        json.dumps(demo_report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/"
        "{manifest.json, arrays.npz, analytic_overlay.npz, demo_report.json}"
    )

    # ------------------------------------------------------------------------
    # Plot — four panels: envelope, ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩
    # ------------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not installed; skipping plot")
        return 0

    times_us = tlist * 1e6  # human-readable axis; canonical arrays store seconds

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8.0, 8.0))

    axes[0].plot(times_us, envelope_samples, color="#9467bd", linewidth=2.0)
    axes[0].fill_between(times_us, 0.0, envelope_samples, color="#9467bd", alpha=0.2)
    axes[0].set_ylabel(r"envelope $f(t)$")
    axes[0].set_ylim(bottom=0.0)
    axes[0].axhline(0.0, color="grey", linewidth=0.3)

    for ax, label, num, ana, color in (
        (axes[1], r"$\langle \sigma_x \rangle$", sx_traj, sx_analytic, "#2ca02c"),
        (axes[2], r"$\langle \sigma_y \rangle$", sy_traj, sy_analytic, "#ff7f0e"),
        (axes[3], r"$\langle \sigma_z \rangle$", sz_traj, sz_analytic, "#1f77b4"),
    ):
        ax.plot(times_us, num, color=color, linewidth=2.0, label="sequences.solve")
        ax.plot(
            times_us,
            ana,
            color="black",
            linewidth=0.8,
            linestyle="--",
            label="analytic",
        )
        ax.axhline(0.0, color="grey", linewidth=0.3)
        ax.set_ylabel(label)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel(r"time (μs)")
    axes[0].set_title(
        f"Gaussian π-pulse through sequences.solve + modulated_carrier — "
        f"pulse area = {final_theta:.4f} rad, max |Δ| = {max_error:.1e}"
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
