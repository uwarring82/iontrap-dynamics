# SPDX-License-Identifier: MIT
r"""Run the Phase 0.F single-ion sideband-flopping benchmark and capture artefacts.

Runs the same scenario as
``tests/benchmarks/test_performance_smoke.py::test_single_ion_sideband_flopping_under_5s``
but additionally:

1. Captures the full time-series of ⟨σ_z⟩(t) and ⟨n̂⟩(t).
2. Writes the arrays and run metadata into
   ``benchmarks/data/01_single_ion_sideband_flopping/``.
3. Generates a two-panel PNG (``plot.png``) for reviewer inspection.

Usage::

    python tools/run_benchmark_sideband.py

Requires matplotlib (``pip install -e ".[plot]"``). The pytest-gated
timing check does NOT depend on matplotlib — this tool exists solely
for human review and for docs-site illustrations.

Physics setup (workplan §0.F item 1)
-----------------------------------

Single ²⁵Mg⁺ ion, axial mode (1.5 MHz), 280 nm drive aligned along the
mode axis so k ∥ b (full Lamb–Dicke projection). Initial state: |↓, 1⟩.
Red-sideband resonance (caller-asserted), interaction-picture
Hamiltonian from :func:`iontrap_dynamics.hamiltonians.red_sideband_hamiltonian`.

Duration: 2 × (one sideband Rabi period) = two full flop cycles between
|↓, 1⟩ and |↑, 0⟩. Time grid: 200 evenly-spaced samples
(matches the "200 steps" in workplan §0.F).

Fock truncation: N=30 (workplan §0.F). Generous for a single-phonon
starting state but matches the benchmark specification exactly.

Threshold: 5.0 s wall time on the canonical laptop. On an M2 MacBook
Air the actual solve runs in well under a second; the benchmark mostly
catches regressions under future operator-cache or backend changes.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import qutip

from iontrap_dynamics.analytic import (
    lamb_dicke_parameter,
    red_sideband_rabi_frequency,
)
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import red_sideband_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.operators import sigma_z_ion, spin_down
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "01_single_ion_sideband_flopping"

# Canonical parameters (workplan §0.F item 1)
N_FOCK = 30
N_STEPS = 200

# Physical parameters
RABI_OVER_2PI_MHZ = 0.1
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

# Derived
RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6  # rad/s
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6  # rad/s
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)  # m⁻¹


def _build_scenario() -> tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, float, float]:
    """Construct HilbertSpace, H, and the |↓, 1⟩ initial state. Returns
    also the Lamb–Dicke parameter η and the red-sideband Rabi rate."""
    mode = ModeConfig(
        label="axial",
        frequency_rad_s=MODE_FREQ_RAD_S,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),  # aligned with +z
    )
    system = IonSystem.homogeneous(
        species=mg25_plus(),
        n_ions=1,
        modes=(mode,),
    )
    hilbert = HilbertSpace(system=system, fock_truncations={"axial": N_FOCK})

    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, WAVENUMBER_M_INV],  # k ∥ b
        carrier_rabi_frequency_rad_s=RABI_RAD_S,
    )
    hamiltonian = red_sideband_hamiltonian(hilbert, drive, "axial", ion_index=0)

    eta = lamb_dicke_parameter(
        k_vec=drive.k_vector_m_inv,
        mode_eigenvector=mode.eigenvector_at_ion(0),
        ion_mass=system.species(0).mass_kg,
        mode_frequency=MODE_FREQ_RAD_S,
    )
    sb_rate = red_sideband_rabi_frequency(
        carrier_rabi_frequency=RABI_RAD_S,
        lamb_dicke_parameter=eta,
        n_initial=1,
    )

    psi_0 = qutip.tensor(spin_down(), qutip.basis(N_FOCK, 1))  # |↓, 1⟩
    return hilbert, hamiltonian, psi_0, eta, sb_rate


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

    hilbert, hamiltonian, psi_0, eta, sb_rate = _build_scenario()
    sb_period = 2 * np.pi / sb_rate
    # Two sideband Rabi periods, 200 samples
    tlist = np.linspace(0.0, 2 * sb_period, N_STEPS)

    print(">>> running single-ion sideband flopping benchmark")
    print(f"    N_Fock = {N_FOCK}, steps = {N_STEPS}")
    print(f"    Ω/2π = {RABI_OVER_2PI_MHZ} MHz, ω_mode/2π = {MODE_FREQ_OVER_2PI_MHZ} MHz")
    print(f"    η = {eta:.4f}, Ω_SB/(2π) = {sb_rate / (2 * np.pi) / 1e3:.3f} kHz")
    print(f"    duration = {tlist[-1] * 1e6:.2f} μs (~2 sideband Rabi periods)")

    t0 = time.perf_counter()
    result = qutip.mesolve(hamiltonian, psi_0, tlist, [], [])
    elapsed = time.perf_counter() - t0
    print(f"    elapsed: {elapsed:.3f} s (threshold 5.0 s)")

    sz_op = hilbert.spin_op_for_ion(sigma_z_ion(), 0)
    n_op = hilbert.number_for_mode("axial")
    sigma_z_traj = np.array([qutip.expect(sz_op, st) for st in result.states])
    n_traj = np.array([qutip.expect(n_op, st) for st in result.states])

    # ------------------------------------------------------------------------
    # Save arrays + metadata
    # ------------------------------------------------------------------------
    times_us = tlist * 1e6
    np.savez(
        OUTPUT_DIR / "arrays.npz",
        times_us=times_us,
        sigma_z=sigma_z_traj,
        n_mode=n_traj,
    )

    metadata = {
        "scenario": "01_single_ion_sideband_flopping",
        "workplan_reference": "§0.F item 1",
        "threshold_seconds": 5.0,
        "elapsed_seconds": elapsed,
        "parameters": {
            "N_fock": N_FOCK,
            "n_steps": N_STEPS,
            "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
            "mode_frequency_over_2pi_MHz": MODE_FREQ_OVER_2PI_MHZ,
            "laser_wavelength_nm": LASER_WAVELENGTH_NM,
            "initial_state": "|↓, 1⟩",
            "eta": eta,
            "sideband_rabi_rate_over_2pi_kHz": sb_rate / (2 * np.pi) / 1e3,
            "duration_us": times_us[-1],
        },
        "environment": _environment(),
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": 1,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/arrays.npz + metadata.json")

    # ------------------------------------------------------------------------
    # Plot (two stacked panels)
    # ------------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not installed; skipping plot (pip install -e '.[plot]')")
        return 0

    fig, (ax_spin, ax_motion) = plt.subplots(2, 1, sharex=True, figsize=(8.0, 5.0))

    ax_spin.plot(times_us, sigma_z_traj, color="#1f77b4", linewidth=1.8)
    ax_spin.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
    ax_spin.set_ylabel(r"$\langle \sigma_z \rangle$")
    ax_spin.set_ylim(-1.05, 1.05)
    ax_spin.set_title(
        f"Single-ion red-sideband flopping — N_Fock={N_FOCK}, {N_STEPS} steps, "
        f"elapsed {elapsed * 1000:.0f} ms"
    )

    ax_motion.plot(times_us, n_traj, color="#d62728", linewidth=1.8)
    ax_motion.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
    ax_motion.axhline(1.0, color="grey", linewidth=0.5, linestyle=":")
    ax_motion.set_ylabel(r"$\langle \hat{n} \rangle$")
    ax_motion.set_xlabel(r"time (μs)")
    ax_motion.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0 if elapsed < 5.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
