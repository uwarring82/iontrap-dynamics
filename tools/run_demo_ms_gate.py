# SPDX-License-Identifier: MIT
"""Detuned Mølmer–Sørensen Bell-gate demo through sequences.solve().

Companion to ``run_demo_gaussian_pulse.py``. That tool showed the
time-dependent list-format path on a single ion (modulated carrier);
this one exercises the same path on a two-ion, one-mode system with
the **gate-closing** detuned MS Hamiltonian. The textbook result:

    δ       = 2 |Ω η| √K   (loop-closing + Bell-state condition)
    t_gate  = π √K / |Ω η|
    |↓↓, 0⟩ → (|↓↓⟩ − i |↑↑⟩) / √2 ⊗ |0⟩    at  t = t_gate

At t_gate the motion has traced K closed loops in phase space and
returned to vacuum; the spin sector has picked up a π/4 rotation
on σ_x^{(0)} σ_x^{(1)} that maps the input product state into a
Bell state with ``P(|↓↓⟩) = P(|↑↑⟩) = 1/2``.

Why this demo exists
--------------------

The first tool that drives the full Phase 1 stack through a
**two-ion** list-format solve — exercising:

- two-ion :class:`IonSystem` + COM mode :class:`ModeConfig`
- :func:`detuned_ms_gate_hamiltonian` (list format)
- analytic helpers :func:`ms_gate_closing_detuning`,
  :func:`ms_gate_closing_time`
- named spin observables + manually constructed two-ion projectors
- :func:`solve` dispatch

Any regression in the list-format dispatch or the MS-gate algebra
shows up as a visible deviation on the plot: motion failing to
return to vacuum at ``t_gate``, populations missing the 0.5 mark,
or the two ions desynchronising.

Usage::

    python tools/run_demo_ms_gate.py

Requires matplotlib (``pip install -e ".[plot]"``). Falls back to
"data only, no plot" if matplotlib is absent.

Output::

    benchmarks/data/ms_gate_bell_demo/
      arrays.npz     — times_us, n_mode, p_dd, p_uu, p_flip, sz_0, sz_1
      metadata.json  — parameters, environment, closing-condition values
      plot.png       — three-panel figure: ⟨n̂⟩, Bell populations, σ_z
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
    ms_gate_closing_detuning,
    ms_gate_closing_time,
)
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import detuned_ms_gate_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import Observable, number, spin_z
from iontrap_dynamics.operators import spin_down, spin_up
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "data" / "ms_gate_bell_demo"

# Physical parameters — two ²⁵Mg⁺ ions sharing a 1.5 MHz axial COM mode
N_FOCK = 12
N_STEPS = 500
LOOPS = 1  # K: number of phase-space loops before the gate closes

RABI_OVER_2PI_MHZ = 0.1
MODE_FREQ_OVER_2PI_MHZ = 1.5
LASER_WAVELENGTH_NM = 280.0

RABI_RAD_S = 2 * np.pi * RABI_OVER_2PI_MHZ * 1e6
MODE_FREQ_RAD_S = 2 * np.pi * MODE_FREQ_OVER_2PI_MHZ * 1e6
WAVENUMBER_M_INV = 2 * np.pi / (LASER_WAVELENGTH_NM * 1e-9)


def _build_scenario() -> tuple[HilbertSpace, list[object], qutip.Qobj, float, float, float]:
    """Return (hilbert, list-format H, psi_0 (|↓↓, 0⟩), eta, delta, t_gate)."""
    mode = ModeConfig(
        label="com",
        frequency_rad_s=MODE_FREQ_RAD_S,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]) / np.sqrt(2.0),
    )
    system = IonSystem(species_per_ion=(mg25_plus(), mg25_plus()), modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"com": N_FOCK})

    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, WAVENUMBER_M_INV],
        carrier_rabi_frequency_rad_s=RABI_RAD_S,
        phase_rad=0.0,
    )

    eta = lamb_dicke_parameter(
        k_vec=drive.k_vector_m_inv,
        mode_eigenvector=mode.eigenvector_at_ion(0),
        ion_mass=mg25_plus().mass_kg,
        mode_frequency=MODE_FREQ_RAD_S,
    )
    delta = ms_gate_closing_detuning(
        carrier_rabi_frequency=RABI_RAD_S,
        lamb_dicke_parameter=eta,
        loops=LOOPS,
    )
    t_gate = ms_gate_closing_time(
        carrier_rabi_frequency=RABI_RAD_S,
        lamb_dicke_parameter=eta,
        loops=LOOPS,
    )

    hamiltonian = detuned_ms_gate_hamiltonian(
        hilbert, drive, "com", ion_indices=(0, 1), detuning_rad_s=delta
    )
    psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(N_FOCK, 0))
    return hilbert, hamiltonian, psi_0, eta, delta, t_gate


def _bell_population_observables(hilbert: HilbertSpace) -> list[Observable]:
    """Construct the three Bell-sector population projectors as
    :class:`Observable` records ready for :func:`solve`.

    ``p_dd``   — ``|↓↓⟩⟨↓↓| ⊗ I_mode``
    ``p_uu``   — ``|↑↑⟩⟨↑↑| ⊗ I_mode``
    ``p_flip`` — ``(|↓↑⟩⟨↓↑| + |↑↓⟩⟨↑↓|) ⊗ I_mode`` (odd-parity, stays
    at 0 throughout an ideal MS gate because the Hamiltonian
    conserves total parity).
    """
    n_fock = hilbert.fock_truncations["com"]
    i_mode = qutip.qeye(n_fock)

    dd = qutip.ket2dm(qutip.tensor(spin_down(), spin_down()))
    du = qutip.ket2dm(qutip.tensor(spin_down(), spin_up()))
    ud = qutip.ket2dm(qutip.tensor(spin_up(), spin_down()))
    uu = qutip.ket2dm(qutip.tensor(spin_up(), spin_up()))

    return [
        Observable(label="p_dd", operator=qutip.tensor(dd, i_mode)),
        Observable(label="p_uu", operator=qutip.tensor(uu, i_mode)),
        Observable(label="p_flip", operator=qutip.tensor(du + ud, i_mode)),
    ]


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

    hilbert, hamiltonian, psi_0, eta, delta, t_gate = _build_scenario()
    tlist = np.linspace(0.0, t_gate, N_STEPS)

    print(">>> running detuned MS gate demo (sequences.solve + list-format)")
    print(f"    N_Fock = {N_FOCK}, steps = {N_STEPS}")
    print(f"    Ω/2π = {RABI_OVER_2PI_MHZ} MHz, η = {eta:.4f}")
    print(f"    Bell detuning   δ = {delta / (2 * np.pi * 1e3):.2f} × 2π kHz")
    print(f"    gate time   t_gate = {t_gate * 1e6:.2f} μs")
    print(f"    loops               K = {LOOPS}")

    observables = [
        number(hilbert, "com"),
        spin_z(hilbert, 0),
        spin_z(hilbert, 1),
        *_bell_population_observables(hilbert),
    ]

    t0 = time.perf_counter()
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=psi_0,
        times=tlist,
        observables=observables,
        storage_mode=StorageMode.OMITTED,
        provenance_tags=("demo", "ms_gate_bell"),
    )
    elapsed = time.perf_counter() - t0
    print(f"    elapsed: {elapsed:.3f} s")

    n_traj = result.expectations["n_com"]
    sz_0 = result.expectations["sigma_z_0"]
    sz_1 = result.expectations["sigma_z_1"]
    p_dd = result.expectations["p_dd"]
    p_uu = result.expectations["p_uu"]
    p_flip = result.expectations["p_flip"]

    # Final-state sanity check — should land on Bell-fidelity indicators
    final_p_dd = float(p_dd[-1])
    final_p_uu = float(p_uu[-1])
    final_p_flip = float(p_flip[-1])
    final_n = float(n_traj[-1])
    final_sz_0 = float(sz_0[-1])
    final_sz_1 = float(sz_1[-1])
    max_sz_asymmetry = float(np.max(np.abs(sz_0 - sz_1)))

    print()
    print("    --- final-state sanity (expected at Bell condition) ---")
    print(f"    P(|↓↓⟩)      = {final_p_dd:.6f}   (target 0.5)")
    print(f"    P(|↑↑⟩)      = {final_p_uu:.6f}   (target 0.5)")
    print(f"    P_flip       = {final_p_flip:.6f}   (target 0)")
    print(f"    ⟨n̂⟩          = {final_n:.6f}   (target 0 — loop closes)")
    print(f"    ⟨σ_z^(0)⟩    = {final_sz_0:.6f}   (target 0)")
    print(f"    ⟨σ_z^(1)⟩    = {final_sz_1:.6f}   (target 0)")
    print(f"    max|Δσ_z|    = {max_sz_asymmetry:.3e}   (ion-exchange symmetry)")

    # ------------------------------------------------------------------------
    # Save arrays + metadata
    # ------------------------------------------------------------------------
    times_us = tlist * 1e6
    np.savez(
        OUTPUT_DIR / "arrays.npz",
        times_us=times_us,
        n_mode=n_traj,
        sigma_z_0=sz_0,
        sigma_z_1=sz_1,
        p_dd=p_dd,
        p_uu=p_uu,
        p_flip=p_flip,
    )

    metadata = {
        "scenario": "ms_gate_bell_demo",
        "purpose": (
            "end-to-end detuned MS gate demonstration — solve() + list-format "
            "Hamiltonian + two-ion population projectors as custom observables"
        ),
        "workplan_reference": (
            "not a Phase 0.F benchmark; complements run_demo_gaussian_pulse.py "
            "with the two-ion list-format path"
        ),
        "analytic_predictions": {
            "final_p_dd_target": 0.5,
            "final_p_uu_target": 0.5,
            "final_p_flip_target": 0.0,
            "final_n_mode_target": 0.0,
            "final_sigma_z_target": 0.0,
        },
        "final_state": {
            "p_dd": final_p_dd,
            "p_uu": final_p_uu,
            "p_flip": final_p_flip,
            "n_mode": final_n,
            "sigma_z_0": final_sz_0,
            "sigma_z_1": final_sz_1,
            "max_sigma_z_asymmetry": max_sz_asymmetry,
        },
        "elapsed_seconds": elapsed,
        "parameters": {
            "N_fock": N_FOCK,
            "n_steps": N_STEPS,
            "loops_K": LOOPS,
            "rabi_over_2pi_MHz": RABI_OVER_2PI_MHZ,
            "mode_freq_over_2pi_MHz": MODE_FREQ_OVER_2PI_MHZ,
            "laser_wavelength_nm": LASER_WAVELENGTH_NM,
            "lamb_dicke_eta": eta,
            "bell_detuning_rad_s": delta,
            "bell_detuning_over_2pi_kHz": delta / (2 * np.pi * 1e3),
            "gate_time_us": t_gate * 1e6,
            "initial_state": "|↓↓, 0⟩",
        },
        "provenance_tags": list(result.metadata.provenance_tags),
        "convention_version": result.metadata.convention_version,
        "storage_mode": result.metadata.storage_mode.value,
        "backend_name": result.metadata.backend_name,
        "backend_version": result.metadata.backend_version,
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
    # Plot — three stacked panels
    # ------------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not installed; skipping plot")
        return 0

    t_gate_us = t_gate * 1e6

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.0, 7.0))

    # Panel 1: ⟨n̂⟩(t) — the phase-space loop closing
    axes[0].plot(times_us, n_traj, color="#9467bd", linewidth=2.0)
    axes[0].axhline(0.0, color="grey", linewidth=0.3)
    axes[0].axvline(t_gate_us, color="black", linewidth=0.6, linestyle=":", alpha=0.7)
    axes[0].set_ylabel(r"$\langle \hat n \rangle$")
    axes[0].set_title(
        "Detuned MS gate — phase-space loop closes and spins form a Bell state "
        f"at t = {t_gate_us:.2f} μs"
    )

    # Panel 2: Bell populations
    axes[1].plot(
        times_us, p_dd, color="#1f77b4", linewidth=2.0, label=r"$P(|\!\downarrow\downarrow\rangle)$"
    )
    axes[1].plot(
        times_us, p_uu, color="#d62728", linewidth=2.0, label=r"$P(|\!\uparrow\uparrow\rangle)$"
    )
    axes[1].plot(
        times_us, p_flip, color="#2ca02c", linewidth=1.4, label=r"$P_{\mathrm{flip}}$ (odd parity)"
    )
    axes[1].axhline(0.5, color="grey", linewidth=0.3, linestyle=":")
    axes[1].axvline(t_gate_us, color="black", linewidth=0.6, linestyle=":", alpha=0.7)
    axes[1].set_ylabel("population")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="center right", fontsize=8)

    # Panel 3: σ_z per ion (overlaid, should be identical throughout)
    axes[2].plot(
        times_us, sz_0, color="#ff7f0e", linewidth=2.4, label=r"$\langle \sigma_z^{(0)} \rangle$"
    )
    axes[2].plot(
        times_us,
        sz_1,
        color="black",
        linewidth=0.9,
        linestyle="--",
        label=r"$\langle \sigma_z^{(1)} \rangle$",
    )
    axes[2].axhline(0.0, color="grey", linewidth=0.3)
    axes[2].axvline(t_gate_us, color="black", linewidth=0.6, linestyle=":", alpha=0.7)
    axes[2].set_ylabel(r"$\langle \sigma_z \rangle$")
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_xlabel(r"time (μs)")
    axes[2].legend(loc="center right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}/plot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
