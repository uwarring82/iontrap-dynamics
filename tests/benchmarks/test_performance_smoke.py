# SPDX-License-Identifier: MIT
"""Phase 0.F performance smoke-test scaffold.

Three canonical benchmarks, frozen against the laptop hardware listed in
``WORKPLAN_v0.3.md`` §0.F (2023 MacBook Air M2 or equivalent):

- Single-ion sideband flopping   (``N_Fock = 30``, 200 steps)   → < 5 s
- Two-ion Mølmer–Sørensen gate   (``N_Fock = 15``, 500 steps)   → < 30 s
- Stroboscopic AC-π/2 drive      (``N_Fock = 40``, 1000 steps)  → < 60 s

Each benchmark is currently skipped pending the Phase 1 builder it
exercises. Replacing ``@pytest.mark.skip(...)`` with a real build-and-solve
body is the unit of work that activates each gate — the three test slots
live here so Phase 1 PRs do not need to invent a new file, and so
``pytest -m benchmark`` already lists the complete target set today.

Breach policy (workplan §0.F)
----------------------------

If a threshold is exceeded after a builder lands, review immediately.
Options include algorithmic optimisation, sparse-path refinement, targeted
caching, or elevating JAX-backend work from Phase 2 to Phase 1. The
integrator selects proportionate to root cause.

Running
-------

``pytest tests/benchmarks -v`` — shows all three as SKIPPED with reason.
``pytest -m benchmark``       — same, filtered by marker.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


# Thresholds are the workplan §0.F contract. Reproduced here as symbolic
# constants so a builder-landing PR can replace the skip without rereading
# the workplan to find the number.
CANONICAL_HARDWARE = "2023 MacBook Air M2 or equivalent"

T_SINGLE_ION_SIDEBAND = 5.0  # seconds
T_TWO_ION_MS_GATE = 30.0  # seconds
T_STROBOSCOPIC_AC = 60.0  # seconds


def test_single_ion_sideband_flopping_under_5s() -> None:
    """Single-ion red-sideband flopping: ``N_Fock = 30``, 200 steps.

    Threshold: ``T_SINGLE_ION_SIDEBAND`` = 5.0 s wall time on the canonical
    hardware. Builds the same scenario as
    ``tools/run_benchmark_sideband.py`` but asserts only on timing —
    physics correctness is already covered by the analytic-regression
    tests and the sideband-Hamiltonian unit tests.

    Setup: ²⁵Mg⁺, 1.5 MHz axial mode, 280 nm drive aligned along +z,
    Ω/2π = 0.1 MHz carrier Rabi. Initial state |↓, 1⟩. Evolve over
    two sideband Rabi periods at 200 tlist samples.
    """
    import time

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
    from iontrap_dynamics.operators import spin_down
    from iontrap_dynamics.species import mg25_plus
    from iontrap_dynamics.system import IonSystem

    n_fock = 30
    rabi_rad_s = 2 * np.pi * 0.1e6
    mode_freq_rad_s = 2 * np.pi * 1.5e6
    wavenumber_m_inv = 2 * np.pi / 280e-9

    mode = ModeConfig(
        label="axial",
        frequency_rad_s=mode_freq_rad_s,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"axial": n_fock})

    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, wavenumber_m_inv],
        carrier_rabi_frequency_rad_s=rabi_rad_s,
    )
    H = red_sideband_hamiltonian(hilbert, drive, "axial", ion_index=0)

    eta = lamb_dicke_parameter(
        k_vec=drive.k_vector_m_inv,
        mode_eigenvector=mode.eigenvector_at_ion(0),
        ion_mass=system.species(0).mass_kg,
        mode_frequency=mode_freq_rad_s,
    )
    sb_rate = red_sideband_rabi_frequency(
        carrier_rabi_frequency=rabi_rad_s,
        lamb_dicke_parameter=eta,
        n_initial=1,
    )
    tlist = np.linspace(0.0, 2 * 2 * np.pi / sb_rate, 200)
    psi_0 = qutip.tensor(spin_down(), qutip.basis(n_fock, 1))

    t0 = time.perf_counter()
    qutip.mesolve(H, psi_0, tlist, [], [])
    elapsed = time.perf_counter() - t0

    assert elapsed < T_SINGLE_ION_SIDEBAND, (
        f"took {elapsed:.2f}s (threshold {T_SINGLE_ION_SIDEBAND}s)"
    )


def test_two_ion_ms_gate_under_30s() -> None:
    """Two-ion Mølmer–Sørensen gate (δ = 0 bichromatic): ``N_Fock = 15``,
    500 steps.

    Threshold: ``T_TWO_ION_MS_GATE`` = 30.0 s wall time on the canonical
    hardware. The smaller Fock truncation reflects the larger Hilbert
    dimension (two spins + one mode) relative to the single-ion case.

    Setup: two ²⁵Mg⁺ ions sharing a 1.5 MHz axial COM mode, 280 nm drive
    aligned along +z, Ω/2π = 0.1 MHz. Initial state |↓↓, 0⟩. Evolve
    over one full coherent-displacement period.
    """
    import time

    import numpy as np
    import qutip

    from iontrap_dynamics.analytic import lamb_dicke_parameter
    from iontrap_dynamics.drives import DriveConfig
    from iontrap_dynamics.hamiltonians import ms_gate_hamiltonian
    from iontrap_dynamics.hilbert import HilbertSpace
    from iontrap_dynamics.modes import ModeConfig
    from iontrap_dynamics.operators import spin_down
    from iontrap_dynamics.species import mg25_plus
    from iontrap_dynamics.system import IonSystem

    n_fock = 15
    rabi_rad_s = 2 * np.pi * 0.1e6
    mode_freq_rad_s = 2 * np.pi * 1.5e6
    wavenumber_m_inv = 2 * np.pi / 280e-9

    mode = ModeConfig(
        label="com",
        frequency_rad_s=mode_freq_rad_s,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]) / np.sqrt(2.0),
    )
    system = IonSystem(species_per_ion=(mg25_plus(), mg25_plus()), modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"com": n_fock})

    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, wavenumber_m_inv],
        carrier_rabi_frequency_rad_s=rabi_rad_s,
    )
    H = ms_gate_hamiltonian(hilbert, drive, "com", ion_indices=(0, 1))

    eta = lamb_dicke_parameter(
        k_vec=drive.k_vector_m_inv,
        mode_eigenvector=mode.eigenvector_at_ion(0),
        ion_mass=system.species(0).mass_kg,
        mode_frequency=mode_freq_rad_s,
    )
    # MS displacement period T ≈ 2π/(Ωη) — one full phase-space loop for the
    # |++⟩ component. Ground-state initial means the actual observable
    # oscillation period differs, but this is the right dimensional scale
    # and puts 500 samples on a physically representative grid.
    tlist = np.linspace(0.0, 2 * np.pi / (rabi_rad_s * abs(eta)), 500)
    psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(n_fock, 0))

    t0 = time.perf_counter()
    qutip.mesolve(H, psi_0, tlist, [], [])
    elapsed = time.perf_counter() - t0

    assert elapsed < T_TWO_ION_MS_GATE, f"took {elapsed:.2f}s (threshold {T_TWO_ION_MS_GATE}s)"


@pytest.mark.skip(
    reason=(
        "awaiting Phase 1 stroboscopic AC-π/2 builder (workplan §0.F item 3); "
        "replace skip with build/solve and assert elapsed < T_STROBOSCOPIC_AC."
    )
)
def test_stroboscopic_ac_halfpi_under_60s() -> None:
    """Stroboscopic AC-π/2 drive: ``N_Fock = 40``, 1000 steps.

    Threshold: ``T_STROBOSCOPIC_AC`` = 60.0 s wall time on the canonical
    hardware. The 1000-step count reflects the fast time-dependence of
    stroboscopic drives — the threshold is deliberately the loosest of the
    three.
    """
    raise AssertionError("unreachable — test is skipped")


# ----------------------------------------------------------------------------
# Meta-sanity: the benchmarks scaffold is registered, reachable, and parametric.
# These tests are NOT skipped — they verify the scaffold's integrity so that a
# broken benchmark module cannot hide behind three silent skips.
# ----------------------------------------------------------------------------


class TestScaffoldSanity:
    def test_thresholds_are_positive(self) -> None:
        assert T_SINGLE_ION_SIDEBAND > 0
        assert T_TWO_ION_MS_GATE > 0
        assert T_STROBOSCOPIC_AC > 0

    def test_thresholds_ordering_matches_workplan(self) -> None:
        """Single-ion is fastest, stroboscopic is slowest — matches the
        problem complexity ordering in workplan §0.F."""
        assert T_SINGLE_ION_SIDEBAND < T_TWO_ION_MS_GATE < T_STROBOSCOPIC_AC

    def test_canonical_hardware_declared(self) -> None:
        assert "M2" in CANONICAL_HARDWARE, (
            "workplan §0.F names the canonical laptop; keep that reference "
            "visible so benchmark owners know what the threshold is calibrated against."
        )
