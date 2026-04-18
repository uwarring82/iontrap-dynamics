# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.hamiltonians`."""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics.analytic import (
    blue_sideband_rabi_frequency,
    lamb_dicke_parameter,
    red_sideband_rabi_frequency,
)
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.exceptions import ConventionError, IonTrapError
from iontrap_dynamics.hamiltonians import (
    blue_sideband_hamiltonian,
    carrier_hamiltonian,
    red_sideband_hamiltonian,
)
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.operators import sigma_x_ion, sigma_y_ion, spin_down, spin_up
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import ground_state
from iontrap_dynamics.system import IonSystem

# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _single_ion_axial_mode() -> ModeConfig:
    return ModeConfig(
        label="axial",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )


def _two_ion_com_mode() -> ModeConfig:
    return ModeConfig(
        label="com",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]) / np.sqrt(2.0),
    )


def _single_ion_hilbert(*, fock: int = 5) -> HilbertSpace:
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(_single_ion_axial_mode(),))
    return HilbertSpace(system=system, fock_truncations={"axial": fock})


def _simple_drive(*, phase_rad: float = 0.0, rabi: float = 2 * np.pi * 1e6) -> DriveConfig:
    return DriveConfig(
        k_vector_m_inv=[2e7, 0.0, 0.0],
        carrier_rabi_frequency_rad_s=rabi,
        phase_rad=phase_rad,
    )


# ----------------------------------------------------------------------------
# Structural correctness
# ----------------------------------------------------------------------------


class TestStructure:
    def test_dims_match_full_hilbert(self) -> None:
        h = _single_ion_hilbert()
        H = carrier_hamiltonian(h, _simple_drive(), ion_index=0)
        assert H.dims == h.qutip_dims()

    def test_hermitian(self) -> None:
        """Carrier Hamiltonian must be Hermitian for any phase."""
        h = _single_ion_hilbert()
        for phi in (0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3):
            H = carrier_hamiltonian(h, _simple_drive(phase_rad=phi), ion_index=0)
            assert (H - H.dag()).norm() < 1e-14, f"non-Hermitian at φ={phi}"

    def test_returns_qutip_qobj(self) -> None:
        H = carrier_hamiltonian(_single_ion_hilbert(), _simple_drive(), ion_index=0)
        assert isinstance(H, qutip.Qobj)


# ----------------------------------------------------------------------------
# Phase conventions (CONVENTIONS.md §5)
# ----------------------------------------------------------------------------


class TestPhaseConvention:
    def test_phi_zero_gives_half_rabi_sigma_x(self) -> None:
        """At φ=0, H/ℏ = (Ω/2) σ_x (embedded on the target ion)."""
        h = _single_ion_hilbert()
        omega = 2 * np.pi * 1e6
        H = carrier_hamiltonian(h, _simple_drive(phase_rad=0.0, rabi=omega), ion_index=0)
        expected = (omega / 2.0) * h.spin_op_for_ion(sigma_x_ion(), 0)
        assert (H - expected).norm() < 1e-14

    def test_phi_pi_over_two_gives_minus_half_rabi_sigma_y(self) -> None:
        """At φ=π/2, H/ℏ = −(Ω/2) σ_y (per CONVENTIONS.md §5
        rewrite using σ_y_ion = −i(σ_+ − σ_−))."""
        h = _single_ion_hilbert()
        omega = 2 * np.pi * 1e6
        H = carrier_hamiltonian(h, _simple_drive(phase_rad=np.pi / 2, rabi=omega), ion_index=0)
        expected = -(omega / 2.0) * h.spin_op_for_ion(sigma_y_ion(), 0)
        assert (H - expected).norm() < 1e-10

    def test_phi_pi_gives_minus_half_rabi_sigma_x(self) -> None:
        """At φ=π, H/ℏ = −(Ω/2) σ_x."""
        h = _single_ion_hilbert()
        omega = 2 * np.pi * 1e6
        H = carrier_hamiltonian(h, _simple_drive(phase_rad=np.pi, rabi=omega), ion_index=0)
        expected = -(omega / 2.0) * h.spin_op_for_ion(sigma_x_ion(), 0)
        assert (H - expected).norm() < 1e-10


# ----------------------------------------------------------------------------
# Action on states
# ----------------------------------------------------------------------------


class TestAction:
    def test_phi_zero_acts_as_sigma_x_on_ground(self) -> None:
        """H |↓, 0⟩ = (Ω/2) σ_x |↓⟩ ⊗ |0⟩ = (Ω/2) |↑, 0⟩."""
        h = _single_ion_hilbert()
        omega = 2 * np.pi * 1e6
        H = carrier_hamiltonian(h, _simple_drive(rabi=omega), ion_index=0)
        psi_down = ground_state(h)
        psi_up = qutip.tensor(spin_up(), qutip.basis(5, 0))
        # (Ω/2) |↑, 0⟩
        expected = (omega / 2.0) * psi_up
        diff = (H * psi_down) - expected
        assert diff.norm() < 1e-10

    def test_phi_zero_acts_as_sigma_x_on_excited(self) -> None:
        """H |↑, 0⟩ = (Ω/2) σ_x |↑⟩ ⊗ |0⟩ = (Ω/2) |↓, 0⟩."""
        h = _single_ion_hilbert()
        omega = 2 * np.pi * 1e6
        H = carrier_hamiltonian(h, _simple_drive(rabi=omega), ion_index=0)
        psi_up = qutip.tensor(spin_up(), qutip.basis(5, 0))
        psi_down = ground_state(h)
        expected = (omega / 2.0) * psi_down
        diff = (H * psi_up) - expected
        assert diff.norm() < 1e-10


# ----------------------------------------------------------------------------
# Multi-ion targeting (only addresses the specified ion)
# ----------------------------------------------------------------------------


class TestMultiIonTargeting:
    def test_drive_on_ion_1_leaves_ion_0_untouched(self) -> None:
        """Driving ion 1 on a two-ion crystal should leave ion 0 in
        its original state (the embedding places identity at ion 0)."""
        system = IonSystem(
            species_per_ion=(mg25_plus(), mg25_plus()),
            modes=(_two_ion_com_mode(),),
        )
        h = HilbertSpace(system=system, fock_truncations={"com": 4})
        H = carrier_hamiltonian(h, _simple_drive(), ion_index=1)

        # Start |↓, ↓, 0⟩ and apply H — it should produce (Ω/2)|↓, ↑, 0⟩
        # (σ_x on ion 1, identity on ion 0 and mode)
        psi_0 = ground_state(h)
        result = H * psi_0
        expected = (_simple_drive().carrier_rabi_frequency_rad_s / 2.0) * qutip.tensor(
            spin_down(), spin_up(), qutip.basis(4, 0)
        )
        diff = result - expected
        assert diff.norm() < 1e-10

    def test_carrier_ion_1_and_ion_0_are_different_operators(self) -> None:
        """H(ion_0) ≠ H(ion_1) — the embedding respects ion_index."""
        system = IonSystem(
            species_per_ion=(mg25_plus(), mg25_plus()),
            modes=(_two_ion_com_mode(),),
        )
        h = HilbertSpace(system=system, fock_truncations={"com": 4})
        drive = _simple_drive()
        H0 = carrier_hamiltonian(h, drive, ion_index=0)
        H1 = carrier_hamiltonian(h, drive, ion_index=1)
        assert (H0 - H1).norm() > 1e-6  # Genuinely different operators


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


class TestValidation:
    def test_detuned_drive_rejected(self) -> None:
        h = _single_ion_hilbert()
        drive = DriveConfig(
            k_vector_m_inv=[1.0, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
            detuning_rad_s=1e3,  # non-zero
        )
        with pytest.raises(ConventionError, match="on-resonance"):
            carrier_hamiltonian(h, drive, ion_index=0)

    def test_out_of_range_ion_index_raises(self) -> None:
        h = _single_ion_hilbert()
        with pytest.raises(IndexError):
            carrier_hamiltonian(h, _simple_drive(), ion_index=1)

    def test_validation_errors_subclass_iontraperror(self) -> None:
        h = _single_ion_hilbert()
        drive = DriveConfig(
            k_vector_m_inv=[1.0, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
            detuning_rad_s=-1.0,
        )
        with pytest.raises(IonTrapError):
            carrier_hamiltonian(h, drive, ion_index=0)


# ----------------------------------------------------------------------------
# Dynamics sanity — drive |↓⟩ with the carrier Hamiltonian and verify a
# π-pulse flips the state. This closes the loop: builder output plugged
# into qutip.mesolve reproduces the analytic carrier-flopping formula.
# ----------------------------------------------------------------------------


class TestDynamicsSanity:
    def test_pi_pulse_flips_ground_to_excited(self) -> None:
        """A π-pulse at t_π = π/Ω should flip |↓⟩ to |↑⟩ at unit fidelity
        (within mesolve numerical precision)."""
        from iontrap_dynamics.operators import sigma_z_ion

        h = _single_ion_hilbert(fock=3)  # small Fock — carrier doesn't couple motion
        omega = 2 * np.pi * 1e6  # 1 MHz Rabi
        H = carrier_hamiltonian(h, _simple_drive(rabi=omega), ion_index=0)

        psi_0 = ground_state(h)
        t_pi = np.pi / omega
        # mesolve with no collapse operators → unitary evolution
        result = qutip.mesolve(H, psi_0, [0.0, t_pi], [], [])
        psi_final = result.states[-1]

        sigma_z = h.spin_op_for_ion(sigma_z_ion(), 0)
        # σ_z = −1 at start, +1 at π-pulse (atomic-physics convention)
        sz_initial = qutip.expect(sigma_z, psi_0)
        sz_final = qutip.expect(sigma_z, psi_final)
        assert sz_initial == pytest.approx(-1.0)
        assert sz_final == pytest.approx(+1.0, abs=1e-6)

    def test_half_pi_pulse_gives_equal_superposition(self) -> None:
        """A π/2-pulse should give ⟨σ_z⟩ = 0 (equal superposition)."""
        from iontrap_dynamics.operators import sigma_z_ion

        h = _single_ion_hilbert(fock=3)
        omega = 2 * np.pi * 1e6
        H = carrier_hamiltonian(h, _simple_drive(rabi=omega), ion_index=0)

        psi_0 = ground_state(h)
        t_half_pi = np.pi / (2 * omega)
        result = qutip.mesolve(H, psi_0, [0.0, t_half_pi], [], [])
        psi_final = result.states[-1]

        sigma_z = h.spin_op_for_ion(sigma_z_ion(), 0)
        assert qutip.expect(sigma_z, psi_final) == pytest.approx(0.0, abs=1e-6)


# ============================================================================
# Sideband fixtures — physical parameters for ²⁵Mg⁺ axial mode, 280 nm drive.
# ============================================================================

# Scalar wavenumber |k| for a 280 nm laser (Mg S↔P transition).
_K_MAGNITUDE = 2 * np.pi / 280e-9  # m⁻¹


def _sideband_drive(*, rabi: float = 2 * np.pi * 0.1e6, phase_rad: float = 0.0) -> DriveConfig:
    """Drive aligned along +z (so k ∥ b for the axial mode → full LD projection)."""
    return DriveConfig(
        k_vector_m_inv=[0.0, 0.0, _K_MAGNITUDE],
        carrier_rabi_frequency_rad_s=rabi,
        phase_rad=phase_rad,
    )


def _expected_eta(h: HilbertSpace) -> float:
    """Compute η for the fixtures above, for cross-checks against builder output."""
    drive = _sideband_drive()
    species = h.system.species(0)
    mode = h.system.mode("axial")
    return lamb_dicke_parameter(
        k_vec=drive.k_vector_m_inv,
        mode_eigenvector=mode.eigenvector_at_ion(0),
        ion_mass=species.mass_kg,
        mode_frequency=mode.frequency_rad_s,
    )


# ----------------------------------------------------------------------------
# Red sideband: structure + Hermiticity
# ----------------------------------------------------------------------------


class TestRedSidebandStructure:
    def test_dims_match_full_hilbert(self) -> None:
        h = _single_ion_hilbert(fock=5)
        H = red_sideband_hamiltonian(h, _sideband_drive(), "axial", ion_index=0)
        assert H.dims == h.qutip_dims()

    def test_hermitian_for_any_phase(self) -> None:
        h = _single_ion_hilbert(fock=5)
        for phi in (0.0, np.pi / 3, np.pi / 2, np.pi, -np.pi / 4):
            H = red_sideband_hamiltonian(h, _sideband_drive(phase_rad=phi), "axial", ion_index=0)
            assert (H - H.dag()).norm() < 1e-12


# ----------------------------------------------------------------------------
# Red sideband: vacuum is frozen, |↓, 1⟩ flops to |↑, 0⟩
# ----------------------------------------------------------------------------


class TestRedSidebandAction:
    def test_vacuum_is_annihilated(self) -> None:
        """H_RSB |↓, 0⟩ = 0: σ_+ a on |↓,0⟩ = |↑⟩·a|0⟩ = 0;
        σ_- a† on |↓,0⟩ = 0·a†|0⟩ = 0. The vacuum red sideband
        has no state to lower into, so the Hamiltonian annihilates it."""
        h = _single_ion_hilbert(fock=5)
        H = red_sideband_hamiltonian(h, _sideband_drive(), "axial", ion_index=0)
        psi_vac = ground_state(h)  # |↓, 0⟩
        assert (H * psi_vac).norm() < 1e-12

    def test_fock_one_maps_to_excited_vacuum(self) -> None:
        """H_RSB |↓, 1⟩ = (Ωη/2)·e^{iφ}·|↑, 0⟩ (with φ=0, prefactor is real
        and positive)."""
        h = _single_ion_hilbert(fock=5)
        rabi = 2 * np.pi * 0.1e6
        drive = _sideband_drive(rabi=rabi)
        H = red_sideband_hamiltonian(h, drive, "axial", ion_index=0)

        psi_down_1 = qutip.tensor(spin_down(), qutip.basis(5, 1))
        result = H * psi_down_1

        eta = _expected_eta(h)
        psi_up_0 = qutip.tensor(spin_up(), qutip.basis(5, 0))
        expected = (rabi * eta / 2.0) * psi_up_0
        diff = result - expected
        assert diff.norm() < 1e-10


# ----------------------------------------------------------------------------
# Red sideband: π-pulse via mesolve at the analytic rate |η|·Ω
# ----------------------------------------------------------------------------


class TestRedSidebandDynamics:
    def test_pi_pulse_from_down_one_to_up_zero(self) -> None:
        """A sideband π-pulse at t = π/(|η|·Ω) should transfer |↓, 1⟩ to
        |↑, 0⟩ with fidelity ~1 (within mesolve numerical precision)."""
        from iontrap_dynamics.operators import sigma_z_ion

        h = _single_ion_hilbert(fock=10)
        rabi = 2 * np.pi * 0.1e6
        drive = _sideband_drive(rabi=rabi)
        H = red_sideband_hamiltonian(h, drive, "axial", ion_index=0)

        eta = _expected_eta(h)
        rate = red_sideband_rabi_frequency(
            carrier_rabi_frequency=rabi,
            lamb_dicke_parameter=eta,
            n_initial=1,
        )
        t_pi = np.pi / rate

        psi_0 = qutip.tensor(spin_down(), qutip.basis(10, 1))
        result = qutip.mesolve(H, psi_0, [0.0, t_pi], [], [])
        psi_final = result.states[-1]

        # σ_z = −1 at start (|↓⟩), +1 at the end (|↑⟩). ⟨n̂⟩: 1 → 0.
        sigma_z = h.spin_op_for_ion(sigma_z_ion(), 0)
        n_op = h.number_for_mode("axial")
        assert qutip.expect(sigma_z, psi_0) == pytest.approx(-1.0)
        assert qutip.expect(sigma_z, psi_final) == pytest.approx(+1.0, abs=1e-4)
        assert qutip.expect(n_op, psi_0) == pytest.approx(1.0)
        assert qutip.expect(n_op, psi_final) == pytest.approx(0.0, abs=1e-4)


# ----------------------------------------------------------------------------
# Blue sideband: vacuum couples, sqrt(n+1) scaling
# ----------------------------------------------------------------------------


class TestBlueSidebandAction:
    def test_vacuum_maps_to_excited_one_phonon(self) -> None:
        """H_BSB |↓, 0⟩ = (Ωη/2)·e^{iφ}·|↑, 1⟩: σ_+ a† creates a phonon."""
        h = _single_ion_hilbert(fock=5)
        rabi = 2 * np.pi * 0.1e6
        drive = _sideband_drive(rabi=rabi)
        H = blue_sideband_hamiltonian(h, drive, "axial", ion_index=0)

        psi_down_0 = ground_state(h)
        result = H * psi_down_0

        eta = _expected_eta(h)
        psi_up_1 = qutip.tensor(spin_up(), qutip.basis(5, 1))
        expected = (rabi * eta / 2.0) * psi_up_1
        diff = result - expected
        assert diff.norm() < 1e-10

    def test_hermitian_for_any_phase(self) -> None:
        h = _single_ion_hilbert(fock=5)
        for phi in (0.0, np.pi / 3, -np.pi / 2):
            H = blue_sideband_hamiltonian(h, _sideband_drive(phase_rad=phi), "axial", ion_index=0)
            assert (H - H.dag()).norm() < 1e-12

    def test_vacuum_blue_rabi_rate_matches_analytic(self) -> None:
        """A blue π-pulse from |↓, 0⟩ takes the state to |↑, 1⟩ at
        rate |η|·Ω (the n=0 case of |η|·√(n+1)·Ω)."""
        from iontrap_dynamics.operators import sigma_z_ion

        h = _single_ion_hilbert(fock=10)
        rabi = 2 * np.pi * 0.1e6
        drive = _sideband_drive(rabi=rabi)
        H = blue_sideband_hamiltonian(h, drive, "axial", ion_index=0)

        eta = _expected_eta(h)
        rate = blue_sideband_rabi_frequency(
            carrier_rabi_frequency=rabi,
            lamb_dicke_parameter=eta,
            n_initial=0,
        )
        t_pi = np.pi / rate

        psi_0 = ground_state(h)  # |↓, 0⟩
        result = qutip.mesolve(H, psi_0, [0.0, t_pi], [], [])
        psi_final = result.states[-1]

        sigma_z = h.spin_op_for_ion(sigma_z_ion(), 0)
        n_op = h.number_for_mode("axial")
        # σ_z: −1 → +1; ⟨n̂⟩: 0 → 1
        assert qutip.expect(sigma_z, psi_final) == pytest.approx(+1.0, abs=1e-4)
        assert qutip.expect(n_op, psi_final) == pytest.approx(1.0, abs=1e-4)


# ----------------------------------------------------------------------------
# Red vs Blue: structural asymmetry
# ----------------------------------------------------------------------------


class TestRedVsBlue:
    def test_red_and_blue_are_different_operators(self) -> None:
        """For the same (drive, mode, ion), red and blue differ by a↔a†."""
        h = _single_ion_hilbert(fock=5)
        H_red = red_sideband_hamiltonian(h, _sideband_drive(), "axial", ion_index=0)
        H_blue = blue_sideband_hamiltonian(h, _sideband_drive(), "axial", ion_index=0)
        assert (H_red - H_blue).norm() > 1e-6

    def test_unknown_mode_label_raises(self) -> None:
        h = _single_ion_hilbert(fock=5)
        with pytest.raises(ConventionError, match="unknown mode"):
            red_sideband_hamiltonian(h, _sideband_drive(), "nonexistent", ion_index=0)
        with pytest.raises(ConventionError, match="unknown mode"):
            blue_sideband_hamiltonian(h, _sideband_drive(), "nonexistent", ion_index=0)

    def test_out_of_range_ion_index_raises(self) -> None:
        h = _single_ion_hilbert(fock=5)
        with pytest.raises(IndexError):
            red_sideband_hamiltonian(h, _sideband_drive(), "axial", ion_index=1)
        with pytest.raises(IndexError):
            blue_sideband_hamiltonian(h, _sideband_drive(), "axial", ion_index=1)
