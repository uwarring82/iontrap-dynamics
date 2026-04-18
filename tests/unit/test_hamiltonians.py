# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.hamiltonians`."""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip

from iontrap_dynamics.analytic import (
    blue_sideband_rabi_frequency,
    lamb_dicke_parameter,
    ms_gate_closing_detuning,
    ms_gate_closing_time,
    ms_gate_phonon_number,
    red_sideband_rabi_frequency,
)
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.exceptions import ConventionError, IonTrapError
from iontrap_dynamics.hamiltonians import (
    blue_sideband_hamiltonian,
    carrier_hamiltonian,
    detuned_ms_gate_hamiltonian,
    modulated_carrier_hamiltonian,
    ms_gate_hamiltonian,
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


# ============================================================================
# Mølmer–Sørensen gate (δ = 0 bichromatic)
# ============================================================================


def _two_ion_ms_hilbert(*, fock: int = 6) -> HilbertSpace:
    """Two ²⁵Mg⁺ ions sharing an axial COM mode (symmetric Lamb–Dicke)."""
    system = IonSystem(
        species_per_ion=(mg25_plus(), mg25_plus()),
        modes=(_two_ion_com_mode(),),
    )
    return HilbertSpace(system=system, fock_truncations={"com": fock})


def _ms_drive(*, rabi: float = 2 * np.pi * 0.1e6, phase_rad: float = 0.0) -> DriveConfig:
    """Drive along +z, so k ∥ b for the axial COM mode."""
    return DriveConfig(
        k_vector_m_inv=[0.0, 0.0, _K_MAGNITUDE],
        carrier_rabi_frequency_rad_s=rabi,
        phase_rad=phase_rad,
    )


def _ms_expected_eta(h: HilbertSpace, ion: int) -> float:
    drive = _ms_drive()
    species = h.system.species(ion)
    mode = h.system.mode("com")
    return lamb_dicke_parameter(
        k_vec=drive.k_vector_m_inv,
        mode_eigenvector=mode.eigenvector_at_ion(ion),
        ion_mass=species.mass_kg,
        mode_frequency=mode.frequency_rad_s,
    )


# ----------------------------------------------------------------------------
# MS gate: structural correctness
# ----------------------------------------------------------------------------


class TestMSGateStructure:
    def test_dims_match_full_hilbert(self) -> None:
        h = _two_ion_ms_hilbert()
        H = ms_gate_hamiltonian(h, _ms_drive(), "com", ion_indices=(0, 1))
        assert H.dims == h.qutip_dims()

    def test_hermitian_for_any_phase(self) -> None:
        h = _two_ion_ms_hilbert()
        for phi in (0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3):
            H = ms_gate_hamiltonian(h, _ms_drive(phase_rad=phi), "com", ion_indices=(0, 1))
            assert (H - H.dag()).norm() < 1e-10, f"non-Hermitian at φ={phi}"

    def test_ion_index_swap_gives_same_operator(self) -> None:
        """H((0, 1)) == H((1, 0)) — the builder is symmetric under ion swap."""
        h = _two_ion_ms_hilbert()
        H_ij = ms_gate_hamiltonian(h, _ms_drive(), "com", ion_indices=(0, 1))
        H_ji = ms_gate_hamiltonian(h, _ms_drive(), "com", ion_indices=(1, 0))
        assert (H_ij - H_ji).norm() < 1e-12

    def test_equals_sum_of_single_ion_pieces(self) -> None:
        """H_MS = H_0 + H_1 where each H_k = (Ωη_k/2) σ_φ^{(k)} ⊗ (a + a†)."""
        h = _two_ion_ms_hilbert()
        drive = _ms_drive()
        H_full = ms_gate_hamiltonian(h, drive, "com", ion_indices=(0, 1))

        omega = drive.carrier_rabi_frequency_rad_s
        a = h.annihilation_for_mode("com")
        a_dag = h.creation_for_mode("com")
        x_mode = a + a_dag

        eta_0 = _ms_expected_eta(h, 0)
        eta_1 = _ms_expected_eta(h, 1)
        sx0 = h.spin_op_for_ion(sigma_x_ion(), 0)
        sx1 = h.spin_op_for_ion(sigma_x_ion(), 1)
        # At φ=0: H = (Ω/2)(η_0 σ_x^{(0)} + η_1 σ_x^{(1)}) ⊗ (a + a†)
        expected = (omega / 2.0) * (eta_0 * sx0 + eta_1 * sx1) * x_mode
        assert (H_full - expected).norm() < 1e-10


# ----------------------------------------------------------------------------
# MS gate: phase conventions (σ_x at φ=0, σ_y at φ=π/2)
# ----------------------------------------------------------------------------


class TestMSGatePhaseConvention:
    def test_phi_zero_reduces_to_sigma_x_form(self) -> None:
        """At φ=0, H/ℏ = Σ_k (Ωη_k/2) σ_x^{(k)} ⊗ (a + a†)."""
        h = _two_ion_ms_hilbert()
        omega = 2 * np.pi * 0.1e6
        H = ms_gate_hamiltonian(h, _ms_drive(rabi=omega), "com", ion_indices=(0, 1))

        eta_0 = _ms_expected_eta(h, 0)
        eta_1 = _ms_expected_eta(h, 1)
        sx0 = h.spin_op_for_ion(sigma_x_ion(), 0)
        sx1 = h.spin_op_for_ion(sigma_x_ion(), 1)
        a = h.annihilation_for_mode("com")
        x_mode = a + a.dag()

        expected = (omega / 2.0) * (eta_0 * sx0 + eta_1 * sx1) * x_mode
        assert (H - expected).norm() < 1e-10

    def test_phi_pi_over_two_gives_minus_sigma_y_form(self) -> None:
        """At φ=π/2, H/ℏ = −Σ_k (Ωη_k/2) σ_y^{(k)} ⊗ (a + a†)."""
        h = _two_ion_ms_hilbert()
        omega = 2 * np.pi * 0.1e6
        H = ms_gate_hamiltonian(
            h, _ms_drive(rabi=omega, phase_rad=np.pi / 2), "com", ion_indices=(0, 1)
        )

        eta_0 = _ms_expected_eta(h, 0)
        eta_1 = _ms_expected_eta(h, 1)
        sy0 = h.spin_op_for_ion(sigma_y_ion(), 0)
        sy1 = h.spin_op_for_ion(sigma_y_ion(), 1)
        a = h.annihilation_for_mode("com")
        x_mode = a + a.dag()

        expected = -(omega / 2.0) * (eta_0 * sy0 + eta_1 * sy1) * x_mode
        assert (H - expected).norm() < 1e-10


# ----------------------------------------------------------------------------
# MS gate: validation
# ----------------------------------------------------------------------------


class TestMSGateValidation:
    def test_duplicate_ion_indices_rejected(self) -> None:
        h = _two_ion_ms_hilbert()
        with pytest.raises(ConventionError, match="distinct"):
            ms_gate_hamiltonian(h, _ms_drive(), "com", ion_indices=(0, 0))

    def test_unknown_mode_label_rejected(self) -> None:
        h = _two_ion_ms_hilbert()
        with pytest.raises(ConventionError, match="unknown mode"):
            ms_gate_hamiltonian(h, _ms_drive(), "nonexistent", ion_indices=(0, 1))

    def test_out_of_range_ion_index_raises(self) -> None:
        h = _two_ion_ms_hilbert()
        with pytest.raises(IndexError):
            ms_gate_hamiltonian(h, _ms_drive(), "com", ion_indices=(0, 2))

    def test_validation_errors_subclass_iontraperror(self) -> None:
        h = _two_ion_ms_hilbert()
        with pytest.raises(IonTrapError):
            ms_gate_hamiltonian(h, _ms_drive(), "com", ion_indices=(1, 1))


# ----------------------------------------------------------------------------
# MS gate: symmetry-conservation integration tests
# ----------------------------------------------------------------------------


def _plus_state() -> qutip.Qobj:
    """|+⟩ = (|↓⟩ + |↑⟩) / √2 — the σ_x = +1 eigenstate."""
    return (spin_down() + spin_up()).unit()


def _minus_state() -> qutip.Qobj:
    """|−⟩ = (|↓⟩ − |↑⟩) / √2 — the σ_x = −1 eigenstate."""
    return (spin_down() - spin_up()).unit()


class TestMSGateConservationLaws:
    """At φ=0 the MS generator is (Ω/2)(η₀ σ_x⁽⁰⁾ + η₁ σ_x⁽¹⁾) ⊗ (a + a†).
    Each σ_x^{(k)} individually commutes with H, so ⟨σ_x^{(k)}⟩ and
    ⟨σ_x^{(0)} σ_x^{(1)}⟩ are exact constants of motion."""

    def test_sigma_x_each_ion_conserved_from_plus_plus(self) -> None:
        """⟨σ_x^{(0)}⟩ = ⟨σ_x^{(1)}⟩ = +1 at every time, starting from |++, 0⟩."""
        h = _two_ion_ms_hilbert(fock=10)
        omega = 2 * np.pi * 0.1e6
        H = ms_gate_hamiltonian(h, _ms_drive(rabi=omega), "com", ion_indices=(0, 1))

        eta = _ms_expected_eta(h, 0)
        # Short evolution over ~one displacement period (arbitrary cap)
        t_end = 2 * np.pi / (omega * abs(eta))
        tlist = np.linspace(0.0, t_end, 10)

        psi_0 = qutip.tensor(_plus_state(), _plus_state(), qutip.basis(10, 0))
        result = qutip.mesolve(H, psi_0, tlist, [], [])

        sx0 = h.spin_op_for_ion(sigma_x_ion(), 0)
        sx1 = h.spin_op_for_ion(sigma_x_ion(), 1)
        for state in result.states:
            assert qutip.expect(sx0, state) == pytest.approx(+1.0, abs=1e-6)
            assert qutip.expect(sx1, state) == pytest.approx(+1.0, abs=1e-6)

    def test_sigma_x_product_conserved_from_plus_minus(self) -> None:
        """⟨σ_x^{(0)} σ_x^{(1)}⟩ = −1 throughout, starting from |+−, 0⟩."""
        h = _two_ion_ms_hilbert(fock=10)
        omega = 2 * np.pi * 0.1e6
        H = ms_gate_hamiltonian(h, _ms_drive(rabi=omega), "com", ion_indices=(0, 1))

        eta = _ms_expected_eta(h, 0)
        t_end = 2 * np.pi / (omega * abs(eta))
        tlist = np.linspace(0.0, t_end, 10)

        psi_0 = qutip.tensor(_plus_state(), _minus_state(), qutip.basis(10, 0))
        result = qutip.mesolve(H, psi_0, tlist, [], [])

        sx_product = h.spin_op_for_ion(sigma_x_ion(), 0) * h.spin_op_for_ion(sigma_x_ion(), 1)
        for state in result.states:
            assert qutip.expect(sx_product, state) == pytest.approx(-1.0, abs=1e-6)

    def test_sigma_z_ion_exchange_symmetry_from_ground(self) -> None:
        """Starting from |↓↓, 0⟩, ⟨σ_z^{(0)}⟩(t) = ⟨σ_z^{(1)}⟩(t) at every time
        (symmetric drive, symmetric mode)."""
        from iontrap_dynamics.operators import sigma_z_ion

        h = _two_ion_ms_hilbert(fock=10)
        omega = 2 * np.pi * 0.1e6
        H = ms_gate_hamiltonian(h, _ms_drive(rabi=omega), "com", ion_indices=(0, 1))

        eta = _ms_expected_eta(h, 0)
        t_end = 0.3 * 2 * np.pi / (omega * abs(eta))
        tlist = np.linspace(0.0, t_end, 8)

        psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(10, 0))
        result = qutip.mesolve(H, psi_0, tlist, [], [])

        sz0 = h.spin_op_for_ion(sigma_z_ion(), 0)
        sz1 = h.spin_op_for_ion(sigma_z_ion(), 1)
        for state in result.states:
            sz_0_t = qutip.expect(sz0, state)
            sz_1_t = qutip.expect(sz1, state)
            assert sz_0_t == pytest.approx(sz_1_t, abs=1e-8)


# ----------------------------------------------------------------------------
# MS gate: analytic coherent-displacement match
# ----------------------------------------------------------------------------


class TestMSGateCoherentDisplacement:
    """The spin-dependent-force picture: starting from a σ_x product eigenstate
    times motional vacuum, the motion becomes a coherent state whose mean
    phonon number follows :func:`analytic.ms_gate_phonon_number`."""

    def test_plus_plus_matches_analytic_displacement(self) -> None:
        """|++, 0⟩ → ⟨n̂⟩(t) = ((Ωt/2)(η₀ + η₁))²."""
        h = _two_ion_ms_hilbert(fock=20)
        omega = 2 * np.pi * 0.1e6
        H = ms_gate_hamiltonian(h, _ms_drive(rabi=omega), "com", ion_indices=(0, 1))

        eta_0 = _ms_expected_eta(h, 0)
        eta_1 = _ms_expected_eta(h, 1)
        # Short horizon so |α(t)|² ≪ fock truncation
        t_end = 0.5 / (omega * abs(eta_0))
        tlist = np.linspace(0.0, t_end, 12)

        psi_0 = qutip.tensor(_plus_state(), _plus_state(), qutip.basis(20, 0))
        result = qutip.mesolve(H, psi_0, tlist, [], [])

        n_op = h.number_for_mode("com")
        numerical = np.array([qutip.expect(n_op, s) for s in result.states])
        analytic = ms_gate_phonon_number(
            carrier_rabi_frequency=omega,
            lamb_dicke_parameters=(eta_0, eta_1),
            spin_eigenvalues=(+1, +1),
            t=tlist,
        )
        np.testing.assert_allclose(numerical, analytic, atol=1e-4)

    def test_plus_minus_dark_state_stays_in_vacuum(self) -> None:
        """|+−, 0⟩ on the COM mode (η₀ = η₁) is a dark state: forces cancel,
        motion stays in vacuum. ⟨n̂⟩(t) ≈ 0 throughout."""
        h = _two_ion_ms_hilbert(fock=10)
        omega = 2 * np.pi * 0.1e6
        H = ms_gate_hamiltonian(h, _ms_drive(rabi=omega), "com", ion_indices=(0, 1))

        eta = _ms_expected_eta(h, 0)
        t_end = 2 * np.pi / (omega * abs(eta))
        tlist = np.linspace(0.0, t_end, 20)

        psi_0 = qutip.tensor(_plus_state(), _minus_state(), qutip.basis(10, 0))
        result = qutip.mesolve(H, psi_0, tlist, [], [])

        n_op = h.number_for_mode("com")
        occupations = np.array([qutip.expect(n_op, s) for s in result.states])
        np.testing.assert_allclose(occupations, np.zeros_like(occupations), atol=1e-8)


# ============================================================================
# Modulated carrier (time-dependent envelope, list-format)
# ============================================================================


class TestModulatedCarrierStructure:
    def test_returns_qutip_list_format(self) -> None:
        """Returns [[H_carrier, coeff_fn]] — a one-entry list of list pairs."""
        h = _single_ion_hilbert(fock=3)
        H = modulated_carrier_hamiltonian(h, _simple_drive(), ion_index=0, envelope=lambda t: 1.0)
        assert isinstance(H, list)
        assert len(H) == 1
        inner = H[0]
        assert isinstance(inner, list)
        assert len(inner) == 2
        assert isinstance(inner[0], qutip.Qobj)
        assert callable(inner[1])

    def test_base_hamiltonian_matches_static_carrier(self) -> None:
        """The [0][0] Qobj is byte-identical to what carrier_hamiltonian
        returns for the same (hilbert, drive, ion_index). The envelope
        lives entirely in the callable coefficient."""
        h = _single_ion_hilbert(fock=3)
        drive = _simple_drive()
        static_H = carrier_hamiltonian(h, drive, ion_index=0)
        modulated_H = modulated_carrier_hamiltonian(h, drive, ion_index=0, envelope=lambda t: 1.0)
        embedded_qobj = modulated_H[0][0]
        assert isinstance(embedded_qobj, qutip.Qobj)
        assert (embedded_qobj - static_H).norm() < 1e-14

    def test_envelope_is_evaluated_through_coeff_fn(self) -> None:
        """The coefficient callable forwards t → envelope(t) and ignores
        the QuTiP ``args`` dict. Structural check only: pure-Python
        invocation of the inner callable."""
        h = _single_ion_hilbert(fock=3)
        H = modulated_carrier_hamiltonian(
            h,
            _simple_drive(),
            ion_index=0,
            envelope=lambda t: 3.7 * t,
        )
        coeff = H[0][1]
        assert coeff(2.0, {}) == pytest.approx(7.4)
        assert coeff(0.0, {}) == pytest.approx(0.0)


class TestModulatedCarrierValidation:
    def test_detuned_drive_rejected(self) -> None:
        h = _single_ion_hilbert(fock=3)
        drive = DriveConfig(
            k_vector_m_inv=[1.0, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
            detuning_rad_s=1e3,
        )
        with pytest.raises(ConventionError, match="on-resonance"):
            modulated_carrier_hamiltonian(h, drive, ion_index=0, envelope=lambda t: 1.0)

    def test_out_of_range_ion_index_raises(self) -> None:
        h = _single_ion_hilbert(fock=3)
        with pytest.raises(IndexError):
            modulated_carrier_hamiltonian(h, _simple_drive(), ion_index=1, envelope=lambda t: 1.0)

    def test_validation_errors_subclass_iontraperror(self) -> None:
        h = _single_ion_hilbert(fock=3)
        drive = DriveConfig(
            k_vector_m_inv=[1.0, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=1.0,
            detuning_rad_s=-1.0,
        )
        with pytest.raises(IonTrapError):
            modulated_carrier_hamiltonian(h, drive, ion_index=0, envelope=lambda t: 1.0)


class TestModulatedCarrierDynamics:
    """End-to-end checks routing the list-format Hamiltonian through mesolve."""

    def test_constant_envelope_matches_static_carrier_evolution(self) -> None:
        """``envelope(t) = 1`` should reproduce the same ⟨σ_z⟩(t) trajectory
        as the static :func:`carrier_hamiltonian`, modulo integrator noise."""
        from iontrap_dynamics.operators import sigma_z_ion

        h = _single_ion_hilbert(fock=3)
        rabi = 2 * np.pi * 1e6
        drive = _simple_drive(rabi=rabi)

        H_static = carrier_hamiltonian(h, drive, ion_index=0)
        H_modulated = modulated_carrier_hamiltonian(h, drive, ion_index=0, envelope=lambda t: 1.0)

        psi_0 = ground_state(h)
        tlist = np.linspace(0.0, 2 * np.pi / rabi, 50)
        sigma_z = h.spin_op_for_ion(sigma_z_ion(), 0)

        static_result = qutip.mesolve(H_static, psi_0, tlist, [], [sigma_z])
        modulated_result = qutip.mesolve(H_modulated, psi_0, tlist, [], [sigma_z])

        np.testing.assert_allclose(
            modulated_result.expect[0],
            static_result.expect[0],
            atol=1e-6,
        )

    def test_zero_envelope_freezes_dynamics(self) -> None:
        """``envelope(t) = 0`` leaves the state at the initial state: no evolution."""
        h = _single_ion_hilbert(fock=3)
        rabi = 2 * np.pi * 1e6
        H = modulated_carrier_hamiltonian(
            h, _simple_drive(rabi=rabi), ion_index=0, envelope=lambda t: 0.0
        )

        psi_0 = ground_state(h)
        tlist = np.linspace(0.0, np.pi / rabi, 5)
        result = qutip.mesolve(H, psi_0, tlist, [], [])

        for state in result.states:
            assert (state - psi_0).norm() < 1e-8

    def test_pulse_area_gives_pi_rotation(self) -> None:
        """A Gaussian envelope with pulse area ``∫Ω·f(t) dt = π`` delivers
        a π-rotation: |↓⟩ → |↑⟩ (⟨σ_z⟩ = −1 → +1). Uses an integrated
        rather than constant-rate schedule — this is the test that
        validates the time-dependent path beyond trivial equivalence."""
        from iontrap_dynamics.operators import sigma_z_ion

        h = _single_ion_hilbert(fock=3)
        rabi = 2 * np.pi * 1e6
        t_end = 5e-6  # 5 μs window
        t_centre = t_end / 2
        sigma_gauss = t_end / 10  # narrow pulse, well inside the window

        # Normalise so the pulse area ∫₀^t_end Ω·f(t) dt = π.
        # f(t) = A · exp(-(t - t_c)² / (2 σ²)), area ≈ A · σ · √(2π) · Ω
        # (the erf correction from truncating at t=0, t=t_end is <1e-30
        # for the chosen σ ≪ t_end).
        amplitude = np.pi / (rabi * sigma_gauss * np.sqrt(2 * np.pi))

        def gaussian(t: float) -> float:
            return float(amplitude * math.exp(-((t - t_centre) ** 2) / (2 * sigma_gauss**2)))

        H = modulated_carrier_hamiltonian(
            h, _simple_drive(rabi=rabi), ion_index=0, envelope=gaussian
        )

        psi_0 = ground_state(h)
        tlist = np.linspace(0.0, t_end, 200)
        result = qutip.mesolve(H, psi_0, tlist, [], [])
        psi_final = result.states[-1]

        sigma_z = h.spin_op_for_ion(sigma_z_ion(), 0)
        assert qutip.expect(sigma_z, psi_0) == pytest.approx(-1.0)
        assert qutip.expect(sigma_z, psi_final) == pytest.approx(+1.0, abs=1e-4)


# ============================================================================
# Detuned Mølmer–Sørensen gate (δ ≠ 0, gate-closing, list-format)
# ============================================================================


class TestDetunedMSGateStructure:
    def test_returns_two_piece_list_format(self) -> None:
        """[[A_x, cos_fn], [A_p, sin_fn]] — two time-dependent pieces."""
        h = _two_ion_ms_hilbert()
        H = detuned_ms_gate_hamiltonian(
            h, _ms_drive(), "com", ion_indices=(0, 1), detuning_rad_s=1e5
        )
        assert isinstance(H, list)
        assert len(H) == 2
        for piece in H:
            assert isinstance(piece, list) and len(piece) == 2
            assert isinstance(piece[0], qutip.Qobj)
            assert callable(piece[1])
            assert piece[0].dims == h.qutip_dims()

    def test_hermitian_at_sampled_times(self) -> None:
        """H(t) = A_x · cos(δt) + A_p · sin(δt) should be Hermitian for all t,
        since A_x and A_p are each independently Hermitian."""
        h = _two_ion_ms_hilbert()
        delta = 2 * np.pi * 50e3
        H = detuned_ms_gate_hamiltonian(
            h, _ms_drive(), "com", ion_indices=(0, 1), detuning_rad_s=delta
        )
        a_x, cos_fn = H[0]
        a_p, sin_fn = H[1]
        # Individual operators Hermitian
        assert (a_x - a_x.dag()).norm() < 1e-10
        assert (a_p - a_p.dag()).norm() < 1e-10
        # Reconstructed H(t) Hermitian at representative times
        for t in (0.0, 1e-6, 3e-6, 5e-6):
            H_t = cos_fn(t, {}) * a_x + sin_fn(t, {}) * a_p
            assert (H_t - H_t.dag()).norm() < 1e-10

    def test_t_zero_matches_static_ms_gate(self) -> None:
        """At t = 0, cos(δt) = 1 and sin(δt) = 0, so the reconstructed
        Hamiltonian reduces to the static δ = 0 MS gate."""
        h = _two_ion_ms_hilbert()
        drive = _ms_drive()
        H_static = ms_gate_hamiltonian(h, drive, "com", ion_indices=(0, 1))
        H_detuned = detuned_ms_gate_hamiltonian(
            h, drive, "com", ion_indices=(0, 1), detuning_rad_s=1e5
        )
        a_x, cos_fn = H_detuned[0]
        a_p, sin_fn = H_detuned[1]
        H_at_zero = cos_fn(0.0, {}) * a_x + sin_fn(0.0, {}) * a_p
        assert (H_at_zero - H_static).norm() < 1e-10


class TestDetunedMSGateValidation:
    def test_zero_detuning_rejected(self) -> None:
        """δ = 0 must use the static ms_gate_hamiltonian instead."""
        h = _two_ion_ms_hilbert()
        with pytest.raises(ConventionError, match="non-zero"):
            detuned_ms_gate_hamiltonian(
                h, _ms_drive(), "com", ion_indices=(0, 1), detuning_rad_s=0.0
            )

    def test_duplicate_ion_indices_rejected(self) -> None:
        h = _two_ion_ms_hilbert()
        with pytest.raises(ConventionError, match="distinct"):
            detuned_ms_gate_hamiltonian(
                h, _ms_drive(), "com", ion_indices=(1, 1), detuning_rad_s=1e5
            )

    def test_unknown_mode_label_rejected(self) -> None:
        h = _two_ion_ms_hilbert()
        with pytest.raises(ConventionError, match="unknown mode"):
            detuned_ms_gate_hamiltonian(
                h, _ms_drive(), "nonexistent", ion_indices=(0, 1), detuning_rad_s=1e5
            )

    def test_out_of_range_ion_index_raises(self) -> None:
        h = _two_ion_ms_hilbert()
        with pytest.raises(IndexError):
            detuned_ms_gate_hamiltonian(
                h, _ms_drive(), "com", ion_indices=(0, 2), detuning_rad_s=1e5
            )

    def test_validation_errors_subclass_iontraperror(self) -> None:
        h = _two_ion_ms_hilbert()
        with pytest.raises(IonTrapError):
            detuned_ms_gate_hamiltonian(
                h, _ms_drive(), "com", ion_indices=(0, 1), detuning_rad_s=0.0
            )


# ----------------------------------------------------------------------------
# Detuned MS — phase-space loop closure and Bell-state condition
# ----------------------------------------------------------------------------


class TestDetunedMSGateClosure:
    """The textbook result: at δ = 2|Ωη|√K and t_gate = πK^{1/2}/|Ωη|,
    the Magnus expansion closes — motion returns to vacuum and the spin
    state picks up exactly a ``σ_x σ_x`` rotation of π/4 (mod π/2), which
    on |↓↓⟩ yields a Bell state."""

    def test_phase_space_loop_closes_at_t_gate(self) -> None:
        """⟨n̂⟩(t_gate) ≈ 0 — the coherent state returns to vacuum after
        the loop closes, regardless of the spin sector."""
        h = _two_ion_ms_hilbert(fock=15)
        omega = 2 * np.pi * 0.1e6
        drive = _ms_drive(rabi=omega)
        eta = _ms_expected_eta(h, 0)
        delta = ms_gate_closing_detuning(
            carrier_rabi_frequency=omega,
            lamb_dicke_parameter=eta,
            loops=1,
        )
        t_gate = ms_gate_closing_time(
            carrier_rabi_frequency=omega,
            lamb_dicke_parameter=eta,
            loops=1,
        )

        H = detuned_ms_gate_hamiltonian(h, drive, "com", ion_indices=(0, 1), detuning_rad_s=delta)

        psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(15, 0))
        result = qutip.mesolve(H, psi_0, [0.0, t_gate], [], [])
        psi_final = result.states[-1]

        n_op = h.number_for_mode("com")
        assert qutip.expect(n_op, psi_final) == pytest.approx(0.0, abs=1e-3)

    def test_bell_populations_at_t_gate(self) -> None:
        """Starting from |↓↓, 0⟩, at t_gate the spin state has
        P(|↓↓⟩) = P(|↑↑⟩) = 0.5 and P(|↓↑⟩) = P(|↑↓⟩) = 0 — a Bell
        state up to local phases."""
        from iontrap_dynamics.operators import sigma_z_ion

        h = _two_ion_ms_hilbert(fock=15)
        omega = 2 * np.pi * 0.1e6
        drive = _ms_drive(rabi=omega)
        eta = _ms_expected_eta(h, 0)
        delta = ms_gate_closing_detuning(
            carrier_rabi_frequency=omega,
            lamb_dicke_parameter=eta,
            loops=1,
        )
        t_gate = ms_gate_closing_time(
            carrier_rabi_frequency=omega,
            lamb_dicke_parameter=eta,
            loops=1,
        )

        H = detuned_ms_gate_hamiltonian(h, drive, "com", ion_indices=(0, 1), detuning_rad_s=delta)

        psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(15, 0))
        result = qutip.mesolve(H, psi_0, [0.0, t_gate], [], [])
        psi_final = result.states[-1]

        # Project onto the four two-spin computational-basis states (trace over motion)
        dd = qutip.ket2dm(qutip.tensor(spin_down(), spin_down())).full()
        du = qutip.ket2dm(qutip.tensor(spin_down(), spin_up())).full()
        ud = qutip.ket2dm(qutip.tensor(spin_up(), spin_down())).full()
        uu = qutip.ket2dm(qutip.tensor(spin_up(), spin_up())).full()

        rho_spin = psi_final.ptrace([0, 1]).full()
        p_dd = float(np.real((rho_spin * dd).trace()))
        p_du = float(np.real((rho_spin * du).trace()))
        p_ud = float(np.real((rho_spin * ud).trace()))
        p_uu = float(np.real((rho_spin * uu).trace()))

        assert p_dd == pytest.approx(0.5, abs=2e-2)
        assert p_uu == pytest.approx(0.5, abs=2e-2)
        assert p_du == pytest.approx(0.0, abs=2e-3)
        assert p_ud == pytest.approx(0.0, abs=2e-3)

        # Same-parity sanity: ⟨σ_z^{(0)} σ_z^{(1)}⟩ = +1 (both |↓↓⟩ and |↑↑⟩ give +1)
        sz0_sz1 = h.spin_op_for_ion(sigma_z_ion(), 0) * h.spin_op_for_ion(sigma_z_ion(), 1)
        assert qutip.expect(sz0_sz1, psi_final) == pytest.approx(+1.0, abs=5e-3)

    def test_sigma_z_zero_at_bell(self) -> None:
        """At t_gate with Bell-condition δ, each ion's ⟨σ_z⟩ is 0:
        equal |↓↓⟩ and |↑↑⟩ populations cancel to zero polarisation."""
        from iontrap_dynamics.operators import sigma_z_ion

        h = _two_ion_ms_hilbert(fock=15)
        omega = 2 * np.pi * 0.1e6
        drive = _ms_drive(rabi=omega)
        eta = _ms_expected_eta(h, 0)
        delta = ms_gate_closing_detuning(
            carrier_rabi_frequency=omega,
            lamb_dicke_parameter=eta,
            loops=1,
        )
        t_gate = ms_gate_closing_time(
            carrier_rabi_frequency=omega,
            lamb_dicke_parameter=eta,
            loops=1,
        )

        H = detuned_ms_gate_hamiltonian(h, drive, "com", ion_indices=(0, 1), detuning_rad_s=delta)

        psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(15, 0))
        result = qutip.mesolve(H, psi_0, [0.0, t_gate], [], [])
        psi_final = result.states[-1]

        sz0 = h.spin_op_for_ion(sigma_z_ion(), 0)
        sz1 = h.spin_op_for_ion(sigma_z_ion(), 1)
        assert qutip.expect(sz0, psi_final) == pytest.approx(0.0, abs=5e-3)
        assert qutip.expect(sz1, psi_final) == pytest.approx(0.0, abs=5e-3)

    def test_two_loop_closure_also_works(self) -> None:
        """For K = 2 (two loops), δ = 2Ωη√2 and t_gate = π√2/(Ωη).
        Same Bell-state outcome, slower gate."""
        h = _two_ion_ms_hilbert(fock=15)
        omega = 2 * np.pi * 0.1e6
        drive = _ms_drive(rabi=omega)
        eta = _ms_expected_eta(h, 0)
        delta = ms_gate_closing_detuning(
            carrier_rabi_frequency=omega,
            lamb_dicke_parameter=eta,
            loops=2,
        )
        t_gate = ms_gate_closing_time(
            carrier_rabi_frequency=omega,
            lamb_dicke_parameter=eta,
            loops=2,
        )

        H = detuned_ms_gate_hamiltonian(h, drive, "com", ion_indices=(0, 1), detuning_rad_s=delta)

        psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(15, 0))
        result = qutip.mesolve(H, psi_0, [0.0, t_gate], [], [])
        psi_final = result.states[-1]

        n_op = h.number_for_mode("com")
        assert qutip.expect(n_op, psi_final) == pytest.approx(0.0, abs=1e-3)

        # Same Bell populations as the K=1 case
        dd = qutip.ket2dm(qutip.tensor(spin_down(), spin_down())).full()
        uu = qutip.ket2dm(qutip.tensor(spin_up(), spin_up())).full()
        rho_spin = psi_final.ptrace([0, 1]).full()
        p_dd = float(np.real((rho_spin * dd).trace()))
        p_uu = float(np.real((rho_spin * uu).trace()))
        assert p_dd == pytest.approx(0.5, abs=3e-2)
        assert p_uu == pytest.approx(0.5, abs=3e-2)


# ----------------------------------------------------------------------------
# Analytic helpers — standalone formula sanity (unit algebra)
# ----------------------------------------------------------------------------


class TestMSGateClosingHelpers:
    def test_closing_detuning_single_loop(self) -> None:
        """δ = 2·|Ωη|·√1 = 2·|Ωη| for a single-loop gate."""
        delta = ms_gate_closing_detuning(
            carrier_rabi_frequency=1e5,
            lamb_dicke_parameter=0.1,
            loops=1,
        )
        assert delta == pytest.approx(2e4)

    def test_closing_detuning_multi_loop_scaling(self) -> None:
        """δ scales as √K — two loops at δ = 2·|Ωη|·√2."""
        delta1 = ms_gate_closing_detuning(
            carrier_rabi_frequency=1e5, lamb_dicke_parameter=0.1, loops=1
        )
        delta4 = ms_gate_closing_detuning(
            carrier_rabi_frequency=1e5, lamb_dicke_parameter=0.1, loops=4
        )
        assert delta4 / delta1 == pytest.approx(2.0)  # √4 / √1 = 2

    def test_closing_detuning_discards_sign(self) -> None:
        """Only |Ωη| matters — flipping either sign gives the same δ."""
        delta_pos = ms_gate_closing_detuning(
            carrier_rabi_frequency=1e5, lamb_dicke_parameter=0.1, loops=1
        )
        delta_neg_eta = ms_gate_closing_detuning(
            carrier_rabi_frequency=1e5, lamb_dicke_parameter=-0.1, loops=1
        )
        assert delta_pos == pytest.approx(delta_neg_eta)

    def test_closing_time_matches_two_pi_k_over_delta(self) -> None:
        """t_gate · δ = 2π K by construction."""
        omega, eta = 1e5, 0.1
        for loops in (1, 2, 3, 5):
            delta = ms_gate_closing_detuning(
                carrier_rabi_frequency=omega,
                lamb_dicke_parameter=eta,
                loops=loops,
            )
            t_gate = ms_gate_closing_time(
                carrier_rabi_frequency=omega,
                lamb_dicke_parameter=eta,
                loops=loops,
            )
            assert t_gate * delta == pytest.approx(2 * np.pi * loops)

    def test_closing_detuning_rejects_zero_loops(self) -> None:
        with pytest.raises(ValueError):
            ms_gate_closing_detuning(carrier_rabi_frequency=1e5, lamb_dicke_parameter=0.1, loops=0)

    def test_closing_time_rejects_zero_coupling(self) -> None:
        with pytest.raises(ValueError):
            ms_gate_closing_time(carrier_rabi_frequency=0.0, lamb_dicke_parameter=0.1, loops=1)
