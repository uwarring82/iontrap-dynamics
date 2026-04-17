# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.states`."""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics.exceptions import ConventionError, IonTrapError
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.operators import sigma_z_ion, spin_down, spin_up
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import compose_density, ground_state
from iontrap_dynamics.system import IonSystem

# ----------------------------------------------------------------------------
# Fixture helpers
# ----------------------------------------------------------------------------


def _single_ion_axial() -> ModeConfig:
    return ModeConfig(
        label="axial",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )


def _two_ion_com() -> ModeConfig:
    return ModeConfig(
        label="com",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]) / np.sqrt(2.0),
    )


def _two_ion_stretch() -> ModeConfig:
    return ModeConfig(
        label="stretch",
        frequency_rad_s=2 * np.pi * 2.6e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]) / np.sqrt(2.0),
    )


def _single_ion_hilbert(*, fock: int = 5) -> HilbertSpace:
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(_single_ion_axial(),))
    return HilbertSpace(system=system, fock_truncations={"axial": fock})


def _two_ion_two_mode_hilbert(*, com_fock: int = 4, stretch_fock: int = 4) -> HilbertSpace:
    system = IonSystem(
        species_per_ion=(mg25_plus(), mg25_plus()),
        modes=(_two_ion_com(), _two_ion_stretch()),
    )
    return HilbertSpace(system=system, fock_truncations={"com": com_fock, "stretch": stretch_fock})


# ----------------------------------------------------------------------------
# ground_state
# ----------------------------------------------------------------------------


class TestGroundState:
    def test_single_ion_dims(self) -> None:
        h = _single_ion_hilbert(fock=10)
        psi = ground_state(h)
        # Subsystem layout is [spin=2, mode=10]; the ket-column side is
        # flat [1] in QuTiP 5 rather than [1, 1]. Assert the stable half.
        assert psi.dims[0] == [2, 10]
        assert psi.shape == (20, 1)
        assert psi.isket

    def test_single_ion_is_normalised(self) -> None:
        psi = ground_state(_single_ion_hilbert())
        assert psi.norm() == pytest.approx(1.0)

    def test_sigma_z_eigenvalue_on_single_ion_is_minus_one(self) -> None:
        """CONVENTIONS.md §3: ground state |↓⟩ has σ_z_ion eigenvalue −1."""
        h = _single_ion_hilbert()
        psi = ground_state(h)
        sz0 = h.spin_op_for_ion(sigma_z_ion(), ion_index=0)
        assert qutip.expect(sz0, psi) == pytest.approx(-1.0)

    def test_sum_of_sigma_z_on_two_ions_is_minus_two(self) -> None:
        """|↓, ↓⟩ has σ_z^(0) + σ_z^(1) = -2."""
        h = _two_ion_two_mode_hilbert()
        psi = ground_state(h)
        sz_total = h.spin_op_for_ion(sigma_z_ion(), 0) + h.spin_op_for_ion(sigma_z_ion(), 1)
        assert qutip.expect(sz_total, psi) == pytest.approx(-2.0)

    def test_mode_occupation_is_zero(self) -> None:
        """Ground state has ⟨n̂⟩ = 0 on every mode."""
        h = _two_ion_two_mode_hilbert()
        psi = ground_state(h)
        n_com = h.number_for_mode("com")
        n_stretch = h.number_for_mode("stretch")
        assert qutip.expect(n_com, psi) == pytest.approx(0.0)
        assert qutip.expect(n_stretch, psi) == pytest.approx(0.0)

    def test_pure_spin_system_ground_state(self) -> None:
        """A pure-spin IonSystem (no modes) still produces a valid
        ground state — just |↓⟩^⊗N with no motional factor."""
        system = IonSystem(species_per_ion=(mg25_plus(), mg25_plus()), modes=())
        h = HilbertSpace(system=system, fock_truncations={})
        psi = ground_state(h)
        assert psi.dims[0] == [2, 2]
        assert psi.shape == (4, 1)
        assert psi.isket
        assert psi.norm() == pytest.approx(1.0)


# ----------------------------------------------------------------------------
# compose_density — happy paths
# ----------------------------------------------------------------------------


class TestComposeDensityHappyPaths:
    def test_compose_from_kets_single_ion(self) -> None:
        """All-ket inputs → density matrix via ket2dm internally."""
        h = _single_ion_hilbert(fock=5)
        rho = compose_density(
            h,
            spin_states_per_ion=[spin_down()],
            mode_states_by_label={"axial": qutip.basis(5, 0)},
        )
        assert rho.isoper
        assert rho.dims == h.qutip_dims()
        # Trace 1
        assert abs(rho.tr() - 1.0) < 1e-14
        # ⟨σ_z⟩ = −1 for |↓⟩
        sz = h.spin_op_for_ion(sigma_z_ion(), 0)
        assert (sz * rho).tr() == pytest.approx(-1.0)

    def test_compose_thermal_motion(self) -> None:
        """Mixed input: |↓⟩ spin ket + thermal density on the mode."""
        h = _single_ion_hilbert(fock=20)
        n_bar = 0.5
        rho = compose_density(
            h,
            spin_states_per_ion=[spin_down()],
            mode_states_by_label={"axial": qutip.thermal_dm(20, n=n_bar)},
        )
        assert rho.isoper
        assert abs(rho.tr() - 1.0) < 1e-10
        # ⟨n̂⟩ on the composite should equal n_bar
        n_op = h.number_for_mode("axial")
        assert (n_op * rho).tr() == pytest.approx(n_bar)

    def test_compose_with_spin_up(self) -> None:
        """|↑⟩ spin → ⟨σ_z⟩ = +1 (verifies the spin slot receives the
        user's choice, not a silent default)."""
        h = _single_ion_hilbert()
        rho = compose_density(
            h,
            spin_states_per_ion=[spin_up()],
            mode_states_by_label={"axial": qutip.basis(5, 0)},
        )
        sz = h.spin_op_for_ion(sigma_z_ion(), 0)
        assert (sz * rho).tr() == pytest.approx(+1.0)

    def test_compose_two_ion_two_mode(self) -> None:
        """Two spins, two modes — dims, trace, per-slot expectations."""
        h = _two_ion_two_mode_hilbert(com_fock=5, stretch_fock=4)
        rho = compose_density(
            h,
            spin_states_per_ion=[spin_down(), spin_up()],
            mode_states_by_label={
                "com": qutip.basis(5, 2),  # Fock |2⟩ on com
                "stretch": qutip.thermal_dm(4, n=0.1),
            },
        )
        assert rho.dims == h.qutip_dims()
        assert abs(rho.tr() - 1.0) < 1e-10
        # Ion 0 (|↓⟩): σ_z = −1; ion 1 (|↑⟩): σ_z = +1; com Fock 2: ⟨n̂⟩ = 2
        sz0 = h.spin_op_for_ion(sigma_z_ion(), 0)
        sz1 = h.spin_op_for_ion(sigma_z_ion(), 1)
        n_com = h.number_for_mode("com")
        assert (sz0 * rho).tr() == pytest.approx(-1.0)
        assert (sz1 * rho).tr() == pytest.approx(+1.0)
        assert (n_com * rho).tr() == pytest.approx(2.0)

    def test_compose_accepts_density_matrix_input_for_spin(self) -> None:
        """Passing a 2×2 density matrix (not a ket) for a spin works."""
        h = _single_ion_hilbert()
        rho_spin = qutip.ket2dm(spin_down())
        rho = compose_density(
            h,
            spin_states_per_ion=[rho_spin],
            mode_states_by_label={"axial": qutip.basis(5, 0)},
        )
        sz = h.spin_op_for_ion(sigma_z_ion(), 0)
        assert (sz * rho).tr() == pytest.approx(-1.0)


# ----------------------------------------------------------------------------
# compose_density — validation
# ----------------------------------------------------------------------------


class TestComposeDensityValidation:
    def test_wrong_spin_count_rejected(self) -> None:
        h = _two_ion_two_mode_hilbert()
        with pytest.raises(ConventionError, match="spin_states_per_ion"):
            compose_density(
                h,
                spin_states_per_ion=[spin_down()],  # only 1, need 2
                mode_states_by_label={
                    "com": qutip.basis(4, 0),
                    "stretch": qutip.basis(4, 0),
                },
            )

    def test_too_many_spin_states_rejected(self) -> None:
        h = _single_ion_hilbert()
        with pytest.raises(ConventionError, match="spin_states_per_ion"):
            compose_density(
                h,
                spin_states_per_ion=[spin_down(), spin_down()],  # 2, need 1
                mode_states_by_label={"axial": qutip.basis(5, 0)},
            )

    def test_missing_mode_rejected(self) -> None:
        h = _two_ion_two_mode_hilbert()
        with pytest.raises(ConventionError, match="missing entries"):
            compose_density(
                h,
                spin_states_per_ion=[spin_down(), spin_down()],
                mode_states_by_label={"com": qutip.basis(4, 0)},  # stretch missing
            )

    def test_extra_mode_rejected(self) -> None:
        h = _single_ion_hilbert()
        with pytest.raises(ConventionError, match="unknown modes"):
            compose_density(
                h,
                spin_states_per_ion=[spin_down()],
                mode_states_by_label={
                    "axial": qutip.basis(5, 0),
                    "nonexistent": qutip.basis(5, 0),
                },
            )

    def test_wrong_spin_dim_rejected(self) -> None:
        """A 3-dim "spin" state is not a valid two-level spin input."""
        h = _single_ion_hilbert()
        with pytest.raises(ConventionError, match="spin_states_per_ion"):
            compose_density(
                h,
                spin_states_per_ion=[qutip.basis(3, 0)],  # dim 3, not 2
                mode_states_by_label={"axial": qutip.basis(5, 0)},
            )

    def test_wrong_mode_cutoff_rejected(self) -> None:
        """Mode state with wrong Fock cutoff fails validation."""
        h = _single_ion_hilbert(fock=5)
        with pytest.raises(ConventionError, match="mode_states_by_label"):
            compose_density(
                h,
                spin_states_per_ion=[spin_down()],
                mode_states_by_label={"axial": qutip.basis(7, 0)},  # dim 7, need 5
            )

    def test_validation_errors_subclass_iontraperror(self) -> None:
        h = _single_ion_hilbert()
        with pytest.raises(IonTrapError):
            compose_density(
                h,
                spin_states_per_ion=[],
                mode_states_by_label={"axial": qutip.basis(5, 0)},
            )


# ----------------------------------------------------------------------------
# Convention compliance — end-to-end sanity
# ----------------------------------------------------------------------------


class TestConventionCompliance:
    def test_ground_state_and_compose_density_agree_on_trivial_case(self) -> None:
        """compose_density with |↓⟩ and |0⟩ should give the density-matrix
        equivalent of ground_state (i.e. ket2dm(ground_state))."""
        h = _single_ion_hilbert(fock=5)
        rho_compose = compose_density(
            h,
            spin_states_per_ion=[spin_down()],
            mode_states_by_label={"axial": qutip.basis(5, 0)},
        )
        rho_via_ground = qutip.ket2dm(ground_state(h))
        diff = rho_compose - rho_via_ground
        assert diff.norm() < 1e-14
