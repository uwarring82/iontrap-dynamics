# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.hilbert`."""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics.exceptions import ConventionError, IonTrapError
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.operators import sigma_z_ion, spin_down
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

# ----------------------------------------------------------------------------
# Fixtures
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


def _single_ion_system() -> IonSystem:
    return IonSystem(species_per_ion=(mg25_plus(),), modes=(_single_ion_axial(),))


def _two_ion_two_mode_system() -> IonSystem:
    return IonSystem(
        species_per_ion=(mg25_plus(), mg25_plus()),
        modes=(_two_ion_com(), _two_ion_stretch()),
    )


# ----------------------------------------------------------------------------
# Construction + validation
# ----------------------------------------------------------------------------


class TestConstruction:
    def test_single_ion_single_mode(self) -> None:
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 10})
        assert h.n_ions == 1
        assert h.n_modes == 1
        assert h.spin_dim == 2
        assert h.total_dim == 2 * 10

    def test_two_ion_two_mode(self) -> None:
        h = HilbertSpace(
            system=_two_ion_two_mode_system(),
            fock_truncations={"com": 8, "stretch": 6},
        )
        assert h.n_ions == 2
        assert h.n_modes == 2
        # 2 spins × 8 × 6 = 192
        assert h.total_dim == 2 * 2 * 8 * 6

    def test_pure_spin_system_no_modes(self) -> None:
        """An IonSystem with no modes is valid — useful for pure-spin
        dynamics. HilbertSpace should handle this as well."""
        system = IonSystem(species_per_ion=(mg25_plus(),), modes=())
        h = HilbertSpace(system=system, fock_truncations={})
        assert h.n_modes == 0
        assert h.total_dim == 2


class TestValidation:
    def test_missing_mode_truncation_rejected(self) -> None:
        with pytest.raises(ConventionError, match="missing entries"):
            HilbertSpace(system=_single_ion_system(), fock_truncations={})

    def test_extra_truncation_key_rejected(self) -> None:
        with pytest.raises(ConventionError, match="unknown modes"):
            HilbertSpace(
                system=_single_ion_system(),
                fock_truncations={"axial": 10, "nonexistent": 5},
            )

    def test_zero_cutoff_rejected(self) -> None:
        with pytest.raises(ConventionError, match="must be >= 1"):
            HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 0})

    def test_negative_cutoff_rejected(self) -> None:
        with pytest.raises(ConventionError, match="must be >= 1"):
            HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": -3})

    def test_validation_errors_subclass_iontraperror(self) -> None:
        with pytest.raises(IonTrapError):
            HilbertSpace(system=_single_ion_system(), fock_truncations={})


# ----------------------------------------------------------------------------
# Dimensions (CONVENTIONS.md §2 tensor ordering: spins first, then modes)
# ----------------------------------------------------------------------------


class TestDimensions:
    def test_subsystem_dims_spins_first_then_modes(self) -> None:
        """CONVENTIONS.md §2: spins first (ascending), then modes (in
        the order they appear in system.modes)."""
        h = HilbertSpace(
            system=_two_ion_two_mode_system(),
            fock_truncations={"com": 8, "stretch": 6},
        )
        # 2 spins (both dim 2), then com (8), then stretch (6)
        assert h.subsystem_dims == [2, 2, 8, 6]

    def test_mode_order_follows_system_modes_order(self) -> None:
        """Swap the mode order in the IonSystem; the Hilbert dims should
        follow."""
        system = IonSystem(
            species_per_ion=(mg25_plus(), mg25_plus()),
            modes=(_two_ion_stretch(), _two_ion_com()),  # stretch first
        )
        h = HilbertSpace(system=system, fock_truncations={"com": 8, "stretch": 6})
        assert h.subsystem_dims == [2, 2, 6, 8]  # stretch (6) before com (8)

    def test_qutip_dims_operator_shape(self) -> None:
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 10})
        assert h.qutip_dims() == [[2, 10], [2, 10]]

    def test_mode_dim_lookup(self) -> None:
        h = HilbertSpace(
            system=_two_ion_two_mode_system(),
            fock_truncations={"com": 8, "stretch": 6},
        )
        assert h.mode_dim("com") == 8
        assert h.mode_dim("stretch") == 6

    def test_mode_dim_unknown_label_raises(self) -> None:
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 10})
        with pytest.raises(ConventionError, match="unknown mode"):
            h.mode_dim("nonexistent")


# ----------------------------------------------------------------------------
# Spin-operator embedding
# ----------------------------------------------------------------------------


class TestSpinEmbedding:
    def test_sigma_z_on_ion_0_of_two_ion_crystal(self) -> None:
        """Embedding σ_z_ion on ion 0 should give σ_z ⊗ I ⊗ I_mode."""
        h = HilbertSpace(
            system=IonSystem(
                species_per_ion=(mg25_plus(), mg25_plus()),
                modes=(_two_ion_com(),),
            ),
            fock_truncations={"com": 4},
        )
        op = h.spin_op_for_ion(sigma_z_ion(), ion_index=0)
        assert op.dims == [[2, 2, 4], [2, 2, 4]]

        # Action on |↓⟩_0 |↓⟩_1 |0⟩_mode should give eigenvalue −1 (ion-0 σ_z).
        down = spin_down()
        vacuum = qutip.basis(4, 0)
        psi = qutip.tensor(down, down, vacuum)
        eigval = qutip.expect(op, psi)
        assert eigval == pytest.approx(-1.0)

    def test_sigma_z_on_ion_1_of_two_ion_crystal(self) -> None:
        """Verifying the embedding places the operator at the correct
        position: ion_0 untouched, σ_z on ion_1."""
        h = HilbertSpace(
            system=IonSystem(
                species_per_ion=(mg25_plus(), mg25_plus()),
                modes=(_two_ion_com(),),
            ),
            fock_truncations={"com": 4},
        )
        op = h.spin_op_for_ion(sigma_z_ion(), ion_index=1)
        # Ion 0 in |↑⟩ (σ_z = +1), ion 1 in |↓⟩ (σ_z = −1). The embedded op
        # acts only on ion 1, so eigenvalue must be −1, not +1 or a sum.
        from iontrap_dynamics.operators import spin_up

        psi = qutip.tensor(spin_up(), spin_down(), qutip.basis(4, 0))
        eigval = qutip.expect(op, psi)
        assert eigval == pytest.approx(-1.0)

    def test_out_of_range_ion_index_raises(self) -> None:
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 5})
        with pytest.raises(IndexError):
            h.spin_op_for_ion(sigma_z_ion(), ion_index=1)

    def test_wrong_op_dims_raises(self) -> None:
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 5})
        wrong_op = qutip.qeye(3)  # 3×3, not 2×2
        with pytest.raises(ConventionError, match="spin operator"):
            h.spin_op_for_ion(wrong_op, ion_index=0)


# ----------------------------------------------------------------------------
# Mode-operator embedding
# ----------------------------------------------------------------------------


class TestModeEmbedding:
    def test_mode_op_by_label(self) -> None:
        h = HilbertSpace(
            system=_two_ion_two_mode_system(),
            fock_truncations={"com": 8, "stretch": 6},
        )
        com_n = h.mode_op_for(qutip.num(8), "com")
        assert com_n.dims == [[2, 2, 8, 6], [2, 2, 8, 6]]

    def test_mode_op_unknown_label_raises(self) -> None:
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 5})
        with pytest.raises(ConventionError, match="unknown mode"):
            h.mode_op_for(qutip.num(5), "nonexistent")

    def test_mode_op_wrong_cutoff_raises(self) -> None:
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 5})
        wrong = qutip.num(7)  # 7-dim op for a 5-cutoff mode
        with pytest.raises(ConventionError, match="mode operator"):
            h.mode_op_for(wrong, "axial")


# ----------------------------------------------------------------------------
# Motional primitives — annihilation / creation / number
# ----------------------------------------------------------------------------


class TestMotionalPrimitives:
    def test_annihilation_of_vacuum_is_zero(self) -> None:
        """a |↓, 0⟩ = 0."""
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 5})
        a = h.annihilation_for_mode("axial")
        psi_vac = qutip.tensor(spin_down(), qutip.basis(5, 0))
        result = a * psi_vac
        assert result.norm() < 1e-14

    def test_creation_of_vacuum_is_one_phonon(self) -> None:
        """a† |↓, 0⟩ = |↓, 1⟩."""
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 5})
        ad = h.creation_for_mode("axial")
        psi_0 = qutip.tensor(spin_down(), qutip.basis(5, 0))
        psi_1 = qutip.tensor(spin_down(), qutip.basis(5, 1))
        diff = (ad * psi_0) - psi_1
        assert diff.norm() < 1e-14

    def test_number_of_fock_n_is_n(self) -> None:
        """n̂ |↓, 3⟩ = 3 |↓, 3⟩."""
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 5})
        n_op = h.number_for_mode("axial")
        psi_3 = qutip.tensor(spin_down(), qutip.basis(5, 3))
        eigval = qutip.expect(n_op, psi_3)
        assert eigval == pytest.approx(3.0)

    def test_number_equals_creation_times_annihilation(self) -> None:
        """n̂ = a† a (consistency of the three primitives)."""
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 5})
        n_direct = h.number_for_mode("axial")
        n_composed = h.creation_for_mode("axial") * h.annihilation_for_mode("axial")
        diff = n_direct - n_composed
        assert diff.norm() < 1e-14

    def test_primitives_respect_multi_mode_ordering(self) -> None:
        """In a two-mode system, a_com must act on mode 0 slot and
        a_stretch on mode 1 slot — verify via expectation."""
        h = HilbertSpace(
            system=_two_ion_two_mode_system(),
            fock_truncations={"com": 4, "stretch": 4},
        )
        a_com = h.annihilation_for_mode("com")
        a_str = h.annihilation_for_mode("stretch")
        # |↓, ↓, 2_com, 0_stretch⟩
        psi = qutip.tensor(spin_down(), spin_down(), qutip.basis(4, 2), qutip.basis(4, 0))
        # a_com lowers com by one → norm = √2; a_stretch annihilates → 0
        assert (a_com * psi).norm() == pytest.approx(np.sqrt(2.0))
        assert (a_str * psi).norm() < 1e-14


# ----------------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------------


class TestIdentity:
    def test_identity_dimensions(self) -> None:
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 10})
        i = h.identity()
        assert i.dims == [[2, 10], [2, 10]]
        assert i.shape == (20, 20)

    def test_identity_squared_equals_identity(self) -> None:
        h = HilbertSpace(system=_single_ion_system(), fock_truncations={"axial": 6})
        i = h.identity()
        diff = (i * i) - i
        assert diff.norm() < 1e-14

    def test_identity_in_two_ion_two_mode(self) -> None:
        h = HilbertSpace(
            system=_two_ion_two_mode_system(),
            fock_truncations={"com": 5, "stretch": 4},
        )
        i = h.identity()
        assert i.shape == (2 * 2 * 5 * 4, 2 * 2 * 5 * 4)
