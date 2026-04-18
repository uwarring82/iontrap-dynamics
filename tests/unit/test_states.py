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
from iontrap_dynamics.states import (
    coherent_mode,
    compose_density,
    ground_state,
    squeezed_coherent_mode,
    squeezed_vacuum_mode,
)
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


# ============================================================================
# Named single-mode state factories (coherent / squeezed / squeezed-coherent)
# ============================================================================


class TestCoherentMode:
    def test_returns_normalised_ket(self) -> None:
        psi = coherent_mode(15, 1.0)
        assert psi.isket
        assert psi.dims == [[15], [1]]
        assert abs(float(psi.norm()) - 1.0) < 1e-12

    def test_alpha_zero_gives_vacuum(self) -> None:
        psi = coherent_mode(10, 0.0)
        assert (psi - qutip.basis(10, 0)).norm() < 1e-12

    def test_mean_n_equals_alpha_magnitude_squared(self) -> None:
        """⟨n̂⟩ = |α|² for a coherent state."""
        for alpha in (0.5, 1.0, 1.5 + 0.5j, -0.7j):
            psi = coherent_mode(25, alpha)
            n_op = qutip.num(25)
            expected = abs(alpha) ** 2
            assert float(qutip.expect(n_op, psi).real) == pytest.approx(expected, abs=5e-3)

    def test_nonpositive_fock_dim_rejected(self) -> None:
        with pytest.raises(ConventionError, match="positive"):
            coherent_mode(0, 1.0)
        with pytest.raises(ConventionError, match="positive"):
            coherent_mode(-5, 1.0)

    def test_a_eigenvalue_matches_alpha(self) -> None:
        """a|α⟩ = α|α⟩ to within truncation error."""
        alpha = 1.2 + 0.3j
        psi = coherent_mode(30, alpha)
        a = qutip.destroy(30)
        lhs = a * psi
        rhs = alpha * psi
        assert (lhs - rhs).norm() < 1e-6


class TestSqueezedVacuumMode:
    def test_returns_normalised_ket(self) -> None:
        psi = squeezed_vacuum_mode(20, 0.3)
        assert psi.isket
        assert abs(float(psi.norm()) - 1.0) < 1e-12

    def test_z_zero_gives_vacuum(self) -> None:
        psi = squeezed_vacuum_mode(10, 0.0)
        assert (psi - qutip.basis(10, 0)).norm() < 1e-12

    def test_mean_n_equals_sinh_squared(self) -> None:
        """⟨n̂⟩ = sinh²(|ξ|) for squeezed vacuum."""
        for r in (0.1, 0.3, 0.5, 0.8):
            psi = squeezed_vacuum_mode(30, r)
            n_op = qutip.num(30)
            expected = np.sinh(r) ** 2
            assert float(qutip.expect(n_op, psi).real) == pytest.approx(expected, abs=5e-3)

    def test_squeezes_x_quadrature_at_phase_zero(self) -> None:
        """At φ=0 (real ξ>0), the X quadrature variance is reduced
        below the vacuum floor."""
        n_fock = 30
        psi = squeezed_vacuum_mode(n_fock, 0.5)
        a = qutip.destroy(n_fock)
        x_op = (a + a.dag()) / np.sqrt(2.0)
        var_x = float((qutip.expect(x_op * x_op, psi) - qutip.expect(x_op, psi) ** 2).real)
        # Vacuum variance is 0.5; squeezed by r=0.5 gives e^{-1} * 0.5 ≈ 0.184
        assert var_x < 0.5  # at least squeezed
        assert var_x == pytest.approx(0.5 * np.exp(-2 * 0.5), abs=1e-3)

    def test_nonpositive_fock_dim_rejected(self) -> None:
        with pytest.raises(ConventionError, match="positive"):
            squeezed_vacuum_mode(0, 0.3)


class TestSqueezedCoherentMode:
    def test_returns_normalised_ket(self) -> None:
        psi = squeezed_coherent_mode(20, z=0.3, alpha=1.0)
        assert psi.isket
        assert abs(float(psi.norm()) - 1.0) < 1e-12

    def test_mean_n_equals_alpha_squared_plus_sinh_squared(self) -> None:
        """⟨n̂⟩ = |α|² + sinh²(|ξ|)."""
        n_fock = 30
        for alpha in (1.0, 0.5 + 0.5j):
            for r in (0.1, 0.3, 0.5):
                psi = squeezed_coherent_mode(n_fock, z=r, alpha=alpha)
                expected = abs(alpha) ** 2 + np.sinh(r) ** 2
                n_op = qutip.num(n_fock)
                assert float(qutip.expect(n_op, psi).real) == pytest.approx(expected, abs=5e-3)

    def test_zero_squeeze_recovers_coherent(self) -> None:
        """ξ=0 reduces to the plain coherent state."""
        psi_sq = squeezed_coherent_mode(20, z=0.0, alpha=1.2)
        psi_coh = coherent_mode(20, 1.2)
        assert (psi_sq - psi_coh).norm() < 1e-10

    def test_zero_alpha_recovers_squeezed_vacuum(self) -> None:
        """α=0 reduces to squeezed vacuum (D(0) = I)."""
        psi_sq_coh = squeezed_coherent_mode(20, z=0.3, alpha=0.0)
        psi_sq_vac = squeezed_vacuum_mode(20, 0.3)
        assert (psi_sq_coh - psi_sq_vac).norm() < 1e-10

    def test_keyword_only_construction(self) -> None:
        """``z`` and ``alpha`` are keyword-only to prevent positional swap."""
        with pytest.raises(TypeError):
            squeezed_coherent_mode(20, 0.3, 1.0)  # type: ignore[misc]

    def test_nonpositive_fock_dim_rejected(self) -> None:
        with pytest.raises(ConventionError, match="positive"):
            squeezed_coherent_mode(0, z=0.3, alpha=1.0)

    def test_composes_with_compose_density(self) -> None:
        """The factory output feeds straight into :func:`compose_density`."""
        mode = _single_ion_axial()
        system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
        fock = 15
        h = HilbertSpace(system=system, fock_truncations={"axial": fock})

        rho = compose_density(
            h,
            spin_states_per_ion=[spin_down()],
            mode_states_by_label={"axial": squeezed_coherent_mode(fock, z=0.3, alpha=1.0)},
        )
        assert rho.isoper
        assert rho.dims == h.qutip_dims()
        # Trace preserved
        assert float(rho.tr().real) == pytest.approx(1.0, abs=1e-10)

    def test_errors_subclass_iontraperror(self) -> None:
        with pytest.raises(IonTrapError):
            squeezed_coherent_mode(-1, z=0.0, alpha=0.0)
