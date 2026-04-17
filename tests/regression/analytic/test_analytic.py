# SPDX-License-Identifier: MIT
"""Analytic-regression tests: closed-form reference physics.

Permanent tier of the three-layer regression harness (workplan §0.B).
Exercises each closed-form reference in :mod:`iontrap_dynamics.analytic`
against values derivable from textbook physics. These tests do not depend
on QuTiP or any solver — they ARE the solver truth in the limits they cover.

Coverage
--------

- Carrier flopping — population at canonical pulse times; ⟨σ_z⟩ sign
  convention (↓ = −1, ↑ = +1); array broadcasting.
- Sideband Rabi frequencies — vacuum-state red sideband is identically
  zero; blue sideband is non-zero from vacuum; √n scaling for red.
- Lamb–Dicke parameter — k ∥ b, k ⊥ b, oblique geometries; sign
  preservation; input validation.
- Coherent-state ⟨n⟩ = |α|².
"""

from __future__ import annotations

import numpy as np
import pytest

from iontrap_dynamics.analytic import (
    HBAR,
    blue_sideband_rabi_frequency,
    carrier_rabi_excited_population,
    carrier_rabi_sigma_z,
    coherent_state_mean_n,
    lamb_dicke_parameter,
    red_sideband_rabi_frequency,
)

pytestmark = pytest.mark.regression_analytic


# ----------------------------------------------------------------------------
# Carrier flopping
# ----------------------------------------------------------------------------


class TestCarrierFlopping:
    def test_population_at_t_zero_is_zero(self) -> None:
        """Start in |↓⟩: no excitation at t = 0."""
        assert carrier_rabi_excited_population(2 * np.pi * 1.0e6, 0.0) == pytest.approx(0.0)

    def test_population_at_pi_pulse_is_one(self) -> None:
        """A π-pulse takes |↓⟩ → |↑⟩: full excitation."""
        omega = 2 * np.pi * 1.0e6
        t_pi = np.pi / omega
        assert carrier_rabi_excited_population(omega, t_pi) == pytest.approx(1.0)

    def test_population_at_half_pi_pulse_is_one_half(self) -> None:
        """A π/2-pulse puts |↓⟩ into an equal superposition."""
        omega = 2 * np.pi * 1.0e6
        t_half_pi = np.pi / (2 * omega)
        assert carrier_rabi_excited_population(omega, t_half_pi) == pytest.approx(0.5)

    def test_population_accepts_array_time(self) -> None:
        """Broadcasting over a time array works."""
        omega = 2 * np.pi * 1.0e6
        times = np.linspace(0.0, 2 * np.pi / omega, 5)  # one full cycle + one extra
        p = carrier_rabi_excited_population(omega, times)
        assert p.shape == times.shape
        assert np.all((p >= 0.0) & (p <= 1.0))

    def test_sigma_z_sign_convention_matches_conventions_md(self) -> None:
        """⟨σ_z⟩(t=0) = −1 (in |↓⟩) per CONVENTIONS.md §3."""
        assert carrier_rabi_sigma_z(2 * np.pi * 1.0e6, 0.0) == pytest.approx(-1.0)

    def test_sigma_z_at_pi_pulse_flips_to_plus_one(self) -> None:
        """After π-pulse, fully in |↑⟩: ⟨σ_z⟩ = +1."""
        omega = 2 * np.pi * 1.0e6
        t_pi = np.pi / omega
        assert carrier_rabi_sigma_z(omega, t_pi) == pytest.approx(1.0)

    def test_population_and_sigma_z_consistency(self) -> None:
        """⟨σ_z⟩ = 2·P↑ − 1 identity (same dynamics, different observable)."""
        omega = 2 * np.pi * 1.0e6
        times = np.linspace(0.0, 2 * np.pi / omega, 17)
        p_excited = carrier_rabi_excited_population(omega, times)
        sz = carrier_rabi_sigma_z(omega, times)
        np.testing.assert_allclose(sz, 2 * p_excited - 1.0, atol=1e-14)


# ----------------------------------------------------------------------------
# Sideband Rabi frequencies
# ----------------------------------------------------------------------------


class TestSidebandRabiFrequencies:
    def test_vacuum_red_sideband_is_exactly_zero(self) -> None:
        """|↓, 0⟩ cannot be red-detuned off vacuum."""
        rate = red_sideband_rabi_frequency(
            carrier_rabi_frequency=2 * np.pi * 1.0e6,
            lamb_dicke_parameter=0.1,
            n_initial=0,
        )
        assert rate == 0.0

    def test_red_sideband_scales_as_sqrt_n(self) -> None:
        """Ω_{n→n−1} = |η|·√n·Ω."""
        omega = 2 * np.pi * 1.0e6
        eta = 0.15
        rate_1 = red_sideband_rabi_frequency(
            carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=1
        )
        rate_4 = red_sideband_rabi_frequency(
            carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=4
        )
        assert rate_1 == pytest.approx(eta * omega)
        assert rate_4 == pytest.approx(2.0 * eta * omega)  # √4 = 2

    def test_red_sideband_rejects_negative_n(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            red_sideband_rabi_frequency(
                carrier_rabi_frequency=1.0, lamb_dicke_parameter=0.1, n_initial=-1
            )

    def test_red_sideband_sign_of_eta_does_not_affect_rate(self) -> None:
        """The sign of η enters the phase, not the Rabi rate magnitude."""
        rate_plus = red_sideband_rabi_frequency(
            carrier_rabi_frequency=1.0, lamb_dicke_parameter=+0.1, n_initial=3
        )
        rate_minus = red_sideband_rabi_frequency(
            carrier_rabi_frequency=1.0, lamb_dicke_parameter=-0.1, n_initial=3
        )
        assert rate_plus == rate_minus

    def test_vacuum_blue_sideband_is_non_zero(self) -> None:
        """|↓, 0⟩ → |↑, 1⟩ at rate |η|·Ω."""
        omega = 2 * np.pi * 1.0e6
        eta = 0.1
        rate = blue_sideband_rabi_frequency(
            carrier_rabi_frequency=omega,
            lamb_dicke_parameter=eta,
            n_initial=0,
        )
        assert rate == pytest.approx(eta * omega)

    def test_blue_sideband_scales_as_sqrt_n_plus_one(self) -> None:
        """Ω_{n→n+1} = |η|·√(n+1)·Ω."""
        omega = 2 * np.pi * 1.0e6
        eta = 0.1
        rate_n0 = blue_sideband_rabi_frequency(
            carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=0
        )
        rate_n3 = blue_sideband_rabi_frequency(
            carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=3
        )
        assert rate_n3 == pytest.approx(2.0 * rate_n0)  # √(3+1) = 2

    def test_blue_sideband_rejects_negative_n(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            blue_sideband_rabi_frequency(
                carrier_rabi_frequency=1.0, lamb_dicke_parameter=0.1, n_initial=-1
            )


# ----------------------------------------------------------------------------
# Lamb–Dicke parameter — the CONVENTIONS.md §10 test checklist
# ----------------------------------------------------------------------------

# Typical ²⁵Mg⁺ values for physical plausibility
_MG25_MASS = 24.99 * 1.66053906660e-27  # kg
_TRAP_OMEGA = 2 * np.pi * 1.5e6  # 1.5 MHz axial secular
_WAVENUMBER = 2 * np.pi / 280e-9  # 280 nm laser (Mg S↔P)


class TestLambDicke:
    def test_k_parallel_b_reduces_to_1d_formula(self) -> None:
        """k ∥ b: η = |k|·√(ℏ/2mω)."""
        k = np.array([_WAVENUMBER, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        eta = lamb_dicke_parameter(
            k_vec=k,
            mode_eigenvector=b,
            ion_mass=_MG25_MASS,
            mode_frequency=_TRAP_OMEGA,
        )
        expected = _WAVENUMBER * np.sqrt(HBAR / (2.0 * _MG25_MASS * _TRAP_OMEGA))
        assert eta == pytest.approx(expected)

    def test_k_perpendicular_b_is_exactly_zero(self) -> None:
        """k ⊥ b: η = 0 exactly (no arithmetic slop allowed here)."""
        k = np.array([_WAVENUMBER, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        eta = lamb_dicke_parameter(
            k_vec=k,
            mode_eigenvector=b,
            ion_mass=_MG25_MASS,
            mode_frequency=_TRAP_OMEGA,
        )
        assert eta == 0.0

    def test_oblique_k_matches_cosine_projection(self) -> None:
        """η at angle θ = |k|·cos(θ)·√(ℏ/2mω)."""
        theta = np.deg2rad(30.0)
        k = np.array([_WAVENUMBER * np.cos(theta), _WAVENUMBER * np.sin(theta), 0.0])
        b = np.array([1.0, 0.0, 0.0])  # along x
        eta = lamb_dicke_parameter(
            k_vec=k,
            mode_eigenvector=b,
            ion_mass=_MG25_MASS,
            mode_frequency=_TRAP_OMEGA,
        )
        expected = _WAVENUMBER * np.cos(theta) * np.sqrt(HBAR / (2.0 * _MG25_MASS * _TRAP_OMEGA))
        assert eta == pytest.approx(expected)

    def test_sign_is_preserved(self) -> None:
        """η can be negative; the sign is physical (CONVENTIONS.md §10)."""
        k = np.array([_WAVENUMBER, 0.0, 0.0])
        b_plus = np.array([1.0, 0.0, 0.0])
        b_minus = np.array([-1.0, 0.0, 0.0])
        eta_plus = lamb_dicke_parameter(
            k_vec=k,
            mode_eigenvector=b_plus,
            ion_mass=_MG25_MASS,
            mode_frequency=_TRAP_OMEGA,
        )
        eta_minus = lamb_dicke_parameter(
            k_vec=k,
            mode_eigenvector=b_minus,
            ion_mass=_MG25_MASS,
            mode_frequency=_TRAP_OMEGA,
        )
        assert eta_plus > 0
        assert eta_minus == pytest.approx(-eta_plus)

    def test_dimensionless_output(self) -> None:
        """η is dimensionless — a sanity check via order of magnitude."""
        k = np.array([_WAVENUMBER, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        eta = lamb_dicke_parameter(
            k_vec=k,
            mode_eigenvector=b,
            ion_mass=_MG25_MASS,
            mode_frequency=_TRAP_OMEGA,
        )
        # ²⁵Mg⁺ at 280 nm, 1.5 MHz axial has η ~ 0.3 (textbook value)
        assert 0.1 < abs(eta) < 0.5

    @pytest.mark.parametrize(
        ("k_shape", "b_shape"),
        [((2,), (3,)), ((3,), (2,)), ((4,), (4,))],
    )
    def test_rejects_non_3_vectors(
        self, k_shape: tuple[int, ...], b_shape: tuple[int, ...]
    ) -> None:
        with pytest.raises(ValueError, match="3-vector"):
            lamb_dicke_parameter(
                k_vec=np.zeros(k_shape),
                mode_eigenvector=np.zeros(b_shape),
                ion_mass=_MG25_MASS,
                mode_frequency=_TRAP_OMEGA,
            )

    def test_rejects_non_positive_mass(self) -> None:
        k = np.array([_WAVENUMBER, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="ion_mass"):
            lamb_dicke_parameter(
                k_vec=k,
                mode_eigenvector=b,
                ion_mass=0.0,
                mode_frequency=_TRAP_OMEGA,
            )

    def test_rejects_non_positive_mode_frequency(self) -> None:
        k = np.array([_WAVENUMBER, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="mode_frequency"):
            lamb_dicke_parameter(
                k_vec=k,
                mode_eigenvector=b,
                ion_mass=_MG25_MASS,
                mode_frequency=-1.0,
            )


# ----------------------------------------------------------------------------
# Coherent-state occupation
# ----------------------------------------------------------------------------


class TestCoherentState:
    def test_vacuum_has_zero_occupation(self) -> None:
        assert coherent_state_mean_n(0.0 + 0.0j) == 0.0

    def test_unit_alpha_has_unit_occupation(self) -> None:
        assert coherent_state_mean_n(1.0 + 0.0j) == pytest.approx(1.0)

    def test_phase_does_not_affect_occupation(self) -> None:
        """⟨n⟩ depends only on |α|."""
        mag = 1.7
        for phase in (0.0, 0.3, np.pi / 2, np.pi, -1.1):
            alpha = mag * np.exp(1j * phase)
            assert coherent_state_mean_n(alpha) == pytest.approx(mag**2)
