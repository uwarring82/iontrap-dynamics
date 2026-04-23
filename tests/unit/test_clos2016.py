# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.clos2016`."""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics import (
    CLOS2016_LEGACY_WAVELENGTH_M,
    clos2016_averaged_effective_dimension,
    clos2016_initial_state,
    clos2016_spin_boson_hamiltonian,
    effective_dimension,
    solve_spectrum,
)
from iontrap_dynamics.exceptions import ConventionError
from iontrap_dynamics.operators import sigma_x_ion, sigma_z_ion, spin_up
from iontrap_dynamics.species import mg25_plus


class TestClos2016SpinBosonHamiltonian:
    def test_dims_and_hermiticity_for_two_modes(self) -> None:
        hamiltonian = clos2016_spin_boson_hamiltonian(
            max_phonons=2,
            axial_frequency_rad_s=2 * np.pi * 1.2e6,
            dimensionless_mode_frequencies=[1.0, 1.35],
            center_mode_weights=[1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
            carrier_rabi_frequency_rad_s=2 * np.pi * 0.5e6,
            detuning_rad_s=2 * np.pi * 0.15e6,
            ion_mass_kg=mg25_plus().mass_kg,
            phase_rad=np.pi / 7.0,
        )

        assert hamiltonian.dims == [[2, 3, 3], [2, 3, 3]]
        assert (hamiltonian - hamiltonian.dag()).norm() < 1e-12

    def test_large_wavelength_limit_reduces_to_carrier_number_and_detuning(self) -> None:
        max_phonons = 3
        mode_dim = max_phonons + 1
        axial_frequency = 2 * np.pi * 1.1e6
        dimensionless_mode_frequency = 1.4
        rabi_frequency = 2 * np.pi * 0.4e6
        detuning = 2 * np.pi * 0.07e6

        hamiltonian = clos2016_spin_boson_hamiltonian(
            max_phonons=max_phonons,
            axial_frequency_rad_s=axial_frequency,
            dimensionless_mode_frequencies=[dimensionless_mode_frequency],
            center_mode_weights=[1.0],
            carrier_rabi_frequency_rad_s=rabi_frequency,
            detuning_rad_s=detuning,
            ion_mass_kg=mg25_plus().mass_kg,
            laser_wavelength_m=1e12,
            phase_rad=0.0,
        )

        expected = (
            qutip.tensor(
                (rabi_frequency / 2.0) * sigma_x_ion() + (detuning / 2.0) * sigma_z_ion(),
                qutip.qeye(mode_dim),
            )
            + axial_frequency
            * dimensionless_mode_frequency
            * qutip.tensor(qutip.qeye(2), qutip.num(mode_dim))
        )

        assert (hamiltonian - expected).norm() < 5e-8

    def test_rejects_mismatched_mode_lengths(self) -> None:
        with pytest.raises(ConventionError, match="same length"):
            clos2016_spin_boson_hamiltonian(
                max_phonons=2,
                axial_frequency_rad_s=2 * np.pi * 1.0e6,
                dimensionless_mode_frequencies=[1.0, 1.2],
                center_mode_weights=[1.0],
                carrier_rabi_frequency_rad_s=2 * np.pi * 0.5e6,
                detuning_rad_s=0.0,
                ion_mass_kg=mg25_plus().mass_kg,
            )


class TestClos2016InitialState:
    def test_zero_temperature_theta_zero_gives_spin_up_times_vacuum(self) -> None:
        rho0 = clos2016_initial_state(max_phonons=2, mean_occupations=[0.0], theta_rad=0.0, phi_rad=0.0)
        expected = qutip.ket2dm(qutip.tensor(spin_up(), qutip.basis(3, 0)))

        assert rho0.dims == expected.dims
        assert (rho0 - expected).norm() < 1e-12

    def test_legacy_default_wavelength_matches_raman_two_photon_reference(self) -> None:
        # `ergodic_ipr_av.m:41` and every `_ipr_av*` sibling use
        # eta_calculator(25, 279.63/sqrt(2.), axial); the `theo_dim_N_*.dat`
        # tables are downstream of that call, so the public default must
        # match it (not the bare 200 nm single-photon wavelength used by the
        # non-`_ipr_av` ergodic.m drivers).
        np.testing.assert_allclose(CLOS2016_LEGACY_WAVELENGTH_M, 279.63e-9 / np.sqrt(2.0))


class TestClos2016AveragedEffectiveDimension:
    def test_legacy_ipr_av_differs_from_standard_mixed_state_deff(self) -> None:
        spectrum = solve_spectrum(np.diag(np.arange(4, dtype=np.float64)))
        rho0 = np.diag([0.5, 0.5, 0.0, 0.0]).astype(np.complex128)

        assert effective_dimension(spectrum, rho0) == pytest.approx(2.0)
        assert clos2016_averaged_effective_dimension(spectrum, rho0) == pytest.approx(1.0)
