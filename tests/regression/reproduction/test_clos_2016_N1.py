# SPDX-License-Identifier: MIT
"""N=1 Clos/Porras reproduction regression."""

from __future__ import annotations

import numpy as np
import pytest

from iontrap_dynamics import (
    CLOS2016_LEGACY_WAVELENGTH_M,
    clos2016_averaged_effective_dimension,
    clos2016_initial_state,
    clos2016_spin_boson_hamiltonian,
    solve_spectrum,
)
from iontrap_dynamics.clos2016_references import (
    load_clos2016_cutoff_convergence,
    load_clos2016_theory_dimension_surface,
)
from iontrap_dynamics.species import mg25_plus

pytestmark = pytest.mark.regression_reproduction


def test_n1_archived_ipr_av_surface_reproduces_with_dense_full_ld_path() -> None:
    convergence = load_clos2016_cutoff_convergence(1)
    surface = load_clos2016_theory_dimension_surface(1)

    cutoff = convergence.inferred_converged_cutoff
    assert cutoff == 7

    cutoff_index = int(np.flatnonzero(surface.cutoffs == cutoff)[0])
    reference = surface.averaged_effective_dimension[cutoff_index]
    initial_state = clos2016_initial_state(
        max_phonons=cutoff,
        mean_occupations=[surface.mean_occupation],
        theta_rad=0.0,
        phi_rad=0.0,
    )

    calculated = []
    for detuning in surface.detunings_legacy_units:
        hamiltonian = clos2016_spin_boson_hamiltonian(
            max_phonons=cutoff,
            axial_frequency_rad_s=surface.omega_axial_legacy_units * 2 * np.pi * 1e6,
            dimensionless_mode_frequencies=[1.0],
            center_mode_weights=[1.0],
            carrier_rabi_frequency_rad_s=surface.omega_rabi_legacy_units * 2 * np.pi * 1e6,
            detuning_rad_s=detuning * 2 * np.pi * 1e6,
            ion_mass_kg=mg25_plus().mass_kg,
            laser_wavelength_m=CLOS2016_LEGACY_WAVELENGTH_M,
        )
        spectrum = solve_spectrum(hamiltonian, initial_state=initial_state)
        calculated.append(clos2016_averaged_effective_dimension(spectrum, initial_state))

    # Achieved: max |Δd_eff/d_eff| ≈ 8.8 % at the sharp mid-resonance peaks
    # (det ≈ 1.0, 2.0 in legacy units), where ``IPR_av`` varies steeply and
    # the published table carries only 3 significant figures. Workplan §5
    # AAE allowed documenting whatever tolerance the pipeline actually
    # achieves; see §6 Risk 3 commentary for the rationale.
    np.testing.assert_allclose(np.asarray(calculated), reference, rtol=0.10, atol=0.0)
