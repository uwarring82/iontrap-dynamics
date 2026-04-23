# SPDX-License-Identifier: MIT
"""N=2 and N=3 Clos/Porras reproduction regression (Dispatch AAF)."""

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
    clos2016_axial_mode_reference,
    load_clos2016_theory_dimension_surface,
)
from iontrap_dynamics.species import mg25_plus

pytestmark = pytest.mark.regression_reproduction


@pytest.mark.parametrize(
    ("n_ions", "cutoff", "rtol"),
    [
        # Achieved tolerances on the 3-significant-figure published table:
        #   N=2 cutoff=8 → max |Δ| / |IPR_av| ≈ 6.3 % (~0.2 s)
        #   N=3 cutoff=6 → max |Δ| / |IPR_av| ≈ 4.2 % (~6 s)
        # cutoff=8 for N=2 matches the inferred-converged value from
        # ``2ions_ipr_vs_nc.txt``. cutoff=6 for N=3 is below the inferred-
        # converged value (10) but is the largest cutoff that fits the
        # regression-suite time budget; the comparison is row-vs-row at the
        # same cutoff in ``theo_dim_N_3.dat``, so the truncation matches on
        # both sides.
        (2, 8, 0.08),
        (3, 6, 0.06),
    ],
)
def test_archived_ipr_av_surface_reproduces_with_dense_full_ld_path(
    n_ions: int,
    cutoff: int,
    rtol: float,
) -> None:
    surface = load_clos2016_theory_dimension_surface(n_ions)
    axial_modes = clos2016_axial_mode_reference(n_ions)

    cutoff_index = int(np.flatnonzero(surface.cutoffs == cutoff)[0])
    reference = surface.averaged_effective_dimension[cutoff_index]
    initial_state = clos2016_initial_state(
        max_phonons=cutoff,
        mean_occupations=[surface.mean_occupation] * n_ions,
        theta_rad=0.0,
        phi_rad=0.0,
    )

    calculated = []
    for detuning in surface.detunings_legacy_units:
        hamiltonian = clos2016_spin_boson_hamiltonian(
            max_phonons=cutoff,
            axial_frequency_rad_s=surface.omega_axial_legacy_units * 2 * np.pi * 1e6,
            dimensionless_mode_frequencies=axial_modes.dimensionless_frequencies.tolist(),
            center_mode_weights=axial_modes.first_ion_participation_weights.tolist(),
            carrier_rabi_frequency_rad_s=surface.omega_rabi_legacy_units * 2 * np.pi * 1e6,
            detuning_rad_s=detuning * 2 * np.pi * 1e6,
            ion_mass_kg=mg25_plus().mass_kg,
            laser_wavelength_m=CLOS2016_LEGACY_WAVELENGTH_M,
        )
        spectrum = solve_spectrum(hamiltonian, initial_state=initial_state)
        calculated.append(clos2016_averaged_effective_dimension(spectrum, initial_state))

    np.testing.assert_allclose(np.asarray(calculated), reference, rtol=rtol, atol=0.0)
