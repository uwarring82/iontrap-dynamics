# SPDX-License-Identifier: MIT
"""Clos/Porras 2016 reproduction helpers.

This module carries the pieces that do not fit the repo's textbook
single-ion / multi-ion RWA builder surface cleanly:

- the legacy full-exponential one-spin-plus-N-modes Hamiltonian
- the matching thermal initial state
- the legacy averaged effective-dimension quantity ``IPR_av``

The quantity named ``IPR_av`` in the MATLAB bundle is *not* the standard
mixed-state effective dimension ``1 / sum(p_alpha^2)``. It is a weighted
average of pure-state effective dimensions over the eigendecomposition of
the initial mixed state. This module keeps that legacy quantity explicit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import qutip
from numpy.typing import NDArray

from .exceptions import ConventionError
from .operators import sigma_minus_ion, sigma_plus_ion, sigma_z_ion, spin_down, spin_up
from .spectrum import SpectrumResult

# Effective two-photon Raman wavelength used by every `ergodic_ipr_av*.m`
# variant in the legacy bundle (e.g. `eta_calculator(25, 279.63/sqrt(2.),
# axial)` in `ergodic_ipr_av.m:41`). The non-`_ipr_av` drivers
# (`ergodic.m`, `sb_evolution*.m`, …) use the bare 200 nm single-photon
# reference instead, but the `theo_dim_N_*.dat` tables — the regression
# anchor consumed by `IPR_av` reproduction — are produced only by the
# `_ipr_av*` family. Keep this constant pinned to the actual data
# provenance.
CLOS2016_LEGACY_WAVELENGTH_M = 279.63e-9 / math.sqrt(2.0)
_HBAR = 1.054571817e-34


def clos2016_spin_boson_hamiltonian(
    *,
    max_phonons: int,
    axial_frequency_rad_s: float,
    dimensionless_mode_frequencies: Sequence[float],
    center_mode_weights: Sequence[float],
    carrier_rabi_frequency_rad_s: float,
    detuning_rad_s: float,
    ion_mass_kg: float,
    laser_wavelength_m: float = CLOS2016_LEGACY_WAVELENGTH_M,
    phase_rad: float = 0.0,
) -> qutip.Qobj:
    r"""Return the legacy full-exponential spin-boson Hamiltonian.

    Implements the MATLAB structure

    .. math::
        H = \frac{\Omega}{2}\left[\sigma_+ e^{\hat{P}} e^{i\phi}
          + \sigma_- e^{-\hat{P}} e^{-i\phi}\right]
          + \sum_n \omega_n a_n^\dagger a_n
          + \frac{\delta}{2}\sigma_z

    with

    .. math::
        \hat{P} = \sum_n \eta_0 \frac{\phi_n}{\sqrt{\omega_n/\omega_1}}
        (a_n - a_n^\dagger),

    where ``ω_1`` is the axial reference frequency passed as
    ``axial_frequency_rad_s``.
    """
    if max_phonons < 0:
        raise ConventionError(f"max_phonons must be >= 0; got {max_phonons}.")
    if axial_frequency_rad_s <= 0.0:
        raise ConventionError(
            f"axial_frequency_rad_s must be positive; got {axial_frequency_rad_s!r}."
        )
    if carrier_rabi_frequency_rad_s <= 0.0:
        raise ConventionError(
            "carrier_rabi_frequency_rad_s must be positive for the Clos 2016 Hamiltonian."
        )
    if ion_mass_kg <= 0.0:
        raise ConventionError(f"ion_mass_kg must be positive; got {ion_mass_kg!r}.")
    if laser_wavelength_m <= 0.0:
        raise ConventionError(f"laser_wavelength_m must be positive; got {laser_wavelength_m!r}.")

    mode_frequencies = np.asarray(dimensionless_mode_frequencies, dtype=np.float64)
    center_weights = np.asarray(center_mode_weights, dtype=np.float64)
    if mode_frequencies.ndim != 1 or center_weights.ndim != 1:
        raise ConventionError("dimensionless_mode_frequencies and center_mode_weights must be 1-D.")
    if mode_frequencies.size == 0:
        raise ConventionError("Clos 2016 Hamiltonian requires at least one bosonic mode.")
    if mode_frequencies.shape != center_weights.shape:
        raise ConventionError(
            "dimensionless_mode_frequencies and center_mode_weights must have the same length."
        )
    if np.any(mode_frequencies <= 0.0):
        raise ConventionError("dimensionless_mode_frequencies must all be positive.")

    mode_dim = max_phonons + 1
    n_modes = int(mode_frequencies.size)
    spin_identity = qutip.qeye(2)
    mode_identity = qutip.qeye(mode_dim)

    def spin_op(op: qutip.Qobj) -> qutip.Qobj:
        return qutip.tensor(op, *[mode_identity for _ in range(n_modes)])

    def mode_op(op: qutip.Qobj, mode_index: int) -> qutip.Qobj:
        factors = [spin_identity]
        for index in range(n_modes):
            factors.append(op if index == mode_index else mode_identity)
        return qutip.tensor(*factors)

    eta_0 = _legacy_eta0(
        ion_mass_kg=ion_mass_kg,
        laser_wavelength_m=laser_wavelength_m,
        axial_frequency_rad_s=axial_frequency_rad_s,
    )

    pol = 0 * spin_op(qutip.qeye(2))
    H_n = 0 * spin_op(qutip.qeye(2))
    for mode_index, (ratio, weight) in enumerate(
        zip(mode_frequencies, center_weights, strict=True)
    ):
        a_mode = mode_op(qutip.destroy(mode_dim), mode_index)
        eta_mode = eta_0 * weight / math.sqrt(float(ratio))
        pol = pol + eta_mode * (a_mode - a_mode.dag())
        H_n = H_n + axial_frequency_rad_s * float(ratio) * (a_mode.dag() * a_mode)

    sigma_plus = spin_op(sigma_plus_ion())
    sigma_minus = spin_op(sigma_minus_ion())
    sigma_z = spin_op(sigma_z_ion())
    phase_plus = complex(np.exp(1j * phase_rad))
    phase_minus = phase_plus.conjugate()

    exp_pol = pol.expm()
    H_sph = (carrier_rabi_frequency_rad_s / 2.0) * (
        phase_plus * sigma_plus * exp_pol + phase_minus * sigma_minus * exp_pol.dag()
    )
    hamiltonian = H_sph + H_n + (detuning_rad_s / 2.0) * sigma_z
    # Match the MATLAB implementation, which explicitly symmetrizes after
    # assembly to remove tiny `expm` roundoff asymmetries.
    return 0.5 * (hamiltonian + hamiltonian.dag())


def clos2016_initial_state(
    *,
    max_phonons: int,
    mean_occupations: Sequence[float],
    theta_rad: float = 0.0,
    phi_rad: float = 0.0,
) -> qutip.Qobj:
    """Return the legacy thermal-bath / Bloch-sphere spin initial state.

    Matches the MATLAB convention

    ``psi = cos(theta/2)|up> + sin(theta/2) e^{-i phi} |down>``.
    """
    if max_phonons < 0:
        raise ConventionError(f"max_phonons must be >= 0; got {max_phonons}.")
    n_bars = np.asarray(mean_occupations, dtype=np.float64)
    if n_bars.ndim != 1 or n_bars.size == 0:
        raise ConventionError("mean_occupations must be a non-empty 1-D sequence.")
    if np.any(n_bars < 0.0):
        raise ConventionError("mean_occupations must all be >= 0.")

    spin_ket = (
        math.cos(theta_rad / 2.0) * spin_up()
        + math.sin(theta_rad / 2.0) * np.exp(-1j * phi_rad) * spin_down()
    )
    spin_dm = qutip.ket2dm(spin_ket.unit())
    mode_dim = max_phonons + 1
    mode_states = [qutip.thermal_dm(mode_dim, float(n_bar)) for n_bar in n_bars]
    return qutip.tensor(spin_dm, *mode_states)


def clos2016_averaged_effective_dimension(
    spectrum: SpectrumResult,
    initial_state: qutip.Qobj | NDArray[np.complexfloating[Any]] | NDArray[np.floating[Any]],
) -> float:
    """Return the legacy ``IPR_av`` quantity from the MATLAB bundle.

    If ``rho_0 = sum_j d_j |psi_j><psi_j|``, the returned value is

    ``sum_j d_j / sum_a |<E_a | psi_j>|^4``.
    """
    eigenvectors = _eigenvector_matrix(spectrum)
    rho = _as_density_matrix(initial_state, eigenvectors.shape[0])
    eigenvalues_rho, eigenvectors_rho = np.linalg.eigh(rho)
    overlaps = eigenvectors.conj().T @ eigenvectors_rho
    deff_per_component = 1.0 / np.sum(np.abs(overlaps) ** 4, axis=0)
    weights = np.real_if_close(eigenvalues_rho, tol=1000)
    if np.iscomplexobj(weights):
        raise ConventionError("initial_state density matrix produced complex eigenvalues.")
    weights = np.asarray(weights, dtype=np.float64)
    weights[np.abs(weights) < 1e-14] = 0.0
    if np.any(weights < -1e-10):
        raise ConventionError("initial_state density matrix must be positive semidefinite.")
    return float(np.sum(deff_per_component * np.clip(weights, 0.0, None)))


def _legacy_eta0(
    *,
    ion_mass_kg: float,
    laser_wavelength_m: float,
    axial_frequency_rad_s: float,
) -> float:
    """Return the legacy axial reference Lamb-Dicke parameter ``eta0``."""
    wavenumber_m_inv = 2.0 * math.pi / laser_wavelength_m
    x0 = math.sqrt(_HBAR / (2.0 * ion_mass_kg * axial_frequency_rad_s))
    return wavenumber_m_inv * x0


def _eigenvector_matrix(spectrum: SpectrumResult) -> NDArray[np.complex128]:
    """Materialise eigenvector columns from a :class:`SpectrumResult`."""
    if spectrum.eigenvectors is not None:
        vectors = np.asarray(spectrum.eigenvectors, dtype=np.complex128)
    else:
        assert spectrum.eigenvectors_loader is not None
        columns = [
            np.asarray(spectrum.eigenvectors_loader(i), dtype=np.complex128).reshape(-1)
            for i in range(spectrum.eigenvalues.shape[0])
        ]
        if not columns:
            raise ConventionError("SpectrumResult must carry at least one eigenvector.")
        vectors = np.column_stack(columns)

    normalized = np.empty_like(vectors, dtype=np.complex128)
    for index in range(vectors.shape[1]):
        column = vectors[:, index]
        norm = np.linalg.norm(column)
        if np.isclose(norm, 0.0):
            raise ConventionError(f"eigenvector column {index} has zero norm.")
        normalized[:, index] = column / norm
    return normalized


def _as_density_matrix(
    initial_state: qutip.Qobj | NDArray[np.complexfloating[Any]] | NDArray[np.floating[Any]],
    dimension: int,
) -> NDArray[np.complex128]:
    """Convert a ket or density matrix to a normalized dense density matrix."""
    if isinstance(initial_state, qutip.Qobj):
        state = np.asarray(initial_state.full(), dtype=np.complex128)
    else:
        state = np.asarray(initial_state, dtype=np.complex128)

    if state.ndim == 1 or (state.ndim == 2 and 1 in state.shape):
        ket = state.reshape(-1)
        if ket.shape[0] != dimension:
            raise ConventionError(
                f"initial_state ket has length {ket.shape[0]}, expected {dimension}."
            )
        norm = np.vdot(ket, ket)
        if np.isclose(norm, 0.0):
            raise ConventionError("initial_state ket must have non-zero norm.")
        ket = ket / np.sqrt(norm)
        return np.outer(ket, ket.conj())

    if state.ndim == 2 and state.shape == (dimension, dimension):
        trace = np.trace(state)
        if np.isclose(trace, 0.0):
            raise ConventionError("initial_state density matrix must have non-zero trace.")
        rho = state / trace
        if not np.allclose(rho, rho.conj().T, atol=1e-12):
            raise ConventionError("initial_state density matrix must be Hermitian.")
        return cast("NDArray[np.complex128]", rho)

    raise ConventionError(
        "initial_state must be a ket vector or a square density matrix matching "
        "the spectrum dimension."
    )


__all__ = [
    "CLOS2016_LEGACY_WAVELENGTH_M",
    "clos2016_averaged_effective_dimension",
    "clos2016_initial_state",
    "clos2016_spin_boson_hamiltonian",
]
