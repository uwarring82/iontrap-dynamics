# SPDX-License-Identifier: MIT
"""Hamiltonian builders for trapped-ion spin–motion dynamics.

Each builder takes a :class:`HilbertSpace` plus configuration
(:class:`DriveConfig`, ion and mode indices, etc.) and returns a QuTiP
:class:`qutip.Qobj` representing the Hamiltonian acting on the full
tensor-product space. The returned operator is stored as ``H/ℏ`` in
rad·s⁻¹ (CONVENTIONS.md §1 internal units) so that consumers can pass
it to ``qutip.mesolve`` without additional rescaling.

Scope for v0.1
--------------

Only the on-resonance carrier builder. Detuned drives require QuTiP's
time-dependent list format and will be added once the first
time-independent path is proven against the analytic-regression
carrier-Rabi formulas.

Follow-on builders (one per dispatch):

- **Carrier, detuned** — ``detuning_rad_s ≠ 0``; list format.
- **Red sideband** — ``Ω_{n → n−1} = |η|·√n·Ω`` with the full
  Lamb–Dicke operator ``e^{iη(a + a†)}`` in the non-LD case or
  leading-order expansion in the LD regime.
- **Blue sideband** — symmetric counterpart.
- **Mølmer–Sørensen** — two-ion bichromatic gate.
- **Stroboscopic AC drive** — time-modulated carrier per the qc.py
  scenario 4 envelope.
"""

from __future__ import annotations

import math

import qutip

from .drives import DriveConfig
from .exceptions import ConventionError
from .hilbert import HilbertSpace
from .operators import sigma_x_ion, sigma_y_ion


def carrier_hamiltonian(
    hilbert: HilbertSpace,
    drive: DriveConfig,
    *,
    ion_index: int,
) -> qutip.Qobj:
    """Return the on-resonance carrier Hamiltonian for a single ion.

    Implements CONVENTIONS.md §5 for ``δ = 0``:

    .. math::
        H_{\\text{carrier}} = \\frac{\\hbar \\Omega}{2}
        \\left[ \\sigma_+ e^{i\\phi} + \\sigma_- e^{-i\\phi} \\right]

    which, using the atomic-physics Pauli convention (§3)
    ``σ_y_ion = −i(σ_+ − σ_−)``, rewrites as

    .. math::
        H / \\hbar = \\frac{\\Omega}{2}
        \\left[ \\cos(\\phi) \\, \\sigma_x - \\sin(\\phi) \\, \\sigma_y \\right].

    The returned :class:`qutip.Qobj` is this ``H/ℏ`` expression embedded
    on the full tensor-product space, with identity operators on the
    other spins and all modes. Units: rad·s⁻¹.

    Parameters
    ----------
    hilbert
        The full tensor-product Hilbert space.
    drive
        :class:`DriveConfig` for the carrier drive. Must be on-resonance
        (``detuning_rad_s == 0`` exactly).
    ion_index
        Zero-based index of the ion the drive couples to. Keyword-only
        to prevent accidental positional swap with multi-argument
        callers (e.g. future two-ion Mølmer–Sørensen builders that take
        two indices).

    Returns
    -------
    qutip.Qobj
        Time-independent Hermitian operator on the full Hilbert space
        with dimensions :meth:`HilbertSpace.qutip_dims`.

    Raises
    ------
    ConventionError
        If ``drive.detuning_rad_s != 0``. (Detuned carrier drives will
        be added once the time-dependent list-format path is proven.)
    IndexError
        If ``ion_index`` is outside ``[0, n_ions)`` — propagated from
        :meth:`HilbertSpace.spin_op_for_ion`.

    Example
    -------

    Single-ion on-resonance π/2-pulse drive (φ = 0, so H/ℏ = (Ω/2) σ_x)::

        from iontrap_dynamics.drives import DriveConfig
        from iontrap_dynamics.hamiltonians import carrier_hamiltonian

        drive = DriveConfig(
            k_vector_m_inv=[2e7, 0.0, 0.0],
            carrier_rabi_frequency_rad_s=2 * np.pi * 1e6,  # 1 MHz Rabi
        )
        H = carrier_hamiltonian(hilbert, drive, ion_index=0)

    The π-pulse duration is then ``t_π = π / Ω`` (CONVENTIONS.md §5
    reference carrier: σ_z = −1 → +1 after a π-pulse).
    """
    if drive.detuning_rad_s != 0.0:
        raise ConventionError(
            f"carrier_hamiltonian currently supports only on-resonance drives "
            f"(detuning_rad_s == 0); got {drive.detuning_rad_s!r}. Detuned "
            "carrier drives require QuTiP's time-dependent list format and "
            "will be added in a follow-on dispatch."
        )

    omega = drive.carrier_rabi_frequency_rad_s
    phi = drive.phase_rad

    # H/ℏ = (Ω/2) · [cos(φ) σ_x − sin(φ) σ_y]  (derivation in module docstring)
    sigma_x = hilbert.spin_op_for_ion(sigma_x_ion(), ion_index)
    sigma_y = hilbert.spin_op_for_ion(sigma_y_ion(), ion_index)
    return (omega / 2.0) * (math.cos(phi) * sigma_x - math.sin(phi) * sigma_y)


__all__ = [
    "carrier_hamiltonian",
]
