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

All builders here operate in the **interaction picture of the atomic
transition**, with the RWA applied. They return time-independent Qobj's
for drives placed exactly at their respective resonances:

- :func:`carrier_hamiltonian` — on-resonance carrier (``δ = 0``).
- :func:`red_sideband_hamiltonian` — ``ω_laser = ω_atom − ω_mode``.
- :func:`blue_sideband_hamiltonian` — ``ω_laser = ω_atom + ω_mode``.

Off-resonance variants (detuned carrier, near-sideband dynamics) require
QuTiP's time-dependent list format and are slated for a follow-on
dispatch. ``detuning_rad_s`` on the :class:`DriveConfig` is currently
**not validated** against the resonance the chosen builder assumes —
the caller is trusted to pick the matching builder. (A near-future
revision will cross-check this and raise :class:`ConventionError` on
mismatch.)

Follow-on builders (one per dispatch):

- **Carrier, detuned** — ``detuning_rad_s ≠ 0``; list format.
- **Near-sideband** — detuning slightly off resonance, list format.
- **Mølmer–Sørensen (δ = 0, bichromatic)** — two-ion spin-dependent
  force, time-independent and therefore compatible with the existing
  :func:`iontrap_dynamics.sequences.solve` dispatcher. Landed via
  :func:`ms_gate_hamiltonian`.
- **Mølmer–Sørensen (δ ≠ 0, gate-closing)** — bichromatic with a
  symmetric detuning around the sideband; time-dependent list format
  required to close a phase-space loop at ``t_gate = 2π/δ``. Future
  dispatch.
- **Stroboscopic AC drive** — time-modulated carrier per the qc.py
  scenario 4 envelope.

Frame / convention note
-----------------------

The interaction-picture + RWA form here is clean textbook physics but
does **not** match the legacy ``qc.py`` output byte-for-byte: ``qc.py``
keeps the full lab-frame Hamiltonian and does not apply RWA, so its
mesolve output carries both the rotating and counter-rotating
contributions. The migration-regression scenarios (workplan §0.B tier
1) therefore need a frame-transform adapter before comparison; that's
deferred to a future dispatch and is not on the critical path for
Phase 1 textbook correctness.
"""

from __future__ import annotations

import cmath
import math

import qutip

from .analytic import lamb_dicke_parameter
from .drives import DriveConfig
from .exceptions import ConventionError
from .hilbert import HilbertSpace
from .operators import sigma_minus_ion, sigma_plus_ion, sigma_x_ion, sigma_y_ion


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


# ----------------------------------------------------------------------------
# Sideband builders (Lamb–Dicke leading-order, RWA)
# ----------------------------------------------------------------------------


def _sideband_lamb_dicke(
    hilbert: HilbertSpace,
    drive: DriveConfig,
    mode_label: str,
    ion_index: int,
) -> float:
    """Compute η = (k · b_{i,m}) · √(ℏ / 2 m_i ω_m) for a given drive/mode/ion.

    Delegates to :func:`analytic.lamb_dicke_parameter` so the 3D projection
    and CODATA ℏ stay in one place. Sign is preserved (CONVENTIONS.md §10).
    """
    species = hilbert.system.species(ion_index)
    mode = hilbert.system.mode(mode_label)
    return lamb_dicke_parameter(
        k_vec=drive.k_vector_m_inv,
        mode_eigenvector=mode.eigenvector_at_ion(ion_index),
        ion_mass=species.mass_kg,
        mode_frequency=mode.frequency_rad_s,
    )


def red_sideband_hamiltonian(
    hilbert: HilbertSpace,
    drive: DriveConfig,
    mode_label: str,
    *,
    ion_index: int,
) -> qutip.Qobj:
    """Return the red-sideband Hamiltonian in the interaction picture (RWA).

    .. math::
        H / \\hbar = \\frac{\\Omega \\eta}{2}
        \\left[ \\sigma_+ a \\, e^{i\\phi} + \\sigma_- a^\\dagger \\, e^{-i\\phi} \\right]

    where η is the Lamb–Dicke parameter for this (drive, mode, ion) triple
    (CONVENTIONS.md §10; computed via
    :func:`iontrap_dynamics.analytic.lamb_dicke_parameter`). The
    coupling ``σ_+ a`` takes ``|↓, n⟩`` to ``|↑, n − 1⟩`` with amplitude
    ``√n``, giving the textbook red-sideband Rabi rate
    ``|η|·√n·Ω`` per :func:`analytic.red_sideband_rabi_frequency`.

    The caller asserts by choosing this builder that
    ``ω_laser = ω_atom − ω_mode`` (exact red-sideband resonance). The
    ``drive.detuning_rad_s`` field is **not validated** against this
    assumption in v0.1; see the module docstring.

    Parameters
    ----------
    hilbert
        The full tensor-product Hilbert space.
    drive
        :class:`DriveConfig`. Reads Rabi frequency, phase, and wavevector.
        Detuning and polarisation are not consulted in v0.1.
    mode_label
        Label of the motional mode the sideband addresses. Must exist in
        ``hilbert.system.modes``.
    ion_index
        Zero-based index of the ion the drive couples to.

    Returns
    -------
    qutip.Qobj
        Time-independent Hermitian operator on the full Hilbert space.

    Raises
    ------
    ConventionError
        If ``mode_label`` is not a mode of the system (from
        :meth:`HilbertSpace.system.mode`).
    IndexError
        If ``ion_index`` is outside ``[0, n_ions)``.
    """
    eta = _sideband_lamb_dicke(hilbert, drive, mode_label, ion_index)
    omega = drive.carrier_rabi_frequency_rad_s
    phi = drive.phase_rad

    sigma_p = hilbert.spin_op_for_ion(sigma_plus_ion(), ion_index)
    sigma_m = hilbert.spin_op_for_ion(sigma_minus_ion(), ion_index)
    a = hilbert.annihilation_for_mode(mode_label)
    a_dag = hilbert.creation_for_mode(mode_label)

    coeff = omega * eta / 2.0
    phase_plus = cmath.exp(1j * phi)
    phase_minus = phase_plus.conjugate()
    return coeff * (phase_plus * sigma_p * a + phase_minus * sigma_m * a_dag)


def blue_sideband_hamiltonian(
    hilbert: HilbertSpace,
    drive: DriveConfig,
    mode_label: str,
    *,
    ion_index: int,
) -> qutip.Qobj:
    """Return the blue-sideband Hamiltonian in the interaction picture (RWA).

    .. math::
        H / \\hbar = \\frac{\\Omega \\eta}{2}
        \\left[ \\sigma_+ a^\\dagger \\, e^{i\\phi} + \\sigma_- a \\, e^{-i\\phi} \\right]

    Couples ``|↓, n⟩`` to ``|↑, n + 1⟩`` at rate ``|η|·√(n + 1)·Ω`` per
    :func:`analytic.blue_sideband_rabi_frequency`. Non-zero from the
    motional vacuum — the blue sideband can always create a phonon —
    in contrast with :func:`red_sideband_hamiltonian`, which annihilates
    ``|↓, 0⟩``.

    Parameters, returns, and raises match
    :func:`red_sideband_hamiltonian`; the caller asserts
    ``ω_laser = ω_atom + ω_mode``.
    """
    eta = _sideband_lamb_dicke(hilbert, drive, mode_label, ion_index)
    omega = drive.carrier_rabi_frequency_rad_s
    phi = drive.phase_rad

    sigma_p = hilbert.spin_op_for_ion(sigma_plus_ion(), ion_index)
    sigma_m = hilbert.spin_op_for_ion(sigma_minus_ion(), ion_index)
    a = hilbert.annihilation_for_mode(mode_label)
    a_dag = hilbert.creation_for_mode(mode_label)

    coeff = omega * eta / 2.0
    phase_plus = cmath.exp(1j * phi)
    phase_minus = phase_plus.conjugate()
    return coeff * (phase_plus * sigma_p * a_dag + phase_minus * sigma_m * a)


# ----------------------------------------------------------------------------
# Mølmer–Sørensen gate (bichromatic, δ = 0)
# ----------------------------------------------------------------------------


def ms_gate_hamiltonian(
    hilbert: HilbertSpace,
    drive: DriveConfig,
    mode_label: str,
    *,
    ion_indices: tuple[int, int],
) -> qutip.Qobj:
    """Return the Mølmer–Sørensen gate Hamiltonian in the interaction picture
    (symmetric bichromatic drive, δ = 0 — time-independent).

    .. math::
        H / \\hbar = \\sum_{k \\in \\text{ions}}
        \\frac{\\Omega \\, \\eta_k}{2}
        \\left[ \\sigma_+^{(k)} e^{i\\phi}
              + \\sigma_-^{(k)} e^{-i\\phi} \\right]
        \\otimes \\bigl(a + a^\\dagger\\bigr)

    This is the Lamb–Dicke, RWA-reduced form of a **symmetric
    bichromatic drive** with two tones at ``ω_atom ± (ω_mode − δ)``
    under ``δ = 0`` — the two sideband contributions sum coherently to
    a spin-dependent force along the ``σ_φ`` axis, coupled to the
    mode's position quadrature ``x̂ ∝ (a + a†)``. In the σ_x/σ_y
    rewrite (atomic-physics convention §3):

    .. math::
        H / \\hbar = \\sum_k \\frac{\\Omega \\eta_k}{2}
        \\left[ \\cos(\\phi) \\sigma_x^{(k)}
              - \\sin(\\phi) \\sigma_y^{(k)} \\right]
        \\otimes \\bigl(a + a^\\dagger\\bigr),

    which makes the physical picture explicit: at ``φ = 0`` each σ_x
    commutes with the Hamiltonian, so the two-ion J_x axis is
    conserved and each J_x eigenstate drives a coherent displacement
    of the mode proportional to its eigenvalue (see
    :func:`iontrap_dynamics.analytic.ms_gate_phonon_number`).

    Scope (v0.1)
    ------------

    The builder supports **exactly two ions**, sharing a single drive
    configuration (a single laser addressing both). Arbitrary ``N``-ion
    MS generalises in the obvious way (symmetric sum across ions) and
    will be added once a three-ion benchmark is agreed upon. The two
    ions' individual ``η_k`` are computed independently from the mode
    eigenvector at each ion — COM and stretch modes are therefore
    handled correctly out of the box.

    The ``δ = 0`` assumption keeps the Hamiltonian time-independent.
    The gate-closing form (``δ ≠ 0``, ``t_gate = 2π/δ`` Bell-state
    target) needs QuTiP's list format and lands in a follow-on
    dispatch.

    Parameters
    ----------
    hilbert
        The full tensor-product Hilbert space.
    drive
        :class:`DriveConfig` for the bichromatic drive. Reads Rabi
        frequency (``carrier_rabi_frequency_rad_s``), phase, and
        wavevector. Detuning is not consulted (the δ = 0 assumption
        is structural, not a field value).
    mode_label
        Label of the motional mode that mediates the gate (typically
        COM or stretch). Must exist in ``hilbert.system.modes``.
    ion_indices
        Tuple of the two zero-based ion indices to entangle. Must be
        distinct and within ``[0, n_ions)``. Order does not matter —
        the Hamiltonian is symmetric under swap.

    Returns
    -------
    qutip.Qobj
        Time-independent Hermitian operator on the full Hilbert space,
        with dimensions :meth:`HilbertSpace.qutip_dims`. Ready to pass
        straight to :func:`iontrap_dynamics.sequences.solve`.

    Raises
    ------
    ConventionError
        If ``ion_indices`` contains a duplicate index, or if
        ``mode_label`` is not a mode of the system.
    IndexError
        If either ion index is outside ``[0, n_ions)``.

    Example
    -------

    Two ²⁵Mg⁺ ions sharing an axial COM mode, driven symmetrically
    with φ = 0::

        system = IonSystem(
            species_per_ion=(mg25_plus(), mg25_plus()),
            modes=(com_mode,),
        )
        hilbert = HilbertSpace(system=system, fock_truncations={"com": 15})
        drive = DriveConfig(
            k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
            carrier_rabi_frequency_rad_s=2 * np.pi * 0.1e6,
        )
        H = ms_gate_hamiltonian(hilbert, drive, "com", ion_indices=(0, 1))
    """
    i, j = ion_indices
    if i == j:
        raise ConventionError(
            f"ms_gate_hamiltonian requires two distinct ions; got ion_indices=({i}, {j})."
        )

    omega = drive.carrier_rabi_frequency_rad_s
    phi = drive.phase_rad
    phase_plus = cmath.exp(1j * phi)
    phase_minus = phase_plus.conjugate()

    a = hilbert.annihilation_for_mode(mode_label)
    a_dag = hilbert.creation_for_mode(mode_label)
    x_mode = a + a_dag

    def _single_ion_term(k: int) -> qutip.Qobj:
        eta_k = _sideband_lamb_dicke(hilbert, drive, mode_label, k)
        sigma_p = hilbert.spin_op_for_ion(sigma_plus_ion(), k)
        sigma_m = hilbert.spin_op_for_ion(sigma_minus_ion(), k)
        coeff = omega * eta_k / 2.0
        return coeff * (phase_plus * sigma_p + phase_minus * sigma_m) * x_mode

    return _single_ion_term(i) + _single_ion_term(j)


__all__ = [
    "blue_sideband_hamiltonian",
    "carrier_hamiltonian",
    "ms_gate_hamiltonian",
    "red_sideband_hamiltonian",
]
