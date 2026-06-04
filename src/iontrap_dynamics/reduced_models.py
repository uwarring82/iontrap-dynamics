# SPDX-License-Identifier: MIT
"""Reduced light–matter model Hamiltonians (CONVENTIONS.md §25).

Abstract qubit–oscillator models — Jaynes–Cummings (JC), anti-Jaynes–Cummings
(AJC), and the quantum Rabi model (QRM) — as **physics-layer** objects: what a
trapped-ion apparatus *approximates*, distinct from the §5 apparatus/drive
builders in :mod:`iontrap_dynamics.hamiltonians` that *realise* them.

Per §25 these are **Schrödinger-picture** Hamiltonians that intentionally carry
the **bare** atomic term ``½ω₀ σ_z`` — unlike the §5 interaction-picture
builders, whose free atomic term is transformed away (see the §5 scope note).
Each is a static, Hermitian :class:`qutip.Qobj` in ``H/ℏ`` units of rad·s⁻¹ on a
single-ion ⊗ single-mode embedding (§2 tensor order, §3 spin basis), with
σ_z = |↑⟩⟨↑| − |↓⟩⟨↓| and σ_+ = |↑⟩⟨↓| and â the mode annihilation operator:

* JC  : ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g(â σ_+ + â† σ_−)``  (co-rotating)
* AJC : ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g(â† σ_+ + â σ_−)``  (counter-rotating)
* QRM : ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g σ_x(â + â†)``       (full dipole, non-RWA)

``ω₀`` is an effective model splitting and may be negative (§25.2); ``ω_f`` is
the oscillator frequency (builder kwarg ``omega_f``); ``g`` the coupling. The
LOCK-3 identity ``H_AJC(ω₀) = σ_x H_JC(−ω₀) σ_x`` (§25.3) and the U(1)/Z₂
symmetry contrast are enforced by ``tests/conventions/test_reduced_models_conventions.py``.
"""

from __future__ import annotations

import math

import qutip

from .exceptions import ConventionError
from .hilbert import HilbertSpace
from .operators import sigma_minus_ion, sigma_plus_ion, sigma_x_ion, sigma_z_ion


def _validate_couplings(*, omega_0: float, omega_f: float, g: float) -> None:
    """Validate the model scalars before they reach a ``Qobj`` (§25).

    ``omega_0`` (an effective splitting, §25.2) and ``g`` may carry either sign but
    must be finite. ``omega_f`` is a genuine oscillator frequency — §25 grants the
    negative-sign semantics to ``omega_0`` alone — so it must be finite and
    positive, like the §10/§11 mode frequency it realises (cf. ``ModeConfig``).
    """
    for name, value in (("omega_0", omega_0), ("g", g)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value}.")
    if not math.isfinite(omega_f) or omega_f <= 0.0:
        raise ConventionError(
            f"omega_f must be a finite positive oscillator frequency (§25); got {omega_f}."
        )


def _bare_hamiltonian(
    hilbert: HilbertSpace,
    mode_label: str,
    ion_index: int,
    omega_0: float,
    omega_f: float,
) -> qutip.Qobj:
    """The bare term ``½ω₀ σ_z + ω_f â†â`` shared by JC/AJC/QRM (§25.1)."""
    sigma_z = hilbert.spin_op_for_ion(sigma_z_ion(), ion_index)
    number = hilbert.number_for_mode(mode_label)
    return 0.5 * omega_0 * sigma_z + omega_f * number


def jaynes_cummings_hamiltonian(
    hilbert: HilbertSpace,
    mode_label: str,
    *,
    ion_index: int,
    omega_0: float,
    omega_f: float,
    g: float,
) -> qutip.Qobj:
    """Return the Jaynes–Cummings Hamiltonian (co-rotating, RWA) — CONVENTIONS.md §25.

    ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g(â σ_+ + â† σ_−)``.

    The co-rotating coupling conserves the excitation number
    ``N̂ = â†â + |↑⟩⟨↑|`` (a U(1) symmetry): it couples ``|↑, n⟩`` to
    ``|↓, n+1⟩`` with matrix element ``g√(n+1)``, leaving ``|↓, 0⟩`` dark.

    Parameters
    ----------
    hilbert
        The full tensor-product Hilbert space (one ion ⊗ the selected mode).
    mode_label
        Label of the bosonic mode the qubit couples to.
    ion_index
        Zero-based index of the qubit ion. Keyword-only to keep the API
        unambiguous on multi-ion / multi-mode spaces.
    omega_0
        Effective qubit splitting ω₀ (rad·s⁻¹); may be negative (§25.2).
    omega_f
        Oscillator frequency ω_f (rad·s⁻¹); must be finite and positive.
    g
        Coupling strength g (rad·s⁻¹); may be negative.

    Returns
    -------
    qutip.Qobj
        Time-independent Hermitian operator on the full space, with dimensions
        :meth:`HilbertSpace.qutip_dims`.

    Raises
    ------
    ValueError
        If ``omega_0`` or ``g`` is non-finite.
    ConventionError
        If ``omega_f`` is not finite and positive, or ``mode_label`` is not a
        mode of ``hilbert``.
    IndexError
        If ``ion_index`` is outside ``[0, n_ions)``.

    See Also
    --------
    anti_jaynes_cummings_hamiltonian
    quantum_rabi_hamiltonian

    Example
    -------
    ::

        import numpy as np
        from iontrap_dynamics import jaynes_cummings_hamiltonian

        H = jaynes_cummings_hamiltonian(
            hilbert, "axial", ion_index=0,
            omega_0=2 * np.pi * 1e6, omega_f=2 * np.pi * 1e6, g=2 * np.pi * 1e3,
        )
    """
    _validate_couplings(omega_0=omega_0, omega_f=omega_f, g=g)
    sigma_p = hilbert.spin_op_for_ion(sigma_plus_ion(), ion_index)
    sigma_m = hilbert.spin_op_for_ion(sigma_minus_ion(), ion_index)
    a = hilbert.annihilation_for_mode(mode_label)
    a_dag = hilbert.creation_for_mode(mode_label)
    bare = _bare_hamiltonian(hilbert, mode_label, ion_index, omega_0, omega_f)
    return bare + g * (a * sigma_p + a_dag * sigma_m)


def anti_jaynes_cummings_hamiltonian(
    hilbert: HilbertSpace,
    mode_label: str,
    *,
    ion_index: int,
    omega_0: float,
    omega_f: float,
    g: float,
) -> qutip.Qobj:
    """Return the anti-Jaynes–Cummings Hamiltonian (counter-rotating) — §25.

    ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g(â† σ_+ + â σ_−)``.

    The counter-rotating coupling conserves the difference number
    ``Ĉ = â†â − |↑⟩⟨↑|``: it couples ``|↓, n⟩`` to ``|↑, n+1⟩`` (its dark state
    is ``|↑, 0⟩``). It is the σ_x conjugate of ``H_JC(−ω₀)`` — the LOCK-3
    identity ``H_AJC(ω₀) = σ_x H_JC(−ω₀) σ_x`` (§25.3).

    Parameters, returns, and raises match :func:`jaynes_cummings_hamiltonian`.
    """
    _validate_couplings(omega_0=omega_0, omega_f=omega_f, g=g)
    sigma_p = hilbert.spin_op_for_ion(sigma_plus_ion(), ion_index)
    sigma_m = hilbert.spin_op_for_ion(sigma_minus_ion(), ion_index)
    a = hilbert.annihilation_for_mode(mode_label)
    a_dag = hilbert.creation_for_mode(mode_label)
    bare = _bare_hamiltonian(hilbert, mode_label, ion_index, omega_0, omega_f)
    return bare + g * (a_dag * sigma_p + a * sigma_m)


def quantum_rabi_hamiltonian(
    hilbert: HilbertSpace,
    mode_label: str,
    *,
    ion_index: int,
    omega_0: float,
    omega_f: float,
    g: float,
) -> qutip.Qobj:
    """Return the quantum Rabi Hamiltonian (full dipole, non-RWA) — §25.

    ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g σ_x(â + â†)``.

    The full dipole coupling ``g σ_x(â + â†)`` is non-RWA (JC + AJC together);
    it conserves neither ``N̂`` nor ``Ĉ``, only the Z₂ parity
    ``P = exp(iπ N̂)``. Below ultra-strong coupling its low-lying spectrum tends
    to the Jaynes–Cummings one as ``g/ω₀ → 0`` (the RWA limit), reached as a
    genuine weak-coupling limit, never by dropping terms.

    Parameters, returns, and raises match :func:`jaynes_cummings_hamiltonian`.
    """
    _validate_couplings(omega_0=omega_0, omega_f=omega_f, g=g)
    sigma_x = hilbert.spin_op_for_ion(sigma_x_ion(), ion_index)
    a = hilbert.annihilation_for_mode(mode_label)
    a_dag = hilbert.creation_for_mode(mode_label)
    bare = _bare_hamiltonian(hilbert, mode_label, ion_index, omega_0, omega_f)
    return bare + g * sigma_x * (a + a_dag)


__all__ = [
    "anti_jaynes_cummings_hamiltonian",
    "jaynes_cummings_hamiltonian",
    "quantum_rabi_hamiltonian",
]
