# SPDX-License-Identifier: MIT
"""State-preparation (SPAM) errors for the systematics layer.

Models imperfect initial-state preparation — the "SP" side of SPAM.
Measurement-side infidelity ("AM") is already captured by
:class:`iontrap_dynamics.measurement.DetectorConfig` per §17.8. This
module covers the preparation side: optical-pumping leakage on the
spin and residual thermal motion after sideband cooling.

Two primitives ship in Dispatch U:

- :class:`SpinPreparationError` — probability ``p`` that the spin is
  pumped to ``|↑⟩`` instead of the intended ``|↓⟩``. Produces the
  classical-mixture spin state
  ``ρ_spin = (1 − p) |↓⟩⟨↓| + p |↑⟩⟨↑|``.
- :class:`ThermalPreparationError` — residual thermal phonon
  occupation ``n̄`` after cooling. Produces a Maxwell–Boltzmann-
  weighted mixture of Fock states,
  ``ρ_mode = qutip.thermal_dm(fock_dim, n̄)``.

Because these produce *density matrices* rather than kets, the
composed initial state is always a density matrix. Users feed the
per-subsystem factors to
:func:`iontrap_dynamics.states.compose_density`, which handles the
mixed ket/operator bookkeeping and returns the full-space ρ. The
solver then runs ``mesolve`` naturally on the density matrix — no
additional wiring needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import qutip

from ..operators import spin_down, spin_up


@dataclass(frozen=True, slots=True, kw_only=True)
class SpinPreparationError:
    """Imperfect spin preparation — probability of ending in ``|↑⟩``.

    Models incomplete optical pumping: intended ground state ``|↓⟩``
    is contaminated by a fraction ``p`` of shots that end up in
    ``|↑⟩``. Produces the classical-mixture density matrix
    ``(1 − p) |↓⟩⟨↓| + p |↑⟩⟨↑|``.

    Parameters
    ----------
    p_up_prep
        Probability of being prepared in ``|↑⟩`` instead of ``|↓⟩``.
        Must lie in ``[0, 1]``. ``0.0`` is a valid no-op — the
        density matrix collapses to pure ``|↓⟩⟨↓|``. Typical
        experimental values are ``p ≈ 10⁻⁴ – 10⁻²`` depending on
        cooling quality.
    label
        Identifier used by downstream aggregation code. Defaults to
        ``"spin_prep_error"``.

    Raises
    ------
    ValueError
        If ``p_up_prep`` lies outside ``[0, 1]``.
    """

    p_up_prep: float
    label: str = "spin_prep_error"

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_up_prep <= 1.0):
            raise ValueError(
                f"SpinPreparationError: p_up_prep must lie in [0, 1]; got {self.p_up_prep}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ThermalPreparationError:
    """Residual thermal motion after cooling — mean phonon number.

    Models sub-ideal sideband cooling: instead of the motional
    ground state ``|0⟩``, the ion sits in a thermal distribution
    with mean occupation ``n̄``. Produces the Fock-basis density
    matrix
    ``ρ_mn = (n̄ / (n̄ + 1))^n / (n̄ + 1)·δ_mn`` truncated at the
    requested Fock dimension.

    Parameters
    ----------
    n_bar_prep
        Mean phonon occupation after preparation. Non-negative.
        ``0.0`` is a valid no-op — the thermal distribution
        collapses to the ground state. Typical experimental values
        are ``n̄ ≈ 10⁻³ – 10⁻¹`` after resolved-sideband cooling.
    label
        Identifier used by downstream aggregation code. Defaults to
        ``"thermal_prep_error"``.

    Raises
    ------
    ValueError
        If ``n_bar_prep`` is negative.
    """

    n_bar_prep: float
    label: str = "thermal_prep_error"

    def __post_init__(self) -> None:
        if self.n_bar_prep < 0.0:
            raise ValueError(
                f"ThermalPreparationError: n_bar_prep must be >= 0; got {self.n_bar_prep}"
            )


def imperfect_spin_ground(error: SpinPreparationError) -> qutip.Qobj:
    """Return the 2 × 2 density matrix with a probability-``p_up_prep`` leak.

    The result is a single-spin density matrix
    ``(1 − p) |↓⟩⟨↓| + p |↑⟩⟨↑|`` suitable for
    :func:`iontrap_dynamics.states.compose_density`'s
    ``spin_states_per_ion`` entries. At ``p = 0`` it collapses to
    the pure ``|↓⟩⟨↓|`` density matrix — equivalent to
    ``qutip.ket2dm(spin_down())`` but explicitly named for readability.
    """
    p = error.p_up_prep
    dn = spin_down()
    up = spin_up()
    return (1.0 - p) * (dn * dn.dag()) + p * (up * up.dag())


def imperfect_motional_ground(
    error: ThermalPreparationError,
    *,
    fock_dim: int,
) -> qutip.Qobj:
    """Return a thermal density matrix on a ``fock_dim``-truncated mode.

    Delegates to :func:`qutip.thermal_dm`. At ``n̄ = 0`` the thermal
    distribution collapses to pure ``|0⟩⟨0|``.

    Parameters
    ----------
    error
        :class:`ThermalPreparationError` spec with ``n_bar_prep``.
    fock_dim
        Fock truncation for the mode (the ``N`` in ``ρ ∈ ℂ^{N×N}``).
        Must be ``>= 1``.

    Raises
    ------
    ValueError
        If ``fock_dim < 1`` or if ``n̄_prep ≥ fock_dim − 1``. The
        upper bound is a solver-quality check: a thermal
        distribution with ``n̄`` close to the Fock cutoff is poorly
        approximated by the truncated space and should raise a
        ``FockConvergenceWarning`` downstream.
    """
    if fock_dim < 1:
        raise ValueError(f"imperfect_motional_ground: fock_dim must be >= 1; got {fock_dim}")
    if error.n_bar_prep >= fock_dim - 1:
        raise ValueError(
            "imperfect_motional_ground: n_bar_prep "
            f"({error.n_bar_prep}) >= fock_dim − 1 ({fock_dim - 1}); "
            "the thermal distribution extends past the Fock truncation. "
            "Either increase fock_dim (recommended: >= 4·n_bar_prep + 4) "
            "or reduce n_bar_prep."
        )
    return qutip.thermal_dm(fock_dim, error.n_bar_prep)


__all__ = [
    "SpinPreparationError",
    "ThermalPreparationError",
    "imperfect_motional_ground",
    "imperfect_spin_ground",
]
