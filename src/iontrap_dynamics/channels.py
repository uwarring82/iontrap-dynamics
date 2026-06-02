# SPDX-License-Identifier: MIT
"""Typed motional CPTP channels for the Lindblad (master-equation) path.

WP-02 WI-3 (dispatch MCC). Application-agnostic: each channel is a typed
dissipator on a labelled motional mode, producing the QuTiP collapse operators
the master equation consumes. They are routed into
:func:`iontrap_dynamics.sequences.solve` via its ``channels=`` argument, which
forces the ``mesolve`` (master-equation) path; with no channels the solver is
unchanged.

The channel parameterisation — Lindblad rates in s⁻¹, bath occupation n̄, and the
collapse-operator map below — is the convention staged for ``CONVENTIONS.md §24``
(the shared v0.3 Convention Freeze; see ``WP/FREEZE-v0.3.md`` and
``WP/WP-02-two-mode-motional.md`` §6). Each channel contributes Lindblad
dissipators ``D[L]ρ = LρL† − ½{L†L, ρ}`` with the collapse operators ``L``:

- :class:`AmplitudeDamping` — zero-temperature decay, ``L = √κ · â``.
- :class:`Heating` — finite-bath amplitude damping, ``L₋ = √(κ(n̄+1)) · â`` and
  ``L₊ = √(κ n̄) · â†`` (the anomalous-heating model; steady-state ``⟨n̂⟩ → n̄``).
- :class:`Dephasing` — pure dephasing, ``L = √γ_φ · n̂``.

All operators are embedded into the full tensor space via the
:class:`~iontrap_dynamics.hilbert.HilbertSpace` mode-operator API
(``annihilation_for_mode`` / ``creation_for_mode`` / ``number_for_mode``), so a
channel names its mode by label and is agnostic to the tensor ordering. An
unknown mode label raises :class:`~iontrap_dynamics.exceptions.ConventionError`
from that API.

Application-agnostic boundary (``WP/WP-02-two-mode-motional.md`` §1): this is the
general typed dissipator *family*. Which channel models a particular physical
process, and any composition (``ε_total``) logic, belong to the consuming
programme, not here.

``Depolarising`` is intentionally **not** provided in WI-3a: a depolarising
channel is canonical for finite-dimensional systems, not for a single bosonic
mode, so it is deferred rather than given an arbitrary truncated-Fock definition
(``WP/WP-02-two-mode-motional.md`` §5; WP-02 logbook). The three Lindblad
dissipators above cover the native heating and dephasing models the surface
needs. Time-windowed (sequence-aware) application and the non-commuting-order
test are WI-3b.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import qutip

from .hilbert import HilbertSpace


def _validate_nonnegative(value: float, name: str) -> None:
    """A rate / occupation must be a finite, non-negative number."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value}")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value}")


@dataclass(frozen=True, slots=True, kw_only=True)
class AmplitudeDamping:
    """Zero-temperature amplitude damping on a motional ``mode``.

    Collapse operator ``L = √rate · â`` (CONVENTIONS.md §24). Drives the mode to
    the ground state: ``⟨n̂(t)⟩ = ⟨n̂(0)⟩ · e^{−rate·t}``.

    Parameters
    ----------
    mode
        Mode label; must be a mode of the :class:`HilbertSpace` it is built on.
    rate
        Damping rate κ in s⁻¹. Non-negative; ``0.0`` is a no-op.

    Raises
    ------
    ValueError
        On a negative or non-finite ``rate``.
    """

    mode: str
    rate: float

    def __post_init__(self) -> None:
        _validate_nonnegative(self.rate, "AmplitudeDamping.rate")

    def collapse_operators(self, hilbert: HilbertSpace) -> list[qutip.Qobj]:
        """Return ``[√rate · â_mode]`` embedded on ``hilbert`` (``[]`` if rate is 0)."""
        if self.rate == 0.0:
            return []
        return [math.sqrt(self.rate) * hilbert.annihilation_for_mode(self.mode)]


@dataclass(frozen=True, slots=True, kw_only=True)
class Heating:
    """Finite-temperature amplitude damping (heating) on a motional ``mode``.

    Collapse operators ``L₋ = √(rate·(n̄+1)) · â`` and ``L₊ = √(rate·n̄) · â†``
    (CONVENTIONS.md §24). The mode relaxes to the bath: ``⟨n̂⟩ → n_bar_bath``.
    This is the anomalous-heating model.

    Parameters
    ----------
    mode
        Mode label.
    rate
        Coupling rate κ in s⁻¹ to the bath. Non-negative.
    n_bar_bath
        Bath mean occupation n̄ ≥ 0; the steady-state ``⟨n̂⟩``.

    Raises
    ------
    ValueError
        On a negative or non-finite ``rate`` or ``n_bar_bath``.
    """

    mode: str
    rate: float
    n_bar_bath: float

    def __post_init__(self) -> None:
        _validate_nonnegative(self.rate, "Heating.rate")
        _validate_nonnegative(self.n_bar_bath, "Heating.n_bar_bath")

    def collapse_operators(self, hilbert: HilbertSpace) -> list[qutip.Qobj]:
        """Return the down/up collapse operators embedded on ``hilbert``.

        ``[√(rate·(n̄+1))·â]`` plus ``√(rate·n̄)·â†`` when ``n̄ > 0`` (``[]`` if
        ``rate`` is 0).
        """
        if self.rate == 0.0:
            return []
        ops = [
            math.sqrt(self.rate * (self.n_bar_bath + 1.0))
            * hilbert.annihilation_for_mode(self.mode)
        ]
        if self.n_bar_bath > 0.0:
            ops.append(
                math.sqrt(self.rate * self.n_bar_bath) * hilbert.creation_for_mode(self.mode)
            )
        return ops


@dataclass(frozen=True, slots=True, kw_only=True)
class Dephasing:
    """Pure dephasing on a motional ``mode``.

    Collapse operator ``L = √rate · n̂`` (CONVENTIONS.md §24). Decoheres
    off-diagonal Fock coherences while leaving ``⟨n̂⟩`` unchanged.

    Parameters
    ----------
    mode
        Mode label.
    rate
        Dephasing rate γ_φ in s⁻¹. Non-negative.

    Raises
    ------
    ValueError
        On a negative or non-finite ``rate``.
    """

    mode: str
    rate: float

    def __post_init__(self) -> None:
        _validate_nonnegative(self.rate, "Dephasing.rate")

    def collapse_operators(self, hilbert: HilbertSpace) -> list[qutip.Qobj]:
        """Return ``[√rate · n̂_mode]`` embedded on ``hilbert`` (``[]`` if rate is 0)."""
        if self.rate == 0.0:
            return []
        return [math.sqrt(self.rate) * hilbert.number_for_mode(self.mode)]


MotionalChannel = AmplitudeDamping | Heating | Dephasing
"""Union of the typed motional CPTP channels delivered by WI-3a."""


def build_collapse_operators(
    channels: Sequence[MotionalChannel], hilbert: HilbertSpace
) -> list[qutip.Qobj]:
    """Concatenate the collapse operators of ``channels`` on ``hilbert``.

    Each channel embeds its operators on its named mode via the HilbertSpace
    mode-operator API (an unknown mode label raises ``ConventionError``).
    Returns ``[]`` for an empty sequence or all-zero-rate channels — the
    no-dissipation case, which leaves :func:`iontrap_dynamics.sequences.solve`
    on its default solver path.
    """
    c_ops: list[qutip.Qobj] = []
    for channel in channels:
        c_ops.extend(channel.collapse_operators(hilbert))
    return c_ops


__all__ = [
    "AmplitudeDamping",
    "Dephasing",
    "Heating",
    "MotionalChannel",
    "build_collapse_operators",
]
