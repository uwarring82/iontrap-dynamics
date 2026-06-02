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

**Sequence-aware (time-windowed) application (WI-3b).** Each channel accepts an
optional ``window = (t0, t1)``: the dissipation is active only on the half-open
interval ``[t0, t1)`` (SI seconds, on the solver's time axis) and off elsewhere.
This is realised with QuTiP's time-dependent collapse-operator format
``[L, coeff]``, where ``coeff`` is a callable obeying the QuTiP ``QobjEvo``
contract **``f(t, args)``** — a two-argument signature; ``args`` is unused here.
No custom solver path is introduced: :func:`solve` passes the mixed
constant / time-dependent list straight to ``mesolve``. When any channel is
windowed, ``solve`` caps the integrator's ``max_step`` at the output-grid
resolution (``min(diff(times))``) so the adaptive stepper cannot take one giant
step across a quiescent interval and **skip a window's turn-on** — the only place
the windowed path differs from the constant-channel path. Because the windows are
half-open, adjacent windows ``[a, b)`` and ``[b, c)`` partition cleanly, and
**ordered or overlapping windows make channel composition order-dependent** — the
library does not assume the dissipators commute (the R8 boundary; see the WP-02
R8 regression test). ``window = None`` (default) keeps the WI-3a constant-rate
behaviour byte-for-byte.

Application-agnostic boundary (``WP/WP-02-two-mode-motional.md`` §1): this is the
general typed dissipator *family*. Which channel models a particular physical
process, and any composition (``ε_total``) logic, belong to the consuming
programme, not here.

``Depolarising`` is intentionally **not** provided: a depolarising channel is
canonical for finite-dimensional systems, not for a single bosonic mode, so it is
deferred rather than given an arbitrary truncated-Fock definition
(``WP/WP-02-two-mode-motional.md`` §5; WP-02 logbook).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import qutip

from .hilbert import HilbertSpace

# A collapse-operator spec for the QuTiP master equation: either a constant
# operator (a Qobj) or QuTiP's time-dependent ``[operator, coefficient]`` pair.
# Mirrors the ``MesolveHamiltonian`` alias in ``sequences.py``.
CollapseOperator: TypeAlias = "qutip.Qobj | list[object]"


def _validate_nonnegative(value: float, name: str) -> None:
    """A rate / occupation must be a finite, non-negative number."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value}")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value}")


def _validate_window(window: tuple[float, float]) -> None:
    """A window ``(t0, t1)`` must be finite with ``0 ≤ t0 < t1``."""
    t0, t1 = window
    if not (math.isfinite(t0) and math.isfinite(t1)):
        raise ValueError(f"window endpoints must be finite; got {window}")
    if t0 < 0.0:
        raise ValueError(f"window start must be non-negative; got {t0}")
    if not t1 > t0:
        raise ValueError(f"window must satisfy t1 > t0; got {window}")


def _window_coefficient(t0: float, t1: float) -> Callable[[float, dict[str, object]], float]:
    """A QuTiP time-dependent coefficient — the ``QobjEvo`` ``f(t, args)`` contract.

    Returns ``1.0`` on the half-open window ``[t0, t1)`` and ``0.0`` outside, so
    the Lindblad rate (∝ |coeff|²) is fully on inside the window and off outside.
    The second parameter **must** be named ``args`` (unused here): QuTiP keys its
    ``(t)`` vs ``(t, args)`` calling convention on that exact name — renaming it
    makes QuTiP call the coefficient with one argument and raise.
    """

    def _coeff(t: float, args: dict[str, object]) -> float:
        return 1.0 if t0 <= t < t1 else 0.0

    return _coeff


def _apply_window(
    operators: list[qutip.Qobj], window: tuple[float, float] | None
) -> list[CollapseOperator]:
    """Return ``operators`` unchanged, or wrapped as ``[op, coeff]`` if windowed."""
    if window is None:
        return list(operators)
    coeff = _window_coefficient(*window)
    return [[op, coeff] for op in operators]


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
    window
        Optional ``(t0, t1)`` activity window in SI seconds (half-open
        ``[t0, t1)``). ``None`` (default) → active over the whole solve.

    Raises
    ------
    ValueError
        On a negative or non-finite ``rate``, or a malformed ``window``.
    """

    mode: str
    rate: float
    window: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative(self.rate, "AmplitudeDamping.rate")
        if self.window is not None:
            _validate_window(self.window)

    def collapse_operators(self, hilbert: HilbertSpace) -> list[CollapseOperator]:
        """Return ``√rate · â_mode`` embedded on ``hilbert`` (``[]`` if rate is 0)."""
        if self.rate == 0.0:
            return []
        base = [math.sqrt(self.rate) * hilbert.annihilation_for_mode(self.mode)]
        return _apply_window(base, self.window)


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
    window
        Optional ``(t0, t1)`` activity window in SI seconds (half-open). ``None``
        (default) → active over the whole solve.

    Raises
    ------
    ValueError
        On a negative or non-finite ``rate`` or ``n_bar_bath``, or a malformed
        ``window``.
    """

    mode: str
    rate: float
    n_bar_bath: float
    window: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative(self.rate, "Heating.rate")
        _validate_nonnegative(self.n_bar_bath, "Heating.n_bar_bath")
        if self.window is not None:
            _validate_window(self.window)

    def collapse_operators(self, hilbert: HilbertSpace) -> list[CollapseOperator]:
        """Return the down/up collapse operators embedded on ``hilbert``.

        ``√(rate·(n̄+1))·â`` plus ``√(rate·n̄)·â†`` when ``n̄ > 0`` (``[]`` if
        ``rate`` is 0), each time-windowed when ``window`` is set.
        """
        if self.rate == 0.0:
            return []
        base = [
            math.sqrt(self.rate * (self.n_bar_bath + 1.0))
            * hilbert.annihilation_for_mode(self.mode)
        ]
        if self.n_bar_bath > 0.0:
            base.append(
                math.sqrt(self.rate * self.n_bar_bath) * hilbert.creation_for_mode(self.mode)
            )
        return _apply_window(base, self.window)


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
    window
        Optional ``(t0, t1)`` activity window in SI seconds (half-open). ``None``
        (default) → active over the whole solve.

    Raises
    ------
    ValueError
        On a negative or non-finite ``rate``, or a malformed ``window``.
    """

    mode: str
    rate: float
    window: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative(self.rate, "Dephasing.rate")
        if self.window is not None:
            _validate_window(self.window)

    def collapse_operators(self, hilbert: HilbertSpace) -> list[CollapseOperator]:
        """Return ``√rate · n̂_mode`` embedded on ``hilbert`` (``[]`` if rate is 0)."""
        if self.rate == 0.0:
            return []
        base = [math.sqrt(self.rate) * hilbert.number_for_mode(self.mode)]
        return _apply_window(base, self.window)


MotionalChannel = AmplitudeDamping | Heating | Dephasing
"""Union of the typed motional CPTP channels."""


def build_collapse_operators(
    channels: Sequence[MotionalChannel], hilbert: HilbertSpace
) -> list[CollapseOperator]:
    """Concatenate the collapse operators of ``channels`` on ``hilbert``.

    Each channel embeds its operators on its named mode via the HilbertSpace
    mode-operator API (an unknown mode label raises ``ConventionError``), and
    time-windows them when it carries a ``window``. Returns ``[]`` for an empty
    sequence or all-zero-rate channels — the no-dissipation case, which leaves
    :func:`iontrap_dynamics.sequences.solve` on its default solver path. The
    returned list mixes constant ``Qobj`` and time-dependent ``[op, coeff]``
    entries; ``mesolve`` consumes both.
    """
    c_ops: list[CollapseOperator] = []
    for channel in channels:
        c_ops.extend(channel.collapse_operators(hilbert))
    return c_ops


__all__ = [
    "AmplitudeDamping",
    "CollapseOperator",
    "Dephasing",
    "Heating",
    "MotionalChannel",
    "build_collapse_operators",
]
