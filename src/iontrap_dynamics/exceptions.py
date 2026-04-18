# SPDX-License-Identifier: MIT
"""Canonical exception and warning hierarchies for :mod:`iontrap_dynamics`.

Both hierarchies are intentionally shallow and stable, matching the
three-level ladder in ``CONVENTIONS.md`` §15:

- Level 3 (hard failure)      → typed exception below, raised
- Level 2 (quality degraded)  → :class:`IonTrapWarning` subclass below,
                                emitted to Python's ``warnings`` channel
                                **and** recorded on
                                :attr:`TrajectoryResult.warnings`
- Level 1 (convergence)       → same channel as Level 2, distinguished
                                by the specific warning subclass
- OK                          → silent
"""


# ----------------------------------------------------------------------------
# Exceptions (Level 3 hard failures)
# ----------------------------------------------------------------------------


class IonTrapError(Exception):
    """Base class for package-defined failures.

    This inherits directly from ``Exception`` rather than ``RuntimeError``
    because the family includes convention, integrity, and validation failures
    in addition to backend runtime failures.
    """


class ConventionError(IonTrapError):
    """Raised when code or input violates published project conventions."""


class BackendError(IonTrapError):
    """Raised for backend-internal failures or unsupported backend requests."""


class IntegrityError(IonTrapError):
    """Raised when cached data or computed results fail integrity checks."""


class ConvergenceError(IonTrapError):
    """Raised when solver convergence fails above the allowed tolerance."""


# ----------------------------------------------------------------------------
# Warnings (Level 1 and Level 2 soft diagnostics)
# ----------------------------------------------------------------------------
#
# These are *Warning* subclasses, not *Exception* subclasses — they travel
# through the ``warnings`` module, never through raise/except. They pair
# with :class:`~iontrap_dynamics.results.ResultWarning` records attached to
# :attr:`TrajectoryResult.warnings`; a well-behaved solver emits both for
# every anomaly it detects.


class IonTrapWarning(UserWarning):
    """Base class for package-defined warnings.

    Subclasses from :class:`UserWarning` so they survive Python's default
    filterset (``default``) and reach the user without explicit filter
    configuration. Downstream code may filter by this base class to
    silence all library-emitted warnings uniformly.
    """


class FockConvergenceWarning(IonTrapWarning):
    """Level 1 (CONVENTIONS.md §15): Fock-truncation slow-convergence.

    Emitted when a mode's top-Fock population lies in
    ``[ε/10, ε)`` — the solver converged but the truncation is close to
    its envelope. Results are returned and trusted for coarse analysis;
    tighten ``fock_truncations`` for publication-grade figures.
    """


class FockQualityWarning(IonTrapWarning):
    """Level 2 (CONVENTIONS.md §15): Fock-truncation quality degradation.

    Emitted when a mode's top-Fock population lies in ``[ε, 10·ε)`` —
    the truncation is demonstrably under-resolved. Results are still
    returned, but ``TrajectoryResult.warnings`` must be consulted before
    any publication use. Ignoring the warning and publishing a figure
    anyway is a convention violation.
    """
