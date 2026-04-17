# SPDX-License-Identifier: MIT
"""Canonical exception hierarchy for :mod:`iontrap_dynamics`.

The hierarchy is intentionally shallow and stable. It matches the Level 3
hard-failure contract in ``CONVENTIONS.md`` section 15.
"""


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
