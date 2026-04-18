# SPDX-License-Identifier: MIT
"""Public package interface for ``iontrap_dynamics``."""

from .cache import (
    CACHE_FORMAT_VERSION,
    compute_request_hash,
    load_trajectory,
    save_trajectory,
)
from .conventions import CONVENTION_VERSION, FOCK_CONVERGENCE_TOLERANCE
from .exceptions import (
    BackendError,
    ConventionError,
    ConvergenceError,
    FockConvergenceWarning,
    FockQualityWarning,
    IntegrityError,
    IonTrapError,
    IonTrapWarning,
)
from .results import (
    Result,
    ResultMetadata,
    ResultWarning,
    StorageMode,
    TrajectoryResult,
    WarningSeverity,
)

__all__ = [
    "CACHE_FORMAT_VERSION",
    "CONVENTION_VERSION",
    "FOCK_CONVERGENCE_TOLERANCE",
    "BackendError",
    "ConventionError",
    "ConvergenceError",
    "FockConvergenceWarning",
    "FockQualityWarning",
    "IntegrityError",
    "IonTrapError",
    "IonTrapWarning",
    "Result",
    "ResultMetadata",
    "ResultWarning",
    "StorageMode",
    "TrajectoryResult",
    "WarningSeverity",
    "compute_request_hash",
    "load_trajectory",
    "save_trajectory",
]
