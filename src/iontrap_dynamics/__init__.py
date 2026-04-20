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
from .measurement import (
    BernoulliChannel,
    BinomialChannel,
    BinomialSummary,
    DetectorConfig,
    ParityScan,
    PoissonChannel,
    SidebandInference,
    SpinReadout,
    binomial_summary,
    clopper_pearson_interval,
    sample_outcome,
    wilson_interval,
)
from .results import (
    MeasurementResult,
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
    "BernoulliChannel",
    "BinomialChannel",
    "BinomialSummary",
    "ConventionError",
    "ConvergenceError",
    "DetectorConfig",
    "FockConvergenceWarning",
    "FockQualityWarning",
    "IntegrityError",
    "IonTrapError",
    "IonTrapWarning",
    "MeasurementResult",
    "ParityScan",
    "PoissonChannel",
    "Result",
    "ResultMetadata",
    "ResultWarning",
    "SidebandInference",
    "SpinReadout",
    "StorageMode",
    "TrajectoryResult",
    "WarningSeverity",
    "binomial_summary",
    "clopper_pearson_interval",
    "compute_request_hash",
    "load_trajectory",
    "sample_outcome",
    "save_trajectory",
    "wilson_interval",
]
