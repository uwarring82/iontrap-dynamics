# SPDX-License-Identifier: MIT
"""Information-theoretic primitives: estimation and (later) quantum Darwinism.

Application-agnostic umbrella sub-package (WP-01, ratified naming root). WI-1
lands the estimation module ``fisher``; WI-2 will add ``redundancy`` and
``recoverability`` under the same umbrella, sharing the nonlinear-in-rho helper
layer.
"""

from .fisher import (
    classical_fisher_information,
    cramer_rao_bound,
    linear_gaussian_fisher,
    quantum_fisher_information_trajectory,
)
from .recoverability import recoverability
from .redundancy import (
    fragment_mutual_information,
    partial_information_plot,
    redundancy,
)

__all__ = [
    "classical_fisher_information",
    "cramer_rao_bound",
    "fragment_mutual_information",
    "linear_gaussian_fisher",
    "partial_information_plot",
    "quantum_fisher_information_trajectory",
    "recoverability",
    "redundancy",
]
