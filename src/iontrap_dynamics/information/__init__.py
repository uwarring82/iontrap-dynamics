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

__all__ = [
    "classical_fisher_information",
    "cramer_rao_bound",
    "linear_gaussian_fisher",
    "quantum_fisher_information_trajectory",
]
