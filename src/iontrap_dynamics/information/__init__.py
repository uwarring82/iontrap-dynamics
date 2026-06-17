# SPDX-License-Identifier: MIT
"""Information-theoretic primitives: estimation and quantum Darwinism.

Application-agnostic umbrella sub-package (WP-01, ratified naming root):
``fisher`` (WI-1 — estimation: CFI / QFI / Cramér–Rao), and ``redundancy`` +
``recoverability`` (WI-2 — quantum Darwinism), sharing the nonlinear-in-ρ helper
layer in ``_common`` (``_ensure_density``, ``_von_neumann_entropy_bits``,
``_validate_indices`` / ``_validate_state_dim``).
"""

from .distinguishability import (
    blp_non_markovianity,
    trace_distance,
    trace_distance_trajectory,
)
from .fisher import (
    classical_fisher_information,
    cramer_rao_bound,
    linear_gaussian_fisher,
    quantum_fisher_information_trajectory,
)
from .qpn_bias import QpnBiasResult, non_markovianity_qpn_bias
from .recoverability import recoverability
from .redundancy import (
    fragment_mutual_information,
    partial_information_plot,
    redundancy,
)

__all__ = [
    "QpnBiasResult",
    "blp_non_markovianity",
    "classical_fisher_information",
    "cramer_rao_bound",
    "fragment_mutual_information",
    "linear_gaussian_fisher",
    "non_markovianity_qpn_bias",
    "partial_information_plot",
    "quantum_fisher_information_trajectory",
    "recoverability",
    "redundancy",
    "trace_distance",
    "trace_distance_trajectory",
]
