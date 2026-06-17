# SPDX-License-Identifier: MIT
"""Contract / invariant tests for the trace-distance + BLP non-Markovianity API.

Phase A / WI-1 of the non-Markovianity task card. Exercises the metric
properties of :func:`trace_distance`, the subsystem partial-trace + validation
of :func:`trace_distance_trajectory`, and the positive-increment accumulation
plus §15 failure paths of :func:`blp_non_markovianity`.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from _helpers import _single_mode_hilbert, _spin_hilbert
from iontrap_dynamics.information import (
    blp_non_markovianity,
    trace_distance,
    trace_distance_trajectory,
)
from iontrap_dynamics.operators import spin_down, spin_up
from iontrap_dynamics.states import compose_density

ATOL = 1e-12


def _plus() -> qutip.Qobj:
    return (spin_up() + spin_down()).unit()


def _minus() -> qutip.Qobj:
    return (spin_up() - spin_down()).unit()


# --------------------------------------------------------------------------
# trace_distance — metric properties
# --------------------------------------------------------------------------


def test_identical_states_zero_distance() -> None:
    assert trace_distance(spin_up(), spin_up()) == pytest.approx(0.0, abs=ATOL)
    rho = _plus() * _plus().dag()
    assert trace_distance(rho, rho) == pytest.approx(0.0, abs=ATOL)


def test_orthogonal_pure_states_unit_distance() -> None:
    assert trace_distance(spin_up(), spin_down()) == pytest.approx(1.0, abs=ATOL)
    assert trace_distance(_plus(), _minus()) == pytest.approx(1.0, abs=ATOL)


def test_distance_is_symmetric() -> None:
    a, b = spin_up(), _plus()
    assert trace_distance(a, b) == pytest.approx(trace_distance(b, a), abs=ATOL)


def test_kets_promote_to_match_density_form() -> None:
    ket_form = trace_distance(spin_up(), _plus())
    dm_form = trace_distance(spin_up() * spin_up().dag(), _plus() * _plus().dag())
    assert ket_form == pytest.approx(dm_form, abs=ATOL)


def test_half_overlap_pair_is_one_over_sqrt_two() -> None:
    # |↑⟩ vs |+⟩ overlap ½ → D = √(1 − ½) = 1/√2 for pure states.
    assert trace_distance(spin_up(), _plus()) == pytest.approx(1.0 / np.sqrt(2.0), abs=ATOL)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shapes"):
        trace_distance(spin_up(), qutip.basis(3, 0))


# --------------------------------------------------------------------------
# trace_distance_trajectory — validation + partial trace
# --------------------------------------------------------------------------


def test_trajectory_basic_values() -> None:
    hilbert = _spin_hilbert(1)
    d = trace_distance_trajectory([spin_up(), spin_up()], [spin_up(), spin_down()], hilbert=hilbert)
    assert d.shape == (2,)
    assert d[0] == pytest.approx(0.0, abs=ATOL)
    assert d[1] == pytest.approx(1.0, abs=ATOL)


def test_trajectory_partial_trace_reduces_to_subsystem() -> None:
    hilbert = _single_mode_hilbert(3)  # spin ⊗ Fock(3)
    s0 = compose_density(
        hilbert, spin_states_per_ion=[spin_up()], mode_states_by_label={"b": qutip.basis(3, 0)}
    )
    s1 = compose_density(
        hilbert, spin_states_per_ion=[spin_up()], mode_states_by_label={"b": qutip.basis(3, 1)}
    )
    # Full states differ only in the mode → orthogonal → D = 1.
    assert trace_distance_trajectory([s0], [s1], hilbert=hilbert)[0] == pytest.approx(1.0, abs=1e-9)
    # Trace out the mode, keep the spin (index 0) → identical spin → D = 0.
    reduced = trace_distance_trajectory([s0], [s1], hilbert=hilbert, subsystem_indices=[0])
    assert reduced[0] == pytest.approx(0.0, abs=1e-9)


def test_trajectory_length_mismatch_raises() -> None:
    hilbert = _spin_hilbert(1)
    with pytest.raises(ValueError, match="length"):
        trace_distance_trajectory([spin_up()], [spin_up(), spin_down()], hilbert=hilbert)


def test_trajectory_empty_raises() -> None:
    hilbert = _spin_hilbert(1)
    with pytest.raises(ValueError, match="non-empty"):
        trace_distance_trajectory([], [], hilbert=hilbert)


def test_trajectory_dim_mismatch_raises() -> None:
    hilbert = _spin_hilbert(1)  # total_dim 2
    bad = qutip.basis(4, 0)
    with pytest.raises(ValueError, match="shape"):
        trace_distance_trajectory([bad], [bad], hilbert=hilbert)


def test_trajectory_bad_subsystem_index_raises() -> None:
    hilbert = _spin_hilbert(1)
    with pytest.raises(ValueError, match="out of range"):
        trace_distance_trajectory([spin_up()], [spin_down()], hilbert=hilbert, subsystem_indices=[5])


# --------------------------------------------------------------------------
# blp_non_markovianity — accumulation + §15 failure paths
# --------------------------------------------------------------------------


def test_blp_zero_for_monotone_decrease() -> None:
    assert blp_non_markovianity([1.0, 0.8, 0.5, 0.2, 0.0]) == pytest.approx(0.0, abs=ATOL)


def test_blp_sums_positive_increments() -> None:
    # Rises 0.2→0.6 (+0.4) and 0.3→0.9 (+0.6) ⇒ 𝒩 = 1.0; the falls do not count.
    assert blp_non_markovianity([0.5, 0.2, 0.6, 0.3, 0.9]) == pytest.approx(1.0, abs=ATOL)


def test_blp_short_input_is_zero() -> None:
    assert blp_non_markovianity([0.7]) == 0.0
    assert blp_non_markovianity([]) == 0.0


def test_blp_nonfinite_raises() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        blp_non_markovianity([0.1, np.nan, 0.3])


def test_blp_non_1d_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        blp_non_markovianity([[0.1, 0.2], [0.3, 0.4]])


def test_blp_rejects_distance_below_zero() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        blp_non_markovianity([0.5, -0.1, 0.3])


def test_blp_rejects_distance_above_one() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        blp_non_markovianity([0.5, 1.2, 0.3])


def test_blp_allows_exact_boundary_values() -> None:
    # exact 0 and 1 are valid trace distances (and round-off slack is tolerated)
    assert blp_non_markovianity([0.0, 1.0, 0.0]) == pytest.approx(1.0, abs=ATOL)
