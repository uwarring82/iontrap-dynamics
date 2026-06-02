# SPDX-License-Identifier: MIT
"""Unit tests for the two-mode SU(1,1) surface (WP-02 WI-1/WI-2, MCA/MCB).

Structural checks on ``two_mode_squeezing_hamiltonian`` / ``beamsplitter_hamiltonian``
(`hamiltonians.py`) and the ``two_mode_squeezed_vacuum`` factory (`states.py`):
Hermiticity, the conserved-charge commutators (su(1,1) difference number, SU(2)
total number), label-based embedding, and input validation. The closed-form
occupation oracles live in
``tests/regression/analytic/test_two_mode_squeezing.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from _helpers import _two_mode_hilbert
from iontrap_dynamics.exceptions import ConventionError
from iontrap_dynamics.hamiltonians import (
    beamsplitter_hamiltonian,
    two_mode_squeezing_hamiltonian,
)
from iontrap_dynamics.states import two_mode_squeezed_vacuum

FOCK_DIM = 8


def _commutator_norm(op_a: qutip.Qobj, op_b: qutip.Qobj) -> float:
    return float((op_a * op_b - op_b * op_a).norm())


# --------------------------------------------------------------------------
# Hamiltonian builders
# --------------------------------------------------------------------------


def test_two_mode_squeezing_hamiltonian_is_hermitian():
    hilbert = _two_mode_hilbert(FOCK_DIM)
    h = two_mode_squeezing_hamiltonian(hilbert, g=3000.0, phase=0.7)
    assert h.isherm
    assert h.dims == hilbert.qutip_dims()


def test_two_mode_squeezing_conserves_difference_number():
    # su(1,1): H_TMS commutes with n_a - n_b (the conserved Casimir label).
    hilbert = _two_mode_hilbert(FOCK_DIM)
    h = two_mode_squeezing_hamiltonian(hilbert, g=3000.0, phase=0.3)
    n_diff = hilbert.number_for_mode("a") - hilbert.number_for_mode("b")
    assert _commutator_norm(h, n_diff) < 1e-9


def test_beamsplitter_is_hermitian_and_conserves_total_number():
    # SU(2): H_BS commutes with n_a + n_b (excitation only rotates between modes).
    hilbert = _two_mode_hilbert(FOCK_DIM)
    h = beamsplitter_hamiltonian(hilbert, coupling=2000.0, phase=0.4)
    assert h.isherm
    n_total = hilbert.number_for_mode("a") + hilbert.number_for_mode("b")
    assert _commutator_norm(h, n_total) < 1e-9


def test_squeezing_does_not_conserve_total_number():
    # Sanity: TMS *creates* pairs, so it must NOT commute with n_a + n_b.
    hilbert = _two_mode_hilbert(FOCK_DIM)
    h = two_mode_squeezing_hamiltonian(hilbert, g=3000.0)
    n_total = hilbert.number_for_mode("a") + hilbert.number_for_mode("b")
    assert _commutator_norm(h, n_total) > 1e-3


def test_unknown_mode_label_raises():
    hilbert = _two_mode_hilbert(FOCK_DIM)
    with pytest.raises(ConventionError, match="unknown mode"):
        two_mode_squeezing_hamiltonian(hilbert, g=1.0, mode_labels=("a", "nope"))


def test_label_order_swaps_the_pair_symmetrically():
    # â†b̂† is symmetric under a<->b, so swapping the labels gives the same H.
    hilbert = _two_mode_hilbert(FOCK_DIM)
    h_ab = two_mode_squeezing_hamiltonian(hilbert, g=1500.0, mode_labels=("a", "b"))
    h_ba = two_mode_squeezing_hamiltonian(hilbert, g=1500.0, mode_labels=("b", "a"))
    assert (h_ab - h_ba).norm() < 1e-9


# --------------------------------------------------------------------------
# Two-mode squeezed-vacuum factory
# --------------------------------------------------------------------------


def test_factory_is_normalised_two_mode_ket():
    psi = two_mode_squeezed_vacuum(FOCK_DIM, 0.4)
    assert psi.norm() == pytest.approx(1.0, abs=1e-9)
    # qutip 5 reports a two-mode ket as [[N_a, N_b], [1]] (trivial column flattened).
    assert psi.dims == [[FOCK_DIM, FOCK_DIM], [1]]
    assert psi.isket


def test_factory_accepts_asymmetric_dims():
    psi = two_mode_squeezed_vacuum((6, 10), 0.3)
    assert psi.dims == [[6, 10], [1]]


def test_factory_zero_squeeze_is_two_mode_vacuum():
    psi = two_mode_squeezed_vacuum(FOCK_DIM, 0.0)
    vacuum = qutip.tensor(qutip.basis(FOCK_DIM, 0), qutip.basis(FOCK_DIM, 0))
    assert abs(abs(psi.overlap(vacuum)) - 1.0) < 1e-12


@pytest.mark.parametrize("bad_dims", [0, -3, (4, 0), (-1, 5)])
def test_factory_nonpositive_dim_raises(bad_dims):
    with pytest.raises(ConventionError, match="positive"):
        two_mode_squeezed_vacuum(bad_dims, 0.3)


@pytest.mark.parametrize("bad_z", [complex(float("nan"), 0.0), complex(0.0, float("inf"))])
def test_factory_nonfinite_z_raises(bad_z):
    with pytest.raises(ConventionError, match="finite"):
        two_mode_squeezed_vacuum(FOCK_DIM, bad_z)


def test_factory_populates_only_equal_occupation_components():
    # Schmidt form Σ cₙ |n, n⟩: zero population off the n_a = n_b diagonal.
    dim = 10
    psi = two_mode_squeezed_vacuum(dim, 0.5)
    probs = np.abs(psi.full().reshape(dim, dim)) ** 2
    off_diagonal = probs.sum() - np.trace(probs)
    assert off_diagonal < 1e-9
