# SPDX-License-Identifier: MIT
"""Convention-enforcement tests for the reduced light–matter models (§25).

WP-03 / Dispatch RLA. These tests anchor the convention *staged* for
``CONVENTIONS.md`` §25 (reduced JC / AJC / QRM Hamiltonians) and the
companion §5 scope note — both proposed in ``WP/RL-conventions-proposal.md``
and sealed by the maintainer under a ``CONVENTION_VERSION`` 0.3 → 0.4 bump
*before* the WI-2 (Dispatch RLB) builders land. The reference Hamiltonians
are built here inline from the canonical library operators; the future
``reduced_models.py`` builders must reproduce these exact forms — *including
the absolute coefficients*, which the magnitude anchors below pin.

Convention under test (§25), on the §3 spin basis
(σ_z = |↑⟩⟨↑| − |↓⟩⟨↓|, σ_+ = |↑⟩⟨↓|) and §2 tensor order (spin ⊗ mode),
all in ``H/ℏ`` (rad·s⁻¹), Schrödinger-picture *bare* terms (NOT the §5
interaction picture — see the §5 scope note). ``ω_f`` is the oscillator/field
frequency (builder kwarg ``omega_f``; the physical motional mode ω_m of §10/§11
in an apparatus realisation):

    H_JC(ω0)  = ½ ω0 σ_z + ω_f a†a + g (a σ_+ + a† σ_−)      co-rotating
    H_AJC(ω0) = ½ ω0 σ_z + ω_f a†a + g (a† σ_+ + a σ_−)      counter-rotating
    H_QRM(ω0) = ½ ω0 σ_z + ω_f a†a + g σ_x (a + a†)          full dipole (non-RWA)

Three binding properties:

* **LOCK-3 identity** ``H_AJC(ω0) = σ_x H_JC(−ω0) σ_x``. Holds exactly under
  §3 (σ_x σ_z σ_x = −σ_z, σ_x σ_± σ_x = σ_∓; σ_x is the identity on the
  motional factor). The negative argument is *essential* for ω0 ≠ 0: the σ_x
  conjugation flips the σ_z sign, so the input frequency must be pre-flipped
  to recover the physical +½ ω0 σ_z AJC term. The "−ω0" is an *effective
  model sign* (e.g. red-sideband selection maps to an effective −ω0 frame),
  NOT a negative physical ion splitting and NOT the §4 drive detuning
  δ = ω_laser − ω_atom.

* **Symmetry contrast** (§25.1): JC conserves the excitation number
  N̂ = a†a + |↑⟩⟨↑| (= a†a + σ_+σ_−, a U(1) symmetry); AJC conserves the
  difference number Ĉ = a†a − |↑⟩⟨↑|; the non-RWA QRM conserves neither, only
  the Z₂ parity P = exp(iπ N̂) = −σ_z(−1)^{a†a} (the card's Π = σ_z(−1)^{a†a}
  up to an immaterial global sign — same ±1 eigenspaces).

* **Absolute coefficients** (§25.1): ½ on the σ_z term, the unit on a†a, and
  g on the coupling. LOCK-3 and the commutators are *relational* — a factor
  shared by JC and AJC cancels — so magnitudes are pinned separately by the
  matrix-element anchors in :class:`TestConventionMagnitudes`.

These are operator-algebra contracts (like ``test_operators.py``), so the
tolerances are named symbolic floors and the parameters are dimensionless
O(1) model values — the identity is scale-free, distinct from the SI rad·s⁻¹
builder inputs (R4). The *absolute* §3 operator signs (σ_z = diag(−1, +1)
etc.) are anchored in ``test_operators.py``, which these tests rely on via the
same library operators; the Pauli sub-identities below are not a sign anchor.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from _helpers import _single_mode_hilbert
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.operators import (
    sigma_minus_ion,
    sigma_plus_ion,
    sigma_x_ion,
    sigma_z_ion,
    spin_down,
    spin_up,
)

pytestmark = pytest.mark.convention

# Machine-precision floors for O(1)-scaled operator-algebra identities.
ATOL_LOCK3 = 1e-12
ATOL_ALGEBRA = 1e-12
# A symmetry that is *broken* must be broken by a margin far above the floor;
# this guards the contrast tests against vacuous "everything commutes" passes.
# Measured broken residuals are O(10) — ~4 orders above this margin.
RTOL_SYMMETRY_BROKEN = 1e-3

_MODE = "b"
_FOCK_DIM = 6  # JC/AJC conserve their numbers; small cutoff is exact for the algebra.


# ---------------------------------------------------------------------------
# Inline reference builders — the forms the WI-2 reduced_models.py must match.
# ---------------------------------------------------------------------------


def _embedded_spins(hilbert: HilbertSpace, ion_index: int = 0):
    """The four embedded spin operators (σ_z, σ_+, σ_−, σ_x) for one ion."""
    return (
        hilbert.spin_op_for_ion(sigma_z_ion(), ion_index),
        hilbert.spin_op_for_ion(sigma_plus_ion(), ion_index),
        hilbert.spin_op_for_ion(sigma_minus_ion(), ion_index),
        hilbert.spin_op_for_ion(sigma_x_ion(), ion_index),
    )


def _jc_reference(hilbert: HilbertSpace, *, w0: float, wf: float, g: float) -> qutip.Qobj:
    sz, sp, sm, _sx = _embedded_spins(hilbert)
    a = hilbert.annihilation_for_mode(_MODE)
    a_dag = hilbert.creation_for_mode(_MODE)
    return 0.5 * w0 * sz + wf * (a_dag * a) + g * (a * sp + a_dag * sm)


def _ajc_reference(hilbert: HilbertSpace, *, w0: float, wf: float, g: float) -> qutip.Qobj:
    sz, sp, sm, _sx = _embedded_spins(hilbert)
    a = hilbert.annihilation_for_mode(_MODE)
    a_dag = hilbert.creation_for_mode(_MODE)
    return 0.5 * w0 * sz + wf * (a_dag * a) + g * (a_dag * sp + a * sm)


def _qrm_reference(hilbert: HilbertSpace, *, w0: float, wf: float, g: float) -> qutip.Qobj:
    sz, _sp, _sm, sx = _embedded_spins(hilbert)
    a = hilbert.annihilation_for_mode(_MODE)
    a_dag = hilbert.creation_for_mode(_MODE)
    return 0.5 * w0 * sz + wf * (a_dag * a) + g * sx * (a + a_dag)


def _projector_up(hilbert: HilbertSpace, ion_index: int = 0) -> qutip.Qobj:
    """Embedded spin-up projector |↑⟩⟨↑| (= σ_+σ_−)."""
    return hilbert.spin_op_for_ion(spin_up() * spin_up().dag(), ion_index)


def _ket(hilbert: HilbertSpace, spin: qutip.Qobj, n: int) -> qutip.Qobj:
    """Basis ket |spin, n⟩ in §2 order (spin ⊗ mode)."""
    return qutip.tensor(spin, qutip.basis(hilbert.mode_dim(_MODE), n))


# Dimensionless O(1) parameter triples; include a negative and a degenerate ω0.
_PARAMS = [
    (1.0, 0.7, 0.3),
    (2.5, 1.0, 0.4),
    (-1.3, 0.9, 0.5),
    (0.0, 0.8, 0.6),
]


# ---------------------------------------------------------------------------
# The §3 Pauli sub-identities that make LOCK-3 work (the algebraic foundation).
# NB: these are NOT an absolute-sign anchor (σ_x enters squared / the ± swap is
# symmetric); the absolute σ_z sign lives in test_operators.py.
# ---------------------------------------------------------------------------


class TestPauliConjugationSubidentities:
    """σ_x conjugation: the three relations LOCK-3 is built on (bare 2×2)."""

    def test_sigma_x_flips_sigma_z(self) -> None:
        """σ_x σ_z σ_x = −σ_z (CONVENTIONS.md §3)."""
        sx, sz = sigma_x_ion(), sigma_z_ion()
        assert (sx * sz * sx + sz).norm() < ATOL_ALGEBRA

    def test_sigma_x_swaps_sigma_plus_to_minus(self) -> None:
        """σ_x σ_+ σ_x = σ_−."""
        sx = sigma_x_ion()
        assert (sx * sigma_plus_ion() * sx - sigma_minus_ion()).norm() < ATOL_ALGEBRA

    def test_sigma_x_swaps_sigma_minus_to_plus(self) -> None:
        """σ_x σ_− σ_x = σ_+."""
        sx = sigma_x_ion()
        assert (sx * sigma_minus_ion() * sx - sigma_plus_ion()).norm() < ATOL_ALGEBRA

    def test_sigma_x_is_involution(self) -> None:
        """σ_x² = I, so conjugation is well-defined."""
        sx = sigma_x_ion()
        assert (sx * sx - qutip.qeye(2)).norm() < ATOL_ALGEBRA


# ---------------------------------------------------------------------------
# LOCK-3 identity on embedded library operators.
# ---------------------------------------------------------------------------


class TestLock3Identity:
    """H_AJC(ω0) = σ_x H_JC(−ω0) σ_x on the one-spin + one-mode embedding."""

    @pytest.mark.parametrize(("w0", "wf", "g"), _PARAMS)
    def test_lock3_holds(self, w0: float, wf: float, g: float) -> None:
        """The identity holds to machine precision for every parameter triple."""
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        sx = hilbert.spin_op_for_ion(sigma_x_ion(), 0)
        ajc = _ajc_reference(hilbert, w0=w0, wf=wf, g=g)
        conjugated_jc = sx * _jc_reference(hilbert, w0=-w0, wf=wf, g=g) * sx
        assert (ajc - conjugated_jc).norm() < ATOL_LOCK3

    @pytest.mark.parametrize(("w0", "wf", "g"), _PARAMS)
    def test_negative_frequency_is_essential(self, w0: float, wf: float, g: float) -> None:
        """σ_x H_JC(+ω0) σ_x = H_AJC(−ω0): for ω0 ≠ 0 this is NOT H_AJC(+ω0).

        Anti-vacuity guard: confirms the "−ω0" in the identity carries weight.
        At the degeneracy ω0 = 0 the spin term vanishes and the sign is moot,
        so the discrepancy collapses to zero there.
        """
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        sx = hilbert.spin_op_for_ion(sigma_x_ion(), 0)
        ajc = _ajc_reference(hilbert, w0=w0, wf=wf, g=g)
        wrong_sign = sx * _jc_reference(hilbert, w0=+w0, wf=wf, g=g) * sx
        discrepancy = (ajc - wrong_sign).norm()
        if w0 == 0.0:
            assert discrepancy < ATOL_LOCK3
        else:
            # The residual is |ω0| · ‖σ_z embedded‖ — comfortably macroscopic.
            assert discrepancy > RTOL_SYMMETRY_BROKEN

    def test_qrm_is_self_conjugate_under_sigma_x(self) -> None:
        """σ_x H_QRM(−ω0) σ_x = H_QRM(ω0): the dipole coupling σ_x(a+a†) is
        σ_x-invariant, so only the bare ω0 sign moves (the JC↔AJC swap is the
        coupling-specific part absent here)."""
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        sx = hilbert.spin_op_for_ion(sigma_x_ion(), 0)
        lhs = sx * _qrm_reference(hilbert, w0=-1.0, wf=0.7, g=0.3) * sx
        rhs = _qrm_reference(hilbert, w0=+1.0, wf=0.7, g=0.3)
        assert (lhs - rhs).norm() < ATOL_LOCK3


# ---------------------------------------------------------------------------
# Absolute coefficients (½ ω0, ω_f, g) — matrix-element anchors.
# The relational tests above cancel any factor shared by JC and AJC, so a
# builder with 1.0·ω0 or 2g would pass them; these anchors close that gap.
# ---------------------------------------------------------------------------


class TestConventionMagnitudes:
    """Pin the absolute coefficients via diagonal energies and coupling elements."""

    W0, WF, G = 1.3, 0.7, 0.4

    def test_jc_bare_splitting_is_half_omega0(self) -> None:
        """⟨↑,0|H_JC|↑,0⟩ = +½ω0 and ⟨↓,0|H_JC|↓,0⟩ = −½ω0 (pins the ½ and the
        §3 σ_z sign together)."""
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        jc = _jc_reference(hilbert, w0=self.W0, wf=self.WF, g=self.G)
        up0, down0 = _ket(hilbert, spin_up(), 0), _ket(hilbert, spin_down(), 0)
        assert up0.overlap(jc * up0).real == pytest.approx(+0.5 * self.W0, abs=ATOL_ALGEBRA)
        assert down0.overlap(jc * down0).real == pytest.approx(-0.5 * self.W0, abs=ATOL_ALGEBRA)

    @pytest.mark.parametrize("builder", [_jc_reference, _ajc_reference, _qrm_reference])
    def test_oscillator_quantum_is_omega_f(self, builder) -> None:
        """⟨↑,1|H|↑,1⟩ − ⟨↑,0|H|↑,0⟩ = ω_f for every model (pins the a†a unit)."""
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        ham = builder(hilbert, w0=self.W0, wf=self.WF, g=self.G)
        up0, up1 = _ket(hilbert, spin_up(), 0), _ket(hilbert, spin_up(), 1)
        gap = up1.overlap(ham * up1).real - up0.overlap(ham * up0).real
        assert gap == pytest.approx(self.WF, abs=ATOL_ALGEBRA)

    def test_jc_coupling_magnitude_is_g(self) -> None:
        """JC couples |↑,0⟩ → |↓,1⟩ with element g (via a†σ_−); |↓,0⟩ is dark."""
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        jc = _jc_reference(hilbert, w0=self.W0, wf=self.WF, g=self.G)
        up0, down1, down0 = (
            _ket(hilbert, spin_up(), 0),
            _ket(hilbert, spin_down(), 1),
            _ket(hilbert, spin_down(), 0),
        )
        assert abs(down1.overlap(jc * up0)) == pytest.approx(self.G, abs=ATOL_ALGEBRA)
        assert (jc * down0 - 0.5 * self.W0 * (-1.0) * down0).norm() < ATOL_ALGEBRA  # dark

    def test_ajc_coupling_magnitude_is_g(self) -> None:
        """AJC couples |↓,0⟩ → |↑,1⟩ with element g (via a†σ_+)."""
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        ajc = _ajc_reference(hilbert, w0=self.W0, wf=self.WF, g=self.G)
        down0, up1 = _ket(hilbert, spin_down(), 0), _ket(hilbert, spin_up(), 1)
        assert abs(up1.overlap(ajc * down0)) == pytest.approx(self.G, abs=ATOL_ALGEBRA)

    def test_qrm_coupling_magnitude_is_g(self) -> None:
        """QRM couples |↑,0⟩ → |↓,1⟩ with element g (via σ_x(a+a†))."""
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        qrm = _qrm_reference(hilbert, w0=self.W0, wf=self.WF, g=self.G)
        up0, down1 = _ket(hilbert, spin_up(), 0), _ket(hilbert, spin_down(), 1)
        assert abs(down1.overlap(qrm * up0)) == pytest.approx(self.G, abs=ATOL_ALGEBRA)


# ---------------------------------------------------------------------------
# Hermiticity + symmetry contrast (§25.1): U(1) numbers vs Z₂ parity.
# ---------------------------------------------------------------------------


class TestReducedModelStructure:
    """Each builder is Hermitian; each conserves exactly its stated symmetry."""

    @staticmethod
    def _commutator_norm(op_a: qutip.Qobj, op_b: qutip.Qobj) -> float:
        return (op_a * op_b - op_b * op_a).norm()

    @pytest.mark.parametrize("builder", [_jc_reference, _ajc_reference, _qrm_reference])
    def test_all_models_are_hermitian(self, builder) -> None:
        """JC / AJC / QRM are Hermitian (H/ℏ is an observable generator)."""
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        ham = builder(hilbert, w0=1.7, wf=0.9, g=0.4)
        assert (ham - ham.dag()).norm() < ATOL_ALGEBRA

    def test_jc_conserves_excitation_number(self) -> None:
        """[H_JC, N̂] = 0 with N̂ = a†a + |↑⟩⟨↑| (U(1) excitation symmetry)."""
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        n_op = hilbert.number_for_mode(_MODE) + _projector_up(hilbert)
        jc = _jc_reference(hilbert, w0=1.0, wf=0.7, g=0.3)
        assert self._commutator_norm(jc, n_op) < ATOL_ALGEBRA

    def test_jc_does_not_conserve_difference_number(self) -> None:
        """[H_JC, Ĉ] ≠ 0 — JC conserves N̂, not the difference Ĉ (anti-vacuity)."""
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        c_op = hilbert.number_for_mode(_MODE) - _projector_up(hilbert)
        jc = _jc_reference(hilbert, w0=1.0, wf=0.7, g=0.3)
        assert self._commutator_norm(jc, c_op) > RTOL_SYMMETRY_BROKEN

    def test_ajc_conserves_difference_number(self) -> None:
        """[H_AJC, Ĉ] = 0 with Ĉ = a†a − |↑⟩⟨↑|."""
        hilbert = _single_mode_hilbert(_FOCK_DIM, label=_MODE)
        c_op = hilbert.number_for_mode(_MODE) - _projector_up(hilbert)
        ajc = _ajc_reference(hilbert, w0=1.0, wf=0.7, g=0.3)
        assert self._commutator_norm(ajc, c_op) < ATOL_ALGEBRA

    @pytest.mark.parametrize("fock_dim", [5, 6, 7])
    def test_qrm_breaks_excitation_number_but_keeps_parity(self, fock_dim: int) -> None:
        """The non-RWA QRM conserves only Z₂ parity P = exp(iπN̂): [H,P] = 0
        while [H, N̂] ≠ 0. Parity conjugation acts entrywise as diag(±1), so the
        equality is exact even in the truncated Fock space — verified here at
        both even and odd cutoffs (no top-Fock boundary leak)."""
        hilbert = _single_mode_hilbert(fock_dim, label=_MODE)
        n_op = hilbert.number_for_mode(_MODE) + _projector_up(hilbert)
        parity = (1j * np.pi * n_op).expm()
        qrm = _qrm_reference(hilbert, w0=1.0, wf=0.7, g=0.3)
        assert self._commutator_norm(qrm, parity) < ATOL_ALGEBRA
        assert self._commutator_norm(qrm, n_op) > RTOL_SYMMETRY_BROKEN
