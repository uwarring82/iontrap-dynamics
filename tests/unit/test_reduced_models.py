# SPDX-License-Identifier: MIT
"""Unit tests for the reduced light–matter model builders (CONVENTIONS.md §25).

WP-03 / Dispatch RLB. Structural checks that ``reduced_models.jaynes_cummings_hamiltonian``,
``anti_jaynes_cummings_hamiltonian``, and ``quantum_rabi_hamiltonian`` conform to
the sealed §25 convention: Hermiticity, dims, the LOCK-3 identity and magnitude
coefficients on the *actual builders* (the conventions test anchors them on inline
references), the JC/AJC Rabi block element ``g√(n+1)``, explicit mode/ion
selection on a multi-mode space, API rejection, a structural QRM weak-coupling
sanity, and QuTiP-vs-JAX spectrum parity. Closed-form dynamics oracles
(ground-state ⟨a†a⟩ bands, 2g√(n±1) sideband relation, full-LD vs leading-order)
are the regression-tier scope of WI-3 (Dispatch RLC).
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from _helpers import _single_mode_hilbert, _two_mode_hilbert
from iontrap_dynamics import (
    anti_jaynes_cummings_hamiltonian,
    jaynes_cummings_hamiltonian,
    quantum_rabi_hamiltonian,
)
from iontrap_dynamics.exceptions import ConventionError
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.operators import sigma_x_ion, spin_down, spin_up
from iontrap_dynamics.spectrum import solve_spectrum

ATOL_EXACT = 1e-12
RTOL_SYMMETRY_BROKEN = 1e-3
FOCK_DIM = 8
_MODE = "b"
_BUILDERS = [
    jaynes_cummings_hamiltonian,
    anti_jaynes_cummings_hamiltonian,
    quantum_rabi_hamiltonian,
]


def _ket(hilbert: HilbertSpace, spin: qutip.Qobj, n: int, label: str = _MODE) -> qutip.Qobj:
    return qutip.tensor(spin, qutip.basis(hilbert.mode_dim(label), n))


def _commutator_norm(op_a: qutip.Qobj, op_b: qutip.Qobj) -> float:
    return float((op_a * op_b - op_b * op_a).norm())


# ---------------------------------------------------------------------------
# Hermiticity, dims, and the §25 forms on the actual builders.
# ---------------------------------------------------------------------------


class TestStructure:
    @pytest.mark.parametrize("builder", _BUILDERS)
    def test_hermitian_with_correct_dims(self, builder) -> None:
        """Each builder returns a Hermitian Qobj on the full space (§25)."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        ham = builder(hilbert, _MODE, ion_index=0, omega_0=1.3, omega_f=0.7, g=0.4)
        assert (ham - ham.dag()).norm() < ATOL_EXACT
        assert ham.dims == hilbert.qutip_dims()

    @pytest.mark.parametrize("builder", _BUILDERS)
    def test_g_zero_reduces_to_bare_term(self, builder) -> None:
        """At g = 0 all three builders collapse to the same bare ½ω₀σ_z + ω_f a†a."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        bare = builder(hilbert, _MODE, ion_index=0, omega_0=1.3, omega_f=0.7, g=0.0)
        reference = jaynes_cummings_hamiltonian(
            hilbert, _MODE, ion_index=0, omega_0=1.3, omega_f=0.7, g=0.0
        )
        assert (bare - reference).norm() < ATOL_EXACT

    def test_lock3_identity_on_builders(self) -> None:
        """H_AJC(ω₀) = σ_x H_JC(−ω₀) σ_x on the real builders (§25.3)."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        sx = hilbert.spin_op_for_ion(sigma_x_ion(), 0)
        ajc = anti_jaynes_cummings_hamiltonian(
            hilbert, _MODE, ion_index=0, omega_0=1.3, omega_f=0.7, g=0.4
        )
        jc_neg = jaynes_cummings_hamiltonian(
            hilbert, _MODE, ion_index=0, omega_0=-1.3, omega_f=0.7, g=0.4
        )
        assert (ajc - sx * jc_neg * sx).norm() < ATOL_EXACT

    def test_bare_splitting_and_quantum_magnitudes(self) -> None:
        """⟨↑,0|H|↑,0⟩ = +½ω₀ and the a†a quantum is ω_f (pins the absolute ½/ω_f)."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        w0, wf, g = 1.3, 0.7, 0.4
        jc = jaynes_cummings_hamiltonian(hilbert, _MODE, ion_index=0, omega_0=w0, omega_f=wf, g=g)
        up0, up1 = _ket(hilbert, spin_up(), 0), _ket(hilbert, spin_up(), 1)
        assert up0.overlap(jc * up0).real == pytest.approx(0.5 * w0, abs=ATOL_EXACT)
        gap = up1.overlap(jc * up1).real - up0.overlap(jc * up0).real
        assert gap == pytest.approx(wf, abs=ATOL_EXACT)


# ---------------------------------------------------------------------------
# Rabi block element g√(n+1) and the symmetry contrast (§25.1).
# ---------------------------------------------------------------------------


class TestCouplingAndSymmetry:
    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_jc_block_element_is_g_root_n_plus_1(self, n: int) -> None:
        """JC couples |↑,n⟩ → |↓,n+1⟩ with element g√(n+1) (the dressed Rabi scale)."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        g = 0.4
        jc = jaynes_cummings_hamiltonian(hilbert, _MODE, ion_index=0, omega_0=1.3, omega_f=0.7, g=g)
        up_n = _ket(hilbert, spin_up(), n)
        down_np1 = _ket(hilbert, spin_down(), n + 1)
        element = abs(down_np1.overlap(jc * up_n))
        assert element == pytest.approx(g * np.sqrt(n + 1), abs=ATOL_EXACT)

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_ajc_block_element_is_g_root_n_plus_1(self, n: int) -> None:
        """AJC couples |↓,n⟩ → |↑,n+1⟩ with element g√(n+1)."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        g = 0.4
        ajc = anti_jaynes_cummings_hamiltonian(
            hilbert, _MODE, ion_index=0, omega_0=1.3, omega_f=0.7, g=g
        )
        down_n = _ket(hilbert, spin_down(), n)
        up_np1 = _ket(hilbert, spin_up(), n + 1)
        element = abs(up_np1.overlap(ajc * down_n))
        assert element == pytest.approx(g * np.sqrt(n + 1), abs=ATOL_EXACT)

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_qrm_block_element_is_g_root_n_plus_1(self, n: int) -> None:
        """QRM couples |↑,n⟩ → |↓,n+1⟩ with element g√(n+1) via σ_x(â+â†).

        Pins the absolute QRM coupling magnitude: the LOCK-3 / symmetry / spectrum
        tests are all blind to a shared g rescale on the QRM (a g→2g slip).
        """
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        g = 0.4
        qrm = quantum_rabi_hamiltonian(hilbert, _MODE, ion_index=0, omega_0=1.3, omega_f=0.7, g=g)
        up_n = _ket(hilbert, spin_up(), n)
        down_np1 = _ket(hilbert, spin_down(), n + 1)
        element = abs(down_np1.overlap(qrm * up_n))
        assert element == pytest.approx(g * np.sqrt(n + 1), abs=ATOL_EXACT)

    def test_jc_dark_state_is_down_zero(self) -> None:
        """|↓,0⟩ is a JC eigenstate (the co-rotating coupling cannot lower it)."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        w0 = 1.3
        jc = jaynes_cummings_hamiltonian(
            hilbert, _MODE, ion_index=0, omega_0=w0, omega_f=0.7, g=0.4
        )
        down0 = _ket(hilbert, spin_down(), 0)
        assert (jc * down0 - (-0.5 * w0) * down0).norm() < ATOL_EXACT

    def test_symmetry_contrast(self) -> None:
        """JC conserves N̂ = a†a + |↑⟩⟨↑|; AJC conserves Ĉ = a†a − |↑⟩⟨↑|; QRM only
        the Z₂ parity P = exp(iπN̂) (and breaks N̂)."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        p_up = hilbert.spin_op_for_ion(spin_up() * spin_up().dag(), 0)
        n_op = hilbert.number_for_mode(_MODE) + p_up
        c_op = hilbert.number_for_mode(_MODE) - p_up
        parity = (1j * np.pi * n_op).expm()
        kw = dict(ion_index=0, omega_0=1.0, omega_f=0.7, g=0.4)
        jc = jaynes_cummings_hamiltonian(hilbert, _MODE, **kw)
        ajc = anti_jaynes_cummings_hamiltonian(hilbert, _MODE, **kw)
        qrm = quantum_rabi_hamiltonian(hilbert, _MODE, **kw)
        assert _commutator_norm(jc, n_op) < ATOL_EXACT
        assert _commutator_norm(ajc, c_op) < ATOL_EXACT
        assert _commutator_norm(qrm, parity) < ATOL_EXACT
        assert _commutator_norm(qrm, n_op) > RTOL_SYMMETRY_BROKEN


# ---------------------------------------------------------------------------
# Explicit mode / ion selection on a multi-mode space.
# ---------------------------------------------------------------------------


class TestModeSelection:
    @pytest.mark.parametrize("builder", _BUILDERS)
    def test_acts_only_on_selected_mode(self, builder) -> None:
        """On a two-mode space, a builder targeting mode 'a' leaves mode 'b'
        untouched ([H, n̂_b] = 0) while genuinely coupling mode 'a' ([H, n̂_a] ≠ 0)."""
        hilbert = _two_mode_hilbert(FOCK_DIM, labels=("a", "b"))
        ham = builder(hilbert, "a", ion_index=0, omega_0=1.0, omega_f=0.7, g=0.4)
        assert _commutator_norm(ham, hilbert.number_for_mode("b")) < ATOL_EXACT
        assert _commutator_norm(ham, hilbert.number_for_mode("a")) > RTOL_SYMMETRY_BROKEN

    @pytest.mark.parametrize("builder", _BUILDERS)
    def test_free_term_lands_on_selected_mode(self, builder) -> None:
        """The ω_f â†â free term is on the selected mode 'a': adding a quantum to
        'a' raises the diagonal energy by ω_f, while a quantum in 'b' adds nothing
        (the commutator test above is blind to the diagonal free term's mode)."""
        hilbert = _two_mode_hilbert(FOCK_DIM, labels=("a", "b"))
        wf = 0.7
        ham = builder(hilbert, "a", ion_index=0, omega_0=1.0, omega_f=wf, g=0.4)
        n_a, n_b = hilbert.mode_dim("a"), hilbert.mode_dim("b")

        def energy(i_a: int, i_b: int) -> float:
            ket = qutip.tensor(spin_up(), qutip.basis(n_a, i_a), qutip.basis(n_b, i_b))
            return ket.overlap(ham * ket).real

        assert energy(1, 0) - energy(0, 0) == pytest.approx(wf, abs=ATOL_EXACT)
        assert energy(0, 1) - energy(0, 0) == pytest.approx(0.0, abs=ATOL_EXACT)


# ---------------------------------------------------------------------------
# API rejection.
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("builder", _BUILDERS)
    @pytest.mark.parametrize("field", ["omega_0", "g"])
    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_signed_scalar_rejected(self, builder, field: str, bad: float) -> None:
        """Non-finite ω₀ / g raise ValueError before reaching a Qobj (signed scalars)."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        kwargs = {"ion_index": 0, "omega_0": 1.0, "omega_f": 0.7, "g": 0.4}
        kwargs[field] = bad
        with pytest.raises(ValueError, match="must be finite"):
            builder(hilbert, _MODE, **kwargs)

    @pytest.mark.parametrize("builder", _BUILDERS)
    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), 0.0, -1.0])
    def test_omega_f_must_be_finite_positive(self, builder, bad: float) -> None:
        """ω_f is an oscillator frequency: non-finite or non-positive raises
        ConventionError (§25 grants the negative-sign semantics to ω₀ only)."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        with pytest.raises(ConventionError, match="positive oscillator frequency"):
            builder(hilbert, _MODE, ion_index=0, omega_0=1.0, omega_f=bad, g=0.4)

    @pytest.mark.parametrize("builder", _BUILDERS)
    def test_unknown_mode_label_rejected(self, builder) -> None:
        """An unknown mode label raises ConventionError (delegated to HilbertSpace)."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        with pytest.raises(ConventionError, match="unknown mode"):
            builder(hilbert, "nonexistent", ion_index=0, omega_0=1.0, omega_f=0.7, g=0.4)

    @pytest.mark.parametrize("builder", _BUILDERS)
    @pytest.mark.parametrize("bad_index", [5, -1])
    def test_out_of_range_ion_index_rejected(self, builder, bad_index: int) -> None:
        """An out-of-range ion_index raises IndexError — including the negative-index
        case, which Python list-indexing would otherwise silently wrap."""
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        with pytest.raises(IndexError):
            builder(hilbert, _MODE, ion_index=bad_index, omega_0=1.0, omega_f=0.7, g=0.4)


# ---------------------------------------------------------------------------
# QRM weak-coupling sanity + backend parity (full oracles are WI-3 / RLC).
# ---------------------------------------------------------------------------


class TestSpectrumAndBackend:
    def test_qrm_ground_state_acquires_virtual_phonons(self) -> None:
        """The non-RWA QRM ground state has ⟨a†a⟩ → 0 as g/ω₀ → 0 and ⟨a†a⟩ > 0 at
        finite coupling (it is not |↓,0⟩) — a structural weak-coupling reference."""
        hilbert = _single_mode_hilbert(16, label=_MODE)
        number = hilbert.number_for_mode(_MODE)
        w0 = wf = 1.0

        def ground_phonons(g: float) -> float:
            qrm = quantum_rabi_hamiltonian(hilbert, _MODE, ion_index=0, omega_0=w0, omega_f=wf, g=g)
            _evals, evecs = qrm.eigenstates(eigvals=1)
            return float(qutip.expect(number, evecs[0]))

        assert ground_phonons(0.01) < 1e-3
        assert ground_phonons(0.5) > 1e-2

    def test_jc_and_qrm_agree_in_weak_coupling(self) -> None:
        """JC ≈ QRM ground-state energy as g/ω₀ → 0 (RWA limit), and they separate
        at strong coupling — pre-committed control points, not monotonicity."""
        hilbert = _single_mode_hilbert(16, label=_MODE)
        w0 = wf = 1.0

        def ground_gap(g: float) -> float:
            jc = jaynes_cummings_hamiltonian(
                hilbert, _MODE, ion_index=0, omega_0=w0, omega_f=wf, g=g
            )
            qrm = quantum_rabi_hamiltonian(hilbert, _MODE, ion_index=0, omega_0=w0, omega_f=wf, g=g)
            return abs(
                float(jc.eigenenergies(eigvals=1)[0]) - float(qrm.eigenenergies(eigvals=1)[0])
            )

        assert ground_gap(0.01) < 1e-3  # weak coupling: JC and QRM agree
        assert ground_gap(1.0) > 1e-2  # ultra-strong: counter-rotating terms matter

    def test_spectrum_backend_parity_qutip_vs_jax(self) -> None:
        """solve_spectrum eigenvalues agree between the scipy and JAX backends
        (the available JAX path; the trajectory backend needs Dynamiqs)."""
        pytest.importorskip("jax")
        hilbert = _single_mode_hilbert(FOCK_DIM, label=_MODE)
        qrm = quantum_rabi_hamiltonian(hilbert, _MODE, ion_index=0, omega_0=1.3, omega_f=0.7, g=0.5)
        scipy_r = solve_spectrum(qrm, backend_name="spectrum-scipy")
        jax_r = solve_spectrum(qrm, backend_name="spectrum-jax")
        np.testing.assert_allclose(jax_r.eigenvalues, scipy_r.eigenvalues, rtol=1e-9, atol=1e-9)
