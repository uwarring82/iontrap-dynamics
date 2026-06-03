# SPDX-License-Identifier: MIT
"""Analytic-regression oracle for the probe-QFI benchmark (WP-02 WI-6, MCF).

Binding closed-form anchor for ``tools/run_benchmark_probe_qfi.py``. WI-6
**consumes** WP-01's SLD-QFI primitive (`information/fisher.py`, dispatch EDA,
CONVENTIONS §19) — it adds **no new public symbol** (§8): a coherent/squeezed
probe is a pure state under unitary encoding ρ(θ)=e^{−iθG}ρe^{+iθG}, exactly the
shape ``quantum_fisher_information_trajectory`` already expresses.

Closed forms validated (pure-state identity ``F_Q = 4·Var(G)``, CONVENTIONS §19):

* coherent ``|α⟩`` phase (G = n̂):           ``F_Q = 4|α|²``  (shot-noise / SQL)
* coherent ``|α⟩`` displacement (G = x, p):  ``F_Q = 2``      (α-independent)
* squeezed ``|ξ=r⟩`` displacement (G = x):   ``F_Q = 2e^{−2r}``  (sub-shot-noise)
* squeezed ``|ξ=r⟩`` displacement (G = p):   ``F_Q = 2e^{+2r}``  (anti-squeezed)
* squeezed ``|ξ=r⟩`` phase (G = n̂):          ``F_Q = 2 sinh²(2r)``
* qubit ``|+⟩`` phase (G = J_z = ½σ_z):      ``F_Q = 1``, and the **σ_y**-basis
  measurement (the SLD eigenbasis) **saturates** the Cramér–Rao bound
  (CFI = QFI), while σ_x is phase-insensitive (CFI = 0).

Convention note (verified by direct computation, not the literature prose):
``squeezed_vacuum_mode(r)`` with real ``r`` squeezes the ``x = (a+a†)/√2``
quadrature (``Var(x) = ½e^{−2r}``), so the *x*-generator gives the sub-shot-noise
``2e^{−2r}`` and the *p*-generator the anti-squeezed ``2e^{+2r}``. Generators are
labelled explicitly; the headline physics (one quadrature beats the shot-noise
floor ``2``) is convention-independent.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from _helpers import _single_mode_hilbert, _spin_hilbert
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.information import (
    classical_fisher_information,
    cramer_rao_bound,
    quantum_fisher_information_trajectory,
)
from iontrap_dynamics.operators import (
    sigma_x_ion,
    sigma_y_ion,
    sigma_z_ion,
    spin_down,
    spin_up,
)
from iontrap_dynamics.states import coherent_mode, compose_density, squeezed_vacuum_mode

pytestmark = pytest.mark.regression_analytic

ATOL_PROBE_QFI = 1e-9  # exact finite-dimensional cases (qubit CFI saturation, CRB)
ATOL_PROBE_QFI_CV = 1e-6  # Fock-truncated continuous-variable probes
FOCK_DIM_COHERENT = 60
FOCK_DIM_SQUEEZED = 100  # converges F_Q to <1e-6 through r = 1.0

_MODE = "b"


def _embed(hilbert: HilbertSpace, mode_ket: qutip.Qobj) -> qutip.Qobj:
    """Embed a single-mode probe as ``|↓⟩⊗|ψ⟩`` (the spin is an inert spectator)."""
    return compose_density(
        hilbert, spin_states_per_ion=[spin_down()], mode_states_by_label={_MODE: mode_ket}
    )


def _number(hilbert: HilbertSpace) -> qutip.Qobj:
    return hilbert.number_for_mode(_MODE)


def _quad_x(hilbert: HilbertSpace) -> qutip.Qobj:
    a = hilbert.annihilation_for_mode(_MODE)
    return (a + a.dag()) / np.sqrt(2.0)


def _quad_p(hilbert: HilbertSpace) -> qutip.Qobj:
    a = hilbert.annihilation_for_mode(_MODE)
    return (a - a.dag()) / (1j * np.sqrt(2.0))


def _qfi(hilbert: HilbertSpace, rho: qutip.Qobj, generator: qutip.Qobj) -> float:
    return float(
        quantum_fisher_information_trajectory([rho], hilbert=hilbert, generator=generator)[0]
    )


# --------------------------------------------------------------------------
# The pure-state identity F_Q = 4·Var(G) — the master oracle every case rests on
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.5, 1.5, 2.0])
def test_qfi_is_four_times_generator_variance(alpha: float) -> None:
    # Independent of any closed form: the API must return 4·Var(G) for a pure
    # probe. Cross-checked against qutip.variance (a different code path).
    hilbert = _single_mode_hilbert(FOCK_DIM_COHERENT)
    rho = _embed(hilbert, coherent_mode(FOCK_DIM_COHERENT, alpha))
    for generator in (_number(hilbert), _quad_x(hilbert), _quad_p(hilbert)):
        expected = 4.0 * float(qutip.variance(generator, rho).real)
        assert _qfi(hilbert, rho, generator) == pytest.approx(expected, abs=ATOL_PROBE_QFI_CV)


# --------------------------------------------------------------------------
# Coherent probe — the standard quantum limit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
def test_coherent_phase_qfi_is_four_n_bar(alpha: float) -> None:
    # Poissonian: Var(n̂) = ⟨n̂⟩ = |α|² → F_Q = 4|α|² (shot-noise / SQL scaling).
    hilbert = _single_mode_hilbert(FOCK_DIM_COHERENT)
    rho = _embed(hilbert, coherent_mode(FOCK_DIM_COHERENT, alpha))
    assert _qfi(hilbert, rho, _number(hilbert)) == pytest.approx(
        4.0 * alpha**2, abs=ATOL_PROBE_QFI_CV
    )


@pytest.mark.parametrize("alpha", [0.0, 0.7, 1.5, 2.0])
def test_coherent_displacement_qfi_is_shot_noise_floor(alpha: float) -> None:
    # Minimum-uncertainty: Var(x) = Var(p) = ½ for every α → F_Q = 2.
    hilbert = _single_mode_hilbert(FOCK_DIM_COHERENT)
    rho = _embed(hilbert, coherent_mode(FOCK_DIM_COHERENT, alpha))
    assert _qfi(hilbert, rho, _quad_x(hilbert)) == pytest.approx(2.0, abs=ATOL_PROBE_QFI_CV)
    assert _qfi(hilbert, rho, _quad_p(hilbert)) == pytest.approx(2.0, abs=ATOL_PROBE_QFI_CV)


# --------------------------------------------------------------------------
# Squeezed-vacuum probe — sub-shot-noise enhancement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("r", [0.2, 0.4, 0.6, 0.8, 1.0])
def test_squeezed_displacement_beats_and_pays_shot_noise(r: float) -> None:
    # The metrological point: the squeezed quadrature (x here) gives F_Q = 2e^{−2r}
    # BELOW the shot-noise floor 2; the conjugate (p) pays 2e^{+2r} above it.
    hilbert = _single_mode_hilbert(FOCK_DIM_SQUEEZED)
    rho = _embed(hilbert, squeezed_vacuum_mode(FOCK_DIM_SQUEEZED, r))
    fq_x = _qfi(hilbert, rho, _quad_x(hilbert))
    fq_p = _qfi(hilbert, rho, _quad_p(hilbert))
    assert fq_x == pytest.approx(2.0 * np.exp(-2.0 * r), abs=ATOL_PROBE_QFI_CV)
    assert fq_p == pytest.approx(2.0 * np.exp(+2.0 * r), abs=ATOL_PROBE_QFI_CV)
    assert fq_x < 2.0 < fq_p  # sub-shot-noise on one quadrature, penalty on the other


@pytest.mark.parametrize("r", [0.2, 0.5, 0.8, 1.0])
def test_squeezed_phase_qfi_is_two_sinh_squared_two_r(r: float) -> None:
    # Phase sensing with a squeezed vacuum: F_Q = 4·Var(n̂) = 2 sinh²(2r).
    hilbert = _single_mode_hilbert(FOCK_DIM_SQUEEZED)
    rho = _embed(hilbert, squeezed_vacuum_mode(FOCK_DIM_SQUEEZED, r))
    assert _qfi(hilbert, rho, _number(hilbert)) == pytest.approx(
        2.0 * np.sinh(2.0 * r) ** 2, abs=ATOL_PROBE_QFI_CV
    )


# --------------------------------------------------------------------------
# CFI saturation — the optimal measurement reaches the quantum Cramér–Rao bound
# --------------------------------------------------------------------------


def _cfi_in_measurement_basis(
    psi0: qutip.Qobj, generator: qutip.Qobj, measurement: qutip.Qobj
) -> float:
    """CFI of a ±1-eigenvalue ``measurement`` on ``ρ(θ)=e^{−iθG}ρe^{+iθG}`` at θ=0.

    Derived from the actual operators (not hand-supplied): the two outcomes are
    ``P_± = (I ± M)/2`` with ``p_± = ⟨ψ|P_±|ψ⟩`` and, for unitary encoding,
    ``∂_θp_±|₀ = i⟨ψ|[G, P_±]|ψ⟩``; then ``CFI = Σ (∂_θp)²/p``.
    """
    identity = qutip.qeye(measurement.dims[0])
    projectors = [(identity + measurement) / 2, (identity - measurement) / 2]
    probabilities = [float(qutip.expect(proj, psi0).real) for proj in projectors]
    derivatives = [
        float((1j * qutip.expect(generator * proj - proj * generator, psi0)).real)
        for proj in projectors
    ]
    return classical_fisher_information(probabilities, parameter_derivative=derivatives)


def test_cfi_saturates_qfi_only_for_the_optimal_measurement() -> None:
    # Probe |+⟩ (a σ_x eigenstate), generator J_z = ½σ_z: QFI = 4·Var(J_z) = 1.
    # The SLD eigenbasis here is σ_y, NOT σ_x: a σ_y measurement saturates the
    # Cramér–Rao bound (CFI = QFI), while σ_x is phase-insensitive (CFI = 0) —
    # |+⟩ already lies along the σ_x axis, so a first-order rotation leaves the
    # σ_x statistics unchanged. Probabilities and derivatives are computed from
    # the probe and the measurement operators, not hand-supplied.
    hilbert = _spin_hilbert(1)
    jz = 0.5 * hilbert.spin_op_for_ion(sigma_z_ion(), 0)
    plus = (spin_up() + spin_down()).unit()
    qfi = _qfi(hilbert, plus, jz)
    assert qfi == pytest.approx(1.0, abs=ATOL_PROBE_QFI)
    cfi_optimal = _cfi_in_measurement_basis(plus, jz, sigma_y_ion())
    assert cfi_optimal == pytest.approx(qfi, abs=ATOL_PROBE_QFI)  # Braunstein–Caves saturation
    cfi_insensitive = _cfi_in_measurement_basis(plus, jz, sigma_x_ion())
    assert cfi_insensitive == pytest.approx(0.0, abs=ATOL_PROBE_QFI)  # phase-insensitive basis
    assert cfi_optimal <= qfi + ATOL_PROBE_QFI  # CFI ≤ QFI always


def test_cramer_rao_bound_inverts_the_fisher_information() -> None:
    # CRB = 1/F; for the |+⟩ / J_z probe F_Q = 1 so the bound is 1.
    assert cramer_rao_bound(1.0) == pytest.approx(1.0, abs=ATOL_PROBE_QFI)
    assert cramer_rao_bound(4.0) == pytest.approx(0.25, abs=ATOL_PROBE_QFI)
