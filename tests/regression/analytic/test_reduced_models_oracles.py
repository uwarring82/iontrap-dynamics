# SPDX-License-Identifier: MIT
"""Analytic-regression oracles for the reduced light–matter models (§25, WP-03 RLC).

Closed-form / committed-band oracles for the §25 builders
(:func:`iontrap_dynamics.reduced_models.jaynes_cummings_hamiltonian` and siblings),
covering the task card's falsifiable cases:

* **Case A — JC ↔ AJC (the "label true" anchor).** The LOCK-3 identity
  ``H_AJC(ω₀) = σ_x H_JC(−ω₀) σ_x`` makes the two spectra coincide, and the JC
  spectrum matches the closed-form dressed-state energies
  ``E_±(n) = ω_f(n+½) ± √((δ/2)² + g²(n+1))`` with ``δ = ω_f − ω₀`` plus the dark
  ground ``|↓,0⟩`` at ``−½ω₀``.
* **Case B — the "knob" anchor.** ``|↓,0⟩`` is dark under JC (red-sideband image:
  the co-rotating coupling annihilates it) but bright under AJC (blue-sideband
  image: it couples to ``|↑,1⟩`` at element ``g`` = ½ the blue-sideband rate).
* **Case C — JC vs QRM (the "label false" anchor).** The non-RWA QRM ground state
  acquires virtual phonons ``⟨a†a⟩ ≈ (g/(ω₀+ω_f))²`` at weak coupling (2nd-order
  perturbation theory) — it is *not* the JC dark state — and JC↔QRM deviation sits
  below a pre-committed weak-coupling band and above a pre-committed USC separation
  band, with no pointwise monotonicity required.
* **Case D — reduced ↔ apparatus.** The reduced-model coupling element ``g√(n+1)``
  is exactly half the leading-order sideband Rabi rate: ``2g√(n+1) =
  blue_sideband_rabi_frequency`` and ``2g√n = red_sideband_rabi_frequency`` with
  ``g = ηΩ/2`` — the falsifiable Axis-A ↔ Axis-B bridge — and the full-Lamb–Dicke
  rate tends to the leading order as ``η²(2n+1) → 0`` (deep), diverging beyond.

Tolerances are named symbolic constants here, never read from an artefact. This
file is separate from the QuTiP-free ``test_analytic.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from _helpers import _single_mode_hilbert
from iontrap_dynamics.analytic import (
    blue_sideband_rabi_frequency,
    lamb_dicke_regime,
    red_sideband_rabi_frequency,
    red_sideband_rabi_frequency_full_ld,
)
from iontrap_dynamics.operators import sigma_x_ion
from iontrap_dynamics.reduced_models import (
    anti_jaynes_cummings_hamiltonian,
    jaynes_cummings_hamiltonian,
    quantum_rabi_hamiltonian,
)
from iontrap_dynamics.spectrum import solve_spectrum

pytestmark = pytest.mark.regression_analytic

# Exact linear-algebra floors.
ATOL_SPECTRUM = 1e-9  # eigenvalues from a dense eigh of a constructed Qobj
ATOL_IDENTITY = 1e-12  # operator / closed-form algebraic identity
RTOL_SIDEBAND = 1e-12  # the 2g√(n±1) bridge (rates ~1e6 rad·s⁻¹)
# Case-C committed bands (deviation = |E0_JC − E0_QRM|).
# The PT residual is the next (4th-) order correction, ~(g/(ω₀+ω_f))² relative,
# so 2e-2 keeps a ~6× margin at g = 0.05 (worst measured rel err ≈ 3e-3).
RTOL_QRM_PERTURBATION = 2e-2  # weak-coupling ⟨a†a⟩ vs 2nd-order PT (g ≤ 0.05)
AGREE_BAND = 1e-2  # weak coupling g/ω₀ ≤ 0.1: JC ≈ QRM
SEPARATION_BAND = 1e-1  # ultra-strong coupling g/ω₀ ≥ 1: JC ≠ QRM
USC_PHONON_FLOOR = 1e-1  # QRM ground ⟨a†a⟩ at USC: virtual phonons, not the dark state
# Case-D committed bands (full-LD vs leading-order relative deviation).
DEEP_BAND = 1e-3  # η²(2n+1) ≪ 1: full-LD ≈ leading order
BEYOND_BAND = 1e-1  # η²(2n+1) ≫ 1: the Laguerre structure departs

FOCK_DIM = 24


def _jc_closed_form_eigenvalues(
    *, omega_0: float, omega_f: float, g: float, n_max: int
) -> np.ndarray:
    """Closed-form low-lying JC spectrum: the dark ground −½ω₀ and the dressed
    pairs ``ω_f(n+½) ± √((δ/2)² + g²(n+1))`` (δ = ω_f − ω₀), ascending."""
    delta = omega_f - omega_0
    energies = [-0.5 * omega_0]
    for n in range(n_max):
        mean = omega_f * (n + 0.5)
        split = np.sqrt((0.5 * delta) ** 2 + g**2 * (n + 1))
        energies.extend((mean - split, mean + split))
    return np.sort(np.asarray(energies, dtype=np.float64))


def _ground_phonons(hamiltonian: qutip.Qobj, number_op: qutip.Qobj) -> float:
    """⟨a†a⟩ of the ground state, via ``solve_spectrum`` eigenvectors."""
    result = solve_spectrum(hamiltonian)
    ground = result.eigenvectors[:, 0]
    dense = number_op.full()
    return float(np.real(ground.conj() @ (dense @ ground)))


def _ground_energy(hamiltonian: qutip.Qobj) -> float:
    return float(solve_spectrum(hamiltonian).eigenvalues[0])


# ---------------------------------------------------------------------------
# Case A — JC ↔ AJC: spectra coincide; JC matches the dressed-state closed form.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("omega_0", "omega_f", "g"), [(1.3, 0.7, 0.4), (2.0, 2.0, 0.5), (-1.1, 0.9, 0.3)]
)
def test_case_a_lock3_spectrum_equality(omega_0: float, omega_f: float, g: float) -> None:
    """spec(H_AJC(ω₀)) = spec(H_JC(−ω₀)): the LOCK-3 σ_x conjugation is unitary."""
    hilbert = _single_mode_hilbert(FOCK_DIM, label="b")
    ajc = anti_jaynes_cummings_hamiltonian(
        hilbert, "b", ion_index=0, omega_0=omega_0, omega_f=omega_f, g=g
    )
    jc_neg = jaynes_cummings_hamiltonian(
        hilbert, "b", ion_index=0, omega_0=-omega_0, omega_f=omega_f, g=g
    )
    spec_ajc = solve_spectrum(ajc).eigenvalues
    spec_jc = solve_spectrum(jc_neg).eigenvalues
    np.testing.assert_allclose(spec_ajc, spec_jc, atol=ATOL_SPECTRUM)


@pytest.mark.parametrize(
    ("omega_0", "omega_f", "g"),
    [(1.0, 1.0, 0.3), (1.5, 0.8, 0.4), (0.6, 0.6, 0.25), (-1.1, 0.9, 0.3)],
)
def test_case_a_jc_dressed_state_spectrum(omega_0: float, omega_f: float, g: float) -> None:
    """The JC eigenvalues match ω_f(n+½) ± √((δ/2)² + g²(n+1)) plus the dark −½ω₀.

    Compared on the lowest 20 levels, well below the Fock-truncation edge. The
    last (ω₀ < 0) case exercises the regime where the dark state is not the ground
    — the closed form (and the sort) is agnostic to that ordering.
    """
    hilbert = _single_mode_hilbert(FOCK_DIM, label="b")
    jc = jaynes_cummings_hamiltonian(
        hilbert, "b", ion_index=0, omega_0=omega_0, omega_f=omega_f, g=g
    )
    numerical = solve_spectrum(jc).eigenvalues[:20]
    analytic = _jc_closed_form_eigenvalues(
        omega_0=omega_0, omega_f=omega_f, g=g, n_max=FOCK_DIM - 2
    )[:20]
    np.testing.assert_allclose(numerical, analytic, atol=ATOL_SPECTRUM)


@pytest.mark.parametrize(("omega_0", "omega_f", "g"), [(1.3, 0.7, 0.4), (-0.9, 1.1, 0.5)])
def test_case_a_lock3_operator_identity(omega_0: float, omega_f: float, g: float) -> None:
    """H_AJC(ω₀) = σ_x H_JC(−ω₀) σ_x as an exact embedded Qobj equality (§25.3).

    The regression-tier "label true" anchor; the conventions tier gates the same
    identity on inline references, the unit tier on the shipped builders. This
    operator equality is also the *sole* anchor for the global coupling sign
    (g → −g leaves the spectrum invariant), so it must not be relaxed to a
    spectrum-only check.
    """
    hilbert = _single_mode_hilbert(FOCK_DIM, label="b")
    sigma_x = hilbert.spin_op_for_ion(sigma_x_ion(), 0)
    ajc = anti_jaynes_cummings_hamiltonian(
        hilbert, "b", ion_index=0, omega_0=omega_0, omega_f=omega_f, g=g
    )
    jc_neg = jaynes_cummings_hamiltonian(
        hilbert, "b", ion_index=0, omega_0=-omega_0, omega_f=omega_f, g=g
    )
    assert (ajc - sigma_x * jc_neg * sigma_x).norm() < ATOL_IDENTITY


def test_case_a_resonant_vacuum_rabi_splitting_is_2g() -> None:
    """At resonance (ω₀ = ω_f) the n = 0 dressed pair splits by exactly 2g."""
    hilbert = _single_mode_hilbert(FOCK_DIM, label="b")
    omega, g = 1.0, 0.3
    jc = jaynes_cummings_hamiltonian(hilbert, "b", ion_index=0, omega_0=omega, omega_f=omega, g=g)
    evals = solve_spectrum(jc).eigenvalues
    # The M_0 dressed pair sits at ω_f/2 ± g — the two eigenvalues nearest ω_f/2
    # (pinned by index, robust to other manifolds for any g, unlike a fixed window).
    pair = np.sort(evals[np.argsort(np.abs(evals - 0.5 * omega))[:2]])
    assert (pair[1] - pair[0]) == pytest.approx(2.0 * g, abs=ATOL_SPECTRUM)


# ---------------------------------------------------------------------------
# Case B — the "knob" anchor: |↓,0⟩ dark under JC (red), bright under AJC (blue).
# ---------------------------------------------------------------------------


def test_case_b_ground_dark_under_jc_bright_under_ajc() -> None:
    """Case B: |↓,0⟩ is dark under JC (the red-sideband image — the co-rotating
    coupling annihilates it, since ``a|0⟩ = 0`` and ``σ_−|↓⟩ = 0``) but bright under
    AJC (the blue-sideband image — ``a†σ_+`` couples it to |↑,1⟩ at element g,
    i.e. half the blue-sideband Rabi rate; see the Case-D bridge)."""
    hilbert = _single_mode_hilbert(FOCK_DIM, label="b")
    g = 0.4
    jc = jaynes_cummings_hamiltonian(hilbert, "b", ion_index=0, omega_0=1.0, omega_f=1.0, g=g)
    ajc = anti_jaynes_cummings_hamiltonian(hilbert, "b", ion_index=0, omega_0=1.0, omega_f=1.0, g=g)
    down0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(FOCK_DIM, 0))
    up1 = qutip.tensor(qutip.basis(2, 1), qutip.basis(FOCK_DIM, 1))
    # Red / JC: |↓,0⟩ is an exact eigenstate — dark, no transition out.
    assert (jc * down0 - qutip.expect(jc, down0) * down0).norm() < ATOL_IDENTITY
    # Blue / AJC: |↓,0⟩ → |↑,1⟩ with element g — bright.
    assert abs(up1.overlap(ajc * down0)) == pytest.approx(g, abs=ATOL_IDENTITY)


# ---------------------------------------------------------------------------
# Case C — JC vs QRM: virtual phonons, weak-coupling agreement, USC separation.
# ---------------------------------------------------------------------------


def test_case_c_jc_ground_is_the_dark_state() -> None:
    """Below the resonance crossing the JC ground state is exactly |↓,0⟩:
    energy −½ω₀ and ⟨a†a⟩ = 0 (the RWA reference QRM departs from)."""
    hilbert = _single_mode_hilbert(FOCK_DIM, label="b")
    number = hilbert.number_for_mode("b")
    for g in (0.05, 0.3, 0.5):
        jc = jaynes_cummings_hamiltonian(hilbert, "b", ion_index=0, omega_0=1.0, omega_f=1.0, g=g)
        assert _ground_energy(jc) == pytest.approx(-0.5, abs=ATOL_SPECTRUM)
        assert _ground_phonons(jc, number) == pytest.approx(0.0, abs=ATOL_SPECTRUM)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_case_c_qrm_coupling_element_pins_g(n: int) -> None:
    """The QRM ``g σ_x(â+â†)`` element ⟨↓,n+1|H_QRM|↑,n⟩ = g√(n+1) — a *linear*-in-g
    anchor pinning the QRM coupling magnitude to machine precision (the weak-coupling
    ⟨a†a⟩ oracle below only scales as g²)."""
    hilbert = _single_mode_hilbert(FOCK_DIM, label="b")
    g = 0.4
    qrm = quantum_rabi_hamiltonian(hilbert, "b", ion_index=0, omega_0=1.3, omega_f=0.7, g=g)
    up_n = qutip.tensor(qutip.basis(2, 1), qutip.basis(FOCK_DIM, n))
    down_np1 = qutip.tensor(qutip.basis(2, 0), qutip.basis(FOCK_DIM, n + 1))
    assert abs(down_np1.overlap(qrm * up_n)) == pytest.approx(g * np.sqrt(n + 1), abs=ATOL_IDENTITY)


@pytest.mark.parametrize(
    ("omega_0", "omega_f", "g"), [(1.0, 1.0, 0.02), (1.3, 0.7, 0.02), (1.3, 0.7, 0.05)]
)
def test_case_c_qrm_ground_phonons_match_perturbation_theory(
    omega_0: float, omega_f: float, g: float
) -> None:
    """The non-RWA QRM ground state acquires virtual phonons ⟨a†a⟩ ≈ (g/(ω₀+ω_f))²
    at weak coupling (2nd-order PT) — an independent oracle, not |↓,0⟩. The detuned
    cases discriminate the (ω₀+ω_f) denominator from (2ω) and (ω₀−ω_f)."""
    hilbert = _single_mode_hilbert(FOCK_DIM, label="b")
    number = hilbert.number_for_mode("b")
    qrm = quantum_rabi_hamiltonian(hilbert, "b", ion_index=0, omega_0=omega_0, omega_f=omega_f, g=g)
    phonons = _ground_phonons(qrm, number)
    predicted = (g / (omega_0 + omega_f)) ** 2
    assert phonons == pytest.approx(predicted, rel=RTOL_QRM_PERTURBATION)
    assert phonons > 0.0  # strictly virtual — the QRM ground is not the dark state


@pytest.mark.parametrize("g", [0.01, 0.05, 0.1])
def test_case_c_weak_coupling_jc_approx_qrm(g: float) -> None:
    """Weak coupling g/ω₀ ≤ 0.1: |E0_JC − E0_QRM| sits below the committed band."""
    hilbert = _single_mode_hilbert(FOCK_DIM, label="b")
    jc = jaynes_cummings_hamiltonian(hilbert, "b", ion_index=0, omega_0=1.0, omega_f=1.0, g=g)
    qrm = quantum_rabi_hamiltonian(hilbert, "b", ion_index=0, omega_0=1.0, omega_f=1.0, g=g)
    assert abs(_ground_energy(jc) - _ground_energy(qrm)) < AGREE_BAND


@pytest.mark.parametrize("g", [1.0, 2.0])
def test_case_c_ultra_strong_jc_separates_from_qrm(g: float) -> None:
    """Ultra-strong coupling g/ω₀ ≥ 1: the counter-rotating terms drive
    |E0_JC − E0_QRM| above the committed separation band (RWA breaks down), and the
    QRM ground state holds real virtual phonons ⟨a†a⟩ > floor (it is *not* the JC
    dark state) — the "label false" discriminator exercised at the breakdown points."""
    hilbert = _single_mode_hilbert(FOCK_DIM, label="b")
    number = hilbert.number_for_mode("b")
    jc = jaynes_cummings_hamiltonian(hilbert, "b", ion_index=0, omega_0=1.0, omega_f=1.0, g=g)
    qrm = quantum_rabi_hamiltonian(hilbert, "b", ion_index=0, omega_0=1.0, omega_f=1.0, g=g)
    assert abs(_ground_energy(jc) - _ground_energy(qrm)) > SEPARATION_BAND
    assert _ground_phonons(qrm, number) > USC_PHONON_FLOOR


# ---------------------------------------------------------------------------
# Case D — reduced ↔ apparatus: the 2g√(n±1) bridge and the LD limit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_case_d_two_g_root_n_bridge(n: int) -> None:
    """g = ηΩ/2 makes the reduced coupling element half the leading-order sideband
    rate: 2|g|√(n+1) = blue rate, 2|g|√n = red rate (red is 0 at n = 0)."""
    omega, eta = 2.0 * np.pi * 1.0e6, 0.1
    g = eta * omega / 2.0
    blue = blue_sideband_rabi_frequency(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
    )
    red = red_sideband_rabi_frequency(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
    )
    np.testing.assert_allclose(blue, 2.0 * abs(g) * np.sqrt(n + 1), rtol=RTOL_SIDEBAND)
    np.testing.assert_allclose(red, 2.0 * abs(g) * np.sqrt(n), rtol=RTOL_SIDEBAND)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_case_d_reduced_block_element_is_half_sideband_rate(n: int) -> None:
    """The shipped AJC builder's |↓,n⟩ → |↑,n+1⟩ matrix element equals exactly half
    the apparatus blue-sideband Rabi rate, with g = ηΩ/2 — the Axis-A↔B bridge."""
    hilbert = _single_mode_hilbert(FOCK_DIM, label="b")
    omega, eta = 2.0 * np.pi * 1.0e6, 0.1
    g = eta * omega / 2.0
    ajc = anti_jaynes_cummings_hamiltonian(hilbert, "b", ion_index=0, omega_0=1.0, omega_f=1.0, g=g)
    down_n = qutip.tensor(qutip.basis(2, 0), qutip.basis(FOCK_DIM, n))
    up_np1 = qutip.tensor(qutip.basis(2, 1), qutip.basis(FOCK_DIM, n + 1))
    element = abs(up_np1.overlap(ajc * down_n))
    blue = blue_sideband_rabi_frequency(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
    )
    np.testing.assert_allclose(2.0 * element, blue, rtol=RTOL_SIDEBAND)


@pytest.mark.parametrize(
    ("eta", "n", "regime", "dev_low", "dev_high"),
    [
        # Deep: η²(2n+1) = 0.0012 ≪ 0.1; the residual is only the zero-point
        # Debye–Waller factor 1 − e^{−η²/2} ≈ 2e-4 (it never fully vanishes).
        (0.02, 1, "deep", 0.0, DEEP_BAND),
        # Intermediate: η²(2n+1) = 0.12 ∈ [0.1, 1.0); deviation between the bands.
        (0.2, 1, "intermediate", DEEP_BAND, BEYOND_BAND),
        # Beyond: η²(2n+1) = 7.2 ≫ 1; the all-orders Laguerre structure departs.
        (1.2, 2, "beyond", BEYOND_BAND, np.inf),
    ],
)
def test_case_d_full_ld_vs_leading_tracks_regime(
    eta: float, n: int, regime: str, dev_low: float, dev_high: float
) -> None:
    """The all-orders red-sideband rate tracks the LD classifier: it equals the
    leading order deep, departs beyond, with the deviation bracketed by the
    committed bands at each regime control point (no pointwise monotonicity)."""
    omega = 2.0 * np.pi * 1.0e6
    assert lamb_dicke_regime(lamb_dicke_parameter=eta, mean_phonon_number=n) == regime
    leading = red_sideband_rabi_frequency(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
    )
    full = red_sideband_rabi_frequency_full_ld(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n
    )
    deviation = abs(full - leading) / leading
    assert dev_low <= deviation < dev_high
