# SPDX-License-Identifier: MIT
"""Convention-enforcement tests for non-adiabatic squeezing (§26).

WP-05 / Dispatch SQ1. These tests anchor the behaviour sealed in
``CONVENTIONS.md`` §26 (proposed in ``WP/SQ-conventions-proposal.md``, sealed by
the maintainer under a ``CONVENTION_VERSION`` 0.4 → 0.5 bump) for the
non-adiabatic-squeezing generator (:func:`nonadiabatic_squeezing_hamiltonian`),
its :class:`FrequencyWaveform` contract, and the single-mode Gaussian readout
(:mod:`iontrap_dynamics.gaussian`) that consumes it.

Convention under test (§26), all ``H/ℏ`` in rad·s⁻¹, pure-motional
Schrödinger-picture (NOT the §5 interaction picture):

* **§26.1 generator** ``H(t)/ℏ = ω(t)(â†â + ½) − (i/4)(d ln ω/dt)(â†² − â²)`` in a
  **fixed** ``ω(0)`` basis; time-list Hermitian split
  ``[[â†â+½, ω(t)], [−i(â†²−â²), ¼·d ln ω/dt]]``. Gates: *sudden* (narrowing
  smooth ramp) → ``r = ½|ln(ω_f/ω_i)|``; *cyclic adiabatic* (wide down/up ramp
  returning to ``ω_ini``) → ``r → 0``.
* **§26.2 quadratures** ``x̂ = â+â†``, ``p̂ = i(â†−â)``, vacuum variance 1
  (``V = 𝟙₂``).
* **§26.4 readout** ``ν = √(det V)``, ``r = ¼·ln(λ_max/λ_min)`` (eigenvalue
  ratio, **not** ``tr V``), ``n̄_sq = sinh²r``, ``α = (⟨x̂⟩+i⟨p̂⟩)/2``.

The sudden/adiabatic gate values were calibrated empirically against the
Silveri-frame evolution (see WP-05 SQ1 LOGBOOK); the fixed-``ω(0)``-basis
squeezing-parameter is invariant under the residual free rotation, so the
readout time past the ramp is immaterial.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import qutip

from iontrap_dynamics import gaussian, phase_space, waveforms
from iontrap_dynamics.exceptions import (
    ConventionError,
    ConvergenceError,
    FockConvergenceWarning,
    FockQualityWarning,
)
from iontrap_dynamics.hamiltonians import (
    displacement_force_hamiltonian,
    nonadiabatic_squeezing_hamiltonian,
)
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.results import StorageMode, WarningSeverity
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import coherent_mode, squeezed_vacuum_mode
from iontrap_dynamics.system import IonSystem

TWOPI = 2.0 * np.pi

# Named tolerance floors.
_EXACT = 1e-6  # operator-algebra / readout exactness
_READOUT_REL = 1e-4  # state-factory readout round-trip
_SUDDEN_REL = 0.05  # narrowing-ramp convergence to ½|ln(ω_f/ω_i)| (calibrated ≈1.3%)
_ADIABATIC_ABS = 1e-2  # wide-ramp residual squeezing floor


def _single_mode(fock: int, freq_hz: float = 2.0e6, *, label: str = "m") -> HilbertSpace:
    mode = ModeConfig(
        label=label,
        frequency_rad_s=TWOPI * freq_hz,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={label: fock})


class TestQuadratureReadout:
    """§26.2 / §26.4 — the covariance readout on states with known Gaussians."""

    def test_vacuum_covariance_is_identity(self) -> None:
        cov, mean = gaussian.covariance_matrix(qutip.basis(40, 0))
        assert np.allclose(cov, np.eye(2), atol=_EXACT)
        assert np.allclose(mean, 0.0, atol=_EXACT)
        assert gaussian.squeezing_parameter(cov) == pytest.approx(0.0, abs=_EXACT)
        assert gaussian.symplectic_eigenvalue(cov) == pytest.approx(1.0, abs=_EXACT)

    @pytest.mark.parametrize("r0", [0.3, 0.6, 0.9])
    def test_squeezed_vacuum_principal_variances(self, r0: float) -> None:
        cov, _ = gaussian.covariance_matrix(squeezed_vacuum_mode(60, r0))
        lam = np.linalg.eigvalsh(cov)
        assert lam[0] == pytest.approx(np.exp(-2.0 * r0), rel=_READOUT_REL)
        assert lam[-1] == pytest.approx(np.exp(2.0 * r0), rel=_READOUT_REL)
        assert gaussian.squeezing_parameter(cov) == pytest.approx(r0, abs=_READOUT_REL)
        assert gaussian.symplectic_eigenvalue(cov) == pytest.approx(1.0, abs=_READOUT_REL)
        assert gaussian.mean_squeezed_occupation(cov) == pytest.approx(
            float(np.sinh(r0) ** 2), rel=_READOUT_REL
        )

    def test_coherent_displacement_and_occupation(self) -> None:
        alpha = 1.2 + 0.5j
        cov, mean = gaussian.covariance_matrix(coherent_mode(60, alpha))
        assert gaussian.coherent_amplitude(mean) == pytest.approx(alpha, abs=_READOUT_REL)
        assert gaussian.squeezing_parameter(cov) == pytest.approx(0.0, abs=_READOUT_REL)
        assert gaussian.mean_occupation(cov, mean) == pytest.approx(
            abs(alpha) ** 2, rel=_READOUT_REL
        )


class TestThermalRegression:
    """§26.4 — the eigenvalue-ratio ``r`` separates squeezing from thermal width; ``tr V`` cannot."""

    @pytest.mark.parametrize("nbar", [0.0, 0.5, 1.3])
    def test_squeezed_thermal_r_invariant_under_nbar(self, nbar: float) -> None:
        fock, r0 = 60, 0.45
        squeeze = qutip.squeeze(fock, r0)
        rho = squeeze * qutip.thermal_dm(fock, nbar) * squeeze.dag()
        cov, _ = gaussian.covariance_matrix(rho)
        # r is the pure-squeezing parameter, invariant under the thermal core...
        assert gaussian.squeezing_parameter(cov) == pytest.approx(r0, abs=1e-3)
        # ...while ν carries the thermal content.
        assert gaussian.symplectic_eigenvalue(cov) == pytest.approx(2.0 * nbar + 1.0, rel=1e-3)

    def test_trace_form_would_conflate_thermal(self) -> None:
        # The naive ½·arccosh(½·tr V) form inflates with n̄_th; the eigenvalue
        # form (§26.4) does not — this is the regression that pins the choice.
        fock, r0, nbar = 60, 0.45, 1.3
        squeeze = qutip.squeeze(fock, r0)
        cov, _ = gaussian.covariance_matrix(squeeze * qutip.thermal_dm(fock, nbar) * squeeze.dag())
        eigenvalue_r = gaussian.squeezing_parameter(cov)
        trace_r = 0.5 * float(np.arccosh(0.5 * np.trace(cov)))
        assert eigenvalue_r == pytest.approx(r0, abs=1e-3)
        assert trace_r > eigenvalue_r + 0.5  # tr V badly conflates thermal width


class TestSuddenAdiabaticGates:
    """§26.1 — the fixed-``ω(0)``-basis sudden and adiabatic limits."""

    # Both ramp directions: a narrow (width = 0.01·T_i) squeeze kick converges to
    # ½|ln(ω_f/ω_i)| regardless of whether ω increases or decreases (the result
    # depends only on the ratio). Calibrated residuals: 0.33 % (down), 1.33 % (up).
    @pytest.mark.parametrize(("wi", "wf", "fock"), [(2.0e6, 1.0e6, 40), (1.0e6, 2.0e6, 60)])
    def test_sudden_narrow_ramp(self, wi: float, wf: float, fock: int) -> None:
        hilbert = _single_mode(fock, wi)
        psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(fock, 0))
        period = 1.0 / wi
        width = 0.01 * period
        center = 25.0 * width
        tmax = 55.0 * width
        wave = waveforms.smooth_ramp(
            omega_i=TWOPI * wi, omega_f=TWOPI * wf, center_s=center, width_s=width
        )
        hamiltonian = nonadiabatic_squeezing_hamiltonian(hilbert, "m", wave)
        result = solve(
            hilbert=hilbert,
            hamiltonian=hamiltonian,
            initial_state=psi0,
            times=np.linspace(0.0, tmax, 1500),
            storage_mode=StorageMode.EAGER,
        )
        cov, _ = gaussian.covariance_matrix(
            gaussian.reduced_single_mode(result.states[-1], hilbert, "m")
        )
        target = 0.5 * abs(np.log(wf / wi))
        assert gaussian.squeezing_parameter(cov) == pytest.approx(target, rel=_SUDDEN_REL)

    def test_cyclic_adiabatic_wide_ramp(self) -> None:
        wi, wf, fock = 2.0e6, 1.0e6, 50
        hilbert = _single_mode(fock, wi)
        psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(fock, 0))
        period = 1.0 / wi
        width = 2.0 * period
        first_center = 5.0 * width
        second_center = 15.0 * width
        tmax = 20.0 * width
        ln_half_swing = 0.5 * math.log(wf / wi)

        def omega(t: float) -> float:
            log_swing = ln_half_swing * (
                math.tanh((t - first_center) / width) - math.tanh((t - second_center) / width)
            )
            return TWOPI * wi * math.exp(log_swing)

        def d_ln_omega_dt(t: float) -> float:
            x1 = (t - first_center) / width
            x2 = (t - second_center) / width
            sech2_first = 1.0 / math.cosh(x1) ** 2
            sech2_second = 1.0 / math.cosh(x2) ** 2
            return ln_half_swing * (sech2_first - sech2_second) / width

        wave = waveforms.FrequencyWaveform(omega=omega, d_ln_omega_dt=d_ln_omega_dt)
        times = np.linspace(0.0, tmax, 2500)
        hamiltonian = nonadiabatic_squeezing_hamiltonian(
            hilbert, "m", wave, validate_at=tuple(float(t) for t in times[::250])
        )
        result = solve(
            hilbert=hilbert,
            hamiltonian=hamiltonian,
            initial_state=psi0,
            times=times,
            storage_mode=StorageMode.EAGER,
        )
        cov, _ = gaussian.covariance_matrix(
            gaussian.reduced_single_mode(result.states[-1], hilbert, "m")
        )
        assert gaussian.squeezing_parameter(cov) < _ADIABATIC_ABS


class TestHamiltonianStructure:
    """§26.1 — the builder emits two Hermitian pieces with real coefficients."""

    def _wave(self) -> waveforms.FrequencyWaveform:
        return waveforms.smooth_ramp(
            omega_i=TWOPI * 2.0e6, omega_f=TWOPI * 1.0e6, center_s=1.0e-6, width_s=1.0e-7
        )

    def test_returns_two_hermitian_pieces(self) -> None:
        hilbert = _single_mode(10, 2.0e6)
        hamiltonian = nonadiabatic_squeezing_hamiltonian(hilbert, "m", self._wave())
        assert isinstance(hamiltonian, list)
        assert len(hamiltonian) == 2
        for piece, coeff in hamiltonian:
            assert piece.isherm  # both H_free and H_sq are Hermitian
            assert callable(coeff)  # real-valued QuTiP coefficient callable

    def test_squeezing_coefficient_is_quarter_dln(self) -> None:
        # The second piece's coefficient is ¼·d ln ω/dt (real).
        hilbert = _single_mode(10, 2.0e6)
        wave = self._wave()
        hamiltonian = nonadiabatic_squeezing_hamiltonian(hilbert, "m", wave)
        _, sq_coeff = hamiltonian[1]
        t = 1.05e-6
        assert sq_coeff(t, None) == pytest.approx(0.25 * wave.d_ln_omega_dt(t), rel=1e-9)

    def test_unknown_backend_raises(self) -> None:
        hilbert = _single_mode(10, 2.0e6)
        with pytest.raises(ConventionError):
            nonadiabatic_squeezing_hamiltonian(hilbert, "m", self._wave(), backend="numpy")


class TestWaveformContract:
    """§26.1 / WP-05 R5 — the FrequencyWaveform validation contract."""

    def test_rejects_nonpositive_omega(self) -> None:
        wave = waveforms.FrequencyWaveform(omega=lambda t: -1.0e6, d_ln_omega_dt=lambda t: 0.0)
        with pytest.raises(ConventionError):
            wave.validate_at([0.0, 1.0e-6])

    def test_rejects_nonfinite_omega(self) -> None:
        wave = waveforms.FrequencyWaveform(
            omega=lambda t: float("nan"), d_ln_omega_dt=lambda t: 0.0
        )
        with pytest.raises(ConventionError):
            wave.validate_at([0.0])

    def test_rejects_nonfinite_derivative(self) -> None:
        wave = waveforms.FrequencyWaveform(
            omega=lambda t: 1.0e6, d_ln_omega_dt=lambda t: float("inf")
        )
        with pytest.raises(ConventionError):
            wave.validate_at([0.0])

    def test_sinusoidal_positivity_guard(self) -> None:
        with pytest.raises(ConventionError):
            waveforms.sinusoidal_modulation(
                omega_ini=1.0e6, mod_amplitude=1.5e6, mod_frequency=2.0e6
            )

    def test_named_shape_derivative_is_analytic(self) -> None:
        # The named shapes supply d ln ω/dt analytically; cross-check it against a
        # finite difference of ln ω (the FD is a test-only sanity check, never used
        # at run time — §26.1 forbids runtime numerical differentiation).
        wave = waveforms.gaussian_quench(
            omega_ini=TWOPI * 2.0e6, amplitude=0.5, width_s=1.0e-7, center_s=5.0e-7
        )
        t, h = 5.3e-7, 1.0e-12
        fd = (np.log(wave.omega(t + h)) - np.log(wave.omega(t - h))) / (2.0 * h)
        assert wave.d_ln_omega_dt(t) == pytest.approx(fd, rel=1e-4)


class TestCrossBackend:
    """WP-05 SQ1 gate — QuTiP vs JAX agreement at the 1e-3 design tolerance."""

    def test_qutip_jax_squeezing_agreement(self) -> None:
        pytest.importorskip("dynamiqs")
        wi, wf, fock = 2.0e6, 1.0e6, 40
        hilbert = _single_mode(fock, wi)
        psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(fock, 0))
        period = 1.0 / wi
        wave = waveforms.smooth_ramp(
            omega_i=TWOPI * wi,
            omega_f=TWOPI * wf,
            center_s=10.0 * period,
            width_s=0.1 * period,
        )
        times = np.linspace(0.0, 22.0 * period, 2000)
        r_by_backend = {}
        for backend in ("qutip", "jax"):
            hamiltonian = nonadiabatic_squeezing_hamiltonian(hilbert, "m", wave, backend=backend)
            result = solve(
                hilbert=hilbert,
                hamiltonian=hamiltonian,
                initial_state=psi0,
                times=times,
                storage_mode=StorageMode.EAGER,
                backend=backend,
            )
            cov, _ = gaussian.covariance_matrix(
                gaussian.reduced_single_mode(result.states[-1], hilbert, "m")
            )
            r_by_backend[backend] = gaussian.squeezing_parameter(cov)
        assert r_by_backend["qutip"] == pytest.approx(r_by_backend["jax"], abs=1e-3)


class TestWignerScaling:
    """§26.3 — the Wigner ``g``-pin gives vacuum-variance-1 phase space."""

    def test_g_is_one(self) -> None:
        assert phase_space.WIGNER_G == 1.0

    def test_vacuum_wigner_variance_is_one(self) -> None:
        xs = np.linspace(-6.0, 6.0, 241)
        w = phase_space.wigner(qutip.basis(40, 0), xs)
        marginal_x = w.sum(axis=0)
        marginal_x = marginal_x / marginal_x.sum()
        var_x = float((marginal_x * xs**2).sum() - (marginal_x * xs).sum() ** 2)
        assert var_x == pytest.approx(1.0, abs=2e-3)

    def test_squeezed_wigner_matches_covariance(self) -> None:
        # The g=1 Wigner ellipse's squeezed-axis variance equals the covariance
        # eigenvalue e^{-2r} (the readout and the phase-space picture agree).
        r0 = 0.6
        xs = np.linspace(-7.0, 7.0, 281)
        w = phase_space.wigner(squeezed_vacuum_mode(60, r0), xs)
        marginal_x = w.sum(axis=0)
        marginal_x = marginal_x / marginal_x.sum()
        var_x = float((marginal_x * xs**2).sum() - (marginal_x * xs).sum() ** 2)
        assert var_x == pytest.approx(float(np.exp(-2.0 * r0)), rel=2e-2)


class TestPhaseSpaceFacade:
    """WP-05 R4 — phase_space façades reduce + delegate to the gaussian core."""

    def test_readout_reduces_full_state(self) -> None:
        hilbert = _single_mode(30, 2.0e6)
        r0 = 0.5
        psi = qutip.tensor(qutip.basis(2, 0), squeezed_vacuum_mode(30, r0))
        readout = phase_space.phase_space_readout(psi, hilbert=hilbert, mode_label="m")
        assert readout.squeezing_parameter == pytest.approx(r0, abs=1e-3)
        assert readout.symplectic_eigenvalue == pytest.approx(1.0, abs=1e-3)

    def test_readout_matches_manual_reduction(self) -> None:
        hilbert = _single_mode(30, 2.0e6)
        psi = qutip.tensor(qutip.basis(2, 0), coherent_mode(30, 0.8 + 0.3j))
        via_facade = phase_space.phase_space_readout(psi, hilbert=hilbert, mode_label="m")
        manual = gaussian.gaussian_readout(gaussian.reduced_single_mode(psi, hilbert, "m"))
        assert via_facade.coherent_amplitude == pytest.approx(manual.coherent_amplitude, abs=1e-6)

    def test_readout_forwards_truncation_controls(self) -> None:
        hilbert = _single_mode(20, 2.0e6)
        squeezed = qutip.squeeze(20, 1.2) * qutip.basis(20, 0)
        psi = qutip.tensor(qutip.basis(2, 0), squeezed)
        with pytest.raises(ConvergenceError):
            phase_space.phase_space_readout(psi, hilbert=hilbert, mode_label="m")
        readout = phase_space.phase_space_readout(
            psi,
            hilbert=hilbert,
            mode_label="m",
            check_truncation=False,
        )
        assert readout.squeezing_parameter >= 0.0

    def test_multi_subsystem_without_labels_raises(self) -> None:
        psi = qutip.tensor(qutip.basis(2, 0), qutip.basis(10, 0))
        with pytest.raises(ValueError):
            phase_space.wigner(psi, np.linspace(-4.0, 4.0, 41))


class TestPhononDistribution:
    """§26.4 — direct ``Pₙ = ⟨n|ρ|n⟩`` and the pure-squeezed-vacuum oracle."""

    def test_vacuum_pn(self) -> None:
        pn = gaussian.phonon_number_distribution(qutip.basis(20, 0))
        assert pn[0] == pytest.approx(1.0, abs=_EXACT)
        assert pn[1:].sum() == pytest.approx(0.0, abs=_EXACT)

    @pytest.mark.parametrize("r0", [0.3, 0.8])
    def test_squeezed_vacuum_pn_is_even_only(self, r0: float) -> None:
        pn = gaussian.phonon_number_distribution(squeezed_vacuum_mode(200, r0))
        oracle = gaussian.pure_squeezed_vacuum_pn(r0, 199)
        assert np.max(np.abs(pn - oracle)) < 1e-12  # matches the closed form
        assert np.max(pn[1::2]) < 1e-12  # odd-n identically zero — the pair signature
        assert pn.sum() == pytest.approx(1.0, rel=1e-9)
        n = np.arange(200)
        assert float((n * pn).sum()) == pytest.approx(float(np.sinh(r0) ** 2), rel=1e-4)


class TestFockTruncationGuard:
    """§13/§15 — the parity-aware tail-window truncation guard (Dispatch SQ4)."""

    @staticmethod
    def _sqz(r: float, fock: int) -> qutip.Qobj:
        return qutip.squeeze(fock, r) * qutip.basis(fock, 0)

    def test_well_truncated_is_silent(self) -> None:
        assert gaussian.check_fock_truncation(self._sqz(0.6, 60)) == ()

    def test_level1_convergence_warning(self) -> None:
        with pytest.warns(FockConvergenceWarning):
            records = gaussian.check_fock_truncation(self._sqz(0.8, 24))
        assert len(records) == 1
        assert records[0].severity == WarningSeverity.CONVERGENCE
        assert records[0].category == "fock_truncation"

    def test_level2_quality_warning(self) -> None:
        with pytest.warns(FockQualityWarning):
            records = gaussian.check_fock_truncation(self._sqz(0.9, 24))
        assert records[0].severity == WarningSeverity.QUALITY

    def test_level3_raises(self) -> None:
        with pytest.raises(ConvergenceError):
            gaussian.check_fock_truncation(self._sqz(1.2, 20))

    def test_parity_blind_hole_is_closed(self) -> None:
        # THE point of SQ4: a squeezed vacuum in an EVEN-dim Fock space has an empty
        # top (odd) level — a top-level-only metric would report convergence — yet
        # the even tail is saturated, so the parity-aware guard raises.
        squeezed = self._sqz(1.2, 20)
        pn = gaussian.phonon_number_distribution(squeezed)
        assert pn[-1] == pytest.approx(0.0, abs=1e-12)  # top (odd) level: the blind spot
        assert pn[-2:].sum() > 1e-3  # but the top-2 edge (its even level) is saturated
        with pytest.raises(ConvergenceError):
            gaussian.check_fock_truncation(squeezed)

    def test_zero_tolerance_raises(self) -> None:
        with pytest.raises(ConventionError):
            gaussian.check_fock_truncation(self._sqz(0.6, 60), tolerance=0.0)

    def test_nonpositive_window_raises(self) -> None:
        with pytest.raises(ConventionError):
            gaussian.check_fock_truncation(self._sqz(0.6, 60), window=0)

    def test_readout_guards_by_default(self) -> None:
        # gaussian_readout runs the guard by default (raises on a biased readout);
        # check_truncation=False returns the (biased) readout without raising.
        with pytest.raises(ConvergenceError):
            gaussian.gaussian_readout(self._sqz(1.2, 20))
        readout = gaussian.gaussian_readout(self._sqz(1.2, 20), check_truncation=False)
        assert readout.squeezing_parameter >= 0.0

    def test_no_false_positive_on_well_truncated_moderate_fock(self) -> None:
        # Regression: a *wide* window would over-flag well-resolved moderate-N
        # states by summing bulk instead of the edge. The edge window (default 2,
        # ground state excluded) must stay silent for states whose leakage beyond
        # the cutoff is negligible — near-vacuum thermal, small coherent, vacuum.
        for state in (
            qutip.thermal_dm(6, 0.05),
            qutip.coherent(8, 0.3),
            qutip.basis(4, 0),
        ):
            assert gaussian.check_fock_truncation(state) == ()


class TestForcedDisplacement:
    """§26.4 / §7 — the SQ6 linear force seeds a convention-consistent displacement (additive)."""

    @staticmethod
    def _evolve_force(force_fn, tmax: float, fock: int = 40) -> qutip.Qobj:
        hilbert = _single_mode(fock, 2.0e6)
        const = waveforms.FrequencyWaveform(
            omega=lambda t: TWOPI * 2.0e6, d_ln_omega_dt=lambda t: 0.0
        )
        hamiltonian = nonadiabatic_squeezing_hamiltonian(
            hilbert, "m", const, validate_at=(0.0, tmax)
        ) + displacement_force_hamiltonian(hilbert, "m", force_fn)
        psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(fock, 0))
        result = solve(
            hilbert=hilbert,
            hamiltonian=hamiltonian,
            initial_state=psi0,
            times=np.linspace(0.0, tmax, 600),
            storage_mode=StorageMode.EAGER,
        )
        return gaussian.reduced_single_mode(result.states[-1], hilbert, "m")

    def test_force_seeds_displacement_along_minus_p(self) -> None:
        # Leading order α = −i f·t (§26.4/§7): a positive force displaces along −p̂
        # (negative imaginary part), with |α| ≈ f·t at short time.
        tmax = 0.02 * (1.0 / 2.0e6)  # ≪ trap period → free rotation negligible
        force = 1.0e7
        alpha = gaussian.gaussian_readout(
            self._evolve_force(lambda t: force, tmax)
        ).coherent_amplitude
        assert alpha.imag < 0.0
        assert abs(alpha.real) < 0.1 * abs(alpha.imag)  # essentially pure −p at short time
        assert abs(alpha) == pytest.approx(force * tmax, rel=0.05)

    @pytest.mark.parametrize("scale", [2.0, 4.0])
    def test_displacement_linear_in_force(self, scale: float) -> None:
        tmax = 0.02 * (1.0 / 2.0e6)
        base = 5.0e6
        a1 = abs(
            gaussian.gaussian_readout(self._evolve_force(lambda t: base, tmax)).coherent_amplitude
        )
        a2 = abs(
            gaussian.gaussian_readout(
                self._evolve_force(lambda t, s=scale: s * base, tmax)
            ).coherent_amplitude
        )
        assert a2 == pytest.approx(scale * a1, rel=1e-3)

    def test_displacement_lifts_parity(self) -> None:
        # The centred squeezing generator keeps P_odd = 0; the force fills the odd n.
        tmax = 6.0 * 0.02 * (1.0 / 2.0e6)
        pn = gaussian.phonon_number_distribution(self._evolve_force(lambda t: 2.0e7, tmax, fock=50))
        assert pn[1::2].sum() > 1e-2

    def test_backend_guard(self) -> None:
        hilbert = _single_mode(10, 2.0e6)
        with pytest.raises(ConventionError):
            displacement_force_hamiltonian(hilbert, "m", lambda t: 1.0e7, backend="jax")


class TestDownUpPulse:
    """§26.1 — the down/up pulse waveform (SQ6 single-pulse optimisation knob)."""

    def test_dips_and_returns_to_omega_ini(self) -> None:
        wave = waveforms.down_up_pulse(
            omega_ini=TWOPI * 2.0e6,
            omega_min=TWOPI * 1.0e6,
            ramp_width_s=1.0e-8,
            hold_s=2.0e-7,
            center_s=5.0e-7,
        )
        assert wave.omega(0.0) == pytest.approx(TWOPI * 2.0e6, rel=1e-6)  # starts at ω_ini
        assert wave.omega(1.0e-6) == pytest.approx(TWOPI * 2.0e6, rel=1e-6)  # returns to ω_ini
        assert wave.omega(5.0e-7) == pytest.approx(TWOPI * 1.0e6, rel=1e-3)  # dips to ω_min

    def test_analytic_derivative(self) -> None:
        wave = waveforms.down_up_pulse(
            omega_ini=TWOPI * 2.0e6,
            omega_min=TWOPI * 1.0e6,
            ramp_width_s=2.0e-8,
            hold_s=1.0e-7,
            center_s=5.0e-7,
        )
        t, h = 4.7e-7, 1.0e-12
        fd = (np.log(wave.omega(t + h)) - np.log(wave.omega(t - h))) / (2.0 * h)
        assert wave.d_ln_omega_dt(t) == pytest.approx(fd, rel=1e-4)

    def test_guards(self) -> None:
        with pytest.raises(ConventionError):
            waveforms.down_up_pulse(
                omega_ini=-1.0, omega_min=1.0e6, ramp_width_s=1e-8, hold_s=1e-7, center_s=5e-7
            )
        with pytest.raises(ConventionError):
            waveforms.down_up_pulse(
                omega_ini=2.0e6, omega_min=1.0e6, ramp_width_s=-1e-8, hold_s=1e-7, center_s=5e-7
            )
