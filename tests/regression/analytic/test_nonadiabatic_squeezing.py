# SPDX-License-Identifier: MIT
"""Analytic regression oracle for the non-adiabatic-squeezing benchmark (SQ5).

WP-05 / Dispatch SQ5. Pins the benchmark-specific physics of Wittemer 2020 that
``tools/run_benchmark_nonadiabatic_squeezing.py`` reproduces (§26): the
**parametric** arm's linear-in-duration squeezing ``r = ½ δω · T_mod`` (so
``n̄_sq = sinh²(2π g T_mod)`` with ``g = δω/(4π)``), displacement-free; and the
**quench** arm's monotone, displacement-free squeezing growth with amplitude.
The sudden / cyclic-adiabatic §26.1 limits are pinned in
``tests/conventions/test_squeezing_conventions.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics import gaussian, waveforms
from iontrap_dynamics.hamiltonians import (
    displacement_force_hamiltonian,
    nonadiabatic_squeezing_hamiltonian,
)
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

pytestmark = pytest.mark.regression_analytic

TWOPI = 2.0 * np.pi
OMEGA_INI = TWOPI * 2.8e6  # ω_ini/2π = 2.8 MHz (paper single ion)
DELTA_OMEGA_MOD = TWOPI * 8.0e3  # δω/2π = 8 kHz (paper)
OMEGA_MOD = 2.0 * OMEGA_INI  # parametric resonance


def _single_mode(fock: int) -> HilbertSpace:
    mode = ModeConfig(
        label="m",
        frequency_rad_s=OMEGA_INI,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={"m": fock})


def _evolve_readout(
    hilbert: HilbertSpace, wave: waveforms.FrequencyWaveform, tmax: float, n_times: int
) -> gaussian.GaussianReadout:
    fock = hilbert.fock_truncations["m"]
    psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(fock, 0))
    hamiltonian = nonadiabatic_squeezing_hamiltonian(hilbert, "m", wave, validate_at=(0.0, tmax))
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=psi0,
        times=np.linspace(0.0, tmax, n_times),
        storage_mode=StorageMode.EAGER,
    )
    return gaussian.gaussian_readout(gaussian.reduced_single_mode(result.states[-1], hilbert, "m"))


class TestParametricArm:
    """Parametric modulation ω_mod = 2ω_ini → r = ½ δω T_mod, displacement-free."""

    @pytest.mark.parametrize("n_periods", [90, 180])
    def test_linear_squeezing_and_displacement_free(self, n_periods: int) -> None:
        hilbert = _single_mode(40)
        t_mod = n_periods * (TWOPI / OMEGA_MOD)
        wave = waveforms.sinusoidal_modulation(
            omega_ini=OMEGA_INI, mod_amplitude=DELTA_OMEGA_MOD, mod_frequency=OMEGA_MOD
        )
        readout = _evolve_readout(hilbert, wave, float(t_mod), max(2000, n_periods * 15))
        r_oracle = 0.5 * DELTA_OMEGA_MOD * t_mod  # r = ½ δω T_mod
        assert readout.squeezing_parameter == pytest.approx(r_oracle, rel=0.02)
        assert readout.mean_squeezed_occupation == pytest.approx(
            float(np.sinh(r_oracle) ** 2), rel=0.05
        )
        assert abs(readout.coherent_amplitude) ** 2 < 1e-9  # displacement-free (centred generator)

    def test_fitted_coupling_matches_rwa(self) -> None:
        # slope of r vs T_mod = ½ δω = 2π g ⇒ g_fit = δω/(4π) (leading-order RWA).
        hilbert = _single_mode(40)
        periods = np.array([60, 120, 180])
        t_mod = periods * (TWOPI / OMEGA_MOD)
        r_sim = np.array(
            [
                _evolve_readout(
                    hilbert,
                    waveforms.sinusoidal_modulation(
                        omega_ini=OMEGA_INI,
                        mod_amplitude=DELTA_OMEGA_MOD,
                        mod_frequency=OMEGA_MOD,
                    ),
                    float(t),
                    max(2000, int(p) * 15),
                ).squeezing_parameter
                for p, t in zip(periods, t_mod, strict=True)
            ]
        )
        g_fit = float(np.polyfit(t_mod, r_sim, 1)[0]) / TWOPI
        assert g_fit == pytest.approx(DELTA_OMEGA_MOD / (4.0 * np.pi), rel=0.02)


class TestQuenchArm:
    """Fast Gaussian quench → monotone, displacement-free squeezing growth with amplitude."""

    def test_monotone_and_displacement_free(self) -> None:
        hilbert = _single_mode(50)
        period = TWOPI / OMEGA_INI
        width = 0.02 * period
        center = 25.0 * width
        tmax = 55.0 * width
        n_sq = []
        for amplitude in (0.2, 0.4, 0.6):
            wave = waveforms.gaussian_quench(
                omega_ini=OMEGA_INI, amplitude=amplitude, width_s=width, center_s=center
            )
            readout = _evolve_readout(hilbert, wave, tmax, 1500)
            n_sq.append(readout.mean_squeezed_occupation)
            assert abs(readout.coherent_amplitude) ** 2 < 1e-9  # displacement-free
        assert n_sq[0] < n_sq[1] < n_sq[2]  # monotone in quench amplitude


class TestSinglePulseOptimisation:
    """Down/up pulse — r oscillates with the hold (WKB phase); optimum ≈ doubles a one-way ramp.

    Paper protocol: "iteratively adjust δτ and Δω to find maximal |r|". The down- and
    up-ramp squeezing contributions add constructively at hold ≈ ½ trap period and cancel
    at hold → 0 (adiabatic-cyclic, r → 0).
    """

    PERIOD = TWOPI / OMEGA_INI
    RAMP = 0.02 * PERIOD

    def _r_at_hold(self, hold: float) -> float:
        hilbert = _single_mode(50)
        center = 10.0 * self.RAMP + 0.5 * hold
        pulse = waveforms.down_up_pulse(
            omega_ini=OMEGA_INI,
            omega_min=0.5 * OMEGA_INI,
            ramp_width_s=self.RAMP,
            hold_s=hold,
            center_s=center,
        )
        tmax = center + 0.5 * hold + 15.0 * self.RAMP
        return _evolve_readout(hilbert, pulse, tmax, 1500).squeezing_parameter

    def test_constructive_hold_doubles_one_way_ramp(self) -> None:
        r_zero = self._r_at_hold(0.0)
        r_opt = self._r_at_hold(0.5 * self.PERIOD)
        r_one_way = 0.5 * abs(np.log(0.5))  # ½|ln(ω_min/ω_ini)| for a single ramp of this depth
        assert r_zero < 0.05  # cyclic hold → 0 leaves (almost) no squeezing
        assert r_opt > r_zero
        assert r_opt > 1.5 * r_one_way  # constructive optimum well past a one-way ramp
        assert r_opt == pytest.approx(2.0 * r_one_way, rel=0.15)  # ≈ doubling


class TestEcho:
    """Two-pulse purifying echo — displacement (∝ω) cancels while squeezing (∝2ω) adds.

    Each quench pulse carries a parasitic linear force. Separated by t_free ≈ ½ trap period
    (shifted by the finite pulse width), the second pulse's displacement cancels the first's
    (δp = n_dsp⁽¹⁾/n_dsp⁽²⁾ ≫ 1) while the squeezing accumulates.
    """

    PERIOD = TWOPI / OMEGA_INI
    AMP = -0.4  # a down-quench (Δω/ω_ini) — squeezes
    WIDTH = 0.02 * PERIOD
    FORCE_AMP = 2.0e7

    def _pulse(self, hilbert: HilbertSpace, centers: list[float], tmax: float):
        width, amp, force_amp = self.WIDTH, self.AMP, self.FORCE_AMP

        def bump(t: float) -> float:
            return sum(np.exp(-0.5 * ((t - c) / width) ** 2) for c in centers)

        def bump_prime(t: float) -> float:
            return sum(-(t - c) / width**2 * np.exp(-0.5 * ((t - c) / width) ** 2) for c in centers)

        wave = waveforms.FrequencyWaveform(
            omega=lambda t: OMEGA_INI * (1.0 + amp * bump(t)),
            d_ln_omega_dt=lambda t: amp * bump_prime(t) / (1.0 + amp * bump(t)),
        )

        def force(t: float) -> float:
            return force_amp * sum(np.exp(-0.5 * ((t - c) / width) ** 2) for c in centers)

        hamiltonian = nonadiabatic_squeezing_hamiltonian(
            hilbert, "m", wave, validate_at=(0.0, tmax)
        ) + displacement_force_hamiltonian(hilbert, "m", force)
        psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(50, 0))
        result = solve(
            hilbert=hilbert,
            hamiltonian=hamiltonian,
            initial_state=psi0,
            times=np.linspace(0.0, tmax, 2000),
            storage_mode=StorageMode.EAGER,
        )
        return gaussian.gaussian_readout(
            gaussian.reduced_single_mode(result.states[-1], hilbert, "m")
        )

    def test_echo_suppresses_displacement_and_adds_squeezing(self) -> None:
        hilbert = _single_mode(50)
        c1 = 20.0 * self.WIDTH
        one = self._pulse(hilbert, [c1], c1 + 20.0 * self.WIDTH)
        n_dsp_one = abs(one.coherent_amplitude) ** 2

        # scan t_free across the ≈½-period null (shifted off 0.50·T by the finite pulse width)
        best_delta_p, best_readout = 0.0, one
        for frac in (0.50, 0.515, 0.53, 0.545, 0.56):
            tf = frac * self.PERIOD
            two = self._pulse(hilbert, [c1, c1 + tf], c1 + tf + 20.0 * self.WIDTH)
            delta_p = n_dsp_one / (abs(two.coherent_amplitude) ** 2)
            if delta_p > best_delta_p:
                best_delta_p, best_readout = delta_p, two

        assert best_delta_p > 20.0  # displacement strongly suppressed at the null (benchmark ~1341)
        assert abs(best_readout.coherent_amplitude) < abs(one.coherent_amplitude)
        assert best_readout.squeezing_parameter > one.squeezing_parameter  # squeezing adds
