# SPDX-License-Identifier: MIT
"""Unit + integration tests for :mod:`iontrap_dynamics.sequences`.

The dispatcher is thin (it wraps mesolve and assembles a
TrajectoryResult), so these tests double as end-to-end integration
tests of the full Phase 1 stack: species → drives → modes → system →
hilbert → states → hamiltonians → observables → sequences.solve.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import qutip

from iontrap_dynamics.cache import compute_request_hash
from iontrap_dynamics.conventions import FOCK_CONVERGENCE_TOLERANCE
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.exceptions import (
    ConventionError,
    ConvergenceError,
    FockConvergenceWarning,
    FockQualityWarning,
    IonTrapError,
)
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import number, spin_x, spin_y, spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.results import StorageMode, TrajectoryResult, WarningSeverity
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import ground_state
from iontrap_dynamics.system import IonSystem

# ----------------------------------------------------------------------------
# Fixtures — a single-ion carrier scenario reused across tests.
# ----------------------------------------------------------------------------


def _carrier_scenario(
    *,
    rabi: float = 2 * np.pi * 1.0e6,
    fock: int = 3,
    n_steps: int = 100,
) -> tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, float]:
    """Return (hilbert, H, psi_0, times, rabi). Carrier at φ=0 on ion 0."""
    mode = ModeConfig(
        label="axial",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"axial": fock})

    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
        carrier_rabi_frequency_rad_s=rabi,
        phase_rad=0.0,
    )
    hamiltonian = carrier_hamiltonian(hilbert, drive, ion_index=0)
    psi_0 = ground_state(hilbert)

    rabi_period = 2 * np.pi / rabi
    times = np.linspace(0.0, rabi_period, n_steps)
    return hilbert, hamiltonian, psi_0, times, rabi


# ----------------------------------------------------------------------------
# Structural — returns a TrajectoryResult with the right shape
# ----------------------------------------------------------------------------


class TestReturnShape:
    def test_returns_trajectory_result(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        result = solve(
            hilbert=h,
            hamiltonian=H,
            initial_state=psi_0,
            times=times,
            observables=[],
        )
        assert isinstance(result, TrajectoryResult)

    def test_times_roundtrip(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        result = solve(
            hilbert=h,
            hamiltonian=H,
            initial_state=psi_0,
            times=times,
        )
        np.testing.assert_array_equal(result.times, times)

    def test_empty_observables_gives_empty_expectations(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        result = solve(hilbert=h, hamiltonian=H, initial_state=psi_0, times=times)
        assert result.expectations == {}

    def test_observables_propagate_to_expectations(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        result = solve(
            hilbert=h,
            hamiltonian=H,
            initial_state=psi_0,
            times=times,
            observables=[spin_z(h, 0), number(h, "axial")],
        )
        assert set(result.expectations.keys()) == {"sigma_z_0", "n_axial"}
        assert result.expectations["sigma_z_0"].shape == (len(times),)
        assert result.expectations["n_axial"].shape == (len(times),)


# ----------------------------------------------------------------------------
# Storage modes (CONVENTIONS.md §0.E)
# ----------------------------------------------------------------------------


class TestStorageModes:
    def test_omitted_default(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        result = solve(hilbert=h, hamiltonian=H, initial_state=psi_0, times=times)
        assert result.metadata.storage_mode is StorageMode.OMITTED
        assert result.states is None
        assert result.states_loader is None

    def test_eager_attaches_state_tuple(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario(n_steps=5)
        result = solve(
            hilbert=h,
            hamiltonian=H,
            initial_state=psi_0,
            times=times,
            storage_mode=StorageMode.EAGER,
        )
        assert result.metadata.storage_mode is StorageMode.EAGER
        assert result.states is not None
        assert len(result.states) == len(times)
        # First state should match the initial state up to numerical noise.
        diff = (
            (result.states[0] - qutip.ket2dm(psi_0))
            if result.states[0].isoper
            else (result.states[0] - psi_0)
        )
        assert diff.norm() < 1e-10

    def test_lazy_rejected(self) -> None:
        """StorageMode.LAZY is not supported from solve() in v0.1."""
        h, H, psi_0, times, _ = _carrier_scenario()
        with pytest.raises(ConventionError, match="LAZY"):
            solve(
                hilbert=h,
                hamiltonian=H,
                initial_state=psi_0,
                times=times,
                storage_mode=StorageMode.LAZY,
            )

    def test_lazy_rejection_is_iontraperror(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        with pytest.raises(IonTrapError):
            solve(
                hilbert=h,
                hamiltonian=H,
                initial_state=psi_0,
                times=times,
                storage_mode=StorageMode.LAZY,
            )


# ----------------------------------------------------------------------------
# Metadata — caller inputs propagate into the result's metadata
# ----------------------------------------------------------------------------


class TestMetadata:
    def test_convention_version_from_hilbert(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        result = solve(hilbert=h, hamiltonian=H, initial_state=psi_0, times=times)
        assert result.metadata.convention_version == h.system.convention_version

    def test_request_hash_propagates(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        hash_value = compute_request_hash({"scenario": "test"})
        result = solve(
            hilbert=h,
            hamiltonian=H,
            initial_state=psi_0,
            times=times,
            request_hash=hash_value,
        )
        assert result.metadata.request_hash == hash_value

    def test_backend_name_default(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        result = solve(hilbert=h, hamiltonian=H, initial_state=psi_0, times=times)
        assert result.metadata.backend_name == "qutip-mesolve"

    def test_backend_name_override(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        result = solve(
            hilbert=h,
            hamiltonian=H,
            initial_state=psi_0,
            times=times,
            backend_name="custom-backend",
        )
        assert result.metadata.backend_name == "custom-backend"

    def test_backend_version_is_qutip_version(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        result = solve(hilbert=h, hamiltonian=H, initial_state=psi_0, times=times)
        assert result.metadata.backend_version == qutip.__version__

    def test_fock_truncations_copied_from_hilbert(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario(fock=7)
        result = solve(hilbert=h, hamiltonian=H, initial_state=psi_0, times=times)
        assert dict(result.metadata.fock_truncations) == {"axial": 7}

    def test_provenance_tags_propagate(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        tags = ("phase1", "integration_test")
        result = solve(
            hilbert=h,
            hamiltonian=H,
            initial_state=psi_0,
            times=times,
            provenance_tags=tags,
        )
        assert result.metadata.provenance_tags == tags

    def test_warnings_is_empty_for_a_well_converged_carrier_run(self) -> None:
        """A carrier drive on |↓, 0⟩ populates only ``|n=0⟩``, so the
        Fock-saturation check (§13, §15) returns no warnings."""
        h, H, psi_0, times, _ = _carrier_scenario()
        result = solve(hilbert=h, hamiltonian=H, initial_state=psi_0, times=times)
        assert result.warnings == ()


# ----------------------------------------------------------------------------
# Integration — full pipeline reproduces carrier Rabi analytics
# ----------------------------------------------------------------------------


class TestCarrierIntegration:
    def test_half_pi_pulse_via_solve(self) -> None:
        """Configuration → solve → expectations: after t = π/(2Ω),
        ⟨σ_z⟩ = 0 (equal superposition)."""
        h, H, psi_0, _, rabi = _carrier_scenario()
        t_half_pi = np.pi / (2 * rabi)
        result = solve(
            hilbert=h,
            hamiltonian=H,
            initial_state=psi_0,
            times=np.array([0.0, t_half_pi]),
            observables=[spin_z(h, 0)],
        )
        np.testing.assert_allclose(result.expectations["sigma_z_0"], [-1.0, 0.0], atol=1e-6)

    def test_full_rabi_period_returns_to_ground(self) -> None:
        """After one full Rabi period, ⟨σ_z⟩ should return to −1."""
        h, H, psi_0, _, rabi = _carrier_scenario()
        t_full = 2 * np.pi / rabi
        result = solve(
            hilbert=h,
            hamiltonian=H,
            initial_state=psi_0,
            times=np.array([0.0, t_full]),
            observables=[spin_z(h, 0)],
        )
        np.testing.assert_allclose(result.expectations["sigma_z_0"], [-1.0, -1.0], atol=1e-6)

    def test_numerical_matches_analytic_formula(self) -> None:
        """Over two full Rabi periods the numerical ⟨σ_z⟩(t) tracks
        the analytic form −cos(Ωt) to ~1e-5 (same bound as the
        standalone tool's cross-check)."""
        h, H, psi_0, _, rabi = _carrier_scenario(n_steps=200)
        times = np.linspace(0.0, 2 * 2 * np.pi / rabi, 200)
        result = solve(
            hilbert=h,
            hamiltonian=H,
            initial_state=psi_0,
            times=times,
            observables=[spin_z(h, 0), spin_y(h, 0), spin_x(h, 0)],
        )
        analytic_sz = -np.cos(rabi * times)
        analytic_sy = np.sin(rabi * times)
        analytic_sx = np.zeros_like(times)
        np.testing.assert_allclose(result.expectations["sigma_z_0"], analytic_sz, atol=5e-5)
        np.testing.assert_allclose(result.expectations["sigma_y_0"], analytic_sy, atol=5e-5)
        np.testing.assert_allclose(result.expectations["sigma_x_0"], analytic_sx, atol=5e-5)


# ============================================================================
# Fock-truncation saturation warnings (CONVENTIONS.md §13, §15)
# ============================================================================
#
# The carrier Hamiltonian does not couple motion, so a Fock-diagonal
# initial state propagates with p_top exactly preserved across the
# trajectory. We construct deterministic mixed initial states at each
# target p_top and verify the solver classifies them into the correct
# level of the §15 ladder.


def _two_mode_hilbert(*, fock_axial: int = 4, fock_radial: int = 4) -> HilbertSpace:
    axial = ModeConfig(
        label="axial",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    radial = ModeConfig(
        label="radial",
        frequency_rad_s=2 * np.pi * 3.0e6,
        eigenvector_per_ion=np.array([[1.0, 0.0, 0.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(axial, radial))
    return HilbertSpace(
        system=system,
        fock_truncations={"axial": fock_axial, "radial": fock_radial},
    )


def _initial_state_with_top_fock(
    hilbert: HilbertSpace,
    mode_label: str,
    p_top: float,
) -> qutip.Qobj:
    """Return a density matrix with population ``p_top`` in the top Fock
    level of ``mode_label`` and ``1 − p_top`` in the vacuum of that mode.

    All other subsystems (spin, other modes) are in their vacuum / |↓⟩.
    The returned state lives on the full Hilbert space.
    """
    if not (0.0 <= p_top <= 1.0):
        raise ValueError(f"p_top must be in [0, 1]; got {p_top}")

    components = []
    for mode in hilbert.system.modes:
        n_fock = hilbert.fock_truncations[mode.label]
        if mode.label == mode_label:
            vac = qutip.basis(n_fock, 0)
            top = qutip.basis(n_fock, n_fock - 1)
        else:
            vac = qutip.basis(n_fock, 0)
            top = qutip.basis(n_fock, 0)
        components.append((vac, top))

    # Build full-space |↓⟩ ⊗ modes_vac and |↓⟩ ⊗ modes_top vectors
    psi_vac_mode_components = [spin_down()] + [c[0] for c in components]
    psi_top_mode_components = [spin_down()] + [c[1] for c in components]
    psi_vac = qutip.tensor(*psi_vac_mode_components)
    psi_top = qutip.tensor(*psi_top_mode_components)

    return (1.0 - p_top) * qutip.ket2dm(psi_vac) + p_top * qutip.ket2dm(psi_top)


def _saturation_scenario(
    p_top: float,
    *,
    target_mode: str = "axial",
    fock_axial: int = 4,
    fock_radial: int = 4,
) -> tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray]:
    """Build a carrier scenario with prescribed top-Fock saturation in
    ``target_mode``. Carrier does not couple motion, so ``p_top`` is
    preserved exactly across the trajectory."""
    hilbert = _two_mode_hilbert(fock_axial=fock_axial, fock_radial=fock_radial)
    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
        carrier_rabi_frequency_rad_s=2 * np.pi * 1.0e6,
        phase_rad=0.0,
    )
    hamiltonian = carrier_hamiltonian(hilbert, drive, ion_index=0)
    rho_0 = _initial_state_with_top_fock(hilbert, target_mode, p_top)
    times = np.linspace(0.0, 2 * np.pi / (2 * np.pi * 1.0e6), 20)
    return hilbert, hamiltonian, rho_0, times


class TestFockSaturationOK:
    """p_top < ε/10 → silent, no warnings."""

    def test_ground_state_emits_no_warnings(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail the test
            result = solve(hilbert=h, hamiltonian=H, initial_state=psi_0, times=times)
        assert result.warnings == ()

    def test_result_warnings_is_empty_tuple_when_converged(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        result = solve(hilbert=h, hamiltonian=H, initial_state=psi_0, times=times)
        assert result.warnings == ()

    def test_slightly_populated_but_below_tenth_epsilon_stays_silent(self) -> None:
        """p_top = ε/100 = 1e-6 is well below ε/10 — no warning."""
        h, H, rho_0, times = _saturation_scenario(1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = solve(hilbert=h, hamiltonian=H, initial_state=rho_0, times=times)
        assert result.warnings == ()


class TestFockSaturationLevel1:
    """ε/10 ≤ p_top < ε → FockConvergenceWarning."""

    def test_fires_fock_convergence_warning(self) -> None:
        h, H, rho_0, times = _saturation_scenario(5e-5)  # between 1e-5 and 1e-4
        with pytest.warns(FockConvergenceWarning, match="top-Fock population"):
            solve(hilbert=h, hamiltonian=H, initial_state=rho_0, times=times)

    def test_appends_result_warning_with_convergence_severity(self) -> None:
        h, H, rho_0, times = _saturation_scenario(5e-5)
        with pytest.warns(FockConvergenceWarning):
            result = solve(hilbert=h, hamiltonian=H, initial_state=rho_0, times=times)
        assert len(result.warnings) == 1
        record = result.warnings[0]
        assert record.severity is WarningSeverity.CONVERGENCE
        assert record.category == "fock_truncation"

    def test_warning_diagnostics_carry_mode_and_p_top(self) -> None:
        h, H, rho_0, times = _saturation_scenario(5e-5, target_mode="axial")
        with pytest.warns(FockConvergenceWarning):
            result = solve(hilbert=h, hamiltonian=H, initial_state=rho_0, times=times)
        diag = result.warnings[0].diagnostics
        assert diag["mode_label"] == "axial"
        assert diag["fock_dim"] == 4
        assert diag["p_top_max"] == pytest.approx(5e-5, rel=1e-6)
        assert diag["tolerance_epsilon"] == pytest.approx(FOCK_CONVERGENCE_TOLERANCE)


class TestFockSaturationLevel2:
    """ε ≤ p_top < 10ε → FockQualityWarning."""

    def test_fires_fock_quality_warning(self) -> None:
        h, H, rho_0, times = _saturation_scenario(5e-4)  # between 1e-4 and 1e-3
        with pytest.warns(FockQualityWarning, match="quality degraded"):
            solve(hilbert=h, hamiltonian=H, initial_state=rho_0, times=times)

    def test_appends_result_warning_with_quality_severity(self) -> None:
        h, H, rho_0, times = _saturation_scenario(5e-4)
        with pytest.warns(FockQualityWarning):
            result = solve(hilbert=h, hamiltonian=H, initial_state=rho_0, times=times)
        assert len(result.warnings) == 1
        assert result.warnings[0].severity is WarningSeverity.QUALITY


class TestFockSaturationLevel3:
    """p_top ≥ 10ε → ConvergenceError, no result returned."""

    def test_raises_convergence_error(self) -> None:
        h, H, rho_0, times = _saturation_scenario(5e-3)  # well above 1e-3
        with pytest.raises(ConvergenceError, match="Level 3"):
            solve(hilbert=h, hamiltonian=H, initial_state=rho_0, times=times)

    def test_convergence_error_subclass_iontraperror(self) -> None:
        h, H, rho_0, times = _saturation_scenario(5e-3)
        with pytest.raises(IonTrapError):
            solve(hilbert=h, hamiltonian=H, initial_state=rho_0, times=times)

    def test_error_message_names_mode_and_p_top(self) -> None:
        h, H, rho_0, times = _saturation_scenario(5e-3, target_mode="axial")
        with pytest.raises(ConvergenceError) as exc_info:
            solve(hilbert=h, hamiltonian=H, initial_state=rho_0, times=times)
        assert "axial" in str(exc_info.value)


class TestFockToleranceOverride:
    def test_looser_tolerance_can_silence_a_warning(self) -> None:
        """p_top = 5e-5 triggers Level 1 at default ε, but with a looser ε=1e-3
        the same p_top is < ε/10 → silent."""
        h, H, rho_0, times = _saturation_scenario(5e-5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = solve(
                hilbert=h,
                hamiltonian=H,
                initial_state=rho_0,
                times=times,
                fock_tolerance=1e-3,
            )
        assert result.warnings == ()

    def test_tighter_tolerance_can_escalate_a_silent_run_to_level_2(self) -> None:
        """p_top = 5e-7 is silent at default ε=1e-4 (< ε/10), but at tight
        ε=1e-7 it lies in [ε, 10ε) = [1e-7, 1e-6) → Level 2."""
        h, H, rho_0, times = _saturation_scenario(5e-7)
        with pytest.warns(FockQualityWarning):
            result = solve(
                hilbert=h,
                hamiltonian=H,
                initial_state=rho_0,
                times=times,
                fock_tolerance=1e-7,
            )
        assert result.warnings[0].severity is WarningSeverity.QUALITY

    def test_non_positive_tolerance_rejected(self) -> None:
        h, H, psi_0, times, _ = _carrier_scenario()
        with pytest.raises(ConventionError, match="positive"):
            solve(
                hilbert=h,
                hamiltonian=H,
                initial_state=psi_0,
                times=times,
                fock_tolerance=0.0,
            )
        with pytest.raises(ConventionError, match="positive"):
            solve(
                hilbert=h,
                hamiltonian=H,
                initial_state=psi_0,
                times=times,
                fock_tolerance=-1e-5,
            )


class TestFockSaturationMultiMode:
    """Each mode is classified independently; the strictest regime wins."""

    def test_per_mode_warnings_are_independent(self) -> None:
        """Saturate only ``axial`` — ``radial`` is silent."""
        h, H, rho_0, times = _saturation_scenario(5e-5, target_mode="axial")
        with pytest.warns(FockConvergenceWarning):
            result = solve(hilbert=h, hamiltonian=H, initial_state=rho_0, times=times)
        assert len(result.warnings) == 1
        assert result.warnings[0].diagnostics["mode_label"] == "axial"

    def test_level_3_on_any_mode_aborts_the_run(self) -> None:
        """radial saturated at Level 3 → ConvergenceError even though
        axial is fine."""
        h, H, rho_0, times = _saturation_scenario(5e-3, target_mode="radial")
        with pytest.raises(ConvergenceError, match="radial"):
            solve(hilbert=h, hamiltonian=H, initial_state=rho_0, times=times)
