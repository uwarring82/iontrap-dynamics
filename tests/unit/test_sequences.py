# SPDX-License-Identifier: MIT
"""Unit + integration tests for :mod:`iontrap_dynamics.sequences`.

The dispatcher is thin (it wraps mesolve and assembles a
TrajectoryResult), so these tests double as end-to-end integration
tests of the full Phase 1 stack: species → drives → modes → system →
hilbert → states → hamiltonians → observables → sequences.solve.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics.cache import compute_request_hash
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.exceptions import ConventionError, IonTrapError
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import number, spin_x, spin_y, spin_z
from iontrap_dynamics.results import StorageMode, TrajectoryResult
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

    def test_warnings_is_empty_in_v01(self) -> None:
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
