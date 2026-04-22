# SPDX-License-Identifier: MIT
"""Integration tests for the Dynamiqs JAX backend (Dispatch β.2).

These tests exercise the real :func:`iontrap_dynamics.backends.jax.
solve_via_jax` path with Dynamiqs installed. They are gated on
``pytest.importorskip("dynamiqs")`` so the base CI environment (no
``[jax]`` extras) skips them cleanly.

The dispatch-plumbing tests in :mod:`test_backends_jax` cover the
install-agnostic surface (``backend=`` kwarg validation, stub
behaviour under mocked availability, solver/backend compatibility).
This file is specifically the numeric / integration surface that
needs a real Dynamiqs install.

β.2 scope (mirrors ``docs/phase-2-jax-backend-design.md`` §7):

- Cross-backend expectation equivalence between QuTiP and JAX at
  library-default integrator tolerances. Target: disagreement
  bounded by 1e-3 over 4 Rabi periods at dim 24. Empirically
  (Dynamiqs 0.3.4 + QuTiP 5.2.3 at default Tsit5 + default scipy
  tolerances) the measured disagreement is ~2e-5; the test
  asserts 1e-3 as a safe margin against integrator-version drift.
- Metadata tagging: ``backend_name="jax-dynamiqs"``,
  ``backend_version="dynamiqs-<ver>+jax-<ver>"``,
  ``convention_version`` inherited from the library.
- Storage modes: ``OMITTED`` computes expectations only (no state
  materialisation); ``EAGER`` returns a tuple of ``Qobj`` with
  the original tensor dims preserved; ``LAZY`` raises
  :class:`NotImplementedError` (β.3 scope).
- Ket (sesolve) and density-matrix (mesolve) dispatch paths.
- Fock-saturation check piggybacks top-level projectors onto
  Dynamiqs ``exp_ops`` so no state materialisation is needed —
  the same §15 warning ladder applies as on the QuTiP path.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

# Gate the entire module on Dynamiqs being importable.
dq = pytest.importorskip("dynamiqs")
jax = pytest.importorskip("jax")

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.exceptions import ConvergenceError
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import number, spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.results import StorageMode, TrajectoryResult
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem


# ---------------------------------------------------------------------------
# Shared fixture — carrier Rabi at Fock=12 (dim 24) for cross-backend tests.
# Parameters match the design-note §7 α.2 scope: carrier Hamiltonian, 4 Rabi
# periods, dim 24.
# ---------------------------------------------------------------------------


@pytest.fixture
def carrier_fock12() -> tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list]:
    mode = ModeConfig(
        label="axial",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"axial": 12})
    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
        carrier_rabi_frequency_rad_s=2 * np.pi * 1e6,
        phase_rad=0.0,
    )
    hamiltonian = carrier_hamiltonian(hilbert, drive, ion_index=0)
    psi_0 = qutip.tensor(spin_down(), qutip.basis(12, 0))
    # 4 Rabi periods × 50 steps/period = 200 time samples.
    rabi_period = 1e-6  # Ω = 2π × 1 MHz → T = 1 μs
    times = np.linspace(0.0, 4 * rabi_period, 200)
    observables = [spin_z(hilbert, 0), number(hilbert, "axial")]
    return hilbert, hamiltonian, psi_0, times, observables


# ---------------------------------------------------------------------------
# Cross-backend numeric equivalence — the Phase 2 exit criterion from
# workplan §5 Phase 2 ("matching the QuTiP reference within cross-platform
# tolerance").
# ---------------------------------------------------------------------------


class TestCrossBackendEquivalence:
    # Empirically 2–3e-5 at Dynamiqs 0.3.4 + QuTiP 5.2.3 defaults. 1e-3
    # is a conservative margin that survives integrator-version drift
    # without flaking; see docs/phase-2-jax-backend-design.md §7.
    TOLERANCE = 1e-3

    def test_sesolve_carrier_matches_qutip(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        hilbert, ham, psi_0, times, obs = carrier_fock12
        r_qutip = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="qutip",
        )
        r_jax = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        for obs_label in r_qutip.expectations:
            delta = np.max(
                np.abs(r_qutip.expectations[obs_label] - r_jax.expectations[obs_label])
            )
            assert delta < self.TOLERANCE, (
                f"observable {obs_label!r}: cross-backend disagreement "
                f"{delta:.2e} exceeds tolerance {self.TOLERANCE:.0e}"
            )

    def test_mesolve_path_matches_sesolve_path_on_pure_state(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        # Same physics (pure state evolution) routed through the two
        # Dynamiqs paths: sesolve via ket input, mesolve via DM input.
        # Both JAX paths must agree within library-default tolerances.
        hilbert, ham, psi_0, times, obs = carrier_fock12
        rho_0 = psi_0 * psi_0.dag()
        r_se = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        r_me = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=rho_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        for obs_label in r_se.expectations:
            delta = np.max(
                np.abs(r_se.expectations[obs_label] - r_me.expectations[obs_label])
            )
            assert delta < self.TOLERANCE


# ---------------------------------------------------------------------------
# Metadata surface — backend_name, backend_version, convention_version.
# ---------------------------------------------------------------------------


class TestResultMetadata:
    def test_backend_name_is_jax_dynamiqs(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        hilbert, ham, psi_0, times, obs = carrier_fock12
        r = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        assert r.metadata.backend_name == "jax-dynamiqs"

    def test_backend_version_embeds_dynamiqs_and_jax(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        hilbert, ham, psi_0, times, obs = carrier_fock12
        r = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        version = r.metadata.backend_version
        assert "dynamiqs-" in version
        assert "jax-" in version
        # Matches the schema-commitment note in
        # docs/phase-2-jax-backend-design.md §4.2.
        assert dq.__version__ in version
        assert jax.__version__ in version

    def test_convention_version_inherited(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        # The JAX path writes the library's CONVENTION_VERSION into
        # metadata — same contract as the QuTiP path. No per-backend
        # convention divergence (that would be a schema break).
        from iontrap_dynamics.conventions import CONVENTION_VERSION

        hilbert, ham, psi_0, times, obs = carrier_fock12
        r = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        assert r.metadata.convention_version == CONVENTION_VERSION

    def test_backend_name_override_honoured(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        hilbert, ham, psi_0, times, obs = carrier_fock12
        r = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
            backend_name="custom-tag",
        )
        assert r.metadata.backend_name == "custom-tag"


# ---------------------------------------------------------------------------
# Storage mode handling — OMITTED / EAGER honoured; LAZY rejected.
# ---------------------------------------------------------------------------


class TestStorageModes:
    def test_omitted_does_not_materialise_states(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        hilbert, ham, psi_0, times, obs = carrier_fock12
        r = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        assert r.states is None
        assert r.states_loader is None
        assert r.metadata.storage_mode is StorageMode.OMITTED

    def test_eager_round_trips_states_with_original_dims(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        hilbert, ham, psi_0, times, obs = carrier_fock12
        r = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.EAGER,
            backend="jax",
        )
        assert r.states is not None
        assert len(r.states) == len(times)
        # Original tensor dims ([[2, 12], [1]] for 1-ion × 12-Fock ket)
        # preserved across the JAX ↔ Qobj conversion.
        assert r.states[0].dims == psi_0.dims
        assert r.states[0].isket
        # First state is psi_0 to numerical precision.
        assert (r.states[0] - psi_0).norm() < 1e-10

    def test_eager_density_matrix_round_trip(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        hilbert, ham, psi_0, times, obs = carrier_fock12
        rho_0 = psi_0 * psi_0.dag()
        r = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=rho_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.EAGER,
            backend="jax",
        )
        assert r.states is not None
        assert r.states[0].isoper
        assert r.states[0].dims == rho_0.dims
        # Trace preserved in unitary evolution.
        assert abs(r.states[50].tr() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Fock-saturation check — the JAX path piggybacks top-level projectors
# onto Dynamiqs exp_ops and classifies via the shared classifier. Same
# §15 warning ladder as the QuTiP path.
# ---------------------------------------------------------------------------


class TestFockSaturation:
    def test_fock_saturation_raises_convergence_error_on_jax(self) -> None:
        # Construct a scenario that saturates the Fock truncation — a
        # large coherent drive on a mode with too few Fock levels. The
        # JAX path must raise ConvergenceError (Level 3) at the same
        # threshold as the QuTiP path does.
        mode = ModeConfig(
            label="axial",
            frequency_rad_s=2 * np.pi * 1.0e6,
            eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
        )
        system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
        # Tiny truncation — coherent state with displacement ~4 will
        # saturate easily.
        hilbert = HilbertSpace(system=system, fock_truncations={"axial": 4})
        drive = DriveConfig(
            k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
            carrier_rabi_frequency_rad_s=2 * np.pi * 1e6,
            phase_rad=0.0,
        )
        ham = carrier_hamiltonian(hilbert, drive, ion_index=0)
        psi_0 = qutip.tensor(spin_down(), qutip.coherent(4, 2.0))
        times = np.linspace(0.0, 2e-6, 50)
        with pytest.raises(ConvergenceError, match="Fock-truncation"):
            solve(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
                observables=[spin_z(hilbert, 0)],
                storage_mode=StorageMode.OMITTED,
                backend="jax",
            )


# ---------------------------------------------------------------------------
# Scope boundaries: time-dependent Hamiltonian and LAZY are β.3 scope.
# ---------------------------------------------------------------------------


class TestNotImplementedScope:
    def test_time_dependent_hamiltonian_rejected(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        hilbert, ham, psi_0, times, obs = carrier_fock12
        # Wrap as a QuTiP list to exercise the list-format guard. The
        # callable is never evaluated — NotImplementedError fires
        # before Dynamiqs is invoked.
        list_ham = [[ham, lambda t, _args: 1.0]]
        with pytest.raises(NotImplementedError, match="Dispatch β.3"):
            solve(
                hilbert=hilbert,
                hamiltonian=list_ham,
                initial_state=psi_0,
                times=times,
                observables=obs,
                storage_mode=StorageMode.OMITTED,
                backend="jax",
            )

    def test_lazy_storage_mode_rejected(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        hilbert, ham, psi_0, times, obs = carrier_fock12
        with pytest.raises(NotImplementedError, match="β.3"):
            solve(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
                observables=obs,
                storage_mode=StorageMode.LAZY,
                backend="jax",
            )


# ---------------------------------------------------------------------------
# Trajectory return type — smoke check that the JAX path produces the same
# frozen TrajectoryResult as the QuTiP path.
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_trajectoryresult(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        hilbert, ham, psi_0, times, obs = carrier_fock12
        r = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        assert isinstance(r, TrajectoryResult)
        # Expectations dict keyed by observable label.
        for obs_record in obs:
            assert obs_record.label in r.expectations
        # No user-requested warnings on this well-conditioned scenario.
        assert r.warnings == ()
