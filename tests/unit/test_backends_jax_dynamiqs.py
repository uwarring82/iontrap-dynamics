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
from iontrap_dynamics.exceptions import ConventionError, ConvergenceError
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

    def test_convention_version_read_from_system(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        # The JAX path must honour whatever ``convention_version`` the
        # caller pinned on the IonSystem — not the library-current
        # CONVENTION_VERSION module constant. Same contract as the
        # QuTiP path (sequences.py reads from ``hilbert.system``).
        # For the default-pinned system these two values coincide.
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
        assert r.metadata.convention_version == hilbert.system.convention_version

    def test_convention_version_honours_archival_pin(
        self,
    ) -> None:
        # Regression against silent relabeling of archived results.
        # A user who archives an IonSystem with a pinned
        # ``convention_version="archive-vX.Y"`` and re-runs it on the
        # JAX backend must get that archival string in the result
        # metadata — not the library-current CONVENTION_VERSION.
        # Without this pin, cached results would round-trip under the
        # wrong convention label on the JAX path while the QuTiP path
        # does the right thing (see sequences.py line 446).
        from iontrap_dynamics.conventions import CONVENTION_VERSION
        from iontrap_dynamics.system import IonSystem

        pinned_version = "archive-vTEST.0.0"
        # Must differ from the library-current string so the test
        # actually catches a regression.
        assert pinned_version != CONVENTION_VERSION

        mode = ModeConfig(
            label="axial",
            frequency_rad_s=2 * np.pi * 1.5e6,
            eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
        )
        pinned_system = IonSystem(
            species_per_ion=(mg25_plus(),),
            modes=(mode,),
            convention_version=pinned_version,
        )
        hilbert = HilbertSpace(system=pinned_system, fock_truncations={"axial": 4})
        drive = DriveConfig(
            k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
            carrier_rabi_frequency_rad_s=2 * np.pi * 1e6,
            phase_rad=0.0,
        )
        from iontrap_dynamics.hamiltonians import carrier_hamiltonian

        ham = carrier_hamiltonian(hilbert, drive, ion_index=0)
        psi_0 = qutip.tensor(spin_down(), qutip.basis(4, 0))
        times = np.linspace(0.0, 1e-6, 20)

        r_jax = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=[spin_z(hilbert, 0)],
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        r_qutip = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=[spin_z(hilbert, 0)],
            storage_mode=StorageMode.OMITTED,
            backend="qutip",
        )
        # Both backends must report the archival version — parity check.
        assert r_jax.metadata.convention_version == pinned_version
        assert r_qutip.metadata.convention_version == pinned_version

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
# LAZY storage mode (Dispatch β.3): states_loader closure over the
# Dynamiqs JAX Array; per-index Qobj materialisation on demand.
# ---------------------------------------------------------------------------


class TestLazyStorage:
    def test_lazy_returns_loader_and_no_states_tuple(
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
            storage_mode=StorageMode.LAZY,
            backend="jax",
        )
        assert r.states is None
        assert r.states_loader is not None
        assert callable(r.states_loader)
        assert r.metadata.storage_mode is StorageMode.LAZY

    def test_lazy_loader_reproduces_initial_state(
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
            storage_mode=StorageMode.LAZY,
            backend="jax",
        )
        assert r.states_loader is not None
        first = r.states_loader(0)
        assert first.isket
        assert first.dims == psi_0.dims
        assert (first - psi_0).norm() < 1e-10

    def test_lazy_loader_matches_eager_states(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        # Bit-identical: LAZY is just EAGER with per-index materialisation
        # — no different integrator or conversion path. Compare both
        # results at a handful of time indices.
        hilbert, ham, psi_0, times, obs = carrier_fock12
        r_lazy = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.LAZY,
            backend="jax",
        )
        r_eager = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.EAGER,
            backend="jax",
        )
        assert r_lazy.states_loader is not None
        assert r_eager.states is not None
        for i in (0, 1, 50, 100, len(times) - 1):
            assert (r_lazy.states_loader(i) - r_eager.states[i]).norm() < 1e-12

    def test_lazy_loader_negative_index_mirrors_python(
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
            storage_mode=StorageMode.LAZY,
            backend="jax",
        )
        assert r.states_loader is not None
        last_neg = r.states_loader(-1)
        last_pos = r.states_loader(len(times) - 1)
        assert (last_neg - last_pos).norm() < 1e-12

    @pytest.mark.parametrize("bad_index", [200, 1000, -201, -10_000])
    def test_lazy_loader_out_of_bounds_raises(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
        bad_index: int,
    ) -> None:
        # JAX silently clamps out-of-range indexing; the loader must
        # raise IndexError instead (CONVENTIONS.md §15 "silent
        # degradation forbidden"). The trajectory length is 200 so
        # indices 200, 1000, -201, -10000 are all out of range.
        hilbert, ham, psi_0, times, obs = carrier_fock12
        r = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.LAZY,
            backend="jax",
        )
        assert r.states_loader is not None
        with pytest.raises(IndexError, match="out of range"):
            r.states_loader(bad_index)

    def test_lazy_with_density_matrix_initial(
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
            storage_mode=StorageMode.LAZY,
            backend="jax",
        )
        assert r.states_loader is not None
        first = r.states_loader(0)
        assert first.isoper
        assert first.dims == rho_0.dims
        # Trace preserved at t = 0 and at a mid-point.
        assert abs(first.tr() - 1.0) < 1e-10
        assert abs(r.states_loader(100).tr() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Scope boundary: direct QuTiP list-format time-dep rejected on JAX.
# (The builder's backend="jax" kwarg is the supported entry — see
# TestTimeDependentDetunedCarrier below.)
# ---------------------------------------------------------------------------


class TestNotImplementedScope:
    def test_qutip_list_format_rejected_on_jax_backend(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        hilbert, ham, psi_0, times, obs = carrier_fock12
        # Wrap as a QuTiP list to exercise the list-format guard. The
        # callable is never evaluated — NotImplementedError fires
        # before Dynamiqs is invoked.
        list_ham = [[ham, lambda t, _args: 1.0]]
        with pytest.raises(NotImplementedError, match="backend='jax'"):
            solve(
                hilbert=hilbert,
                hamiltonian=list_ham,
                initial_state=psi_0,
                times=times,
                observables=obs,
                storage_mode=StorageMode.OMITTED,
                backend="jax",
            )


# ---------------------------------------------------------------------------
# β.4.1 — detuned_carrier_hamiltonian with backend="jax" emits a
# Dynamiqs TimeQArray; dq.modulated(cos, A_φ) + dq.modulated(sin, A_⊥).
# Cross-backend numeric equivalence against the QuTiP time-dep list.
# ---------------------------------------------------------------------------


class TestTimeDependentDetunedCarrier:
    TOLERANCE = 1e-3  # same as β.2 cross-backend target

    @pytest.fixture
    def detuned_setup(
        self,
    ) -> tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list]:
        mode = ModeConfig(
            label="axial",
            frequency_rad_s=2 * np.pi * 1.5e6,
            eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
        )
        system = IonSystem.homogeneous(
            species=mg25_plus(), n_ions=1, modes=(mode,)
        )
        hilbert = HilbertSpace(system=system, fock_truncations={"axial": 12})
        drive = DriveConfig(
            k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
            carrier_rabi_frequency_rad_s=2 * np.pi * 1e6,
            detuning_rad_s=2 * np.pi * 0.5e6,
            phase_rad=0.0,
        )
        psi_0 = qutip.tensor(spin_down(), qutip.basis(12, 0))
        # 4 generalised-Rabi periods of a δ = Ω/2 detuned carrier
        # (Ω_gen = √(Ω² + δ²) ≈ 1.118 Ω → T ≈ 0.894 μs). 200 samples.
        times = np.linspace(0.0, 4e-6, 200)
        observables = [spin_z(hilbert, 0)]
        return hilbert, drive, psi_0, times, observables

    def test_jax_backend_returns_time_qarray(
        self,
        detuned_setup: tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list],
    ) -> None:
        from iontrap_dynamics.hamiltonians import detuned_carrier_hamiltonian

        hilbert, drive, *_ = detuned_setup
        H_jax = detuned_carrier_hamiltonian(
            hilbert, drive, ion_index=0, backend="jax"
        )
        # Exact class is a Dynamiqs internal (SummedTimeQArray); what
        # matters is it's not a QuTiP list, and solve(backend="jax")
        # accepts it (covered by cross-backend test below).
        assert not isinstance(H_jax, list)
        # Duck-typed TimeQArray: must be callable at a time point and
        # return something matrix-shaped.
        sample = H_jax(0.0)
        assert sample is not None

    def test_cross_backend_expectation_equivalence(
        self,
        detuned_setup: tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list],
    ) -> None:
        from iontrap_dynamics.hamiltonians import detuned_carrier_hamiltonian

        hilbert, drive, psi_0, times, obs = detuned_setup
        H_qutip = detuned_carrier_hamiltonian(
            hilbert, drive, ion_index=0, backend="qutip"
        )
        H_jax = detuned_carrier_hamiltonian(
            hilbert, drive, ion_index=0, backend="jax"
        )
        r_qutip = solve(
            hilbert=hilbert,
            hamiltonian=H_qutip,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="qutip",
        )
        r_jax = solve(
            hilbert=hilbert,
            hamiltonian=H_jax,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        for obs_label in r_qutip.expectations:
            delta = np.max(
                np.abs(
                    r_qutip.expectations[obs_label]
                    - r_jax.expectations[obs_label]
                )
            )
            assert delta < self.TOLERANCE, (
                f"observable {obs_label!r}: cross-backend disagreement "
                f"{delta:.2e} exceeds tolerance {self.TOLERANCE:.0e}"
            )

    def test_backend_name_tag(
        self,
        detuned_setup: tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list],
    ) -> None:
        # β.4 results tag identically to β.2 (single "jax-dynamiqs"
        # string across time-independent and time-dep paths — design
        # note §6 Q2 default).
        from iontrap_dynamics.hamiltonians import detuned_carrier_hamiltonian

        hilbert, drive, psi_0, times, obs = detuned_setup
        H_jax = detuned_carrier_hamiltonian(
            hilbert, drive, ion_index=0, backend="jax"
        )
        r = solve(
            hilbert=hilbert,
            hamiltonian=H_jax,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        assert r.metadata.backend_name == "jax-dynamiqs"

    def test_lazy_storage_works_with_time_dependent_hamiltonian(
        self,
        detuned_setup: tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list],
    ) -> None:
        # Regression: β.3's LAZY loader must work for time-dep
        # results too (the loader is storage-layer, orthogonal to
        # the Hamiltonian shape).
        from iontrap_dynamics.hamiltonians import detuned_carrier_hamiltonian

        hilbert, drive, psi_0, times, obs = detuned_setup
        H_jax = detuned_carrier_hamiltonian(
            hilbert, drive, ion_index=0, backend="jax"
        )
        r = solve(
            hilbert=hilbert,
            hamiltonian=H_jax,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.LAZY,
            backend="jax",
        )
        assert r.states is None
        assert r.states_loader is not None
        first = r.states_loader(0)
        assert first.isket
        assert (first - psi_0).norm() < 1e-10


# ---------------------------------------------------------------------------
# β.4.2 — detuned red / blue sideband Hamiltonians with backend="jax".
# Both share the _list_format_cos_sin QuTiP helper; β.4.2 routes
# backend="jax" through the sibling timeqarray_cos_sin helper in
# backends/jax/_coefficients.py. Parametrized over (red, blue) — the
# two builders produce structurally identical forms (H_static +
# H_quadrature with cos/sin envelopes), differing only in the
# spin-motion coupling pattern (σ_+ a for red, σ_+ a† for blue).
# ---------------------------------------------------------------------------


class TestTimeDependentDetunedSideband:
    TOLERANCE = 1e-3  # same as β.4.1

    @pytest.fixture
    def sideband_setup(
        self,
    ) -> tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list, float]:
        mode = ModeConfig(
            label="axial",
            frequency_rad_s=2 * np.pi * 1.5e6,
            eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
        )
        system = IonSystem.homogeneous(
            species=mg25_plus(), n_ions=1, modes=(mode,)
        )
        hilbert = HilbertSpace(system=system, fock_truncations={"axial": 8})
        drive = DriveConfig(
            k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
            carrier_rabi_frequency_rad_s=2 * np.pi * 1e6,
            phase_rad=0.0,
        )
        # Fock=1 so the red sideband has somewhere to lower into and
        # the blue sideband isn't degenerate with vacuum.
        psi_0 = qutip.tensor(spin_down(), qutip.basis(8, 1))
        times = np.linspace(0.0, 4e-6, 200)
        observables = [spin_z(hilbert, 0), number(hilbert, "axial")]
        detuning = 2 * np.pi * 0.3e6  # 0.3 MHz off-resonance
        return hilbert, drive, psi_0, times, observables, detuning

    @pytest.mark.parametrize(
        ("builder_name",),
        [
            ("detuned_red_sideband_hamiltonian",),
            ("detuned_blue_sideband_hamiltonian",),
        ],
    )
    def test_jax_backend_returns_time_qarray(
        self,
        sideband_setup,
        builder_name: str,
    ) -> None:
        import iontrap_dynamics.hamiltonians as h_mod

        builder = getattr(h_mod, builder_name)
        hilbert, drive, *_, detuning = sideband_setup
        H_jax = builder(
            hilbert,
            drive,
            "axial",
            ion_index=0,
            detuning_rad_s=detuning,
            backend="jax",
        )
        assert not isinstance(H_jax, list)
        # Duck-check: TimeQArray is callable at a time point.
        sample = H_jax(0.0)
        assert sample is not None

    @pytest.mark.parametrize(
        ("builder_name",),
        [
            ("detuned_red_sideband_hamiltonian",),
            ("detuned_blue_sideband_hamiltonian",),
        ],
    )
    def test_cross_backend_expectation_equivalence(
        self,
        sideband_setup,
        builder_name: str,
    ) -> None:
        import iontrap_dynamics.hamiltonians as h_mod

        builder = getattr(h_mod, builder_name)
        hilbert, drive, psi_0, times, obs, detuning = sideband_setup
        H_qutip = builder(
            hilbert,
            drive,
            "axial",
            ion_index=0,
            detuning_rad_s=detuning,
            backend="qutip",
        )
        H_jax = builder(
            hilbert,
            drive,
            "axial",
            ion_index=0,
            detuning_rad_s=detuning,
            backend="jax",
        )
        r_qutip = solve(
            hilbert=hilbert,
            hamiltonian=H_qutip,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="qutip",
        )
        r_jax = solve(
            hilbert=hilbert,
            hamiltonian=H_jax,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        for obs_label in r_qutip.expectations:
            delta = np.max(
                np.abs(
                    r_qutip.expectations[obs_label]
                    - r_jax.expectations[obs_label]
                )
            )
            assert delta < self.TOLERANCE, (
                f"{builder_name} {obs_label!r}: cross-backend "
                f"disagreement {delta:.2e} exceeds tolerance "
                f"{self.TOLERANCE:.0e}"
            )


# ---------------------------------------------------------------------------
# β.4.3 — detuned Mølmer–Sørensen gate (two ions, single mode) with
# backend="jax". Same cos/sin structural form as the single-ion
# builders; the A_X / A_P operator pair is different but the
# timeqarray_cos_sin assembly helper is shared.
# ---------------------------------------------------------------------------


class TestTimeDependentDetunedMSGate:
    TOLERANCE = 1e-3

    @pytest.fixture
    def ms_gate_setup(
        self,
    ) -> tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list, float]:
        com = ModeConfig(
            label="com",
            frequency_rad_s=2 * np.pi * 1.5e6,
            eigenvector_per_ion=np.array(
                [
                    [0.0, 0.0, 1.0 / np.sqrt(2.0)],
                    [0.0, 0.0, 1.0 / np.sqrt(2.0)],
                ]
            ),
        )
        system = IonSystem.homogeneous(
            species=mg25_plus(), n_ions=2, modes=(com,)
        )
        hilbert = HilbertSpace(system=system, fock_truncations={"com": 8})
        drive = DriveConfig(
            k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
            carrier_rabi_frequency_rad_s=2 * np.pi * 0.1e6,
            phase_rad=0.0,
        )
        # Bell-gate parameters derived from analytic helpers. One loop
        # closes in t_gate ≈ 27 μs at δ/2π ≈ 36.85 kHz for the COM-
        # mode on ²⁵Mg⁺ with Ω/2π = 100 kHz.
        from iontrap_dynamics.analytic import (
            lamb_dicke_parameter,
            ms_gate_closing_detuning,
        )

        eta = lamb_dicke_parameter(
            k_vec=[0.0, 0.0, 2 * np.pi / 280e-9],
            mode_eigenvector=[0.0, 0.0, 1.0 / np.sqrt(2.0)],
            ion_mass=mg25_plus().mass_kg,
            mode_frequency=com.frequency_rad_s,
        )
        detuning = ms_gate_closing_detuning(
            carrier_rabi_frequency=2 * np.pi * 0.1e6,
            lamb_dicke_parameter=eta,
            loops=1,
        )
        # Quarter of the gate-closing time — enough to see dynamics
        # and cheap to integrate on both backends (dim = 4 × 8 = 32).
        psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(8, 0))
        times = np.linspace(0.0, 0.25 * 27e-6, 100)
        observables = [
            spin_z(hilbert, 0),
            spin_z(hilbert, 1),
            number(hilbert, "com"),
        ]
        return hilbert, drive, psi_0, times, observables, detuning

    def test_jax_backend_returns_time_qarray(
        self,
        ms_gate_setup: tuple[
            HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list, float
        ],
    ) -> None:
        from iontrap_dynamics.hamiltonians import detuned_ms_gate_hamiltonian

        hilbert, drive, *_, detuning = ms_gate_setup
        H_jax = detuned_ms_gate_hamiltonian(
            hilbert,
            drive,
            "com",
            ion_indices=(0, 1),
            detuning_rad_s=detuning,
            backend="jax",
        )
        assert not isinstance(H_jax, list)
        assert H_jax(0.0) is not None

    def test_cross_backend_expectation_equivalence(
        self,
        ms_gate_setup: tuple[
            HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list, float
        ],
    ) -> None:
        from iontrap_dynamics.hamiltonians import detuned_ms_gate_hamiltonian

        hilbert, drive, psi_0, times, obs, detuning = ms_gate_setup
        H_qutip = detuned_ms_gate_hamiltonian(
            hilbert,
            drive,
            "com",
            ion_indices=(0, 1),
            detuning_rad_s=detuning,
            backend="qutip",
        )
        H_jax = detuned_ms_gate_hamiltonian(
            hilbert,
            drive,
            "com",
            ion_indices=(0, 1),
            detuning_rad_s=detuning,
            backend="jax",
        )
        r_qutip = solve(
            hilbert=hilbert,
            hamiltonian=H_qutip,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="qutip",
        )
        r_jax = solve(
            hilbert=hilbert,
            hamiltonian=H_jax,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        for obs_label in r_qutip.expectations:
            delta = np.max(
                np.abs(
                    r_qutip.expectations[obs_label]
                    - r_jax.expectations[obs_label]
                )
            )
            assert delta < self.TOLERANCE, (
                f"MS gate {obs_label!r}: cross-backend disagreement "
                f"{delta:.2e} exceeds tolerance {self.TOLERANCE:.0e}"
            )


# ---------------------------------------------------------------------------
# β.4.4 — modulated_carrier_hamiltonian with a user-supplied
# envelope_jax kwarg. Unlike the four structured detuning builders,
# the modulated carrier wraps an arbitrary user envelope; the library
# can't auto-translate scipy-traced callables to JAX, so the caller
# supplies both envelope= (QuTiP path) and envelope_jax= (JAX path).
# ---------------------------------------------------------------------------


class TestModulatedCarrierUserEnvelope:
    TOLERANCE = 1e-3

    @pytest.fixture
    def modulated_setup(
        self,
    ) -> tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list]:
        mode = ModeConfig(
            label="axial",
            frequency_rad_s=2 * np.pi * 1.5e6,
            eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
        )
        system = IonSystem.homogeneous(
            species=mg25_plus(), n_ions=1, modes=(mode,)
        )
        hilbert = HilbertSpace(system=system, fock_truncations={"axial": 4})
        drive = DriveConfig(
            k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
            carrier_rabi_frequency_rad_s=2 * np.pi * 1e6,
            detuning_rad_s=0.0,  # modulated carrier requires on-resonance
            phase_rad=0.0,
        )
        psi_0 = qutip.tensor(spin_down(), qutip.basis(4, 0))
        times = np.linspace(0.0, 1e-6, 200)
        observables = [spin_z(hilbert, 0)]
        return hilbert, drive, psi_0, times, observables

    def test_missing_envelope_jax_raises_convention_error(
        self,
        modulated_setup: tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list],
    ) -> None:
        from iontrap_dynamics.hamiltonians import modulated_carrier_hamiltonian

        hilbert, drive, *_ = modulated_setup
        with pytest.raises(ConventionError, match="envelope_jax"):
            modulated_carrier_hamiltonian(
                hilbert,
                drive,
                ion_index=0,
                envelope=lambda t: 1.0,
                backend="jax",
            )

    def test_jax_backend_returns_modulated_time_qarray(
        self,
        modulated_setup: tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list],
    ) -> None:
        # β.4.4 returns a single-piece ModulatedTimeQArray (not a
        # SummedTimeQArray like the detuning builders) because the
        # modulated carrier has only one Hermitian piece × envelope.
        import jax.numpy as jnp

        from iontrap_dynamics.hamiltonians import modulated_carrier_hamiltonian

        hilbert, drive, *_ = modulated_setup
        H_jax = modulated_carrier_hamiltonian(
            hilbert,
            drive,
            ion_index=0,
            envelope=lambda t: 1.0,
            envelope_jax=lambda t: jnp.asarray(1.0),
            backend="jax",
        )
        assert not isinstance(H_jax, list)
        assert H_jax(0.0) is not None

    def test_cross_backend_expectation_equivalence_gaussian_envelope(
        self,
        modulated_setup: tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list],
    ) -> None:
        # Canonical Gaussian-envelope pulse. Both callables compute
        # the same function; the numpy version is scipy-traceable,
        # the jnp version is JAX-traceable. Cross-backend agreement
        # should be under the library-default 1e-3 bound.
        import jax.numpy as jnp

        from iontrap_dynamics.hamiltonians import modulated_carrier_hamiltonian

        hilbert, drive, psi_0, times, obs = modulated_setup
        t0 = 0.5e-6
        sigma = 0.1e-6

        def env_np(t: float) -> float:
            return float(np.exp(-0.5 * ((t - t0) / sigma) ** 2))

        def env_jax(t: float) -> object:
            return jnp.exp(-0.5 * ((t - t0) / sigma) ** 2)

        H_qutip = modulated_carrier_hamiltonian(
            hilbert,
            drive,
            ion_index=0,
            envelope=env_np,
            backend="qutip",
        )
        H_jax = modulated_carrier_hamiltonian(
            hilbert,
            drive,
            ion_index=0,
            envelope=env_np,  # unused on JAX path but required by the signature
            envelope_jax=env_jax,
            backend="jax",
        )
        r_qutip = solve(
            hilbert=hilbert,
            hamiltonian=H_qutip,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="qutip",
        )
        r_jax = solve(
            hilbert=hilbert,
            hamiltonian=H_jax,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        for obs_label in r_qutip.expectations:
            delta = np.max(
                np.abs(
                    r_qutip.expectations[obs_label]
                    - r_jax.expectations[obs_label]
                )
            )
            assert delta < self.TOLERANCE, (
                f"modulated carrier {obs_label!r}: cross-backend "
                f"disagreement {delta:.2e} exceeds tolerance "
                f"{self.TOLERANCE:.0e}"
            )

    def test_constant_envelope_reproduces_static_carrier(
        self,
        modulated_setup: tuple[HilbertSpace, DriveConfig, qutip.Qobj, np.ndarray, list],
    ) -> None:
        # envelope_jax(t) = 1 should produce dynamics equivalent to
        # the static carrier on the JAX path (modulo library-default
        # integrator tolerances).
        import jax.numpy as jnp

        from iontrap_dynamics.hamiltonians import (
            carrier_hamiltonian,
            modulated_carrier_hamiltonian,
        )

        hilbert, drive, psi_0, times, obs = modulated_setup

        H_mod = modulated_carrier_hamiltonian(
            hilbert,
            drive,
            ion_index=0,
            envelope=lambda t: 1.0,
            envelope_jax=lambda t: jnp.asarray(1.0),
            backend="jax",
        )
        H_static = carrier_hamiltonian(hilbert, drive, ion_index=0)
        r_mod = solve(
            hilbert=hilbert,
            hamiltonian=H_mod,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        r_static = solve(
            hilbert=hilbert,
            hamiltonian=H_static,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
        )
        for obs_label in r_mod.expectations:
            delta = np.max(
                np.abs(
                    r_mod.expectations[obs_label]
                    - r_static.expectations[obs_label]
                )
            )
            assert delta < self.TOLERANCE


# ---------------------------------------------------------------------------
# solve_ensemble(..., backend="jax") — real execution coverage. The
# kwarg-validation tests in test_backends_jax.py exercise the
# dispatch surface; this test confirms that the full loop through
# joblib + solve_via_jax + dq.sesolve actually produces the expected
# TrajectoryResult tuple.
# ---------------------------------------------------------------------------


class TestSolveEnsembleOnJaxBackend:
    def test_serial_ensemble_returns_per_trial_results(
        self,
        carrier_fock12: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray, list],
    ) -> None:
        # Three trials at slightly different Rabi frequencies —
        # enough to distinguish trajectories so the test actually
        # exercises the per-Hamiltonian dispatch, not just the
        # kwarg-forwarding path. n_jobs=1 avoids joblib's
        # process-spawn overhead and keeps the test deterministic
        # and self-contained on CPU.
        from iontrap_dynamics.hamiltonians import carrier_hamiltonian
        from iontrap_dynamics.sequences import solve_ensemble

        hilbert, _, psi_0, times, obs = carrier_fock12
        rabis = [2 * np.pi * rate for rate in (0.9e6, 1.0e6, 1.1e6)]
        hamiltonians = tuple(
            carrier_hamiltonian(
                hilbert,
                DriveConfig(
                    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
                    carrier_rabi_frequency_rad_s=omega,
                    phase_rad=0.0,
                ),
                ion_index=0,
            )
            for omega in rabis
        )
        results = solve_ensemble(
            hilbert=hilbert,
            hamiltonians=hamiltonians,
            initial_state=psi_0,
            times=times,
            observables=obs,
            storage_mode=StorageMode.OMITTED,
            backend="jax",
            n_jobs=1,
        )
        assert len(results) == len(rabis)
        for r in results:
            assert isinstance(r, TrajectoryResult)
            assert r.metadata.backend_name == "jax-dynamiqs"
        # Different Rabi frequencies produce observably different
        # σ_z trajectories — confirms each trial actually integrated
        # its own Hamiltonian rather than sharing a cached result.
        sz = [r.expectations["sigma_z_0"] for r in results]
        assert float(np.max(np.abs(sz[0] - sz[1]))) > 1e-3
        assert float(np.max(np.abs(sz[1] - sz[2]))) > 1e-3


# ---------------------------------------------------------------------------
# β.4.1 / β.4.2 / β.4.3 shared helper: timeqarray_cos_sin.
# ---------------------------------------------------------------------------


class TestTimeQArrayCosSin:
    def test_returns_non_list_time_qarray(self) -> None:
        from iontrap_dynamics.backends.jax._coefficients import (
            timeqarray_cos_sin,
        )

        # Two trivial operators; the helper just needs QArrayLikes.
        h_static = qutip.sigmax()
        h_quadrature = qutip.sigmay()
        result = timeqarray_cos_sin(h_static, h_quadrature, 1.0e6)
        assert not isinstance(result, list)
        # Callable at a time point.
        assert result(0.0) is not None


# ---------------------------------------------------------------------------
# β.4.1 coefficient-callable factory unit tests.
# ---------------------------------------------------------------------------


class TestCoefficientFactories:
    def test_cos_detuning_evaluates_to_jnp_cos(self) -> None:
        from iontrap_dynamics.backends.jax._coefficients import cos_detuning_jax

        delta = 2 * np.pi * 1e6  # 1 MHz detuning
        coeff = cos_detuning_jax(delta)
        assert callable(coeff)
        # At t = 0 → cos(0) = 1.
        assert float(coeff(0.0)) == pytest.approx(1.0, abs=1e-14)
        # At t = π / δ → cos(π) = -1.
        assert float(coeff(np.pi / delta)) == pytest.approx(-1.0, abs=1e-10)

    def test_sin_detuning_evaluates_to_jnp_sin(self) -> None:
        from iontrap_dynamics.backends.jax._coefficients import sin_detuning_jax

        delta = 2 * np.pi * 1e6
        coeff = sin_detuning_jax(delta)
        assert callable(coeff)
        assert float(coeff(0.0)) == pytest.approx(0.0, abs=1e-14)
        # sin(π/2) at t = (π/2) / δ → 1.
        assert float(coeff(0.5 * np.pi / delta)) == pytest.approx(1.0, abs=1e-10)

    def test_closure_captures_delta_by_value(self) -> None:
        # Regression: the factory must snapshot delta so the caller
        # can mutate their own local binding without perturbing the
        # returned closure. JAX traces the closure when Dynamiqs
        # builds the TimeQArray; late-bound delta would silently
        # produce wrong dynamics.
        from iontrap_dynamics.backends.jax._coefficients import cos_detuning_jax

        delta = 1.0e6
        coeff = cos_detuning_jax(delta)
        delta = 999.0  # mutate caller's reference
        # Closure still uses 1e6: cos(1e6 * 1e-6) = cos(1.0) ≈ 0.5403.
        assert float(coeff(1e-6)) == pytest.approx(np.cos(1.0), abs=1e-10)
        del delta  # appease linters



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
