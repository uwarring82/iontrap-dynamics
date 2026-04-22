# SPDX-License-Identifier: MIT
"""Tests for the JAX-backend dispatch plumbing (Dispatch β.1 skeleton).

These tests cover the dispatch surface — the ``backend=`` kwarg on
:func:`iontrap_dynamics.sequences.solve`, its interaction with the
existing ``solver=`` kwarg, and the availability-check path through
the :mod:`iontrap_dynamics.backends.jax` subpackage. They do **not**
exercise any JAX or Dynamiqs code: Dispatch β.1 ships only the
skeleton, and the tests here are correspondingly scoped.

When β.2 wires the Dynamiqs integrator, these tests stay useful as
the dispatch-plumbing regression surface; new tests are added in a
``test_backends_jax_dynamiqs.py`` (or similar) sibling that covers
the integrator numerics.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics.backends.jax import _core as jax_core
from iontrap_dynamics.backends.jax import solve_via_jax
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.exceptions import BackendError, ConventionError
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.results import StorageMode, TrajectoryResult
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

# ---------------------------------------------------------------------------
# Shared fixtures — a minimal carrier Rabi setup that exercises the solve()
# entry point without caring about physics correctness (these are dispatch
# tests, not numerics tests).
# ---------------------------------------------------------------------------


@pytest.fixture
def carrier_setup() -> tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray]:
    mode = ModeConfig(
        label="axial",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"axial": 4})
    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
        carrier_rabi_frequency_rad_s=2 * np.pi * 1e6,
        phase_rad=0.0,
    )
    hamiltonian = carrier_hamiltonian(hilbert, drive, ion_index=0)
    psi_0 = qutip.tensor(spin_down(), qutip.basis(4, 0))
    times = np.linspace(0.0, 1e-6, 20)
    return hilbert, hamiltonian, psi_0, times


# ---------------------------------------------------------------------------
# Default behaviour — backend="qutip" is the default and must not change
# existing user-visible behaviour.
# ---------------------------------------------------------------------------


class TestDefaultBackendIsQutip:
    def test_default_backend_produces_trajectory_result(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
    ) -> None:
        hilbert, ham, psi_0, times = carrier_setup
        result = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            storage_mode=StorageMode.OMITTED,
        )
        assert isinstance(result, TrajectoryResult)
        # Default backend kwarg routes through QuTiP; backend_name records
        # the specific solver QuTiP picked.
        assert result.metadata.backend_name in {"qutip-sesolve", "qutip-mesolve"}

    def test_explicit_qutip_backend_matches_default(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
    ) -> None:
        hilbert, ham, psi_0, times = carrier_setup
        default = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            storage_mode=StorageMode.OMITTED,
        )
        explicit = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            storage_mode=StorageMode.OMITTED,
            backend="qutip",
        )
        assert default.metadata.backend_name == explicit.metadata.backend_name
        for key in default.expectations:
            np.testing.assert_array_equal(default.expectations[key], explicit.expectations[key])


# ---------------------------------------------------------------------------
# Unknown-backend rejection — exercises the first line of _validate_backend.
# ---------------------------------------------------------------------------


class TestUnknownBackendRejected:
    def test_unknown_backend_raises_convention_error(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
    ) -> None:
        hilbert, ham, psi_0, times = carrier_setup
        with pytest.raises(ConventionError, match="unknown backend"):
            solve(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
                backend="numpy",  # type: ignore[arg-type]
            )

    def test_error_message_names_valid_backends(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
    ) -> None:
        hilbert, ham, psi_0, times = carrier_setup
        with pytest.raises(ConventionError) as excinfo:
            solve(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
                backend="banana",
            )
        msg = str(excinfo.value)
        assert "'qutip'" in msg
        assert "'jax'" in msg


# ---------------------------------------------------------------------------
# solver= / backend= compatibility — documented in §4.1 of the JAX design
# note: solver= stays QuTiP-specific; explicit sesolve/mesolve with
# backend="jax" raises ConventionError.
# ---------------------------------------------------------------------------


class TestSolverBackendCompatibility:
    @pytest.mark.parametrize("explicit_solver", ["sesolve", "mesolve"])
    def test_explicit_qutip_solver_with_jax_backend_rejected(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
        explicit_solver: str,
    ) -> None:
        hilbert, ham, psi_0, times = carrier_setup
        with pytest.raises(ConventionError, match="QuTiP-specific"):
            solve(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
                backend="jax",
                solver=explicit_solver,
            )

    def test_unknown_solver_with_jax_backend_rejected(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
    ) -> None:
        # Unknown solver string → "unknown solver" message, *not* the
        # "QuTiP-specific" message. Vocabulary check precedes the
        # backend-compatibility check so the error matches the actual
        # failure mode.
        hilbert, ham, psi_0, times = carrier_setup
        with pytest.raises(ConventionError, match="unknown solver"):
            solve(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
                backend="jax",
                solver="banana",
            )

    def test_auto_solver_with_jax_backend_passes_validation(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # With backend="jax" and solver="auto", validation must pass and
        # the call proceeds into the JAX entry. Without the extras
        # installed that entry raises BackendError; with extras but no
        # integrator it raises NotImplementedError. Either way the call
        # must get *past* _validate_backend.
        hilbert, ham, psi_0, times = carrier_setup
        # Force the availability check to report unavailable so we
        # reliably hit BackendError regardless of local install.
        monkeypatch.setattr(jax_core, "_is_jax_available", lambda: False)
        with pytest.raises(BackendError):
            solve(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
                backend="jax",
                solver="auto",
            )

    @pytest.mark.parametrize("explicit_solver", ["sesolve", "mesolve", "auto"])
    def test_explicit_qutip_solver_with_qutip_backend_still_works(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
        explicit_solver: str,
    ) -> None:
        # Regression: the backend validation must not interfere with the
        # existing QuTiP solver= contract.
        hilbert, ham, psi_0, times = carrier_setup
        result = solve(
            hilbert=hilbert,
            hamiltonian=ham,
            initial_state=psi_0,
            times=times,
            backend="qutip",
            solver=explicit_solver,
        )
        assert isinstance(result, TrajectoryResult)


# ---------------------------------------------------------------------------
# JAX availability & stub — the skeleton's actual dispatch behaviour.
# ---------------------------------------------------------------------------


class TestJaxAvailabilityAndStub:
    def test_install_hint_raised_when_extras_missing(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hilbert, ham, psi_0, times = carrier_setup
        monkeypatch.setattr(jax_core, "_is_jax_available", lambda: False)
        with pytest.raises(BackendError) as excinfo:
            solve_via_jax(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
            )
        msg = str(excinfo.value)
        assert "iontrap-dynamics[jax]" in msg
        assert "JAX" in msg or "Dynamiqs" in msg

    def test_install_hint_text_lists_install_command(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hilbert, ham, psi_0, times = carrier_setup
        monkeypatch.setattr(jax_core, "_is_jax_available", lambda: False)
        with pytest.raises(BackendError) as excinfo:
            solve_via_jax(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
            )
        assert "pip install iontrap-dynamics[jax]" in str(excinfo.value)

    def test_beta2_stub_raised_when_extras_present(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Direct-call path: the stub surfaces the NotImplementedError
        # when the availability check reports extras present. Signature
        # was locked in β.1, so all required kwargs must be supplied.
        hilbert, ham, psi_0, times = carrier_setup
        monkeypatch.setattr(jax_core, "_is_jax_available", lambda: True)
        with pytest.raises(NotImplementedError) as excinfo:
            solve_via_jax(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
            )
        msg = str(excinfo.value)
        assert "Dispatch β.2" in msg or "β.2" in msg

    def test_beta2_stub_reachable_through_sequences_solve(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # End-to-end regression for the dispatch wiring. Going through
        # sequences.solve with backend="jax" and mocked availability
        # must reach the same NotImplementedError surface as a direct
        # solve_via_jax call — confirms the kwarg-forwarding in
        # sequences.py matches solve_via_jax's locked signature.
        hilbert, ham, psi_0, times = carrier_setup
        monkeypatch.setattr(jax_core, "_is_jax_available", lambda: True)
        with pytest.raises(NotImplementedError):
            solve(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
                backend="jax",
            )

    def test_availability_check_returns_bool(self) -> None:
        # Real call — whatever the local environment has, the predicate
        # must return a bool without raising.
        result = jax_core._is_jax_available()
        assert isinstance(result, bool)

    def test_solve_via_jax_exported_from_subpackage(self) -> None:
        # Stable internal API: sequences.solve imports solve_via_jax
        # from iontrap_dynamics.backends.jax.
        from iontrap_dynamics.backends.jax import solve_via_jax as imported

        assert imported is jax_core.solve_via_jax


# ---------------------------------------------------------------------------
# solve_ensemble — backend kwarg propagates through the batch entry.
# ---------------------------------------------------------------------------


class TestDispatchOrdering:
    """Lock in that the JAX dispatch fires *before* QuTiP-path guards.

    The QuTiP path rejects ``StorageMode.LAZY`` because ``qutip.mesolve``
    materialises all states eagerly (see `sequences.solve` docstring).
    The JAX path has its own storage-mode semantics (per the design
    note §2: LAZY → ``states_loader`` stays JAX-lazy until the caller
    fetches an index) and must not inherit the QuTiP-only restriction.
    Verifying the ordering with a regression test now — before β.2
    introduces real JAX behaviour that would mask an ordering bug.
    """

    def test_lazy_storage_does_not_raise_on_jax_dispatch(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hilbert, ham, psi_0, times = carrier_setup
        # Force availability to True so the call reaches the β.2 stub.
        # The assertion we care about: the error is NotImplementedError
        # (β.2 stub) or BackendError (if extras missing), not the
        # ConventionError QuTiP raises on LAZY.
        monkeypatch.setattr(jax_core, "_is_jax_available", lambda: True)
        with pytest.raises(NotImplementedError):
            solve(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
                storage_mode=StorageMode.LAZY,
                backend="jax",
            )

    def test_lazy_storage_still_rejected_on_qutip_backend(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
    ) -> None:
        # The LAZY restriction on the QuTiP backend must persist —
        # the dispatch-ordering fix on the JAX side is not a general
        # LAZY-support change.
        hilbert, ham, psi_0, times = carrier_setup
        with pytest.raises(ConventionError, match="LAZY"):
            solve(
                hilbert=hilbert,
                hamiltonian=ham,
                initial_state=psi_0,
                times=times,
                storage_mode=StorageMode.LAZY,
                backend="qutip",
            )


class TestSolveEnsembleForwardsBackend:
    def test_ensemble_rejects_unknown_backend(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
    ) -> None:
        from iontrap_dynamics.sequences import solve_ensemble

        hilbert, ham, psi_0, times = carrier_setup
        with pytest.raises(ConventionError, match="unknown backend"):
            solve_ensemble(
                hilbert=hilbert,
                hamiltonians=(ham,),
                initial_state=psi_0,
                times=times,
                backend="numpy",  # type: ignore[arg-type]
            )

    def test_ensemble_rejects_jax_with_explicit_qutip_solver(
        self,
        carrier_setup: tuple[HilbertSpace, qutip.Qobj, qutip.Qobj, np.ndarray],
    ) -> None:
        from iontrap_dynamics.sequences import solve_ensemble

        hilbert, ham, psi_0, times = carrier_setup
        with pytest.raises(ConventionError, match="QuTiP-specific"):
            solve_ensemble(
                hilbert=hilbert,
                hamiltonians=(ham,),
                initial_state=psi_0,
                times=times,
                backend="jax",
                solver="sesolve",
            )
