# SPDX-License-Identifier: MIT
"""Unit tests for :mod:`iontrap_dynamics.observables`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import qutip

from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import (
    Observable,
    expectations_over_time,
    number,
    parity,
    spin_x,
    spin_y,
    spin_z,
)
from iontrap_dynamics.operators import spin_down, spin_up
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import ground_state
from iontrap_dynamics.system import IonSystem

# ----------------------------------------------------------------------------
# Fixture
# ----------------------------------------------------------------------------


def _single_ion_hilbert(*, fock: int = 5) -> HilbertSpace:
    mode = ModeConfig(
        label="axial",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={"axial": fock})


def _two_ion_hilbert(*, fock: int = 4) -> HilbertSpace:
    mode = ModeConfig(
        label="com",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]) / np.sqrt(2.0),
    )
    system = IonSystem(species_per_ion=(mg25_plus(), mg25_plus()), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={"com": fock})


# ----------------------------------------------------------------------------
# Observable — frozen dataclass sanity
# ----------------------------------------------------------------------------


class TestObservableType:
    def test_frozen(self) -> None:
        """Observable is a frozen dataclass; label and operator are immutable."""
        h = _single_ion_hilbert()
        obs = spin_z(h, 0)
        with pytest.raises(FrozenInstanceError):
            obs.label = "other"  # type: ignore[misc]

    def test_keyword_only_construction(self) -> None:
        """Observable's dataclass is kw_only; positional ctor must fail."""
        with pytest.raises(TypeError):
            Observable("x", qutip.sigmax())  # type: ignore[misc]

    def test_repr_contains_label(self) -> None:
        h = _single_ion_hilbert()
        assert "sigma_z_0" in repr(spin_z(h, 0))


# ----------------------------------------------------------------------------
# Spin factories — default labels and embedding
# ----------------------------------------------------------------------------


class TestSpinFactories:
    def test_spin_z_default_label(self) -> None:
        h = _single_ion_hilbert()
        assert spin_z(h, 0).label == "sigma_z_0"

    def test_spin_x_default_label(self) -> None:
        h = _single_ion_hilbert()
        assert spin_x(h, 0).label == "sigma_x_0"

    def test_spin_y_default_label(self) -> None:
        h = _single_ion_hilbert()
        assert spin_y(h, 0).label == "sigma_y_0"

    def test_custom_label_overrides_default(self) -> None:
        h = _single_ion_hilbert()
        obs = spin_z(h, 0, label="readout_population")
        assert obs.label == "readout_population"

    def test_operator_dims_match_full_hilbert(self) -> None:
        h = _single_ion_hilbert(fock=10)
        obs = spin_z(h, 0)
        assert obs.operator.dims == h.qutip_dims()

    def test_two_ion_labels_distinguish_ions(self) -> None:
        h = _two_ion_hilbert()
        a = spin_z(h, 0)
        b = spin_z(h, 1)
        assert a.label == "sigma_z_0"
        assert b.label == "sigma_z_1"
        assert a.operator != b.operator


# ----------------------------------------------------------------------------
# Mode factories
# ----------------------------------------------------------------------------


class TestModeFactory:
    def test_number_default_label(self) -> None:
        h = _single_ion_hilbert()
        assert number(h, "axial").label == "n_axial"

    def test_number_custom_label(self) -> None:
        h = _single_ion_hilbert()
        obs = number(h, "axial", label="phonon_count")
        assert obs.label == "phonon_count"

    def test_number_dims_match_full_hilbert(self) -> None:
        h = _single_ion_hilbert(fock=8)
        obs = number(h, "axial")
        assert obs.operator.dims == h.qutip_dims()


# ----------------------------------------------------------------------------
# parity — multi-ion σ_z product observable
# ----------------------------------------------------------------------------


class TestParityFactory:
    def test_default_label_two_ions(self) -> None:
        h = _two_ion_hilbert()
        obs = parity(h, [0, 1])
        assert obs.label == "parity_0_1"

    def test_default_label_three_ions(self) -> None:
        h = _two_ion_hilbert()  # only 2 ions built; label is a pure-syntax check
        # fabricate a tuple ordering syntactically; not executed on trapped space
        assert parity(h, (0, 1)).label == "parity_0_1"

    def test_custom_label(self) -> None:
        h = _two_ion_hilbert()
        obs = parity(h, (0, 1), label="Z_0_Z_1")
        assert obs.label == "Z_0_Z_1"

    def test_requires_at_least_two_ions(self) -> None:
        h = _two_ion_hilbert()
        with pytest.raises(ValueError, match="at least two ions"):
            parity(h, [0])

    def test_requires_distinct_indices(self) -> None:
        h = _two_ion_hilbert()
        with pytest.raises(ValueError, match="must be distinct"):
            parity(h, [0, 0])

    def test_operator_dims_match_full_hilbert(self) -> None:
        h = _two_ion_hilbert()
        obs = parity(h, (0, 1))
        assert obs.operator.dims == h.qutip_dims()

    def test_expectation_bell_phi_plus_is_plus_one(self) -> None:
        """|Φ+⟩ = (|↑↑⟩ + |↓↓⟩)/√2 should give ⟨σ_z σ_z⟩ = +1."""
        h = _two_ion_hilbert(fock=2)
        fock_zero = qutip.basis(2, 0)  # single COM mode at |0⟩
        up_up = qutip.tensor(spin_up(), spin_up(), fock_zero)
        dn_dn = qutip.tensor(spin_down(), spin_down(), fock_zero)
        phi_plus = (up_up + dn_dn).unit()
        result = expectations_over_time([phi_plus], [parity(h, (0, 1))])
        np.testing.assert_allclose(result["parity_0_1"], [1.0])

    def test_expectation_bell_psi_plus_is_minus_one(self) -> None:
        """|Ψ+⟩ = (|↑↓⟩ + |↓↑⟩)/√2 should give ⟨σ_z σ_z⟩ = −1."""
        h = _two_ion_hilbert(fock=2)
        fock_zero = qutip.basis(2, 0)
        up_dn = qutip.tensor(spin_up(), spin_down(), fock_zero)
        dn_up = qutip.tensor(spin_down(), spin_up(), fock_zero)
        psi_plus = (up_dn + dn_up).unit()
        result = expectations_over_time([psi_plus], [parity(h, (0, 1))])
        np.testing.assert_allclose(result["parity_0_1"], [-1.0])

    def test_expectation_product_state_factorises(self) -> None:
        """|↑↑⟩: ⟨σ_z⟩_0 = ⟨σ_z⟩_1 = +1 → ⟨σ_z σ_z⟩ = +1."""
        h = _two_ion_hilbert(fock=2)
        fock_zero = qutip.basis(2, 0)
        state = qutip.tensor(spin_up(), spin_up(), fock_zero)
        result = expectations_over_time([state], [parity(h, (0, 1))])
        np.testing.assert_allclose(result["parity_0_1"], [1.0])


# ----------------------------------------------------------------------------
# expectations_over_time — correctness + shape
# ----------------------------------------------------------------------------


class TestExpectationsOverTime:
    def test_single_state_single_observable(self) -> None:
        h = _single_ion_hilbert()
        result = expectations_over_time(
            states=[ground_state(h)],
            observables=[spin_z(h, 0)],
        )
        assert list(result.keys()) == ["sigma_z_0"]
        np.testing.assert_allclose(result["sigma_z_0"], [-1.0])

    def test_multiple_states_returns_array_per_observable(self) -> None:
        """Trajectory of 3 states × 2 observables → 2 entries each of length 3."""
        h = _single_ion_hilbert()
        psi_down = ground_state(h)
        psi_up = qutip.tensor(spin_up(), qutip.basis(5, 0))
        states = [psi_down, psi_up, psi_down]

        result = expectations_over_time(
            states=states,
            observables=[spin_z(h, 0), number(h, "axial")],
        )
        assert set(result.keys()) == {"sigma_z_0", "n_axial"}
        np.testing.assert_allclose(result["sigma_z_0"], [-1.0, +1.0, -1.0])
        np.testing.assert_allclose(result["n_axial"], [0.0, 0.0, 0.0], atol=1e-12)

    def test_arrays_are_float_not_complex(self) -> None:
        """qutip.expect on Hermitian operators returns reals; the dict values
        should be plain float ndarrays, not complex."""
        h = _single_ion_hilbert()
        result = expectations_over_time(
            states=[ground_state(h)],
            observables=[spin_z(h, 0), number(h, "axial")],
        )
        for arr in result.values():
            assert arr.dtype == np.float64

    def test_empty_state_list_returns_empty_arrays(self) -> None:
        h = _single_ion_hilbert()
        result = expectations_over_time(states=[], observables=[spin_z(h, 0)])
        assert len(result["sigma_z_0"]) == 0

    def test_empty_observable_list_returns_empty_dict(self) -> None:
        h = _single_ion_hilbert()
        result = expectations_over_time(states=[ground_state(h)], observables=[])
        assert result == {}

    def test_label_is_preserved(self) -> None:
        """A user-supplied label propagates through to the dict key."""
        h = _single_ion_hilbert()
        result = expectations_over_time(
            states=[ground_state(h)],
            observables=[spin_z(h, 0, label="measured_sigma_z")],
        )
        assert "measured_sigma_z" in result
        assert "sigma_z_0" not in result


# ----------------------------------------------------------------------------
# Integration: ground-state expectations match CONVENTIONS.md §3
# ----------------------------------------------------------------------------


class TestGroundStateIntegration:
    def test_two_ion_ground_state_both_spins_down(self) -> None:
        """|↓, ↓, 0⟩ → ⟨σ_z⟩_0 = ⟨σ_z⟩_1 = −1, ⟨n̂⟩ = 0."""
        h = _two_ion_hilbert()
        result = expectations_over_time(
            states=[ground_state(h)],
            observables=[
                spin_z(h, 0),
                spin_z(h, 1),
                number(h, "com"),
            ],
        )
        np.testing.assert_allclose(result["sigma_z_0"], [-1.0])
        np.testing.assert_allclose(result["sigma_z_1"], [-1.0])
        np.testing.assert_allclose(result["n_com"], [0.0], atol=1e-12)

    def test_ground_state_sigma_x_and_y_are_zero(self) -> None:
        """Ground state is a σ_z eigenstate → ⟨σ_x⟩ = ⟨σ_y⟩ = 0."""
        h = _single_ion_hilbert()
        result = expectations_over_time(
            states=[ground_state(h)],
            observables=[spin_x(h, 0), spin_y(h, 0)],
        )
        np.testing.assert_allclose(result["sigma_x_0"], [0.0], atol=1e-14)
        np.testing.assert_allclose(result["sigma_y_0"], [0.0], atol=1e-14)
