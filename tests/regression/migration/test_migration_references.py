# SPDX-License-Identifier: MIT
"""Migration-regression tests (workplan §0.B tier 1).

Phase 0 only — retired after Phase 1 per workplan §0.B. The migration tier
compares Phase 1 builder output against frozen reference arrays generated
by the legacy ``qc.py`` script (tagged ``qc-legacy-v1.0``). The references
themselves live in ``tests/regression/migration/references/<scenario>/``
and are produced by ``tools/generate_migration_references.py``.

This file has two responsibilities:

1. **Reference-bundle validation** (runs today). Each reference directory
   must contain a well-formed ``metadata.json`` and an ``arrays.npz`` that
   match on shape and declared observable keys. Catches corruption or
   schema drift in the stored bundles.

2. **Builder comparison** (skipped, activates with Phase 1 builders). For
   each of the five canonical scenarios, a test compares Phase 1 builder
   output against the frozen reference at 10⁻¹⁰ tolerance per workplan
   §0.B item 5. Skipped until the relevant builder lands.

Migration is a check, not a truth criterion. The analytic and invariant
tiers remain the permanent physics anchors; a disagreement between a Phase
1 builder and the qc.py reference points at either a builder bug, a
legacy-script bug, or a convention difference — resolution requires
physics judgment, not blind regression.

Tolerance reality
-----------------

Workplan §0.B item 5 originally specified 10⁻¹⁰ "bit-identity" tolerance.
In practice the bit-identity goal is incompatible with the Phase 1 design:

- qc.py uses ``qutip.sigmaz`` (computational basis); iontrap-dynamics
  uses the atomic-physics convention (CONVENTIONS.md §3). This flips
  the sign of ⟨σ_z⟩ and ⟨σ_y⟩ on every reference. A **convention
  translator** is therefore required on the qc.py side before comparison
  (see :func:`_qc_to_iontrap_convention`).
- qc.py keeps the full Lamb–Dicke exponential in its Hamiltonian; the
  Phase 1 builders use the leading-order (LD-truncated) RWA form. Even
  at ``LD_regime=True`` in qc.py, residual differences of order
  ``η² · n̄`` remain.
- Some scenarios have a physics mismatch that is structural, not
  numerical — notably qc.py's scenario 3 drives a single-tone blue
  sideband rather than the symmetric bichromatic MS gate we build via
  :func:`ms_gate_hamiltonian` / :func:`detuned_ms_gate_hamiltonian`.

Scenario 1 therefore activates with a physical-level tolerance
(``atol = 5e-3`` on all convention-translated observables), rather than
``1e-10``. Scenarios 2–5 stay skipped with honest, specific blockers
(see each skip reason). The tier remains a check against accidental
convention drift; the analytic-regression tests are the permanent
physics anchor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import qutip

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import number, spin_x, spin_y, spin_z
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

pytestmark = pytest.mark.regression_migration

REFERENCES_DIR = Path(__file__).resolve().parent / "references"

CANONICAL_SCENARIOS = (
    "01_single_ion_carrier_thermal",
    "02_single_ion_red_sideband_fock1",
    "03_two_ion_ms_gate",
    "04_single_ion_stroboscopic_ac_halfpi",
    "05_single_mode_squeeze_displace",
)

# Which scenarios currently have reference bundles. Extend as new scenarios
# land in tools/generate_migration_references.py.
AVAILABLE_SCENARIOS = tuple(
    name
    for name in CANONICAL_SCENARIOS
    if (REFERENCES_DIR / name / "metadata.json").is_file()
    and (REFERENCES_DIR / name / "arrays.npz").is_file()
)

PENDING_SCENARIOS = tuple(name for name in CANONICAL_SCENARIOS if name not in AVAILABLE_SCENARIOS)


# ----------------------------------------------------------------------------
# Reference-bundle validation — runs today
# ----------------------------------------------------------------------------


class TestReferenceInventory:
    def test_references_directory_exists(self) -> None:
        assert REFERENCES_DIR.is_dir(), (
            f"missing directory {REFERENCES_DIR}; "
            "run tools/generate_migration_references.py to populate."
        )

    def test_at_least_one_scenario_available(self) -> None:
        """Migration-regression tier exists only if at least one reference
        bundle has been generated. Zero is a clear signal the user still
        needs to run the generator."""
        assert AVAILABLE_SCENARIOS, (
            "no reference bundles present under "
            f"{REFERENCES_DIR}. Run tools/generate_migration_references.py."
        )


@pytest.mark.parametrize(
    "scenario",
    AVAILABLE_SCENARIOS
    or [pytest.param("<none>", marks=pytest.mark.skip(reason="no references generated yet"))],
)
class TestReferenceBundle:
    """Per-scenario bundle validation: metadata schema, array consistency."""

    def _bundle_dir(self, scenario: str) -> Path:
        return REFERENCES_DIR / scenario

    def _load_metadata(self, scenario: str) -> dict:
        return json.loads((self._bundle_dir(scenario) / "metadata.json").read_text())

    def test_metadata_has_required_fields(self, scenario: str) -> None:
        meta = self._load_metadata(scenario)
        required = {
            "scenario_name",
            "scenario_index",
            "description",
            "qc_legacy_tag",
            "parameters",
            "parameters_hash",
            "environment",
            "generated_at",
            "schema_version",
            "observable_keys",
            "n_samples",
        }
        missing = required - meta.keys()
        assert not missing, f"metadata.json missing fields: {missing}"

    def test_metadata_scenario_name_matches_directory(self, scenario: str) -> None:
        meta = self._load_metadata(scenario)
        assert meta["scenario_name"] == scenario

    def test_metadata_schema_version(self, scenario: str) -> None:
        meta = self._load_metadata(scenario)
        assert meta["schema_version"] == 1, (
            "unknown reference schema version; migration-regression tier "
            "does not yet know how to compare against it."
        )

    def test_arrays_npz_loads(self, scenario: str) -> None:
        with np.load(self._bundle_dir(scenario) / "arrays.npz") as npz:
            assert len(npz.files) > 0

    def test_arrays_match_declared_observables(self, scenario: str) -> None:
        meta = self._load_metadata(scenario)
        with np.load(self._bundle_dir(scenario) / "arrays.npz") as npz:
            array_keys = set(npz.files)
        declared = set(meta["observable_keys"])
        assert array_keys == declared, (
            f"array keys {array_keys} do not match declared observables {declared}"
        )

    def test_arrays_have_consistent_length(self, scenario: str) -> None:
        meta = self._load_metadata(scenario)
        n = meta["n_samples"]
        with np.load(self._bundle_dir(scenario) / "arrays.npz") as npz:
            for key in npz.files:
                assert npz[key].shape[0] == n, (
                    f"{key} has {npz[key].shape[0]} samples, expected {n}"
                )

    def test_times_array_is_monotonic_non_decreasing(self, scenario: str) -> None:
        with np.load(self._bundle_dir(scenario) / "arrays.npz") as npz:
            times = npz["times"]
        assert np.all(np.diff(times) >= 0), "times must be monotonic non-decreasing"


# ----------------------------------------------------------------------------
# Convention translator — qc.py ↔ iontrap-dynamics
# ----------------------------------------------------------------------------
#
# qc.py uses qutip.sigmaz (|↓⟩ = +1 eigenstate); iontrap-dynamics follows
# the atomic-physics convention (|↓⟩ = −1 eigenstate). See CONVENTIONS.md
# §3 for the rationale. The translator is a plain sign flip on σ_z and
# σ_y; σ_x and scalar invariants pass through unchanged. Motional-frame
# translations (⟨X̂⟩, ⟨P̂⟩ rotations at ω_mode) would be additional
# per-scenario work — deferred along with scenarios 2–5.


def _qc_to_iontrap_convention(qc_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Translate a qc.py reference-array dict to iontrap-dynamics
    conventions (CONVENTIONS.md §3)."""
    sign_flip = {"sigma_y", "sigma_z", "sigma_y_A", "sigma_y_B", "sigma_z_A", "sigma_z_B"}
    return {
        key: (-np.asarray(value) if key in sign_flip else np.asarray(value))
        for key, value in qc_arrays.items()
    }


# ----------------------------------------------------------------------------
# Scenario 1 — on-resonance carrier on thermal motion, ACTIVE
# ----------------------------------------------------------------------------
#
# Matches the qc.py reference to atol = 5e-3 on σ_x, σ_y, σ_z, and n_mode
# after the convention flip. The residual is dominated by integrator
# differences and sub-leading LD corrections that qc.py retains but the
# Phase 1 carrier_hamiltonian drops; it is not "bit-identity" at 1e-10
# (see module docstring "Tolerance reality").


def _load_reference(scenario_name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    bundle = REFERENCES_DIR / scenario_name
    metadata = json.loads((bundle / "metadata.json").read_text())
    with np.load(bundle / "arrays.npz") as npz:
        arrays = {k: np.asarray(npz[k]) for k in npz.files}
    return metadata, arrays


def test_builder_matches_scenario_1() -> None:
    """Compare :func:`carrier_hamiltonian` + mesolve output against the
    qc.py reference for scenario 1 (single-ion carrier, thermal, δ = 0).

    Runs the full configuration → Hilbert → Hamiltonian → mesolve →
    expectations pipeline on parameters decoded from the reference
    metadata, applies the qc.py→iontrap convention translator to the
    reference arrays, then compares element-wise.
    """
    metadata, qc_arrays = _load_reference("01_single_ion_carrier_thermal")
    qc_translated = _qc_to_iontrap_convention(qc_arrays)

    p = metadata["parameters"]
    omega = p["Omega_over_2pi_MHz"] * 2 * np.pi * 1e6
    mode_freq = p["omega_mode_over_2pi_MHz"] * 2 * np.pi * 1e6
    n_thermal = p["n_thermal"]
    phi = p["phi_drive_rad"]

    # Fock truncation 12 is generous for n_th=0.5 (P(n=12) < 1e-7)
    n_fock = 12

    mode = ModeConfig(
        label="axial",
        frequency_rad_s=mode_freq,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"axial": n_fock})

    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
        carrier_rabi_frequency_rad_s=omega,
        phase_rad=phi,
    )
    H = carrier_hamiltonian(hilbert, drive, ion_index=0)

    # Thermal motion ⊗ |↓⟩ initial density matrix
    rho_spin = qutip.ket2dm(qutip.basis(2, 0))
    rho_motion = qutip.thermal_dm(n_fock, n_thermal)
    rho_0 = qutip.tensor(rho_spin, rho_motion)

    # qc.py times are in μs; convert to seconds
    tlist_s = qc_arrays["times"] * 1e-6

    obs = [
        spin_x(hilbert, 0).operator,
        spin_y(hilbert, 0).operator,
        spin_z(hilbert, 0).operator,
        number(hilbert, "axial").operator,
    ]
    result = qutip.mesolve(H, rho_0, tlist_s, [], obs)
    my_sx, my_sy, my_sz, my_n = result.expect

    atol = 5e-3
    np.testing.assert_allclose(my_sx, qc_translated["sigma_x"], atol=atol)
    np.testing.assert_allclose(my_sy, qc_translated["sigma_y"], atol=atol)
    np.testing.assert_allclose(my_sz, qc_translated["sigma_z"], atol=atol)
    np.testing.assert_allclose(my_n, qc_translated["n_mode"], atol=atol)


# ----------------------------------------------------------------------------
# Scenarios 2–5 — skipped with specific, honest blockers
# ----------------------------------------------------------------------------
#
# These scenarios have structural frame/physics differences that go beyond
# the simple convention flip scenario 1 uses. Each skip reason names the
# specific blocker rather than the generic "awaiting builder" placeholder
# (the builders now exist; the gap is frame/convention reconciliation).


@pytest.mark.skip(
    reason=(
        "qc.py keeps sub-leading Lamb–Dicke terms on the sideband transition "
        "that red_sideband_hamiltonian (leading-order RWA) does not; "
        "element-wise match at the convention-translator level is not "
        "achievable. Activation path: either compare only integrated "
        "invariants (σ_z parity, ⟨n̂⟩ peak amplitude) with a looser "
        "tolerance, or build a higher-order LD variant of the sideband "
        "Hamiltonian."
    )
)
def test_builder_matches_scenario_2() -> None:
    """Scenario 2 — single-ion red-sideband from Fock |1⟩."""
    raise AssertionError("unreachable — skipped")


@pytest.mark.skip(
    reason=(
        "qc.py drives a single-tone blue sideband on both spins; "
        "iontrap-dynamics' ms_gate_hamiltonian implements the symmetric "
        "bichromatic MS form. The two are physically distinct generators "
        "and will not match at any tolerance. Activation path: either "
        "add a single-tone two-ion-sideband builder, or retire this "
        "scenario in favour of a bichromatic reference generated from "
        "the Phase 1 MS builders themselves."
    )
)
def test_builder_matches_scenario_3() -> None:
    """Scenario 3 — two-ion MS-like gate via a single blue-sideband tone."""
    raise AssertionError("unreachable — skipped")


@pytest.mark.skip(
    reason=(
        "qc.py uses the full-exponential Lamb–Dicke operator "
        "``C = exp(iη(a+a†))`` and the lab-frame (non-RWA) Hamiltonian for "
        "its stroboscopic drive; modulated_carrier_hamiltonian applies the "
        "envelope to the leading-order RWA carrier. The Silveri-type "
        "spin-motion coupling that emerges from the full-exponential path "
        "is absent in our reduction. Activation path: extend the builder "
        "with a Lamb–Dicke-expansion order parameter or implement a "
        "dedicated frequency-modulated carrier builder."
    )
)
def test_builder_matches_scenario_4() -> None:
    """Scenario 4 — single-ion stroboscopic AC-π/2 drive."""
    raise AssertionError("unreachable — skipped")


@pytest.mark.skip(
    reason=(
        "Requires squeezing + displacement state-prep builders (workplan "
        "1.D), not yet landed in Phase 1. A ``squeeze`` / ``displace`` "
        "pair in :mod:`iontrap_dynamics.states` would unblock a clean "
        "activation at convention-translator tolerance, parallel to "
        "scenario 1."
    )
)
def test_builder_matches_scenario_5() -> None:
    """Scenario 5 — single-mode squeezing + displacement."""
    raise AssertionError("unreachable — skipped")
