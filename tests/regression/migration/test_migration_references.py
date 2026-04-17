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
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
# Builder comparison — skipped until Phase 1 builders land
# ----------------------------------------------------------------------------
#
# Each scenario has one comparison test. When the corresponding Phase 1
# builder is implemented, replace the skip body with a real build/solve +
# per-observable allclose against the reference at 10⁻¹⁰ tolerance.


def _activation_template_reason(scenario: str, blocking_builder: str) -> str:
    return (
        f"awaiting Phase 1 builder ({blocking_builder}) for {scenario}. "
        "Activation: invoke the builder with metadata['parameters'], compare "
        "per-observable against arrays.npz using np.testing.assert_allclose "
        "(atol=1e-10, same-platform bit-identity per workplan §0.B item 5)."
    )


@pytest.mark.skip(
    reason=_activation_template_reason(
        "01_single_ion_carrier_thermal", "red_sideband / carrier_flopping builder"
    )
)
def test_builder_matches_scenario_1() -> None:
    """Compare Phase 1 carrier-flopping builder output against the qc.py
    reference for scenario 1 (thermal, on-resonance)."""
    raise AssertionError("unreachable — skipped until builder lands")


@pytest.mark.skip(
    reason=_activation_template_reason("02_single_ion_red_sideband_fock1", "red_sideband builder")
)
def test_builder_matches_scenario_2() -> None:
    """Compare Phase 1 red-sideband builder output (Fock |1⟩ initial state)
    against the qc.py reference for scenario 2."""
    raise AssertionError("unreachable — skipped until builder lands")


@pytest.mark.skip(reason=_activation_template_reason("03_two_ion_ms_gate", "MS gate builder"))
def test_builder_matches_scenario_3() -> None:
    """Compare Phase 1 Mølmer–Sørensen gate builder output against the
    qc.py reference for scenario 3 (two ions, single mode)."""
    raise AssertionError("unreachable — skipped until builder lands")


@pytest.mark.skip(
    reason=_activation_template_reason(
        "04_single_ion_stroboscopic_ac_halfpi", "stroboscopic-AC builder"
    )
)
def test_builder_matches_scenario_4() -> None:
    """Compare Phase 1 stroboscopic AC-π/2 builder output against the qc.py
    reference for scenario 4."""
    raise AssertionError("unreachable — skipped until builder lands")


@pytest.mark.skip(
    reason=_activation_template_reason(
        "05_single_mode_squeeze_displace", "squeeze + displace state-prep"
    )
)
def test_builder_matches_scenario_5() -> None:
    """Compare Phase 1 squeeze + displacement output against the qc.py
    reference for scenario 5."""
    raise AssertionError("unreachable — skipped until builder lands")
