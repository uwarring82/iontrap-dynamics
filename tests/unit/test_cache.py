# SPDX-License-Identifier: MIT
"""Cache-integrity tests for :mod:`iontrap_dynamics.cache`.

Exercises every :class:`IntegrityError` path: mismatched hash, missing files,
unparseable JSON, wrong format version, missing/extra npz arrays, malformed
metadata and warnings. Also round-trips happy-path saves and confirms the
hash-computation helper is stable and sensitive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from iontrap_dynamics import (
    CACHE_FORMAT_VERSION,
    CONVENTION_VERSION,
    ConventionError,
    IntegrityError,
    IonTrapError,
    ResultMetadata,
    ResultWarning,
    StorageMode,
    TrajectoryResult,
    WarningSeverity,
    compute_request_hash,
    load_trajectory,
    save_trajectory,
)

# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

HASH_A = "a" * 64
HASH_B = "b" * 64


def _metadata(
    *,
    request_hash: str = HASH_A,
    storage_mode: StorageMode = StorageMode.OMITTED,
) -> ResultMetadata:
    return ResultMetadata(
        convention_version=CONVENTION_VERSION,
        request_hash=request_hash,
        backend_name="test-backend",
        backend_version="0.0.0",
        storage_mode=storage_mode,
        fock_truncations={"axial": 12},
        provenance_tags=("test",),
    )


def _sample_result(
    *,
    request_hash: str = HASH_A,
    with_warnings: bool = False,
) -> TrajectoryResult:
    expectations = {
        "sigma_z": np.array([1.0, 0.5, 0.0, -0.5, -1.0]),
        "n_axial": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
    }
    warnings: tuple[ResultWarning, ...] = ()
    if with_warnings:
        warnings = (
            ResultWarning(
                severity=WarningSeverity.CONVERGENCE,
                category="fock-truncation",
                message="p_top within [eps/10, eps)",
                diagnostics={"p_top": 5e-5, "eps": 1e-4},
            ),
        )
    return TrajectoryResult(
        metadata=_metadata(request_hash=request_hash),
        times=np.linspace(0.0, 1.0e-6, 5),
        expectations=expectations,
        warnings=warnings,
    )


# ----------------------------------------------------------------------------
# compute_request_hash — helper behaviour
# ----------------------------------------------------------------------------


class TestComputeRequestHash:
    def test_stable_across_insertion_order(self) -> None:
        h1 = compute_request_hash({"a": 1, "b": 2})
        h2 = compute_request_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_length_is_64_hex_chars(self) -> None:
        assert len(compute_request_hash({"x": 1})) == 64

    def test_sensitive_to_value_change(self) -> None:
        assert compute_request_hash({"a": 1}) != compute_request_hash({"a": 2})

    def test_sensitive_to_key_change(self) -> None:
        assert compute_request_hash({"a": 1}) != compute_request_hash({"b": 1})

    def test_non_json_value_raises_typeerror(self) -> None:
        """Fail-fast on unserialisable values (numpy array, object, etc.)."""
        with pytest.raises(TypeError):
            compute_request_hash({"x": np.array([1, 2, 3])})


# ----------------------------------------------------------------------------
# Round-trip happy paths
# ----------------------------------------------------------------------------


class TestRoundTrip:
    def test_roundtrip_preserves_fields(self, tmp_path: Path) -> None:
        original = _sample_result()
        save_trajectory(original, tmp_path / "r1")
        loaded = load_trajectory(tmp_path / "r1", expected_request_hash=HASH_A)

        np.testing.assert_array_equal(loaded.times, original.times)
        assert loaded.expectations.keys() == original.expectations.keys()
        for key in original.expectations:
            np.testing.assert_array_equal(loaded.expectations[key], original.expectations[key])
        assert loaded.metadata == original.metadata
        assert loaded.warnings == original.warnings

    def test_roundtrip_with_warnings(self, tmp_path: Path) -> None:
        original = _sample_result(with_warnings=True)
        save_trajectory(original, tmp_path / "r2")
        loaded = load_trajectory(tmp_path / "r2", expected_request_hash=HASH_A)

        assert len(loaded.warnings) == 1
        assert loaded.warnings[0].severity is WarningSeverity.CONVERGENCE
        assert loaded.warnings[0].diagnostics == {"p_top": 5e-5, "eps": 1e-4}

    def test_roundtrip_writes_two_files(self, tmp_path: Path) -> None:
        save_trajectory(_sample_result(), tmp_path / "r3")
        assert (tmp_path / "r3" / "manifest.json").is_file()
        assert (tmp_path / "r3" / "arrays.npz").is_file()


# ----------------------------------------------------------------------------
# Save-side failures
# ----------------------------------------------------------------------------


class TestSaveFailures:
    def test_eager_result_rejected(self, tmp_path: Path) -> None:
        result = TrajectoryResult(
            metadata=_metadata(storage_mode=StorageMode.EAGER),
            times=np.linspace(0.0, 1.0, 3),
            states=("a", "b", "c"),
        )
        with pytest.raises(ConventionError, match="OMITTED"):
            save_trajectory(result, tmp_path / "r")

    def test_lazy_result_rejected(self, tmp_path: Path) -> None:
        result = TrajectoryResult(
            metadata=_metadata(storage_mode=StorageMode.LAZY),
            times=np.linspace(0.0, 1.0, 3),
            states_loader=lambda i: f"s{i}",
        )
        with pytest.raises(ConventionError, match="OMITTED"):
            save_trajectory(result, tmp_path / "r")

    def test_refuses_to_overwrite_existing_cache(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        save_trajectory(_sample_result(), dest)
        with pytest.raises(FileExistsError):
            save_trajectory(_sample_result(), dest)

    def test_overwrite_flag_replaces_cache(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        save_trajectory(_sample_result(request_hash=HASH_A), dest)
        save_trajectory(_sample_result(request_hash=HASH_B), dest, overwrite=True)
        loaded = load_trajectory(dest, expected_request_hash=HASH_B)
        assert loaded.metadata.request_hash == HASH_B


# ----------------------------------------------------------------------------
# Load-side integrity failures
# ----------------------------------------------------------------------------


class TestLoadIntegrity:
    def test_hash_mismatch_raises(self, tmp_path: Path) -> None:
        save_trajectory(_sample_result(request_hash=HASH_A), tmp_path / "r")
        with pytest.raises(IntegrityError, match="request_hash mismatch"):
            load_trajectory(tmp_path / "r", expected_request_hash=HASH_B)

    def test_hash_mismatch_is_iontrap_error(self, tmp_path: Path) -> None:
        """Blanket catches should work."""
        save_trajectory(_sample_result(), tmp_path / "r")
        with pytest.raises(IonTrapError):
            load_trajectory(tmp_path / "r", expected_request_hash=HASH_B)

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        save_trajectory(_sample_result(), dest)
        (dest / "manifest.json").unlink()
        with pytest.raises(IntegrityError, match="manifest missing"):
            load_trajectory(dest, expected_request_hash=HASH_A)

    def test_missing_arrays_raises(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        save_trajectory(_sample_result(), dest)
        (dest / "arrays.npz").unlink()
        with pytest.raises(IntegrityError, match="arrays file missing"):
            load_trajectory(dest, expected_request_hash=HASH_A)

    def test_unparseable_manifest_raises(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        save_trajectory(_sample_result(), dest)
        (dest / "manifest.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(IntegrityError, match="not valid JSON"):
            load_trajectory(dest, expected_request_hash=HASH_A)

    def test_unknown_cache_format_version_raises(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        save_trajectory(_sample_result(), dest)
        manifest = json.loads((dest / "manifest.json").read_text())
        manifest["cache_format_version"] = CACHE_FORMAT_VERSION + 99
        (dest / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(IntegrityError, match="cache_format_version"):
            load_trajectory(dest, expected_request_hash=HASH_A)

    def test_missing_npz_key_raises(self, tmp_path: Path) -> None:
        """Remove an expectation array that the manifest still declares."""
        dest = tmp_path / "r"
        save_trajectory(_sample_result(), dest)
        with np.load(dest / "arrays.npz") as npz:
            kept = {"times": npz["times"], "expectation__sigma_z": npz["expectation__sigma_z"]}
        np.savez(dest / "arrays.npz", **kept)  # drop expectation__n_axial
        with pytest.raises(IntegrityError, match="missing keys"):
            load_trajectory(dest, expected_request_hash=HASH_A)

    def test_extra_npz_key_raises(self, tmp_path: Path) -> None:
        """An unexpected array is a tamper signal."""
        dest = tmp_path / "r"
        save_trajectory(_sample_result(), dest)
        with np.load(dest / "arrays.npz") as npz:
            payload = {k: npz[k] for k in npz.files}
        payload["expectation__injected"] = np.zeros(5)
        np.savez(dest / "arrays.npz", **payload)
        with pytest.raises(IntegrityError, match="unexpected keys"):
            load_trajectory(dest, expected_request_hash=HASH_A)

    def test_missing_required_times_key_raises(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        save_trajectory(_sample_result(), dest)
        with np.load(dest / "arrays.npz") as npz:
            kept = {k: npz[k] for k in npz.files if k != "times"}
        np.savez(dest / "arrays.npz", **kept)
        with pytest.raises(IntegrityError, match="times"):
            load_trajectory(dest, expected_request_hash=HASH_A)

    def test_malformed_metadata_raises(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        save_trajectory(_sample_result(), dest)
        manifest = json.loads((dest / "manifest.json").read_text())
        del manifest["metadata"]["backend_name"]
        (dest / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(IntegrityError, match="metadata malformed"):
            load_trajectory(dest, expected_request_hash=HASH_A)

    def test_non_object_root_raises(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        save_trajectory(_sample_result(), dest)
        (dest / "manifest.json").write_text('"just a string"', encoding="utf-8")
        with pytest.raises(IntegrityError, match="root is not a JSON object"):
            load_trajectory(dest, expected_request_hash=HASH_A)

    def test_malformed_warning_raises(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        save_trajectory(_sample_result(with_warnings=True), dest)
        manifest = json.loads((dest / "manifest.json").read_text())
        manifest["warnings"][0]["severity"] = "not-a-severity"
        (dest / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(IntegrityError, match="warnings malformed"):
            load_trajectory(dest, expected_request_hash=HASH_A)
