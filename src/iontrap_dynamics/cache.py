# SPDX-License-Identifier: MIT
"""Hash-verified persistence for :class:`~iontrap_dynamics.results.TrajectoryResult`.

Implements Design Principle 7 of ``WORKPLAN_v0.3.md`` ("Cache integrity
non-negotiable. No serialised file loaded without verifying parameter-hash
match"). The archival format follows §3 of the workplan: ``.npz`` for numeric
arrays plus a JSON manifest for metadata, never pickle.

API
---

* :func:`save_trajectory` — writes a cache directory.
* :func:`load_trajectory` — reads a cache directory, verifying hash integrity.
* :func:`compute_request_hash` — stable hex SHA-256 of a canonical-JSON
  payload, useful for callers wiring :attr:`ResultMetadata.request_hash`.

Scope for v0.1
--------------

Only :attr:`~iontrap_dynamics.results.StorageMode.OMITTED` results can be
cached. :class:`EAGER<iontrap_dynamics.results.StorageMode>` and
:class:`LAZY<iontrap_dynamics.results.StorageMode>` raise :class:`ConventionError`
because quantum-state objects are backend-specific and the library avoids
pickle. Phase 1+ will add backend-annotated state serialisation.

Failure semantics
-----------------

Every load-time failure raises :class:`IntegrityError`, matching
``CONVENTIONS.md`` §15 Level 3. The message names the specific failure
(hash mismatch, missing file, unknown cache-format version, extra/missing
npz array, unparseable JSON). Silent pass-through on corrupted caches is
forbidden.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import ConventionError, IntegrityError
from .results import (
    ResultMetadata,
    ResultWarning,
    StorageMode,
    TrajectoryResult,
    WarningSeverity,
)

#: Bumped whenever the on-disk layout changes in a way old loaders cannot read.
#: Old caches are not auto-migrated; producers must re-generate after a bump.
CACHE_FORMAT_VERSION: int = 1

_MANIFEST_NAME = "manifest.json"
_ARRAYS_NAME = "arrays.npz"
_TIMES_KEY = "times"
_EXPECTATION_PREFIX = "expectation__"


# ----------------------------------------------------------------------------
# Request-hash helper
# ----------------------------------------------------------------------------


def compute_request_hash(payload: Mapping[str, Any]) -> str:
    """Return a stable hex SHA-256 digest of a canonical-JSON payload.

    Produces identical output for equal-by-value dicts regardless of insertion
    order. Non-JSON-serialisable values raise :class:`TypeError` at hash time
    (fail-fast — callers must convert numpy scalars, Path objects, etc. to
    their primitive equivalents before hashing).
    """
    canonical = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------


def save_trajectory(
    result: TrajectoryResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Persist ``result`` as a hash-verified cache directory at ``path``.

    The directory contains two files:

    * ``manifest.json`` — metadata, warnings, request hash, format version.
    * ``arrays.npz``    — the ``times`` array plus one array per observable
      under key ``expectation__<label>``.

    Parameters
    ----------
    result
        Must have ``metadata.storage_mode == StorageMode.OMITTED``. EAGER or
        LAZY results raise :class:`ConventionError`; see module docstring.
    path
        Destination directory. Created if it does not exist.
    overwrite
        If ``False`` (default), an existing non-empty destination raises
        :class:`FileExistsError`. If ``True``, the two cache files inside
        are overwritten; any *other* files inside the destination are left
        untouched.
    """
    if result.metadata.storage_mode is not StorageMode.OMITTED:
        raise ConventionError(
            f"cache v{CACHE_FORMAT_VERSION} only persists storage_mode=OMITTED "
            f"results; got {result.metadata.storage_mode.value}. "
            "Re-run the solver with states discarded, or wait for backend-"
            "annotated state serialisation in Phase 1+."
        )

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)

    manifest_path = target / _MANIFEST_NAME
    arrays_path = target / _ARRAYS_NAME

    if not overwrite and (manifest_path.exists() or arrays_path.exists()):
        raise FileExistsError(
            f"cache files already present in {target}; pass overwrite=True to replace them."
        )

    # Numeric payload: times + one array per expectation.
    npz_payload: dict[str, np.ndarray] = {_TIMES_KEY: np.asarray(result.times)}
    for label, array in result.expectations.items():
        npz_payload[_EXPECTATION_PREFIX + label] = np.asarray(array)
    np.savez(arrays_path, **npz_payload)  # type: ignore[arg-type]

    # Manifest: metadata + warnings + the list of expectation labels so load
    # can detect both missing and extra arrays.
    manifest = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "metadata": {
            "convention_version": result.metadata.convention_version,
            "request_hash": result.metadata.request_hash,
            "backend_name": result.metadata.backend_name,
            "backend_version": result.metadata.backend_version,
            "storage_mode": result.metadata.storage_mode.value,
            "fock_truncations": dict(result.metadata.fock_truncations),
            "provenance_tags": list(result.metadata.provenance_tags),
        },
        "expectation_labels": sorted(result.expectations.keys()),
        "warnings": [
            {
                "severity": w.severity.value,
                "category": w.category,
                "message": w.message,
                "diagnostics": dict(w.diagnostics),
            }
            for w in result.warnings
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------


def load_trajectory(
    path: str | Path,
    *,
    expected_request_hash: str,
) -> TrajectoryResult:
    """Load a cached :class:`TrajectoryResult`, verifying hash integrity.

    Parameters
    ----------
    path
        Cache directory written by :func:`save_trajectory`.
    expected_request_hash
        Hex SHA-256 the caller expects the cache to be bound to. Must equal
        the value recorded in the manifest; mismatch raises
        :class:`IntegrityError`.

    Raises
    ------
    IntegrityError
        On any of: mismatched hash; missing ``manifest.json`` or
        ``arrays.npz``; unparseable manifest; unknown cache-format version;
        missing or extra npz arrays relative to the manifest's declared
        expectation labels; malformed metadata fields.
    """
    source = Path(path)
    manifest_path = source / _MANIFEST_NAME
    arrays_path = source / _ARRAYS_NAME

    if not manifest_path.is_file():
        raise IntegrityError(f"cache manifest missing: {manifest_path}")
    if not arrays_path.is_file():
        raise IntegrityError(f"cache arrays file missing: {arrays_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IntegrityError(f"manifest is not valid JSON: {manifest_path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise IntegrityError(f"manifest root is not a JSON object: {manifest_path}")

    cache_version = manifest.get("cache_format_version")
    if cache_version != CACHE_FORMAT_VERSION:
        raise IntegrityError(
            f"unknown cache_format_version {cache_version!r}; "
            f"this library reads version {CACHE_FORMAT_VERSION}."
        )

    meta_raw = manifest.get("metadata")
    if not isinstance(meta_raw, dict):
        raise IntegrityError("manifest.metadata is missing or not an object")

    recorded_hash = meta_raw.get("request_hash")
    if recorded_hash != expected_request_hash:
        raise IntegrityError(
            "request_hash mismatch: "
            f"expected {expected_request_hash!r}, cache recorded {recorded_hash!r}."
        )

    try:
        metadata = ResultMetadata(
            convention_version=meta_raw["convention_version"],
            request_hash=meta_raw["request_hash"],
            backend_name=meta_raw["backend_name"],
            backend_version=meta_raw["backend_version"],
            storage_mode=StorageMode(meta_raw["storage_mode"]),
            fock_truncations=dict(meta_raw.get("fock_truncations", {})),
            provenance_tags=tuple(meta_raw.get("provenance_tags", ())),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise IntegrityError(f"manifest.metadata malformed: {exc}") from exc

    expected_labels = manifest.get("expectation_labels", [])
    if not isinstance(expected_labels, list):
        raise IntegrityError("manifest.expectation_labels must be a JSON array")

    warnings_raw = manifest.get("warnings", [])
    if not isinstance(warnings_raw, list):
        raise IntegrityError("manifest.warnings must be a JSON array")
    try:
        warnings = tuple(
            ResultWarning(
                severity=WarningSeverity(w["severity"]),
                category=w["category"],
                message=w["message"],
                diagnostics=dict(w.get("diagnostics", {})),
            )
            for w in warnings_raw
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise IntegrityError(f"manifest.warnings malformed: {exc}") from exc

    with np.load(arrays_path) as npz:
        array_keys = set(npz.files)
        if _TIMES_KEY not in array_keys:
            raise IntegrityError(f"arrays.npz is missing required key {_TIMES_KEY!r}")

        expected_npz_keys = {_TIMES_KEY} | {_EXPECTATION_PREFIX + lbl for lbl in expected_labels}
        missing = expected_npz_keys - array_keys
        extra = array_keys - expected_npz_keys
        if missing:
            raise IntegrityError(f"arrays.npz is missing keys: {sorted(missing)}")
        if extra:
            raise IntegrityError(f"arrays.npz has unexpected keys: {sorted(extra)}")

        times = np.array(npz[_TIMES_KEY])
        expectations = {lbl: np.array(npz[_EXPECTATION_PREFIX + lbl]) for lbl in expected_labels}

    return TrajectoryResult(
        metadata=metadata,
        times=times,
        expectations=expectations,
        warnings=warnings,
    )


__all__ = [
    "CACHE_FORMAT_VERSION",
    "compute_request_hash",
    "load_trajectory",
    "save_trajectory",
]
