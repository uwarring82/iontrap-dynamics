# SPDX-License-Identifier: MIT
"""Clos 2016 reproduction anchors for Dispatch AAA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iontrap_dynamics.clos2016_references import (
    DEFAULT_LEGACY_CLOS2016_DIR,
    clos2016_axial_mode_reference,
    load_all_clos2016_cutoff_convergences,
    load_clos2016_cutoff_convergence,
)

pytestmark = pytest.mark.regression_reproduction


def _allclose_up_to_global_sign(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-12) -> bool:
    return bool(np.allclose(actual, expected, atol=atol) or np.allclose(actual, -expected, atol=atol))


def test_legacy_bundle_directory_exists() -> None:
    assert Path("legacy/clos 2016 prl").resolve() == DEFAULT_LEGACY_CLOS2016_DIR
    assert DEFAULT_LEGACY_CLOS2016_DIR.is_dir()


@pytest.mark.parametrize(
    ("n_ions", "expected_frequencies", "expected_first_ion_weights"),
    [
        (
            2,
            np.asarray([1.0, 1.12266952429]),
            np.asarray([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]),
        ),
        (
            3,
            np.asarray([1.0, 1.110697060554, 1.249301846942]),
            np.asarray([1.0 / np.sqrt(3.0), -1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)]),
        ),
    ],
)
def test_axial_mode_references_match_pinned_values(
    n_ions: int,
    expected_frequencies: np.ndarray,
    expected_first_ion_weights: np.ndarray,
) -> None:
    ref = clos2016_axial_mode_reference(n_ions)

    assert ref.n_ions == n_ions
    np.testing.assert_allclose(ref.dimensionless_frequencies, expected_frequencies, atol=1e-12)
    assert _allclose_up_to_global_sign(
        ref.first_ion_participation_weights,
        expected_first_ion_weights,
    )


def test_axial_mode_reference_arrays_are_copied_on_return() -> None:
    ref = clos2016_axial_mode_reference(2)
    ref.dimensionless_frequencies[0] = 999.0
    ref.first_ion_participation_weights[0] = 999.0

    ref_again = clos2016_axial_mode_reference(2)
    np.testing.assert_allclose(ref_again.dimensionless_frequencies, [1.0, 1.12266952429], atol=1e-12)
    assert _allclose_up_to_global_sign(
        ref_again.first_ion_participation_weights,
        np.asarray([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]),
    )


def test_axial_mode_reference_rejects_unsupported_chain_sizes() -> None:
    with pytest.raises(ValueError, match="N=2 and N=3"):
        clos2016_axial_mode_reference(4)


@pytest.mark.parametrize(
    ("n_ions", "expected_omega_rabi_over_omega_axial", "expected_inferred_cutoff"),
    [
        (1, 0.7, 7),
        (2, 0.9555, 8),
        (3, 1.1993, 10),
        (4, 1.4333, None),
        (5, 1.6604, None),
    ],
)
def test_cutoff_tables_parse_expected_metadata_and_plateaus(
    n_ions: int,
    expected_omega_rabi_over_omega_axial: float,
    expected_inferred_cutoff: int | None,
) -> None:
    record = load_clos2016_cutoff_convergence(n_ions)

    assert record.n_ions == n_ions
    assert record.cutoffs[0] == 1
    assert np.all(np.diff(record.cutoffs) == 1)
    assert record.omegaz_over_omega_axial == pytest.approx(0.7)
    assert record.omega_rabi_over_omega_axial == pytest.approx(expected_omega_rabi_over_omega_axial)
    assert record.inferred_converged_cutoff == expected_inferred_cutoff


def test_cutoff_summary_loads_all_chain_sizes() -> None:
    records = load_all_clos2016_cutoff_convergences()
    assert tuple(record.n_ions for record in records) == (1, 2, 3, 4, 5)
