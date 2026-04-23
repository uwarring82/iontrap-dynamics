# SPDX-License-Identifier: MIT
"""Repo-local Clos 2016 reference anchors used by reproduction tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_LEGACY_CLOS2016_DIR = Path(__file__).resolve().parents[2] / "legacy/clos 2016 prl"
DEFAULT_CUTOFF_RELATIVE_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True, kw_only=True)
class Clos2016AxialModeReference:
    """Legacy axial-mode reference for a small ion chain."""

    n_ions: int
    dimensionless_frequencies: NDArray[np.float64]
    first_ion_participation_weights: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class Clos2016CutoffConvergence:
    """Cutoff-convergence table parsed from the legacy bundle."""

    n_ions: int
    cutoffs: NDArray[np.int64]
    ipr_average: NDArray[np.float64]
    ipr: NDArray[np.float64]
    omegaz_over_omega_axial: float
    omega_rabi_over_omega_axial: float
    inferred_converged_cutoff: int | None
    relative_tolerance: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Clos2016TheoryDimensionSurface:
    """Parsed ``theo_dim_N_*.dat`` surface from the legacy bundle.

    The archived numeric columns use the legacy MATLAB frequency convention:
    a numeric value of ``1`` denotes ``2pi x 1 MHz``. This loader preserves
    those raw values rather than guessing an SI conversion.
    """

    n_ions: int
    cutoffs: NDArray[np.int64]
    detunings_legacy_units: NDArray[np.float64]
    averaged_effective_dimension: NDArray[np.float64]
    omega_axial_legacy_units: float
    omega_rabi_legacy_units: float
    mean_occupation: float


# Reference values are the output of the legacy `cchain(N, center=1)` from
# `ergodic_ipr_av.m` (Coulomb chain in a quadratic axial trap with `beta=4`),
# reproduced numerically in Python. For N ions in a harmonic axial trap with
# Coulomb repulsion the COM mode sits at the trap frequency ω_z and the
# breathing mode at √3·ω_z; for N=3 the third mode is √(29/5)·ω_z. These are
# textbook results, not Porras-specific. The participation-weight rows are
# the first column of the Hessian eigenvector matrix V (the "spin-ion at
# index 0" projection used by `pol = Σ_n eta0·center_wf(n)·…`).
_AXIAL_MODE_REFERENCES: dict[int, Clos2016AxialModeReference] = {
    2: Clos2016AxialModeReference(
        n_ions=2,
        dimensionless_frequencies=np.asarray(
            [1.0, np.sqrt(3.0)],
            dtype=np.float64,
        ),
        first_ion_participation_weights=np.asarray(
            [-1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
            dtype=np.float64,
        ),
    ),
    3: Clos2016AxialModeReference(
        n_ions=3,
        dimensionless_frequencies=np.asarray(
            [1.0, np.sqrt(3.0), np.sqrt(29.0 / 5.0)],
            dtype=np.float64,
        ),
        first_ion_participation_weights=np.asarray(
            [1.0 / np.sqrt(3.0), -1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
            dtype=np.float64,
        ),
    ),
}


def clos2016_axial_mode_reference(n_ions: int) -> Clos2016AxialModeReference:
    """Return the N=2/N=3 axial-mode references used by Dispatch AAA."""
    try:
        ref = _AXIAL_MODE_REFERENCES[n_ions]
    except KeyError as exc:
        raise ValueError(
            f"Clos 2016 axial-mode references are only pinned for N=2 and N=3; got {n_ions}."
        ) from exc

    return Clos2016AxialModeReference(
        n_ions=ref.n_ions,
        dimensionless_frequencies=ref.dimensionless_frequencies.copy(),
        first_ion_participation_weights=ref.first_ion_participation_weights.copy(),
    )


def _relative_error(values: NDArray[np.float64], reference: float) -> NDArray[np.float64]:
    if reference == 0.0:
        return np.abs(values - reference)
    return np.abs(values - reference) / abs(reference)


def _infer_converged_cutoff(
    cutoffs: NDArray[np.int64],
    ipr_average: NDArray[np.float64],
    ipr: NDArray[np.float64],
    *,
    relative_tolerance: float,
) -> int | None:
    """Return the earliest cutoff with a two-point plateau versus the last sample."""
    final_ipr_average = float(ipr_average[-1])
    final_ipr = float(ipr[-1])

    for index, cutoff in enumerate(cutoffs):
        if cutoffs.size - index < 2:
            continue
        tail_ipr_average = ipr_average[index:]
        tail_ipr = ipr[index:]
        if np.all(
            _relative_error(tail_ipr_average, final_ipr_average) <= relative_tolerance
        ) and np.all(_relative_error(tail_ipr, final_ipr) <= relative_tolerance):
            return int(cutoff)
    return None


def load_clos2016_cutoff_convergence(
    n_ions: int,
    *,
    legacy_dir: Path = DEFAULT_LEGACY_CLOS2016_DIR,
    relative_tolerance: float = DEFAULT_CUTOFF_RELATIVE_TOLERANCE,
) -> Clos2016CutoffConvergence:
    """Parse one ``*ions_ipr_vs_nc.txt`` table from the legacy bundle."""
    table_path = legacy_dir / f"{n_ions}ions_ipr_vs_nc.txt"
    data = np.loadtxt(table_path, delimiter="\t", skiprows=1, dtype=np.float64)

    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"unexpected cutoff-table shape for {table_path}: {data.shape!r}")

    cutoffs = data[:, 0].astype(np.int64)
    ipr_average = data[:, 1].astype(np.float64)
    ipr = data[:, 2].astype(np.float64)
    n_ions_column = data[:, 3].astype(np.int64)
    omegaz_over_omega_axial = float(data[0, 4])
    omega_rabi_over_omega_axial = float(data[0, 5])

    if not np.all(n_ions_column == n_ions):
        raise ValueError(f"table {table_path} declares inconsistent ion counts: {n_ions_column!r}")
    if not np.allclose(data[:, 4], omegaz_over_omega_axial):
        raise ValueError(f"table {table_path} has non-constant omegaz/omega_axial column")
    if not np.allclose(data[:, 5], omega_rabi_over_omega_axial):
        raise ValueError(f"table {table_path} has non-constant OmegaR/omega_axial column")

    return Clos2016CutoffConvergence(
        n_ions=n_ions,
        cutoffs=cutoffs,
        ipr_average=ipr_average,
        ipr=ipr,
        omegaz_over_omega_axial=omegaz_over_omega_axial,
        omega_rabi_over_omega_axial=omega_rabi_over_omega_axial,
        inferred_converged_cutoff=_infer_converged_cutoff(
            cutoffs,
            ipr_average,
            ipr,
            relative_tolerance=relative_tolerance,
        ),
        relative_tolerance=relative_tolerance,
    )


def load_all_clos2016_cutoff_convergences(
    *,
    legacy_dir: Path = DEFAULT_LEGACY_CLOS2016_DIR,
    relative_tolerance: float = DEFAULT_CUTOFF_RELATIVE_TOLERANCE,
) -> tuple[Clos2016CutoffConvergence, ...]:
    """Return the full N=1..5 cutoff-convergence summary."""
    return tuple(
        load_clos2016_cutoff_convergence(
            n_ions,
            legacy_dir=legacy_dir,
            relative_tolerance=relative_tolerance,
        )
        for n_ions in range(1, 6)
    )


def load_clos2016_theory_dimension_surface(
    n_ions: int,
    *,
    legacy_dir: Path = DEFAULT_LEGACY_CLOS2016_DIR,
) -> Clos2016TheoryDimensionSurface:
    """Parse one archived ``theo_dim_N_*.dat`` cutoff-by-detuning surface."""
    table_path = legacy_dir / "DP num res_fig_1_2015_07_30" / f"theo_dim_N_{n_ions}.dat"
    data = np.loadtxt(table_path, delimiter="\t", skiprows=1, dtype=np.float64)

    if data.ndim != 2 or data.shape[1] != 8:
        raise ValueError(f"unexpected theory-surface shape for {table_path}: {data.shape!r}")

    cutoffs = np.unique(data[:, 0].astype(np.int64))
    detunings = np.unique(data[:, 4].astype(np.float64))
    expected_rows = cutoffs.size * detunings.size
    if data.shape[0] != expected_rows:
        raise ValueError(
            f"table {table_path} cannot be reshaped to cutoff x detuning grid: "
            f"{data.shape[0]} rows vs {cutoffs.size}x{detunings.size}."
        )

    n_ions_column = data[:, 2].astype(np.int64)
    omega_axial = float(data[0, 3])
    omega_rabi = float(data[0, 5])
    mean_occupation = float(data[0, 6])

    if not np.all(n_ions_column == n_ions):
        raise ValueError(f"table {table_path} declares inconsistent ion counts: {n_ions_column!r}")
    if not np.allclose(data[:, 3], omega_axial):
        raise ValueError(f"table {table_path} has non-constant omega_axial column")
    if not np.allclose(data[:, 5], omega_rabi):
        raise ValueError(f"table {table_path} has non-constant omega_rabi column")
    if not np.allclose(data[:, 6], mean_occupation):
        raise ValueError(f"table {table_path} has non-constant mean-occupation column")

    deff_grid = data[:, 1].reshape(detunings.size, cutoffs.size).T.astype(np.float64)

    return Clos2016TheoryDimensionSurface(
        n_ions=n_ions,
        cutoffs=cutoffs.astype(np.int64),
        detunings_legacy_units=detunings.astype(np.float64),
        averaged_effective_dimension=deff_grid,
        omega_axial_legacy_units=omega_axial,
        omega_rabi_legacy_units=omega_rabi,
        mean_occupation=mean_occupation,
    )


__all__ = [
    "DEFAULT_CUTOFF_RELATIVE_TOLERANCE",
    "DEFAULT_LEGACY_CLOS2016_DIR",
    "Clos2016AxialModeReference",
    "Clos2016CutoffConvergence",
    "Clos2016TheoryDimensionSurface",
    "clos2016_axial_mode_reference",
    "load_all_clos2016_cutoff_convergences",
    "load_clos2016_cutoff_convergence",
    "load_clos2016_theory_dimension_surface",
]
