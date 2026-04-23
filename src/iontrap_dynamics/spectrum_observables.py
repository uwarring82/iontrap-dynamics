# SPDX-License-Identifier: MIT
"""Spectrum-domain analyses for exact-diagonalization results.

These helpers operate on :class:`~iontrap_dynamics.spectrum.SpectrumResult`
objects, not on trajectory observables. They cover the AAD scope for the
Clos/Porras integration track: IPR, effective dimension, ETH diagonals, and
phonon-number diagonals.

Naming note
-----------

The legacy MATLAB bundle labels the effective dimension
``1 / sum(p_alpha^2)`` as ``IPR``. This module uses the modern separation:

- :func:`inverse_participation_ratio` returns ``sum(p_alpha^2)``
- :func:`effective_dimension` returns its reciprocal

so later regression code can match the bundled ``d_eff`` tables without
baking the MATLAB naming quirk into the public API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import qutip
from numpy.typing import NDArray

from .exceptions import ConventionError
from .hilbert import HilbertSpace
from .spectrum import SpectrumResult


def inverse_participation_ratio(
    spectrum: SpectrumResult,
    initial_state: qutip.Qobj | NDArray[np.complexfloating[Any]] | NDArray[np.floating[Any]],
) -> float:
    """Return ``sum(p_alpha^2)`` in the eigenbasis carried by ``spectrum``.

    Here ``p_alpha = <E_alpha|rho_0|E_alpha>`` are the spectral populations
    of the supplied ket or density matrix. A single eigenstate gives 1;
    a uniform pure superposition over ``d`` eigenstates gives ``1 / d``.
    """
    populations = _spectral_populations(spectrum, initial_state)
    return float(np.sum(populations**2))


def effective_dimension(
    spectrum: SpectrumResult,
    initial_state: qutip.Qobj | NDArray[np.complexfloating[Any]] | NDArray[np.floating[Any]],
) -> float:
    """Return ``d_eff = 1 / IPR`` for the supplied initial state."""
    ipr = inverse_participation_ratio(spectrum, initial_state)
    if np.isclose(ipr, 0.0):
        raise ConventionError("effective_dimension is undefined for zero IPR.")
    return float(1.0 / ipr)


def eth_diagonal(
    spectrum: SpectrumResult,
    operator: qutip.Qobj | NDArray[np.complexfloating[Any]] | NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Return per-eigenstate diagonal expectations ``<E_alpha|O|E_alpha>``.

    The operator must be Hermitian and match the physical Hilbert-space
    dimension of the eigenvectors stored on ``spectrum``.
    """
    eigenvectors = _eigenvector_matrix(spectrum)
    operator_matrix = _as_hermitian_operator_matrix(operator, eigenvectors.shape[0])
    diagonals = np.einsum(
        "ia,ij,ja->a",
        eigenvectors.conj(),
        operator_matrix,
        eigenvectors,
        optimize=True,
    )
    diagonals = np.real_if_close(diagonals, tol=1000)
    if np.iscomplexobj(diagonals):
        raise ConventionError("eth_diagonal expected a Hermitian operator with real diagonals.")
    return np.asarray(diagonals, dtype=np.float64)


def phonon_number_diagonals(
    spectrum: SpectrumResult,
    hilbert: HilbertSpace,
    *,
    mode_labels: Sequence[str] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Return legacy-style ``nphdiag_cell`` arrays for the requested modes.

    The returned mapping is keyed by mode label, each value holding the
    diagonal of that mode's number operator in the spectrum eigenbasis.
    """
    labels = tuple(mode_labels) if mode_labels is not None else tuple(
        mode.label for mode in hilbert.system.modes
    )
    return {label: eth_diagonal(spectrum, hilbert.number_for_mode(label)) for label in labels}


def _spectral_populations(
    spectrum: SpectrumResult,
    initial_state: qutip.Qobj | NDArray[np.complexfloating[Any]] | NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Return ``p_alpha = <E_alpha|rho_0|E_alpha>`` as a real 1-D array."""
    eigenvectors = _eigenvector_matrix(spectrum)
    rho = _as_density_matrix(initial_state, eigenvectors.shape[0])
    populations = np.einsum(
        "ia,ij,ja->a",
        eigenvectors.conj(),
        rho,
        eigenvectors,
        optimize=True,
    )
    populations = np.real_if_close(populations, tol=1000)
    if np.iscomplexobj(populations):
        raise ConventionError("Spectral populations must be real for a valid density operator.")
    populations = np.asarray(populations, dtype=np.float64)
    populations[np.abs(populations) < 1e-14] = 0.0
    if np.any(populations < -1e-10):
        raise ConventionError("Spectral populations must be non-negative.")
    return np.clip(populations, 0.0, None)


def _eigenvector_matrix(spectrum: SpectrumResult) -> NDArray[np.complex128]:
    """Materialise the eigenvector columns from ``spectrum`` as a dense matrix."""
    if spectrum.eigenvectors is not None:
        vectors = np.asarray(spectrum.eigenvectors, dtype=np.complex128)
    else:
        assert spectrum.eigenvectors_loader is not None
        columns = [
            np.asarray(spectrum.eigenvectors_loader(i), dtype=np.complex128).reshape(-1)
            for i in range(spectrum.eigenvalues.shape[0])
        ]
        if not columns:
            raise ConventionError("SpectrumResult must carry at least one eigenvector.")
        vectors = np.column_stack(columns)

    normalized = np.empty_like(vectors, dtype=np.complex128)
    for index in range(vectors.shape[1]):
        column = vectors[:, index]
        norm = np.linalg.norm(column)
        if np.isclose(norm, 0.0):
            raise ConventionError(f"eigenvector column {index} has zero norm.")
        normalized[:, index] = column / norm
    return normalized


def _as_density_matrix(
    initial_state: qutip.Qobj | NDArray[np.complexfloating[Any]] | NDArray[np.floating[Any]],
    dimension: int,
) -> NDArray[np.complex128]:
    """Convert a ket or density matrix to a normalized dense density matrix."""
    if isinstance(initial_state, qutip.Qobj):
        state = np.asarray(initial_state.full(), dtype=np.complex128)
    else:
        state = np.asarray(initial_state, dtype=np.complex128)

    if state.ndim == 1 or (state.ndim == 2 and 1 in state.shape):
        ket = state.reshape(-1)
        if ket.shape[0] != dimension:
            raise ConventionError(
                f"initial_state ket has length {ket.shape[0]}, expected {dimension}."
            )
        norm = np.vdot(ket, ket)
        if np.isclose(norm, 0.0):
            raise ConventionError("initial_state ket must have non-zero norm.")
        ket = ket / np.sqrt(norm)
        return np.outer(ket, ket.conj())

    if state.ndim == 2 and state.shape == (dimension, dimension):
        trace = np.trace(state)
        if np.isclose(trace, 0.0):
            raise ConventionError("initial_state density matrix must have non-zero trace.")
        rho = state / trace
        if not np.allclose(rho, rho.conj().T, atol=1e-12):
            raise ConventionError("initial_state density matrix must be Hermitian.")
        return rho

    raise ConventionError(
        "initial_state must be a ket vector or a square density matrix matching "
        "the spectrum dimension."
    )


def _as_hermitian_operator_matrix(
    operator: qutip.Qobj | NDArray[np.complexfloating[Any]] | NDArray[np.floating[Any]],
    dimension: int,
) -> NDArray[np.complex128]:
    """Convert ``operator`` to a dense Hermitian matrix with the given dimension."""
    if isinstance(operator, qutip.Qobj):
        matrix = np.asarray(operator.full(), dtype=np.complex128)
    else:
        matrix = np.asarray(operator, dtype=np.complex128)

    if matrix.shape != (dimension, dimension):
        raise ConventionError(
            f"operator shape {matrix.shape!r} does not match spectrum dimension {dimension}."
        )
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12):
        raise ConventionError("eth_diagonal requires a Hermitian operator.")
    return matrix


__all__ = [
    "effective_dimension",
    "eth_diagonal",
    "inverse_participation_ratio",
    "phonon_number_diagonals",
]
