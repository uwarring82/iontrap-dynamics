# SPDX-License-Identifier: MIT
"""Exact-diagonalization entry points and result schema.

AAC scope: dense full-spectrum diagonalization via :func:`scipy.linalg.eigh`
plus a frozen result object aligned with the existing result-family style.
Iterative interior-window methods are intentionally deferred; callers may
request them by name, but only ``method="dense"`` is implemented here.

BBA scope: a second dense backend via ``jax.numpy.linalg.eigh``. The
``backend_name="spectrum-jax"`` value is device-neutral; an optional
``device="gpu"|"cpu"|None`` kwarg routes the eigensolve to a specific JAX
device when provided. Device selection is provenance, not identity: the
``backend_name`` is the same regardless of whether the JAX path ran on
CPU, CUDA, or Metal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import qutip
import scipy
from numpy.typing import NDArray
from scipy import linalg as scipy_linalg

from .conventions import CONVENTION_VERSION
from .exceptions import ConventionError
from .results import Result, ResultWarning

SpectrumVector = NDArray[np.complex128]

_ALLOWED_BACKEND_NAMES = frozenset({"spectrum-scipy", "spectrum-jax"})
_ALLOWED_DEVICE_VALUES = frozenset({"cpu", "gpu"})


@dataclass(frozen=True, slots=True, kw_only=True)
class SpectrumMetadata:
    """Provenance and context recorded for a :class:`SpectrumResult`."""

    convention_version: str
    request_hash: str
    backend_name: str
    backend_version: str
    fock_truncations: Mapping[str, int] = field(default_factory=dict)
    provenance_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class SpectrumResult(Result):
    """Frozen exact-diagonalization output.

    Mirrors the established result-family style while carrying the spectrum
    payload needed by later Clos/Porras analyses.
    """

    metadata: SpectrumMetadata  # type: ignore[assignment]  # spectrum metadata omits storage_mode by design; eigenvectors_loader handles laziness separately
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128] | None = None
    eigenvectors_loader: Callable[[int], SpectrumVector] | None = None
    window_center_energy: float | None = None
    window_width_energy: float | None = None
    initial_state_mean_energy: float | None = None
    initial_state_energy_std: float | None = None
    method: str = "dense"
    warnings: tuple[ResultWarning, ...] = ()

    def __post_init__(self) -> None:
        if self.method not in {"dense", "shift_invert"}:
            raise ConventionError(
                f"SpectrumResult method must be 'dense' or 'shift_invert'; got {self.method!r}."
            )
        if self.eigenvalues.ndim != 1:
            raise ConventionError("SpectrumResult requires `eigenvalues` to be a 1-D real array.")
        have_vectors = self.eigenvectors is not None
        have_loader = self.eigenvectors_loader is not None
        if have_vectors and have_loader:
            raise ConventionError(
                "SpectrumResult forbids setting both `eigenvectors` and "
                "`eigenvectors_loader`; choose one source."
            )
        if not have_vectors and not have_loader:
            raise ConventionError(
                "SpectrumResult requires either materialised `eigenvectors` or "
                "an `eigenvectors_loader`."
            )
        if have_vectors:
            assert self.eigenvectors is not None
            if self.eigenvectors.ndim != 2:
                raise ConventionError("SpectrumResult requires `eigenvectors` to be a 2-D array.")
            if self.eigenvectors.shape[1] != self.eigenvalues.shape[0]:
                raise ConventionError(
                    "SpectrumResult eigenvector columns must match the number of eigenvalues."
                )


def solve_spectrum(
    hamiltonian: qutip.Qobj | NDArray[np.complexfloating[Any]] | NDArray[np.floating[Any]],
    *,
    method: str = "dense",
    request_hash: str = "",
    backend_name: str | None = None,
    device: str | None = None,
    fock_truncations: Mapping[str, int] | None = None,
    provenance_tags: tuple[str, ...] = (),
    initial_state: qutip.Qobj
    | NDArray[np.complexfloating[Any]]
    | NDArray[np.floating[Any]]
    | None = None,
) -> SpectrumResult:
    """Diagonalize a Hermitian Hamiltonian and return a :class:`SpectrumResult`.

    Parameters
    ----------
    hamiltonian
        Hermitian operator as a QuTiP ``Qobj`` or dense square array.
    method
        Currently only ``"dense"`` is implemented. ``"shift_invert"`` is
        reserved for a later dispatch.
    request_hash
        Reproducibility token copied into the result metadata.
    backend_name
        Optional backend selector. ``None`` or ``"spectrum-scipy"`` uses
        :func:`scipy.linalg.eigh` (default). ``"spectrum-jax"`` uses
        :func:`jax.numpy.linalg.eigh`; the JAX path is device-neutral, so
        the same ``backend_name`` covers CPU-JAX, CUDA, and Metal.
    device
        Optional JAX device selector. Accepted values are ``"cpu"`` /
        ``"gpu"`` / ``None`` (JAX's default device). Only applicable to
        ``backend_name="spectrum-jax"``; passing ``device=...`` with the
        scipy backend is a ``ConventionError``. The selected device is
        recorded in ``metadata.provenance_tags`` as ``"device:<value>"``.
    fock_truncations
        Mode-label to cutoff mapping recorded verbatim in the metadata.
    provenance_tags
        Free-form provenance tags stored in the metadata.
    initial_state
        Optional state used only to record mean energy and energy spread on
        the result. Supports kets and density matrices as QuTiP objects or
        dense arrays.
    """
    if method != "dense":
        raise ConventionError(
            f"solve_spectrum(method={method!r}) is not implemented yet; AAC ships "
            "only the dense `scipy.linalg.eigh` reference path."
        )

    resolved_backend = backend_name if backend_name is not None else "spectrum-scipy"
    if resolved_backend not in _ALLOWED_BACKEND_NAMES:
        raise ConventionError(
            f"solve_spectrum(backend_name={resolved_backend!r}) is not recognised; "
            f"allowed values are {sorted(_ALLOWED_BACKEND_NAMES)}."
        )
    if device is not None and device not in _ALLOWED_DEVICE_VALUES:
        raise ConventionError(
            f"solve_spectrum(device={device!r}) is not recognised; "
            f"allowed values are {sorted(_ALLOWED_DEVICE_VALUES)} or None."
        )
    if device is not None and resolved_backend != "spectrum-jax":
        raise ConventionError(
            f"solve_spectrum(device={device!r}) is only applicable to "
            f"backend_name='spectrum-jax'; got backend_name={resolved_backend!r}."
        )

    matrix = _as_hermitian_matrix(hamiltonian)

    if resolved_backend == "spectrum-jax":
        eigenvalues, eigenvectors, backend_version, device_tag = _solve_dense_jax(matrix, device)
        extra_tags: tuple[str, ...] = (f"device:{device_tag}",)
    else:
        eigenvalues, eigenvectors = scipy_linalg.eigh(matrix)
        backend_version = scipy.__version__
        extra_tags = ()

    mean_energy: float | None = None
    energy_std: float | None = None
    if initial_state is not None:
        mean_energy, energy_std = _state_energy_moments(matrix, initial_state)

    metadata = SpectrumMetadata(
        convention_version=CONVENTION_VERSION,
        request_hash=request_hash,
        backend_name=resolved_backend,
        backend_version=backend_version,
        fock_truncations=dict(fock_truncations or {}),
        provenance_tags=tuple(provenance_tags) + extra_tags,
    )
    return SpectrumResult(
        metadata=metadata,
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        eigenvectors=np.asarray(eigenvectors, dtype=np.complex128),
        initial_state_mean_energy=mean_energy,
        initial_state_energy_std=energy_std,
        method="dense",
    )


def _solve_dense_jax(
    matrix: NDArray[np.complex128],
    device: str | None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128], str, str]:
    """Dispatch dense ``eigh`` through JAX; return eigendata + backend metadata.

    Returns a 4-tuple ``(eigenvalues, eigenvectors, backend_version, device_tag)``
    where ``device_tag`` is the JAX platform string the solve actually ran on
    (``"cpu"`` / ``"gpu"`` / whatever ``jax.default_backend()`` reports).
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ConventionError(
            "solve_spectrum(backend_name='spectrum-jax') requires JAX; install "
            "with the [jax] extra (CPU) or [gpu] extra (CUDA build)."
        ) from exc

    # Force x64 — the library's dtype contract is double precision. Same rationale
    # as backends/jax/_core.py; documented in docs/phase-2-jax-backend-design.md §8.
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call, unused-ignore]

    available_platforms = {d.platform for d in jax.devices()}
    if device is not None and device not in available_platforms:
        raise ConventionError(
            f"solve_spectrum(device={device!r}) requested but unavailable; "
            f"JAX platforms on this install: {sorted(available_platforms)}."
        )

    if device is not None:
        target_device = jax.devices(device)[0]
        jax_matrix = jax.device_put(jnp.asarray(matrix, dtype=jnp.complex128), target_device)
        device_tag = device
    else:
        jax_matrix = jnp.asarray(matrix, dtype=jnp.complex128)
        device_tag = jax.default_backend()

    eigenvalues_jax, eigenvectors_jax = jnp.linalg.eigh(jax_matrix)
    eigenvalues_jax.block_until_ready()

    return (
        np.asarray(eigenvalues_jax, dtype=np.float64),
        np.asarray(eigenvectors_jax, dtype=np.complex128),
        f"jax-{jax.__version__}",
        device_tag,
    )


def _as_hermitian_matrix(
    hamiltonian: qutip.Qobj | NDArray[np.complexfloating[Any]] | NDArray[np.floating[Any]],
) -> NDArray[np.complex128]:
    """Convert ``hamiltonian`` to a dense Hermitian matrix or raise."""
    if isinstance(hamiltonian, qutip.Qobj):
        matrix = np.asarray(hamiltonian.full(), dtype=np.complex128)
    else:
        matrix = np.asarray(hamiltonian, dtype=np.complex128)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ConventionError(
            f"solve_spectrum requires a square operator; got shape {matrix.shape!r}."
        )
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12):
        raise ConventionError("solve_spectrum requires a Hermitian Hamiltonian.")
    return matrix


def _state_energy_moments(
    hamiltonian: NDArray[np.complex128],
    initial_state: qutip.Qobj | NDArray[np.complexfloating[Any]] | NDArray[np.floating[Any]],
) -> tuple[float, float]:
    """Return ``(<H>, sqrt(<H²> - <H>²))`` for a ket or density matrix."""
    if isinstance(initial_state, qutip.Qobj):
        state = np.asarray(initial_state.full(), dtype=np.complex128)
    else:
        state = np.asarray(initial_state, dtype=np.complex128)

    dim = hamiltonian.shape[0]
    if state.ndim == 1 or (state.ndim == 2 and 1 in state.shape):
        ket = state.reshape(-1)
        if ket.shape[0] != dim:
            raise ConventionError(f"initial_state ket has length {ket.shape[0]}, expected {dim}.")
        norm = np.vdot(ket, ket)
        if np.isclose(norm, 0.0):
            raise ConventionError("initial_state ket must have non-zero norm.")
        ket = ket / np.sqrt(norm)
        mean = np.vdot(ket, hamiltonian @ ket)
        second = np.vdot(ket, hamiltonian @ (hamiltonian @ ket))
    elif state.ndim == 2 and state.shape == (dim, dim):
        trace = np.trace(state)
        if np.isclose(trace, 0.0):
            raise ConventionError("initial_state density matrix must have non-zero trace.")
        rho = state / trace
        mean = np.trace(rho @ hamiltonian)
        second = np.trace(rho @ hamiltonian @ hamiltonian)
    else:
        raise ConventionError(
            "initial_state must be a ket vector or a square density matrix "
            "matching the Hamiltonian dimension."
        )

    mean_real = float(np.real_if_close(mean, tol=1000))
    variance = float(np.real_if_close(second - mean * mean, tol=1000))
    variance = max(variance, 0.0)
    return mean_real, float(np.sqrt(variance))


__all__ = [
    "SpectrumMetadata",
    "SpectrumResult",
    "solve_spectrum",
]
