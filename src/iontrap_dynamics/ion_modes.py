# SPDX-License-Identifier: MIT
"""Ion normal→local symplectic adapter (WP-07 GT3b, consumer side).

Builds the ion-specific symplectic map ``S`` that re-expresses a Gaussian covariance
from the **normal-mode** basis (where squeezing dynamics live) into the **local-ion**
basis, so an ion-partition entanglement can be read out with :func:`gaussian.log_negativity`.
The generic congruence ``V ↦ S V Sᵀ`` and its ``S Ω Sᵀ = Ω`` guard are GT3a
(:func:`gaussian.congruence`); this module only *builds* ``S`` from a mode basis and hands
it over — no phase-space or entanglement math lives here beyond the construction.

This is the **consumer** half of the ratified cross-repo handshake
(``task cards/TC-gt3b-ion-symplectic-adapter.md`` v0.3, ``1a88145``). The **producer**
(``iontrap-structure``) emits a serialization-neutral ``IonModeBasis`` payload — plain
arrays + metadata, **no shared runtime class** — and this module *validates and
materializes its own immutable record* (:func:`materialize_ion_mode_basis`). ``iontrap-dynamics``
owns the schema version + consumer contract; ``iontrap-structure`` owns field correctness.

Map (card §3, §27 quadratures ``x̂ = √(2ω/ℏ) q̂`` ⇒ vacuum variance 1). With the
mass-symmetrised, orthonormal basis ``B`` (`B[j, m]` = coordinate ``j``, mode ``m``),
mode frequencies ``ω_m``, and local reference frequencies ``ω_local,j`` the ``√2``, ``ℏ``
and **mass all cancel**, leaving in block ordering ``R = (x₁…x_n, p₁…p_n)``::

    S = diag(X, P),   X_{jm} = B_{jm} √(ω_local,j / ω_m),   P_{jm} = B_{jm} √(ω_m / ω_local,j).

``S Ω Sᵀ = Ω`` because ``X Pᵀ = I`` (rows of ``B`` orthonormal) — symplectic and mass-free.
The adapter emits ``S`` permuted into §27 per-mode ordering ``(x̂₁, p̂₁, …)``. The local
reference frequencies are an explicit **local-coordinate gauge** (a product of local
single-mode squeezings): they change the covariance representation / occupation / ``T_eff``
but **not** ion-cut ``E_N`` (invariant under local symplectics; card D3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaussian import congruence

__all__ = [
    "ION_MODE_BASIS_SCHEMA_VERSION",
    "IonModeBasis",
    "ion_mode_indices",
    "materialize_ion_mode_basis",
    "normal_to_local_symplectic",
    "to_local_covariance",
]

#: Consumer-owned schema version of the ``IonModeBasis`` wire payload (card D2: the schema +
#: version are owned here, in ``iontrap-dynamics``; the producer emits this exact value). A
#: payload declaring any other version is rejected — a loud mismatch, not a silent misread.
ION_MODE_BASIS_SCHEMA_VERSION = 1

#: Canonical wire ordering the payload must follow: coordinate rows are **ion-major, axis-minor**
#: (row ``r = axes_per_ion·i + c`` for ion ``i``, axis ``c``); column ``m`` of ``B`` aligns with
#: ``frequencies_rad_s[m]``. Emitted as the ``coordinate_frame`` tag by the producer and enforced by
#: exact match in :func:`materialize_ion_mode_basis`. It is a **declared, trusted** invariant: ``B``
#: carries no coordinate labels, so the row ordering cannot be cross-checked against it — producer
#: conformance to this ordering is the producer's guaranteed responsibility (card D2, field
#: correctness). A mismatched tag is a loud rejection; a *mislabelled* payload (wrong rows, canonical
#: tag) is a producer-side contract violation the consumer cannot detect, by design.
_CANONICAL_COORDINATE_FRAME = "ion-major-axis-minor;row=axes_per_ion*i+c;col=mode"

#: ``‖BᵀB − 𝟙‖`` tolerance for the mode basis, set **below** GT3a :func:`gaussian.congruence`'s
#: ``1e-11`` symplecticity gate so that **materialize acceptance implies the built ``S`` is accepted**
#: (accept ⇒ usable, no confusing downstream "S is not symplectic"). The built ``S``'s relative
#: ``SΩSᵀ=Ω`` residual is the Gram residual times a **mode-frequency-spread** amplification: ``≈2.6``
#: across the realistic ion band (mode frequencies within a few ×), rising to ``~40`` only for
#: extreme ``≥1e3×`` spreads. The **local-reference-frequency gauge does not amplify** — congruence's
#: per-entry, scale-aware relative gate cancels the ``√(ω_local)`` factors exactly. At ``1e-13`` the
#: coherence margin is ``≳30×`` in-band and still holds (``≳2×``) even at the extreme spread. A
#: genuine eigenbasis is orthonormal to ``~1e-15`` (``≳25×`` margin here); only a lossy wire (too few
#: significant figures) trips this — a clear, actionable rejection.
_GRAM_TOL = 1e-13


@dataclass(frozen=True)
class IonModeBasis:
    """A validated, immutable normal-mode basis for the GT3b adapter.

    Materialized from a producer payload by :func:`materialize_ion_mode_basis` — do not build
    directly. ``n = n_coords`` local coordinates / normal modes; ``N = n_ions`` ions;
    ``axes_per_ion = n // N`` (3 for a full trap, 1 for an axial reduction).
    """

    frequencies_rad_s: np.ndarray  #: (n,) mode frequencies ``ω_m > 0``
    mass_weighted_eigenvectors: np.ndarray  #: (n, n) orthonormal ``B``; ``B[:, m]`` is mode ``m``
    masses_kg: np.ndarray  #: (N,) species masses — *values* are mass-free (``S`` never uses them),
    #: but the *length* ``N`` is load-bearing: it sets ``n_ions`` and hence the ``ion_mode_indices`` partition.
    local_reference_frequencies_rad_s: np.ndarray  #: (n,) local gauge ``ω_local,j > 0``
    coordinate_frame: str  #: wire-ordering tag (see :data:`_CANONICAL_COORDINATE_FRAME`)

    @property
    def n_coords(self) -> int:
        return int(self.frequencies_rad_s.shape[0])

    @property
    def n_ions(self) -> int:
        return int(self.masses_kg.shape[0])

    @property
    def axes_per_ion(self) -> int:
        return self.n_coords // self.n_ions


def materialize_ion_mode_basis(
    *,
    schema_version: int,
    frequencies_rad_s: np.ndarray,
    mass_weighted_eigenvectors: np.ndarray,
    masses_kg: np.ndarray,
    local_reference_frequencies_rad_s: np.ndarray,
    coordinate_frame: str,
) -> IonModeBasis:
    """Validate a producer ``IonModeBasis`` payload and materialize an immutable record.

    Enforces the consumer contract (card §4): matching :data:`ION_MODE_BASIS_SCHEMA_VERSION`;
    real (not complex), finite arrays of consistent shape (``frequencies_rad_s`` a 1-D ``(n,)``,
    ``B`` a square ``(n, n)``, ``n = N`` or ``3N``); positive mode/local frequencies and masses;
    and an orthonormal basis ``B`` (``‖BᵀB − 𝟙‖ ≤`` :data:`_GRAM_TOL`). Returns a record whose arrays
    are **private copies, marked read-only** — the materialized record is genuinely immutable. The
    definitive symplecticity of the built ``S`` is still enforced downstream by
    :func:`gaussian.congruence`.

    Raises
    ------
    ValueError
        On schema mismatch (``schema_version`` must be exactly the bound int), a non-canonical
        ``coordinate_frame``, complex arrays, wrong/inconsistent shapes (scalar or ``(n, 1)``
        frequencies; ``axes_per_ion`` outside ``{1, 3}``), non-finite entries, non-positive
        frequencies/masses, or a non-orthonormal basis.
    """
    if type(schema_version) is not int or schema_version != ION_MODE_BASIS_SCHEMA_VERSION:
        raise ValueError(
            f"IonModeBasis: schema_version {schema_version!r} ≠ the consumer-owned int "
            f"{ION_MODE_BASIS_SCHEMA_VERSION!r}; producer/consumer contract mismatch (the wire type "
            f"is a bound int — ``True``/``1.0`` are not accepted)."
        )
    if coordinate_frame != _CANONICAL_COORDINATE_FRAME:
        raise ValueError(
            f"IonModeBasis: coordinate_frame {coordinate_frame!r} ≠ the canonical wire ordering "
            f"{_CANONICAL_COORDINATE_FRAME!r}. The row/mode ordering is a pinned wire invariant "
            f"that ion_mode_indices depends on; a mismatch is a loud error, not a silent misread "
            f"(a non-canonical ordering yields a wrong-but-symplectic S that congruence cannot catch)."
        )

    raw = {
        "frequencies_rad_s": frequencies_rad_s,
        "mass_weighted_eigenvectors": mass_weighted_eigenvectors,
        "masses_kg": masses_kg,
        "local_reference_frequencies_rad_s": local_reference_frequencies_rad_s,
    }
    for name, value in raw.items():
        if np.iscomplexobj(np.asarray(value)):
            raise ValueError(
                f"IonModeBasis: {name} is complex — emit real float64 arrays; casting to float "
                f"would silently discard the imaginary part."
            )

    freqs = np.asarray(frequencies_rad_s, dtype=float)
    basis = np.asarray(mass_weighted_eigenvectors, dtype=float)
    masses = np.asarray(masses_kg, dtype=float)
    local = np.asarray(local_reference_frequencies_rad_s, dtype=float)

    for name, arr in (
        ("frequencies_rad_s", freqs),
        ("mass_weighted_eigenvectors", basis),
        ("masses_kg", masses),
        ("local_reference_frequencies_rad_s", local),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"IonModeBasis: {name} has non-finite entries (NaN/inf).")

    if freqs.ndim != 1 or freqs.shape[0] == 0:
        raise ValueError(
            f"IonModeBasis: frequencies_rad_s must be a 1-D (n,) array with n > 0; got shape "
            f"{freqs.shape} (a scalar or (n, 1) column is rejected)."
        )
    n = freqs.shape[0]
    if basis.shape != (n, n):
        raise ValueError(
            f"IonModeBasis: mass_weighted_eigenvectors must be square (n, n) = ({n}, {n}) to "
            f"match frequencies_rad_s; got {basis.shape}."
        )
    if local.shape != (n,):
        raise ValueError(
            f"IonModeBasis: local_reference_frequencies_rad_s must have shape ({n},); "
            f"got {local.shape}."
        )
    if masses.ndim != 1 or masses.shape[0] == 0 or n % masses.shape[0] != 0:
        raise ValueError(
            f"IonModeBasis: masses_kg must be (N,) with n_coords ({n}) a positive multiple of "
            f"N; got shape {masses.shape}."
        )
    axes_per_ion = n // masses.shape[0]
    if axes_per_ion not in (1, 3):
        raise ValueError(
            f"IonModeBasis: n_coords ({n}) must be N (axial reduction) or 3N (full trap) for "
            f"N = {masses.shape[0]} ions — axes_per_ion ∈ {{1, 3}}; got {axes_per_ion}."
        )
    if not np.all(freqs > 0.0):
        raise ValueError("IonModeBasis: mode frequencies_rad_s must be strictly positive.")
    if not np.all(local > 0.0):
        raise ValueError(
            "IonModeBasis: local_reference_frequencies_rad_s must be strictly positive."
        )
    if not np.all(masses > 0.0):
        raise ValueError("IonModeBasis: masses_kg must be strictly positive.")

    gram_residual = float(np.max(np.abs(basis.T @ basis - np.eye(n))))
    if gram_residual > _GRAM_TOL:
        raise ValueError(
            f"IonModeBasis: mass_weighted_eigenvectors is not orthonormal — ‖BᵀB − 𝟙‖ = "
            f"{gram_residual:.2e} exceeds {_GRAM_TOL:.0e} (§11 mode-basis convention). A genuine "
            f"eigenbasis is orthonormal to ~1e-15; a residual this large usually means a lossy "
            f"wire (too few significant figures) — emit B at full float64 precision."
        )

    # Materialize a genuinely immutable record: own a private copy of each array (so later mutation
    # of the caller's payload cannot reach in) and mark it read-only (so the record's own arrays
    # cannot be mutated to bypass the validated Gram / positivity guarantees).
    def _frozen(arr: np.ndarray) -> np.ndarray:
        out = np.array(arr, dtype=float, copy=True)
        out.flags.writeable = False
        return out

    return IonModeBasis(
        frequencies_rad_s=_frozen(freqs),
        mass_weighted_eigenvectors=_frozen(basis),
        masses_kg=_frozen(masses),
        local_reference_frequencies_rad_s=_frozen(local),
        coordinate_frame=coordinate_frame,
    )


def normal_to_local_symplectic(basis: IonModeBasis) -> np.ndarray:
    r"""Build the normal→local symplectic ``S`` (§27 per-mode ordering) from a mode basis.

    ``S`` maps ``R_local = S R_normal``; feed it (with a normal-basis covariance) to
    :func:`gaussian.congruence`, or use :func:`to_local_covariance`. Symplectic by construction
    for an orthonormal ``B`` (``X Pᵀ = 𝟙``); ``congruence`` re-verifies ``S Ω Sᵀ = Ω`` exactly.
    """
    n = basis.n_coords
    b = basis.mass_weighted_eigenvectors
    omega_mode = basis.frequencies_rad_s
    omega_local = basis.local_reference_frequencies_rad_s

    ratio = omega_local[:, None] / omega_mode[None, :]  # (ω_local,j / ω_m)
    x_block = b * np.sqrt(ratio)  # X_{jm} = B_{jm} √(ω_local,j / ω_m)
    p_block = b / np.sqrt(ratio)  # P_{jm} = B_{jm} √(ω_m / ω_local,j)

    s_block = np.zeros((2 * n, 2 * n))  # block ordering R = (x₁…x_n, p₁…p_n)
    s_block[:n, :n] = x_block
    s_block[n:, n:] = p_block

    # Permute to §27 per-mode ordering (x̂₁, p̂₁, …): x_k → 2k, p_k → 2k+1.
    perm = np.empty(2 * n, dtype=int)
    perm[0::2] = np.arange(n)
    perm[1::2] = np.arange(n) + n
    return np.asarray(s_block[np.ix_(perm, perm)], dtype=float)


def to_local_covariance(basis: IonModeBasis, cov_normal: np.ndarray) -> np.ndarray:
    """Transform a normal-basis covariance ``V`` into the local-ion basis ``S V Sᵀ``.

    Thin wrapper over :func:`gaussian.congruence` with ``S`` from
    :func:`normal_to_local_symplectic`; inherits its ``S Ω Sᵀ = Ω`` gate and input/output
    physicality validation (so a rough non-orthonormal basis surfaces as a clear error).
    """
    return congruence(normal_to_local_symplectic(basis), np.asarray(cov_normal, dtype=float))


def ion_mode_indices(basis: IonModeBasis, ion_index: int) -> list[int]:
    """Per-mode indices (in the local covariance) of the coordinates belonging to ``ion_index``.

    Use as the ``mode_indices`` cut for :func:`gaussian.log_negativity` to read the
    ion-vs-rest entanglement. For a full trap this is a 3-coordinate block; for an axial
    reduction (``axes_per_ion = 1``) it is a single index.
    """
    if not 0 <= ion_index < basis.n_ions:
        raise ValueError(
            f"ion_mode_indices: ion_index {ion_index} out of range [0, {basis.n_ions})."
        )
    a = basis.axes_per_ion
    return [a * ion_index + c for c in range(a)]
