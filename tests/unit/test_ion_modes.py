# SPDX-License-Identifier: MIT
"""Unit tests for the GT3b ion normal→local symplectic adapter (``ion_modes``).

Physics acceptance (card ``TC-gt3b-ion-symplectic-adapter.md`` v0.3 §7): canonicality
(``SΩSᵀ=Ω`` via GT3a), the passive/common-frequency separability limit, a selected coupled
two-ion ground state cross-checked against an **independent QuTiP diagonalization**, a squeezed
normal input, the equal-mass COM/stretch ``3^{±1/4}`` weights, mixed mass (mass cancellation),
and the local-reference-frequency gauge invariance of the ion-cut ``E_N``.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip

from iontrap_dynamics import gaussian as g
from iontrap_dynamics import ion_modes as im

_FRAME = "ion-major-axis-minor;row=axes_per_ion*i+c;col=mode"


def _modes_from_stiffness(k: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mass-symmetrised normal modes of a physical stiffness ``K``: ``D = M^{-1/2} K M^{-1/2}``."""
    m_half = np.diag(1.0 / np.sqrt(masses))
    eigvals, eigvecs = np.linalg.eigh(m_half @ k @ m_half)
    return eigvecs, np.sqrt(eigvals)  # columns = modes, ω_m = √eigval


def _make_basis(
    b: np.ndarray, omega: np.ndarray, omega_local: np.ndarray, masses: np.ndarray
) -> im.IonModeBasis:
    return im.materialize_ion_mode_basis(
        schema_version=im.ION_MODE_BASIS_SCHEMA_VERSION,
        frequencies_rad_s=omega,
        mass_weighted_eigenvectors=b,
        masses_kg=masses,
        local_reference_frequencies_rad_s=omega_local,
        coordinate_frame=_FRAME,
    )


def _qutip_ground_covariance(
    k: np.ndarray, masses: np.ndarray, omega_local: np.ndarray, fock: int = 24
) -> np.ndarray:
    """Independent oracle: local-basis covariance of the coupled quadratic-Hamiltonian ground
    state, from a full Fock-space diagonalization (no reference to the adapter's ``S``).

    Nondimensionalises each local coordinate ``i`` at ``(m_i, ω_local,i)`` (§26.2, vacuum
    variance 1): ``p_i²/2m_i → (ω_local,i/4) p̂_i²`` and ``½ Σ K_ij u_i u_j →
    Σ K_ij x̂_i x̂_j / (4√(m_i m_j ω_local,i ω_local,j))``.
    """
    n = len(masses)
    annih = [
        qutip.tensor([qutip.destroy(fock) if q == i else qutip.qeye(fock) for q in range(n)])
        for i in range(n)
    ]
    x = [o + o.dag() for o in annih]
    p = [1j * (o.dag() - o) for o in annih]
    ham = sum((omega_local[i] / 4.0) * p[i] * p[i] for i in range(n))
    for i in range(n):
        for j in range(n):
            scale = 4.0 * np.sqrt(masses[i] * masses[j] * omega_local[i] * omega_local[j])
            ham += (k[i, j] / scale) * x[i] * x[j]
    cov, _ = g.covariance_matrix(ham.groundstate()[1])
    return cov


# --- payload contract (materialize) ---------------------------------------------


def test_materialize_validates_the_contract() -> None:
    b = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    good = dict(
        schema_version=im.ION_MODE_BASIS_SCHEMA_VERSION,
        frequencies_rad_s=np.array([1.0, np.sqrt(3.0)]),
        mass_weighted_eigenvectors=b,
        masses_kg=np.array([1.0, 1.0]),
        local_reference_frequencies_rad_s=np.array([1.0, 1.0]),
        coordinate_frame=_FRAME,
    )
    basis = im.materialize_ion_mode_basis(**good)  # type: ignore[arg-type]
    assert (basis.n_coords, basis.n_ions, basis.axes_per_ion) == (2, 2, 1)

    with pytest.raises(ValueError, match="schema_version"):
        im.materialize_ion_mode_basis(**{**good, "schema_version": 999})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not orthonormal"):
        im.materialize_ion_mode_basis(**{**good, "mass_weighted_eigenvectors": np.eye(2) * 1.3})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly positive"):
        im.materialize_ion_mode_basis(**{**good, "frequencies_rad_s": np.array([1.0, -2.0])})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square"):
        im.materialize_ion_mode_basis(**{**good, "mass_weighted_eigenvectors": np.ones((2, 3))})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-finite"):
        im.materialize_ion_mode_basis(  # type: ignore[arg-type]
            **{**good, "local_reference_frequencies_rad_s": np.array([1.0, np.nan])}
        )
    with pytest.raises(ValueError, match="canonical wire ordering"):
        im.materialize_ion_mode_basis(**{**good, "coordinate_frame": ""})  # type: ignore[arg-type]


def test_materialized_record_is_immutable() -> None:
    # Regression (adversarial): the "immutable" record must own read-only copies — a caller must not
    # be able to mutate B after validation and bypass the Gram guarantee, nor reach in through the
    # original payload arrays.
    b = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    freqs = np.array([1.0, np.sqrt(3.0)])
    basis = _make_basis(b, freqs, np.array([1.0, 1.0]), np.ones(2))
    with pytest.raises(ValueError, match="read-only"):
        basis.mass_weighted_eigenvectors[0, 0] = 99.0  # the record's arrays are read-only
    b[0, 0] = 99.0  # mutating the caller's originals after materialize …
    freqs[0] = 42.0
    assert basis.mass_weighted_eigenvectors[0, 0] == pytest.approx(
        1.0 / np.sqrt(2.0)
    )  # … no effect
    assert basis.frequencies_rad_s[0] == pytest.approx(1.0)


def test_materialize_rejects_malformed_boundary_inputs() -> None:
    b = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    good = dict(
        schema_version=im.ION_MODE_BASIS_SCHEMA_VERSION,
        frequencies_rad_s=np.array([1.0, np.sqrt(3.0)]),
        mass_weighted_eigenvectors=b,
        masses_kg=np.array([1.0, 1.0]),
        local_reference_frequencies_rad_s=np.array([1.0, 1.0]),
        coordinate_frame=_FRAME,
    )
    for bad_version in (True, 1.0):  # exact int — bool/float that == 1 are still rejected
        with pytest.raises(ValueError, match="schema_version"):
            im.materialize_ion_mode_basis(**{**good, "schema_version": bad_version})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="complex"):  # complex not silently cast to real
        im.materialize_ion_mode_basis(**{**good, "mass_weighted_eigenvectors": b.astype(complex)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1-D"):  # (n, 1) column frequencies rejected
        im.materialize_ion_mode_basis(**{**good, "frequencies_rad_s": np.array([[1.0], [2.0]])})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="axes_per_ion"):  # n=2, N=1 → axes_per_ion=2 rejected
        im.materialize_ion_mode_basis(**{**good, "masses_kg": np.array([1.0])})  # type: ignore[arg-type]


def test_non_canonical_coordinate_frame_is_rejected() -> None:
    # Regression (adversarial): the coordinate_frame is a pinned wire invariant that
    # ion_mode_indices depends on — a mislabelled/different ordering yields a wrong-but-symplectic
    # S that congruence cannot catch, so it must be a loud rejection at materialize (not silently
    # accepted like any non-empty tag). Only the exact canonical ordering is admitted.
    b = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    kw = dict(
        schema_version=im.ION_MODE_BASIS_SCHEMA_VERSION,
        frequencies_rad_s=np.array([1.0, np.sqrt(3.0)]),
        mass_weighted_eigenvectors=b,
        masses_kg=np.array([1.0, 1.0]),
        local_reference_frequencies_rad_s=np.array([1.0, 1.0]),
    )
    with pytest.raises(ValueError, match="canonical wire ordering"):
        im.materialize_ion_mode_basis(**kw, coordinate_frame="axis-major-ion-minor")  # type: ignore[arg-type]
    # the exact canonical string is accepted
    assert im.materialize_ion_mode_basis(**kw, coordinate_frame=_FRAME).coordinate_frame == _FRAME  # type: ignore[arg-type]


def test_gram_tolerance_is_coherent_with_congruence() -> None:
    # Regression (adversarial): the materialize Gram gate must be aligned with congruence's 1e-11
    # symplecticity gate so that *acceptance implies usability* — a basis that materializes must
    # not then crash in to_local_covariance with "S is not symplectic". A basis whose Gram residual
    # lands in the old (1e-11, 1e-8] window is now rejected *at materialize* with a clear error…
    theta = 0.6
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    b_rough = rot @ (np.eye(2) + 1e-9 * np.array([[0.0, 1.0], [1.0, 0.0]]))  # Gram ~2e-9
    with pytest.raises(ValueError, match="not orthonormal"):
        _make_basis(b_rough, np.array([1.0, 5.0]), np.array([1.0, 1.0]), np.ones(2))
    # … while a machine-precision eigenbasis materializes AND transforms without raising.
    b, omega = _modes_from_stiffness(np.array([[2.0, -0.7], [-0.7, 1.9]]), np.array([1.0, 2.3]))
    basis = _make_basis(b, omega, np.array([3.1, 0.4]), np.array([1.0, 2.3]))
    im.to_local_covariance(basis, np.eye(4))  # no "S is not symplectic" — accept ⇒ usable


# --- canonicality + purity preservation -----------------------------------------


def test_symplectic_map_is_canonical_and_purity_preserving() -> None:
    rng = np.random.default_rng(0)
    cases = [
        # equal-mass COM/stretch
        (
            np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0),
            np.array([1.0, np.sqrt(3.0)]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        ),
    ]
    for _ in range(4):  # mixed-mass random stiffness + gauge
        a = rng.uniform(0.5, 2.0, size=(3, 3))
        k = a @ a.T + 3.0 * np.eye(3)  # SPD stiffness
        masses = rng.uniform(1.0, 4.0, size=3)
        b, omega = _modes_from_stiffness(k, masses)
        cases.append((b, omega, rng.uniform(0.4, 2.5, size=3), masses))

    for b, omega, omega_local, masses in cases:
        basis = _make_basis(b, omega, omega_local, masses)
        n = basis.n_coords
        s = im.normal_to_local_symplectic(basis)
        # congruence accepts S (SΩSᵀ=Ω) and preserves the Williamson spectrum of any physical V
        out_vac = g.congruence(s, np.eye(2 * n))
        assert g.symplectic_eigenvalues(out_vac) == pytest.approx(np.ones(n), abs=1e-9)
        thermal = np.diag(np.repeat(rng.uniform(1.5, 4.0, size=n), 2))
        assert np.sort(g.symplectic_eigenvalues(im.to_local_covariance(basis, thermal))) == (
            pytest.approx(np.sort(g.symplectic_eigenvalues(thermal)), rel=1e-8)
        )


def test_com_stretch_weights_are_three_to_the_quarter() -> None:
    # Equal-mass 2 ions: participation both 1/√2, ω_stretch/ω_COM = √3, so the stretch column
    # of X carries 3^{-1/4} and of P carries 3^{+1/4} (card §7.5) — NOT a √3 participation.
    b = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    basis = _make_basis(b, np.array([1.0, np.sqrt(3.0)]), np.array([1.0, 1.0]), np.ones(2))
    s = im.normal_to_local_symplectic(basis)  # §27 per-mode ordering (x0,p0,x1,p1)
    # x rows are 0,2; p rows 1,3. Column 1 (per-mode index 2,3) is the stretch mode.
    x_stretch = np.abs(s[np.ix_([0, 2], [2])]).ravel()
    p_stretch = np.abs(s[np.ix_([1, 3], [3])]).ravel()
    assert x_stretch == pytest.approx(np.full(2, 3.0**-0.25 / np.sqrt(2.0)), rel=1e-12)
    assert p_stretch == pytest.approx(np.full(2, 3.0**0.25 / np.sqrt(2.0)), rel=1e-12)


# --- passive limit: thermal product stays separable -----------------------------


def test_passive_limit_maps_thermal_product_to_separable() -> None:
    # Common frequency (ω_local = ω_m = common) ⇒ X = P = B (a rotation): a thermal normal
    # product maps to a separable local state (Kim 2002 passive-mixing regime, card §7.2).
    theta = 0.7
    b = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    basis = _make_basis(b, np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.ones(2))
    v_normal = np.diag([3.0, 3.0, 5.0, 5.0])  # two thermal modes, ν = 3 and 5
    assert g.log_negativity(im.to_local_covariance(basis, v_normal), [0]) == pytest.approx(
        0.0, abs=1e-9
    )


# --- the strong independent oracle: coupled ground state vs QuTiP ----------------


@pytest.mark.parametrize(
    ("k", "masses", "omega_local"),
    [
        (np.array([[2.0, -1.0], [-1.0, 2.0]]), np.array([1.0, 1.0]), np.array([1.0, 1.0])),
        (np.array([[2.0, -0.7], [-0.7, 1.5]]), np.array([1.0, 2.5]), np.array([1.3, 0.8])),
    ],
)
def test_local_covariance_matches_independent_qutip_ground_state(
    k: np.ndarray, masses: np.ndarray, omega_local: np.ndarray
) -> None:
    b, omega = _modes_from_stiffness(k, masses)
    basis = _make_basis(b, omega, omega_local, masses)
    v_local = im.to_local_covariance(basis, np.eye(4))  # map the normal-mode vacuum
    v_qutip = _qutip_ground_covariance(k, masses, omega_local)
    assert v_local == pytest.approx(v_qutip, abs=1e-6)
    assert g.log_negativity(v_local, im.ion_mode_indices(basis, 0)) == pytest.approx(
        g.log_negativity(v_qutip, [0]), abs=1e-6
    )
    assert g.log_negativity(v_local, im.ion_mode_indices(basis, 0)) > 0.05  # actively entangled


def test_squeezed_normal_input_is_propagated_canonically() -> None:
    # A squeezed normal-mode input (globally pure) maps to a pure local state (ν preserved),
    # cross-checked against an inline, independent S = diag(X, P) construction.
    b, omega = _modes_from_stiffness(np.array([[3.0, -0.9], [-0.9, 2.2]]), np.array([1.0, 1.7]))
    omega_local = np.array([1.1, 0.9])
    basis = _make_basis(b, omega, omega_local, np.array([1.0, 1.7]))
    r = 0.6
    v_normal = np.diag([np.exp(2 * r), np.exp(-2 * r), 1.0, 1.0])  # mode 0 squeezed, pure
    v_local = im.to_local_covariance(basis, v_normal)
    assert g.symplectic_eigenvalues(v_local) == pytest.approx(np.ones(2), abs=1e-9)  # still pure
    # inline independent S (block form, permuted) — a separate code path
    ratio = omega_local[:, None] / omega[None, :]
    x, p = b * np.sqrt(ratio), b / np.sqrt(ratio)
    s_block = np.block([[x, np.zeros((2, 2))], [np.zeros((2, 2)), p]])
    perm = [0, 2, 1, 3]
    s_ref = s_block[np.ix_(perm, perm)]
    assert v_local == pytest.approx(s_ref @ v_normal @ s_ref.T, abs=1e-12)


# --- D3: local reference frequency is a gauge (E_N invariant) --------------------


def test_local_frequency_is_an_entanglement_neutral_gauge() -> None:
    b, omega = _modes_from_stiffness(np.array([[2.0, -0.8], [-0.8, 1.6]]), np.array([1.0, 1.4]))
    masses = np.array([1.0, 1.4])
    base = im.to_local_covariance(_make_basis(b, omega, np.array([1.0, 1.0]), masses), np.eye(4))
    gauged = im.to_local_covariance(_make_basis(b, omega, np.array([4.2, 0.3]), masses), np.eye(4))
    # ion-cut E_N is invariant under the local-squeeze gauge …
    assert g.log_negativity(gauged, [0]) == pytest.approx(g.log_negativity(base, [0]), abs=1e-9)
    # … while the local occupation genuinely changes (it is a different representation).
    assert g.mean_occupation(gauged, np.zeros(4)) != pytest.approx(
        g.mean_occupation(base, np.zeros(4)), rel=1e-3
    )


def test_ion_mode_indices() -> None:
    b, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((6, 6)))  # random orthonormal
    basis = _make_basis(b, np.arange(1.0, 7.0), np.ones(6), np.array([1.0, 1.0]))  # 2 ions × 3 axes
    assert basis.axes_per_ion == 3
    assert im.ion_mode_indices(basis, 0) == [0, 1, 2]
    assert im.ion_mode_indices(basis, 1) == [3, 4, 5]
    with pytest.raises(ValueError, match="out of range"):
        im.ion_mode_indices(basis, 2)
