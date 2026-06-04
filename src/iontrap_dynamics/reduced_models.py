# SPDX-License-Identifier: MIT
"""Reduced light–matter model Hamiltonians (CONVENTIONS.md §25).

Abstract qubit–oscillator models — Jaynes–Cummings (JC), anti-Jaynes–Cummings
(AJC), and the quantum Rabi model (QRM) — as **physics-layer** objects: what a
trapped-ion apparatus *approximates*, distinct from the §5 apparatus/drive
builders in :mod:`iontrap_dynamics.hamiltonians` that *realise* them.

Per §25 these are **Schrödinger-picture** Hamiltonians that intentionally carry
the **bare** atomic term ``½ω₀ σ_z`` — unlike the §5 interaction-picture
builders, whose free atomic term is transformed away (see the §5 scope note).
Each is a static, Hermitian :class:`qutip.Qobj` in ``H/ℏ`` units of rad·s⁻¹ on a
single-ion ⊗ single-mode embedding (§2 tensor order, §3 spin basis), with
σ_z = |↑⟩⟨↑| − |↓⟩⟨↓| and σ_+ = |↑⟩⟨↓| and â the mode annihilation operator:

* JC  : ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g(â σ_+ + â† σ_−)``  (co-rotating)
* AJC : ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g(â† σ_+ + â σ_−)``  (counter-rotating)
* QRM : ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g σ_x(â + â†)``       (full dipole, non-RWA)

``ω₀`` is an effective model splitting and may be negative (§25.2); ``ω_f`` is
the oscillator frequency (builder kwarg ``omega_f``); ``g`` the coupling. The
LOCK-3 identity ``H_AJC(ω₀) = σ_x H_JC(−ω₀) σ_x`` (§25.3) and the U(1)/Z₂
symmetry contrast are enforced by ``tests/conventions/test_reduced_models_conventions.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import qutip
from numpy.typing import NDArray

from .exceptions import ConventionError
from .hilbert import HilbertSpace
from .operators import sigma_minus_ion, sigma_plus_ion, sigma_x_ion, sigma_z_ion
from .results import TrajectoryResult


def _validate_couplings(*, omega_0: float, omega_f: float, g: float) -> None:
    """Validate the model scalars before they reach a ``Qobj`` (§25).

    ``omega_0`` (an effective splitting, §25.2) and ``g`` may carry either sign but
    must be finite. ``omega_f`` is a genuine oscillator frequency — §25 grants the
    negative-sign semantics to ``omega_0`` alone — so it must be finite and
    positive, like the §10/§11 mode frequency it realises (cf. ``ModeConfig``).
    """
    for name, value in (("omega_0", omega_0), ("g", g)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value}.")
    if not math.isfinite(omega_f) or omega_f <= 0.0:
        raise ConventionError(
            f"omega_f must be a finite positive oscillator frequency (§25); got {omega_f}."
        )


def _bare_hamiltonian(
    hilbert: HilbertSpace,
    mode_label: str,
    ion_index: int,
    omega_0: float,
    omega_f: float,
) -> qutip.Qobj:
    """The bare term ``½ω₀ σ_z + ω_f â†â`` shared by JC/AJC/QRM (§25.1)."""
    sigma_z = hilbert.spin_op_for_ion(sigma_z_ion(), ion_index)
    number = hilbert.number_for_mode(mode_label)
    return 0.5 * omega_0 * sigma_z + omega_f * number


def jaynes_cummings_hamiltonian(
    hilbert: HilbertSpace,
    mode_label: str,
    *,
    ion_index: int,
    omega_0: float,
    omega_f: float,
    g: float,
) -> qutip.Qobj:
    """Return the Jaynes–Cummings Hamiltonian (co-rotating, RWA) — CONVENTIONS.md §25.

    ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g(â σ_+ + â† σ_−)``.

    The co-rotating coupling conserves the excitation number
    ``N̂ = â†â + |↑⟩⟨↑|`` (a U(1) symmetry): it couples ``|↑, n⟩`` to
    ``|↓, n+1⟩`` with matrix element ``g√(n+1)``, leaving ``|↓, 0⟩`` dark.

    Parameters
    ----------
    hilbert
        The full tensor-product Hilbert space (one ion ⊗ the selected mode).
    mode_label
        Label of the bosonic mode the qubit couples to.
    ion_index
        Zero-based index of the qubit ion. Keyword-only to keep the API
        unambiguous on multi-ion / multi-mode spaces.
    omega_0
        Effective qubit splitting ω₀ (rad·s⁻¹); may be negative (§25.2).
    omega_f
        Oscillator frequency ω_f (rad·s⁻¹); must be finite and positive.
    g
        Coupling strength g (rad·s⁻¹); may be negative.

    Returns
    -------
    qutip.Qobj
        Time-independent Hermitian operator on the full space, with dimensions
        :meth:`HilbertSpace.qutip_dims`.

    Raises
    ------
    ValueError
        If ``omega_0`` or ``g`` is non-finite.
    ConventionError
        If ``omega_f`` is not finite and positive, or ``mode_label`` is not a
        mode of ``hilbert``.
    IndexError
        If ``ion_index`` is outside ``[0, n_ions)``.

    See Also
    --------
    anti_jaynes_cummings_hamiltonian
    quantum_rabi_hamiltonian

    Example
    -------
    ::

        import numpy as np
        from iontrap_dynamics import jaynes_cummings_hamiltonian

        H = jaynes_cummings_hamiltonian(
            hilbert, "axial", ion_index=0,
            omega_0=2 * np.pi * 1e6, omega_f=2 * np.pi * 1e6, g=2 * np.pi * 1e3,
        )
    """
    _validate_couplings(omega_0=omega_0, omega_f=omega_f, g=g)
    sigma_p = hilbert.spin_op_for_ion(sigma_plus_ion(), ion_index)
    sigma_m = hilbert.spin_op_for_ion(sigma_minus_ion(), ion_index)
    a = hilbert.annihilation_for_mode(mode_label)
    a_dag = hilbert.creation_for_mode(mode_label)
    bare = _bare_hamiltonian(hilbert, mode_label, ion_index, omega_0, omega_f)
    return bare + g * (a * sigma_p + a_dag * sigma_m)


def anti_jaynes_cummings_hamiltonian(
    hilbert: HilbertSpace,
    mode_label: str,
    *,
    ion_index: int,
    omega_0: float,
    omega_f: float,
    g: float,
) -> qutip.Qobj:
    """Return the anti-Jaynes–Cummings Hamiltonian (counter-rotating) — §25.

    ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g(â† σ_+ + â σ_−)``.

    The counter-rotating coupling conserves the difference number
    ``Ĉ = â†â − |↑⟩⟨↑|``: it couples ``|↓, n⟩`` to ``|↑, n+1⟩`` (its dark state
    is ``|↑, 0⟩``). It is the σ_x conjugate of ``H_JC(−ω₀)`` — the LOCK-3
    identity ``H_AJC(ω₀) = σ_x H_JC(−ω₀) σ_x`` (§25.3).

    Parameters, returns, and raises match :func:`jaynes_cummings_hamiltonian`.
    """
    _validate_couplings(omega_0=omega_0, omega_f=omega_f, g=g)
    sigma_p = hilbert.spin_op_for_ion(sigma_plus_ion(), ion_index)
    sigma_m = hilbert.spin_op_for_ion(sigma_minus_ion(), ion_index)
    a = hilbert.annihilation_for_mode(mode_label)
    a_dag = hilbert.creation_for_mode(mode_label)
    bare = _bare_hamiltonian(hilbert, mode_label, ion_index, omega_0, omega_f)
    return bare + g * (a_dag * sigma_p + a * sigma_m)


def quantum_rabi_hamiltonian(
    hilbert: HilbertSpace,
    mode_label: str,
    *,
    ion_index: int,
    omega_0: float,
    omega_f: float,
    g: float,
) -> qutip.Qobj:
    """Return the quantum Rabi Hamiltonian (full dipole, non-RWA) — §25.

    ``H/ℏ = ½ω₀ σ_z + ω_f â†â + g σ_x(â + â†)``.

    The full dipole coupling ``g σ_x(â + â†)`` is non-RWA (JC + AJC together);
    it conserves neither ``N̂`` nor ``Ĉ``, only the Z₂ parity
    ``P = exp(iπ N̂)``. Below ultra-strong coupling its low-lying spectrum tends
    to the Jaynes–Cummings one as ``g/ω₀ → 0`` (the RWA limit), reached as a
    genuine weak-coupling limit, never by dropping terms.

    Parameters, returns, and raises match :func:`jaynes_cummings_hamiltonian`.
    """
    _validate_couplings(omega_0=omega_0, omega_f=omega_f, g=g)
    sigma_x = hilbert.spin_op_for_ion(sigma_x_ion(), ion_index)
    a = hilbert.annihilation_for_mode(mode_label)
    a_dag = hilbert.creation_for_mode(mode_label)
    bare = _bare_hamiltonian(hilbert, mode_label, ion_index, omega_0, omega_f)
    return bare + g * sigma_x * (a + a_dag)


# ---------------------------------------------------------------------------
# Model-deviation comparison (WP-03 WI-5 / RM2, Dispatch RLE)
# ---------------------------------------------------------------------------

_DEVIATION_METHODS = ("auto", "state_fidelity", "observable_rms")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelDeviation:
    """Deviation summary between two matched trajectories (a reduced model vs a
    realisation), per the RM2 convention.

    Parameters
    ----------
    value
        The pinned scalar deviation: the worst-case (max over time) of ``per_time``.
    method
        Which metric produced it — ``"state_fidelity"`` (``1 − qutip.fidelity`` per
        time step, requires materialised states) or ``"observable_rms"`` (RMS over
        the shared observable expectations, the explicit fallback). Self-describing
        so a caller can tell which path was taken.
    per_time
        The per-time-sample deviation series, aligned to ``times``.
    times
        The shared time grid (seconds).
    """

    value: float
    method: str
    per_time: NDArray[np.float64]
    times: NDArray[np.float64]


def _materialised_states(trajectory: TrajectoryResult) -> tuple[Any, ...] | None:
    """The trajectory's states if materialised (``StorageMode.EAGER`` directly, or
    ``LAZY`` via its ``states_loader``); ``None`` if expectation-only."""
    if trajectory.states is not None:
        return trajectory.states
    if trajectory.states_loader is not None:
        loader = trajectory.states_loader
        return tuple(loader(i) for i in range(len(trajectory.times)))
    return None


def _state_fidelity_deviation(
    states_a: tuple[Any, ...], states_b: tuple[Any, ...]
) -> NDArray[np.float64]:
    """Per-step ``1 − qutip.fidelity`` (the RM2 pin), clamped to ``[0, 1]``.

    ``qutip.fidelity`` is called on the raw states — it handles ket-vs-ket (a
    plain overlap, no matrix square root), ket-vs-density-matrix, and
    density-matrix-vs-density-matrix uniformly, so a pure sesolve trajectory can
    be compared against a mixed mesolve one without promoting either."""
    deviations = np.empty(len(states_a), dtype=np.float64)
    for index, (state_a, state_b) in enumerate(zip(states_a, states_b, strict=True)):
        # Compare the Hilbert-space (ket) dimensions — a ket and a density matrix
        # on the same space have different full Qobj dims but are comparable.
        if state_a.dims[0] != state_b.dims[0]:
            raise ValueError(
                f"model_deviation: states at index {index} live on different Hilbert "
                f"spaces ({state_a.dims[0]} vs {state_b.dims[0]})."
            )
        fidelity = float(qutip.fidelity(state_a, state_b))
        # Clamp: rounding in the matrix square root can push F slightly outside [0, 1].
        deviations[index] = min(max(1.0 - fidelity, 0.0), 1.0)
    return deviations


def _observable_rms_deviation(
    reference: TrajectoryResult,
    comparison: TrajectoryResult,
    observables: Sequence[str] | None,
) -> NDArray[np.float64]:
    """Per-time RMS across the shared observable channels of the two trajectories."""
    shared = set(reference.expectations) & set(comparison.expectations)
    if observables is not None:
        requested = set(observables)
        missing = requested - shared
        if missing:
            raise ValueError(
                f"model_deviation: observables {sorted(missing)} are not present in both trajectories."
            )
        shared = requested
    if not shared:
        raise ValueError(
            "model_deviation: the two trajectories share no observable labels for the RMS fallback."
        )
    labels = sorted(shared)
    n_times = len(reference.times)
    for label in labels:
        for source, array in (
            ("reference", reference.expectations[label]),
            ("comparison", comparison.expectations[label]),
        ):
            shaped = np.asarray(array)
            if shaped.ndim != 1 or shaped.shape[0] != n_times:
                raise ValueError(
                    f"model_deviation: {source} observable {label!r} has shape {shaped.shape}, "
                    f"expected ({n_times},) — one value per time sample."
                )
    diffs = np.stack(
        [
            np.asarray(reference.expectations[label], dtype=np.float64)
            - np.asarray(comparison.expectations[label], dtype=np.float64)
            for label in labels
        ]
    )
    return np.asarray(np.sqrt(np.mean(diffs**2, axis=0)), dtype=np.float64)


def model_deviation(
    reference: TrajectoryResult,
    comparison: TrajectoryResult,
    *,
    method: str = "auto",
    observables: Sequence[str] | None = None,
) -> ModelDeviation:
    """Deviation between two matched trajectories (e.g. a reduced model vs a
    full-ion realisation) — the RM2 comparison summary.

    The two trajectories must share a time grid. With materialised states the
    metric is the worst-case state infidelity ``max_t (1 − qutip.fidelity(ρ_ref(t),
    ρ_cmp(t)))`` (the §25/RM2 pin, QuTiP's public fidelity value); without them it
    falls back to the per-time RMS over the shared observable expectations, and the
    returned :class:`ModelDeviation` says which path was taken via ``method``.

    Parameters
    ----------
    reference, comparison
        The two :class:`~iontrap_dynamics.results.TrajectoryResult` objects to
        compare (order does not affect the deviation).
    method
        ``"auto"`` (default) uses state fidelity when *both* trajectories carry
        materialised states and the observable-RMS fallback otherwise;
        ``"state_fidelity"`` requires materialised states (raises otherwise);
        ``"observable_rms"`` always uses the observable RMS.
    observables
        Optional subset of observable labels for the RMS path; defaults to every
        label shared by both trajectories.

    Returns
    -------
    ModelDeviation
        The scalar deviation, the per-time series, and the metric used.

    Raises
    ------
    ValueError
        If the time grids are not identical or are empty, ``method`` is unknown,
        the RMS path has no shared observables, a per-time array (materialised
        states or an expectation channel) does not match the time grid, compared
        states have mismatched dimensions, or the resulting deviation is non-finite
        (§15 forbids silent degradation).
    ConventionError
        If ``method="state_fidelity"`` but the states are not materialised
        (``StorageMode.EAGER``, or ``StorageMode.LAZY`` with a ``states_loader``).
    """
    if method not in _DEVIATION_METHODS:
        raise ValueError(
            f"model_deviation: unknown method {method!r}; expected one of {_DEVIATION_METHODS}."
        )
    if len(reference.times) != len(comparison.times):
        raise ValueError(
            "model_deviation: the two trajectories have different time-grid lengths "
            f"({len(reference.times)} vs {len(comparison.times)})."
        )
    if len(reference.times) == 0:
        raise ValueError("model_deviation: empty time grid; nothing to compare.")
    # "Matched trajectories" must share one grid: require an identical time axis
    # (np.allclose's default atol=1e-8 would accept a ~10 ns shift on a µs grid).
    if not np.array_equal(reference.times, comparison.times):
        raise ValueError("model_deviation: the two trajectories are on different time grids.")

    states_a = _materialised_states(reference)
    states_b = _materialised_states(comparison)
    both_materialised = states_a is not None and states_b is not None

    if method == "state_fidelity" or (method == "auto" and both_materialised):
        if states_a is None or states_b is None:
            raise ConventionError(
                "model_deviation: state-fidelity requires materialised states; pass trajectories "
                "solved with storage_mode=StorageMode.EAGER (or LAZY with a states_loader), "
                "or use method='observable_rms'."
            )
        n_times = len(reference.times)
        if len(states_a) != n_times or len(states_b) != n_times:
            raise ValueError(
                "model_deviation: materialised state count does not match the time grid "
                f"({len(states_a)} / {len(states_b)} states vs {n_times} times)."
            )
        per_time = _state_fidelity_deviation(states_a, states_b)
        used = "state_fidelity"
    else:
        per_time = _observable_rms_deviation(reference, comparison, observables)
        used = "observable_rms"

    # §15: a non-finite deviation must not propagate silently into a threshold.
    if not np.isfinite(per_time).all():
        raise ValueError(
            "model_deviation: non-finite deviation (NaN/Inf) in the per-time series; "
            "check the input expectation arrays / states for divergence."
        )

    return ModelDeviation(
        value=float(np.max(per_time)),
        method=used,
        per_time=per_time,
        times=np.asarray(reference.times, dtype=np.float64),
    )


__all__ = [
    "ModelDeviation",
    "anti_jaynes_cummings_hamiltonian",
    "jaynes_cummings_hamiltonian",
    "model_deviation",
    "quantum_rabi_hamiltonian",
]
