# SPDX-License-Identifier: MIT
"""Protocol-layer composers — named measurement procedures.

A *protocol* wraps the channel-and-detector plumbing into a single
procedure the experimentalist recognises by name (``spin_readout``,
``parity_scan``, ``sideband_flopping``). Each protocol spec is a frozen
dataclass; its ``.run(trajectory, *, shots, seed)`` method consumes a
:class:`TrajectoryResult`, executes the measurement, and returns a
:class:`MeasurementResult` with the dual-view ideal / sampled payload.

Dispatch M adds :class:`SpinReadout` — the prototype single-ion
protocol. Dispatch N adds :class:`ParityScan`, which reads two ions
jointly on the same shot so entanglement-bearing correlations (Bell
states, CHSH, Mølmer–Sørensen verification) survive the measurement.
Sideband-flopping inference (Dispatch O) follows the same shape:
construct a spec, call ``.run()``, consume the
:class:`MeasurementResult`.

Projective-shot readout model
-----------------------------

:class:`SpinReadout` uses the experimentally faithful *projective*
readout model, not the rate-averaged model used by the raw
:class:`PoissonChannel` pipeline:

    each shot  →  project qubit to bright (prob ``p_↑``) or dark
               →  sample Poisson at state-conditional rate
                     bright-branch rate = η · λ_bright + γ_d
                     dark-branch   rate = η · λ_dark   + γ_d
               →  threshold count against ``N̂`` → bright/dark bit

This model is correct for real ion-trap readout, where the detection
laser optically pumps the qubit into a pinned bright or dark cycling
transition for the detection window — the qubit "collapses" at the
start of the window and emits photons at a single state-conditional
rate for the remainder. The rate-averaged model in Dispatch L's demo
is a different limit (coherent emission during fast dynamics), and its
infinite-shots envelope is non-linear in ``p_↑``; the projective model
gives the clean linear envelope

    bright_fraction∞(t) = TP · p_↑(t) + (1 − TN) · (1 − p_↑(t))

where ``TP`` and ``TN`` come from
:meth:`DetectorConfig.classification_fidelity`. Callers comparing
dynamics predictions to experimental readout should use this protocol,
not the raw ``sample_outcome(PoissonChannel(), …)`` pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..exceptions import ConventionError
from ..results import (
    MeasurementResult,
    ResultMetadata,
    StorageMode,
    TrajectoryResult,
)
from .detectors import DetectorConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class SpinReadout:
    """Projective spin-state readout protocol (Dispatch M).

    Parameters
    ----------
    ion_index
        Zero-based index of the ion being read out. The protocol looks
        up ``trajectory.expectations[f"sigma_z_{ion_index}"]`` at
        :meth:`run` time; the trajectory must carry that observable.
    detector
        :class:`DetectorConfig` capturing the readout apparatus
        (efficiency, dark-count rate, threshold). See §17.8.
    lambda_bright
        Emitted photon rate per shot when the qubit is pinned bright,
        in counts per detection window. Non-negative.
    lambda_dark
        Emitted photon rate per shot when the qubit is pinned dark,
        in counts per detection window. Non-negative and
        ``<= lambda_bright``.
    label
        Identifier prefix for the entries the protocol writes into
        :attr:`MeasurementResult.sampled_outcome`. Defaults to
        ``"spin_readout"``; override when a result carries multiple
        simultaneous readouts (e.g. on different ions).

    Raises
    ------
    ValueError
        At construction, if ``ion_index < 0``, rates are negative, or
        ``lambda_dark > lambda_bright``.
    """

    ion_index: int
    detector: DetectorConfig
    lambda_bright: float
    lambda_dark: float
    label: str = "spin_readout"

    def __post_init__(self) -> None:
        if self.ion_index < 0:
            raise ValueError(f"SpinReadout: ion_index must be >= 0; got {self.ion_index}")
        if self.lambda_bright < 0.0 or self.lambda_dark < 0.0:
            raise ValueError(
                "SpinReadout: rates must be >= 0; "
                f"got lambda_bright={self.lambda_bright}, "
                f"lambda_dark={self.lambda_dark}"
            )
        if self.lambda_dark > self.lambda_bright:
            raise ValueError(
                "SpinReadout: lambda_dark must be <= lambda_bright; "
                f"got {self.lambda_dark} > {self.lambda_bright}"
            )

    def run(
        self,
        trajectory: TrajectoryResult,
        *,
        shots: int,
        seed: int | None = None,
        provenance_tags: tuple[str, ...] = (),
    ) -> MeasurementResult:
        """Execute the protocol against ``trajectory``.

        Parameters
        ----------
        trajectory
            Upstream :class:`TrajectoryResult`; must carry
            ``sigma_z_{ion_index}`` under :attr:`expectations`.
        shots
            Number of independent readout shots per time point. ``>= 1``.
        seed
            Optional seed for :func:`numpy.random.default_rng`. When
            supplied, the result is bit-reproducible given
            ``(protocol, trajectory, shots, seed)``.
        provenance_tags
            Extra tags concatenated onto the inherited provenance
            chain after ``"measurement"`` and ``"spin_readout"``.

        Returns
        -------
        MeasurementResult
            With ``ideal_outcome = {"p_up": ..., "bright_fraction_envelope": ...}``
            and ``sampled_outcome = {f"{label}_counts": (shots, n_times) int64,
            f"{label}_bits": (shots, n_times) int8,
            f"{label}_bright_fraction": (n_times,) float64}``. The
            ``trajectory_hash`` field inherits the upstream
            ``request_hash``.

        Raises
        ------
        ConventionError
            If the trajectory has no ``sigma_z_{ion_index}`` expectation.
        ValueError
            If ``shots < 1`` or if the trajectory's ``p_up`` leaves the
            valid ``[0, 1]`` range (indicates a buggy upstream solve).
        """
        if shots < 1:
            raise ValueError(f"SpinReadout.run: shots must be >= 1; got {shots}")

        observable_key = f"sigma_z_{self.ion_index}"
        if observable_key not in trajectory.expectations:
            raise ConventionError(
                f"SpinReadout.run: trajectory has no '{observable_key}' "
                f"expectation (available: {sorted(trajectory.expectations)})"
            )

        sigma_z = np.asarray(trajectory.expectations[observable_key], dtype=np.float64)
        p_up = 0.5 * (1.0 + sigma_z)
        # A well-formed trajectory should keep |⟨σ_z⟩| ≤ 1 to within
        # floating-point noise; clip the tiny over-/under-shoots from
        # ODE integrator slop before they propagate into negative
        # probabilities downstream.
        if np.any((p_up < -1e-9) | (p_up > 1.0 + 1e-9)):
            raise ValueError(
                f"SpinReadout.run: p_up lies outside [0, 1] by more than 1e-9; "
                f"got min={p_up.min()}, max={p_up.max()} — upstream solve is likely buggy"
            )
        p_up = np.clip(p_up, 0.0, 1.0)

        counts, bits, bright_fraction, envelope = _project_and_sample(
            p_up=p_up,
            detector=self.detector,
            lambda_bright=self.lambda_bright,
            lambda_dark=self.lambda_dark,
            shots=shots,
            seed=seed,
        )

        metadata = _inherit_metadata(
            upstream=trajectory,
            provenance_tags=(self.label, *provenance_tags),
        )
        return MeasurementResult(
            metadata=metadata,
            shots=shots,
            rng_seed=seed,
            ideal_outcome={
                "p_up": p_up,
                "bright_fraction_envelope": envelope,
            },
            sampled_outcome={
                f"{self.label}_counts": counts,
                f"{self.label}_bits": bits,
                f"{self.label}_bright_fraction": bright_fraction,
            },
            trajectory_hash=trajectory.metadata.request_hash,
        )


def _project_and_sample(
    *,
    p_up: NDArray[np.float64],
    detector: DetectorConfig,
    lambda_bright: float,
    lambda_dark: float,
    shots: int,
    seed: int | None,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int8],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Shared core of the projective-shot readout pipeline.

    Returns (counts, bits, bright_fraction, envelope) with shapes
    ((shots, n_times), (shots, n_times), (n_times,), (n_times,)).
    """
    rng = np.random.default_rng(seed)
    n_times = p_up.size

    # Per-shot state projection: Bernoulli(p_↑) giving a bright mask.
    uniforms = rng.random(size=(shots, n_times))
    state_bright = uniforms < p_up  # (shots, n_times) bool

    # Effective per-branch rates after detector thinning + dark counts.
    rate_bright_eff = detector.efficiency * lambda_bright + detector.dark_count_rate
    rate_dark_eff = detector.efficiency * lambda_dark + detector.dark_count_rate

    # Per-shot per-time rate — bright branch where state is bright,
    # dark branch elsewhere. Poisson samples are drawn from the combined
    # array in one call (np.random.Generator.poisson accepts arbitrary-
    # shape lam and matches the output shape to it).
    per_shot_rate = np.where(state_bright, rate_bright_eff, rate_dark_eff)
    counts = rng.poisson(per_shot_rate).astype(np.int64)
    bits = detector.discriminate(counts)
    bright_fraction = bits.mean(axis=0).astype(np.float64)

    # Analytic infinite-shots envelope — linear in p_↑ by the
    # projective-shot model (§17.9).
    fidelities = detector.classification_fidelity(
        lambda_bright=lambda_bright, lambda_dark=lambda_dark
    )
    envelope = fidelities["true_positive_rate"] * p_up + (
        1.0 - fidelities["true_negative_rate"]
    ) * (1.0 - p_up)

    return counts, bits, bright_fraction, envelope


def _inherit_metadata(
    *,
    upstream: TrajectoryResult,
    provenance_tags: tuple[str, ...],
) -> ResultMetadata:
    """Copy upstream metadata, appending the measurement chain tags."""
    upstream_meta = upstream.metadata
    return ResultMetadata(
        convention_version=upstream_meta.convention_version,
        request_hash=upstream_meta.request_hash,
        backend_name=upstream_meta.backend_name,
        backend_version=upstream_meta.backend_version,
        storage_mode=StorageMode.OMITTED,
        fock_truncations=upstream_meta.fock_truncations,
        provenance_tags=(*upstream_meta.provenance_tags, "measurement", *provenance_tags),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ParityScan:
    """Two-ion joint parity readout protocol (Dispatch N).

    Reads the joint readout distribution ``P(s_0, s_1)`` of two ions at
    every trajectory time point — entangled states (where the joint
    distribution does **not** factorise into marginals) are correctly
    handled because each shot samples ``(s_0, s_1)`` together rather
    than projecting each ion with an independent Bernoulli.

    Inputs required on the trajectory
    ---------------------------------

    - ``sigma_z_{i}`` and ``sigma_z_{j}`` — single-ion marginals
      (§3 atomic-physics convention; provided by
      :func:`iontrap_dynamics.observables.spin_z`).
    - ``parity_{i}_{j}`` — two-body correlation ``⟨σ_z^{(i)} σ_z^{(j)}⟩``
      (provided by :func:`iontrap_dynamics.observables.parity`). The
      four joint probabilities are reconstructed as

          P(↑↑) = (1 + ⟨σ_z^i⟩ + ⟨σ_z^j⟩ + ⟨σ_z^i σ_z^j⟩) / 4
          P(↑↓) = (1 + ⟨σ_z^i⟩ − ⟨σ_z^j⟩ − ⟨σ_z^i σ_z^j⟩) / 4
          P(↓↑) = (1 − ⟨σ_z^i⟩ + ⟨σ_z^j⟩ − ⟨σ_z^i σ_z^j⟩) / 4
          P(↓↓) = (1 − ⟨σ_z^i⟩ − ⟨σ_z^j⟩ + ⟨σ_z^i σ_z^j⟩) / 4

      These are the four Pauli-decomposition tomography components of
      the ZZ-subspace restricted density matrix; any trajectory-state
      consistent with quantum mechanics keeps them non-negative.

    Per-shot sampling
    -----------------

    For each shot at each time:

    1. Draw ``(s_0, s_1) ∈ {↑↑, ↑↓, ↓↑, ↓↓}`` from the reconstructed
       joint distribution (one categorical sample).
    2. For each ion, Poisson-sample at the state-conditional effective
       rate (``η · λ_bright + γ_d`` if ``|↑⟩``,
       ``η · λ_dark + γ_d`` if ``|↓⟩``).
    3. Threshold each ion's count against ``N̂`` → bright bit.
    4. Shot parity = ``(−1)^(bit_0 + bit_1)`` — ``+1`` if the two bits
       agree (both bright or both dark), ``−1`` if they differ.

    Parameters
    ----------
    ion_indices
        Two-tuple of zero-based ion indices. Must be distinct. The
        trajectory must carry ``sigma_z_{i0}``, ``sigma_z_{i1}``, and
        ``parity_{i0}_{i1}`` expectations.
    detector
        Shared :class:`DetectorConfig` used on both ions. Per-ion
        asymmetric detectors are out of scope for Dispatch N; when
        needed, sum in from a ``ParityScan.from_per_ion_detectors``
        factory a later dispatch is free to add.
    lambda_bright
        Emitted photon rate per shot when a single ion is pinned
        bright. Non-negative. Shared across both ions.
    lambda_dark
        Emitted photon rate per shot when a single ion is pinned dark.
        Non-negative, ``<= lambda_bright``.
    label
        Identifier prefix for ``MeasurementResult.sampled_outcome``
        entries. Defaults to ``"parity_scan"``.

    Raises
    ------
    ValueError
        At construction, on any of: non-distinct ion_indices, negative
        rates, ``lambda_dark > lambda_bright``, or negative ion index.
    """

    ion_indices: tuple[int, int]
    detector: DetectorConfig
    lambda_bright: float
    lambda_dark: float
    label: str = "parity_scan"

    def __post_init__(self) -> None:
        if len(self.ion_indices) != 2:
            raise ValueError(f"ParityScan: ion_indices must be a 2-tuple; got {self.ion_indices}")
        i0, i1 = self.ion_indices
        if i0 < 0 or i1 < 0:
            raise ValueError(f"ParityScan: ion indices must be >= 0; got {self.ion_indices}")
        if i0 == i1:
            raise ValueError(f"ParityScan: ion indices must be distinct; got {self.ion_indices}")
        if self.lambda_bright < 0.0 or self.lambda_dark < 0.0:
            raise ValueError(
                "ParityScan: rates must be >= 0; "
                f"got lambda_bright={self.lambda_bright}, "
                f"lambda_dark={self.lambda_dark}"
            )
        if self.lambda_dark > self.lambda_bright:
            raise ValueError(
                "ParityScan: lambda_dark must be <= lambda_bright; "
                f"got {self.lambda_dark} > {self.lambda_bright}"
            )

    def run(
        self,
        trajectory: TrajectoryResult,
        *,
        shots: int,
        seed: int | None = None,
        provenance_tags: tuple[str, ...] = (),
    ) -> MeasurementResult:
        """Execute the protocol against ``trajectory``.

        Returns
        -------
        MeasurementResult
            ``ideal_outcome`` carries:

            - ``"p_up_{i}"`` for each ion — single-ion marginal
              ``(1 + ⟨σ_z^i⟩) / 2``.
            - ``"parity"`` — the correlation trajectory
              ``⟨σ_z^{i0} σ_z^{i1}⟩(t)`` (the infinite-shots limit of
              the estimator under an ideal detector).
            - ``"parity_envelope"`` — what the estimator converges to
              under the projective-shot model with the finite-fidelity
              detector (see §17.11).
            - ``"joint_probabilities"`` — shape ``(4, n_times)`` array
              in basis order ``[↑↑, ↑↓, ↓↑, ↓↓]``.

            ``sampled_outcome`` carries:

            - ``f"{label}_counts_{i}"`` — per-ion ``(shots, n_times)``
              int64 photon counts.
            - ``f"{label}_bits_{i}"`` — per-ion ``(shots, n_times)``
              int8 bright/dark bits.
            - ``f"{label}_parity"`` — ``(shots, n_times)`` int8 per-shot
              parity in ``{−1, +1}``.
            - ``f"{label}_parity_estimate"`` — ``(n_times,)`` float64
              shot-averaged estimator ``⟨parity⟩``.

        Raises
        ------
        ConventionError
            If any of the three required expectations (``sigma_z_{i0}``,
            ``sigma_z_{i1}``, ``parity_{i0}_{i1}``) is missing from the
            trajectory.
        ValueError
            If ``shots < 1``, or if the reconstructed joint
            probabilities leave ``[0, 1]`` by more than ``1e-9`` —
            indicates the upstream solve produced an unphysical state.
        """
        if shots < 1:
            raise ValueError(f"ParityScan.run: shots must be >= 1; got {shots}")

        i0, i1 = self.ion_indices
        required = (
            f"sigma_z_{i0}",
            f"sigma_z_{i1}",
            f"parity_{i0}_{i1}",
        )
        for key in required:
            if key not in trajectory.expectations:
                raise ConventionError(
                    f"ParityScan.run: trajectory has no '{key}' expectation "
                    f"(available: {sorted(trajectory.expectations)})"
                )

        sz0 = np.asarray(trajectory.expectations[required[0]], dtype=np.float64)
        sz1 = np.asarray(trajectory.expectations[required[1]], dtype=np.float64)
        szz = np.asarray(trajectory.expectations[required[2]], dtype=np.float64)

        # Reconstruct joint probabilities. Order: [↑↑, ↑↓, ↓↑, ↓↓].
        joint = np.stack(
            [
                0.25 * (1.0 + sz0 + sz1 + szz),
                0.25 * (1.0 + sz0 - sz1 - szz),
                0.25 * (1.0 - sz0 + sz1 - szz),
                0.25 * (1.0 - sz0 - sz1 + szz),
            ],
            axis=0,
        )
        if np.any((joint < -1e-9) | (joint > 1.0 + 1e-9)):
            raise ValueError(
                "ParityScan.run: reconstructed joint probabilities leave [0, 1] "
                f"by more than 1e-9 (min={joint.min()}, max={joint.max()}) — "
                "upstream trajectory is unphysical."
            )
        joint = np.clip(joint, 0.0, 1.0)

        counts, bits, parity_shots, parity_est = _parity_project_and_sample(
            joint=joint,
            detector=self.detector,
            lambda_bright=self.lambda_bright,
            lambda_dark=self.lambda_dark,
            shots=shots,
            seed=seed,
        )

        # Projective-shot envelope for the parity estimator (§17.11).
        # Each shot's parity = (+1) with prob p_agree, (-1) with prob
        # p_disagree, where agree/disagree are conditioned on both the
        # underlying state AND the detector's classification. In the
        # identical-ion detector limit used here (shared η, γ_d, N̂),
        # p_agree(t) = (TP·TP + (1-TN)·(1-TN))·p_↑↑
        #           + (TP·(1-TN) + (1-TN)·TP)·p_↑↓
        #           + (1-TN)·TP + TP·(1-TN))·p_↓↑
        #           + ((1-TP)·(1-TP) + TN·TN)·p_↓↓   — wait this needs care
        # Simpler: compute the envelope as a direct 4×4 confusion-weighted
        # sum. See helper below.
        fid = self.detector.classification_fidelity(
            lambda_bright=self.lambda_bright, lambda_dark=self.lambda_dark
        )
        parity_envelope = _parity_envelope(joint=joint, fidelities=fid)

        p_up_0 = 0.5 * (1.0 + sz0)
        p_up_1 = 0.5 * (1.0 + sz1)

        metadata = _inherit_metadata(
            upstream=trajectory,
            provenance_tags=(self.label, *provenance_tags),
        )
        return MeasurementResult(
            metadata=metadata,
            shots=shots,
            rng_seed=seed,
            ideal_outcome={
                f"p_up_{i0}": p_up_0,
                f"p_up_{i1}": p_up_1,
                "parity": szz,
                "parity_envelope": parity_envelope,
                "joint_probabilities": joint,
            },
            sampled_outcome={
                f"{self.label}_counts_{i0}": counts[0],
                f"{self.label}_counts_{i1}": counts[1],
                f"{self.label}_bits_{i0}": bits[0],
                f"{self.label}_bits_{i1}": bits[1],
                f"{self.label}_parity": parity_shots,
                f"{self.label}_parity_estimate": parity_est,
            },
            trajectory_hash=trajectory.metadata.request_hash,
        )


def _parity_project_and_sample(
    *,
    joint: NDArray[np.float64],
    detector: DetectorConfig,
    lambda_bright: float,
    lambda_dark: float,
    shots: int,
    seed: int | None,
) -> tuple[
    tuple[NDArray[np.int64], NDArray[np.int64]],
    tuple[NDArray[np.int8], NDArray[np.int8]],
    NDArray[np.int8],
    NDArray[np.float64],
]:
    """Joint projective-shot sampling for a two-ion parity readout.

    ``joint`` has shape ``(4, n_times)`` in basis order
    ``[↑↑, ↑↓, ↓↑, ↓↓]`` and column-sums close to 1.

    Returns ``(counts_0, counts_1), (bits_0, bits_1), parity_shots,
    parity_estimate``. Each per-ion count / bit array has shape
    ``(shots, n_times)``; parity is ``(shots, n_times)`` int8 in
    ``{−1, +1}``; parity_estimate is ``(n_times,)`` float64.
    """
    rng = np.random.default_rng(seed)
    n_times = joint.shape[1]

    # Cumulative joint distribution along the 4-outcome axis,
    # broadcast-ready for inverse-CDF sampling.
    cumulative = np.cumsum(joint, axis=0)  # (4, n_times)

    # One uniform draw per (shot, time); categorical index in {0,1,2,3}.
    uniforms = rng.random(size=(shots, n_times))
    # cumulative[k, t] ≥ u iff shot falls in class k at time t; argmax
    # along class axis picks the smallest such k.
    # Shape bookkeeping: compare (4, 1, n_times) ≥ (1, shots, n_times).
    cat = np.argmax(cumulative[:, None, :] >= uniforms[None, :, :], axis=0)
    # cat shape: (shots, n_times); entries in {0,1,2,3} for [↑↑, ↑↓, ↓↑, ↓↓].

    # Per-ion bright state (bit 1 if ion is in |↑⟩).
    state_0_bright = (cat == 0) | (cat == 1)  # ion 0 bright when ↑↑ or ↑↓
    state_1_bright = (cat == 0) | (cat == 2)  # ion 1 bright when ↑↑ or ↓↑

    rate_bright_eff = detector.efficiency * lambda_bright + detector.dark_count_rate
    rate_dark_eff = detector.efficiency * lambda_dark + detector.dark_count_rate

    rate_0 = np.where(state_0_bright, rate_bright_eff, rate_dark_eff)
    rate_1 = np.where(state_1_bright, rate_bright_eff, rate_dark_eff)

    counts_0 = rng.poisson(rate_0).astype(np.int64)
    counts_1 = rng.poisson(rate_1).astype(np.int64)
    bits_0 = detector.discriminate(counts_0)
    bits_1 = detector.discriminate(counts_1)

    # Parity = (+1) if bits agree, (−1) if they differ.
    parity_shots = np.where(bits_0 == bits_1, 1, -1).astype(np.int8)
    parity_estimate = parity_shots.mean(axis=0).astype(np.float64)

    return (
        (counts_0, counts_1),
        (bits_0, bits_1),
        parity_shots,
        parity_estimate,
    )


def _parity_envelope(
    *,
    joint: NDArray[np.float64],
    fidelities: dict[str, float],
) -> NDArray[np.float64]:
    """Infinite-shots envelope of the parity estimator (§17.11).

    With shared per-ion detector, the bit-level confusion matrix has
    ``P(bit=1 | state=↑) = TP`` and ``P(bit=1 | state=↓) = 1 − TN``.
    Because the two ions' readouts are independent conditional on the
    joint state, the probability the two bits agree from joint state
    ``(s_0, s_1)`` is

        P(bits agree | s_0, s_1) = P(b_0=1|s_0)·P(b_1=1|s_1)
                                 + (1−P(b_0=1|s_0))·(1−P(b_1=1|s_1))

    and the envelope is ``sum_s P(s)·(2·P(agree|s) − 1)``. This reduces
    to a linear combination of ``⟨σ_z^{0}⟩``, ⟨σ_z^{1}⟩, and the true
    parity ``⟨σ_z^{0} σ_z^{1}⟩`` weighted by fidelity contrast
    ``(TP + TN − 1)``.
    """
    tp = fidelities["true_positive_rate"]
    tn = fidelities["true_negative_rate"]
    p_up_given_up = tp
    p_up_given_dn = 1.0 - tn
    p_dn_given_up = 1.0 - tp
    p_dn_given_dn = tn

    # joint order: [↑↑, ↑↓, ↓↑, ↓↓]
    p_agree_uu = p_up_given_up * p_up_given_up + p_dn_given_up * p_dn_given_up
    p_agree_ud = p_up_given_up * p_up_given_dn + p_dn_given_up * p_dn_given_dn
    p_agree_du = p_up_given_dn * p_up_given_up + p_dn_given_dn * p_dn_given_up
    p_agree_dd = p_up_given_dn * p_up_given_dn + p_dn_given_dn * p_dn_given_dn

    p_agree = (
        joint[0] * p_agree_uu
        + joint[1] * p_agree_ud
        + joint[2] * p_agree_du
        + joint[3] * p_agree_dd
    )
    return np.asarray(2.0 * p_agree - 1.0, dtype=np.float64)


__all__ = [
    "ParityScan",
    "SpinReadout",
]
