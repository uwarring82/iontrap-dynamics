# SPDX-License-Identifier: MIT
"""Protocol-layer composers — named measurement procedures.

A *protocol* wraps the channel-and-detector plumbing into a single
procedure the experimentalist recognises by name (``spin_readout``,
``parity_scan``, ``sideband_flopping``). Each protocol spec is a frozen
dataclass; its ``.run(trajectory, *, shots, seed)`` method consumes a
:class:`TrajectoryResult`, executes the measurement, and returns a
:class:`MeasurementResult` with the dual-view ideal / sampled payload.

Dispatch M adds :class:`SpinReadout` — the prototype protocol. Parity
scan (Dispatch N) and sideband-flopping inference (Dispatch O) follow
the same shape: construct a spec, call ``.run()``, consume the
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


__all__ = [
    "SpinReadout",
]
