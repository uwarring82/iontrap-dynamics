# SPDX-License-Identifier: MIT
"""Tensor-product Hilbert space built from an :class:`IonSystem`.

Implements CONVENTIONS.md §2 tensor ordering: spins first (ascending ion
index), then modes (order as they appear in ``system.modes``,
left-to-right). Every Phase 1 Hamiltonian builder and state-prep routine
builds its operators against a :class:`HilbertSpace` so the tensor
layout is guaranteed consistent across the library.

Responsibilities
----------------

1. **Validate** that the caller-supplied per-mode Fock truncations cover
   every mode in the system and that each cutoff is at least 1.
2. **Expose** total dimension, per-subsystem dimensions, and QuTiP's
   ``dims`` list in the conventional ordering.
3. **Embed** single-spin and single-mode operators into the full
   tensor-product space by filling with identities on the other
   subsystems.
4. **Construct** common motional primitives (annihilation, creation,
   number) for a named mode, pre-embedded into the full space.

Design
------

Frozen dataclass — no lazy operator cache in v0.1. Per-call embedding
uses QuTiP's ``tensor`` with identity operators on the other
subsystems; cost is a few milliseconds for the ~100-dimensional spaces
the Phase 0.F benchmarks target. If profiling under larger systems
shows this to be a hot path, a lazy ``(op_id, position)`` cache can be
added without changing the public API.

Extension points intentionally deferred to Phase 1+:

- **Operator cache** (workplan §3): add a lazy dict on the instance
  once there's profiling data demonstrating the need.
- **Tensor-product builders for multi-subsystem operators** (e.g.
  ``σ_z_i ⊗ σ_z_j``): the current API composes these by multiplying
  two single-subsystem embeddings; a direct helper is useful later.
- **Partial-trace conveniences**: users call QuTiP's ``ptrace`` against
  indices computed from :meth:`subsystem_indices`; a named helper can
  come later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import qutip

from .exceptions import ConventionError
from .system import IonSystem


@dataclass(frozen=True, kw_only=True)
class HilbertSpace:
    """Tensor-product Hilbert space for a given :class:`IonSystem`.

    Parameters
    ----------
    system
        The :class:`IonSystem` specifying the crystal and its modes.
    fock_truncations
        Mapping from mode label to Fock truncation ``N`` (dimension of
        that mode's Hilbert factor). Must have an entry for every mode
        in ``system.modes`` and no extras. Each cutoff must be ≥ 1.
    """

    system: IonSystem
    fock_truncations: Mapping[str, int]

    def __post_init__(self) -> None:
        system_mode_labels = {m.label for m in self.system.modes}
        trunc_labels = set(self.fock_truncations.keys())

        missing = system_mode_labels - trunc_labels
        if missing:
            raise ConventionError(
                f"fock_truncations is missing entries for modes: {sorted(missing)}. "
                "Every mode in the IonSystem must have a declared Fock truncation."
            )
        extra = trunc_labels - system_mode_labels
        if extra:
            raise ConventionError(
                f"fock_truncations has keys for unknown modes: {sorted(extra)}. "
                "The IonSystem carries no mode with these labels."
            )
        for label, n in self.fock_truncations.items():
            if n < 1:
                raise ConventionError(
                    f"fock_truncations[{label!r}] = {n} is invalid; must be >= 1."
                )

    # ------------------------------------------------------------------------
    # Structural properties
    # ------------------------------------------------------------------------

    @property
    def n_ions(self) -> int:
        """Number of two-level spin subsystems (= number of ions)."""
        return self.system.n_ions

    @property
    def n_modes(self) -> int:
        """Number of motional modes."""
        return self.system.n_modes

    @property
    def spin_dim(self) -> int:
        """Dimension of a single spin subsystem (always 2 in v0.1)."""
        return 2

    def mode_dim(self, label: str) -> int:
        """Return the Fock truncation for the named mode."""
        if label not in self.fock_truncations:
            available = sorted(self.fock_truncations.keys())
            raise ConventionError(f"unknown mode label: {label!r}. Available: {available!r}")
        return self.fock_truncations[label]

    @property
    def total_dim(self) -> int:
        """Total Hilbert-space dimension: 2^n_ions × ∏ mode_cutoffs."""
        # Explicit int annotation pins the type through the *= chain.
        # Without it, mypy strict narrows `dim` to Any after the loop
        # (the Mapping value-type resolution interacts unexpectedly with
        # augmented assignment; same category of failure as math.prod).
        dim: int = self.spin_dim**self.n_ions
        for m in self.system.modes:
            dim *= int(self.fock_truncations[m.label])
        return dim

    @property
    def subsystem_dims(self) -> list[int]:
        """Dimensions of each subsystem in CONVENTIONS.md §2 order.

        Returns a list: ``[2, 2, ..., 2, N_0, N_1, ...]`` where the ``2``s
        are the ion spin subsystems (``n_ions`` of them) and the ``N_i``
        are the Fock truncations in the order the modes appear in
        ``system.modes``.
        """
        return [self.spin_dim] * self.n_ions + [
            self.fock_truncations[m.label] for m in self.system.modes
        ]

    def qutip_dims(self) -> list[list[int]]:
        """Return the QuTiP ``dims`` list for a ket or operator on this space.

        Format: ``[subsystem_dims, subsystem_dims]`` for an operator;
        ``[subsystem_dims, [1] * total_n_subsystems]`` for a ket. This
        method returns the operator form; ket form is
        ``[subsystem_dims(), [1] * len(subsystem_dims())]``.
        """
        dims = self.subsystem_dims
        return [dims, dims]

    # ------------------------------------------------------------------------
    # Operator embedding
    # ------------------------------------------------------------------------

    def spin_op_for_ion(self, op: qutip.Qobj, ion_index: int) -> qutip.Qobj:
        """Embed a single-spin ``2×2`` operator into the full space at ``ion_index``.

        Other spins and all modes receive identity operators. Ordering
        follows CONVENTIONS.md §2.

        Raises
        ------
        IndexError
            If ``ion_index`` is outside ``[0, n_ions)``.
        ConventionError
            If ``op.dims`` is not ``[[2], [2]]``.
        """
        if not 0 <= ion_index < self.n_ions:
            raise IndexError(
                f"ion_index {ion_index} out of range for a crystal with {self.n_ions} ions"
            )
        if op.dims != [[self.spin_dim], [self.spin_dim]]:
            raise ConventionError(
                f"spin operator has dims {op.dims}, expected "
                f"[[{self.spin_dim}], [{self.spin_dim}]] (single-spin 2×2)."
            )

        subsystems: list[qutip.Qobj] = []
        for i in range(self.n_ions):
            subsystems.append(op if i == ion_index else qutip.qeye(self.spin_dim))
        for m in self.system.modes:
            subsystems.append(qutip.qeye(self.fock_truncations[m.label]))
        return qutip.tensor(*subsystems)

    def mode_op_for(self, op: qutip.Qobj, mode_label: str) -> qutip.Qobj:
        """Embed a single-mode ``N×N`` operator into the full space at the named mode.

        Raises
        ------
        ConventionError
            If ``mode_label`` is not a mode of the system, or if
            ``op.dims`` does not match the mode's Fock truncation.
        """
        target_index = None
        for i, m in enumerate(self.system.modes):
            if m.label == mode_label:
                target_index = i
                break
        if target_index is None:
            available = [m.label for m in self.system.modes]
            raise ConventionError(f"unknown mode label: {mode_label!r}. Available: {available!r}")

        expected_dim = self.fock_truncations[mode_label]
        if op.dims != [[expected_dim], [expected_dim]]:
            raise ConventionError(
                f"mode operator for {mode_label!r} has dims {op.dims}, expected "
                f"[[{expected_dim}], [{expected_dim}]] (cutoff matches fock_truncations)."
            )

        subsystems: list[qutip.Qobj] = [qutip.qeye(self.spin_dim) for _ in range(self.n_ions)]
        for i, m in enumerate(self.system.modes):
            if i == target_index:
                subsystems.append(op)
            else:
                subsystems.append(qutip.qeye(self.fock_truncations[m.label]))
        return qutip.tensor(*subsystems)

    # ------------------------------------------------------------------------
    # Common motional primitives, pre-embedded
    # ------------------------------------------------------------------------

    def annihilation_for_mode(self, label: str) -> qutip.Qobj:
        """Return the annihilation operator ``a_label`` embedded into the full space."""
        return self.mode_op_for(qutip.destroy(self.mode_dim(label)), label)

    def creation_for_mode(self, label: str) -> qutip.Qobj:
        """Return the creation operator ``a†_label`` embedded into the full space."""
        return self.mode_op_for(qutip.create(self.mode_dim(label)), label)

    def number_for_mode(self, label: str) -> qutip.Qobj:
        """Return the number operator ``n̂_label = a†a`` embedded into the full space."""
        return self.mode_op_for(qutip.num(self.mode_dim(label)), label)

    # ------------------------------------------------------------------------
    # Identities
    # ------------------------------------------------------------------------

    def identity(self) -> qutip.Qobj:
        """Return the identity operator on the full tensor-product space."""
        subsystems: list[qutip.Qobj] = [qutip.qeye(self.spin_dim) for _ in range(self.n_ions)]
        for m in self.system.modes:
            subsystems.append(qutip.qeye(self.fock_truncations[m.label]))
        return qutip.tensor(*subsystems)


__all__ = [
    "HilbertSpace",
]
