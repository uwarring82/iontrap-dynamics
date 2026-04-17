# SPDX-License-Identifier: MIT
"""Phase 0.F performance smoke-test scaffold.

Three canonical benchmarks, frozen against the laptop hardware listed in
``WORKPLAN_v0.3.md`` §0.F (2023 MacBook Air M2 or equivalent):

- Single-ion sideband flopping   (``N_Fock = 30``, 200 steps)   → < 5 s
- Two-ion Mølmer–Sørensen gate   (``N_Fock = 15``, 500 steps)   → < 30 s
- Stroboscopic AC-π/2 drive      (``N_Fock = 40``, 1000 steps)  → < 60 s

Each benchmark is currently skipped pending the Phase 1 builder it
exercises. Replacing ``@pytest.mark.skip(...)`` with a real build-and-solve
body is the unit of work that activates each gate — the three test slots
live here so Phase 1 PRs do not need to invent a new file, and so
``pytest -m benchmark`` already lists the complete target set today.

Breach policy (workplan §0.F)
----------------------------

If a threshold is exceeded after a builder lands, review immediately.
Options include algorithmic optimisation, sparse-path refinement, targeted
caching, or elevating JAX-backend work from Phase 2 to Phase 1. The
integrator selects proportionate to root cause.

Running
-------

``pytest tests/benchmarks -v`` — shows all three as SKIPPED with reason.
``pytest -m benchmark``       — same, filtered by marker.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


# Thresholds are the workplan §0.F contract. Reproduced here as symbolic
# constants so a builder-landing PR can replace the skip without rereading
# the workplan to find the number.
CANONICAL_HARDWARE = "2023 MacBook Air M2 or equivalent"

T_SINGLE_ION_SIDEBAND = 5.0   # seconds
T_TWO_ION_MS_GATE = 30.0      # seconds
T_STROBOSCOPIC_AC = 60.0      # seconds


@pytest.mark.skip(
    reason=(
        "awaiting Phase 1 red-sideband builder (workplan §0.F item 1); "
        "replace skip with build/solve and assert elapsed < T_SINGLE_ION_SIDEBAND."
    )
)
def test_single_ion_sideband_flopping_under_5s() -> None:
    """Single-ion red-sideband flopping: ``N_Fock = 30``, 200 steps.

    Threshold: ``T_SINGLE_ION_SIDEBAND`` = 5.0 s wall time on the canonical
    hardware. Typical shape of the activation body (Phase 1):

    .. code-block:: python

        import time
        system = IonSystem(species=Mg25, drives=[red_sideband], modes=[axial])
        hilbert = HilbertSpace(system, n_fock={"axial": 30})
        state = thermal_state(system, n_bar=0.05)
        sequence = red_sideband_pulse(system, duration=...)
        t0 = time.perf_counter()
        sequence.run(initial_state=state, steps=200)
        elapsed = time.perf_counter() - t0
        assert elapsed < T_SINGLE_ION_SIDEBAND, (
            f"took {elapsed:.2f}s (threshold {T_SINGLE_ION_SIDEBAND}s)"
        )
    """
    raise AssertionError("unreachable — test is skipped")


@pytest.mark.skip(
    reason=(
        "awaiting Phase 1 Mølmer–Sørensen gate builder (workplan §0.F item 2); "
        "replace skip with build/solve and assert elapsed < T_TWO_ION_MS_GATE."
    )
)
def test_two_ion_ms_gate_under_30s() -> None:
    """Two-ion Mølmer–Sørensen gate: ``N_Fock = 15``, 500 steps.

    Threshold: ``T_TWO_ION_MS_GATE`` = 30.0 s wall time on the canonical
    hardware. The smaller Fock truncation reflects the larger Hilbert
    dimension (two spins + one mode) relative to the single-ion case.
    """
    raise AssertionError("unreachable — test is skipped")


@pytest.mark.skip(
    reason=(
        "awaiting Phase 1 stroboscopic AC-π/2 builder (workplan §0.F item 3); "
        "replace skip with build/solve and assert elapsed < T_STROBOSCOPIC_AC."
    )
)
def test_stroboscopic_ac_halfpi_under_60s() -> None:
    """Stroboscopic AC-π/2 drive: ``N_Fock = 40``, 1000 steps.

    Threshold: ``T_STROBOSCOPIC_AC`` = 60.0 s wall time on the canonical
    hardware. The 1000-step count reflects the fast time-dependence of
    stroboscopic drives — the threshold is deliberately the loosest of the
    three.
    """
    raise AssertionError("unreachable — test is skipped")


# ----------------------------------------------------------------------------
# Meta-sanity: the benchmarks scaffold is registered, reachable, and parametric.
# These tests are NOT skipped — they verify the scaffold's integrity so that a
# broken benchmark module cannot hide behind three silent skips.
# ----------------------------------------------------------------------------


class TestScaffoldSanity:
    def test_thresholds_are_positive(self) -> None:
        assert T_SINGLE_ION_SIDEBAND > 0
        assert T_TWO_ION_MS_GATE > 0
        assert T_STROBOSCOPIC_AC > 0

    def test_thresholds_ordering_matches_workplan(self) -> None:
        """Single-ion is fastest, stroboscopic is slowest — matches the
        problem complexity ordering in workplan §0.F."""
        assert T_SINGLE_ION_SIDEBAND < T_TWO_ION_MS_GATE < T_STROBOSCOPIC_AC

    def test_canonical_hardware_declared(self) -> None:
        assert "M2" in CANONICAL_HARDWARE, (
            "workplan §0.F names the canonical laptop; keep that reference "
            "visible so benchmark owners know what the threshold is calibrated against."
        )
