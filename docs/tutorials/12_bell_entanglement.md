# Tutorial 12 — Two-ion Bell-state entanglement

**Goal.** Close the tutorials track by taking the
[Tutorial 4](04_ms_gate_bell.md) MS-gate scenario and exercising
the **two measurement surfaces** that the library specifically
provides for two-ion entangling gates:

1. **Parity-scan readout** via the `ParityScan` protocol — the
   finite-shot parity estimator that experimentalists measure
   directly, including the detector-classification envelope.
2. **Nonlinear entanglement observables** via the registered
   trajectory evaluators — `concurrence_trajectory`,
   `entanglement_of_formation_trajectory`, and
   `log_negativity_trajectory`. These quantify the
   entanglement content at every time step, which the
   linear-expectation observable surface cannot.

By the end you will have verified the gate-closing Bell state
against **three independent witnesses**: parity swinging from
+1 → −1 → +1 (with explicit detector envelope), spin-spin
concurrence reaching 1.0, and spin-vs-motion log-negativity
returning cleanly to 0 as the motional loop closes.

**Expected time.** ~15 min reading; ~2 s runtime.

**Prerequisites.** [Tutorial 4](04_ms_gate_bell.md) — the MS-gate
scenario used verbatim. [Tutorial 10](10_finite_shot_statistics.md)
for the finite-shot parity-estimator error-bar framing.
[Tutorial 5](05_custom_observables.md) for the `StorageMode.EAGER`
pattern that the entanglement evaluators require.

---

## The two measurement surfaces

Most of what we've covered so far composes at the
**linear-expectation** level: every `Observable` is an operator
`O`, and the solver returns `⟨ψ(t)| O |ψ(t)⟩` at every time
step. That's enough for single-ion physics and for the
population-projector tomography of Tutorials
[4](04_ms_gate_bell.md) and [5](05_custom_observables.md). But
two entangling-gate analyses that the library handles natively
**are not** linear expectations:

| Measurement need                     | Interface                                          | Storage mode required |
|--------------------------------------|----------------------------------------------------|-----------------------|
| Finite-shot parity estimator         | `ParityScan` protocol                              | `OMITTED` (expectations only) |
| Two-ion concurrence / EoF            | `concurrence_trajectory(states, ...)`              | `EAGER` (full states needed) |
| Spin-vs-motion log-negativity        | `log_negativity_trajectory(states, partition=...)` | `EAGER`                      |

Parity-scan readout works from expectations alone (it
reconstructs the joint `(s_0, s_1)` distribution from the
three two-ion σ_z expectations). The entanglement evaluators
need the full density matrix at every time step — that's what
`StorageMode.EAGER` gives up from `sequences.solve`.

## Step 1 — Build the MS-gate trajectory twice

Same Tutorial 4 scenario, but this time we need **two solves**:
one in `OMITTED` storage mode for parity-scan readout (the
expectations feed the reconstruction), one in `EAGER` for the
entanglement evaluators (the state trajectory feeds the partial
traces):

```python
import numpy as np
import qutip

from iontrap_dynamics.analytic import (
    lamb_dicke_parameter,
    ms_gate_closing_detuning,
    ms_gate_closing_time,
)
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import detuned_ms_gate_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import parity, spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

N_FOCK = 12

mode = ModeConfig(
    label="com",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    ) / np.sqrt(2.0),
)
system = IonSystem(species_per_ion=(mg25_plus(), mg25_plus()), modes=(mode,))
hilbert = HilbertSpace(system=system, fock_truncations={"com": N_FOCK})

drive = DriveConfig(
    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
    carrier_rabi_frequency_rad_s=2 * np.pi * 0.1e6,
    phase_rad=0.0,
)
eta = lamb_dicke_parameter(
    k_vec=drive.k_vector_m_inv,
    mode_eigenvector=mode.eigenvector_at_ion(0),
    ion_mass=mg25_plus().mass_kg,
    mode_frequency=mode.frequency_rad_s,
)
delta = ms_gate_closing_detuning(
    carrier_rabi_frequency=drive.carrier_rabi_frequency_rad_s,
    lamb_dicke_parameter=eta, loops=1,
)
t_gate = ms_gate_closing_time(
    carrier_rabi_frequency=drive.carrier_rabi_frequency_rad_s,
    lamb_dicke_parameter=eta, loops=1,
)
hamiltonian = detuned_ms_gate_hamiltonian(
    hilbert, drive, "com",
    ion_indices=(0, 1), detuning_rad_s=delta,
)
psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(N_FOCK, 0))
times = np.linspace(0.0, t_gate, 200)

# Run 1: expectations only — for ParityScan.
result_expectations = solve(
    hilbert=hilbert, hamiltonian=hamiltonian, initial_state=psi_0,
    times=times,
    observables=[
        spin_z(hilbert, 0),
        spin_z(hilbert, 1),
        parity(hilbert, ion_indices=(0, 1)),
    ],
    storage_mode=StorageMode.OMITTED,
)

# Run 2: full state trajectory — for the entanglement evaluators.
result_states = solve(
    hilbert=hilbert, hamiltonian=hamiltonian, initial_state=psi_0,
    times=times, observables=[spin_z(hilbert, 0)],
    storage_mode=StorageMode.EAGER,
)
```

!!! tip "Why not one solve with `storage_mode=EAGER` and reuse for everything?"

    You could. `ParityScan` only reads expectations, so it
    doesn't care whether states are attached. The reason the
    examples separate them is to make the cost explicit:
    `EAGER` storage holds all 200 density matrices of dimension
    `2 × 2 × 12 = 48` (so `48 × 48 = 2304` entries each) in
    memory. For a long jitter ensemble that's the difference
    between "runs comfortably" and "swaps to disk".

## Step 2 — Parity-scan readout (three `Observable` inputs)

`ParityScan` looks up three expectations on the trajectory:
`sigma_z_0`, `sigma_z_1`, and `parity_0_1`. These are
provided by the three factories already loaded into
`result_expectations`. At every time step the protocol
reconstructs the joint `P(s_0, s_1)` distribution over
`{↑↑, ↑↓, ↓↑, ↓↓}`, draws `shots` categorical samples,
Poisson-samples photon counts per ion conditioned on the drawn
state, thresholds to bright/dark bits, and computes per-shot
parity `(+1)^(bit_0 + bit_1)`:

```python
from iontrap_dynamics import DetectorConfig, ParityScan

detector = DetectorConfig(
    efficiency=0.85, dark_count_rate=0.3, threshold=3,
)
parity_scan = ParityScan(
    ion_indices=(0, 1),
    detector=detector,
    lambda_bright=20.0, lambda_dark=0.0,
)
measurement = parity_scan.run(
    result_expectations, shots=500, seed=20260421,
)

parity_estimate = measurement.sampled_outcome["parity_scan_parity_estimate"]
parity_envelope = measurement.ideal_outcome["parity_envelope"]
```

Two series are interesting side-by-side:

- **`parity_envelope`** — what the estimator converges to at
  infinite shots under this exact detector. At an ideal
  detector (`efficiency=1`, `dark_count_rate=0`) this equals
  the ideal `⟨σ_z^(0) σ_z^(1)⟩`; with finite fidelity it's
  attenuated by the detector classification errors.
- **`parity_estimate`** — the 500-shot Wilson estimator for
  the parity at each time step. Tracks the envelope within
  statistical error (`σ ≈ 0.045` for `shots = 500`).

### Numbers at the gate-closing time

For the canonical Tutorial 4 parameters (`η = 0.18`,
`δ / 2π = 36.85 kHz`, `t_gate = 27.14 μs`):

```
ideal parity ⟨σ_z σ_z⟩(t_gate)   = +1.0000   (two-body correlator)
parity envelope (detector-limited) = +0.9928  (85 % efficiency + 0.3 dark)
parity estimate (500 shots)        = +0.992   (Wilson CI ±0.007)
```

The 0.72 % gap between ideal and envelope is the detector
classification-fidelity loss — a reader scaling this up to
a publication-grade scenario will tighten `efficiency` or
`threshold` to close that gap.

## Step 3 — Concurrence trajectory (spin-spin entanglement)

`concurrence_trajectory` takes the full state sequence, partial-
traces over everything except the two named spin subsystems,
and computes Wootters' concurrence on the resulting 4×4
reduced density matrix:

```python
from iontrap_dynamics import (
    concurrence_trajectory,
    entanglement_of_formation_trajectory,
    log_negativity_trajectory,
)

c_trajectory = concurrence_trajectory(
    result_states.states, hilbert=hilbert, ion_indices=(0, 1),
)
eof_trajectory = entanglement_of_formation_trajectory(
    result_states.states, hilbert=hilbert, ion_indices=(0, 1),
)

assert c_trajectory[0] == 0.0               # starts separable
assert abs(c_trajectory[-1] - 1.0) < 1e-4   # Bell state at t_gate
```

`entanglement_of_formation_trajectory` follows the closed-form
Wootters relation from concurrence for two-qubit reduced
states, so its trajectory is a monotonic function of
`c_trajectory` with the same zeros and ones but a different
shape in between.

## Step 4 — Log-negativity (spin-vs-motion entanglement)

`log_negativity_trajectory` uses a different bipartition —
**spin subsystem vs. mode subsystem** — and reports
`E_N = log₂ ‖ρ^{T_A}‖₁`. This is the quantity that tells you
how much the spins are **entangled with the motion** mid-gate
(as opposed to the concurrence's between-spin entanglement
after the motion has been traced out):

```python
ln_trajectory = log_negativity_trajectory(
    result_states.states, hilbert=hilbert, partition="spins",
)
```

!!! note "Log-negativity's `partition` argument, not `ion_indices`"

    `concurrence_trajectory` and
    `entanglement_of_formation_trajectory` take
    `ion_indices=(i, j)` — they're two-qubit-specific and
    measure between-spin entanglement. `log_negativity_trajectory`
    is bipartite-generic and takes `partition="spins"` or
    `"modes"` — the bipartition is always
    "all spins ↔ all modes". The two interfaces are
    complementary, not interchangeable.

### Three witnesses, one gate

The three measures tell three different stories at three points
in the gate:

| t           | Concurrence | EoF    | Log-negativity (spin\|motion) | What's happening                                   |
|-------------|-------------|--------|-------------------------------|----------------------------------------------------|
| 0.00 µs     | 0.000       | 0.000  | 0.000                         | Product state `|↓↓, 0⟩` — no entanglement anywhere |
| 13.64 µs    | 0.267       | 0.131  | 1.310                         | Mid-gate: spins heavily entangled **with motion**  |
| 27.14 µs    | 1.000       | 1.000  | 0.000                         | Bell state `|Φ⁻⟩ ⊗ |0⟩` — motion has disentangled  |

Reading the table:

1. **`t = 0`**: all three zero — the product state is separable
   in every bipartition.
2. **Mid-gate**: log-negativity between spins and motion peaks
   at `1.310` — the spins are sharing information with
   phonons. Spin-spin concurrence is only `0.267`; the
   **reduced** state after tracing out motion is mixed because
   it's entangled with the phonons, and mixed states have less
   concurrence than pure Bell states.
3. **`t = t_gate`**: spin-spin concurrence hits `1.000` — a
   maximally-entangled Bell state — while spin-motion
   log-negativity returns to `0`. The phase-space loop closed
   and the motion left; all entanglement moved into the spin
   pair. This is exactly what the MS gate is *for*.

No single measure captures all three phases of the gate. The
concurrence alone would miss the mid-gate spin-motion
entanglement; the log-negativity alone would miss the
qualitative difference between "no entanglement" and "maximal
Bell" at the endpoints. Together they paint a complete picture.

## The runnable reference

The full scenario is packaged as
[`tools/run_demo_bell_entanglement.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/run_demo_bell_entanglement.py)
with a committed artefact bundle under
[`benchmarks/data/bell_entanglement_demo/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data/bell_entanglement_demo).
The companion parity-scan focused demo is
[`tools/run_demo_parity_scan.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/run_demo_parity_scan.py);
the MS-gate dynamics demo used in Tutorial 4 is
`tools/run_demo_ms_gate.py`.

## What's next — beyond the tutorials

This is the last of the twelve planned tutorials. The remaining
library surface that the track doesn't cover directly (but that
composes cleanly with everything you've now seen):

- **SPAM primitives.** Tutorial 1 + `SpinPreparationError` /
  `ThermalPreparationError` for imperfect preparation /
  readout fidelities. See `src/iontrap_dynamics/systematics/spam.py`.
- **Drift primitives.** `RabiDrift` / `DetuningDrift` /
  `PhaseDrift` for slow (shot-to-shot non-stationary) systematic
  drift; Tutorial 11's pipeline applies with the drift
  primitive in place of a jitter primitive. See
  `src/iontrap_dynamics/systematics/drift.py`.
- **Sideband inference.** The `SidebandInference` protocol
  (measurement layer) extracts a motional-thermometry estimate
  `n̄` from red / blue sideband flop amplitudes — introduced
  briefly at the end of Tutorial 2. See the Dispatch O
  material in CHANGELOG.
- **Custom factory contributions.** Upstream a factory to
  `observables.py` if your team hits the same observable
  repeatedly — the pattern is documented in Tutorial 5.

## Where to next

- [Tutorial 4](04_ms_gate_bell.md) — the MS-gate dynamics
  this tutorial reads out.
- [Tutorial 5](05_custom_observables.md) — the `StorageMode.EAGER`
  pattern required by the entanglement evaluators.
- [Tutorial 10](10_finite_shot_statistics.md) — the
  finite-shot Wilson estimator framing for the parity
  estimate.
- [Phase 1 Architecture](../phase-1-architecture.md) — full
  reference for `ParityScan`, the entanglement evaluators, and
  the `MeasurementResult` schema.
- [`src/iontrap_dynamics/entanglement.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/entanglement.py)
  — reference implementation of all three trajectory
  evaluators.

---

## Licence

Sail material — adaptive guidance with specific parameter choices,
not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
