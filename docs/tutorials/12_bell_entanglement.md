# Tutorial 12 — Two-ion Bell-state entanglement

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/12_bell_entanglement.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

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

**Level.** `advanced` — a specialised or research-grade surface; do the core first.

**Prerequisites.** [Tutorial 4](04_ms_gate_bell.md) — the MS-gate
scenario used verbatim. [Tutorial 10](10_finite_shot_statistics.md)
for the finite-shot parity-estimator error-bar framing.
[Tutorial 5](05_custom_observables.md) for the `StorageMode.EAGER`
pattern that the entanglement evaluators require.

---

!!! note "New here? Read this first"

    - This tutorial runs **no new physics** — it re-runs the Tutorial 4
      MS gate and reads its gate-closing Bell state out **three ways**.
    - Populations alone are blind here: at `t_gate` each ion on its own
      reads 50/50 (`⟨σ_z⟩ = 0`). Only **joint** measurements — the
      two-body parity `⟨σ_z σ_z⟩` and the entanglement measures —
      reveal the Bell state.
    - Parity gives the z-basis **correlation** (are the two spins the
      same or different?); the concurrence gives the **coherence** an
      incoherent even-parity mixture would lack. You need both.
    - Entanglement **moves** during the gate: mid-gate the spins entangle
      with the **motion** (log-negativity crests near `1.31`), then the
      motional loop closes at `t_gate` and hands all of it to the spin
      pair (concurrence `0 → 1`).
    - Two solves, two storage modes: parity needs only expectations
      (`StorageMode.OMITTED`); the entanglement measures need **every**
      density matrix (`StorageMode.EAGER`).

    **In a hurry?** Step 2 (parity readout) plus Steps 3–4 (concurrence
    and log-negativity) are the core; the rest is cost-accounting and
    interpretation.

**Symbols in this tutorial**

| Symbol | Meaning |
|--------|---------|
| `ParityScan` | Protocol that reconstructs the joint `P(s₀,s₁)` outcome distribution from the three σ_z expectations (including `⟨σ_z σ_z⟩`), then adds a detector envelope and a finite-shot estimate. |
| `⟨σ_z σ_z⟩` (parity) | Two-body z-correlation: `+1` both spins the same, `−1` different. Necessary but not sufficient for the Bell state. |
| `C` (concurrence) | Wootters spin–spin entanglement of the two-ion reduced state; `0` separable → `1` maximal. |
| `E_F` (entanglement of formation) | A fixed monotone function of `C` — same `0`/`1` endpoints, different curve between. |
| `E_N` (log-negativity) | Spin↔motion entanglement across the spin/mode cut, `log₂‖ρ^{T_A}‖₁`; a mixed-state measure concurrence cannot give. |
| `t_gate` | MS-gate closing time — the motional loop closes and the spins are left in a pure Bell state. |
| `StorageMode` | `OMITTED` keeps expectations only (enough for parity); `EAGER` keeps every density matrix (required by the entanglement evaluators). |

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
import matplotlib.pyplot as plt
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

# House colours
BLUE, RED, GREEN, PURPLE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#444444"

# Motional Fock-space truncation — a sizing choice, not physics.
N_FOCK = 12

# --- Physics: two 25Mg+ ions on one shared COM mode; the MS gate below is
# --- tuned (δ, t_gate) to close its phase-space loop in exactly one loop.
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

print(f"MS-gate parameters: η = {eta:.4f},  δ/2π = {delta / (2 * np.pi) * 1e-3:.2f} kHz,  t_gate = {t_gate * 1e6:.2f} µs")

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

# Ideal ⟨σ_z σ_z⟩ from the expectations trajectory (label set by the parity factory).
parity_ideal = np.asarray(result_expectations.expectations["parity_0_1"], dtype=float)

print(f"Parity at t_gate — ideal ⟨σ_z σ_z⟩: {parity_ideal[-1]:+.4f},  "
      f"envelope (detector-limited): {float(parity_envelope[-1]):+.4f},  "
      f"500-shot estimate: {float(parity_estimate[-1]):+.4f}")

t_us = times * 1e6  # convert to µs for the x-axis
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(t_us, parity_ideal, color=GREY, linewidth=1.0, label=r"ideal $\langle\sigma_z^{(0)}\sigma_z^{(1)}\rangle$")
ax.plot(t_us, parity_envelope, color=BLUE, linewidth=1.5, label="envelope (detector-limited)")
ax.scatter(t_us, parity_estimate, color=RED, s=6, zorder=3, label="500-shot estimate")
ax.axvline(t_gate * 1e6, color=GREY, linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xlabel("time (µs)")
ax.set_ylabel("parity")
ax.set_title("Parity-scan readout — MS gate")
ax.legend(frameon=False)
plt.show()
```

**Takeaway.** Parity `⟨σ_z σ_z⟩ → +1` at `t_gate` proves the two spins
are perfectly z-correlated (never one bright and one dark) — but a
*classical* mixture `½|↓↓⟩⟨↓↓| + ½|↑↑⟩⟨↑↑|` gives the identical `+1`
and identical single-ion marginals, so `ParityScan` cannot tell them
apart. That is precisely why Steps 3–4 add entanglement witnesses a
mixture cannot fake.

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

print(f"Concurrence: C(0) = {c_trajectory[0]:.4f}  →  C(t_gate) = {c_trajectory[-1]:.4f}")
print(f"EoF:         EoF(0) = {eof_trajectory[0]:.4f}  →  EoF(t_gate) = {eof_trajectory[-1]:.4f}")
assert c_trajectory[0] == 0.0, "initial product state |↓↓, 0⟩ is fully separable, so spin-spin concurrence must be exactly 0"               # starts separable
assert abs(c_trajectory[-1] - 1.0) < 1e-4, "concurrence must reach its maximum of 1 at t_gate — the gate has produced a maximally-entangled Bell state"   # Bell state at t_gate

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(times * 1e6, c_trajectory, color=BLUE, label="concurrence $C$")
ax.plot(times * 1e6, eof_trajectory, color=GREEN, linestyle="--", label="EoF $E_F$")
ax.axvline(t_gate * 1e6, color=GREY, linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xlabel("time (µs)")
ax.set_ylabel("spin–spin entanglement")
ax.set_title("Concurrence and EoF — MS gate")
ax.legend(frameon=False)
plt.show()
```

`entanglement_of_formation_trajectory` follows the closed-form
Wootters relation from concurrence for two-qubit reduced
states, so its trajectory is a monotonic function of
`c_trajectory` with the same zeros and ones but a different
shape in between.

!!! warning "Common confusion — `C = 1` certifies *maximal*, not *which*"

    Concurrence hitting `1.0` certifies the spins are **maximally
    entangled** — it does **not** name the state. `|Φ⁻⟩`, `|Φ⁺⟩`, and
    `|Ψ±⟩` all have `C = 1`. Only a **fidelity** against a named target
    names the state: [Tutorial 5](05_custom_observables.md) does exactly
    that with the `|Φ⁻⟩⟨Φ⁻|` projector. The `|Φ⁻⟩` label on this
    tutorial's plots is knowledge of the MS-gate physics, not something
    `C` measured.

## Step 4 — Log-negativity (spin-vs-motion entanglement)

`log_negativity_trajectory` uses a different bipartition —
**spin subsystem vs. mode subsystem** — and reports
`E_N = log₂ ‖ρ^{T_A}‖₁`. This is the quantity that tells you
how much the spins are **entangled with the motion** mid-gate
(as opposed to the concurrence's between-spin entanglement
after the motion has been traced out):

!!! warning "Common confusion — the mid-gate `E_N` crest is *not* the gate's output"

    `E_N` measures **spin↔motion** entanglement, and it **must** fall
    back to `0` at `t_gate`. A residual `E_N > 0` there means the
    phase-space loop has **not** closed — the motion is still entangled
    with the spins, which is a gate *error*, not a product. The gate's
    deliverable is the spin↔spin **concurrence → 1**; the mid-gate
    `E_N ≈ 1.31` crest is a transient you want gone by the end.

```python
ln_trajectory = log_negativity_trajectory(
    result_states.states, hilbert=hilbert, partition="spins",
)

mid_idx = len(times) // 2
print(f"Log-negativity (spin|motion): E_N(0) = {ln_trajectory[0]:.4f},  "
      f"E_N(t_mid) = {ln_trajectory[mid_idx]:.4f},  "
      f"E_N(t_gate) = {ln_trajectory[-1]:.4f}")

# Three witnesses on one panel: concurrence, EoF, and log-negativity (scaled to [0,1]).
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(times * 1e6, c_trajectory, color=BLUE, label="concurrence $C$ (spin–spin)")
ax.plot(times * 1e6, eof_trajectory, color=GREEN, linestyle="--", label="EoF $E_F$ (spin–spin)")
ax.plot(times * 1e6, ln_trajectory, color=PURPLE, linestyle=":", label=r"log-negativity $E_N$ (spin|motion)")
ax.axvline(t_gate * 1e6, color=GREY, linestyle="--", linewidth=0.8, alpha=0.6, label="$t_\\mathrm{gate}$")
ax.set_xlabel("time (µs)")
ax.set_ylabel("entanglement measure")
ax.set_title("Three witnesses — Bell entanglement via MS gate")
ax.legend(frameon=False)
plt.show()
```

**Takeaway.** The three curves never crest together: log-negativity
peaks mid-gate exactly where concurrence is still suppressed. The
motion is acting as an **entangling bus** — its mid-gate spin↔motion
entanglement is transient, and closing the phase-space loop converts
it into the spin↔spin entanglement the gate is built to deliver.

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
