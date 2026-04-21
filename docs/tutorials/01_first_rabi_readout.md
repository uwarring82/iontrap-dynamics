# Tutorial 1 — Carrier Rabi flopping with finite-shot readout

**Goal.** By the end of this tutorial you will have built a complete
simulation pipeline that drives a single ²⁵Mg⁺ ion through a carrier
Rabi flop, reads out the spin with a projective Poisson-counting
detector, and puts 95 % Wilson confidence intervals on a finite-shot
bright-fraction estimator. Every step mirrors a line in the
reference tool [`tools/run_demo_wilson_ci.py`][demo], which you can
also run unmodified.

[demo]: https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/run_demo_wilson_ci.py

This is the canonical "Hello world" for `iontrap-dynamics`. The
scenario exercises every architectural layer introduced through
v0.2: configuration objects (`IonSystem`, `DriveConfig`,
`ModeConfig`), the `HilbertSpace`, a Hamiltonian builder, the
`solve()` dispatcher, a `SpinReadout` protocol, and the Wilson
`binomial_summary` estimator. Each layer is motivated in ≤ 3
sentences — the point is to show **how the pieces compose**, not to
re-derive ion physics.

**Expected time.** ~10 min reading; ~1 s runtime.

**Prerequisites.** A working install (`pip install -e ".[dev]"` in
the repo root) and familiarity with ion-trap terminology at the
level of [`CONVENTIONS.md`](../conventions.md) §1 and §3. No
physics derivations required — the formulas appear only for
orientation.

---

## The scenario

One ²⁵Mg⁺ ion held in a Paul trap with an axial motional mode at
`ω_mode / 2π = 1.5 MHz`. A laser drives the `|↓⟩ ↔ |↑⟩` electronic
transition on resonance (zero detuning) at Rabi frequency
`Ω / 2π = 1 MHz`. Starting from the ground state `|↓, 0⟩`, we
expect the textbook carrier flop

```
⟨σ_z⟩(t) = −cos(Ω t)
```

oscillating between `−1` (spin down) and `+1` (spin up) with period
`T_Ω = 2π / Ω = 1 μs`. After simulating the noise-free dynamics,
we'll "read out" the spin at 80 shots per time point using a
finite-fidelity Poisson detector and plot the 95 % Wilson CI on
the bright-fraction estimator.

## Step 1 — Configure the physical system

Four configuration objects describe what's in the lab: which ion
species, which motional mode, which drive, and how those pieces
compose into an `IonSystem`. None of them commit to a basis or a
Hilbert-space dimension yet — that comes in Step 2.

```python
import numpy as np

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

# Single axial mode along +z. Eigenvector is per-ion; for a single
# ion this is just one row of shape (3,).
mode = ModeConfig(
    label="axial",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
)

# One ion, ²⁵Mg⁺. mg25_plus() returns a pre-built IonSpecies record
# with the 280 nm cycling-transition metadata.
system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))

# Carrier drive: on-resonance (no detuning field), zero phase. The
# wavevector magnitude comes from the 280 nm laser wavelength.
drive = DriveConfig(
    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
    carrier_rabi_frequency_rad_s=2 * np.pi * 1.0e6,
    phase_rad=0.0,
)
```

!!! note "Why the wavevector direction matters"

    The Lamb–Dicke coupling is controlled by `k · b_{i,m}` — the
    projection of the laser wavevector onto the mode's
    eigenvector at ion `i`. Here both vectors point along `+z` so
    the coupling is maximal. `CONVENTIONS.md` §10 nails down the
    sign convention so your dynamics reproduces exactly.

## Step 2 — Build the Hilbert space and Hamiltonian

`HilbertSpace` is the first object that commits to a finite
dimension. The `fock_truncations` dict gives a per-mode Fock
cutoff; the carrier Hamiltonian doesn't couple the motion so a
small cutoff (`N_Fock = 3`) is plenty for this scenario.

```python
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace

hilbert = HilbertSpace(system=system, fock_truncations={"axial": 3})
hamiltonian = carrier_hamiltonian(hilbert, drive, ion_index=0)
```

`carrier_hamiltonian` returns a QuTiP `Qobj` whose dims match
`hilbert.qutip_dims()` — spin subsystems first, modes after, per
`CONVENTIONS.md` §2. The builder chose the leading-order Lamb–Dicke
reduction by default; see Tutorial 8 (planned) for when the
`full_lamb_dicke=True` variant matters.

## Step 3 — Solve the noise-free dynamics

`sequences.solve()` wraps QuTiP's `sesolve` / `mesolve` and
packages the output as a frozen `TrajectoryResult`. We hand it the
Hamiltonian, a ket initial state, a time grid, and a list of
`Observable` records whose expectations we want computed.

```python
import qutip

from iontrap_dynamics.observables import spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.sequences import solve

psi_0 = qutip.tensor(spin_down(), qutip.basis(3, 0))  # |↓, n=0⟩

rabi_period = 2 * np.pi / drive.carrier_rabi_frequency_rad_s
times = np.linspace(0.0, 2 * rabi_period, 100)  # two full Rabi periods

result = solve(
    hilbert=hilbert,
    hamiltonian=hamiltonian,
    initial_state=psi_0,
    times=times,
    observables=[spin_z(hilbert, 0)],
)

sigma_z_trajectory = result.expectations["sigma_z_0"]
```

At this point `sigma_z_trajectory` is the **ideal** (noise-free)
expectation-value curve. Plotting it against `times` would give the
textbook cosine `−cos(Ω t)`. The `result.metadata.backend_name` is
`"qutip-sesolve"` because `psi_0` is a ket — `solve()` auto-
dispatches (see [Benchmarks](../benchmarks.md) for why sesolve is
the default for pure kets).

## Step 4 — Project the dynamics onto shot outcomes

The measurement layer turns an ideal expectation trajectory into
a finite-shot record. The `SpinReadout` protocol models the full
pipeline:

1. **Project** the qubit onto `|↑⟩` (bright) or `|↓⟩` (dark) per
   shot, with probability `p_↑(t) = (1 + ⟨σ_z⟩(t)) / 2`.
2. **Poisson-count** photons at the state-conditional rate — bright
   shots emit Poisson(`η · λ_bright + γ_d`), dark shots emit
   Poisson(`η · λ_dark + γ_d`).
3. **Threshold** the count against `N̂` to classify each shot as
   bright (1) or dark (0).

```python
from iontrap_dynamics import DetectorConfig, SpinReadout

detector = DetectorConfig(
    efficiency=0.5,             # 50 % collection + quantum efficiency
    dark_count_rate=0.3,        # 0.3 counts/shot from stray light
    threshold=3,                # classify bright if count ≥ 3
)

readout = SpinReadout(
    ion_index=0,
    detector=detector,
    lambda_bright=20.0,         # ~20 photons per shot when bright
    lambda_dark=0.0,            # no fluorescence when dark
)

measurement = readout.run(result, shots=80, seed=20260420)

bits = measurement.sampled_outcome["spin_readout_bits"]
# bits has shape (80, 100) — 80 shots × 100 time points.
```

`SpinReadout.run` returns a `MeasurementResult` with a **dual
view**: the ideal inputs (`p_up`, `bright_fraction_envelope`) that
tell you what the estimator converges to at infinite shots, and the
sampled outcomes (`counts`, `bits`, `bright_fraction`) that
approximate that envelope at the supplied shot budget.

## Step 5 — Put error bars on the finite-shot estimator

Raw bits aren't a number — the publishable quantity is the
**bright-fraction point estimate** `k / n` plus a confidence
interval. The `binomial_summary` helper returns both, with a choice
between Wilson (recommended default, near-nominal coverage at
modest `n`) and Clopper–Pearson (conservative, exact).

```python
from iontrap_dynamics.measurement import binomial_summary

successes = bits.sum(axis=0)  # shape (100,) — per-time-bin bright count
summary = binomial_summary(successes, trials=80, confidence=0.95, method="wilson")

# summary.point_estimate, summary.lower, summary.upper all have shape (100,)
```

The `summary` record names its method and confidence level so
downstream code or a plot caption can tag the CI correctly:

```python
print(summary.method, summary.confidence)
# wilson 0.95
```

A typical time bin has `(p̂ − lower, upper − p̂) ≈ (0.07, 0.07)` —
enough spread that the plotted CI band is visibly wider than the
point-estimate markers, but not so wide that the underlying cosine
signal is drowned out. Shot budget 80 is deliberately small here
so the CI band is legible; at 1000 shots it would be ~3× tighter.

## Putting it together

Wall-clock for the whole pipeline is under a second:

```text
>>> running Wilson-CI demo (carrier Rabi with finite-shot error bars)
    shots = 80, seed = 20260420, confidence = 0.95
    detector: η = 0.5, γ_d = 0.3, N̂ = 3, F = 0.9971
    trajectory elapsed: 0.003 s
    readout elapsed:    0.001 s
    Wilson coverage (single seed) = 0.960  (nominal 0.95; many-seed expectation)
    mean half-width = 0.071
```

![Wilson CI on carrier Rabi](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/wilson_ci_demo/plot.png)

The black dashed curve is the ideal `p_↑(t)`; the blue dashed curve
is the fidelity-limited envelope (what the estimator converges to at
infinite shots for this detector); the red markers are the 80-shot
point estimates; the red band is the Wilson 95 % CI. For this seed
96 % of the envelope lies inside the CI band — close to nominal.

## Next steps

Natural follow-ons, each parallel to a planned tutorial in the
[tutorials index](index.md):

- **Red-sideband physics** (Tutorial 2) — swap `carrier_hamiltonian`
  for `red_sideband_hamiltonian` and start from `|↓, n=1⟩` to watch
  a phonon get absorbed during the spin flip.
- **Pulse shaping** (Tutorial 3) — `modulated_carrier_hamiltonian`
  with a Gaussian envelope to build a calibrated π-pulse.
- **Entanglement observables** (Tutorial 4, via Dispatch Q) — run
  the MS-gate Bell-formation scenario and compute concurrence +
  spin-motion log-negativity.
- **Systematics study** (Tutorial 5, via Dispatch R / S) — add
  `RabiJitter` or `DetuningJitter` and see inhomogeneous dephasing
  emerge in the ensemble mean.

Each follow-on uses the same four-step pattern this tutorial
established (configure → build → solve → read out). The public
surface is small enough that once you've internalised this one
example, the rest of the library is mostly swapping builder / state
/ protocol calls into the same skeleton.

---

!!! tip "Reference implementation"

    The full script that produced the numbers and plot above is
    [`tools/run_demo_wilson_ci.py`][demo]. Run it with
    `python tools/run_demo_wilson_ci.py` (requires `matplotlib` for
    the plot; prints the summary line regardless). Every number in
    this tutorial is reproducible from that script.

## Licence

This tutorial is Sail material — adaptive guidance with specific
parameter choices, not a coastline constraint. Licensed under
**CC BY-NC-SA 4.0** per [`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
