# Tutorial 11 — Systematics: jitter ensembles

**Goal.** Take the single-shot carrier-Rabi scenario from
[Tutorial 1](01_first_rabi_readout.md), layer a **shot-to-shot
Rabi-amplitude jitter** on it (σ = 3 % — a realistic
well-stabilised-laser figure), and run an ensemble of 200
independent trajectories to measure the **inhomogeneous-dephasing
signature**: the familiar Gaussian-envelope decay of
`⟨σ_z⟩` oscillations as the mis-calibrated Rabi frequencies
dephase from one another.

By the end you will have:

1. Wired up the three jitter primitives — `RabiJitter`,
   `DetuningJitter`, `PhaseJitter` — and seen why all three
   follow the same two-function pattern (dataclass spec +
   `perturb_*` materialiser).
2. Run a 200-trial Rabi-jitter ensemble through
   `sequences.solve_ensemble` in a single call.
3. Verified the ensemble-averaged `⟨σ_z⟩` against the analytic
   Gaussian-envelope dephasing prediction
   `⟨σ_z⟩(t) = −cos(Ω̄t) · exp(−(σΩ̄t)² / 2)`.
4. Understood the `n_jobs=1` default and when it's worth
   flipping to `n_jobs=-1` for parallel solves.

**Expected time.** ~12 min reading; ~3 s runtime.

**Prerequisites.** [Tutorial 1](01_first_rabi_readout.md) for the
single-shot baseline. [Tutorial 10](10_finite_shot_statistics.md)
is useful background for the "statistical error bar" vs
"ensemble error bar" distinction we return to at the end.

---

## The three jitter primitives

`iontrap_dynamics.systematics` ships three parameter-jitter
classes, each paired with a `perturb_*` helper that materialises
a tuple of perturbed `DriveConfig`s ready for
`solve_ensemble`:

| Spec              | Perturbation type       | Helper                    | Typical σ               |
|-------------------|--------------------------|---------------------------|---------------------------|
| `RabiJitter`      | multiplicative `(1+ε)·Ω` | `perturb_carrier_rabi`    | 0.01–0.05 (dimensionless) |
| `DetuningJitter`  | additive `δ + Δδ`        | `perturb_detuning`        | 2π · 10 Hz – 1 kHz        |
| `PhaseJitter`     | additive `φ + Δφ`        | `perturb_phase`           | 0.01–0.1 rad              |

All three share the same two-step pipeline: sample per-shot
perturbations (Gaussian, mean zero), build a tuple of perturbed
`DriveConfig`s, feed one Hamiltonian per drive into
`solve_ensemble`. The rest is numpy-stack aggregation.

!!! note "Physical meaning of each"

    - **RabiJitter** — laser-intensity noise, AOM amplitude
      jitter, magnetic-field-gradient variation across the ion
      crystal. Produces the inhomogeneous-dephasing Rabi-curve
      decay worked through below.
    - **DetuningJitter** — laser-frequency jitter, Zeeman-shift
      drift from a noisy magnetic field, AOM / EOM
      frequency-reference instability. Dephases Ramsey fringes
      and shifts carrier-vs-sideband selectivity.
    - **PhaseJitter** — optical-path-length fluctuations,
      vibration, AOM RF-phase jitter. Dephases
      multi-pulse interferometric observables but is invisible
      to single-pulse Rabi flopping (a uniform phase offset is
      a gauge choice).

## Step 1 — Build the scenario and declare the jitter

Same Tutorial 1 scenario, one extra line:

```python
import numpy as np
import qutip

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.sequences import solve_ensemble
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem
from iontrap_dynamics.systematics import RabiJitter, perturb_carrier_rabi

N_SHOTS = 200

mode = ModeConfig(
    label="axial",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
)
system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
hilbert = HilbertSpace(system=system, fock_truncations={"axial": 3})

drive = DriveConfig(
    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
    carrier_rabi_frequency_rad_s=2 * np.pi * 1.0e6,
    phase_rad=0.0,
)

jitter = RabiJitter(sigma=0.03)   # 3 % relative amplitude noise
```

`RabiJitter` is a frozen dataclass — it carries the distribution
parameter but does not itself sample. Sampling happens in
`perturb_carrier_rabi`, which takes a `seed` for
bit-reproducibility.

## Step 2 — Materialise 200 perturbed drives

```python
perturbed_drives = perturb_carrier_rabi(
    drive, jitter, shots=N_SHOTS, seed=20260421,
)
# Each drive is a DriveConfig with carrier_rabi scaled by an
# independent (1 + ε_i), ε_i ~ Normal(0, 0.03).
assert len(perturbed_drives) == N_SHOTS

# Sanity check on the empirical mean and std of the Rabi samples:
omega_samples = np.array(
    [d.carrier_rabi_frequency_rad_s for d in perturbed_drives]
) / drive.carrier_rabi_frequency_rad_s
print(f"empirical mean (1+ε): {omega_samples.mean():.4f}")   # ≈ 1.00
print(f"empirical std  (σ):  {omega_samples.std():.4f}")    # ≈ 0.03
```

!!! tip "Fixing the seed earns reproducibility"

    `perturb_carrier_rabi(..., seed=20260421)` makes the
    resulting tuple bit-reproducible given
    `(drive, jitter, shots, seed)`. For CI-level regression
    tests, fix the seed. For production ensemble estimates,
    leave it `None` so each run samples a fresh realisation
    (the ensemble-mean statistics are the same; only the
    per-trial particular trajectories change).

## Step 3 — Run the ensemble with `solve_ensemble`

One Hamiltonian per perturbed drive, all with the same shared
initial state / times / observables:

```python
hamiltonians = [
    carrier_hamiltonian(hilbert, d, ion_index=0)
    for d in perturbed_drives
]

psi_0 = qutip.tensor(spin_down(), qutip.basis(3, 0))
# 10 Rabi periods — long enough that dephasing is unmistakable.
times = np.linspace(0.0, 10e-6, 400)

results = solve_ensemble(
    hilbert=hilbert,
    hamiltonians=hamiltonians,
    initial_state=psi_0,
    times=times,
    observables=[spin_z(hilbert, 0)],
    n_jobs=1,         # see performance note below
)
assert len(results) == N_SHOTS
```

!!! note "`n_jobs=1` is the right default"

    The carrier-Rabi single-solve is ~1 ms on modern hardware —
    well under the joblib crossover documented in
    [Benchmarks](../benchmarks.md). For this exact scenario,
    `n_jobs=1` (serial) is **faster** than `n_jobs=-1` because
    the loky process-spawn + pickle overhead dominates. Flip
    to `n_jobs=-1` for scenarios where each solve takes >15 ms
    (two-ion MS gates with wide Fock truncation; long-duration
    full-Lamb–Dicke sideband solves).

## Step 4 — Aggregate and verify the dephasing envelope

Stack the per-trial `σ_z` trajectories into a `(N_SHOTS,
n_times)` array, then collapse the shot axis to get the
ensemble mean and dispersion:

```python
sz_stack = np.stack(
    [r.expectations["sigma_z_0"] for r in results], axis=0
)
assert sz_stack.shape == (N_SHOTS, 400)

ensemble_mean = sz_stack.mean(axis=0)
ensemble_std = sz_stack.std(axis=0)        # spread across trials
sem = ensemble_std / np.sqrt(N_SHOTS)      # standard error of the mean

# Analytic prediction for Rabi-jitter inhomogeneous dephasing:
#   ⟨σ_z⟩(t) = −cos(Ω̄·t) · exp(−(σ·Ω̄·t)² / 2)
omega_bar = drive.carrier_rabi_frequency_rad_s
envelope = np.exp(-0.5 * (jitter.sigma * omega_bar * times) ** 2)
predicted = -np.cos(omega_bar * times) * envelope
```

### What the numbers look like

A snapshot of the first, fourth, and tenth `σ_z` minima (exact
multiples of the carrier period at `Ω̄ / 2π = 1 MHz`, so
`t = n μs`):

| t       | ensemble mean | analytic envelope | ensemble std |
|---------|---------------|-------------------|--------------|
| 1 µs    | −0.986        | −0.982            | 0.020        |
| 4 µs    | −0.801        | −0.753            | 0.252        |
| 10 µs   | −0.242        | −0.169            | 0.662        |

Three things worth taking from the table:

1. **The ensemble mean tracks the analytic Gaussian envelope
   closely.** At 1 µs the match is within 0.4 %; at 4 µs the
   4-point gap is 6 % of the envelope magnitude (~1σ of
   ensemble noise for `N = 200`); at 10 µs the gap widens but
   remains consistent with SEM ≈ 0.047.
2. **`ensemble_std` grows as the oscillations dephase.** Early
   on, all 200 trajectories are tightly bunched around the
   same `⟨σ_z⟩`; by 10 µs they spread across nearly the full
   `[−1, +1]` range, driven by per-trial Rabi-frequency
   differences. This is the *spread* of individual trials, not
   the uncertainty on the mean.
3. **The SEM is the uncertainty on the mean.** At 10 µs
   `SEM = 0.047` — this is what a plotted error-bar band
   should use when you're claiming the ensemble-averaged
   `⟨σ_z⟩` matches the theoretical prediction, not the wider
   `ensemble_std`.

## Step 5 — Plot the mean with its SEM band, overlaid with theory

```python
import matplotlib.pyplot as plt

t_us = times * 1e6
plt.plot(t_us, predicted, color="black", linewidth=1.2,
         linestyle="--", label="analytic ⟨σ_z⟩")
plt.plot(t_us, ensemble_mean, color="steelblue", linewidth=1.8,
         label=f"ensemble mean (N={N_SHOTS})")
plt.fill_between(
    t_us, ensemble_mean - sem, ensemble_mean + sem,
    color="steelblue", alpha=0.25, label="±1 SEM",
)
plt.fill_between(
    t_us, predicted - np.abs(envelope), predicted + np.abs(envelope),
    color="black", alpha=0.07, label="±|envelope| (spread)",
)
plt.xlabel("time (µs)")
plt.ylabel(r"$\langle \sigma_z \rangle$")
plt.legend(loc="lower right")
```

The ensemble-mean curve hugs the analytic dashed line; the
inner SEM band is narrow even at 10 µs (it would continue to
shrink as `1/√N_SHOTS`), while the outer "spread" band marks
the region individual trials still fluctuate within.

## Variation — detuning jitter

The same template works for the other two primitives. For
`DetuningJitter` the `DriveConfig` initially has
`detuning_rad_s = 0` (on-resonance), so the jitter produces a
*centred* Gaussian distribution around zero detuning. Dephasing
here is Lorentzian-in-frequency-space rather than Gaussian, but
the pipeline pattern is identical:

```python
from iontrap_dynamics.hamiltonians import detuned_carrier_hamiltonian
from iontrap_dynamics.systematics import DetuningJitter, perturb_detuning

detuning_jitter = DetuningJitter(sigma_rad_s=2 * np.pi * 1e3)  # 1 kHz
perturbed_drives = perturb_detuning(
    drive, detuning_jitter, shots=N_SHOTS, seed=20260421,
)
hamiltonians = [
    detuned_carrier_hamiltonian(hilbert, d, ion_index=0)
    for d in perturbed_drives
]
# solve_ensemble + aggregate as before
```

The observable signature of `DetuningJitter` at small `σ` is a
shifted effective-Rabi-frequency distribution (on-resonance
`Ω_eff = Ω`; off-resonance `Ω_eff = √(Ω² + δ²)`), so the
dephasing envelope is **asymmetric** — the ensemble mean sags
earlier and harder than pure Rabi jitter would.

## Where to next

- [Tutorial 1](01_first_rabi_readout.md) — the single-shot
  baseline this tutorial layered jitter over.
- [Tutorial 10](10_finite_shot_statistics.md) — the
  complementary **statistical** uncertainty channel (finite
  shots on a single trajectory) vs this tutorial's
  **ensemble** channel (many trajectories, infinite shots per).
- [CONVENTIONS §18](../conventions.md) — the systematics-layer
  spec: primitive classes, their distributions, and the
  `perturb_*` helpers' bit-reproducibility guarantees.
- [`tools/run_demo_rabi_jitter.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/run_demo_rabi_jitter.py)
  — the runnable demo matching this scenario.
- Planned — **Tutorial 12** will push the same ensemble
  machinery through a two-ion MS gate with parity readout,
  showing how jitter sensitivity on a gate fidelity differs
  from jitter on a single-ion Rabi flop.

---

## Licence

Sail material — adaptive guidance with specific parameter choices,
not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
