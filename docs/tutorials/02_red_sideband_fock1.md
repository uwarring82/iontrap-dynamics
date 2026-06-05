# Tutorial 2 — Red-sideband flopping from Fock ∣1⟩

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/02_red_sideband_fock1.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** Take the four-step pattern from [Tutorial 1](01_first_rabi_readout.md)
(configure → build → solve → read out), swap the carrier Hamiltonian
for the red-sideband variant, and start the motion in the first
phonon Fock state `|1⟩`. The flop now couples spin **and** motion:
driving the red sideband simultaneously flips `|↓⟩ → |↑⟩` and
removes one phonon. By the end of the tutorial you'll have built a
full `|↓, 1⟩ → |↑, 0⟩ → |↓, 1⟩ → …` trajectory and seen the
Lamb–Dicke-reduced Rabi rate `Ω_RSB = Ω η √n` emerge from the
physics helpers.

The reference script is
[`tools/run_benchmark_sideband.py`][demo] — the same scenario, run
as the Phase 0.F performance tripwire. Its committed output bundle
under [`benchmarks/data/01_single_ion_sideband_flopping/`][bundle]
includes the plot embedded below.

[demo]: https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/run_benchmark_sideband.py
[bundle]: https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data/01_single_ion_sideband_flopping

**Expected time.** ~10 min reading; ~1 s runtime.

**Prerequisites.** [Tutorial 1](01_first_rabi_readout.md) — this
tutorial reuses its four-step skeleton without re-explaining the
shared bits. Background physics: a passing familiarity with the
Lamb–Dicke regime at the level of
[`CONVENTIONS.md`](../conventions.md) §10.

---

## The scenario

One ²⁵Mg⁺ ion in a trap with an axial mode at `ω_mode / 2π = 1.5 MHz`.
A 280 nm laser tuned to the **red sideband** — one mode quantum
below the atomic resonance, so each photon absorbed removes one
phonon. Rabi frequency `Ω / 2π = 100 kHz`, and the drive wavevector
is aligned along the mode axis (`k ∥ b`) so the Lamb–Dicke
projection is maximal. The initial state is `|↓, n=1⟩` — one phonon
of motion, spin in the ground state.

The analytic prediction for the flop is Rabi oscillation at the
Lamb–Dicke-suppressed rate

```
Ω_RSB(n) = Ω · η · √n
```

between `|↓, 1⟩` and `|↑, 0⟩`, with all other Fock states unpopulated
(the single-phonon manifold is closed under the RSB coupling). For
`η ≈ 0.26` and `n = 1`, `Ω_RSB / 2π ≈ 26 kHz`, so one flop cycle
takes `T_RSB = 2π / Ω_RSB ≈ 38 μs`.

## Step 1 — Configure

Identical to Tutorial 1's Step 1, but with the Rabi frequency
turned down by 10× — the sideband coupling is already
η-suppressed, so using `Ω / 2π = 100 kHz` keeps the flop duration
in the tens-of-microseconds range instead of hundreds:

```python
import matplotlib.pyplot as plt
import numpy as np

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

# House colours — shared across all figures in this tutorial.
BLUE, RED, GREEN, PURPLE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#444444"

mode = ModeConfig(
    label="axial",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
)
system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))

drive = DriveConfig(
    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
    carrier_rabi_frequency_rad_s=2 * np.pi * 0.1e6,  # Ω/2π = 100 kHz
    phase_rad=0.0,
)
```

!!! note "Why the drive is still called the 'carrier' rate"

    `DriveConfig.carrier_rabi_frequency_rad_s` is the **on-resonance
    Rabi frequency** of the laser — a property of the drive itself,
    not of which sideband it's tuned to. The sideband builders
    compute their own coupling rates from this carrier Ω plus the
    Lamb–Dicke parameter η; you never hand them `Ω_RSB` directly.

## Step 2 — Derive η from the setup and build the Hamiltonian

The Lamb–Dicke parameter

```
η = |k · b| · √(ℏ / (2 m ω_mode))
```

is a derived quantity — it falls out of the laser wavevector, the
mode eigenvector, the ion mass, and the mode frequency.
`iontrap_dynamics.analytic.lamb_dicke_parameter` computes it for
you from those four ingredients; knowing the value before the solve
starts is useful for sanity-checking the physics.

```python
from iontrap_dynamics.analytic import lamb_dicke_parameter, red_sideband_rabi_frequency
from iontrap_dynamics.hamiltonians import red_sideband_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace

hilbert = HilbertSpace(system=system, fock_truncations={"axial": 30})

eta = lamb_dicke_parameter(
    k_vec=drive.k_vector_m_inv,
    mode_eigenvector=mode.eigenvector_at_ion(0),
    ion_mass=mg25_plus().mass_kg,
    mode_frequency=mode.frequency_rad_s,
)
sideband_rabi_rate = drive.carrier_rabi_frequency_rad_s * eta  # Ω η for n=1

omega_rsb = red_sideband_rabi_frequency(
    carrier_rabi_frequency=drive.carrier_rabi_frequency_rad_s,
    lamb_dicke_parameter=eta,
    n_initial=1,
)
flop_period = 2 * np.pi / omega_rsb
print(f"η = {eta:.4f}")
print(f"Ω_RSB(n=1)/2π = {omega_rsb / (2 * np.pi):.1f} Hz  (Ω·η·√1 = {sideband_rabi_rate / (2 * np.pi):.1f} Hz)")
print(f"T_RSB = {flop_period * 1e6:.1f} μs  →  π-pulse at {flop_period / 2 * 1e6:.1f} μs")

hamiltonian = red_sideband_hamiltonian(hilbert, drive, "axial", ion_index=0)
```

`red_sideband_hamiltonian` returns the leading-order Lamb–Dicke
interaction-picture Hamiltonian (CONVENTIONS.md §10). Every
`|n⟩ ↔ |n − 1⟩` transition gets its correct `√n` prefactor —
the builder embeds the annihilation operator on the named mode and
the spin-raising / -lowering operators on the chosen ion. For
off-resonant (detuned) sideband dynamics, swap in
`detuned_red_sideband_hamiltonian` — same signature plus a
`detuning_rad_s` argument.

!!! tip "Fock truncation choice"

    `N_Fock = 30` is the workplan §0.F gate for this scenario; for
    a pure `|↓, 1⟩` start the population stays pinned at
    `{|↓, 1⟩, |↑, 0⟩}` so anything above `N_Fock = 3` would
    technically suffice. Keeping `N_Fock = 30` lets the same code
    run on thermally-warm initial states without retuning, and the
    CONVENTIONS.md §13 Fock-saturation check will tell you if your
    choice becomes a problem mid-trajectory.

## Step 3 — Solve and inspect both spin and motion

The RSB coupling entangles spin with motion, so the **interesting**
observable is no longer `⟨σ_z⟩` alone — we also want `⟨n̂⟩` to see
the phonon being removed as the spin flips up. The
`observables.number` factory returns the phonon-number operator
embedded on the full Hilbert space; the solver computes both
expectations in one call.

```python
import qutip

from iontrap_dynamics.observables import number, spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.sequences import solve

# |↓, n = 1⟩ — one phonon of motion, spin down.
psi_0 = qutip.tensor(spin_down(), qutip.basis(30, 1))

# Duration = two full flop cycles at the expected Ω_RSB rate.
times = np.linspace(0.0, 2 * flop_period, 200)

result = solve(
    hilbert=hilbert,
    hamiltonian=hamiltonian,
    initial_state=psi_0,
    times=times,
    observables=[spin_z(hilbert, 0), number(hilbert, "axial")],
)

sigma_z = np.asarray(result.expectations["sigma_z_0"])
n_mode = np.asarray(result.expectations["n_axial"])

# π-pulse: at t = T_RSB/2 the spin has flipped and the phonon has been absorbed.
idx_pi = np.argmin(np.abs(times - flop_period / 2))
print(f"At π-pulse  (t = {times[idx_pi] * 1e6:.1f} μs):  ⟨σ_z⟩ = {sigma_z[idx_pi]:+.4f},  ⟨n̂⟩ = {n_mode[idx_pi]:.4f}")
print(f"At 2π-pulse (t = {times[-1] * 1e6:.1f} μs):  ⟨σ_z⟩ = {sigma_z[-1]:+.4f},  ⟨n̂⟩ = {n_mode[-1]:.4f}")

# Plot the coupled spin-motion trajectory over two flop cycles.
t_us = times * 1e6
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(t_us, sigma_z, color=BLUE, label=r"$\langle\sigma_z\rangle$")
ax.plot(t_us, n_mode, color=RED, label=r"$\langle\hat{n}\rangle$")
ax.axvline(flop_period / 2 * 1e6, color=GREY, linewidth=0.8, linestyle="--")
ax.set_xlabel(r"time  $t$  (μs)")
ax.set_ylabel("expectation value")
ax.set_title(r"RSB flop: $|\downarrow,1\rangle \leftrightarrow |\uparrow,0\rangle$")
ax.legend(frameon=False)
plt.show()
```

Two observables, one solve. At half the flop period `t = π / Ω_RSB`
the ion has completed a `|↓, 1⟩ → |↑, 0⟩` π-pulse: `⟨σ_z⟩ = +1`
(spin flipped) and `⟨n̂⟩ = 0` (phonon absorbed). At the full flop
period the state returns to `|↓, 1⟩`. The two trajectories are
phase-locked by the conservation law — during the RSB flop,
`⟨σ_z⟩ = 1 − 2 ⟨n̂⟩` holds identically, and verifying that bears
out a useful sanity check on your setup:

```python
print(f"Conservation law  max |⟨σ_z⟩ − (1 − 2⟨n̂⟩)| = {np.max(np.abs(sigma_z - (1 - 2 * n_mode))):.2e}")
assert np.max(np.abs(sigma_z - (1 - 2 * n_mode))) < 1e-6
```

## Step 4 — Read out spin and motion independently

If you want the finite-shot readout numbers from Tutorial 1, the
same `SpinReadout` protocol works unchanged — the readout layer is
agnostic to the dynamics that produced the trajectory. The
`bright_fraction` time series shows how the sampled bright count
tracks the underlying `⟨σ_z⟩` trajectory:

```python
from iontrap_dynamics import DetectorConfig, SpinReadout

detector = DetectorConfig(efficiency=0.5, dark_count_rate=0.3, threshold=3)
readout = SpinReadout(
    ion_index=0, detector=detector, lambda_bright=20.0, lambda_dark=0.0
)
measurement = readout.run(result, shots=500, seed=20260421)
bright_fraction = measurement.sampled_outcome["spin_readout_bright_fraction"]

print(f"Bright fraction at π-pulse   (t = {times[idx_pi]*1e6:.1f} μs): {bright_fraction[idx_pi]:.3f}  (ideal ≈ 1.0)")
print(f"Bright fraction at 2π-pulse  (t = {times[-1]*1e6:.1f} μs): {bright_fraction[-1]:.3f}  (ideal ≈ 0.0)")

# The sampled bright fraction follows the spin-up population with
# finite-statistics noise from the detector model.
p_up_ideal = (sigma_z + 1.0) / 2.0  # ⟨P_↑⟩ from the exact trajectory
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(t_us, p_up_ideal, color=GREY, linewidth=1.0, label=r"$\langle P_\uparrow\rangle$ (exact)")
ax.scatter(t_us, bright_fraction, color=BLUE, s=6, zorder=3, label="bright fraction (500 shots)")
ax.set_xlabel(r"time  $t$  (μs)")
ax.set_ylabel("spin-up population")
ax.set_title("Step 4 · sampled readout vs exact trajectory")
ax.legend(frameon=False)
plt.show()
```

A complementary "motional thermometry" step — reading out the
phonon number — uses the Dispatch O
[`SidebandInference`](../phase-1-architecture.md) protocol instead.
That's its own Tutorial (planned).

## Putting it together

The committed reference run of this exact scenario (plus two full
flop cycles at `N_Fock = 30`, 200 time points) produces:

![Single-ion sideband flopping](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/01_single_ion_sideband_flopping/plot.png)

Top panel: `⟨σ_z⟩` oscillates between −1 and +1 at the Ω_RSB rate.
Bottom panel: `⟨n̂⟩` oscillates between 1 and 0 in anti-phase — the
conservation law `⟨σ_z⟩ + 2⟨n̂⟩ = 1` holds element-wise. First
minimum of `⟨n̂⟩` (and first maximum of `⟨σ_z⟩`) lands at
`t ≈ 19 μs = T_RSB / 2` — the π-pulse timing for a sideband-cooling
step.

Wall-clock for the full 200-step, `N_Fock = 30` solve on a 2023
M2 MacBook Air: ~1.5 s (see [Benchmarks](../benchmarks.md)).

## Physics you can probe next

The four-step template carries over cleanly; only the builder or
initial state changes. Three natural modifications:

### Detuned sideband

Swap `red_sideband_hamiltonian` for
`detuned_red_sideband_hamiltonian` and add a non-zero detuning in
the `DriveConfig`. The trajectory gains off-resonant modulation
on top of the bare Rabi oscillation — Fourier content at the
generalised sideband rate `√(Ω_RSB² + δ²)`.

### Higher-n starting state

Replace `qutip.basis(30, 1)` with `qutip.basis(30, 5)`. Because
`Ω_RSB(n) = Ω η √n`, the flop at `n = 5` runs `√5 ≈ 2.24×` faster
than at `n = 1` — the analytic rate prediction from Step 2 updates
to `Ω_RSB(5) = Ω η √5`.

### Full Lamb–Dicke vs leading order

`red_sideband_hamiltonian(..., full_lamb_dicke=True)` builds the
full Wineland–Itano Laguerre-polynomial coupling matrix instead of
the leading-order `η √n` approximation. The difference is
invisible at `η ≈ 0.26` and `n = 1` (both rates agree to
sub-percent). It becomes important at `η > 0.3` or `n > 10` — the
leading-order approximation systematically over-estimates the
coupling. See Tutorial 8 (planned) for the full comparison.

## Where to next

- [Tutorial 1](01_first_rabi_readout.md) — the carrier-resonance
  baseline this tutorial builds on.
- [Phase 1 Architecture](../phase-1-architecture.md) — reference
  documentation for the full builder / observable / protocol
  surface.
- [`tools/run_benchmark_sideband.py`][demo] — the runnable script
  that produced the plot embedded above; diff it against this
  tutorial for the exact code.

---

## Licence

Sail material — adaptive guidance with specific parameter choices,
not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
