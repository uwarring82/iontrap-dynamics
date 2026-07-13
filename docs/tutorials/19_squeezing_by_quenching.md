# Tutorial 19 — Squeezing a trapped ion by quenching its trap frequency

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/19_squeezing_by_quenching.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** Generate motional squeezing by varying an ion's trap frequency `ω(t)`
in time — no laser, just a fast change of the confinement. By the end you will
have built the time-dependent-frequency squeezing Hamiltonian, read the squeezing
back from the phase-space **covariance matrix**, seen the **sudden vs adiabatic**
crossover, watched the Wigner ellipse squeeze, grown squeezing linearly by
**parametric modulation**, and **optimised a single down/up pulse** for maximal
squeezing. This reproduces the single-ion physics of Wittemer et al., *Phil.
Trans. R. Soc. A* **378**, 20190230 (2020).

**Reference implementation.** `tools/run_benchmark_nonadiabatic_squeezing.py`,
with the committed plot under
[`benchmarks/data/nonadiabatic_squeezing/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data/nonadiabatic_squeezing).

**Expected time.** ~16 min reading; ~4 s runtime.

**Prerequisites.** [Tutorial 9](09_squeezed_coherent_prep.md) (single-mode
squeezed-state factory) and [Tutorial 6](06_fock_truncation.md) (Fock-truncation
diagnosis). CONVENTIONS.md **§26** fixes the squeezing generator, the
vacuum-variance-1 quadrature normalisation, and the Wigner scaling used here.

!!! note "New here? Read this first"
    - A **motional mode** is just a quantum harmonic oscillator — the ion sloshing back and forth in its trap.
    - `fock=40` keeps phonon states `|0⟩ … |39⟩`: the truncated ladder we actually compute on.
    - `ω(t)` is the **trap frequency**, the one knob we vary in time.
    - Changing `ω(t)` **quickly** creates **squeezing**; changing it slowly and steadily (adiabatically) does not.
    - `r` measures **how squeezed** the state is (`r = 0` → not squeezed at all); `ν ≈ 1` means it is still **pure** (no heating).
    - **In a hurry?** Run Steps 1, 2, and 5 for the core story; the rest cover the sudden/adiabatic limits, the Wigner picture, and single-pulse optimisation.

**Symbols in this tutorial**

| Symbol | Meaning |
|---|---|
| `ω(t)` | trap frequency — the control knob |
| `r` | squeezing strength (`r = 0` → none) |
| `ν` | symplectic eigenvalue (`ν = 1` → pure, `ν > 1` → mixed) |
| `n̄_sq` | mean phonons created by squeezing (`= sinh²r`) |
| `α` | coherent displacement — the state's centre (`0` here) |

---

## The scenario

A harmonic oscillator whose frequency `ω(t)` is changed in time is one of the
oldest problems in quantum mechanics — and the workhorse of *analogue-gravity*
experiments. In the **fixed** operator basis of the initial frequency `ω(0)`, the
evolution is (CONVENTIONS §26.1, after Silveri 2015)

```
H(t)/ℏ = ω(t) (â†â + ½) − (i/4)(d ln ω/dt)(â†² − â²).
```

The second term — switched on **only while ω is changing** — is a squeezing
generator. Change `ω` slowly (adiabatically) and the state just follows the
instantaneous ground state; change it fast (non-adiabatically) and the state is
left **squeezed**: pairs of phonons are torn out of the vacuum. We drive this
with a [`FrequencyWaveform`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/waveforms.py)
that carries both `ω(t)` and its analytic log-derivative.

## Step 1 — Build the ω(t) squeezing Hamiltonian and evolve a fast ramp

We use a smooth `tanh` ramp from `ω_i` down to `ω_f = ½ ω_i`. Its total log-swing
is fixed at `ln(ω_f/ω_i)` regardless of width, so narrowing the ramp takes us to
the **sudden** limit, where the generated squeezing is exactly
`r = ½|ln(ω_f/ω_i)|`.

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip

from iontrap_dynamics import gaussian, phase_space, waveforms
from iontrap_dynamics.hamiltonians import nonadiabatic_squeezing_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#444444"
TWOPI = 2.0 * np.pi


# Boilerplate: build a one-ion, one-mode Hilbert space. The spin is an inert
# spectator here — all the physics lives in the motional mode.
def single_mode(fock, freq_hz=2.0e6):
    mode = ModeConfig(
        label="m",
        frequency_rad_s=TWOPI * freq_hz,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={"m": fock})


# The one function to focus on: waveform → Hamiltonian → solve → final motional state.
def evolve(hilbert, wave, tmax, n_times):
    fock = hilbert.fock_truncations["m"]
    psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(fock, 0))
    hamiltonian = nonadiabatic_squeezing_hamiltonian(hilbert, "m", wave, validate_at=(0.0, tmax))
    return solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=psi0,
        times=np.linspace(0.0, tmax, n_times),
        storage_mode=StorageMode.EAGER,
    )


w_i, w_f = 2.0e6, 1.0e6
hilbert = single_mode(fock=40, freq_hz=w_i)
period = 1.0 / w_i
width = 0.01 * period  # narrow → sudden
tmax = 55.0 * width
n_times = 1500
ramp = waveforms.smooth_ramp(
    omega_i=TWOPI * w_i, omega_f=TWOPI * w_f, center_s=25.0 * width, width_s=width
)
result = evolve(hilbert, ramp, tmax, n_times)
mode_state = gaussian.reduced_single_mode(result.states[-1], hilbert, "m")

r_sudden = gaussian.squeezing_parameter(gaussian.covariance_matrix(mode_state)[0])
r_oracle = 0.5 * abs(np.log(w_f / w_i))
print(f"sudden ramp: r = {r_sudden:.4f}   oracle ½|ln(ω_f/ω_i)| = {r_oracle:.4f}")
assert abs(r_sudden - r_oracle) / r_oracle < 0.05
```

The squeezing appears **exactly while `ω(t) is changing`**, then stays put. Reading
the covariance along the trajectory (the eigenvalue-ratio `r` is rotation-invariant,
so it plateaus after the quench) tells the whole story in one figure:

```python
t_grid = np.linspace(0.0, tmax, n_times)
sample = np.arange(0, n_times, 20)
omega_profile = np.array([ramp.omega(t) for t in t_grid[sample]]) / (TWOPI * w_i)
r_of_t = np.array([
    gaussian.squeezing_parameter(
        gaussian.covariance_matrix(gaussian.reduced_single_mode(result.states[k], hilbert, "m"))[0]
    )
    for k in sample
])

fig, ax = plt.subplots(figsize=(6.5, 4.0))
t_us = t_grid[sample] * 1e6
ax.plot(t_us, r_of_t, color=BLUE, lw=2, label="squeezing r(t)")
ax.axhline(r_oracle, color=RED, ls="--", label="sudden oracle")
ax.set_xlabel("time [µs]")
ax.set_ylabel("squeezing r", color=BLUE)
ax.set_title("squeezing appears exactly when the trap is quenched")
ax.legend(loc="center right")
ax2 = ax.twinx()
ax2.plot(t_us, omega_profile, color=GREEN, ls="--")
ax2.set_ylabel("trap frequency ω(t)/ω_i", color=GREEN)
fig.tight_layout()

# r is rotation-invariant, so its final value is the sudden oracle.
assert abs(r_of_t[-1] - r_oracle) / r_oracle < 0.05
```

**Takeaway.** Squeezing only grows *while* `ω(t)` is changing; once the ramp
finishes, `r` plateaus and stays put.

## Step 2 — Read the squeezing back from the covariance matrix

The readout lives in [`iontrap_dynamics.gaussian`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/gaussian.py).
From the reduced mode state it builds the 2×2 covariance `V` (quadratures
`x̂ = â + â†`, `p̂ = i(â† − â)`, vacuum variance 1) and reports the squeezing
`r = ¼ ln(λ_max/λ_min)` (the **eigenvalue ratio**, not `tr V`), the symplectic
eigenvalue `ν = √(det V)` (purity: `ν = 1` for a pure state), and `n̄_sq = sinh²r`.

```python
readout = phase_space.phase_space_readout(mode_state)
print(f"r = {readout.squeezing_parameter:.4f}   ν = {readout.symplectic_eigenvalue:.4f}"
      f"   n̄_sq = {readout.mean_squeezed_occupation:.4f}   |α| = {abs(readout.coherent_amplitude):.2e}")

# The state stayed pure (ν = 1) and displacement-free (the centred generator
# preserves parity: ⟨â⟩ = 0 from vacuum).
assert abs(readout.symplectic_eigenvalue - 1.0) < 1e-3, "ν ≈ 1: the squeezing is unitary — no heating. If ν > 1, the Fock cutoff is likely too small."
assert abs(readout.coherent_amplitude) < 1e-6, "Squeezing must not move the state's centre (α ≈ 0)."
assert abs(readout.mean_squeezed_occupation - np.sinh(r_oracle) ** 2) / np.sinh(r_oracle) ** 2 < 0.1
```

## Step 3 — Sudden kick vs cyclic adiabatic return

Step 1 gave the **sudden** one-way squeeze kick: a narrow ramp from `ω_i` to
`ω_f` gives `r = ½|ln(ω_f/ω_i)|`. The clean **adiabatic** oracle in §26 is
cyclic: ramp down slowly and then back up to the original frequency. In that
case the state returns to the original oscillator and the residual squeezing
goes to zero (`r → 0`). Do not use a one-way frequency change as the `r → 0`
regression; the convention gate is the cyclic waveform.

!!! warning "Common confusion — one-way ≠ cyclic"
    A slow **one-way** ramp leaves the oscillator at a *different* frequency, so the
    readout basis is no longer the instantaneous ground state — it is not the same as
    the **cyclic** adiabatic test. The `r → 0` oracle applies only when the trap
    returns to its original frequency.

```python
def cyclic_down_up(width_s):
    """Smoothly ramp ω_i → ω_f → ω_i with analytic d ln ω/dt."""
    first_center = 5.0 * width_s
    second_center = 15.0 * width_s
    ln_half_swing = 0.5 * np.log(w_f / w_i)

    def omega(t):
        log_swing = ln_half_swing * (
            np.tanh((t - first_center) / width_s) - np.tanh((t - second_center) / width_s)
        )
        return TWOPI * w_i * np.exp(log_swing)

    def d_ln_omega_dt(t):
        s1 = 1.0 / np.cosh((t - first_center) / width_s) ** 2
        s2 = 1.0 / np.cosh((t - second_center) / width_s) ** 2
        return ln_half_swing * (s1 - s2) / width_s

    return waveforms.FrequencyWaveform(omega=omega, d_ln_omega_dt=d_ln_omega_dt)


widths = np.array([0.1, 0.3, 1.0, 3.0]) * period
r_cyclic = []
for w in widths:
    hil_c = single_mode(50, w_i)
    state = gaussian.reduced_single_mode(evolve(hil_c, cyclic_down_up(w), 20.0 * w, 2500).states[-1], hil_c, "m")
    r_cyclic.append(gaussian.squeezing_parameter(gaussian.covariance_matrix(state)[0]))
r_cyclic = np.array(r_cyclic)
print("width/T_i :", widths / period)
print("cyclic r  :", np.round(r_cyclic, 5))

# The one-way sudden kick hits the analytic oracle; the cyclic ramp's residual
# squeezing vanishes as the down/up waveform becomes adiabatic.
assert abs(r_sudden - r_oracle) / r_oracle < 0.05
assert np.all(np.diff(r_cyclic) < 0.0)
assert r_cyclic[-1] < 1e-2

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.semilogx(widths / period, r_cyclic, "o-", color=BLUE, label="cyclic down/up residual")
ax.axhline(r_oracle, color=RED, ls="--", label="one-way sudden kick")
ax.axhline(0.0, color=GREY, ls=":", label="adiabatic limit")
ax.set_xlabel("ramp width / trap period")
ax.set_ylabel("squeezing r")
ax.set_title("cyclic residual squeezing vanishes adiabatically")
ax.legend()
fig.tight_layout()
```

## Step 4 — See it in phase space: the Wigner ellipse

The [`phase_space.wigner`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/phase_space.py)
wrapper pins QuTiP's scaling to the §26.2 vacuum-variance-1 convention (`g = 1`),
so the Wigner ellipse's principal widths are exactly the covariance eigenvalues
`e^{∓2r}`. The vacuum is a unit circle; squeezing flattens it.

```python
grid = np.linspace(-4.0, 4.0, 201)
w_vac = phase_space.wigner(qutip.basis(40, 0), grid)
w_sqz = phase_space.wigner(mode_state, grid)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.0, 4.2))
for ax, w, title in ((ax0, w_vac, "vacuum"), (ax1, w_sqz, f"squeezed (r = {r_sudden:.2f})")):
    ax.contourf(grid, grid, w, levels=30, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\hat x$")
    ax.set_ylabel(r"$\hat p$")
    ax.set_title(title)
fig.tight_layout()

# The squeezed ellipse's narrow axis is e^{-2r} of the vacuum's unit variance.
cov, _ = gaussian.covariance_matrix(mode_state)
narrow_axis = float(np.min(np.linalg.eigvalsh(cov)))
assert abs(narrow_axis - np.exp(-2.0 * r_sudden)) < 0.02
```

## Step 5 — Grow squeezing continuously: parametric modulation

Instead of one fast quench, **modulate** `ω(t)` sinusoidally at twice the trap
frequency, `ω_mod = 2 ω_ini`. This is degenerate parametric amplification: the
squeezing grows **linearly** with the modulation duration, `r = ½ δω · T_mod`, so
`n̄_sq = sinh²(2π g T_mod)` with coupling `g = δω/(4π)`. It is intrinsically
displacement-free.

```python
w_ini = 2.8e6
dw_mod = TWOPI * 8.0e3
hil_p = single_mode(fock=40, freq_hz=w_ini)
mod_period = TWOPI / (2.0 * TWOPI * w_ini)
t_list = np.array([40, 90, 140]) * mod_period
r_param = []
for tmax in t_list:
    wave = waveforms.sinusoidal_modulation(
        omega_ini=TWOPI * w_ini, mod_amplitude=dw_mod, mod_frequency=2.0 * TWOPI * w_ini
    )
    st = gaussian.reduced_single_mode(evolve(hil_p, wave, float(tmax), 2200).states[-1], hil_p, "m")
    r_param.append(gaussian.squeezing_parameter(gaussian.covariance_matrix(st)[0]))
r_param = np.array(r_param)
r_pred = 0.5 * dw_mod * t_list
print("T_mod [µs]:", np.round(t_list * 1e6, 2))
print("r sim     :", np.round(r_param, 4))
print("r = ½δω·T :", np.round(r_pred, 4))

# Linear growth in the modulation duration.
assert np.allclose(r_param, r_pred, rtol=0.03)

# The pairs pile up: n̄_sq = sinh²r grows as sinh²(½δω·T_mod).
t_fine = np.linspace(0.0, t_list[-1], 100)
fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(t_fine * 1e6, np.sinh(0.5 * dw_mod * t_fine) ** 2, color=BLUE,
        label=r"$\sinh^2(\frac{1}{2}\delta\omega\,T_{mod})$")
ax.plot(t_list * 1e6, np.sinh(r_param) ** 2, "o", color=RED, ms=8, label="simulation")
ax.set_xlabel("modulation duration [µs]")
ax.set_ylabel(r"mean squeezed phonons $\bar n_{sq}$")
ax.set_title("parametric amplification: squeezing grows with duration")
ax.legend()
fig.tight_layout()
```

## Step 6 — Optimise a single pulse: the hold knob

Step 3 ramped `ω` down and back up **slowly** and got `r → 0`. But a down/up
**pulse** with *fast* ramps is a different animal: the down-ramp and the up-ramp
each deliver a sudden squeeze kick, and whether they **add or cancel** depends on
the phase the state accumulates during the hold in between (`∫ω dt`). At zero hold
the two ramps coincide — no net frequency excursion, no squeezing. Open the hold
and the two kicks interfere: constructively when the phase accumulated during the
hold reaches `∫ω dt = π/2` — which, because the hold sits at `ω_min = ½ω_i`, lands
at **half a trap period** — where the squeezing reaches ≈ **twice** a one-way ramp
of the same depth. Scanning the hold to find that maximum reproduces the `δτ` half
of the paper's joint *"iteratively adjust δτ and Δω to find maximal |r|"*
optimisation (the depth `Δω` is fixed here; it could be a second scan axis). We
drive it with the named [`down_up_pulse`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/waveforms.py)
shape (`ω_ini → ω_min → ω_ini`, carrying its analytic `d ln ω/dt`).

```python
ramp = 0.02 * period  # fast ramps → each transition is a sudden squeeze kick
holds = np.linspace(0.0, 1.0, 11) * period
r_of_hold = []
for hold in holds:
    center = 10.0 * ramp + 0.5 * hold
    pulse = waveforms.down_up_pulse(
        omega_ini=TWOPI * w_i, omega_min=0.5 * TWOPI * w_i,  # same depth as the one-way ramp
        ramp_width_s=ramp, hold_s=float(hold), center_s=center,
    )
    tmax = center + 0.5 * hold + 15.0 * ramp
    hil = single_mode(40, w_i)
    st = gaussian.reduced_single_mode(evolve(hil, pulse, tmax, 1500).states[-1], hil, "m")
    r_of_hold.append(gaussian.squeezing_parameter(gaussian.covariance_matrix(st)[0]))
r_of_hold = np.array(r_of_hold)
best = int(np.argmax(r_of_hold))
print("hold/T :", np.round(holds / period, 2))
print("r      :", np.round(r_of_hold, 4))
print(f"optimum: r = {r_of_hold[best]:.4f} at hold = {holds[best] / period:.2f}·T "
      f"= {r_of_hold[best] / r_oracle:.2f}× the one-way ramp")

# Zero hold cancels (r → 0); the constructive optimum near ½ trap period ≈ doubles
# a one-way ramp of the same depth.
assert r_of_hold[0] < 1e-2, "Zero hold = no frequency excursion = no squeezing."
assert abs(best - 5) <= 1  # optimum at the hold = 0.5·T grid point (± one grid step)
assert r_of_hold[best] > 1.5 * r_oracle
assert abs(r_of_hold[best] - 2.0 * r_oracle) < 0.15 * (2.0 * r_oracle), "The tuned pulse ≈ doubles a one-way ramp of the same depth."

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(holds / period, r_of_hold, "o-", color=BLUE, label="down/up pulse")
ax.axhline(r_oracle, color=RED, ls="--", label=r"one-way ramp $\frac{1}{2}|\ln(\omega_f/\omega_i)|$")
ax.axhline(2.0 * r_oracle, color=GREEN, ls=":", label="2× one-way")
ax.set_xlabel("hold time / trap period")
ax.set_ylabel("squeezing r")
ax.set_title("single-pulse optimisation: r oscillates with the hold")
ax.legend()
fig.tight_layout()
```

**Takeaway.** The *same* down/up shape gives anything from zero squeezing to twice
a one-way ramp — the hold time is a free knob, so tune it (not just the depth) for
maximal `r`.

## What you built

You generated squeezing with nothing but a time-dependent trap frequency, read it
back from the covariance matrix (the eigenvalue-ratio `r`, the purity `ν`, the
occupation `n̄_sq`), checked the sudden kick and cyclic adiabatic limits, visualised
the squeezed Wigner ellipse on the vacuum-variance-1 grid, grew squeezing
linearly by parametric modulation, and optimised a single down/up pulse — whose
squeezing oscillates with the hold time and peaks at ≈ twice a one-way ramp.
[Tutorial 20](20_phonon_pair_creation.md) reads the same states out as a
**phonon-number distribution**, shows the even-only phonon-**pair** signature, and
removes a parasitic displacement with a **purifying echo**.
