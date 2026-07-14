# Tutorial 9 — Squeezed / coherent state preparation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/09_squeezed_coherent_prep.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** Move past `qutip.basis(N, n)` and
`qutip.thermal_dm(N, n̄)` for the motional initial state. This
tutorial walks through the three named single-mode state
factories — `coherent_mode`, `squeezed_vacuum_mode`,
`squeezed_coherent_mode` — plus the `compose_density` helper
that glues per-subsystem kets and density matrices into a
full-space initial state. By the end you will:

1. Build a coherent `|α⟩` and verify `⟨n̂⟩ = |α|²` on a
   Fock-truncated mode.
2. Build a squeezed vacuum `|ξ⟩` and see the quadrature
   variances compress and stretch by `e^(±2r)`.
3. Build a displaced-squeezed state `|α, ξ⟩` in the
   qc.py-compatible ordering (squeeze first, then displace).
4. Compose any of these with a spin initial condition through
   `compose_density` to produce the full-space initial state.
5. Observe the classical **Rabi-rate collapse** that falls out
   of driving the red sideband from a coherent state — a
   signature spectrum of many interfering Rabi rates, invisible
   from a pure `|n⟩` start.

**Expected time.** ~12 min reading; ~2 s runtime.

**Level.** `core` — assumes the basics (Tutorials 0–1).

**Prerequisites.** [Tutorial 2](02_red_sideband_fock1.md) — the
RSB dynamics used for the collapse demo. [Tutorial
8](08_full_lamb_dicke.md) — we use `full_lamb_dicke=True`
throughout this tutorial because a coherent state with
`|α|² = 4` populates Fock levels where leading-order rates are
already off by > 10 %.

---

!!! note "New here? Read this first"

    - This tutorial builds *motional* initial states beyond a pure Fock `|n⟩` or a thermal mixture: three named factories make coherent, squeezed-vacuum, and squeezed-coherent kets.
    - A **coherent** state `|α⟩` is the vacuum pushed off-centre in phase space — same round shape, just displaced; `⟨n̂⟩ = |α|²`.
    - A **squeezed vacuum** `|ξ⟩` stays centred at the origin but narrows one quadrature, widens the other, and fills *only even* Fock levels (it makes phonon pairs); `⟨n̂⟩ = sinh²r`.
    - A **squeezed-coherent** state does both — and the order is fixed by convention (squeeze first, then displace) because `D` and `S` do not commute.
    - `compose_density` glues a per-ion spin state and a per-mode motional state into the full-space initial density matrix `rho_0`.
    - Driving the red sideband from a coherent (not Fock) start makes `⟨σ_z⟩` oscillations **collapse**, because many Fock levels flop at different Rabi rates at once.
    - **In a hurry?** The three factory cells under *The three factories* are the core; the collapse scenario is the payoff demo.

**Symbols in this tutorial**

| symbol | plain meaning |
| --- | --- |
| `r` | squeeze strength; the quadrature *widths* scale by `e^(∓r)` |
| `α` | coherent (displacement) amplitude, complex; sets `⟨n̂⟩ = |α|²` |
| `φ` | squeeze angle; picks which quadrature is narrowed |
| `ξ = r·e^(2iφ)` | complex squeeze parameter (`z` in code); bundles `r` and `φ` |
| `n̄ = ⟨n̂⟩` | mean phonon number of the prepared state |
| `X, Y` | the two motional quadratures; squeezing compresses one, stretches the other |
| `η` | Lamb–Dicke parameter; sets the red-sideband Rabi coupling |

## The three factories

`iontrap_dynamics.states` exposes three helpers for non-Fock
motional prep. All three return a **ket** on a single mode of
dimension `fock_dim`; to embed in the full Hilbert space, pass
through `compose_density` (or `qutip.tensor`).

### `coherent_mode(fock_dim, alpha)`

```
|α⟩ = D(α) |0⟩,    D(α) = exp(α·a† − α*·a)
```

Mean phonon number `⟨n̂⟩ = |α|²`. The amplitude `α` is complex —
its phase rotates the coherent state around phase space.
Each step below **prints its key numbers and renders a P(n) bar
chart live**: in the notebook you watch the calculation produce
the result instead of trusting a static image.

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip
from iontrap_dynamics.states import coherent_mode

# House colours — match the exemplar Tutorial 18 palette.
BLUE, RED, GREEN, PURPLE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#444444"

N = 40

psi_coh = coherent_mode(N, alpha=2.0)
n_mean_coh = float(qutip.expect(qutip.num(N), psi_coh))
print(f"Coherent |α=2⟩  ⟨n̂⟩ = {n_mean_coh:.6f}  (expected 4.000000)")
assert abs(n_mean_coh - 4.0) < 1e-10, "coherent <n> = |alpha|^2 = 2^2 = 4 exactly (up to Fock truncation)"
# |α=2⟩ carries ⟨n̂⟩ = 4 exactly (within Fock truncation).

psi_rotated = coherent_mode(N, alpha=2.0 * np.exp(1j * np.pi / 2))
n_mean_rot = float(qutip.expect(qutip.num(N), psi_rotated))
print(f"Rotated |α=2i⟩  ⟨n̂⟩ = {n_mean_rot:.6f}  (same distribution, 90° phase shift)")
# Same ⟨n̂⟩ = 4, but rotated 90° in phase space.

pn_coh = np.abs(psi_coh.full().flatten()) ** 2

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.bar(range(N), pn_coh, color=BLUE, label=r"$|\alpha=2\rangle$ coherent")
ax.set_xlabel("Fock level $n$")
ax.set_ylabel("$P(n) = |\\langle n|\\alpha\\rangle|^2$")
ax.set_title(r"Coherent state $|\alpha=2\rangle$: Poissonian $P(n)$")
ax.set_xlim(-0.5, 20.5)
ax.legend(frameon=False)
plt.show()
```

!!! tip "Choosing `fock_dim` for coherent states"

    A coherent state with `|α|² = 4` concentrates its population
    around `n ≈ 4` with Poissonian tails
    `σ_n = |α| = 2`. A good rule is `fock_dim ≥ |α|² + 6·|α|`
    — for `α = 2` that's `≥ 16`; for `α = 5` it's `≥ 55`.
    [Tutorial 6](06_fock_truncation.md)'s `fock_tolerance`
    ladder will tell you if you have chosen too tight.

### `squeezed_vacuum_mode(fock_dim, z)`

```
|ξ⟩ = S(ξ) |0⟩,    S(ξ) = exp((ξ*·a² − ξ·a†²) / 2)
```

with `ξ = r·e^(2iφ)`. Mean phonon number
`⟨n̂⟩ = sinh²(|ξ|)`.

!!! note "Why the phase carries a factor of 2"

    `CONVENTIONS.md` §6 sets `z = r·exp(2iφ)` — the factor of 2
    reflects the π-period of the squeezing ellipse (a squeezing
    axis at physical angle φ maps to complex argument 2φ on the
    squeeze parameter). QuTiP's `qutip.squeeze(N, z)` uses this
    same convention; `squeezed_vacuum_mode` is a named alias
    that records the convention explicitly.

```python
from iontrap_dynamics.states import squeezed_vacuum_mode

# Real z = r squeezes the X-quadrature and anti-squeezes the Y-quadrature.
r = 1.0
psi_sq = squeezed_vacuum_mode(N, z=r)

# Quadrature-variance sanity check: Var(X) = e^(-2r)/2, Var(Y) = e^(+2r)/2.
a = qutip.destroy(N)
X = (a + a.dag()) / np.sqrt(2)
Y = -1j * (a - a.dag()) / np.sqrt(2)
var_x = float(qutip.variance(X, psi_sq))
var_y = float(qutip.variance(Y, psi_sq))
n_mean_sq = float(qutip.expect(qutip.num(N), psi_sq))
print(f"Squeezed vacuum |r=1.0⟩  ⟨n̂⟩ = {n_mean_sq:.6f}  (expected {np.sinh(r)**2:.6f})")
print(f"  Var(X) = {var_x:.5f}  (expected {np.exp(-2*r)/2:.5f},  compressed)")
print(f"  Var(Y) = {var_y:.5f}  (expected {np.exp(+2*r)/2:.5f},  stretched)")
assert abs(var_x - np.exp(-2 * r) / 2) < 1e-3  # 0.068
assert abs(var_y - np.exp(+2 * r) / 2) < 1e-3  # 3.695
assert abs(n_mean_sq - np.sinh(r) ** 2) < 1e-4, "squeezed-vacuum <n> = sinh^2(r) — squeezing raises the phonon number above the vacuum"  # 1.381 (Fock-truncation tolerance)

pn_sq = np.abs(psi_sq.full().flatten()) ** 2

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.bar(range(N), pn_sq, color=RED, label=r"$|r=1.0\rangle$ squeezed vac.")
ax.set_xlabel("Fock level $n$")
ax.set_ylabel("$P(n)$")
ax.set_title(r"Squeezed vacuum $|r=1.0\rangle$: even-$n$ only")
ax.set_xlim(-0.5, 20.5)
ax.legend(frameon=False)
plt.show()
```

**Takeaway.** Real `r` compresses `Var(X)` to `e^(−2r)/2` and stretches `Var(Y)` to `e^(+2r)/2`, but their product stays pinned at the vacuum's minimum-uncertainty floor `1/4` — squeezing *reshapes* the noise (rather than adding noise), even as it does raise the energy `⟨n̂⟩` to `sinh²r`.

Even with `⟨n̂⟩ = 1.38`, the state is **pure** (it's a ket). The
non-trivial motional character comes from the quadrature
asymmetry, not from a classical Fock mixture.

### `squeezed_coherent_mode(fock_dim, *, z, alpha)`

```
|α, ξ⟩ = D(α) · S(ξ) |0⟩
```

Squeeze first, then displace. Mean phonon number
`⟨n̂⟩ = |α|² + sinh²(|ξ|)`.

!!! warning "Common confusion — squeezing vs displacement"

    These are two *different* operations. **Displacement** `D(α)`
    slides the state's centre through phase space (`⟨x̂⟩, ⟨p̂⟩`
    shift) while leaving its shape and quadrature widths
    untouched. **Squeezing** `S(ξ)` keeps the centre at the
    origin but reshapes the noise — narrowing one quadrature,
    widening the other, and populating even Fock pairs. A
    squeezed-coherent state does both: `α` slides the blob, `r`
    squashes it.

```python
from iontrap_dynamics.states import squeezed_coherent_mode

psi_sc = squeezed_coherent_mode(N, z=1.0, alpha=2.0)
# ⟨n̂⟩ = |α|² + sinh²(|ξ|) = 4 + 1.381 = 5.381
n_mean_sc = float(qutip.expect(qutip.num(N), psi_sc))
print(f"Squeezed-coherent |α=2, r=1⟩  ⟨n̂⟩ = {n_mean_sc:.6f}  (expected {4.0 + np.sinh(1.0)**2:.6f})")
assert abs(n_mean_sc - 5.381) < 1e-3, "squeezed-coherent <n> = |alpha|^2 + sinh^2(r) = 4 + 1.381 (displacement and squeezing energies add)"

pn_sc = np.abs(psi_sc.full().flatten()) ** 2

# Plot all three P(n) distributions together for comparison.
ns = np.arange(N)
fig, ax = plt.subplots(figsize=(5.0, 3.2))
width = 0.28
ax.bar(ns - width, pn_coh, width=width, color=BLUE,   label=r"coherent $|\alpha=2\rangle$")
ax.bar(ns,          pn_sq,  width=width, color=RED,    label=r"squeezed vac. $|r=1\rangle$")
ax.bar(ns + width,  pn_sc,  width=width, color=GREEN,  label=r"squeezed coh. $|\alpha=2,r=1\rangle$")
ax.set_xlabel("Fock level $n$")
ax.set_ylabel("$P(n)$")
ax.set_title("Phonon distributions: three prepared states")
ax.set_xlim(-0.5, 18.5)
ax.legend(frameon=False)
plt.show()
```

**Takeaway.** Side by side, the three `P(n)` distributions show the division of labour — displacement lifts the coherent peak toward `n ≈ |α|²`, squeezing thins population onto even levels only, and the squeezed-coherent state shifts that rippled distribution up toward `n ≈ |α|²`, now filling the odd levels too.

!!! note "Why squeeze-then-displace, not the other way"

    The ordering matches the legacy `qc.py`
    `initialise_single_mode` reference scenario that the
    migration test tier diffs against. Physically the two
    orderings are related by a rotation of the squeezing
    ellipse plus a rescaling of the displacement amplitude;
    neither is "more correct", but the library picks one and
    pins it. The `z` and `alpha` arguments are keyword-only so
    a caller who wants the other ordering cannot write it
    accidentally — they have to invoke
    `qutip.squeeze(...) * qutip.displace(...)` by hand.

## Composing into the full Hilbert space

`compose_density` takes per-subsystem states (one spin state per
ion, one mode state per named mode) and returns a full-space
density matrix. Kets are promoted to density matrices
automatically:

```python
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import red_sideband_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import compose_density
from iontrap_dynamics.system import IonSystem

# --- Boilerplate: one axial mode on a single Mg-25 ion, Fock-truncated Hilbert space ---
mode = ModeConfig(
    label="axial",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
)
system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
hilbert = HilbertSpace(system=system, fock_truncations={"axial": N})

# --- The physics that matters: compose spin-down ⊗ coherent-axial into rho_0 ---
rho_0 = compose_density(
    hilbert,
    spin_states_per_ion=[spin_down()],               # ket — auto-promoted
    mode_states_by_label={"axial": coherent_mode(N, alpha=2.0)},
)
# rho_0 is a density matrix on the full space with dims [[2, 40], [2, 40]].
print(f"rho_0 dims = {rho_0.dims}  (full-space density matrix: spin ⊗ axial)")
```

`compose_density` enforces the CONVENTIONS §2 tensor order
internally and raises `ConventionError` if the spin list has the
wrong number of ions, or if `mode_states_by_label` has missing /
extra keys. Having one mode labelled `"axial"` and trying to
pass `{"radial": ...}` is an error at compose time rather than
silent tensor-mismatch debugging later.

!!! tip "`compose_density` vs inline `qutip.tensor`"

    For one-mode, one-ion scenarios, `qutip.tensor(spin_dm,
    mode_dm)` is shorter. `compose_density` earns its keep once
    the system has multiple modes (whose tensor order has to
    match `hilbert.system.modes`) or multiple ions, or when a
    single error message for "wrong number of subsystems
    supplied" is more useful than digging through a dims
    mismatch. Use `qutip.tensor` for quick work; use
    `compose_density` for anything you'll read three months
    later.

## The collapse-and-revive scenario

Driving the red sideband from `|↓, α⟩` populates a band of Fock
levels simultaneously, each with its own Rabi rate
`Ω_{n, n-1}^full` (Tutorial 8). The spin-motion entanglement
acquires *different* phases at different `n`, and the
superposition **dephases** as the phases decorrelate. The
signature is a collapse of `⟨σ_z⟩` oscillations toward zero on a
time scale set by `1 / Δω_rate`, where `Δω_rate` is the spread
of Rabi rates across the Fock distribution:

```python
from iontrap_dynamics.analytic import lamb_dicke_parameter
from iontrap_dynamics.observables import number, spin_z
from iontrap_dynamics.sequences import solve

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
print(f"Lamb–Dicke parameter η = {eta:.4f}")

# Full Lamb–Dicke because |α|²=4 populates n≈4 where the leading-order
# correction is already > 10% off (Tutorial 8).
hamiltonian = red_sideband_hamiltonian(
    hilbert, drive, "axial", ion_index=0, full_lamb_dicke=True,
)

# Characteristic Rabi period at ⟨n̂⟩ = 4 (leading-order estimate — the
# full rate is ~16% slower, but this is just a time-axis baseline).
flop_period = 2 * np.pi / (abs(eta) * np.sqrt(4.0) * drive.carrier_rabi_frequency_rad_s)
print(f"Estimated flop period at ⟨n̂⟩=4: {flop_period * 1e6:.2f} μs")
times = np.linspace(0.0, 8 * flop_period, 400)

result = solve(
    hilbert=hilbert,
    hamiltonian=hamiltonian,
    initial_state=rho_0,
    times=times,
    observables=[spin_z(hilbert, 0), number(hilbert, "axial")],
)

sigma_z = result.expectations["sigma_z_0"]
n_mode = result.expectations["n_axial"]

sz_max = float(np.max(sigma_z))
sz_std = float(np.std(sigma_z))
n_final = float(n_mode[-1])
print(f"Collapse demo — ⟨σ_z⟩ max = {sz_max:.3f},  σ(⟨σ_z⟩) = {sz_std:.3f}")
print(f"  ⟨n̂⟩ drift: {float(n_mode[0]):.3f} → {n_final:.3f} phonons  (red-sideband cooling)")
assert sz_max > 0.5       # clear Rabi oscillation at the start
assert sz_std < 0.5       # oscillations collapse — std well below ideal 1/√2 ≈ 0.71

times_us = times * 1e6
fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.2), sharex=True)
axes[0].plot(times_us, sigma_z, color=BLUE, linewidth=0.8)
axes[0].axhline(0.0, color=GREY, linewidth=0.5, linestyle="--")
axes[0].set_ylabel(r"$\langle\sigma_z\rangle$")
axes[0].set_title(r"Rabi-rate collapse from $|\downarrow,\alpha=2\rangle$")
axes[1].plot(times_us, n_mode, color=RED, linewidth=0.8)
axes[1].set_ylabel(r"$\langle\hat{n}\rangle$  (phonons)")
axes[1].set_xlabel(r"time (μs)")
plt.tight_layout()
plt.show()
```

!!! warning "Common confusion — collapse is not decoherence"

    No collapse operators are passed to `solve` here, so the
    evolution is fully **unitary**. The decaying `⟨σ_z⟩` is
    *reversible dephasing* — the spin entangles with the
    motional Fock level and the many Rabi rates fall out of
    step. The information lives in spin–motion correlations
    rather than being lost; an ideal Jaynes–Cummings spectrum
    would even show a partial revival. A `c_ops` decoherence
    channel, by contrast, is irreversible.

What you see:

- `⟨σ_z⟩(t)` starts at `−1` (spin down). A short burst of Rabi
  oscillations is visible early on, reaching `⟨σ_z⟩ ≈ +0.8` at
  the first maximum. But the oscillations **decay in
  amplitude** as the trajectory goes on — by `t ≈ 150 μs`
  (eight `n = 4` periods), the trajectory std over time is
  `σ(σ_z) ≈ 0.26` against a naïve `√2 / 2 ≈ 0.71` for a
  single pure flop — unmistakable collapse.
- `⟨n̂⟩(t)` decays steadily from 4 toward ~3 — the ion is
  unilaterally transferring phonons to the spin because the
  red sideband drives `|↓, n⟩ → |↑, n−1⟩` at every `n > 0`
  component of the coherent state.
- For an ideal Jaynes-Cummings spectrum (no η-corrections), a
  **revival** would appear at
  `t_rev ≈ 2π · √n̄ / (|η| · Ω)` — a partial return of the
  oscillation amplitude as the rate-spread phases re-align.
  With full Lamb–Dicke this revival gets smeared out by the
  Laguerre structure; the collapse is the robust signature of
  the coherent superposition, the revival less so.

The same builder + solve pipeline with a pure `|↓, 4⟩`
initial state would give a *clean* `√4 = 2` Rabi oscillation
with no collapse — the collapse is a specific signature of the
coherent superposition, not a generic feature of high-`n̄`
starts.

## Two variations to try

### Squeezed-vacuum on the red sideband

Replace `coherent_mode(N, alpha=2.0)` with
`squeezed_vacuum_mode(N, z=1.4)`. Same `⟨n̂⟩ ≈ 3.6`, but the
Fock populations are concentrated on the *even* levels only
(`|0⟩`, `|2⟩`, `|4⟩`, …) — squeezing creates photon pairs from
the vacuum. The red sideband's `|n⟩ → |n−1⟩` coupling mixes
only even ↔ odd, so the trajectory looks substantially
different from the coherent-start case at matched `⟨n̂⟩`. A
useful contrast for anyone building intuition.

### Heterogeneous motional prep on a two-mode system

Add a second motional mode to the `IonSystem` (a radial mode,
say). `compose_density` then wants one state per mode, and
different modes can have different prep:

```python
# hilbert_two_mode is not constructed in this tutorial —
# the snippet below illustrates the calling convention only.
try:
    rho_0_two_mode = compose_density(
        hilbert_two_mode,
        spin_states_per_ion=[spin_down()],
        mode_states_by_label={
            "axial": coherent_mode(N, alpha=1.5),
            "radial": squeezed_vacuum_mode(N, z=0.5),
        },
    )
except NameError as exc:
    print("expected (hilbert_two_mode not built in this tutorial):", exc)
```

Useful for state-prep dispatches on multi-mode systems (e.g.
radial-mode squeezing for axial-mode gates — the radial mode
is a spectator during the gate but non-trivial during the
prep).

## Where to next

- [Tutorial 2](02_red_sideband_fock1.md) — the clean-Fock
  sideband baseline this tutorial's collapse scenario contrasts
  against.
- [Tutorial 6](06_fock_truncation.md) — the `fock_tolerance`
  ladder you'll need to tune if `|α|` or `r` climb.
- [Tutorial 8](08_full_lamb_dicke.md) — the
  `full_lamb_dicke=True` flag this tutorial's collapse scenario
  relies on for quantitative accuracy across the Fock band.
- [CONVENTIONS §6, §7](../conventions.md) — the binding spec
  for the squeeze-parameter phase convention and the coherent
  amplitude.
- [`src/iontrap_dynamics/states.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/states.py)
  — reference implementation of the three factories and
  `compose_density`.

---

## Licence

Sail material — adaptive guidance with specific parameter choices,
not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
