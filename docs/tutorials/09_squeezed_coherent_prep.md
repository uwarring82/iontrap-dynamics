# Tutorial 9 — Squeezed / coherent state preparation

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

**Prerequisites.** [Tutorial 2](02_red_sideband_fock1.md) — the
RSB dynamics used for the collapse demo. [Tutorial
8](08_full_lamb_dicke.md) — we use `full_lamb_dicke=True`
throughout this tutorial because a coherent state with
`|α|² = 4` populates Fock levels where leading-order rates are
already off by > 10 %.

---

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
its phase rotates the coherent state around phase space:

```python
import numpy as np
import qutip
from iontrap_dynamics.states import coherent_mode

N = 40

psi = coherent_mode(N, alpha=2.0)
assert abs(qutip.expect(qutip.num(N), psi) - 4.0) < 1e-10
# |α=2⟩ carries ⟨n̂⟩ = 4 exactly (within Fock truncation).

psi_rotated = coherent_mode(N, alpha=2.0 * np.exp(1j * np.pi / 2))
# Same ⟨n̂⟩ = 4, but rotated 90° in phase space.
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
psi = squeezed_vacuum_mode(N, z=r)

# Quadrature-variance sanity check: Var(X) = e^(-2r)/2, Var(Y) = e^(+2r)/2.
a = qutip.destroy(N)
X = (a + a.dag()) / np.sqrt(2)
Y = -1j * (a - a.dag()) / np.sqrt(2)
assert abs(qutip.variance(X, psi) - np.exp(-2 * r) / 2) < 1e-3  # 0.068
assert abs(qutip.variance(Y, psi) - np.exp(+2 * r) / 2) < 1e-3  # 3.695
assert abs(qutip.expect(qutip.num(N), psi) - np.sinh(r) ** 2) < 1e-6  # 1.381
```

Even with `⟨n̂⟩ = 1.38`, the state is **pure** (it's a ket). The
non-trivial motional character comes from the quadrature
asymmetry, not from a classical Fock mixture.

### `squeezed_coherent_mode(fock_dim, *, z, alpha)`

```
|α, ξ⟩ = D(α) · S(ξ) |0⟩
```

Squeeze first, then displace. Mean phonon number
`⟨n̂⟩ = |α|² + sinh²(|ξ|)`.

```python
from iontrap_dynamics.states import squeezed_coherent_mode

psi = squeezed_coherent_mode(N, z=1.0, alpha=2.0)
# ⟨n̂⟩ = |α|² + sinh²(|ξ|) = 4 + 1.381 = 5.381
assert abs(qutip.expect(qutip.num(N), psi) - 5.381) < 1e-3
```

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

mode = ModeConfig(
    label="axial",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
)
system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
hilbert = HilbertSpace(system=system, fock_truncations={"axial": N})

rho_0 = compose_density(
    hilbert,
    spin_states_per_ion=[spin_down()],               # ket — auto-promoted
    mode_states_by_label={"axial": coherent_mode(N, alpha=2.0)},
)
# rho_0 is a density matrix on the full space with dims [[2, 40], [2, 40]].
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

# Full Lamb–Dicke because |α|²=4 populates n≈4 where the leading-order
# correction is already > 10% off (Tutorial 8).
hamiltonian = red_sideband_hamiltonian(
    hilbert, drive, "axial", ion_index=0, full_lamb_dicke=True,
)

# Characteristic Rabi period at ⟨n̂⟩ = 4 (leading-order estimate — the
# full rate is ~16% slower, but this is just a time-axis baseline).
flop_period = 2 * np.pi / (abs(eta) * np.sqrt(4.0) * drive.carrier_rabi_frequency_rad_s)
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
```

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
rho_0_two_mode = compose_density(
    hilbert_two_mode,
    spin_states_per_ion=[spin_down()],
    mode_states_by_label={
        "axial": coherent_mode(N, alpha=1.5),
        "radial": squeezed_vacuum_mode(N, z=0.5),
    },
)
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
