# Tutorial 8 — Full Lamb–Dicke for hot-ion regimes

**Goal.** Every sideband builder in `iontrap-dynamics`
(`red_sideband_hamiltonian`, `blue_sideband_hamiltonian`, their
detuned siblings, the two-ion variants) accepts a
`full_lamb_dicke: bool = False` keyword. Default off, the builder
uses the leading-order `η·a` coupling that Tutorials
[2](02_red_sideband_fock1.md) and [4](04_ms_gate_bell.md) relied
on. Flip it on, and the same builder constructs the **full
Wineland–Itano** coupling — the Laguerre-polynomial Rabi-rate
structure that emerges from the exact
`e^{iη(a+a†)}` operator without any truncation in η. This
tutorial walks through **when the leading-order approximation is
enough**, **when it isn't**, and **what switching the flag costs
you** (almost nothing at solve time).

By the end you will have:

1. Seen the Wineland–Itano Rabi rate for `|n⟩ → |n−1⟩` and the
   `η² · n ≳ 0.1` rule-of-thumb that separates the two
   regimes.
2. Run three red-sideband flops at increasing Fock levels
   (`n = 1, 5, 10`) with the same Hamiltonian builder, seeing
   leading-order and full Lamb–Dicke answers diverge from 3 %
   to 30 % rate shortfall.
3. Understood where the extra operator structure goes (a
   one-time mode-level matrix exponentiation at build time;
   solve cost is unchanged).
4. Learned why the flag is an operational choice, not a
   physics-truth toggle — leading-order is correct *physics in
   its regime of validity*, not a bug waiting to be switched off.

**Expected time.** ~12 min reading; ~3 s runtime.

**Prerequisites.** [Tutorial 2](02_red_sideband_fock1.md) — the
RSB scenario used throughout this tutorial. The Lamb–Dicke
parameter definition at
[`CONVENTIONS.md`](../conventions.md) §10 is useful background
for why the default is leading-order.

---

## The Wineland–Itano closed form, and when you need it

The exact `|n⟩ → |n − 1⟩` red-sideband Rabi rate is

```
Ω_{n,n−1}^full = Ω · |η| · e^(−η²/2) · √((n−1)! / n!) · L_{n−1}^(1)(η²)
```

where `L_{n−1}^(1)` is the generalised Laguerre polynomial of
degree `n−1`, order 1. Expanding to lowest order in η recovers
the library's default:

```
Ω_{n,n−1}^lead = Ω · |η| · √n
```

Two things matter here:

1. **Debye–Waller amplitude** — the `e^(−η²/2)` prefactor
   uniformly reduces every sideband coupling. For
   `η = 0.26`, it's `0.9666` (3.4 % shortfall), independent of
   `n`. Small, but not zero.
2. **Laguerre-polynomial structure** — `L_{n−1}^(1)(η²)` is
   `≈ n` at leading order, which combined with
   `√((n−1)! / n!) = 1/√n` reproduces the `√n` factor. But at
   higher `n` the polynomial has non-trivial structure
   (zeros, sign changes, oscillations) that the leading-order
   expression cannot capture — the point where the full form
   earns its keep.

The rule of thumb: **leading-order is safe while `η² · n ≲ 0.1`
across every `n` the trajectory populates.** Above that
threshold the Laguerre corrections are measurable; well above
(say `η² · n̄ ≳ 0.3`) the two rates disagree by tens of percent.

## Quantitative at `η = 0.26` (Tutorial 2's scenario)

```
n    leading-order rate (·Ω)    full rate (·Ω)    shortfall
------------------------------------------------------------
 1            0.2606               0.2519           +3.3 %
 5            0.5827               0.4893          +16.0 %
10            0.8240               0.5743          +30.3 %
```

Tutorial 2 noted the single-percent agreement at `n = 1` —
which is true *for the Rabi rate at `n = 1`*, but ignores that
phase drift accumulates. Over two flop periods at the
leading-order rate, the two σ_z trajectories at `n = 1` peel
apart by ~9 % in the final-state projection; at `n = 5` they
completely desynchronise. The regime of validity is *narrower*
than "η ≲ 0.3", and
**the flag starts mattering at Fock levels the library happily
hosts.**

## Step 1 — One builder, two flavours

The flag flips under exactly the same builder signature — every
downstream layer (`solve`, `Observable`, readout) is unchanged:

```python
import numpy as np
import qutip

from iontrap_dynamics.analytic import lamb_dicke_parameter
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import red_sideband_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

N_FOCK = 30
mode = ModeConfig(
    label="axial",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
)
system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
hilbert = HilbertSpace(system=system, fock_truncations={"axial": N_FOCK})

drive = DriveConfig(
    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
    carrier_rabi_frequency_rad_s=2 * np.pi * 0.1e6,
    phase_rad=0.0,
)

# Same signature, one flag.
hamiltonian_leading = red_sideband_hamiltonian(
    hilbert, drive, "axial", ion_index=0,
    full_lamb_dicke=False,       # default — can be omitted
)
hamiltonian_full = red_sideband_hamiltonian(
    hilbert, drive, "axial", ion_index=0,
    full_lamb_dicke=True,
)
```

## Step 2 — Run the three scenarios, overlay the trajectories

Three starting Fock levels, same drive, same total duration
(always two flop periods at the **leading-order** rate — that's
the natural baseline):

```python
eta = lamb_dicke_parameter(
    k_vec=drive.k_vector_m_inv,
    mode_eigenvector=mode.eigenvector_at_ion(0),
    ion_mass=mg25_plus().mass_kg,
    mode_frequency=mode.frequency_rad_s,
)
omega = drive.carrier_rabi_frequency_rad_s

trajectories = {}
for n in (1, 5, 10):
    leading_rate = abs(eta) * np.sqrt(n) * omega
    flop_period_leading = 2 * np.pi / leading_rate
    times = np.linspace(0.0, 2 * flop_period_leading, 400)

    psi_0 = qutip.tensor(spin_down(), qutip.basis(N_FOCK, n))

    r_lead = solve(
        hilbert=hilbert, hamiltonian=hamiltonian_leading,
        initial_state=psi_0, times=times,
        observables=[spin_z(hilbert, 0)],
    )
    r_full = solve(
        hilbert=hilbert, hamiltonian=hamiltonian_full,
        initial_state=psi_0, times=times,
        observables=[spin_z(hilbert, 0)],
    )
    trajectories[n] = (
        times * 1e6,
        r_lead.expectations["sigma_z_0"],
        r_full.expectations["sigma_z_0"],
    )
```

### What the numbers show

`n = 1` (the Tutorial 2 scenario): `⟨σ_z⟩` completes two
flops in both builders, landing at exactly `−1` for leading-order
(by construction — two leading-order periods) and at
`−0.91` for full Lamb–Dicke. The peak trajectory deviation is
~0.37 — the full-LD curve is running at 96.7 % of the leading
rate, and that 3 % phase lag accumulates over two cycles.

`n = 5`: leading-order hits `−1.00`; full-LD is at `+0.43`.
The two curves are no longer the same flop under a phase lag —
they're qualitatively different. The full-LD rate has dropped
to 84 % of leading-order and the accumulated phase lag is now
close to half a cycle.

`n = 10`: leading-order still lands at `−1.00`; full-LD is at
`+0.79`. The full-LD curve has only just started its second
flop — the rate has fallen to 70 % of leading-order.

!!! tip "A reader-run sanity check"

    Just compute `abs(eta) * np.sqrt(n) * np.exp(-eta**2 / 2)`
    against `abs(eta) * np.sqrt(n)` for your specific
    `(η, n)`. The ratio between the two — the
    Debye–Waller factor — is the *floor* on how bad the
    disagreement can be; the Laguerre structure adds on top.
    For any `η² · n > 0.1`, the scenario is in the regime
    where the flag matters.

## Step 3 — What does `full_lamb_dicke=True` actually cost?

The flag switches the mode-subsystem operator from
`η · a` (cheap — a constant times a tridiagonal annihilation
operator) to `M̂_- = P_{Δn = −1}(e^{iη(a+a†)})` (a single
matrix exponentiation on the truncated mode, then projected to
the `Δn = −1` band). The cost is:

- **Build time.** One matrix exponentiation on the mode
  subsystem — negligible for modest `N_Fock`, and done once
  before `solve` starts.
- **Solve time.** Unchanged. Both operators embed into the full
  Hilbert space as a sparse operator with the same non-zero
  pattern (Δn = −1 only); the solver's ODE step cost is
  identical.

In other words: **there is no solve-time reason to leave the
flag off.** The only reason the default is `False` is semantic —
the library promises you the specific "leading-order
Lamb–Dicke" Hamiltonian when you ask for
`red_sideband_hamiltonian` by default, and the full form is
opt-in so a caller can't accidentally get one when they meant
the other.

## Step 4 — The flag applies uniformly

The same `full_lamb_dicke` keyword exists on every sideband
builder in the library, with identical semantics:

```python
from iontrap_dynamics.hamiltonians import (
    blue_sideband_hamiltonian,
    detuned_red_sideband_hamiltonian,
    detuned_blue_sideband_hamiltonian,
    two_ion_red_sideband_hamiltonian,
    two_ion_blue_sideband_hamiltonian,
)

# Every one accepts full_lamb_dicke=True; meaning is the same.
H_bsb_full = blue_sideband_hamiltonian(
    hilbert, drive, "axial", ion_index=0, full_lamb_dicke=True,
)
H_rsb_detuned_full = detuned_red_sideband_hamiltonian(
    hilbert, drive, "axial", ion_index=0,
    detuning_rad_s=2 * np.pi * 10e3,
    full_lamb_dicke=True,
)
```

The Mølmer–Sørensen gate Hamiltonian (from
[Tutorial 4](04_ms_gate_bell.md)) is a composition of red and
blue sideband builders internally, so the flag carries through
the same way: `detuned_ms_gate_hamiltonian(...,
full_lamb_dicke=True)` switches both tones simultaneously.

## When to flip the flag

Not every scenario needs the full form. A practical decision
tree:

- **Pure-Fock initial state, single-phonon-manifold dynamics**
  (Tutorial 2's `|↓, 1⟩ → |↑, 0⟩`). Only the `n = 1` rate
  matters; the 3 % shortfall is cosmetically visible on the
  final `⟨σ_z⟩` but doesn't change the qualitative physics.
  Leading-order is fine for scoping work.
- **Thermal start with `n̄ ≥ 3`.** `η² · n̄ ≳ 0.2` typically —
  over the threshold. Every flop rate in the mixture is
  **different**, and the Doppler-cooling sensitivity studies
  that are the point of running thermally start to mis-predict
  if you're on leading-order.
- **MS gate at `η > 0.1`.** The coherent-state phase-space
  excursion during the gate populates `n` up to a handful;
  Laguerre corrections on the per-level rates shift the
  gate-closing time. If you're tuning `t_gate` to match an
  experiment, flip the flag.
- **Sideband cooling from hot initial states.** The sideband
  spectrum (how fast population flows from `|↑, n⟩` to
  `|↓, n−1⟩` at each `n`) directly controls the cooling rate
  at each step of the cascade. Always full Lamb–Dicke.
- **Publication-grade results in any of the above.** Flip the
  flag even if `η² · n̄` is in the grey zone. The cost is zero
  and the result carries the stronger claim.

## Where to next

- [Tutorial 2](02_red_sideband_fock1.md) — the leading-order
  sideband scenario this tutorial revisits.
- [Tutorial 4](04_ms_gate_bell.md) — the MS gate that composes
  both sideband builders; the flag carries through.
- [CONVENTIONS §10](../conventions.md) — the Lamb–Dicke
  parameter definition and the leading-order / full-form
  distinction, made precise.
- [`src/iontrap_dynamics/hamiltonians.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/hamiltonians.py)
  — the `_full_ld_lowering_single_mode` engine (Wineland–Itano
  matrix-exponential construction) plus every sideband builder's
  full-LD dispatch path.

---

## Licence

Sail material — adaptive guidance with specific parameter choices,
not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
