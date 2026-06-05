# Tutorial 8 — Full Lamb–Dicke for hot-ion regimes

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/08_full_lamb_dicke.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

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
import matplotlib.pyplot as plt
import numpy as np
import qutip

from iontrap_dynamics.analytic import (
    debye_waller_factor,
    lamb_dicke_confinement,
    lamb_dicke_parameter,
    lamb_dicke_regime,
    red_sideband_rabi_frequency,
    red_sideband_rabi_frequency_full_ld,
)
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import red_sideband_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

# House colours — used throughout this tutorial.
BLUE, RED, GREEN, PURPLE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#444444"

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

eta = lamb_dicke_parameter(
    k_vec=drive.k_vector_m_inv,
    mode_eigenvector=mode.eigenvector_at_ion(0),
    ion_mass=mg25_plus().mass_kg,
    mode_frequency=mode.frequency_rad_s,
)
omega = drive.carrier_rabi_frequency_rad_s

dw = debye_waller_factor(lamb_dicke_parameter=eta, mean_phonon_number=0)
print(f"Step 1 — η = {eta:.4f},  η² = {eta**2:.4f},  Debye–Waller e^(−η²/2) = {dw:.4f}")

# Same signature, one flag.
hamiltonian_leading = red_sideband_hamiltonian(
    hilbert, drive, "axial", ion_index=0,
    full_lamb_dicke=False,       # default — can be omitted
)
hamiltonian_full = red_sideband_hamiltonian(
    hilbert, drive, "axial", ion_index=0,
    full_lamb_dicke=True,
)

# Rate comparison: leading-order vs full Lamb–Dicke for n = 1, 5, 10.
print(f"\n{'n':>3}  {'leading (kHz)':>15}  {'full-LD (kHz)':>14}  {'shortfall':>10}  {'η²·n':>6}  {'regime'}")
for n in (1, 5, 10):
    r_lead = red_sideband_rabi_frequency(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n,
    )
    r_full = red_sideband_rabi_frequency_full_ld(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n,
    )
    shortfall = (r_lead - r_full) / r_lead * 100
    conf = lamb_dicke_confinement(lamb_dicke_parameter=eta, mean_phonon_number=n)
    regime = lamb_dicke_regime(lamb_dicke_parameter=eta, mean_phonon_number=n)
    print(f"{n:>3}  {r_lead/(2*np.pi)*1e-3:>15.3f}  {r_full/(2*np.pi)*1e-3:>14.3f}  {shortfall:>9.1f}%  {eta**2*n:>6.3f}  {str(regime)}")

# Sweep rate vs Fock to show the divergence visually.
n_grid = np.arange(1, 21)
leading_n = np.array([
    red_sideband_rabi_frequency(carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=int(n))
    for n in n_grid
]) / (2 * np.pi * 1e3)
full_n = np.array([
    red_sideband_rabi_frequency_full_ld(carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=int(n))
    for n in n_grid
]) / (2 * np.pi * 1e3)

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(n_grid, leading_n, color=GREY, marker="o", markersize=4, label="leading-order  |η|√n·Ω")
ax.plot(n_grid, full_n, color=BLUE, marker="s", markersize=4, label="full Lamb–Dicke (Laguerre)")
ax.set_xlabel("initial Fock level $n$")
ax.set_ylabel(r"red-sideband rate $\Omega_{n,n-1}$ (kHz)")
ax.set_title(rf"RSB rate vs $n$ at $\eta = {eta:.3f}$")
ax.legend(frameon=False)
plt.show()
```

## Step 2 — Run the three scenarios, overlay the trajectories

Three starting Fock levels, same drive, same total duration
(always two flop periods at the **leading-order** rate — that's
the natural baseline):

```python
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
    sz_lead = np.asarray(r_lead.expectations["sigma_z_0"], dtype=float)
    sz_full = np.asarray(r_full.expectations["sigma_z_0"], dtype=float)
    trajectories[n] = (times * 1e6, sz_lead, sz_full)

    # Rate shortfall and final-state gap for this Fock level.
    r_lead_rate = abs(eta) * np.sqrt(n) * omega
    r_full_rate = red_sideband_rabi_frequency_full_ld(
        carrier_rabi_frequency=omega, lamb_dicke_parameter=eta, n_initial=n,
    )
    shortfall_pct = (r_lead_rate - r_full_rate) / r_lead_rate * 100
    final_gap = abs(float(sz_full[-1]) - float(sz_lead[-1]))
    print(
        f"n = {n:>2}: leading rate = {r_lead_rate/(2*np.pi)*1e-3:.3f} kHz, "
        f"full-LD rate = {r_full_rate/(2*np.pi)*1e-3:.3f} kHz, "
        f"shortfall = {shortfall_pct:.1f}%,  final ⟨σ_z⟩ gap = {final_gap:.2f}"
    )

# Three-panel overlay: one row per Fock level.
fig, axes = plt.subplots(3, 1, figsize=(5.0, 7.0), sharex=False)
for ax, n in zip(axes, (1, 5, 10)):
    t_us, sz_lead, sz_full = trajectories[n]
    ax.plot(t_us, sz_lead, color=GREY, linewidth=1.0, label="leading-order")
    ax.plot(t_us, sz_full, color=RED, linewidth=1.0, label="full Lamb–Dicke")
    ax.set_ylabel(r"$\langle\sigma_z\rangle$")
    ax.set_title(rf"$n = {n}$,  $\eta^2 n = {eta**2*n:.3f}$")
    ax.legend(frameon=False, fontsize=8)
ax.set_xlabel("time (µs)")
fig.tight_layout()
plt.show()
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
builder in the library, with identical semantics. The plot
below sweeps the confinement parameter `η²(2n+1)` for a fixed
`n = 5` and shows how the relative shortfall grows from the
deep into the beyond regime:

```python
from iontrap_dynamics.hamiltonians import (
    blue_sideband_hamiltonian,
    detuned_red_sideband_hamiltonian,
    detuned_blue_sideband_hamiltonian,
    two_ion_red_sideband_hamiltonian,
    two_ion_blue_sideband_hamiltonian,
)

# The on-resonance red and blue sideband builders accept full_lamb_dicke=True.
H_bsb_full = blue_sideband_hamiltonian(
    hilbert, drive, "axial", ion_index=0, full_lamb_dicke=True,
)
print(f"Step 4 — blue_sideband_hamiltonian (full-LD) dims: {H_bsb_full.dims}")

# Sweep η at n = 5 to show the confinement → shortfall trend.
# Use a synthetic carrier Ω so the rate ratio is normalisation-independent.
eta_sweep = np.array([0.02, 0.05, 0.10, 0.18, 0.26, 0.40, 0.60, 0.80])
n_sweep = 5
omega_ref = 2 * np.pi * 1.0e6  # 1 MHz reference carrier
confinement_sweep = np.array(
    [lamb_dicke_confinement(lamb_dicke_parameter=e, mean_phonon_number=n_sweep) for e in eta_sweep]
)
leading_sweep = np.array([
    red_sideband_rabi_frequency(carrier_rabi_frequency=omega_ref, lamb_dicke_parameter=e, n_initial=n_sweep)
    for e in eta_sweep
])
full_sweep = np.array([
    red_sideband_rabi_frequency_full_ld(carrier_rabi_frequency=omega_ref, lamb_dicke_parameter=e, n_initial=n_sweep)
    for e in eta_sweep
])
rel_shortfall = (leading_sweep - full_sweep) / leading_sweep

print(f"Step 4 — rate shortfall at n = {n_sweep} vs η:")
print(f"  {'η':>6}  {'η²(2n+1)':>10}  {'regime':>12}  {'shortfall':>10}")
for e, c, r_short in zip(eta_sweep, confinement_sweep, rel_shortfall):
    regime = lamb_dicke_regime(lamb_dicke_parameter=e, mean_phonon_number=n_sweep)
    print(f"  {e:>6.2f}  {c:>10.3f}  {str(regime):>12}  {r_short*100:>9.1f}%")

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.axvspan(1e-4, 0.1, color=GREEN, alpha=0.08, label="deep")
ax.axvspan(0.1, 1.0, color=BLUE, alpha=0.08, label="intermediate")
ax.axvspan(1.0, 1e2, color=RED, alpha=0.08, label="beyond")
ax.plot(confinement_sweep, rel_shortfall, color=GREY, marker="o", markersize=5)
ax.set_xscale("log")
ax.set_xlabel(r"confinement $\eta^2(2n+1)$")
ax.set_ylabel("leading-order vs full-LD shortfall")
ax.set_title(rf"Regime map at $n = {n_sweep}$")
ax.legend(frameon=False, fontsize=8)
plt.show()
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
