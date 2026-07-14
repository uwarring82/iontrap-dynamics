# Tutorial 4 — Mølmer–Sørensen Bell gate

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/04_ms_gate_bell.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** Scale the four-step pattern from Tutorials
[1](01_first_rabi_readout.md)–[3](03_gaussian_pi_pulse.md) up to
a **two-ion** system and exercise the flagship entangling operation
of trapped-ion quantum computing: the Mølmer–Sørensen (MS) gate.
The drive bichromatically straddles the red and blue sidebands of
a shared motional mode; at the carefully-tuned gate-closing
detuning `δ = 2 |Ω η| √K`, the motion traces `K` closed loops in
phase space and the spins pick up a π/4 rotation on
`σ_x⁽⁰⁾ σ_x⁽¹⁾` that maps `|↓↓, 0⟩` onto the Bell state
`(|↓↓⟩ − i |↑↑⟩) / √2 ⊗ |0⟩`.

By the end you will have built this scenario from
`IonSystem` up, derived the Bell-closing detuning and gate time
from **physics inputs** (Ω, η, K) via the
`analytic.ms_gate_closing_detuning` / `ms_gate_closing_time`
helpers — not magic numbers — and verified three independent
observables at the gate time: loop closure `⟨n̂⟩ → 0`, equal
Bell populations `P(|↓↓⟩) = P(|↑↑⟩) = 1/2`, and odd-parity
leakage `P_flip ≡ 0` throughout.

The reference script is
[`tools/run_demo_ms_gate.py`][demo] — same scenario as this
tutorial. Its committed output bundle under
[`benchmarks/data/ms_gate_bell_demo/`][bundle] includes the plot
embedded below.

[demo]: https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/run_demo_ms_gate.py
[bundle]: https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data/ms_gate_bell_demo

**Expected time.** ~15 min reading; ~1 s runtime.

**Level.** `core` — assumes the basics (Tutorials 0–1).

**Prerequisites.** [Tutorial 2](02_red_sideband_fock1.md) for the
Lamb–Dicke parameter helper and sideband physics vocabulary.
Optionally [Tutorial 3](03_gaussian_pi_pulse.md) for the
list-format dispatch through `sequences.solve` (the detuned MS
Hamiltonian is list-format too). Background on the MS gate at the
level of [`CONVENTIONS.md`](../conventions.md) §9 and §10.

---

!!! note "New here? Read this first"

    - Two ions share **one** motional mode (the centre-of-mass, or COM, mode); the gate uses that shared **motion** as a temporary bus to entangle the two **spins** — there is no direct spin–spin term in the Hamiltonian.
    - A **bichromatic** drive (two tones) straddles the mode's red and blue sidebands; the two tones sit symmetrically offset from the first-order sideband by `δ`.
    - During the gate the motion traces a loop in phase space. That loop **must close** at the gate time `t_gate` — `⟨n̂⟩` returns to `0` — or leftover spin–motion entanglement spoils the gate.
    - Nothing here is a magic number: `δ` and `t_gate` are **derived** from the physics inputs `Ω`, `η`, `K` through the `ms_gate_closing_*` helpers.
    - At closure the state is a **Bell state** — equal populations `P(|↓↓⟩) = P(|↑↑⟩) = 1/2` and zero odd-parity leakage.
    - Same four-step skeleton as Tutorials 1–3 (configure → build → solve → read out), now scaled to two ions.

    **In a hurry?** Step 2 derives `(δ, t_gate)` from `(Ω, η)`; Step 3 runs the solve and checks loop closure and the Bell populations — that pair is the core.

**Symbols in this tutorial**

| Symbol | Plain meaning |
| --- | --- |
| `η` | Lamb–Dicke parameter — how strongly the drive couples each spin to the shared COM motion (here `η_COM ≈ 0.1843`). |
| `Ω` | carrier Rabi frequency — the bare on-resonance drive strength (`Ω/2π = 100 kHz`). |
| `δ` | symmetric offset of the two drive tones from the first-order sideband; tuned to close the phase-space loop (scales as `√K`). |
| `t_gate` | gate time — the instant the loop closes and the Bell state is complete (scales as `√K`). |
| `⟨n̂⟩` | mean phonon number of the COM mode; returns to `0` at `t_gate` when the loop closes. |
| `P(↓↓), P(↑↑)` | Bell populations — the weights of the two even-parity spin states; each lands at `1/2` at `t_gate`. |
| `P_flip` | odd-parity population (the two spins pointing oppositely); parity-forbidden here, so pinned at `0`. |

## The scenario

Two identical ²⁵Mg⁺ ions sharing the axial centre-of-mass (COM)
mode at `ω_mode / 2π = 1.5 MHz`. The COM eigenvector places each
ion at `(0, 0, 1/√2)` — both ions move in phase with equal
amplitude, and the per-ion participation factor picks up a `1/√2`
relative to the single-ion limit. A 280 nm bichromatic drive
addresses both ions at carrier Rabi frequency
`Ω / 2π = 100 kHz`; the two tones sit **symmetrically** above and
below the carrier at detuning `±δ`, so the drive couples only to
the first-order sidebands of the COM mode.

The Bell-closing condition fixes `δ` and the gate time `t_gate`
simultaneously. For a single loop `K = 1`:

```
δ      = 2 |Ω η| √K      (loop-closing detuning)
t_gate = π √K / |Ω η|    (equivalently 2π K / δ)
```

With the COM eigenvector's `1/√2` per-ion factor, the single-ion
`η ≈ 0.2606` from Tutorial 2 becomes `η_COM ≈ 0.1843`. Plugging
through: `δ / 2π ≈ 36.85 kHz`, `t_gate ≈ 27.14 μs`. At `t_gate`
the joint state is

```
|ψ(t_gate)⟩ = (|↓↓⟩ − i |↑↑⟩) / √2 ⊗ |0⟩
```

— motion back in vacuum, spins in a maximally-entangled Bell
state.

## Step 1 — Configure the two-ion system

The COM mode's eigenvector is the only new configuration object
compared to Tutorial 2. Each row is one ion's Cartesian
participation in the mode; normalisation
(Σ‖b_i‖² = 1, CONVENTIONS §11) is enforced at construction.

```python
import matplotlib.pyplot as plt
import numpy as np

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

# House colours — match the reference figure palette.
BLUE, RED, GREEN, PURPLE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#444444"

# --- The physics that matters: the COM eigenvector. Both ions move in phase,
#     so each carries a 1/√2 share of the mode — the only new object vs. Tutorial 2. ---
mode = ModeConfig(
    label="com",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array(
        [[0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0]]
    ) / np.sqrt(2.0),  # ‖(0,0,1/√2)‖² · 2 = 1 ✓
)
system = IonSystem(
    species_per_ion=(mg25_plus(), mg25_plus()),
    modes=(mode,),
)

# --- Boilerplate: the base drive (carrier Rabi Ω + 280 nm geometry). The two
#     symmetric ±δ tones that make it an MS drive are added by the builder in Step 2. ---
drive = DriveConfig(
    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
    carrier_rabi_frequency_rad_s=2 * np.pi * 0.1e6,  # Ω/2π = 100 kHz
    phase_rad=0.0,
)

print(f"Step 1 — two-ion system configured")
print(f"  COM mode frequency  ω/2π = {mode.frequency_rad_s / (2*np.pi*1e6):.2f} MHz")
print(f"  Carrier Rabi freq   Ω/2π = {drive.carrier_rabi_frequency_rad_s / (2*np.pi*1e3):.1f} kHz")
print(f"  Drive wavelength         = {2*np.pi / float(np.linalg.norm(drive.k_vector_m_inv)) * 1e9:.0f} nm")
```

!!! note "Why `IonSystem(...)` instead of `IonSystem.homogeneous(...)`"

    The `.homogeneous` classmethod is a convenience for the common
    single-species case and hides the per-ion species tuple. The
    direct `IonSystem(...)` constructor makes the two-ion composition
    explicit — useful here as a template for heterogeneous chains
    (mixed-species cooling, dual-species gates) where
    `.homogeneous` would not apply. For two identical ²⁵Mg⁺ ions
    you could use `.homogeneous(species=mg25_plus(), n_ions=2, …)`
    interchangeably.

## Step 2 — Derive the Bell-closing (δ, t_gate) and build the Hamiltonian

The MS gate parameters are **derived**, not chosen. Feed the
carrier Rabi Ω and the Lamb–Dicke η into
`ms_gate_closing_detuning` and `ms_gate_closing_time`; the loop
count `K` is the only discrete knob, and `K = 1` is the shortest
gate for a given Ω η.

```python
from iontrap_dynamics.analytic import (
    lamb_dicke_parameter,
    ms_gate_closing_detuning,
    ms_gate_closing_time,
)
from iontrap_dynamics.hamiltonians import detuned_ms_gate_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace

hilbert = HilbertSpace(system=system, fock_truncations={"com": 12})

eta = lamb_dicke_parameter(
    k_vec=drive.k_vector_m_inv,
    mode_eigenvector=mode.eigenvector_at_ion(0),
    ion_mass=mg25_plus().mass_kg,
    mode_frequency=mode.frequency_rad_s,
)
# η ≈ 0.1843 — the single-ion value ÷ √2 from the COM sharing.

delta = ms_gate_closing_detuning(
    carrier_rabi_frequency=drive.carrier_rabi_frequency_rad_s,
    lamb_dicke_parameter=eta,
    loops=1,
)
t_gate = ms_gate_closing_time(
    carrier_rabi_frequency=drive.carrier_rabi_frequency_rad_s,
    lamb_dicke_parameter=eta,
    loops=1,
)
print(f"Step 2 — Bell-closing parameters (K = 1)")
print(f"  η_COM   = {eta:.4f}  (single-ion η / √2)")
print(f"  δ/2π    = {delta / (2*np.pi*1e3):.2f} kHz  (loop-closing detuning)")
print(f"  t_gate  = {t_gate*1e6:.2f} μs  (single-loop gate time)")
print(f"  |α|_max = {drive.carrier_rabi_frequency_rad_s * eta / delta:.4f}  (peak phase-space excursion)")
# η = 0.1843
# δ/2π = 36.85 kHz
# t_gate = 27.14 μs

hamiltonian = detuned_ms_gate_hamiltonian(
    hilbert, drive, "com", ion_indices=(0, 1), detuning_rad_s=delta
)

# Illustrate the loop-closing condition: δ = 2|Ω η|√K as a function of loop count K.
k_vals = np.arange(1, 6)
omega_eta = drive.carrier_rabi_frequency_rad_s * eta
delta_k = 2 * omega_eta * np.sqrt(k_vals) / (2 * np.pi * 1e3)   # kHz
tgate_k = np.pi * np.sqrt(k_vals) / omega_eta * 1e6              # µs

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(k_vals, delta_k, color=BLUE, marker="o", markersize=6, label=r"$\delta/2\pi$ (kHz)")
ax.plot(k_vals, tgate_k, color=RED, marker="s", markersize=6, label=r"$t_\mathrm{gate}$ (µs)")
ax.axvline(1, color=GREY, linewidth=0.8, linestyle="--")
ax.set_xlabel("loop count $K$")
ax.set_ylabel("kHz  /  µs")
ax.set_title(r"Loop-closing detuning and gate time vs $K$")
ax.legend(frameon=False)
plt.show()
```

`detuned_ms_gate_hamiltonian` returns the list-format
(`[[H_0, 1.0], [H_1, coeff_fn]]`) bichromatic MS Hamiltonian that
addresses both ions on the named mode. The `ion_indices=(0, 1)`
argument selects which ions the drive couples to — the same
builder scales to longer chains by coupling a subset of ions
while the rest remain spectator to this drive.

!!! warning "Common confusion — the MS gate has no direct spin–spin term"

    The two spins never couple to each other directly. The
    Hamiltonian couples each **spin** to the shared **motion**; the
    entangling `σ_x⁽⁰⁾ σ_x⁽¹⁾` interaction only appears at *second*
    order, after both spins have pushed on — and been pushed back by —
    the same motional mode. Think of the motion as a temporary bus: it
    must be handed back clean at `t_gate` (next step), otherwise the
    correlation you are left with is spin–motion, not spin–spin.

!!! tip "Fock truncation choice for MS gates"

    During the gate the motional coherent state excursion peaks at
    `|α|_max = Ω η / δ = 1 / (2 √K)` — roughly half a phonon
    for `K = 1`. The committed benchmark uses `N_Fock = 12`,
    which is overkill for the expectation value but gives a wide
    safety margin for thermal-start extensions (Tutorial 9,
    planned: squeezed / coherent prep). For a pure `|0⟩` start,
    `N_Fock = 6` would suffice; the Phase 0.F
    Fock-saturation check (CONVENTIONS §13) will flag a
    truncation that is actually too tight.

## Step 3 — Solve with six observables (population + motion + spins)

The Bell state is fully characterised by three population
projectors: `P(|↓↓⟩)`, `P(|↑↑⟩)`, and odd-parity
`P_flip = P(|↓↑⟩) + P(|↑↓⟩)`. The first two should each approach
`0.5` at the gate time; the third should stay at exactly `0`
throughout because the MS Hamiltonian conserves total parity. On
top of these, we watch `⟨n̂⟩` for the phase-space loop closure and
`⟨σ_z⟩` on each ion as an ion-exchange-symmetry cross-check.

!!! warning "Common confusion — the phase-space loop must close"

    `⟨n̂⟩ → 0` at `t_gate` is not a bonus check; it is a
    *requirement*. If the loop does not close, the motion is still
    correlated with the spins, so tracing out the motion leaves the
    two spins in a *mixed* (degraded) state rather than the pure Bell
    state. A non-zero final `⟨n̂⟩` is the first thing to inspect when
    an MS gate under-performs — it usually points to a mis-tuned `δ`
    or a `t_gate` that is not an integer number of loops.

Population projectors aren't in the built-in `observables`
factory — they're custom to the Bell-state scenario. The
`Observable` record is the intended hook for this: wrap a bare
`qutip.Qobj` embedded on the full Hilbert space, give it a
label, and the solver accepts it alongside the named factories:

```python
import qutip

from iontrap_dynamics.observables import Observable, number, spin_z
from iontrap_dynamics.operators import spin_down, spin_up
from iontrap_dynamics.sequences import solve

n_fock = 12
i_mode = qutip.qeye(n_fock)

dd = qutip.ket2dm(qutip.tensor(spin_down(), spin_down()))
du = qutip.ket2dm(qutip.tensor(spin_down(), spin_up()))
ud = qutip.ket2dm(qutip.tensor(spin_up(), spin_down()))
uu = qutip.ket2dm(qutip.tensor(spin_up(), spin_up()))

bell_observables = [
    Observable(label="p_dd", operator=qutip.tensor(dd, i_mode)),
    Observable(label="p_uu", operator=qutip.tensor(uu, i_mode)),
    Observable(label="p_flip", operator=qutip.tensor(du + ud, i_mode)),
]

psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(n_fock, 0))
times = np.linspace(0.0, t_gate, 500)

result = solve(
    hilbert=hilbert,
    hamiltonian=hamiltonian,
    initial_state=psi_0,
    times=times,
    observables=[
        number(hilbert, "com"),
        spin_z(hilbert, 0),
        spin_z(hilbert, 1),
        *bell_observables,
    ],
)

p_dd_final = float(result.expectations["p_dd"][-1])
p_uu_final = float(result.expectations["p_uu"][-1])
p_flip_final = float(result.expectations["p_flip"][-1])
n_final = float(result.expectations["n_com"][-1])
print(f"Step 3 — observables at t_gate = {t_gate*1e6:.2f} µs")
print(f"  P(|↓↓⟩)  = {p_dd_final:.5f}  (target 0.5)")
print(f"  P(|↑↑⟩)  = {p_uu_final:.5f}  (target 0.5)")
print(f"  P_flip   = {p_flip_final:.2e}  (target 0)")
print(f"  ⟨n̂⟩      = {n_final:.2e}  (loop closure → 0)")
```

At `t = t_gate` all four final-state targets land within solver
tolerance:

```python
assert abs(result.expectations["p_dd"][-1]   - 0.5) < 1e-5, "Bell state carries equal weight in |↓↓⟩ and |↑↑⟩, so P(|↓↓⟩) lands at exactly 1/2 at t_gate"
assert abs(result.expectations["p_uu"][-1]   - 0.5) < 1e-5
assert abs(result.expectations["p_flip"][-1] - 0.0) < 1e-5, "total spin parity is conserved, so the odd-parity population P(|↓↑⟩)+P(|↑↓⟩) is forbidden and reads 0 at t_gate"
assert abs(result.expectations["n_com"][-1]  - 0.0) < 1e-5, "the phase-space loop closes at t_gate — ⟨n̂⟩ returns to 0, leaving no spin-motion entanglement to degrade the Bell state"

# Ion-exchange symmetry — σ_z^(0) and σ_z^(1) trajectories are identical
sz0 = result.expectations["sigma_z_0"]
sz1 = result.expectations["sigma_z_1"]
assert np.max(np.abs(sz0 - sz1)) < 1e-12, "ion-exchange symmetry: a symmetric Hamiltonian and symmetric initial state force the two σ_z trajectories to coincide to machine precision"

times_us = times * 1e6  # µs — shared x-axis for all panels

# Panel 1: phonon number — phase-space loop, peaks mid-gate, closes to zero.
n_traj = np.asarray(result.expectations["n_com"], dtype=float)
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(times_us, n_traj, color=PURPLE)
ax.axvline(t_gate * 1e6, color=GREY, linewidth=0.8, linestyle="--", label=r"$t_\mathrm{gate}$")
ax.set_xlabel(r"time $t$ (µs)")
ax.set_ylabel(r"$\langle \hat{n} \rangle$")
ax.set_title("Phase-space loop closure")
ax.legend(frameon=False)
plt.show()

# Panel 2: Bell populations — |↓↓⟩ and |↑↑⟩ land at 0.5, P_flip stays at 0.
p_dd = np.asarray(result.expectations["p_dd"], dtype=float)
p_uu = np.asarray(result.expectations["p_uu"], dtype=float)
p_flip = np.asarray(result.expectations["p_flip"], dtype=float)
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(times_us, p_dd,   color=BLUE,  label=r"$P(|{\downarrow\downarrow}\rangle)$")
ax.plot(times_us, p_uu,   color=RED,   label=r"$P(|{\uparrow\uparrow}\rangle)$")
ax.plot(times_us, p_flip, color=GREEN, linestyle="--", label=r"$P_\mathrm{flip}$ (parity-forbidden)")
ax.axvline(t_gate * 1e6, color=GREY, linewidth=0.8, linestyle="--")
ax.axhline(0.5, color=GREY, linewidth=0.5, linestyle=":")
ax.set_xlabel(r"time $t$ (µs)")
ax.set_ylabel("population")
ax.set_title(r"Bell populations: $|{\downarrow\downarrow}\rangle \to (|{\downarrow\downarrow}\rangle - i|{\uparrow\uparrow}\rangle)/\sqrt{2}$")
ax.legend(frameon=False)
plt.show()

# Panel 3: σ_z trajectories — overlap to machine precision (ion-exchange symmetry).
sz0_arr = np.asarray(sz0, dtype=float)
sz1_arr = np.asarray(sz1, dtype=float)
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(times_us, sz0_arr, color=BLUE, linewidth=2.0, label=r"$\langle\sigma_z^{(0)}\rangle$")
ax.plot(times_us, sz1_arr, color=RED,  linewidth=1.0, linestyle="--",
        label=r"$\langle\sigma_z^{(1)}\rangle$ (hidden by symmetry)")
ax.axvline(t_gate * 1e6, color=GREY, linewidth=0.8, linestyle="--")
ax.set_xlabel(r"time $t$ (µs)")
ax.set_ylabel(r"$\langle\sigma_z\rangle$")
ax.set_title(r"Ion-exchange symmetry: $\sigma_z^{(0)} = \sigma_z^{(1)}$")
ax.legend(frameon=False)
plt.show()

print(f"Step 4 — ion-exchange symmetry: max |σ_z⁽⁰⁾ − σ_z⁽¹⁾| = {float(np.max(np.abs(sz0_arr - sz1_arr))):.2e}")
```

**Takeaway.** Loop closure (`⟨n̂⟩ → 0`) and equal populations `P(|↓↓⟩) = P(|↑↑⟩) = 1/2` are *necessary but not sufficient* for a Bell state — the classical mixture `½|↓↓⟩⟨↓↓| + ½|↑↑⟩⟨↑↑|` gives the *same final-time readouts* in all three panels. It is the *coherence* between `|↓↓⟩` and `|↑↑⟩` — exposed by a parity scan (introduced in Step 4, run in [Tutorial 12](12_bell_entanglement.md)) — that certifies the entanglement.

The ion-exchange-symmetry check is the strongest of the four —
the two `σ_z` trajectories agree to machine precision because the
Hamiltonian is symmetric in `0 ↔ 1` and the initial state is
likewise symmetric. Any accidental asymmetry in the Hamiltonian
builder would show up here as a non-zero residual.

!!! note "Wrapping custom `Qobj`s as `Observable` records"

    The built-in factories (`spin_z`, `number`, `parity`, …) are
    convenience wrappers that embed a subsystem operator on the
    full Hilbert space and attach a canonical label. Anything more
    specific — Bell projectors, two-mode correlators, non-hermitian
    operators for virtual diagnostics — goes through the `Observable`
    constructor directly: you build the full-space `qutip.Qobj`
    yourself and give it a string label for `result.expectations`.
    Tutorial 5 (planned) covers the custom-observable pattern in
    more depth.

## Step 4 — Read out (parity, not single-spin)

The natural two-ion readout for MS-gate tomography is a **parity
scan**: rotate both ions by a variable analysis angle and measure
the `⟨σ_x⁽⁰⁾ σ_x⁽¹⁾⟩` fringe. `iontrap-dynamics` ships a
`ParityScan` protocol that wraps this pattern; its detailed use
lands in Tutorial 12 (planned). The single-ion `SpinReadout` from
Tutorial 1 still works on one ion at a time if you just want a
population-level sanity check — useful for smoke-testing the
pipeline before the full parity analysis.

## Putting it together

The committed reference run of this exact scenario (`Ω / 2π = 100
kHz`, `η ≈ 0.1843`, `δ / 2π ≈ 36.85 kHz`, `t_gate ≈ 27.14 μs`,
`N_Fock = 12`, 500 time points) produces:

![MS Bell gate](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/ms_gate_bell_demo/plot.png)

Top panel: `⟨n̂⟩` ramps up to ~0.25 mid-gate (the peak of the
phase-space loop), then closes back to zero at `t_gate`.
Middle panel: the Bell populations. `P(|↓↓⟩)` starts at 1 (the
`|↓↓, 0⟩` initial state) and `P(|↑↑⟩)` starts at 0; they cross at
`t_gate / 2` and both land at exactly `0.5` at the gate time. The
odd-parity population `P_flip` stays pinned at 0 across the whole
gate — the clearest visual signature that the MS Hamiltonian is
doing what it should.
Bottom panel: `⟨σ_z⁽⁰⁾⟩` and `⟨σ_z⁽¹⁾⟩` overlap to machine
precision (the black dashed line hides the orange solid one
perfectly).

Wall-clock for the full 500-step, two-ion, `N_Fock = 12` solve on
a 2023 M2 MacBook Air: ~10 ms.

## Physics you can probe next

### Higher loop counts

Set `loops=2` (or `loops=3`) in both analytic helpers.
`δ = 2 |Ω η| √K` scales as `√K` — larger detuning, looser drive,
less sensitivity to off-resonant carrier excitation. The trade-off
is a longer gate: `t_gate ∝ √K`. The middle panel of the plot
gains `K` mid-gate excursions before the final Bell-state landing.

### Thermal-state start

Replace `qutip.basis(n_fock, 0)` with a density matrix built from
a thermal distribution (`states.thermal_mode` + `compose_density`,
or compose one inline via `qutip.thermal_dm(n_fock, n_bar)`). The
gate fidelity degrades as `1 − (π²/8) (Ω η / δ)² n_bar` to
leading order — visible as a non-zero residual `P_flip` at the
gate time, and a non-zero final `⟨n̂⟩`. This is the standard
motional-heating sensitivity study.

### Detuning miscalibration sweep

Run `solve_ensemble` over a span of `detuning_rad_s` values
around the nominal `δ` and plot `P(|↓↓⟩) + P(|↑↑⟩)` at `t_gate` —
you'll recover the textbook `sinc²`-like gate-error curve. This
is a natural warm-up for Tutorial 11 (planned, jitter ensembles):
same ensemble machinery, but the detuning is drawn from a noise
distribution rather than swept deterministically.

## Where to next

- [Tutorial 2](02_red_sideband_fock1.md) — the single-ion
  sideband scenario whose `lamb_dicke_parameter` helper is reused
  here.
- [Tutorial 3](03_gaussian_pi_pulse.md) — the list-format
  dispatch path that the detuned MS Hamiltonian also runs on.
- [Phase 1 Architecture](../phase-1-architecture.md) — reference
  for `detuned_ms_gate_hamiltonian`, the `ms_gate_closing_*`
  analytic helpers, and the `Observable` record.
- [`tools/run_demo_ms_gate.py`][demo] — the runnable script that
  produced the plot embedded above; diff it against this tutorial
  for the exact code plus the canonical-cache artefact layout.

---

## Licence

Sail material — adaptive guidance with specific parameter choices,
not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
