# Tutorial 4 — Mølmer–Sørensen Bell gate

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

**Prerequisites.** [Tutorial 2](02_red_sideband_fock1.md) for the
Lamb–Dicke parameter helper and sideband physics vocabulary.
Optionally [Tutorial 3](03_gaussian_pi_pulse.md) for the
list-format dispatch through `sequences.solve` (the detuned MS
Hamiltonian is list-format too). Background on the MS gate at the
level of [`CONVENTIONS.md`](../conventions.md) §9 and §10.

---

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
import numpy as np

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

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

drive = DriveConfig(
    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
    carrier_rabi_frequency_rad_s=2 * np.pi * 0.1e6,  # Ω/2π = 100 kHz
    phase_rad=0.0,
)
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
print(f"η = {eta:.4f}")
print(f"δ/2π = {delta / (2*np.pi*1e3):.2f} kHz")
print(f"t_gate = {t_gate*1e6:.2f} μs")
# η = 0.1843
# δ/2π = 36.85 kHz
# t_gate = 27.14 μs

hamiltonian = detuned_ms_gate_hamiltonian(
    hilbert, drive, "com", ion_indices=(0, 1), detuning_rad_s=delta
)
```

`detuned_ms_gate_hamiltonian` returns the list-format
(`[[H_0, 1.0], [H_1, coeff_fn]]`) bichromatic MS Hamiltonian that
addresses both ions on the named mode. The `ion_indices=(0, 1)`
argument selects which ions the drive couples to — the same
builder scales to longer chains by coupling a subset of ions
while the rest remain spectator to this drive.

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
```

At `t = t_gate` all four final-state targets land within solver
tolerance:

```python
assert abs(result.expectations["p_dd"][-1]   - 0.5) < 1e-5
assert abs(result.expectations["p_uu"][-1]   - 0.5) < 1e-5
assert abs(result.expectations["p_flip"][-1] - 0.0) < 1e-5
assert abs(result.expectations["n_com"][-1]  - 0.0) < 1e-5

# Ion-exchange symmetry — σ_z^(0) and σ_z^(1) trajectories are identical
sz0 = result.expectations["sigma_z_0"]
sz1 = result.expectations["sigma_z_1"]
assert np.max(np.abs(sz0 - sz1)) < 1e-12
```

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
