# Tutorial 5 — Custom observables

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/05_custom_observables.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** Move past the built-in `spin_x` / `spin_y` / `spin_z` /
`parity` / `number` factories and learn the hook that lets you
stream **any** operator through `sequences.solve` as a named
expectation: the `Observable` record. By the end you will have
written four independent custom observables for the
Mølmer–Sørensen Bell-gate scenario from
[Tutorial 4](04_ms_gate_bell.md):

1. A **Bell-state population projector** `|Φ⁻⟩⟨Φ⁻|` — the
   actual target state.
2. A **joint spin correlator** `⟨σ_x⁽⁰⁾ σ_x⁽¹⁾⟩` — the
   MS-gate fringe observable, pre-parity-scan.
3. A **Fock-1 projector** `|1⟩⟨1|` on the mode — tells you
   how much the motion leaks out of the vacuum manifold
   mid-gate.
4. An **off-diagonal coherence** `|↓↓⟩⟨↑↑|` — a
   non-Hermitian "virtual diagnostic" that exposes the phase
   of the Bell superposition.

These four cover the three cases you're likely to hit: Hermitian
multi-subsystem projectors, Hermitian two-ion correlators, and
non-Hermitian virtuals. Each slots into the same list-argument of
`sequences.solve` as the built-ins, with no special handling.

**Expected time.** ~12 min reading; ~1 s runtime.

**Level.** `core` — assumes the basics (Tutorials 0–1).

**Prerequisites.** [Tutorial 4](04_ms_gate_bell.md) — we reuse
its MS-gate scenario verbatim (both ions, COM mode, derived δ
and `t_gate`). No new physics; this tutorial is about the
observable-construction hook.

---

## The one-line rule

An `Observable` is a two-field frozen dataclass:

```python
# Illustrative pseudo-code — the real class lives in iontrap_dynamics.observables
from dataclasses import dataclass
import qutip as _qutip


@dataclass(frozen=True, slots=True, kw_only=True)
class Observable:
    label: str             # appears in result.expectations[...]
    operator: _qutip.Qobj  # embedded on the FULL Hilbert space
```

Everything else follows. As long as `operator.dims` matches
`hilbert.qutip_dims()`, the solver computes `⟨ψ(t)| O |ψ(t)⟩` (or
`tr(ρ(t) O)` for density matrices) at every requested time — no
matter whether `O` is Hermitian, positive semi-definite, or a
random matrix of the right shape.

The subtlety is **embedding**: the built-in factories look easy
because they hide the tensor-product boilerplate
(`spin_op_for_ion`, `mode_op_for`). When you build a custom
observable you either call those helpers yourself, or construct
the operator directly via `qutip.tensor(...)` in the §2 subsystem
order (spins before modes).

## The shared scenario

This is the same two-ion MS-gate setup from Tutorial 4, compressed
into one block so the custom observables are the interesting part.
`t_gate` and `δ` are derived from the physics inputs exactly as
before:

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip

from iontrap_dynamics.analytic import (
    lamb_dicke_parameter,
    ms_gate_closing_detuning,
    ms_gate_closing_time,
)
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import detuned_ms_gate_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.operators import (
    sigma_x_ion, spin_down, spin_up,
)
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

# House colours — shared across all figures in this tutorial.
BLUE, RED, GREEN, PURPLE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#444444"

N_FOCK = 12
mode = ModeConfig(
    label="com",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    ) / np.sqrt(2.0),
)
system = IonSystem(species_per_ion=(mg25_plus(), mg25_plus()), modes=(mode,))
hilbert = HilbertSpace(system=system, fock_truncations={"com": N_FOCK})

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
delta = ms_gate_closing_detuning(
    carrier_rabi_frequency=drive.carrier_rabi_frequency_rad_s,
    lamb_dicke_parameter=eta, loops=1,
)
t_gate = ms_gate_closing_time(
    carrier_rabi_frequency=drive.carrier_rabi_frequency_rad_s,
    lamb_dicke_parameter=eta, loops=1,
)
hamiltonian = detuned_ms_gate_hamiltonian(
    hilbert, drive, "com", ion_indices=(0, 1), detuning_rad_s=delta,
)
psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(N_FOCK, 0))
times = np.linspace(0.0, t_gate, 500)

print(f"Scenario — η = {eta:.4f},  δ/(2π) = {delta / (2 * np.pi) * 1e-3:.3f} kHz,  "
      f"t_gate = {t_gate * 1e6:.2f} µs")
```

## Pattern A — Hermitian multi-subsystem projector (`|Φ⁻⟩⟨Φ⁻|`)

Tutorial 4's three Bell-population projectors (`|↓↓⟩⟨↓↓|`,
`|↑↑⟩⟨↑↑|`, odd-parity) are all **diagonal** in the
computational basis. The target of the MS gate, though, is a
**coherent superposition** `|Φ⁻⟩ = (|↓↓⟩ − i|↑↑⟩) / √2`. The
projector `|Φ⁻⟩⟨Φ⁻|` is off-diagonal — it doesn't reduce to a
sum of computational-basis projectors, and its expectation is
the **Bell-state fidelity**, the quantity you actually want as the
gate figure of merit.

Build the ket in the two-ion spin subspace, embed by tensoring on
the mode identity, and wrap in `Observable`:

```python
from iontrap_dynamics.observables import Observable

# |Φ-⟩ = (|↓↓⟩ − i |↑↑⟩) / √2, in the two-ion spin subspace
phi_minus = (
    qutip.tensor(spin_down(), spin_down())
    - 1j * qutip.tensor(spin_up(), spin_up())
).unit()
phi_minus_projector = qutip.ket2dm(phi_minus)

# Embed on full Hilbert space: spin projector ⊗ I_mode
fidelity_op = qutip.tensor(phi_minus_projector, qutip.qeye(N_FOCK))
bell_fidelity = Observable(label="bell_fidelity", operator=fidelity_op)
```

!!! note "Why tensor order matters — and how to get it right"

    `HilbertSpace` enforces the CONVENTIONS.md §2 tensor order
    **spins first, then modes** — for this two-ion + one-mode
    system that's `ion_0 ⊗ ion_1 ⊗ com`. The operator you hand
    `Observable` must match `hilbert.qutip_dims()` exactly: here
    `[[2, 2, 12], [2, 2, 12]]`. `qutip.tensor(...)` builds from
    left to right, so writing `qutip.tensor(phi_minus_projector,
    qutip.qeye(N_FOCK))` gets the order right because
    `phi_minus_projector` already covers both spins in the right
    order. If you ever need a single-ion Hermitian op on ion ``i``,
    use `hilbert.spin_op_for_ion(...)` — it handles the embedding
    so you don't have to count qeyes.

## Pattern B — Two-ion correlator (`⟨σ_x⁽⁰⁾ σ_x⁽¹⁾⟩`) via `HilbertSpace`

The joint correlator is the time-domain signal that a physical
parity scan samples once per analysis phase. For the `|Φ⁻⟩` Bell
state produced by this gate, `⟨σ_x σ_x⟩` returns to `0` at
`t_gate` — the two equal-weight but phase-shifted components
interfere destructively in the σ_x–σ_x basis.

`HilbertSpace.spin_op_for_ion` is the idiomatic way to build
per-ion operators — no counting qeyes, no risk of transposing
the two ions:

```python
sigma_x_0 = hilbert.spin_op_for_ion(sigma_x_ion(), 0)
sigma_x_1 = hilbert.spin_op_for_ion(sigma_x_ion(), 1)
sigma_xx_op = sigma_x_0 * sigma_x_1   # operator product, commutes
sigma_xx = Observable(label="sigma_xx", operator=sigma_xx_op)
```

!!! tip "Parity factory for the σ_z–σ_z case"

    A corresponding `⟨σ_z σ_z⟩` observable is one line with the
    built-in factory: `parity(hilbert, ion_indices=(0, 1))`. The
    custom construction above is the template for observables
    that the built-in factories don't cover —
    `⟨σ_x σ_x⟩` / `⟨σ_y σ_y⟩` / mixed-axis correlators /
    three-ion parities of mixed spin operators, etc.

## Pattern C — Mode Fock-state projector (`|1⟩⟨1|_mode`)

Tutorial 4 monitored `⟨n̂⟩` on the mode — the *mean* phonon
number. That's insufficient if you want to know *where* the motion
goes when it's not in vacuum. A Fock-state projector
`|1⟩⟨1|` gives you `P(n = 1)` directly, and the same pattern
extends to `|2⟩⟨2|`, `|3⟩⟨3|`, ….

`qutip.fock_dm(N_FOCK, 1)` builds the mode-subsystem operator;
embed via `HilbertSpace.mode_op_for`:

```python
fock1_mode = qutip.fock_dm(N_FOCK, 1)  # mode-subsystem |1⟩⟨1|
p_fock1 = Observable(
    label="p_fock1_com",
    operator=hilbert.mode_op_for(fock1_mode, "com"),
)
```

During the MS gate, `P(n = 1)` rises from 0 at `t = 0`, peaks
mid-gate (when the coherent-state amplitude `α(t)` has the most
overlap with `|1⟩`), and returns to 0 at `t_gate` — exactly in
step with the loop-closure condition. Compare to `⟨n̂⟩`: both go
to zero at `t_gate`, but their mid-gate shapes differ because
`⟨n̂⟩ = Σ n · P(n)` is weighted by `n` while `P(1)` is just the
first term.

## Pattern D — Non-Hermitian virtual (`|↓↓⟩⟨↑↑|`)

Everything so far has been Hermitian — physical observables
correspond to self-adjoint operators. But `Observable` doesn't
*require* hermiticity. `sequences.solve` happily evaluates
`⟨ψ(t)| O |ψ(t)⟩` for any operator with the right dims, and the
result is complex-valued. This is useful as a **virtual
diagnostic** — something you can't measure directly in the lab
but that exposes information a physical measurement would have
to reconstruct from multiple Hermitian observables.

The off-diagonal coherence `|↓↓⟩⟨↑↑|` is a classic:

```python
down_down = qutip.tensor(spin_down(), spin_down())
up_up = qutip.tensor(spin_up(), spin_up())
coherence_spin = down_down * up_up.dag()
coherence_op = qutip.tensor(coherence_spin, qutip.qeye(N_FOCK))
coherence = Observable(label="coherence_dd_uu", operator=coherence_op)
```

At `t = 0`: expectation is 0 (state is `|↓↓, 0⟩`, no `|↑↑⟩`
component). At `t = t_gate`: expectation is `−i / 2` — the gate
output `(0.5−0.5i)|↓↓⟩ + (−0.5−0.5i)|↑↑⟩` gives
`⟨↓↓|ρ|↑↑⟩ = (0.5+0.5i)(−0.5−0.5i) = −i/2`. The
**imaginary-part trajectory** is the coherence-oscillation fringe
you'd reconstruct from a Ramsey-style analysis; getting it
straight from one observable saves a lot of reconstruction
bookkeeping during development.

!!! warning "Complex expectations break some downstream tooling"

    Most of the library stores expectations as `np.ndarray` of
    real `float64`. `sequences.solve` promotes to complex
    automatically when an observable is non-Hermitian, but
    downstream measurement protocols (`SpinReadout`, `ParityScan`)
    expect real inputs and will reject a complex-valued
    observable. Non-Hermitian observables are a diagnostic /
    analysis tool, not a measurement-pipeline input.

## Solve with all four side-by-side

The custom observables slot into `sequences.solve` the same way as
the built-ins:

```python
from iontrap_dynamics.observables import number, parity, spin_z
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve

# The non-Hermitian coherence observable yields complex expectations, which
# we evaluate directly from the stored states rather than passing to solve
# (the solver's expectations engine uses float64 storage for Hermitian ops).
# We store states with StorageMode.EAGER so we can also compute the coherence
# trajectory and reuse the states for concurrence below.
result = solve(
    hilbert=hilbert,
    hamiltonian=hamiltonian,
    initial_state=psi_0,
    times=times,
    storage_mode=StorageMode.EAGER,
    observables=[
        # Built-in factories (Tutorials 1–4)
        spin_z(hilbert, 0),
        spin_z(hilbert, 1),
        number(hilbert, "com"),
        parity(hilbert, ion_indices=(0, 1)),
        # Custom Hermitian observables (this tutorial)
        bell_fidelity,
        sigma_xx,
        p_fock1,
    ],
)

# Evaluate the non-Hermitian coherence trajectory from the stored states.
coherence_traj = np.array(
    [complex(qutip.expect(coherence_op, s)) for s in result.states]
)
final_coherence = coherence_traj[-1]

fid_final = float(result.expectations["bell_fidelity"][-1])
sxx_final = float(result.expectations["sigma_xx"][-1])
p1_final  = float(result.expectations["p_fock1_com"][-1])

print(f"At t_gate = {t_gate * 1e6:.2f} µs:")
print(f"  Bell fidelity  ⟨Φ⁻|ρ|Φ⁻⟩  = {fid_final:.5f}  (target 1.0)")
print(f"  ⟨σ_x σ_x⟩                  = {sxx_final:.5f}  (target 0.0)")
print(f"  P(n_com = 1)               = {p1_final:.2e}  (target 0.0)")
print(f"  coherence Im[⟨↓↓|ρ|↑↑⟩]   = {final_coherence.imag:.5f}  (target −0.5)")
print(f"  coherence Re[⟨↓↓|ρ|↑↑⟩]   = {final_coherence.real:.2e}  (target 0.0)")

assert abs(result.expectations["bell_fidelity"][-1] - 1.0) < 1e-4
assert abs(result.expectations["sigma_xx"][-1] - 0.0) < 1e-4
assert abs(result.expectations["p_fock1_com"][-1] - 0.0) < 1e-4

# Non-Hermitian result is complex; inspect imag / real separately.
assert abs(final_coherence.imag - (-0.5)) < 1e-4
assert abs(final_coherence.real) < 1e-4

# Plot all four custom observables vs time so you can watch each approach
# its Bell-gate target continuously rather than just checking the endpoint.
t_us = times * 1e6  # time axis in µs
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(t_us, result.expectations["bell_fidelity"],
        color=BLUE,   label=r"Bell fidelity $\langle\Phi^-|\rho|\Phi^-\rangle$")
ax.plot(t_us, result.expectations["sigma_xx"],
        color=RED,    label=r"$\langle\sigma_x\sigma_x\rangle$")
ax.plot(t_us, result.expectations["p_fock1_com"],
        color=GREEN,  label=r"$P(n_\mathrm{com}=1)$")
ax.plot(t_us, coherence_traj.imag,
        color=PURPLE, label=r"$\mathrm{Im}\langle{\downarrow\downarrow}|\rho|{\uparrow\uparrow}\rangle$")
ax.axvline(t_gate * 1e6, color=GREY, linewidth=0.8, linestyle="--", label=r"$t_\mathrm{gate}$")
ax.set_xlabel(r"time $t$ (µs)")
ax.set_ylabel("expectation value")
ax.set_title("Custom observables through the MS gate")
ax.legend(frameon=False)
plt.show()
```

All four land on their expected Bell-gate targets at `t_gate`.
The Bell fidelity saturates at 1; the σ_x σ_x correlator stays at 0
(`|Φ⁻⟩ = (|↓↓⟩ − i|↑↑⟩)/√2` has equal-weight computational-basis
components with opposite phases, so `⟨σ_x σ_x⟩` averages to zero);
the Fock-1 population returns to 0 (loop closure); and the coherence
`⟨↓↓|ρ|↑↑⟩` acquires its full `−i / 2` phase (the `−i` amplitude on
`|↑↑⟩` in `|Φ⁻⟩` contributes a negative imaginary part).

## When to build a brand-new factory vs. a one-off `Observable`

If you need the same observable across multiple scenarios — e.g.
every tutorial dispatches into a Bell-fidelity check — the right
pattern is a **factory function** that takes a `HilbertSpace`
(and scenario-specific indices) and returns an `Observable`:

```python
def bell_fidelity_phi_minus(
    hilbert: HilbertSpace,
    ion_indices: tuple[int, int] = (0, 1),
    *,
    label: str | None = None,
) -> Observable:
    i, j = ion_indices
    if hilbert.n_ions < 2:
        raise ValueError("Bell fidelity requires at least two ions")
    # (validate indices; build |Φ-⟩ projector; embed; return Observable)
    ...
```

This mirrors how `spin_z` / `parity` / `number` are structured —
look at
[`src/iontrap_dynamics/observables.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/observables.py)
for the template. Contributing a factory upstream is a natural
Phase 2 dispatch if your team hits the same observable repeatedly;
until then, an inline `Observable` record in your script is the
lightweight option.

## Post-hoc trajectory analysis — the full-state alternative

Every custom observable above lives inside the solve call — the
solver evaluates it step-by-step and discards the state
afterwards. If instead you want the full trajectory of density
matrices for offline analysis (slice arbitrary observables, run
the registered-entanglement evaluators like
`concurrence_trajectory`, reconstruct the reduced state on a
subsystem), pass `storage_mode=StorageMode.EAGER`:

```python
from iontrap_dynamics.entanglement import concurrence_trajectory

# result.states was stored with StorageMode.EAGER in the previous block —
# reuse it directly so we avoid a second solve call.
# result_full.states is a tuple of qutip.Qobj — one per time step
c_traj = concurrence_trajectory(
    result.states, hilbert=hilbert, ion_indices=(0, 1),
)

print(f"Concurrence at t_gate: C = {c_traj[-1]:.5f}  (target 1.0)")
assert abs(c_traj[-1] - 1.0) < 1e-3  # maximally entangled at t_gate

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(times * 1e6, c_traj, color=BLUE)
ax.axvline(t_gate * 1e6, color=GREY, linewidth=0.8, linestyle="--", label=r"$t_\mathrm{gate}$")
ax.set_xlabel(r"time $t$ (µs)")
ax.set_ylabel("Wootters concurrence $C$")
ax.set_title("Entanglement build-up through the MS gate")
ax.legend(frameon=False)
plt.show()
```

For a two-ion Bell gate `c_traj` should reach `1.0` at `t_gate`.
`storage_mode=EAGER` trades memory for flexibility — fine for
single-gate trajectories, expensive for long ensemble sweeps.
The default `StorageMode.OMITTED` keeps only the expectation
arrays and is what you want for parameter scans.

## Where to next

- [Tutorial 4](04_ms_gate_bell.md) — the MS-gate scenario
  whose `Observable`-record foothold (population projectors) this
  tutorial generalises.
- [Phase 1 Architecture](../phase-1-architecture.md) — reference
  for the `Observable` dataclass, the `observables.*` factory
  family, `StorageMode`, and the registered-entanglement
  trajectory evaluators.
- [`src/iontrap_dynamics/observables.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/observables.py)
  — the factory-function template (`spin_z`, `parity`, `number`)
  for when an inline `Observable` outgrows its scope.

---

## Licence

Sail material — adaptive guidance with specific parameter choices,
not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
