# Tutorial 0 — The 30-second mental model

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/00_mental_model.ipynb) — run it live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** Before any physics: see the whole library in one breath. `iontrap-dynamics`
turns a *description of an ion-trap experiment* into a *simulated measurement* — that's the
entire job. This page shows the mental model and runs one short simulation that flops a
spin, so you get a result **before** meeting Fock truncation, storage modes, or warnings.

**Expected time.** ~3 min reading; ~1 s runtime.

**Level.** `intro` — no prerequisites; start here.

**Prerequisites.** None. A working install (`pip install iontrap-dynamics`) — or just press
**Open in Colab** above.

---

**Symbols in this tutorial.**

| Symbol | Plain meaning |
|---|---|
| `⟨σ_z⟩` | Expectation value of Pauli-`z`: the spin population difference — `−1` is spin-down, `+1` is spin-up |
| `Ω` | Carrier Rabi frequency — how fast a resonant drive flops the spin; one down→up→down cycle takes `2π/Ω` |
| \|↓⟩, \|↑⟩ | The two qubit basis states: spin-down and spin-up |
| `fock` | The motional number-state (Fock) basis \|n⟩, truncated to a finite dimension via `fock_truncations` (here `3`) |

## The one idea

Every simulation in this library is the same five-step pipeline. The **physics** and the
**code** line up one-to-one:

| Physics | Code | What it is |
|---|---|---|
| ion + laser + trap | `IonSystem`, `DriveConfig`, `ModeConfig` | *what's in the lab* |
| choose a state space | `HilbertSpace` | *how big a basis* |
| the interaction | a `hamiltonian` builder | *what drives the dynamics* |
| let time run | `solve(...)` | *the simulation* |
| what you measure | an `Observable` → readout | *the number you get out* |

```
IonSystem ──► HilbertSpace ──► Hamiltonian ──► solve ──► readout
    │              │               │             │          │
  species       Fock dim       operators      backend    observable
  modes         truncation     + waveform     + times    sampling
```

Read that spine once. Every one of the tutorials that follows is a variation on it — a
different Hamiltonian, a different readout, a different systematic — but the five boxes
never change.

## Run it: flop a spin

Here is the whole pipeline in one cell. It drives a single ²⁵Mg⁺ ion on resonance and
watches its spin flop from down (`⟨σ_z⟩ = −1`) to up (`+1`) and back — the "hello world"
of trapped-ion dynamics. Run it and you have your first result.

!!! warning "Common confusion — ⟨σ_z⟩ is a population difference, not a negative spin"
    `⟨σ_z⟩` is an *expectation value* in `[−1, +1]` — it is `P↑ − P↓`, not a spin that has
    gone negative. So `⟨σ_z⟩ = 0` means equal down/up *populations*, which could be an equal
    superposition *or* a 50/50 mixture — `⟨σ_z⟩` alone cannot tell the two apart. The sign
    just says which state dominates.

```python
import numpy as np
import qutip

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

# what's in the lab: one Mg-25 ion, one axial mode, one resonant laser
mode = ModeConfig(label="axial", frequency_rad_s=2 * np.pi * 1.5e6,
                  eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]))
system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
drive = DriveConfig(k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
                    carrier_rabi_frequency_rad_s=2 * np.pi * 1.0e6, phase_rad=0.0)

# a basis, the drive Hamiltonian, and one Rabi period of evolution from the ground state
hilbert = HilbertSpace(system=system, fock_truncations={"axial": 3})
hamiltonian = carrier_hamiltonian(hilbert, drive, ion_index=0)
psi0 = qutip.tensor(spin_down(), qutip.basis(3, 0))  # |↓, n = 0⟩
times = np.linspace(0.0, 2 * np.pi / drive.carrier_rabi_frequency_rad_s, 50)

result = solve(hilbert=hilbert, hamiltonian=hamiltonian, initial_state=psi0,
               times=times, observables=[spin_z(hilbert, 0)])

sz = np.array(result.expectations["sigma_z_0"])
print(f"⟨σ_z⟩:  start {sz[0]:+.2f}  →  half-way {sz[len(sz) // 2]:+.2f}  →  end {sz[-1]:+.2f}")

# The spin started down (−1), flopped fully up (+1), and came back — one Rabi cycle.
assert sz[0] < -0.99 and sz[len(sz) // 2] > 0.99 and sz[-1] < -0.99, \
    "The ideal on-resonance carrier drive must flop |↓⟩ → |↑⟩ → |↓⟩ over one Rabi period."
```

That is a complete, correct trapped-ion simulation — five objects, one `solve`. Everything
else in the track just adds detail to one of those five boxes.

## Where to go next

- **[Tutorial 1](01_first_rabi_readout.md)** takes this exact scenario and adds the missing
  realism: a finite-shot photon detector and 95 % confidence intervals on the readout.
- **New to the vocabulary?** Every recurring term (Fock truncation, Lamb–Dicke, adiabatic,
  squeezing, symplectic eigenvalue, …) is defined once in the **[Glossary](../glossary.md)**.
- **In a hurry?** The newcomer path is **0 → 1 → 2 → 6**. After those four you can build a
  Hamiltonian, read it out, and diagnose the one thing that bites beginners first: Fock
  truncation.
