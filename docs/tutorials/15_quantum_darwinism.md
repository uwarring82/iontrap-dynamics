# Tutorial 15 — Quantum Darwinism: why the world looks classical

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/15_quantum_darwinism.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** Quantify *how* a quantum system becomes objective — measure the
redundant records an environment keeps about a system, and the
recoverability of that information from an accessible fragment. By the end
you will have plotted the quantum-Darwinism partial-information curve,
counted the redundancy `R_δ`, and watched recoverability climb as a noisy
record is purified.

**Reference implementation.** `tools/run_benchmark_darwinism_redundancy.py`
and `tools/run_benchmark_recoverability.py`, with committed plots under
[`benchmarks/data/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data).

**Expected time.** ~12 min reading; ~1 s runtime.

**Level.** `advanced` — a specialised or research-grade surface; do the core first.

**Prerequisites.** [Tutorial 14](14_quantum_metrology_qfi.md) (the
`iontrap_dynamics.information` module and the GHZ factory) and a working
notion of von Neumann entropy. CONVENTIONS.md §20 fixes the redundancy and
recoverability definitions. No prior decoherence theory is assumed.

---

!!! note "New here? Read this first"

    - **The puzzle.** Quantum states are fragile and can't be cloned, yet the everyday world looks solid and objective. Quantum Darwinism explains the gap.
    - **The mechanism.** A system does not "collapse" on its own — its environment (photons, phonons, stray qubits) keeps many *independent copies* of one preferred (pointer) observable.
    - **Objectivity = redundancy.** Because the record is copied many times, many observers can each grab a *different* fragment and all read the same fact, without disturbing the system or each other.
    - **The plot.** `partial_information_plot` gives `I(S:F)` — how much a growing environment fragment `F` tells you about the system `S`. A redundant world shows a flat **plateau** at `H_S`: one small fragment already carries the whole classical bit.
    - **The count.** The redundancy `R_δ` counts *how many* such independent fragments exist (`= N` for the ideal GHZ cascade). Plateau *height* and redundancy `R_δ` are two different axes of the same plot.
    - **The catch.** Only the *classical* information is copied this way; the *quantum* (coherent) information is monogamous and can't be redundant — Step 3's `recoverability` measures that quantum information directly.

    **In a hurry?** Step 1 (the plateau, `I(S:F) = H_S`) is the whole idea; Step 2 just counts the copies (`R_δ = N`).

**Symbols in this tutorial**

| Symbol | Meaning |
| --- | --- |
| `S`, `F` | the system (qubit 0) and an accessible fragment of the environment |
| `I(S:F)` | partial information — mutual information `F` holds about `S`; the y-axis of the plateau plot |
| `H_S` | von Neumann entropy of `S` (1 bit for a GHZ qubit) — the classical information on offer |
| plateau | the flat height of `I(S:F)`, equal to `H_S`, that *every* non-empty fragment already reaches |
| `R_δ` | redundancy — how many independent fragments each carry `(1 − δ)·H_S`; `R_δ = N / N_δ` |
| `N`, `N_δ` | environment size, and the smallest fragment size reaching the `(1 − δ)` threshold |
| recoverability | clamped coherent information `max(0, S(ρ_A) − S(ρ_{S∪A}))` — the *quantum* info one fragment returns |

## The scenario

A classical fact is *objective* because many observers can read it
independently without disturbing it — the page you are reading reflects
photons into countless eyes at once. **Quantum Darwinism** makes this
precise: a system looks classical when its environment stores *redundant*
records of one preferred (pointer) observable, so many small fragments each
carry the full system information.

The library quantifies this with three measures (§20):

- the **partial-information plot** `I(S:F)` — the mutual information a
  fragment `F` of the environment holds about the system `S`, as the
  fragment grows;
- the **redundancy** `R_δ` — how many independent fragments each carry
  almost all (`1 − δ`) of the system information;
- the **recoverability** — the clamped coherent information, how much of
  the system's *quantum* information survives in an accessible fragment.

We will read all three off a GHZ cascade (the canonical
maximally-redundant model) and a noisy Bell pair.

## Step 1 — The partial-information plot and the Darwinism plateau

Model the system as qubit 0 of a GHZ state and the environment as the
remaining qubits. For a GHZ state every *single* environment qubit already
carries the full one bit of system information, so the curve `I(S:F)` jumps
to a **plateau** at the first fragment and stays there — the signature of
classical objectivity — only doubling to two bits when the whole
environment (which restores the global quantum coherence) is included.

```python
# --- setup (boilerplate): plotting libs, house colours, and a helper that builds an N-qubit spin register ---
import matplotlib.pyplot as plt
import numpy as np
import qutip
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.system import IonSystem
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import ghz_state
from iontrap_dynamics.information import partial_information_plot

# House colours — match the shipped benchmark figures.
BLUE, RED, GREEN, PURPLE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#444444"

def spin_hilbert(n_ions):
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})

# --- the physics: a 6-qubit GHZ state = 1 system qubit + 5 redundant environment records ---
hilbert = spin_hilbert(6)                     # 1 system + 5 environment qubits
state = ghz_state(hilbert)
pip = partial_information_plot(
    state, hilbert=hilbert, system_indices=[0], environment_indices=[1, 2, 3, 4, 5]
)
# pip = [0, 1, 1, 1, 1, 2]  — zero, then a flat 1-bit plateau, then 2 bits at the full env
print(f"Step 1 — partial-information plot (bits):  {[round(float(v), 6) for v in pip]}")
print(f"  plateau value pip[1..4] = {[round(float(v), 6) for v in pip[1:-1]]}  (expect all 1.0)")
print(f"  full-environment value pip[-1] = {float(pip[-1]):.6f}  (expect 2.0)")
assert pip[0] == 0.0
assert np.allclose(pip[1:-1], 1.0), "the plateau: every non-empty proper fragment already carries the full system bit H_S = 1"            # every proper fragment: the plateau
assert abs(pip[-1] - 2.0) < 1e-9              # full environment restores global coherence

# Fragment sizes: 0 env qubits → 5 env qubits (i.e. fragment = 0..5 out of 5)
frag_sizes = list(range(len(pip)))
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(frag_sizes, pip, color=BLUE, marker="o", markersize=5, linewidth=1.4,
        label=r"$I(S{:}F)$  GHZ")
ax.axhline(1.0, color=GREY, linewidth=0.8, linestyle="--", label=r"plateau: $H_S = 1$ bit")
ax.set_xlabel("fragment size (# env qubits)")
ax.set_ylabel(r"$I(S{:}F)$  (bits)")
ax.set_title("Darwinism plateau — each fragment carries the full system bit")
ax.legend(frameon=False)
plt.show()
```

**Takeaway.** Because the curve is *flat*, it does not matter which fragment an observer intercepts — every independent observer reads the same one-bit fact and none disturbs the others, which is the operational meaning of an objective outcome.

![Quantum-Darwinism partial-information plot rising to a one-bit plateau, and redundancy growing linearly with environment size](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/darwinism_redundancy/plot.png)

The left panel is this plot: a sharp rise to the one-bit plateau, flat
across every proper fragment (the system fact is *out there*, many times
over), then the jump to two bits. A plateau at `H_S` is the quantitative
definition of an objective, redundantly-recorded observable.

!!! warning "Common confusion — the 2-bit endpoint is not extra redundancy"

    Read only the **plateau** (`= H_S = 1` bit) as the redundant, objective
    classical information. The final rise to 2 bits happens *only* at the last
    point, where the fragment is the *entire* environment — that excess is
    quantum mutual information (the global coherence of the pure GHZ state)
    and no small fragment, hence no realistic observer, can access it.
    Objectivity lives in the flat part of the curve, not its endpoint.

## Step 2 — Counting the records: redundancy `R_δ`

The plateau says *each* fragment is informative; the **redundancy** counts
*how many* independent such records exist. `R_δ = N / N_δ`, where `N_δ` is
the smallest fragment that already carries `(1 − δ)` of the system entropy.
For the GHZ cascade a single qubit suffices, so `N_δ = 1` and `R_δ = N` —
the system bit is imprinted `N` times over.

```python
from iontrap_dynamics.information import redundancy

r_values = []
n_env_values = list(range(1, 7))
for n_env in n_env_values:
    hilbert_r = spin_hilbert(n_env + 1)
    state_r = ghz_state(hilbert_r)
    r = redundancy(
        state_r, hilbert=hilbert_r, system_indices=[0],
        environment_indices=list(range(1, n_env + 1)), delta=0.1,
    )
    r_values.append(float(r))
    assert abs(r - n_env) < 1e-9, "R_δ = N here because one qubit (N_δ = 1) already carries the full system bit"              # R_δ = N: one independent record per qubit
# redundancy = [1, 2, 3, 4, 5, 6]

print(f"Step 2 — redundancy R_δ (δ=0.1) vs environment size:")
for n_env, r in zip(n_env_values, r_values):
    print(f"  N_env = {n_env}  →  R_δ = {r:.1f}  (expect {n_env})")

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(n_env_values, r_values, color=GREEN, marker="o", markersize=5, linewidth=1.4,
        label=r"$R_\delta$  GHZ  ($\delta = 0.1$)")
ax.plot(n_env_values, n_env_values, color=GREY, linewidth=0.8, linestyle="--",
        label=r"$R_\delta = N$  (ideal)")
ax.set_xlabel("environment size $N$")
ax.set_ylabel(r"redundancy $R_\delta$")
ax.set_title(r"$R_\delta = N$: every qubit is an independent record")
ax.legend(frameon=False)
plt.show()
```

**Takeaway.** For a perfect GHZ record a single qubit already holds the *entire* bit, so `R_δ = N` holds for *any* deficit `δ ∈ (0, 1)` — the redundancy of an ideal record is insensitive to how strict you set the tolerance, and `R_δ = N` is the ceiling any `N`-qubit environment can reach.

The right panel of the figure above is `R_δ` versus environment size — a
straight line `R_δ = N`. The more redundant the encoding, the more robust
the classical fact: losing any one record changes nothing.

!!! note "Plateau height vs redundancy"

    These two numbers answer different questions. The **plateau height**
    (`pip[1]`, here 1 bit) is *how much* each fragment knows; the
    **redundancy** (`R_δ`, here `N`) is *how many* fragments know it. A
    fragile encoding can have a high plateau but `R_δ ≈ 1`; objectivity
    needs both — a large plateau *and* a large redundancy.

## Step 3 — Recoverability: the quantum information that survives

Mutual information and redundancy are about *classical* records. The
**recoverability** — the clamped coherent information
`max(0, S(ρ_A) − S(ρ_{S∪A}))` — asks instead how much of the system's
*quantum* information can be recovered from an accessible fragment. It is
exactly zero for a classically-correlated record and rises to `H_S` as the
shared state is purified. We sweep a Werner-mixed Bell pair from fully
decohered to pure.

!!! warning "Common confusion — only classical information is copied"

    Do not expect the recoverability of a *quantum* record to be redundant the
    way `I(S:F)` is. Only *classical* information can be stamped into many
    fragments at once — that redundancy is exactly what makes an outcome
    objective. The *coherent* (quantum) information is monogamous: a system can
    be maximally entangled with at most one party, so at most one fragment can
    return `H_S`, and it can never be read off many fragments independently.

```python
from iontrap_dynamics.information import recoverability

pair = spin_hilbert(2)                        # system qubit 0, accessible qubit 1
bell = qutip.bell_state("00")                 # |Φ⁺⟩
maximally_mixed = qutip.qeye([2, 2]) / 4.0
p_values = [0.0, 0.25, 0.5, 0.75, 1.0]
rec_values = []
for p in p_values:
    werner = p * (bell * bell.dag()) + (1.0 - p) * maximally_mixed
    rec = recoverability(werner, hilbert=pair, system_indices=[0], accessible_indices=[1])
    rec_values.append(float(rec))
    # rec = 0 at p=0 (no quantum info), rising monotonically to 1 bit at p=1 (pure Bell)

print("Step 3 — recoverability (bits) vs Werner mixing parameter p:")
for p, r in zip(p_values, rec_values):
    print(f"  p = {p:.2f}  →  recoverability = {r:.4f} bits")
assert abs(recoverability(bell, hilbert=pair, system_indices=[0], accessible_indices=[1]) - 1.0) < 1e-9, "a pure Bell pair returns the full one-bit H_S of quantum (coherent) information"
assert recoverability(maximally_mixed, hilbert=pair, system_indices=[0], accessible_indices=[1]) == 0.0, "a fully decohered (maximally mixed) record returns zero recoverable quantum information"

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(p_values, rec_values, color=RED, marker="o", markersize=5, linewidth=1.4,
        label="recoverability")
ax.set_xlabel(r"Werner purity $p$  ($p=0$: maximally mixed, $p=1$: pure Bell)")
ax.set_ylabel("recoverability (bits)")
ax.set_title("Quantum information recovered from a Werner-mixed Bell pair")
ax.legend(frameon=False)
plt.show()
```

![Recoverability rising from zero to one bit as a Werner-mixed Bell pair is purified](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/recoverability/plot.png)

The curve is monotone: zero recoverable quantum information from a fully
decohered record, exactly one bit from a pure Bell pair. This is the
complement of Darwinism's story — redundancy explains why *classical*
information proliferates, recoverability tracks the *quantum* information
that does not.

## Where to next

- [Tutorial 14 — Quantum metrology](14_quantum_metrology_qfi.md): the
  upstream half of the `iontrap_dynamics.information` surface, where the
  same GHZ state gives the Heisenberg-limited Fisher information.
- The benchmarks `tools/run_benchmark_darwinism_redundancy.py` and
  `tools/run_benchmark_recoverability.py` reproduce both figures, and
  `tools/run_benchmark_ghz_cat.py` shows the GHZ parity fringes that make
  the pointer observable concrete.

---

## Licence

Sail material — adaptive guidance with specific parameter choices, not a
coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
