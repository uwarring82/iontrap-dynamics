# Tutorial 15 — Quantum Darwinism: why the world looks classical

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

**Prerequisites.** [Tutorial 14](14_quantum_metrology_qfi.md) (the
`iontrap_dynamics.information` module and the GHZ factory) and a working
notion of von Neumann entropy. CONVENTIONS.md §20 fixes the redundancy and
recoverability definitions. No prior decoherence theory is assumed.

---

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
import numpy as np
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.system import IonSystem
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import ghz_state
from iontrap_dynamics.information import partial_information_plot

def spin_hilbert(n_ions):
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})

hilbert = spin_hilbert(6)                     # 1 system + 5 environment qubits
state = ghz_state(hilbert)
pip = partial_information_plot(
    state, hilbert=hilbert, system_indices=[0], environment_indices=[1, 2, 3, 4, 5]
)
# pip = [0, 1, 1, 1, 1, 2]  — zero, then a flat 1-bit plateau, then 2 bits at the full env
assert pip[0] == 0.0
assert np.allclose(pip[1:-1], 1.0)            # every proper fragment: the plateau
assert abs(pip[-1] - 2.0) < 1e-9              # full environment restores global coherence
```

![Quantum-Darwinism partial-information plot rising to a one-bit plateau, and redundancy growing linearly with environment size](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/darwinism_redundancy/plot.png)

The left panel is this plot: a sharp rise to the one-bit plateau, flat
across every proper fragment (the system fact is *out there*, many times
over), then the jump to two bits. A plateau at `H_S` is the quantitative
definition of an objective, redundantly-recorded observable.

## Step 2 — Counting the records: redundancy `R_δ`

The plateau says *each* fragment is informative; the **redundancy** counts
*how many* independent such records exist. `R_δ = N / N_δ`, where `N_δ` is
the smallest fragment that already carries `(1 − δ)` of the system entropy.
For the GHZ cascade a single qubit suffices, so `N_δ = 1` and `R_δ = N` —
the system bit is imprinted `N` times over.

```python
from iontrap_dynamics.information import redundancy

for n_env in range(1, 7):
    hilbert = spin_hilbert(n_env + 1)
    state = ghz_state(hilbert)
    r = redundancy(
        state, hilbert=hilbert, system_indices=[0],
        environment_indices=list(range(1, n_env + 1)), delta=0.1,
    )
    assert abs(r - n_env) < 1e-9              # R_δ = N: one independent record per qubit
# redundancy = [1, 2, 3, 4, 5, 6]
```

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

```python
import qutip
from iontrap_dynamics.information import recoverability

pair = spin_hilbert(2)                        # system qubit 0, accessible qubit 1
bell = qutip.bell_state("00")                 # |Φ⁺⟩
maximally_mixed = qutip.qeye([2, 2]) / 4.0
for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    werner = p * (bell * bell.dag()) + (1.0 - p) * maximally_mixed
    rec = recoverability(werner, hilbert=pair, system_indices=[0], accessible_indices=[1])
    # rec = 0 at p=0 (no quantum info), rising monotonically to 1 bit at p=1 (pure Bell)
assert abs(recoverability(bell, hilbert=pair, system_indices=[0], accessible_indices=[1]) - 1.0) < 1e-9
assert recoverability(maximally_mixed, hilbert=pair, system_indices=[0], accessible_indices=[1]) == 0.0
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
