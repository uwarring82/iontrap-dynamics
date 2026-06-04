# Tutorial 14 — Quantum metrology: the Fisher-information limit

**Goal.** Estimate a parameter encoded on a quantum probe, read off the
best possible precision with the quantum Fisher information (QFI), and
see which measurement actually reaches it. By the end you will have
computed `F_Q` for single- and multi-qubit probes, watched a GHZ state
beat the standard quantum limit, and confirmed that the Cramér–Rao bound
is saturated only in the right measurement basis.

**Reference implementation.** `tools/run_benchmark_qfi_scaling.py`,
`tools/run_benchmark_cfi_linear_gaussian.py`, and
`tools/run_benchmark_probe_qfi.py`, with committed plots under
[`benchmarks/data/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data).

**Expected time.** ~12 min reading; ~1 s runtime.

**Prerequisites.** [Tutorial 4](04_ms_gate_bell.md) (two-ion systems) and
[Tutorial 9](09_squeezed_coherent_prep.md) (motional state factories), plus
CONVENTIONS.md §19 (the SLD quantum Fisher information convention). No prior
estimation theory is assumed — the relevant formulas are introduced inline.

---

## The scenario

A *metrology* problem has three parts: a probe state `ρ`, a parameter `θ`
encoded on it, and a measurement that reads `θ` back out. The library
encodes `θ` **unitarily** — `ρ(θ) = e^{−iθG} ρ e^{+iθG}` for a Hermitian
generator `G` — which is the right model for a phase accrued under a
known Hamiltonian.

The quantum Fisher information `F_Q[ρ, G]` is the best the *physics*
allows: any unbiased estimator obeys the quantum Cramér–Rao bound
`Var(θ̂) ≥ 1 / F_Q`. For a **pure** state it collapses to a one-line
identity,

```
F_Q[|ψ⟩, G] = 4 · Var_ψ(G),
```

so the whole story is the variance of the generator in the probe. We will
build that intuition on a single qubit, scale it to N qubits to expose the
Heisenberg limit, and then ask the operational question the QFI does *not*
answer on its own: which measurement realises it?

## Step 1 — A single-qubit probe and `F_Q = 4·Var(G)`

Take the probe `|+⟩ = (|↑⟩ + |↓⟩)/√2` and the phase generator
`G = J_z = ½σ_z`. Since `Var_{|+⟩}(J_z) = ¼`, the identity predicts
`F_Q = 4·¼ = 1`. The library computes the full SLD quantum Fisher
information (which agrees on pure states and is strictly smaller on mixed
ones), so we can check the number directly.

```python
import numpy as np
import qutip
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.system import IonSystem
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.operators import sigma_z_ion, spin_up, spin_down
from iontrap_dynamics.information import quantum_fisher_information_trajectory

def spin_hilbert(n_ions):
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})

h1 = spin_hilbert(1)
jz = 0.5 * h1.spin_op_for_ion(sigma_z_ion(), 0)         # generator on the FULL space
plus = (spin_up() + spin_down()).unit()

qfi = quantum_fisher_information_trajectory([plus], hilbert=h1, generator=jz)[0]
assert abs(qfi - 1.0) < 1e-12                            # F_Q = 4·Var(J_z) = 1
assert abs(qfi - 4.0 * qutip.variance(jz, plus)) < 1e-12 # the pure-state identity
```

!!! note "Two easy-to-miss conventions"

    `quantum_fisher_information_trajectory` is keyword-only for `hilbert=`
    and `generator=`, and takes a **sequence** of states (the `[0]` slice is
    the `t = 0` probe). The generator must be **Hermitian** and embedded on
    the full Hilbert space — use `hilbert.spin_op_for_ion(...)`, not a bare
    `2×2` matrix. And the `½` in `J_z = ½σ_z` is load-bearing: drop it and
    you compute `F_Q` for a generator that is twice as large, so the QFI
    comes out 4× too big.

## Step 2 — Heisenberg vs the standard quantum limit

Now scale up. With `N` independent probes the variance of the collective
`J_z = ½ Σ_i σ_z^{(i)}` adds, so a product state gives `F_Q = N` — the
**standard quantum limit** (SQL). A maximally-entangled GHZ probe
concentrates the variance and reaches `F_Q = N²` — the **Heisenberg
limit**, a quadratic metrological gain.

```python
from iontrap_dynamics.states import ghz_state

def collective_jz(h):
    total = h.spin_op_for_ion(sigma_z_ion(), 0)
    for i in range(1, h.n_ions):
        total = total + h.spin_op_for_ion(sigma_z_ion(), i)
    return 0.5 * total

def product_plus(h):
    plus = (spin_up() + spin_down()).unit()
    return qutip.tensor([plus] * h.n_ions)

for N in range(1, 7):
    h = spin_hilbert(N); jz = collective_jz(h)
    qfi_ghz = quantum_fisher_information_trajectory([ghz_state(h)], hilbert=h, generator=jz)[0]
    qfi_prod = quantum_fisher_information_trajectory([product_plus(h)], hilbert=h, generator=jz)[0]
    assert abs(qfi_ghz - N**2) < 1e-9 and abs(qfi_prod - N) < 1e-9
# F_Q(GHZ) = [1, 4, 9, 16, 25, 36];  F_Q(product) = [1, 2, 3, 4, 5, 6]
```

The two scalings are the headline of the estimation surface:

![QFI scaling: GHZ Heisenberg N squared versus product standard quantum limit N](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/qfi_scaling/plot.png)

On the log–log axis the GHZ curve has slope 2 (`N²`) and the product curve
slope 1 (`N`); the numerical points sit exactly on the textbook lines. The
widening gap is the metrological payoff of entanglement — at `N = 6` the
GHZ probe is six times more informative per shot.

## Step 3 — The Cramér–Rao bound and the *optimal* measurement

The QFI is a ceiling. A real experiment measures something, and the
*classical* Fisher information `F_C = Σ_x (∂_θ p_x)² / p_x` of that
measurement's outcome distribution obeys `F_C ≤ F_Q` (Braunstein–Caves).
Equality — saturation of the bound — holds only for the optimal
measurement, the eigenbasis of the symmetric logarithmic derivative.

For the `|+⟩` / `J_z` probe the optimal basis is **σ_y**, not σ_x. That is
easy to get wrong: `|+⟩` is a σ_x eigenstate, so a σ_x measurement is
*phase-blind* — to first order in `θ` the outcome probabilities do not
move, and `F_C = 0`. We compute both, deriving the probabilities and their
derivatives from the actual operators rather than hand-typing them.

```python
from iontrap_dynamics.operators import sigma_x_ion, sigma_y_ion
from iontrap_dynamics.information import classical_fisher_information, cramer_rao_bound

def cfi_in_basis(measurement):
    identity = qutip.qeye(measurement.dims[0])
    projectors = [(identity + measurement) / 2, (identity - measurement) / 2]
    p = [float(qutip.expect(P, plus).real) for P in projectors]            # outcome probabilities
    dp = [float((1j * qutip.expect(jz * P - P * jz, plus)).real) for P in projectors]  # ∂_θ p at θ=0
    return classical_fisher_information(p, parameter_derivative=dp)

h1 = spin_hilbert(1); jz = 0.5 * h1.spin_op_for_ion(sigma_z_ion(), 0)
qfi = quantum_fisher_information_trajectory([plus], hilbert=h1, generator=jz)[0]
assert abs(cfi_in_basis(sigma_y_ion()) - qfi) < 1e-12   # σ_y SATURATES: F_C = F_Q = 1
assert abs(cfi_in_basis(sigma_x_ion()) - 0.0) < 1e-12   # σ_x is phase-blind: F_C = 0
assert abs(cramer_rao_bound(qfi) - 1.0) < 1e-12         # Var(θ̂) ≥ 1/F_Q = 1
```

![Classical Fisher information of a linear-Gaussian model matching the analytic reference, and Cramér–Rao saturation on the single-qubit phase model](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/cfi_linear_gaussian/plot.png)

The lesson is operational: **the QFI is only useful if you measure in the
right basis.** A perfectly good probe read out the wrong way carries zero
information about `θ`.

## Step 4 — Continuous-variable probes (optional)

The same machinery applies to a motional mode. A coherent probe `|α⟩` under
phase encoding (`G = n̂`) gives `F_Q = 4|α|²` — the CV standard quantum
limit. A squeezed-vacuum probe beats it: sensing a displacement along the
squeezed quadrature gives `F_Q = 2e^{−2r}`, dropping *below* the shot-noise
floor of 2 as the squeezing `r` grows.

![Quantum Fisher information versus squeezing parameter for a squeezed-vacuum probe, dipping below the shot-noise floor on the squeezed quadrature](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/probe_qfi/plot.png)

The squeezed-quadrature curve (`2e^{−2r}`) dives under the floor; the
conjugate quadrature pays `2e^{+2r}` above it. This is sub-shot-noise
metrology — the continuous-variable analogue of the GHZ gain in Step 2 —
and it is built on exactly the same `quantum_fisher_information_trajectory`
call, just with a mode probe and a quadrature generator.

## Where to next

- [Tutorial 15 — Quantum Darwinism](15_quantum_darwinism.md): the same
  `iontrap_dynamics.information` module and the GHZ probe, but a different
  question — why a quantum system *looks* classical.
- The probe-QFI benchmark `tools/run_benchmark_probe_qfi.py` reproduces the
  Step 4 figure across coherent and squeezed probes.

---

## Licence

Sail material — adaptive guidance with specific parameter choices, not a
coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
