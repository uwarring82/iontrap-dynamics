# Tutorial 14 — Quantum metrology: the Fisher-information limit

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/14_quantum_metrology_qfi.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

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
import matplotlib.pyplot as plt
import numpy as np
import qutip
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.system import IonSystem
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.operators import sigma_z_ion, spin_up, spin_down
from iontrap_dynamics.information import quantum_fisher_information_trajectory

# House colours — match the reference style throughout.
BLUE, RED, GREEN, PURPLE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#444444"


def spin_hilbert(n_ions):
    system = IonSystem(species_per_ion=tuple(mg25_plus() for _ in range(n_ions)))
    return HilbertSpace(system=system, fock_truncations={})

h1 = spin_hilbert(1)
jz = 0.5 * h1.spin_op_for_ion(sigma_z_ion(), 0)         # generator on the FULL space
plus = (spin_up() + spin_down()).unit()

qfi = quantum_fisher_information_trajectory([plus], hilbert=h1, generator=jz)[0]
var_g = float(qutip.variance(jz, plus))
print(f"Step 1 — F_Q[|+⟩, J_z] = {qfi:.6f}  (predicted 1.000000)")
print(f"Step 1 — 4·Var(J_z)    = {4.0 * var_g:.6f}  (pure-state identity check)")
assert abs(qfi - 1.0) < 1e-12                            # F_Q = 4·Var(J_z) = 1
assert abs(qfi - 4.0 * var_g) < 1e-12                   # the pure-state identity

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.bar(["$F_Q$ (SLD)", r"$4\cdot\mathrm{Var}(J_z)$"], [qfi, 4.0 * var_g],
       color=[BLUE, GREY], width=0.4)
ax.axhline(1.0, color=RED, linewidth=1.0, linestyle="--", label="predicted = 1")
ax.set_ylabel(r"Fisher information")
ax.set_title(r"Step 1 · $F_Q[|+\rangle, J_z] = 4\,\mathrm{Var}(J_z) = 1$")
ax.legend(frameon=False)
plt.show()
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

ns = list(range(1, 7))
qfi_ghz_vals, qfi_prod_vals = [], []
for N in ns:
    h = spin_hilbert(N); jz = collective_jz(h)
    qfi_ghz = quantum_fisher_information_trajectory([ghz_state(h)], hilbert=h, generator=jz)[0]
    qfi_prod = quantum_fisher_information_trajectory([product_plus(h)], hilbert=h, generator=jz)[0]
    assert abs(qfi_ghz - N**2) < 1e-9 and abs(qfi_prod - N) < 1e-9
    qfi_ghz_vals.append(qfi_ghz)
    qfi_prod_vals.append(qfi_prod)
# F_Q(GHZ) = [1, 4, 9, 16, 25, 36];  F_Q(product) = [1, 2, 3, 4, 5, 6]

print("Step 2 — QFI scaling:  N   GHZ (N²)   product (N)")
for N, g, p in zip(ns, qfi_ghz_vals, qfi_prod_vals):
    print(f"                      {N}   {g:6.1f}     {p:.1f}")
print(f"         Heisenberg gain at N=6: {qfi_ghz_vals[-1] / qfi_prod_vals[-1]:.0f}×")

n_arr = np.array(ns)
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(n_arr, n_arr**2, color=GREY, linewidth=1.0, label=r"$N^2$ (Heisenberg)")
ax.scatter(n_arr, qfi_ghz_vals, color=PURPLE, s=40, zorder=3, label="GHZ probe")
ax.plot(n_arr, n_arr, color=GREY, linewidth=1.0, linestyle="--", label="$N$ (SQL)")
ax.scatter(n_arr, qfi_prod_vals, color=BLUE, marker="s", s=40, zorder=3, label="product probe")
ax.set_xlabel("number of ions $N$")
ax.set_ylabel(r"$F_Q$")
ax.set_title("Step 2 · Heisenberg vs SQL scaling")
ax.legend(frameon=False)
plt.show()
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
cfi_y = cfi_in_basis(sigma_y_ion())
cfi_x = cfi_in_basis(sigma_x_ion())
crb = cramer_rao_bound(qfi)
print(f"Step 3 — F_Q         = {qfi:.6f}  (the ceiling)")
print(f"Step 3 — F_C(σ_y)   = {cfi_y:.6f}  (saturates: optimal basis)")
print(f"Step 3 — F_C(σ_x)   = {cfi_x:.6f}  (phase-blind: F_C = 0)")
print(f"Step 3 — CRB 1/F_Q  = {crb:.6f}  → σ_θ ≥ {crb**0.5:.6f} rad")
assert abs(cfi_y - qfi) < 1e-12   # σ_y SATURATES: F_C = F_Q = 1
assert abs(cfi_x - 0.0) < 1e-12   # σ_x is phase-blind: F_C = 0
assert abs(crb - 1.0) < 1e-12     # Var(θ̂) ≥ 1/F_Q = 1

fig, ax = plt.subplots(figsize=(5.0, 3.2))
labels = [r"$F_Q$  (ceiling)", r"$F_C(\sigma_y)$  optimal", r"$F_C(\sigma_x)$  blind"]
values = [qfi, cfi_y, cfi_x]
colours = [GREY, GREEN, RED]
ax.bar(labels, values, color=colours, width=0.5)
ax.axhline(qfi, color=GREY, linewidth=1.0, linestyle="--")
ax.set_ylabel("Fisher information")
ax.set_title(r"Step 3 · Cramér–Rao saturation: right vs wrong basis")
plt.show()
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
call, just with a mode probe and a quadrature generator. The code below
computes both curves live and redraws the figure.

```python
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.states import coherent_mode, squeezed_vacuum_mode

# FOCK4=40 is safe for |α|² ≤ 9 and r ≤ 1.0 (sinh²(1) ≈ 1.4 phonons).
FOCK4 = 40

def mode_hilbert(fock_dim: int) -> HilbertSpace:
    """One ion + one axial mode — the minimal CV Hilbert space."""
    mode = ModeConfig(
        label="m",
        frequency_rad_s=2 * np.pi * 1e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={"m": fock_dim})

hm = mode_hilbert(FOCK4)
# Mode operators embedded on the full (spin ⊗ mode) space.
n_hat  = hm.number_for_mode("m")                              # G = n̂  (phase encoding, F_Q = 4|α|²)
a_op   = hm.mode_op_for(qutip.destroy(FOCK4), "m")
# Normalised quadratures: Var(x̂)_{vac} = ½, so F_Q = 4·½ = 2 at shot-noise floor.
x_hat  = (a_op + a_op.dag()) / np.sqrt(2)                     # squeezed-quadrature generator
p_hat  = 1j * (a_op.dag() - a_op) / np.sqrt(2)               # anti-squeezed-quadrature generator

# ------- coherent probe: F_Q = 4|α|² -------
alpha_vals = np.linspace(0.5, 3.0, 10)
qfi_coherent = []
for alpha in alpha_vals:
    psi_m = coherent_mode(FOCK4, float(alpha))
    psi = qutip.tensor(spin_down(), psi_m)
    qfi_coherent.append(
        quantum_fisher_information_trajectory([psi], hilbert=hm, generator=n_hat)[0]
    )
qfi_coherent = np.array(qfi_coherent)
analytic_coherent = 4.0 * alpha_vals**2

print("Step 4 — coherent probe F_Q = 4|α|²:")
for a, fq, fa in zip(alpha_vals, qfi_coherent, analytic_coherent):
    print(f"    α = {a:.2f}  F_Q = {fq:.4f}  (predicted {fa:.4f})")

# ------- squeezed probe: squeezed / anti-squeezed quadrature (r ≤ 1.0) -------
# F_Q(x̂, S(r)|0⟩) = 4·Var(x̂) = 2e^{−2r} < 2 (sub-shot-noise)
# F_Q(p̂, S(r)|0⟩) = 4·Var(p̂) = 2e^{+2r} > 2 (anti-squeezed)
r_vals = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
qfi_sq, qfi_antisq = [], []
for r in r_vals:
    psi_m = squeezed_vacuum_mode(FOCK4, float(r))
    psi = qutip.tensor(spin_down(), psi_m)
    qfi_sq.append(
        quantum_fisher_information_trajectory([psi], hilbert=hm, generator=x_hat)[0]
    )
    qfi_antisq.append(
        quantum_fisher_information_trajectory([psi], hilbert=hm, generator=p_hat)[0]
    )
qfi_sq     = np.array(qfi_sq)
qfi_antisq = np.array(qfi_antisq)
analytic_sq     = 2.0 * np.exp(-2.0 * r_vals)   # squeezed quadrature: 4·Var(x̂) = 2e^{-2r}
analytic_antisq = 2.0 * np.exp(+2.0 * r_vals)   # anti-squeezed quadrature: 2e^{+2r}

print("\nStep 4 — squeezed probe F_Q vs squeezing r (shot-noise floor = 2.0):")
for r, fs, fa in zip(r_vals, qfi_sq, qfi_antisq):
    print(f"    r = {r:.1f}   squeezed F_Q = {fs:.4f}  anti-sq F_Q = {fa:.4f}")
print(f"    sub-shot-noise at r=1.0: F_Q = {qfi_sq[-1]:.4f} < 2.0 = floor")

# The squeezed quadrature must be sub-shot-noise for r > 0.
assert qfi_sq[-1] < 1.0          # deep sub-shot-noise at r = 1.0  (2e^{-2} ≈ 0.27)
assert qfi_antisq[-1] > 10.0     # anti-squeezed quadrature: 2e^{+2} ≈ 14.8 >> floor

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))

ax = axes[0]
ax.plot(alpha_vals, analytic_coherent, color=GREY, linewidth=1.0, label=r"$4|\alpha|^2$")
ax.scatter(alpha_vals, qfi_coherent, color=BLUE, s=30, zorder=3, label="coherent probe")
ax.set_xlabel(r"coherent amplitude $|\alpha|$")
ax.set_ylabel(r"$F_Q$")
ax.set_title(r"CV SQL: coherent probe $F_Q = 4|\alpha|^2$")
ax.legend(frameon=False)

ax = axes[1]
ax.plot(r_vals, analytic_sq,     color=GREEN, linewidth=1.0, label=r"$2e^{-2r}$ (squeezed)")
ax.scatter(r_vals, qfi_sq,       color=GREEN, s=30, zorder=3)
ax.plot(r_vals, analytic_antisq, color=RED,   linewidth=1.0, label=r"$2e^{+2r}$ (anti-sq.)")
ax.scatter(r_vals, qfi_antisq,   color=RED,   s=30, zorder=3)
ax.axhline(2.0, color=GREY, linewidth=1.0, linestyle="--", label="shot-noise floor")
ax.set_yscale("log")
ax.set_xlabel(r"squeezing parameter $r$")
ax.set_ylabel(r"$F_Q$  (log scale)")
ax.set_title(r"CV sub-shot-noise: squeezed probe")
ax.legend(frameon=False)

plt.tight_layout()
plt.show()
```

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
