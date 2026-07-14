# Tutorial 20 — Phonon-pair creation and reading it out

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/20_phonon_pair_creation.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** Read a squeezed motional state out as a **phonon-number distribution**
`Pₙ` and see the fingerprint of squeezing: population in **even `n` only** — phonons
are created in **pairs**. By the end you will have compared the numerical `Pₙ`
against its analytic closed form, grown the pairs by parametric modulation, and met
the parity-aware Fock-truncation guard that keeps the readout honest. This is the
single-ion readout physics of Wittemer et al., *Phil. Trans. R. Soc. A* **378**,
20190230 (2020), whose two-ion version is an analogue of cosmological particle
creation.

**Reference implementation.** `tools/run_benchmark_nonadiabatic_squeezing.py`
([`benchmarks/data/nonadiabatic_squeezing/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data/nonadiabatic_squeezing)).

**Expected time.** ~11 min reading; ~2 s runtime.

**Level.** `advanced` — a specialised or research-grade surface; do the core first.

**Prerequisites.** [Tutorial 19](19_squeezing_by_quenching.md) (the `ω(t)`
squeezing engine and the covariance readout). CONVENTIONS.md **§26.4** (the
observable-only readout) and **§13/§15** (the Fock-truncation failure ladder)
govern this page.

---

## The scenario

A squeezed vacuum is built entirely out of **pairs** of phonons: `S(r)|0⟩` is a
superposition of `|0⟩, |2⟩, |4⟩, …` with the odd levels **exactly empty**. That
even-only structure is the number-basis signature of pair creation — in the two-ion
experiment it is literally the creation of correlated phonon pairs, the laboratory
analogue of particles torn from the quantum vacuum by a rapidly changing spacetime.
Here we read it off a single mode.

## Step 1 — The even-only phonon-pair signature

`phonon_number_distribution` returns `Pₙ = ⟨n|ρ|n⟩`; `pure_squeezed_vacuum_pn`
gives the analytic closed form `P_{2k} = [(2k)!/(2ᵏk!)²]·tanh²ᵏr/cosh r`, `P_odd = 0`.

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip

from iontrap_dynamics import gaussian, waveforms
from iontrap_dynamics.exceptions import ConvergenceError
from iontrap_dynamics.hamiltonians import nonadiabatic_squeezing_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.results import StorageMode
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.states import squeezed_vacuum_mode
from iontrap_dynamics.system import IonSystem

BLUE, RED, GREEN = "#1f77b4", "#d62728", "#2ca02c"
TWOPI = 2.0 * np.pi

r0 = 0.8
fock = 120  # generous: r=0.8 has a long even-n tail, needed for the closed-form match
p_n = gaussian.phonon_number_distribution(squeezed_vacuum_mode(fock, r0))
p_oracle = gaussian.pure_squeezed_vacuum_pn(r0, fock - 1)

print(f"P_0 = {p_n[0]:.4f}   P_1 = {p_n[1]:.2e}   P_2 = {p_n[2]:.4f}   P_3 = {p_n[3]:.2e}")
print(f"Σ Pₙ = {p_n.sum():.6f}   ⟨n⟩ = {(np.arange(fock) * p_n).sum():.4f}   sinh²r = {np.sinh(r0) ** 2:.4f}")

# Odd n are identically empty; the numerics match the closed form; ⟨n⟩ = sinh²r.
assert np.max(p_n[1::2]) < 1e-12
assert np.max(np.abs(p_n - p_oracle)) < 1e-12
assert abs((np.arange(fock) * p_n).sum() - np.sinh(r0) ** 2) < 1e-3

fig, ax = plt.subplots(figsize=(6.5, 4.0))
n = np.arange(14)
ax.bar(n, p_n[:14], color=[BLUE if k % 2 == 0 else RED for k in n])
ax.set_xlabel("phonon number $n$")
ax.set_ylabel(r"$P_n$")
ax.set_title(f"squeezed vacuum (r = {r0}) — even-only pair signature")
fig.tight_layout()
```

## Step 2 — Grow the pairs dynamically (parametric modulation)

Rather than a factory state, generate the pairs *dynamically* with the `ω(t)`
engine from Tutorial 19: parametric modulation at `ω_mod = 2 ω_ini` pumps pairs
into the mode, and the distribution stays even-only (the centred generator
preserves parity).

```python
def single_mode(fock, freq_hz):
    mode = ModeConfig(label="m", frequency_rad_s=TWOPI * freq_hz,
                      eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]))
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={"m": fock})


w_ini = 2.8e6
dw_mod = TWOPI * 8.0e3
hil = single_mode(fock=50, freq_hz=w_ini)
psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(50, 0))
t_mod = 120 * TWOPI / (2.0 * TWOPI * w_ini)  # 120 modulation periods
wave = waveforms.sinusoidal_modulation(
    omega_ini=TWOPI * w_ini, mod_amplitude=dw_mod, mod_frequency=2.0 * TWOPI * w_ini
)
ham = nonadiabatic_squeezing_hamiltonian(hil, "m", wave, validate_at=(0.0, t_mod))
result = solve(hilbert=hil, hamiltonian=ham, initial_state=psi0,
               times=np.linspace(0.0, t_mod, 2000), storage_mode=StorageMode.EAGER)
mode_state = gaussian.reduced_single_mode(result.states[-1], hil, "m")

readout = gaussian.gaussian_readout(mode_state)
p_dyn = gaussian.phonon_number_distribution(mode_state)
r_pred = 0.5 * dw_mod * t_mod
print(f"r = {readout.squeezing_parameter:.4f} (½δω·T = {r_pred:.4f})   "
      f"n̄_sq = {readout.mean_squeezed_occupation:.4f}   odd-n weight = {p_dyn[1::2].sum():.2e}")

# Even-only, displacement-free, and r matches the linear-growth prediction.
assert p_dyn[1::2].sum() < 1e-6
assert abs(readout.coherent_amplitude) < 1e-6
assert abs(readout.squeezing_parameter - r_pred) / r_pred < 0.05

# The dynamically-pumped distribution is the same even-only pair comb as the
# closed form for the squeezing r it reached — pairs created, not single phonons.
p_closed = gaussian.pure_squeezed_vacuum_pn(readout.squeezing_parameter, 49)
fig, ax = plt.subplots(figsize=(6.5, 4.0))
idx = np.arange(16)
ax.bar(idx - 0.18, p_dyn[:16], width=0.36, color=BLUE, label="parametric (dynamical)")
ax.bar(idx + 0.18, p_closed[:16], width=0.36, color=GREEN, label="closed form")
ax.set_xlabel("phonon number $n$")
ax.set_ylabel(r"$P_n$")
ax.set_title(f"pairs created dynamically (r = {readout.squeezing_parameter:.2f}) — still even-only")
ax.legend()
fig.tight_layout()

assert np.max(np.abs(p_dyn - p_closed)) < 5e-3  # dynamical generation matches the oracle
```

## Step 3 — Keep the readout honest: the parity-aware truncation guard

A squeezed state has a long even-`n` tail. Truncate the Fock space too tightly and
the covariance readout is silently **biased** — `r` reads low, `ν` reads as if
thermal — while the state norm stays 1. Worse, a naïve "is the top Fock level
populated?" check is fooled: in an even-dimensional space the top level is *odd*,
so it is empty for a squeezed vacuum even when the even tail is saturated.

`check_fock_truncation` closes that hole with a parity-aware edge metric and the
§13/§15 ladder — it **raises** on a badly under-resolved state instead of returning
a wrong number (`gaussian_readout` runs it for you by default).

```python
under_truncated = qutip.squeeze(20, 1.2) * qutip.basis(20, 0)  # r=1.2 needs Fock ≫ 20
p = gaussian.phonon_number_distribution(under_truncated)
print(f"top (odd) level P_19 = {p[-1]:.2e}  ← a top-level-only check would say 'converged'")
print(f"even tail P_18 = {p[-2]:.3e}          ← but the pairs are piling up at the edge")

raised = False
try:
    gaussian.check_fock_truncation(under_truncated)
except ConvergenceError as exc:
    raised = True
    print("guard raised:", str(exc).split("(")[0].strip())
assert raised  # the parity-aware guard catches what the top-level metric misses

# A generous cutoff is silent (well-resolved), and the guard has a documented escape hatch.
assert gaussian.check_fock_truncation(squeezed_vacuum_mode(60, 0.8)) == ()
```

The guard's metric makes the honesty visible: for a fixed squeezing, sweep the Fock
cutoff and watch the parity-aware tail population fall through the §13/§15 bands —
from the **raise** zone (too small) down to **silent** (generous enough).

```python
from iontrap_dynamics.conventions import FOCK_CONVERGENCE_TOLERANCE as EPS

r_demo = 1.0
focks = np.arange(14, 61, 2)
p_tail = np.array([
    gaussian.phonon_number_distribution(squeezed_vacuum_mode(int(nf), r_demo))[-2:].sum()
    for nf in focks
])

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.semilogy(focks, p_tail, "o-", color=BLUE, label="p_tail (edge window)")
ax.axhline(10 * EPS, color=RED, ls="-", label="10ε — raise above")
ax.axhline(EPS, color="#ff7f0e", ls="--", label="ε — quality warning")
ax.axhline(EPS / 10, color=GREEN, ls=":", label="ε/10 — silent below")
ax.set_xlabel("Fock dimension")
ax.set_ylabel("parity-aware tail population")
ax.set_title(f"the truncation guard's metric (squeezed vacuum, r = {r_demo})")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()

# Too-small cutoffs sit in the raise band; a generous cutoff is silent.
assert p_tail[0] > 10 * EPS
assert p_tail[-1] < EPS / 10
```

## Step 4 — The bigger picture: pairs out of the vacuum

The even-only distribution you just measured is not a curiosity. In Wittemer 2020 a
**single** ion demonstrates the mechanism; in Wittemer 2019 (*PRL* **123**, 180502)
**two** ions share a squeezed common mode, and separating them turns the phonon
pairs into **spatial entanglement** — the trapped-ion analogue of cosmological
particle creation, where a rapidly changing spacetime (here, a rapidly changing
trap frequency) tears pairs of particles out of the quantum vacuum. Read out one
ion alone and its reduced state looks **thermal**, with an effective temperature set
by the squeezing — the same signature discussed for Hawking radiation. That two-ion
entanglement + effective-temperature story is a future work-package (it consumes the
Gaussian-toolbox card's multimode machinery); this single-mode readout is its
foundation.

## What you built

You read squeezing out as a phonon-number distribution and saw the even-only
phonon-**pair** signature, matched it to the analytic closed form, grew the pairs
dynamically with parametric modulation, and let the parity-aware truncation guard
protect you from a silently-biased readout. Together with
[Tutorial 19](19_squeezing_by_quenching.md) you now have the full single-ion
non-adiabatic-squeezing toolkit — engine, phase-space readout, and number-basis
readout.
