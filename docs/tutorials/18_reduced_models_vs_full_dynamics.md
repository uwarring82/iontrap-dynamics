# Tutorial 18 — Reduced models vs full dynamics

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/18_reduced_models_vs_full_dynamics.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** By the end you will have confirmed that the Jaynes–Cummings and anti-Jaynes–Cummings models are the same model up to a label, watched that label become a physical knob on the ion, measured the rotating-wave approximation breaking down with `model_deviation`, and tied the reduced-model coupling back to the apparatus sideband rate through the `2g√(n±1)` limit.

**Reference implementation.** [`tools/plot_reduced_models_comparison.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/plot_reduced_models_comparison.py) regenerates the four-panel figure below and writes [`benchmarks/data/reduced_models_comparison/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data/reduced_models_comparison) (`report.json` + `arrays.npz` + `plot.png`); its `max_numerical_vs_analytic_error` is the oracle bar.

**Expected time.** ~12 min reading; ~5 s runtime.

**Level.** `advanced` — a specialised or research-grade surface; do the core first.

**Prerequisites.** [Tutorial 8](08_full_lamb_dicke.md) (full Lamb–Dicke sidebands) and [Tutorial 2](02_red_sideband_fock1.md) (red-sideband flopping). The reduced models are governed by `CONVENTIONS.md` §25 (Schrödinger-picture bare-term Hamiltonians, the LOCK-3 identity, the `ω₀` effective-sign semantics) and the §5 scope note that exempts them from the §5 interaction-picture mandate. This is the first tutorial to use the physics-layer `reduced_models` module (`jaynes_cummings_hamiltonian`, `anti_jaynes_cummings_hamiltonian`, `quantum_rabi_hamiltonian`, `model_deviation`) and to compare it, rung by rung, against the apparatus sideband builders.

---

## The scenario

A reduced light–matter model is a piece of *physics* — what a trapped ion approximates — while a sideband Hamiltonian is a piece of *apparatus* — how a real ion realises it. Keeping those two layers apart is the whole point: it lets us ask whether the reduction is faithful and, where it is not, measure by how much it fails. The [model-hierarchy note](../models-hierarchy.md) (now vendored; cited by section below) lays out four falsifiable cases, and this tutorial walks all four against the shipped library. They span the claim "the Jaynes–Cummings and anti-Jaynes–Cummings models are merely relabelled" from **true in isolation** (Case A), to **a physical knob** (Case B), to **false under strong coupling** (Case C), with the Lamb–Dicke reduction itself put to the test in Case D.

![Four panels. Panel A: the Jaynes-Cummings spectrum at minus omega-zero and the anti-Jaynes-Cummings spectrum at plus omega-zero lie exactly on top of each other. Panel B: from spin-down with no phonons, the spin-up population stays flat at zero under the red sideband but oscillates fully between zero and one under the blue sideband. Panel C: as the coupling g over omega-zero grows, the Jaynes-Cummings to quantum Rabi ground-energy deviation and the quantum Rabi ground-state phonon number both rise from near zero. Panel D: the relative deviation between the full-Lamb-Dicke and leading-order red-sideband rate rises with the confinement parameter eta squared times two n plus one, crossing the deep, intermediate, and beyond bands.](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/reduced_models_comparison/plot.png)

Each step below **prints its key numbers and redraws its panel of this figure live**: in the [Colab notebook](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/18_reduced_models_vs_full_dynamics.ipynb) you watch the calculation produce the result instead of trusting a static image. The `assert`s stay the oracle — a step that runs and renders without error is a step that passed.

## Step 1 — Case A: "only a label" is true (LOCK-3)

In isolation the anti-Jaynes–Cummings model is the Jaynes–Cummings model conjugated by `σ_x` with the qubit splitting flipped — the LOCK-3 identity `H_AJC(ω₀) = σ_x H_JC(−ω₀) σ_x` (`jaynes_cummings_hamiltonian`, `anti_jaynes_cummings_hamiltonian`). A unitary conjugation cannot move eigenvalues, so the two spectra coincide. This is the [model-hierarchy note](../models-hierarchy.md)'s §6 (LOCK-3) / §8 regime 1.

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem
from iontrap_dynamics.reduced_models import (
    anti_jaynes_cummings_hamiltonian,
    jaynes_cummings_hamiltonian,
    model_deviation,
    quantum_rabi_hamiltonian,
)
from iontrap_dynamics.spectrum import solve_spectrum

OMEGA = 2 * np.pi * 1.0e6  # ω₀ = ω_f, the resonant model scale (rad·s⁻¹)
FOCK = 30

# House colours — match the four-panel reference figure above.
BLUE, RED, GREEN, PURPLE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#444444"


def single_mode(fock_dim: int) -> HilbertSpace:
    """One ion (the qubit) plus one axial motional mode ``m``."""
    mode = ModeConfig(label="m", frequency_rad_s=OMEGA, eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]))
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={"m": fock_dim})


h = single_mode(FOCK)
g = 0.4 * OMEGA
ajc = anti_jaynes_cummings_hamiltonian(h, "m", ion_index=0, omega_0=OMEGA, omega_f=OMEGA, g=g)
jc_negative = jaynes_cummings_hamiltonian(h, "m", ion_index=0, omega_0=-OMEGA, omega_f=OMEGA, g=g)

# The σ_x conjugation is unitary, so the two spectra coincide (LOCK-3).
eig_ajc = solve_spectrum(ajc).eigenvalues[:24] / OMEGA  # lowest 24, in units of ω₀
eig_jc = solve_spectrum(jc_negative).eigenvalues[:24] / OMEGA
max_gap = float(np.max(np.abs(eig_ajc - eig_jc)))
print(f"Case A — largest |E_AJC(+ω₀) − E_JC(−ω₀)| / ω₀ = {max_gap:.2e}")
assert max_gap < 1e-6  # spec(H_AJC(ω₀)) = spec(H_JC(−ω₀))

idx = np.arange(eig_jc.size)
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(idx, eig_jc, color=GREY, linewidth=1.0, label=r"JC$(-\omega_0)$")
ax.scatter(idx, eig_ajc, color=RED, s=18, zorder=3, label=r"AJC$(+\omega_0)$")
ax.set_xlabel("eigenvalue index")
ax.set_ylabel(r"$E / \omega_0$")
ax.set_title("A · LOCK-3: spectra coincide")
ax.legend(frameon=False)
plt.show()
```

!!! note "The −ω₀ is a model sign, not a negative ion"
    The identity needs the qubit splitting *flipped* (`omega_0=-OMEGA` on the JC side). That negative `ω₀` is an effective/model parameter — the relabelling that turns a red-sideband image into a blue-sideband image — **not** a physically negative ion splitting. For a real ion `ω₀ > 0` and `(ω₀/2)σ_z` keeps `|↑⟩` above `|↓⟩` (CONVENTIONS.md §3, §25.2).

## Step 2 — Case B: the label becomes a knob

On a real ion the same two couplings are selected by *which sideband you drive*. From `|↓,0⟩` the **red** sideband (the JC image) couples `|↓,n⟩ ↔ |↑,n−1⟩`, and since `â|0⟩ = 0` the ground state is **dark** — nothing flops. The **blue** sideband (the AJC image) couples `|↓,0⟩ ↔ |↑,1⟩`, so the ion is **bright** and flops at the blue-sideband Rabi rate (`red_sideband_hamiltonian`, `blue_sideband_hamiltonian`, `blue_sideband_rabi_frequency`). This is the note's §8 regime 2 — the crispest red ≠ blue anchor.

```python
from iontrap_dynamics.analytic import blue_sideband_rabi_frequency, lamb_dicke_parameter
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import blue_sideband_hamiltonian, red_sideband_hamiltonian
from iontrap_dynamics.observables import Observable
from iontrap_dynamics.operators import spin_down, spin_up
from iontrap_dynamics.sequences import solve

k_vector = np.array([0.0, 0.0, 8.0e6])  # along the axial mode → η ≈ 0.11
rabi = 2 * np.pi * 50.0e3
hb = single_mode(12)
drive = DriveConfig(k_vector_m_inv=k_vector, carrier_rabi_frequency_rad_s=rabi, phase_rad=0.0)
eta = lamb_dicke_parameter(
    k_vec=k_vector, mode_eigenvector=np.array([0.0, 0.0, 1.0]),
    ion_mass=mg25_plus().mass_kg, mode_frequency=OMEGA,
)
blue_rate = blue_sideband_rabi_frequency(carrier_rabi_frequency=rabi, lamb_dicke_parameter=eta, n_initial=0)

psi0 = qutip.tensor(spin_down(), qutip.basis(12, 0))
times = np.linspace(0.0, 2 * np.pi / blue_rate, 80)
p_up = Observable(label="p_up", operator=hb.spin_op_for_ion(spin_up() * spin_up().dag(), 0))
pop_red = np.asarray(
    solve(hilbert=hb, hamiltonian=red_sideband_hamiltonian(hb, drive, "m", ion_index=0),
          initial_state=psi0, times=times, observables=(p_up,)).expectations["p_up"],
    dtype=float,
)
pop_blue = np.asarray(
    solve(hilbert=hb, hamiltonian=blue_sideband_hamiltonian(hb, drive, "m", ion_index=0),
          initial_state=psi0, times=times, observables=(p_up,)).expectations["p_up"],
    dtype=float,
)
analytic_blue = np.sin(blue_rate * times / 2.0) ** 2  # ideal two-level flop at Ω_b

print(f"Case B — η = {eta:.3f};  blue max ⟨P↑⟩ = {pop_blue.max():.3f} (bright),  "
      f"red max ⟨P↑⟩ = {pop_red.max():.1e} (dark)")
assert np.max(pop_red) < 1e-4   # |↓,0⟩ is DARK under the red sideband (→ JC, a|0⟩ = 0)
assert np.max(pop_blue) > 0.99  # |↓,0⟩ is BRIGHT under the blue sideband (→ AJC, |↓,0⟩↔|↑,1⟩)

tau = times * blue_rate / (2.0 * np.pi)  # phase in units of the blue flop period
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(tau, analytic_blue, color=GREY, linewidth=1.0, label=r"$\sin^2(\Omega_b t/2)$")
ax.scatter(tau, pop_blue, color=BLUE, s=14, zorder=3, label="blue → AJC (bright)")
ax.scatter(tau, pop_red, color=RED, marker="s", s=14, zorder=3, label="red → JC (dark)")
ax.set_xlabel(r"blue flop phase $\Omega_b t / 2\pi$")
ax.set_ylabel(r"spin-up population $\langle P_\uparrow\rangle$")
ax.set_title(r"B · $|{\downarrow},0\rangle$ dark vs bright")
ax.legend(frameon=False)
plt.show()
```

!!! tip "Dark is not the same as decoupled"
    `|↓,0⟩` is dark under the *red* sideband only because it sits at the bottom of the ladder. `|↓,1⟩` flops freely under the red sideband — the darkness is a property of the state, not of the JC coupling.

## Step 3 — Case C: "only a label" is false (RWA breakdown)

The Jaynes–Cummings model drops the counter-rotating terms the quantum Rabi model keeps. As the dimensionless coupling `g/ω₀` grows, those terms matter: the QRM ground state stops being the dark state `|↓,0⟩` and acquires virtual phonons, and the two trajectories part company. `model_deviation` measures the parting as a worst-case state infidelity — small in the common (weak-coupling) regime, large at strong coupling ([model-hierarchy note](../models-hierarchy.md) §4 / §8 regime 3). No terms are dropped by hand; the agreement *emerges* from the `g/ω₀ → 0` limit.

```python
from iontrap_dynamics.results import StorageMode


def qrm_ground_phonons(coupling: float) -> float:
    qrm = quantum_rabi_hamiltonian(h, "m", ion_index=0, omega_0=OMEGA, omega_f=OMEGA, g=coupling)
    ground = solve_spectrum(qrm).eigenvectors[:, 0]
    number = h.number_for_mode("m").full()
    return float(np.real(ground.conj() @ (number @ ground)))


assert qrm_ground_phonons(0.05 * OMEGA) < 1e-2  # weak: ground ≈ |↓,0⟩, ⟨a†a⟩ ≈ (g/2ω₀)²
assert qrm_ground_phonons(1.0 * OMEGA) > 0.1    # ultra-strong: real virtual phonons


def trajectory(builder, coupling: float):
    hamiltonian = builder(h, "m", ion_index=0, omega_0=OMEGA, omega_f=OMEGA, g=coupling)
    psi1 = qutip.tensor(spin_down(), qutip.basis(FOCK, 1))
    times_c = np.linspace(0.0, 4.0e-6, 60)
    return solve(hilbert=h, hamiltonian=hamiltonian, initial_state=psi1,
                 times=times_c, storage_mode=StorageMode.EAGER)


weak = model_deviation(trajectory(jaynes_cummings_hamiltonian, 0.05 * OMEGA),
                       trajectory(quantum_rabi_hamiltonian, 0.05 * OMEGA))
strong = model_deviation(trajectory(jaynes_cummings_hamiltonian, 0.5 * OMEGA),
                         trajectory(quantum_rabi_hamiltonian, 0.5 * OMEGA))

assert weak.method == "state_fidelity"  # materialised states → 1 − qutip.fidelity per step
assert weak.value < 1e-2                # common regime: JC ≈ QRM
assert strong.value > 10 * weak.value   # breakdown: the counter-rotating terms separate them

# Sweep the dimensionless coupling g/ω₀ from weak to ultra-strong, tracking two
# probes: the JC↔QRM ground-energy gap, and the QRM virtual-phonon number.
g_over_w0 = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5])
e0_jc, e0_qrm, n_qrm = [], [], []
for ratio in g_over_w0:
    coupling = ratio * OMEGA
    jc = jaynes_cummings_hamiltonian(h, "m", ion_index=0, omega_0=OMEGA, omega_f=OMEGA, g=coupling)
    qrm = quantum_rabi_hamiltonian(h, "m", ion_index=0, omega_0=OMEGA, omega_f=OMEGA, g=coupling)
    e0_jc.append(solve_spectrum(jc).eigenvalues[0] / OMEGA)
    e0_qrm.append(solve_spectrum(qrm).eigenvalues[0] / OMEGA)
    n_qrm.append(qrm_ground_phonons(coupling))
deviation = np.abs(np.array(e0_jc) - np.array(e0_qrm))
n_qrm = np.array(n_qrm)

print(f"Case C — trajectory infidelity: weak {weak.value:.2e}  →  strong {strong.value:.2e}")
print(f"Case C — at g/ω₀ = 1.5: energy gap {deviation[-1]:.3f} ω₀,  QRM ⟨a†a⟩ = {n_qrm[-1]:.3f}")

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(g_over_w0, deviation, color=PURPLE, marker="o", markersize=4,
        label=r"$|E_0^{\mathrm{JC}}-E_0^{\mathrm{QRM}}| / \omega_0$")
ax.plot(g_over_w0, n_qrm, color=GREEN, marker="s", markersize=4,
        label=r"QRM $\langle a^\dagger a\rangle$")
ax.set_xscale("log")
ax.set_xlabel(r"$g / \omega_0$")
ax.set_ylabel(r"deviation,  $\langle a^\dagger a\rangle$")
ax.set_title("C · RWA breakdown")
ax.legend(frameon=False)
plt.show()
```

!!! warning "Fock truncation is the trap in the strong-coupling panel"
    The QRM ground state spreads over many phonons at `g/ω₀ ≈ 1`, so a small `fock_truncations` silently under-resolves it and `solve` will raise the §13 convergence error. The benchmark uses `FOCK = 30`; do not shrink it when you push `g/ω₀` past `~0.5`.

## Step 4 — Case D: tying the reduction back to the apparatus

The reduced-model coupling `g` is not free — it is the Lamb–Dicke image of the apparatus drive, `g = ηΩ/2`. In that limit the blue-sideband rate is exactly twice the reduced AJC block element: `Ω|η|√(n+1) = 2g√(n+1)` (`blue_sideband_rabi_frequency`). Beyond the deep regime the all-orders Debye–Waller × Laguerre structure bends the rate away from that line, tracked by the confinement parameter `η²(2n+1)` and the `lamb_dicke_regime` classifier ([model-hierarchy note](../models-hierarchy.md) §5, nonlinear branch).

```python
from iontrap_dynamics.analytic import (
    lamb_dicke_confinement,
    lamb_dicke_regime,
    red_sideband_rabi_frequency,
    red_sideband_rabi_frequency_full_ld,
)

drive_rabi = 2 * np.pi * 1.0e6
eta_d = 0.1
g_effective = eta_d * drive_rabi / 2  # the reduction g = ηΩ/2
for n in (0, 1, 2, 3):
    rate = blue_sideband_rabi_frequency(carrier_rabi_frequency=drive_rabi, lamb_dicke_parameter=eta_d, n_initial=n)
    assert np.isclose(rate, 2 * g_effective * np.sqrt(n + 1))  # Ω|η|√(n+1) = 2g√(n+1)

# Deep regime: the all-orders rate matches leading order to the zero-point factor.
assert lamb_dicke_regime(lamb_dicke_parameter=0.05, mean_phonon_number=1) == "deep"
assert lamb_dicke_confinement(lamb_dicke_parameter=0.05, mean_phonon_number=1) < 0.1
full = red_sideband_rabi_frequency_full_ld(carrier_rabi_frequency=drive_rabi, lamb_dicke_parameter=0.05, n_initial=1)
leading = red_sideband_rabi_frequency(carrier_rabi_frequency=drive_rabi, lamb_dicke_parameter=0.05, n_initial=1)
assert abs(full - leading) / leading < 1e-2  # full-LD ≈ leading order deep in the regime

# Sweep η at n = 1, so the confinement η²(2n+1) = 3η² runs deep → beyond. The
# all-orders full-LD rate bends away from the leading-order 2g√n line as it grows.
eta_grid = np.array([0.02, 0.06, 0.12, 0.2, 0.35, 0.6, 1.0])
confinement = np.array([lamb_dicke_confinement(lamb_dicke_parameter=e, mean_phonon_number=1) for e in eta_grid])
leading_d = np.array([red_sideband_rabi_frequency(carrier_rabi_frequency=drive_rabi, lamb_dicke_parameter=e, n_initial=1) for e in eta_grid])
full_d = np.array([red_sideband_rabi_frequency_full_ld(carrier_rabi_frequency=drive_rabi, lamb_dicke_parameter=e, n_initial=1) for e in eta_grid])
rel_dev = np.abs(full_d - leading_d) / leading_d
regimes = [lamb_dicke_regime(lamb_dicke_parameter=e, mean_phonon_number=1) for e in eta_grid]

print("Case D —   η    η²(2n+1)   regime        full-LD vs leading")
for e, c, r, d in zip(eta_grid, confinement, regimes, rel_dev):
    print(f"          {e:4.2f}   {c:7.3f}   {r:<12s}  {d:.2e}")

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.axvspan(1e-4, 0.1, color=GREEN, alpha=0.08)
ax.axvspan(0.1, 1.0, color=BLUE, alpha=0.08)
ax.axvspan(1.0, 1e2, color=RED, alpha=0.08)
ax.plot(confinement, rel_dev, color=GREY, marker="o", markersize=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\eta^2(2n+1)$")
ax.set_ylabel("full-LD vs leading rel. dev.")
ax.set_title("D · deep → intermediate → beyond")
plt.show()
```

!!! note "Signed matrix element vs the helper magnitude"
    The full-Lamb–Dicke matrix element `Ω e^{−η²/2} η L_n^{(1)}(η²)/√(n+1)` is *signed* — the Laguerre polynomial changes sign and nulls. The library helpers return the rate **magnitude** (`|·|`), so a vanishing `red_sideband_rabi_frequency_full_ld` marks a Laguerre null, not a sign you can read off the rate alone.

## The analytical picture

The three reduced Hamiltonians (`H/ℏ`, rad·s⁻¹, §25.1) on one qubit ⊗ one mode, with `σ_z = |↑⟩⟨↑| − |↓⟩⟨↓|`, `σ_+ = |↑⟩⟨↓|`, `â` the mode annihilation operator:

- **Quantum Rabi:** `H_QRM = (ω₀/2) σ_z + ω_f â†â + g σ_x(â + â†)` — full dipole, non-RWA.
- **Jaynes–Cummings:** `H_JC = (ω₀/2) σ_z + ω_f â†â + g(â σ_+ + â† σ_−)` — co-rotating.
- **Anti-Jaynes–Cummings:** `H_AJC = (ω₀/2) σ_z + ω_f â†â + g(â† σ_+ + â σ_−)` — counter-rotating.

**Symmetry contrast.** JC conserves the U(1) excitation number `â†â + σ_+σ_−`; AJC conserves the difference number `â†â − σ_+σ_−`; the QRM conserves only the Z₂ parity `Π = σ_z(−1)^{â†â}` (excitation number not conserved). **LOCK-3:** `H_AJC(ω₀) = σ_x H_JC(−ω₀) σ_x`, using `σ_x σ_z σ_x = −σ_z` and `σ_x σ_± σ_x = σ_∓`; the `−ω₀` is an effective/model sign (Step 1 caveat).

**Ion → sideband.** The schematic full-ion drive `H = (ω_at/2) σ_z + ν â†â + Ω σ_x cos(η(â + â†) − ω_L t + φ)`, after the optical RWA, yields first-order sideband terms `∝ σ_+ â e^{−i(δ+ν)t}` and `∝ σ_+ â† e^{−i(δ−ν)t}` — the red tone selects JC, the blue tone selects AJC. The all-orders matrix elements are `Ω_{n,n+1} = Ω e^{−η²/2} η L_n^{(1)}(η²)/√(n+1)` (blue) and `Ω_{n,n−1} = Ω e^{−η²/2} η L_{n−1}^{(1)}(η²)/√n` (red, zero at `n = 0`), reducing in the Lamb–Dicke limit to `Ω|η|√(n+1) = 2g√(n+1)` and `Ω|η|√n = 2g√n` with `g = ηΩ/2`. The visible regime knob throughout is `η²(2n+1)` (Fock `n`) or `η²(2n̄+1)` (thermal), the argument of `lamb_dicke_confinement` and the deep/intermediate/beyond classifier.

**Deferred — Case E (bichromatic simulated QRM).** Driving both sidebands at once can *simulate* a quantum Rabi model on the ion, with the schematic retained interaction `H_I ≃ g(σ_+ â e^{−iΔ_r t} + σ_+ â† e^{−iΔ_b t} + h.c.)` and detunings `δ_r = −ν + Δ_r`, `δ_b = +ν + Δ_b`. The effective-parameter map (`ω₀^eff`, `ω_f^eff`) is **not** committed here — it requires a first-class two-tone sideband builder and a derived convention under the shipped detuned-sideband signs, which is future work ([model-hierarchy note](../models-hierarchy.md) §5 laboratory branch).

## Where to next

- [Tutorial 8 — Full Lamb–Dicke for hot-ion regimes](08_full_lamb_dicke.md) — the apparatus side of Case D, where the all-orders Debye–Waller × Laguerre comb is the headline.
- [Tutorial 16 — Two-mode SU(1,1) squeezing](16_two_mode_squeezing.md) — another "build it twice and make the two agree" comparison, on the motional side.
- Regenerate the figure and the oracle report with [`tools/plot_reduced_models_comparison.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/plot_reduced_models_comparison.py); the closed-form anchors live in `tests/regression/analytic/test_reduced_models_oracles.py`.

---

## Licence

Sail material — adaptive guidance with specific parameter choices, not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per [`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
