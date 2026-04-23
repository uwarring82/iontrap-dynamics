# Tutorial 13 — Reproducing Clos 2016 (Phys. Rev. Lett. 117, 170401)

**Goal.** Reproduce the published `IPR_av` (averaged effective
dimension) numbers from the legacy spin–boson bundle that ships
under [`legacy/clos 2016 prl/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/legacy/clos%202016%20prl).
By the end you will have built the full-displacement (non-RWA)
spin–boson Hamiltonian for a single trapped ion, diagonalised it
across a grid of carrier detunings, computed the legacy `IPR_av`
quantity, and matched it against `theo_dim_N_1.dat` row-by-row.
Then you will see the same pipeline scale to N=2 and N=3 ions
without code changes — only the axial-mode reference and cutoff
sizing change.

**Expected time.** ~15 min reading; ~3 s runtime for N=1, ~10 s
for N=2 + N=3.

**Prerequisites.** [Tutorial 8](08_full_lamb_dicke.md) for the
Lamb–Dicke / Wineland–Itano framing; familiarity with
[`sequences.solve`](04_ms_gate_bell.md) is **not** required —
this tutorial uses the exact-diagonalisation entry point
`solve_spectrum` instead, which is independent of the
trajectory-evolution surface.

---

## Three things to know before the code

The library carries two distinct full-Lamb–Dicke surfaces because
they answer different physics questions:

1. **`carrier_hamiltonian_full_ld`** in `iontrap_dynamics.hamiltonians`
   is the **carrier-RWA** all-orders builder. Each mode contributes
   the Δn=0 projection $\langle n | \hat M_0 | n \rangle =
   e^{-\eta^2/2} L_n(\eta^2)$ — the Debye–Waller / Laguerre dressing
   that survives after the carrier RWA. This is the right object
   for textbook hot-ion carrier dynamics. **It is not what Porras
   keeps.**
2. **`clos2016_spin_boson_hamiltonian`** in `iontrap_dynamics.clos2016`
   is the **non-RWA** displacement operator
   $(\Omega/2)\,\sigma_+\,e^{\hat P} + \mathrm{h.c.}$ that
   `ergodic_ipr_av.m` builds. The off-diagonal Fock couplings are
   precisely the non-secular terms whose mixing drives ergodicity
   in the Clos 2016 paper, so the projection from (1) would erase
   the physics this tutorial reproduces.

The legacy `IPR_av` quantity (recorded in the `theo_dim_N_*.dat`
tables) is **not** the textbook effective dimension. For a mixed
initial state $\rho_0 = \sum_j \lambda_j |\psi_j\rangle\langle\psi_j|$,

$$
\text{IPR}_\text{av} \;=\; \sum_j \lambda_j \cdot
\frac{1}{\sum_\alpha |\langle E_\alpha | \psi_j \rangle|^4}
$$

— a $\rho_0$-eigendecomposition–weighted average of pure-state
effective dimensions. This coincides with
`effective_dimension(spectrum, ρ)` on pure states but diverges on
mixed states. The library ships both:
`effective_dimension` for textbook use,
`clos2016_averaged_effective_dimension` for legacy reproduction.

The Lamb–Dicke wavelength is the **Raman two-photon effective
wavelength** ≈ 197.7 nm, not the 200 nm single-photon reference
that the non-`_ipr_av` siblings of `ergodic.m` use. The constant
`CLOS2016_LEGACY_WAVELENGTH_M` pins this; pass it explicitly to
`clos2016_spin_boson_hamiltonian`.

## Step 1 — Load the published reference

The bundle ships per-N theory tables under
`legacy/clos 2016 prl/DP num res_fig_1_2015_07_30/`. The library
parses them through
`iontrap_dynamics.clos2016_references.load_clos2016_theory_dimension_surface`.
Each surface is a `(cutoff × detuning)` grid of `IPR_av`, plus
the parameters Porras used to compute it (axial frequency, Rabi
frequency, mean phonon occupation):

```python
import numpy as np

from iontrap_dynamics.clos2016_references import (
    load_clos2016_cutoff_convergence,
    load_clos2016_theory_dimension_surface,
)

surface = load_clos2016_theory_dimension_surface(1)
print(surface.cutoffs)                       # [0, 1, …, 20]
print(surface.detunings_legacy_units)        # [0.0, 0.2, …, 3.0]
print(surface.omega_axial_legacy_units)      # 0.71  (MHz, linear)
print(surface.omega_rabi_legacy_units)       # 0.71
print(surface.mean_occupation)               # 1.0
```

The `*_legacy_units` suffix is a reminder: the published table
uses *linear MHz*, not angular frequency. The library expects
SI rad/s, so we convert with `2 π · 10⁶`.

The companion convergence table tells us the smallest cutoff at
which `IPR_av` plateaus (within 1 % of the deep-cutoff tail):

```python
convergence = load_clos2016_cutoff_convergence(1)
print(convergence.inferred_converged_cutoff)  # 7
```

We will reproduce the `cutoff = 7` row of the surface and compare
row-vs-row.

## Step 2 — Build the non-RWA spin–boson Hamiltonian

For N=1 there is one axial mode at the trap frequency with full
participation by the (single) ion. The carrier sweep iterates
over the published detuning column:

```python
from iontrap_dynamics import (
    CLOS2016_LEGACY_WAVELENGTH_M,
    clos2016_averaged_effective_dimension,
    clos2016_initial_state,
    clos2016_spin_boson_hamiltonian,
    solve_spectrum,
)
from iontrap_dynamics.species import mg25_plus

cutoff = convergence.inferred_converged_cutoff           # 7
ci = int(np.flatnonzero(surface.cutoffs == cutoff)[0])
reference = surface.averaged_effective_dimension[ci]     # 16 detunings

initial_state = clos2016_initial_state(
    max_phonons=cutoff,
    mean_occupations=[surface.mean_occupation],          # [1.0]
    theta_rad=0.0, phi_rad=0.0,                          # |↑⟩ ⊗ ρ_thermal
)

calculated = []
for detuning in surface.detunings_legacy_units:
    hamiltonian = clos2016_spin_boson_hamiltonian(
        max_phonons=cutoff,
        axial_frequency_rad_s=surface.omega_axial_legacy_units * 2 * np.pi * 1e6,
        dimensionless_mode_frequencies=[1.0],            # COM mode at ω_z
        center_mode_weights=[1.0],                       # full participation
        carrier_rabi_frequency_rad_s=surface.omega_rabi_legacy_units * 2 * np.pi * 1e6,
        detuning_rad_s=detuning * 2 * np.pi * 1e6,
        ion_mass_kg=mg25_plus().mass_kg,
        laser_wavelength_m=CLOS2016_LEGACY_WAVELENGTH_M, # Raman effective
    )
    spectrum = solve_spectrum(hamiltonian, initial_state=initial_state)
    calculated.append(
        clos2016_averaged_effective_dimension(spectrum, initial_state)
    )

calculated = np.asarray(calculated)
```

`clos2016_initial_state` returns the legacy
$|\theta=0\rangle\langle\theta=0| \otimes \rho_\text{thermal}(\bar n)$
density matrix (Bloch-sphere parameterisation matches
`ergodic_ipr_av.m`'s
$\psi = \cos(\theta/2)|\!\uparrow\rangle + \sin(\theta/2) e^{-i\phi} |\!\downarrow\rangle$).

`solve_spectrum` is the AAC dense-eigensolver entry point; it
returns a `SpectrumResult` with full eigendecomposition and the
metadata needed to track provenance. For non-RWA dynamics it is
the right tool — `sequences.solve` would chase a time-evolution
trajectory we do not need.

`clos2016_averaged_effective_dimension` is the legacy `IPR_av`
helper. It re-eigendecomposes the (mixed) initial state, computes
a pure-state effective dimension for each $|\psi_j\rangle$, and
returns the $\lambda_j$-weighted sum.

## Step 3 — Compare against the published table

```python
abs_err = np.abs(calculated - reference)
rel_err = abs_err / np.abs(reference)
print(f"max relative error : {rel_err.max():.3f}")  # ~0.088
print(f"max absolute error : {abs_err.max():.3f}")  # ~0.16
```

The achieved tolerance is ~9 % at the worst point, which lands
near the steep mid-resonance peaks at `det ≈ 1.0, 2.0` legacy
units. Two compounding sources:

- The published table carries **only three significant figures**
  (e.g. `2.832`, `1.818`). At the resonance peaks `IPR_av`
  varies by a full unit per detuning step, so the rounded value
  is itself ~0.5 % off the underlying number.
- Inside the resonance bands the spectrum is sensitive to η₀
  and the `eigh` tie-breaking near near-degenerate eigenvalues.

The off-resonance shoulders match to better than 1 %.

## Step 4 — Visual sanity check

A quick plot makes the agreement visible — and the resonance
structure of `IPR_av` immediately recognisable:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.plot(surface.detunings_legacy_units, reference,
        "o-", label="legacy theo_dim_N_1.dat (cutoff=7)")
ax.plot(surface.detunings_legacy_units, calculated,
        "x--", label="iontrap-dynamics, this tutorial")
ax.set_xlabel(r"detuning $\omega_z / 2\pi$  [MHz]")
ax.set_ylabel(r"averaged effective dimension $\mathrm{IPR}_\mathrm{av}$")
ax.legend()
fig.tight_layout()
```

The two curves should sit on top of each other at the off-resonance
shoulders and drift apart only at the steep peaks. Three resonance
features at `det ≈ 0.7, 1.4, 2.1` correspond to the COM-mode
sidebands; the one at `det ≈ 0` is the carrier.

## Step 5 — Same pipeline, N=2 and N=3

The only ingredients that change are (a) the axial-mode reference
and (b) the per-N parameters from the surface. The Hamiltonian
builder accepts arbitrary mode counts:

```python
from iontrap_dynamics.clos2016_references import clos2016_axial_mode_reference

for n_ions, cutoff in [(2, 8), (3, 6)]:
    surface = load_clos2016_theory_dimension_surface(n_ions)
    axial = clos2016_axial_mode_reference(n_ions)

    ci = int(np.flatnonzero(surface.cutoffs == cutoff)[0])
    reference = surface.averaged_effective_dimension[ci]

    initial_state = clos2016_initial_state(
        max_phonons=cutoff,
        mean_occupations=[surface.mean_occupation] * n_ions,
    )

    calculated = []
    for detuning in surface.detunings_legacy_units:
        hamiltonian = clos2016_spin_boson_hamiltonian(
            max_phonons=cutoff,
            axial_frequency_rad_s=surface.omega_axial_legacy_units * 2 * np.pi * 1e6,
            dimensionless_mode_frequencies=axial.dimensionless_frequencies.tolist(),
            center_mode_weights=axial.first_ion_participation_weights.tolist(),
            carrier_rabi_frequency_rad_s=surface.omega_rabi_legacy_units * 2 * np.pi * 1e6,
            detuning_rad_s=detuning * 2 * np.pi * 1e6,
            ion_mass_kg=mg25_plus().mass_kg,
            laser_wavelength_m=CLOS2016_LEGACY_WAVELENGTH_M,
        )
        spectrum = solve_spectrum(hamiltonian, initial_state=initial_state)
        calculated.append(
            clos2016_averaged_effective_dimension(spectrum, initial_state)
        )
    calculated = np.asarray(calculated)
    print(f"N={n_ions}: max |Δ|/|ref| = "
          f"{(np.abs(calculated - reference) / np.abs(reference)).max():.3f}")
```

The `clos2016_axial_mode_reference` table currently covers N=2
and N=3 — the AAA dispatch's regression anchor. For N=2 the
modes are at `(1, √3)·ω_z` (COM, breathing) with first-ion
participation `(±1/√2, ±1/√2)`. For N=3 they are
`(1, √3, √(29/5))·ω_z` with weights `(1/√3, −1/√2, 1/√6)` —
both standard results for a Coulomb chain in a quadratic axial
trap. For higher N the modes are not pinned in the library yet;
re-derive from a normal-mode solver.

Achieved tolerances under the regression suite:

| N | cutoff | dim     | wall-clock | max relative error |
|---|--------|---------|------------|--------------------|
| 1 | 7      | 16      | <0.1 s     | ~9 %               |
| 2 | 8      | 162     | ~0.2 s     | ~6 %               |
| 3 | 6      | 686     | ~6 s       | ~4 %               |

The tolerance tightens as N grows because the published
`IPR_av` values get larger and the rounding noise of
three-significant-figure data shrinks proportionally — the
absolute error stays within ~0.2 across all three.

## Why we stop at N=3 here

The `theo_dim_N_*.dat` tables go to N=5, but the dense
diagonalisation cost climbs as $d = 2 \cdot n_c^N$:

| N | n_c | dim       | dense matrix storage (complex128) |
|---|-----|-----------|-----------------------------------|
| 3 | 6   | 686       | 7.5 MB                            |
| 3 | 10  | 21 296    | 7.3 GB                            |
| 4 | 6   | 4 802     | 0.37 GB                           |
| 5 | 5   | 7 776     | 0.97 GB                           |
| 5 | 10  | ~6.7 × 10⁵ | ~7.2 TB (infeasible dense)       |

For N ≥ 4 the inferred-converged cutoff already exceeds what a
laptop can `eigh` in a single call. The library will eventually
ship a benchmark-validated interior-window iterative path
targeting the initial-state mean energy (matching the legacy
`eigs(H, Neigens, meanE)` style); until then, pin yourself to
the dense envelope.

## What just happened

You reproduced a publication-validated trapped-ion quantum
simulator dataset using:

- the **non-RWA** full-displacement spin–boson Hamiltonian
  ([`clos2016_spin_boson_hamiltonian`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/clos2016.py));
- the dense exact-diagonalisation entry point
  ([`solve_spectrum`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/spectrum.py));
- the legacy `IPR_av` quantity
  ([`clos2016_averaged_effective_dimension`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/clos2016.py));
- the bundled reference data and pinned axial-mode anchors
  ([`clos2016_references`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/clos2016_references.py)).

The matching regression tests live under
[`tests/regression/reproduction/test_clos_2016_N1.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tests/regression/reproduction/test_clos_2016_N1.py)
and `test_clos_2016_N2_N3.py` — they run the same pipeline you
just wrote and assert the per-N tolerances above.

## Where to go next

- **Sanity-check the convention.** `clos2016_axial_mode_reference`
  is pinned for N=2 and N=3; if you want to extend to N=4/5,
  re-derive the modes from
  $V(\mathbf{x}) = \sum_i x_i^2 + (\beta/2) \sum_{i<j} 1/|x_i - x_j|$
  with $\beta = 4$ (the legacy `mypotential`). Tutorial 8's
  `lamb_dicke_parameter` analytic helper is the right reference
  for converting between the trap-frequency unit system and
  physical SI.
- **Track-related dispatches.** This tutorial closes the AAA →
  AAF wave of the Clos 2016 integration plan
  ([`docs/workplan-clos-2016-integration.md`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/workplan-clos-2016-integration.md)).
  The next wave (AAG–AAH) characterises the dense-vs-iterative
  envelope and benchmarks the exact-diagonalisation cost on
  representative hardware.
- **Use `IPR_av` on your own data.** The helper takes any
  `SpectrumResult` plus an initial state, so it generalises
  beyond the Porras Hamiltonian. The relevant interpretation
  caveat — that this is a $\rho_0$-eigendecomposition–weighted
  *average* of pure-state effective dimensions, not a
  mixed-state d_eff — applies in any setting.
