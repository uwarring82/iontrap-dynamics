# Tutorial 17 — Motional decoherence and the Lamb–Dicke regime

**Goal.** Characterise and budget the imperfections of a real motional
mode. By the end you will have driven a mode with typed open-system
channels through `solve(channels=…)`, read the resulting contrast loss off
an interferometer fringe, mapped where in the Lamb–Dicke regime the
leading-order formulas can still be trusted, and folded a trap-frequency
drift into the error budget.

**Reference implementation.** `tools/run_benchmark_motional_channels.py` and
`tools/run_benchmark_lamb_dicke_regime.py`, with committed plots under
[`benchmarks/data/`](https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data).

**Expected time.** ~15 min reading; ~2 s runtime.

**Prerequisites.** [Tutorial 9](09_squeezed_coherent_prep.md) (motional
state prep), [Tutorial 8](08_full_lamb_dicke.md) (the full-Lamb–Dicke
sideband), and [Tutorial 16](16_two_mode_squeezing.md) (the `ModeConfig`
setup). CONVENTIONS.md §24 fixes the channel parameterisation and §10 the
Lamb–Dicke parameter. This tutorial bundles four small surfaces into one
"characterising a noisy mode" workflow.

---

## The scenario

The earlier tutorials evolve a *closed* system. Real ion motion is open: it
heats from the trap, damps, and dephases, and the strength of every
spin–motion coupling depends on where you sit in the Lamb–Dicke regime.
This tutorial walks the four library surfaces you need to *characterise*
those imperfections, in the order a real analysis hits them — inject the
noise, read it out as contrast loss, classify the regime, then budget a
systematic drift.

## Step 1 — Typed open-system channels through `solve(channels=…)`

Three frozen channel dataclasses cover the canonical motional baths
(CONVENTIONS §24). Passing any of them to `solve(channels=…)` switches the
solver from the unitary path onto the master equation. Each follows a
textbook decay law: **Heating** relaxes the ground state *up* to a bath
occupation, **AmplitudeDamping** relaxes a Fock state *down* to the ground,
and **Dephasing** kills the coherence quadrature while leaving the energy
untouched.

```python
import numpy as np
import qutip
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.system import IonSystem
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import Observable
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.states import coherent_mode, compose_density
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics import AmplitudeDamping, Heating, Dephasing

def single_mode_hilbert(fock_dim):
    mode = ModeConfig(label="b", frequency_rad_s=2 * np.pi * 1.0e6,
                      eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]))
    system = IonSystem(species_per_ion=(mg25_plus(),), modes=(mode,))
    return HilbertSpace(system=system, fock_truncations={"b": fock_dim})

fock = 30
h = single_mode_hilbert(fock)
n_op = h.number_for_mode("b")
a = h.annihilation_for_mode("b"); x_op = (a + a.dag()) / np.sqrt(2)
H0 = 0.0 * qutip.tensor(qutip.qeye(2), qutip.qeye(fock))   # zero drift: pure dissipation
times = np.linspace(0, 1e-3, 80)

# Heating: ground state relaxes UP to the bath occupation n̄ = 2
ground = qutip.tensor(qutip.basis(2, 0), qutip.basis(fock, 0))
res_h = solve(hilbert=h, hamiltonian=H0, initial_state=ground, times=times,
              observables=(Observable(label="n", operator=n_op),),
              channels=[Heating(mode="b", rate=3000.0, n_bar_bath=2.0)])
n_t = np.asarray(res_h.expectations["n"])
assert res_h.metadata.backend_name == "qutip-mesolve"     # channels FORCE the master equation
assert abs(n_t[-1] - 2.0 * (1 - np.exp(-3000.0 * times[-1]))) < 5e-3   # n̄(1 − e^{−κt})

# Dephasing: coherence ⟨x⟩ of a coherent state decays as e^{−γt/2}; ⟨n⟩ is untouched
coherent = compose_density(h, spin_states_per_ion=[spin_down()],
                           mode_states_by_label={"b": coherent_mode(fock, 2.0)})
res_d = solve(hilbert=h, hamiltonian=H0, initial_state=coherent, times=times,
              observables=(Observable(label="x", operator=x_op), Observable(label="n", operator=n_op)),
              channels=[Dephasing(mode="b", rate=4000.0)])
x_t = np.asarray(res_d.expectations["x"]); n_t2 = np.asarray(res_d.expectations["n"])
assert abs(x_t[-1] - x_t[0] * np.exp(-4000.0 * times[-1] / 2)) < 5e-3   # coherence decay
assert abs(n_t2[-1] - n_t2[0]) < 1e-6                                    # energy preserved
```

![Occupation relaxation under heating, amplitude damping and a windowed heating run; and coherence decay under dephasing while the occupation stays flat](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/motional_channels/plot.png)

The left panel shows the occupation laws — heating rising to the bath,
amplitude damping decaying to the ground, and a **windowed** heating run
(`window=(0, T/2)`) whose rise flattens the instant its window closes, the
sequence-aware noise model. The right panel is the dephasing signature:
coherence decays while energy does not.

!!! note "Three channel rules worth pinning"

    `channels=` (and all of `solve`'s arguments) are keyword-only. Any
    dissipative channel forces `backend_name == "qutip-mesolve"`; with
    `channels=()` the solver stays byte-for-byte on the unitary path.
    `rate=0.0` is a no-op. A `window=(t0, t1)` is half-open `[t0, t1)` in SI
    seconds; when any channel is windowed, `solve` caps the integrator's
    step so a short window can never be stepped over. Channel *order* in the
    list is irrelevant — only the temporal schedule of windows matters (the
    R8 boundary the library refuses to assume away).

## Step 2 — Reading decoherence as contrast loss

That dephasing coherence decay is exactly what an interferometer measures
as **visibility loss**. The `observables` module turns a phase scan into a
contrast and a phase with two helpers: `fringe_visibility` (model-free,
from the extrema) and `fit_fringe` (a robust least-squares
`A + B·cos(θ − φ)` fit). They are pure numerics — feed them a measured
array or the output of `solve`.

```python
from iontrap_dynamics.observables import fringe_visibility, fit_fringe

# A phase scan with finite-shot noise and a contrast set by the decoherence above
theta = np.linspace(0, 2 * np.pi, 25)
V_true, phi_true, offset = 0.8, 0.6, 0.5
rng = np.random.default_rng(0)
signal = offset * (1 + V_true * np.cos(theta - phi_true)) + 0.01 * rng.standard_normal(theta.size)
signal = np.clip(signal, 0.0, None)            # a readout probability must be non-negative

v = fringe_visibility(signal)                  # model-free contrast ≈ 0.82 (biased high by noise)
fit = fit_fringe(theta, signal)                # robust fit → FringeFit(offset, amplitude, phase_rad, visibility)
assert abs(fit.phase_rad - phi_true) < 0.05    # recovers the interferometer phase
assert abs(fit.visibility - V_true) < 0.05     # and the contrast B/A
```

!!! tip "Which contrast estimator?"

    `fringe_visibility` reads the extrema directly — fast, but biased high
    under noise or coarse sampling, and it *requires* a non-negative signal
    (it raises on any negative entry). `fit_fringe` is the robust choice:
    it needs ≥ 3 points and **rejects rank-deficient scans** (repeated or
    collinear phases) rather than silently mis-fitting, and its
    `phase_rad = atan2(c₂, c₁)` recovers the sign of the phase. Use the fit
    for anything you will report; use the model-free value as a quick sanity
    check on clean data.

## Step 3 — The Lamb–Dicke regime map

How badly does any of this matter? That depends on the Lamb–Dicke
parameter `η` and the temperature. The carrier is thermally suppressed by
the **Debye–Waller factor** `e^{−η²(2n̄+1)/2}`, and the regime classifier
partitions the dimensionless `η²(2n̄+1)` into `deep` / `intermediate` /
`beyond` — a direct "can I trust the leading-order formula" map.

```python
from iontrap_dynamics.analytic import (
    debye_waller_factor, lamb_dicke_regime, LambDickeRegime, lamb_dicke_parameter,
    blue_sideband_rabi_frequency, blue_sideband_rabi_frequency_full_ld)

# A physical η: 25Mg+ axial mode at 1 MHz, ~280 nm drive along z
eta = lamb_dicke_parameter(k_vec=np.array([0.0, 0.0, 2 * np.pi / 280e-9]),
                           mode_eigenvector=np.array([0.0, 0.0, 1.0]),
                           ion_mass=mg25_plus().mass_kg, mode_frequency=2 * np.pi * 1.0e6)

# a tightly-confined mode (small η) is comfortably deep…
assert lamb_dicke_regime(lamb_dicke_parameter=0.15, mean_phonon_number=0.0) is LambDickeRegime.DEEP
# …but this physical η (≈ 0.32) already sits at the deep/intermediate boundary even
# ground-state cooled, and a few phonons push it BEYOND — the classifier keys on
# η²(2n̄+1), not η alone
assert lamb_dicke_regime(lamb_dicke_parameter=eta, mean_phonon_number=0.0) is LambDickeRegime.INTERMEDIATE
assert lamb_dicke_regime(lamb_dicke_parameter=eta, mean_phonon_number=5.0) is LambDickeRegime.BEYOND
# the vacuum is already suppressed below 1 — the zero-point term in the exponent
assert debye_waller_factor(lamb_dicke_parameter=eta, mean_phonon_number=0.0) < 1.0

# leading-order over-predicts the sideband Rabi rate as the Fock level climbs
Omega0 = 2 * np.pi * 1e5
lo = blue_sideband_rabi_frequency(carrier_rabi_frequency=Omega0, lamb_dicke_parameter=0.5, n_initial=8)
full = blue_sideband_rabi_frequency_full_ld(carrier_rabi_frequency=Omega0, lamb_dicke_parameter=0.5, n_initial=8)
assert full < lo                               # the linearised √(n+1) form breaks down
```

![Debye–Waller factor versus the regime parameter with the deep, intermediate and beyond bands, and the sideband Rabi frequency where the leading-order form diverges from the exact all-orders curve](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/lamb_dicke_regime/plot.png)

The left panel is the map: the Debye–Waller suppression versus
`η²(2n̄+1)`, banded into the three regimes at the documented thresholds
`0.1` and `1.0`. The right panel shows *why* the regime matters — the
leading-order `|η|√(n+1)` sideband rate climbs without bound while the exact
all-orders curve bends over and eventually nulls. A small `η` can still be
`BEYOND` at high `n̄`: the classifier keys on `η²(2n̄+1)`, not `η` alone.

!!! note "The zero-point trap"

    `debye_waller_factor` is `e^{−η²(2n̄+1)/2} = e^{−η²(n̄+½)}` — the `+½`
    means even the **vacuum** (`n̄ = 0`) is suppressed below 1, not equal to
    it. Expecting `DW(n̄=0) = 1` is a common slip.

## Step 4 — Budgeting a trap-frequency drift

Finally, the systematic. A slow trap-frequency wobble rescales the
Lamb–Dicke parameter: since `η ∝ ω_m^{−1/2}`, a fractional drift `δ` gives
`η → η/√(1+δ)`. `ModeFrequencyDrift` is the multiplicative, dimensionless
knob (parallel to `RabiDrift`); applying it returns a new, frozen
`ModeConfig` from which you re-derive the affected quantities.

```python
from iontrap_dynamics.systematics.drift import ModeFrequencyDrift, apply_mode_frequency_drift

mode = ModeConfig(label="z", frequency_rad_s=2 * np.pi * 1.0e6,
                  eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]))
k_vec = np.array([0.0, 0.0, 2 * np.pi / 280e-9]); ev = np.array([0.0, 0.0, 1.0])
eta0 = lamb_dicke_parameter(k_vec=k_vec, mode_eigenvector=ev,
                            ion_mass=mg25_plus().mass_kg, mode_frequency=mode.frequency_rad_s)

for delta in [-0.10, -0.02, 0.02, 0.10]:
    drifted = apply_mode_frequency_drift(mode, ModeFrequencyDrift(delta=delta))  # ω → ω(1+δ), a NEW config
    eta_d = lamb_dicke_parameter(k_vec=k_vec, mode_eigenvector=ev,
                                 ion_mass=mg25_plus().mass_kg, mode_frequency=drifted.frequency_rad_s)
    assert np.isclose(eta_d, eta0 / np.sqrt(1.0 + delta))   # η rescales as 1/√(1+δ)
# the fractional η change is ≈ −δ/2, NOT −δ — half as sensitive as a naive estimate
```

!!! tip "Half the sensitivity you'd guess"

    Because `η ∝ ω_m^{−1/2}` is sub-linear, a fractional frequency drift `δ`
    moves `η` by `(1+δ)^{−1/2} − 1 ≈ −δ/2`, *half* of `−δ`. So a 1 % trap
    wobble is a ~0.5 % `η` error, which then feeds straight back into the
    Debye–Waller factor from Step 3. `ModeFrequencyDrift` is multiplicative
    and dimensionless (contrast `DetuningDrift`/`PhaseDrift`, which are
    additive with SI units); `apply_mode_frequency_drift` only updates the
    frequency, so always re-derive `η` from the returned mode.

## Where to next

- [Tutorial 11 — Systematics: jitter ensembles](11_jitter_ensembles.md):
  the stochastic, shot-to-shot complement to this deterministic drift.
- The benchmarks `tools/run_benchmark_motional_channels.py` and
  `tools/run_benchmark_lamb_dicke_regime.py` reproduce both figures with
  their analytic oracles.

---

## Licence

Sail material — adaptive guidance with specific parameter choices, not a
coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
