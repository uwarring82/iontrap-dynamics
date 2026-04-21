# Tutorial 3 — Gaussian π-pulse with `modulated_carrier_hamiltonian`

**Goal.** Keep the four-step skeleton from
[Tutorial 1](01_first_rabi_readout.md) and
[Tutorial 2](02_red_sideband_fock1.md) — configure → build → solve
→ read out — and swap the static carrier Hamiltonian for a
**time-dependent** one whose instantaneous Rabi frequency
`Ω(t) = Ω · f(t)` is shaped by a Gaussian envelope. By the end
you'll have delivered a clean `|↓⟩ → |↑⟩` π-rotation by choosing
the envelope amplitude so that the **pulse area** integrates to
exactly π, and you'll have seen the Bloch vector trace out a
smooth meridian in the y–z plane (not a bang-bang square-wave
trajectory).

The reference script is [`tools/run_demo_gaussian_pulse.py`][demo] —
the same scenario, used as the first Phase 1 end-to-end public-API
exercise. Its committed output bundle under
[`benchmarks/data/gaussian_pi_pulse_demo/`][bundle] includes the
plot embedded below.

[demo]: https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/run_demo_gaussian_pulse.py
[bundle]: https://github.com/uwarring82/iontrap-dynamics/tree/main/benchmarks/data/gaussian_pi_pulse_demo

**Expected time.** ~10 min reading; ~1 s runtime.

**Prerequisites.** [Tutorial 1](01_first_rabi_readout.md) — the
four-step skeleton. Nothing from Tutorial 2 is needed; the pulse
stays on the carrier (no sideband physics). Passing familiarity
with the interaction-picture carrier Hamiltonian at the level of
[`CONVENTIONS.md`](../conventions.md) §9.

---

## The scenario

One ²⁵Mg⁺ ion, same trap as Tutorials 1–2 (axial mode at
`ω_mode / 2π = 1.5 MHz`, but the carrier-only dynamics don't touch
it — the mode slot is present only so `HilbertSpace` has something
to attach to). The laser is tuned on resonance at carrier Rabi
frequency `Ω / 2π = 1 MHz` — the same value as Tutorial 1.
Starting state: `|↓, 0⟩`.

What's different: the laser intensity is **gated** by a Gaussian
envelope centred at `t_c = 2.5 μs` with `σ = 0.5 μs`, over a total
pulse window `T = 5 μs`. The instantaneous Rabi frequency is

```
Ω(t) = Ω · f(t),    f(t) = A · exp(−(t − t_c)² / (2σ²))
```

and the envelope amplitude `A` is chosen so that the pulse area

```
θ(T) = ∫₀^T Ω · f(t) dt = π
```

comes out to exactly π — a clean single π-rotation. For a Gaussian
well inside the window (σ ≪ t_c, T − t_c), the integral collapses
to `Ω · A · σ · √(2π)`, so

```
A = π / (Ω · σ · √(2π))
```

Plugging in `Ω / 2π = 1 MHz`, `σ = 0.5 μs` gives `A ≈ 0.399`. The
Bloch trajectory during the pulse is the integrated rotation:

```
⟨σ_x⟩(t) = 0                    (phase φ = 0)
⟨σ_y⟩(t) = sin(θ(t))            θ(t) = ∫₀^t Ω · f(t') dt'
⟨σ_z⟩(t) = −cos(θ(t))
```

so the state smoothly traces a meridian in the y–z plane from the
south pole (`⟨σ_z⟩ = −1`) at `t = 0` to the north pole
(`⟨σ_z⟩ = +1`) at `t = T`.

## Step 1 — Configure (same three objects as Tutorial 1)

Identical to Tutorial 1's Step 1 — the pulse shape is a property of
the *Hamiltonian builder*, not of the `DriveConfig`. The
`carrier_rabi_frequency_rad_s` field stays the **peak-envelope**
Rabi frequency Ω; the envelope scales it from there.

```python
import numpy as np

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

mode = ModeConfig(
    label="axial",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
)
system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))

drive = DriveConfig(
    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
    carrier_rabi_frequency_rad_s=2 * np.pi * 1.0e6,  # Ω/2π = 1 MHz
    phase_rad=0.0,
)
```

## Step 2 — Define the envelope and build the Hamiltonian

`modulated_carrier_hamiltonian` is the generic pulse-envelope
primitive. It takes a Python callable `envelope: Callable[[float], float]`
and returns the Hamiltonian in QuTiP's **time-dependent list format**
`[[H_carrier, coeff_fn]]` — not a bare `Qobj` like the static
builders. You don't need to care about the internal wrapping;
`sequences.solve` dispatches both formats transparently.

```python
import math

from iontrap_dynamics.hamiltonians import modulated_carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace

hilbert = HilbertSpace(system=system, fock_truncations={"axial": 3})

# Pulse parameters
pulse_duration_s = 5.0e-6
pulse_centre_s = pulse_duration_s / 2.0
pulse_sigma_s = pulse_duration_s / 10.0  # = 0.5 μs

# Pulse-area normalisation — area(Gaussian · Ω) = π
rabi_rad_s = drive.carrier_rabi_frequency_rad_s
envelope_amplitude = np.pi / (rabi_rad_s * pulse_sigma_s * np.sqrt(2 * np.pi))

def gaussian_envelope(t: float) -> float:
    return envelope_amplitude * math.exp(
        -((t - pulse_centre_s) ** 2) / (2 * pulse_sigma_s**2)
    )

hamiltonian = modulated_carrier_hamiltonian(
    hilbert, drive, ion_index=0, envelope=gaussian_envelope
)
```

!!! note "Why the envelope is a plain `Callable[[float], float]`"

    Many QuTiP time-dependent-list coefficients are written as
    `(t, args) -> float` closures keyed on a QuTiP-specific `args`
    dict. The builder lifts that boilerplate for you: pass a pure
    Python function of `t` (SI seconds), and the wrapping into
    QuTiP's step-callback is handled internally. The envelope must
    be deterministic — the solver samples it at every sub-step,
    and stochasticity there would invalidate the integrator's
    step-size control.

!!! tip "Choosing `pulse_sigma` and the window"

    Keep the Gaussian well inside `[0, T]` — the pulse-area formula
    `∫Ω·f = Ω·A·σ·√(2π)` uses the Gaussian's full-plane integral,
    and the window-truncation error is the erf-complement of the
    clipped tails. For `σ = T/10` centred at `T/2` the truncation
    is ~1e−30 (negligible); dropping to `σ = T/4` brings it up to
    ~1e−2, and you'd need to either widen the window or replace
    the analytic pulse-area formula with a numerical
    `scipy.integrate.quad` over the finite window.

## Step 3 — Solve with three Bloch-component observables

The on-resonance carrier with zero laser phase drives the spin
along the y axis in the `{σ_x, σ_y, σ_z}` basis. `⟨σ_x⟩` stays at
zero for the whole pulse; `⟨σ_y⟩` and `⟨σ_z⟩` together trace out
the rotation. Requesting all three in one `solve` call gives you
the full Bloch trajectory to compare against analytics.

```python
from iontrap_dynamics.observables import spin_x, spin_y, spin_z
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.states import ground_state

psi_0 = ground_state(hilbert)
times = np.linspace(0.0, pulse_duration_s, 400)

result = solve(
    hilbert=hilbert,
    hamiltonian=hamiltonian,
    initial_state=psi_0,
    times=times,
    observables=[
        spin_x(hilbert, 0),
        spin_y(hilbert, 0),
        spin_z(hilbert, 0),
    ],
)

sigma_x = result.expectations["sigma_x_0"]
sigma_y = result.expectations["sigma_y_0"]
sigma_z = result.expectations["sigma_z_0"]
```

The analytic prediction is the integrated rotation angle
`θ(t) = ∫₀^t Ω · f(t') dt'`, which you can evaluate with a
cumulative trapezoidal rule on a finer grid than the solver output:

```python
fine = np.linspace(times[0], times[-1], 20 * len(times))
omega_f_fine = rabi_rad_s * np.array([gaussian_envelope(t) for t in fine])
theta_fine = np.concatenate(
    ([0.0], np.cumsum(0.5 * (omega_f_fine[:-1] + omega_f_fine[1:])) * np.diff(fine)[0])
)
theta = np.interp(times, fine, theta_fine)

max_error = max(
    np.max(np.abs(sigma_x)),                    # analytic σ_x ≡ 0
    np.max(np.abs(sigma_y - np.sin(theta))),
    np.max(np.abs(sigma_z + np.cos(theta))),
)
assert max_error < 1e-5
assert abs(theta[-1] - np.pi) < 1e-5   # total pulse area = π
assert sigma_z[-1] > 0.9999             # ended at the north pole
```

Three independent assertions: the numerical Bloch trajectory
matches the analytic `(0, sin θ, −cos θ)` curve to better than
`1e-5`, the integrated pulse area comes out to π, and the final
spin projection is essentially `+1` (a clean π-rotation).

## Step 4 — Read out (still the same `SpinReadout`)

The readout layer is agnostic to how the trajectory was produced —
static carrier, sideband flop, or shaped pulse all feed the same
`SpinReadout` / `DetectorConfig` pair:

```python
from iontrap_dynamics import DetectorConfig, SpinReadout

detector = DetectorConfig(efficiency=0.5, dark_count_rate=0.3, threshold=3)
readout = SpinReadout(
    ion_index=0, detector=detector, lambda_bright=20.0, lambda_dark=0.0
)
measurement = readout.run(result, shots=500, seed=20260421)
bright_fraction = measurement.sampled_outcome["spin_readout_bright_fraction"]
```

At the final time step the bright fraction should land near the
expected `0.5 · (1 + 0.9999) ≈ 0.9999` brightening probability
(modulo detector inefficiency and dark counts, both absorbed into
the `DetectorConfig` model from Tutorial 1).

## Putting it together

The committed reference run (`Ω/2π = 1 MHz`, `σ = 0.5 μs`, 400
time points, `N_Fock = 3`) produces:

![Gaussian π-pulse](https://raw.githubusercontent.com/uwarring82/iontrap-dynamics/main/benchmarks/data/gaussian_pi_pulse_demo/plot.png)

Top panel: the Gaussian envelope `f(t)` — peaks at `t = 2.5 μs`,
amplitude `A ≈ 0.399`. Below it: the three Bloch components.
`⟨σ_x⟩` stays pinned at 0. `⟨σ_y⟩` rises, peaks at `sin(π/2) = 1`
around `t = 2.5 μs` (the half-area point), then decays back to 0
as the pulse tail fills in the second half-rotation. `⟨σ_z⟩`
swings cleanly from `−1` to `+1` — a π-rotation, closed to
sub-ppm tolerance:

```
final pulse area θ(T) = 3.1415909 rad (target π)
max |⟨σ_i⟩_numeric − analytic| ≈ 3.4e−06
final ⟨σ_z⟩ = +0.99999999…
```

Wall-clock for the full 400-step time-dependent solve on a 2023
M2 MacBook Air: ~7 ms.

## Physics you can probe next

The envelope is a pure-Python callable, so arbitrary pulse shapes
come for free — no new builder needed. Three natural modifications:

### Blackman window instead of Gaussian

The Blackman window is a compact-support, smoother-edged
alternative to the Gaussian that sits *inside* `[0, T]` with zero
slope at the endpoints. Its cumulative pulse area has a
closed-form expression, but for normalisation you can also just
evaluate it numerically with `scipy.integrate.quad`:

```python
def blackman_envelope(t: float) -> float:
    if not 0.0 <= t <= pulse_duration_s:
        return 0.0
    x = t / pulse_duration_s
    return (
        0.42
        - 0.5 * math.cos(2 * math.pi * x)
        + 0.08 * math.cos(4 * math.pi * x)
    )

from scipy.integrate import quad
unscaled_area, _ = quad(blackman_envelope, 0.0, pulse_duration_s)
blackman_amplitude = np.pi / (rabi_rad_s * unscaled_area)
# wrap in a final envelope that multiplies in blackman_amplitude
```

### Stroboscopic square-wave drive

A square envelope on for a fraction of each mode period — the
`workplan §0.F` benchmark 3 exercises this regime. The envelope
is a simple `1.0` / `0.0` step function; the pulse-area condition
now determines the *duty cycle*, not a continuous amplitude. This
is the closest the modulated-carrier builder gets to emulating a
bang-bang AC drive.

### Adiabatic amplitude ramp

A slowly-varying envelope (e.g. a raised-cosine at both ends plus
a flat middle) is the standard tool for soft turn-on of a drive —
the transient spectral content of a hard step excites unwanted
sidebands when you're running near a mode frequency. This is
purely an envelope change; the rest of the pipeline is unchanged.

## Where to next

- [Tutorial 1](01_first_rabi_readout.md) — the static-carrier
  baseline with finite-shot readout.
- [Tutorial 2](02_red_sideband_fock1.md) — the sibling swap, but
  to the red-sideband Hamiltonian and Fock `|1⟩` initial state.
- [Phase 1 Architecture](../phase-1-architecture.md) — reference
  for the `modulated_carrier_hamiltonian` builder and the full
  list-format dispatch path through `sequences.solve`.
- [`tools/run_demo_gaussian_pulse.py`][demo] — the runnable script
  that produced the plot embedded above; diff it against this
  tutorial for the exact code plus the cumulative-integral
  analytic overlay.

---

## Licence

Sail material — adaptive guidance with specific parameter choices,
not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
