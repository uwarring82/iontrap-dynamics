# Tutorial 10 — Finite-shot statistics

**Goal.** Tutorial 1 ended with a Wilson 95 % CI on a
bright-fraction estimate. This tutorial zooms in on that
one step — the three statistics functions
(`wilson_interval`, `clopper_pearson_interval`,
`binomial_summary`), the `BinomialSummary` dataclass, and the
Wilson-vs-Clopper–Pearson choice — and applies them at scale
across a full readout trajectory. By the end you will have:

1. Computed Wilson and Clopper–Pearson 95 % CIs on the
   bright-fraction estimator at a fistful of canonical
   `(k, n)` points, and seen Wilson's characteristic tighter
   span.
2. Vectorised `binomial_summary` across an entire 200-point
   Rabi-flop trajectory in one call, with no Python-level
   for-loop.
3. Chosen the right estimator for a given purpose — Wilson for
   routine operation, Clopper–Pearson for conservative
   worst-case uncertainty reporting, **never Wald**.
4. Understood why both intervals hit `[0, 0.x]` / `[0.x, 1]`
   cleanly at the extreme `k ∈ {0, n}` boundaries that
   routinely arise in ion-trap readout (ground-state sideband
   probes, high-fidelity detectors on `|↓⟩`-prepared shots).

**Expected time.** ~10 min reading; ~1 s runtime.

**Prerequisites.** [Tutorial 1](01_first_rabi_readout.md) — we
reuse its carrier-Rabi scenario and the `SpinReadout` protocol.
`DetectorConfig` and `BinomialChannel` background at the level
of [`CONVENTIONS.md`](../conventions.md) §17.

---

## The three functions and when to call each

| Function                     | Purpose                                          | Typical caller             |
|------------------------------|--------------------------------------------------|----------------------------|
| `wilson_interval(k, n)`      | Tuple `(lower, upper)` — **Wilson** score CI     | Ad-hoc sanity checks       |
| `clopper_pearson_interval(k, n)` | Tuple `(lower, upper)` — **Clopper–Pearson** (exact) CI | Conservative reports       |
| `binomial_summary(k, n)`     | `BinomialSummary` record: point + CI + metadata  | Anything publication-facing |

All three accept scalar or array inputs and broadcast standard
NumPy rules. The `BinomialSummary` returned by
`binomial_summary` is a frozen dataclass with five numerical
fields (`successes`, `trials`, `point_estimate`, `lower`,
`upper` — all same-shape) plus the `confidence` and `method`
metadata. Wilson is the default method; pass
`method="clopper-pearson"` for the exact variant.

!!! note "Why no Wald interval"

    The textbook `p̂ ± z·√(p̂(1−p̂)/n)` Wald formula *degenerates*
    at the extremes: at `p̂ = 0` or `p̂ = 1` the half-width
    collapses to zero, silently claiming a degenerate CI like
    `[1.0, 1.0]` even for tiny `n`. In ion-trap experiments
    those extremes are routine (high-fidelity detectors on
    ground-state preparations give `p̂ ≈ 1`; RSB probes on
    vacuum motional states give `p̂ ≈ 0`), so Wald is a silent
    footgun. The library ships only Wilson and Clopper–Pearson,
    both of which degrade gracefully at the extremes.

## Canonical `(k, n)` sanity-check table

A handful of anchor points, Wilson vs Clopper–Pearson at
95 % coverage:

| k   | n   | Wilson                    | Clopper–Pearson           |
|-----|-----|---------------------------|---------------------------|
| 0   | 10  | `[0.0000, 0.2775]`        | `[0.0000, 0.3085]`        |
| 1   | 10  | `[0.0179, 0.4042]`        | `[0.0025, 0.4450]`        |
| 5   | 10  | `[0.2366, 0.7634]`        | `[0.1871, 0.8129]`        |
| 9   | 10  | `[0.5958, 0.9821]`        | `[0.5550, 0.9975]`        |
| 10  | 10  | `[0.7225, 1.0000]`        | `[0.6915, 1.0000]`        |
| 50  | 100 | `[0.4038, 0.5962]`        | `[0.3983, 0.6017]`        |
| 95  | 100 | `[0.8882, 0.9785]`        | `[0.8872, 0.9836]`        |

```python
from iontrap_dynamics import wilson_interval, clopper_pearson_interval

for k, n in [(0, 10), (5, 10), (10, 10), (50, 100), (95, 100)]:
    wl, wu = wilson_interval(k, n)
    cl, cu = clopper_pearson_interval(k, n)
    print(f"{k:3d}/{n:3d}: Wilson [{float(wl):.4f}, {float(wu):.4f}]"
          f"   CP [{float(cl):.4f}, {float(cu):.4f}]")
```

Three observations worth taking from the table:

1. **Clopper–Pearson is uniformly wider.** For `5/10`, CP is
   `[0.187, 0.813]` vs Wilson's `[0.237, 0.763]` — a 13 % wider
   interval. Conservatism has a cost in interval width.
2. **Both shrink as `n` grows.** At `50/100` (point
   estimate 0.5) both bracket close to `[0.40, 0.60]`; the
   extra conservatism of CP at `n = 100` is under 1 % of
   interval width. For shot budgets in the hundreds the two
   methods agree for practical purposes.
3. **Both handle the boundaries.** `k = 0` gives `lower = 0`
   exactly; `k = n` gives `upper = 1` exactly. No degenerate
   zero-width intervals.

## Vectorised across a full trajectory

The idiomatic way to get CIs over a whole `result.times` axis
is to pass broadcasted `(k, n)` arrays directly to
`binomial_summary` — one call, no Python loop. `SpinReadout`
returns the per-time-step counts under the same label schema it
uses for expectations:

```python
import numpy as np
import qutip
from iontrap_dynamics import (
    DetectorConfig, SpinReadout, binomial_summary,
)
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

mode = ModeConfig(
    label="axial",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
)
system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
hilbert = HilbertSpace(system=system, fock_truncations={"axial": 3})

drive = DriveConfig(
    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
    carrier_rabi_frequency_rad_s=2 * np.pi * 1.0e6,
    phase_rad=0.0,
)
hamiltonian = carrier_hamiltonian(hilbert, drive, ion_index=0)

psi_0 = qutip.tensor(spin_down(), qutip.basis(3, 0))
times = np.linspace(0.0, 2e-6, 200)

result = solve(
    hilbert=hilbert, hamiltonian=hamiltonian, initial_state=psi_0,
    times=times, observables=[spin_z(hilbert, 0)],
)

detector = DetectorConfig(efficiency=0.5, dark_count_rate=0.3, threshold=3)
readout = SpinReadout(
    ion_index=0, detector=detector, lambda_bright=20.0, lambda_dark=0.0,
)
shots = 80
measurement = readout.run(result, shots=shots, seed=20260421)

# The measurement carries per-shot bright/dark bits as a (shots, n_times)
# int64 array; sum over axis 0 to recover the per-time-step bright count.
bits = measurement.sampled_outcome["spin_readout_bits"]
bright_counts = bits.sum(axis=0)  # shape (n_times,), dtype int64

# One call — binomial_summary vectorises over the whole trajectory.
summary = binomial_summary(
    bright_counts, shots, confidence=0.95, method="wilson",
)

assert summary.point_estimate.shape == times.shape
assert summary.lower.shape == times.shape
assert summary.upper.shape == times.shape
```

`summary.point_estimate` is the per-time-step
`bright_count / shots` estimator. `summary.lower` and
`summary.upper` are the 95 % Wilson CI bounds, also
per-time-step. You can plot the three as:

```python
import matplotlib.pyplot as plt

times_us = times * 1e6
plt.fill_between(
    times_us, summary.lower, summary.upper,
    color="steelblue", alpha=0.25, label="95 % Wilson CI",
)
plt.plot(times_us, summary.point_estimate, color="steelblue",
         marker=".", linestyle="None", label=f"bright fraction ({shots} shots)")
plt.xlabel("time (µs)")
plt.ylabel("bright fraction")
plt.legend()
```

The CI band widens where the fringe is near `0.5` (maximum
Bernoulli variance) and narrows at the extremes — a direct
visual of the Wilson interval's shape dependence on the point
estimate.

## Choosing Wilson vs Clopper–Pearson

The practical decision tree:

- **Exploratory work, automated sanity checks, routine
  reporting.** Wilson. Close to nominal coverage at modest `n`,
  well-behaved at the extremes, cheap to compute
  (closed-form).
- **Publication results where you must claim coverage that is
  *guaranteed* ≥ nominal.** Clopper–Pearson. Actual coverage is
  always ≥ stated — "conservative" in the sense of never
  under-reporting uncertainty.
- **Low-shot regimes where Wilson and CP visibly disagree.**
  Use both. If the two intervals bracket your reported bound
  tightly (say, within 10 % of each other), pick Wilson for
  its closer-to-nominal coverage. If they diverge significantly
  (small `n`, extreme `p̂`), pick CP and explain why.
- **Deep extremes with many trials.** At `k = 0`, `n = 1000`
  Wilson gives `[0, 3.7e-3]`; CP gives `[0, 3.7e-3]` as well
  (the two converge as `n → ∞`). For `n ≳ 200` the choice
  rarely matters at 95 % coverage.

## Scaling the shot budget

Increasing the shot budget shrinks the CI. The relative
precision goes as `1 / √n` — doubling `n` multiplies the
interval width by `1/√2 ≈ 0.71`. A quick way to turn a
required interval width into a shot budget:

```python
# Width target: ±2 % absolute at p̂ = 0.5 (widest Bernoulli variance).
# Wilson half-width ≈ z·√(p̂(1-p̂)/n) for moderate n; solve for n.
target_half_width = 0.02
p_max_var = 0.5
z = 1.96
n_required = (z ** 2) * p_max_var * (1 - p_max_var) / (target_half_width ** 2)
print(f"n ≥ {int(np.ceil(n_required))}")  # n ≥ 2401
```

At `p̂ = 0.5` the answer is about 2400 shots for a ±2 % band.
For smaller `p̂` the required shot count drops (variance is
smaller); for very small `p̂` (say 1 %) the CI shape becomes
asymmetric and the Wilson formula — not the Wald-style
`p ± z·σ` approximation — is what gives the correct answer.

## Where to next

- [Tutorial 1](01_first_rabi_readout.md) — the single-CI
  worked example this tutorial expands on.
- [Tutorial 11 (planned)](index.md) — jitter ensembles,
  where `binomial_summary` vectorises across per-trial
  readout counts in a cross-ensemble aggregate.
- [CONVENTIONS §17](../conventions.md) — the measurement
  layer's z-score convention and the Wilson / CP formulas.
- [`src/iontrap_dynamics/measurement/statistics.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/measurement/statistics.py)
  — reference implementation; the `_validate_counts` helper
  lifts the "k ≤ n element-wise" guardrail for vectorised
  calls.
- [`tools/run_demo_wilson_ci.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/run_demo_wilson_ci.py)
  — the runnable demo behind the Tutorial 1 plot.

---

## Licence

Sail material — adaptive guidance with specific parameter choices,
not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
