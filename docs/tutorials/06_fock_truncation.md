# Tutorial 6 — Fock truncation diagnosis

**Goal.** Every `sequences.solve` call runs a silent bookkeeper
that classifies your Fock truncation against a tolerance ε. This
tutorial shows what the three levels mean, how to read the
`result.warnings` record, how to override ε via the
`fock_tolerance` argument, and — most importantly — how to turn
"my solve raised `ConvergenceError`" into a diagnosis rather than
a guessing game about how large `N_Fock` should be. Unlike the
preceding tutorials, this one is about a **diagnostic layer**
rather than a new physics scenario.

By the end you will have:

1. Run the same scenario with five `N_Fock` values and seen each
   one of the four CONVENTIONS §15 statuses (silent OK, Level 1,
   Level 2, Level 3 raise) in turn.
2. Read the `result.warnings` tuple to extract the exact
   mode label, per-mode top-Fock population, and tolerance
   diagnostics.
3. Tightened ε below the library default via `fock_tolerance` to
   promote a Level 1 warning into a Level 3 raise — the
   publication-grade workflow.
4. Understood why the default ε = 1e-4 is the right floor to
   catch silent truncation pathology without false positives.

**Expected time.** ~12 min reading; ~2 s runtime.

**Prerequisites.** [Tutorial 1](01_first_rabi_readout.md) for the
`solve()` signature and the `result.expectations` access pattern.
CONVENTIONS [§13 (Fock truncation)](../conventions.md) and §15
(severity ladder) for the spec this tutorial walks you through.

---

## The three-level ladder in one picture

For each mode `m`, the solver computes the **top-Fock
population** across the whole trajectory:

```
p_top(m) = max_t  ⟨N_Fock − 1 | ρ_m(t) | N_Fock − 1⟩
```

and classifies it against tolerance ε
(`iontrap_dynamics.conventions.FOCK_CONVERGENCE_TOLERANCE`, default
`1e-4`):

| Regime                       | Classification         | Behaviour                                       |
|------------------------------|------------------------|-------------------------------------------------|
| `p_top < ε / 10`             | OK                     | Silent                                          |
| `ε / 10 ≤ p_top < ε`         | **Level 1** warning    | `FockConvergenceWarning` + `result.warnings`    |
| `ε ≤ p_top < 10·ε`           | **Level 2** warning    | `FockQualityWarning` + `result.warnings`        |
| `p_top ≥ 10·ε`               | **Level 3** failure    | `ConvergenceError` **raised** — no result       |

The asymmetric decade either side of ε is deliberate:
Level 1 is a soft "tighten for publication" nudge, Level 2 a
"quality degraded — check before you publish" warning, and
Level 3 a hard refusal to return a potentially-contaminated
result.

## The canonical demonstration scenario

A single ²⁵Mg⁺ ion in a thermal motional state with `n̄ = 0.5`
(barely-not-cooled — a realistic post-Doppler starting point).
Driving the carrier on resonance doesn't couple spin to motion,
so the Fock distribution is **static** throughout the
trajectory — `p_top` just equals the thermal-tail population
`P_thermal(N_Fock − 1) = (0.5 / 1.5)^(N_Fock − 1) / 1.5`. This
makes the warning classification a deterministic function of
`N_Fock` alone, perfect for walking the ladder end-to-end.

```python
import numpy as np
import qutip

from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import carrier_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.system import IonSystem

N_BAR = 0.5

def build_scenario(n_fock: int):
    mode = ModeConfig(
        label="axial",
        frequency_rad_s=2 * np.pi * 1.5e6,
        eigenvector_per_ion=np.array([[0.0, 0.0, 1.0]]),
    )
    system = IonSystem.homogeneous(species=mg25_plus(), n_ions=1, modes=(mode,))
    hilbert = HilbertSpace(system=system, fock_truncations={"axial": n_fock})

    drive = DriveConfig(
        k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
        carrier_rabi_frequency_rad_s=2 * np.pi * 1.0e6,
        phase_rad=0.0,
    )
    hamiltonian = carrier_hamiltonian(hilbert, drive, ion_index=0)

    spin = qutip.ket2dm(spin_down())
    motion = qutip.thermal_dm(n_fock, N_BAR)
    rho_0 = qutip.tensor(spin, motion)
    return hilbert, hamiltonian, rho_0
```

A thermal initial state is a **density matrix**, not a ket —
solvers will dispatch to `qutip.mesolve` rather than `sesolve`,
which is required for any mixed-state input.

## Level 3 — the hard failure (ConvergenceError)

Start with `N_Fock = 5` — deliberately tight. The top-Fock
population is `P_thermal(4) ≈ 8.23e-3`, well above `10·ε = 1e-3`:

```python
from iontrap_dynamics.exceptions import ConvergenceError

hilbert, hamiltonian, rho_0 = build_scenario(n_fock=5)
try:
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=rho_0,
        times=np.linspace(0.0, 1e-6, 20),
        observables=[spin_z(hilbert, 0)],
    )
except ConvergenceError as exc:
    print(f"ConvergenceError raised:\n  {exc}")
# ConvergenceError raised:
#   Fock-truncation failure (CONVENTIONS.md §13, §15 Level 3):
#   top-level populations meet or exceed 10·ε = 1.000e-03 for one
#   or more modes [axial: p_top = 8.264e-03]. Increase
#   fock_truncations for the affected mode(s) and re-run.
```

The solver **completed the integration** — the ODE ran, the
trajectory exists inside the wrapper — but the Fock check
refused to hand it back because the result is potentially
contaminated by truncation. This is intentional: a silently
degraded trajectory that looks fine in a headline plot but is
wrong by a few percent in a publication table is the pathology
the ladder is designed to prevent.

## Level 2 — `FockQualityWarning`

`N_Fock = 7` brings `p_top` into the `[ε, 10·ε)` band:

```python
import warnings
from iontrap_dynamics.exceptions import FockQualityWarning

hilbert, hamiltonian, rho_0 = build_scenario(n_fock=7)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=rho_0,
        times=np.linspace(0.0, 1e-6, 20),
        observables=[spin_z(hilbert, 0)],
    )
    (warning_record,) = [w for w in caught if issubclass(w.category, FockQualityWarning)]
print(warning_record.message)
# mode 'axial': top-Fock population p_top = 9.149e-04 exceeds ε = 1.000e-04
# (N_Fock = 7); quality degraded (CONVENTIONS.md §15 Level 2).
# Consult result.warnings before publication use.
```

The result **is** returned — Level 2 degrades quality but the
trajectory is deliverable. The signal is that if you are
generating this for a publication-grade figure, you should widen
`N_Fock` before the result ships.

## Level 1 — `FockConvergenceWarning`

`N_Fock = 11` drops `p_top` into `[ε/10, ε)`:

```python
from iontrap_dynamics.exceptions import FockConvergenceWarning

hilbert, hamiltonian, rho_0 = build_scenario(n_fock=11)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=rho_0,
        times=np.linspace(0.0, 1e-6, 20),
        observables=[spin_z(hilbert, 0)],
    )
    (warning_record,) = [w for w in caught if issubclass(w.category, FockConvergenceWarning)]
print(warning_record.message)
# mode 'axial': top-Fock population p_top = 1.129e-05 approaches ε = 1.000e-04
# (N_Fock = 11); solver converged but the truncation is close to its envelope
# (CONVENTIONS.md §15 Level 1). Consider tightening fock_truncations for
# publication-grade results.
```

## Silent OK

`N_Fock = 13` pushes `p_top` below `ε / 10 = 1e-5`. No warning,
`result.warnings` is empty:

```python
hilbert, hamiltonian, rho_0 = build_scenario(n_fock=13)
result = solve(
    hilbert=hilbert,
    hamiltonian=hamiltonian,
    initial_state=rho_0,
    times=np.linspace(0.0, 1e-6, 20),
    observables=[spin_z(hilbert, 0)],
)
assert result.warnings == ()
```

The empty `result.warnings` tuple is the positive affirmation —
the solver ran clean, no pathology detected, no per-mode record
to inspect.

## Reading `result.warnings` programmatically

The Python `warnings` channel is one delivery surface; the
structured `result.warnings` tuple is the other. Each entry is a
frozen `ResultWarning` record with four fields: `severity`,
`category`, `message`, and `diagnostics`. The diagnostics dict is
the machine-readable payload:

```python
hilbert, hamiltonian, rho_0 = build_scenario(n_fock=9)
result = solve(
    hilbert=hilbert,
    hamiltonian=hamiltonian,
    initial_state=rho_0,
    times=np.linspace(0.0, 1e-6, 20),
    observables=[spin_z(hilbert, 0)],
)
for w in result.warnings:
    print(f"{w.severity.value:>11s}  {w.category}")
    for k, v in w.diagnostics.items():
        print(f"    {k}: {v!r}")
# quality      fock_truncation
#     mode_label: 'axial'
#     fock_dim: 9
#     p_top_max: 0.0001016...
#     tolerance_epsilon: 0.0001
```

The `diagnostics` dict is the preferred hook for automated CI
gates — read `p_top_max` and decide programmatically whether to
accept the result, escalate to a wider `N_Fock`, or abort a
parameter sweep. Relying on message-string matching is fragile;
the dict is stable.

!!! tip "Aggregating warnings across an ensemble"

    `sequences.solve_ensemble` returns a tuple of
    `TrajectoryResult` objects, each with its own `.warnings`
    tuple. A one-liner that flattens the lot and returns the
    worst-case `p_top_max` across every trial:
    ```python
    worst = max(
        (w.diagnostics["p_top_max"]
         for r in ensemble_results
         for w in r.warnings
         if w.category == "fock_truncation"),
        default=0.0,
    )
    ```
    A jitter sweep that produces any Level 2 warning across its
    ~1000 trials deserves a wider `N_Fock` for the full production
    run — worth catching before the overnight sweep lands rather
    than after.

## Overriding ε via `fock_tolerance`

`fock_tolerance` is a per-call override. Useful for two
situations:

### Tightening below the default (publication-grade)

Set `fock_tolerance=1e-6` (two decades below the default) to
demand a stricter envelope. A result that passed silently at the
library default can now emit warnings or fail outright — turning
a "looks fine" answer into a diagnosed one:

```python
hilbert, hamiltonian, rho_0 = build_scenario(n_fock=13)
# p_top ≈ 1.25e-6. Default ε=1e-4: silent. Tightened ε=1e-6: Level 2.
try:
    result = solve(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        initial_state=rho_0,
        times=np.linspace(0.0, 1e-6, 20),
        observables=[spin_z(hilbert, 0)],
        fock_tolerance=1e-6,
    )
    print(f"passed at ε=1e-6 with {len(result.warnings)} warning(s)")
except ConvergenceError as exc:
    print(f"tightened-ε ConvergenceError: {exc}")
```

Tightening is the recommended pattern for any result that ends up
in a paper — the library's default ε = 1e-4 is a
development-grade floor, not a publication guarantee.

### Loosening (rare, and carries a burden of proof)

Loosening ε is **not** recommended in general — it masks exactly
the degradation the ladder is there to catch. But for
rapid-iteration exploratory work where you know the physics is
truncation-limited and want to see *qualitative* behaviour before
committing to a wider `N_Fock`, `fock_tolerance=1e-2` relaxes the
ladder by two decades. Document why in your notebook; the
lowered threshold doesn't travel with the result's metadata in a
way that downstream readers can easily find.

!!! warning "`fock_tolerance=0` is a ConventionError, not a shortcut"

    Passing zero to disable the check raises immediately — silent
    degradation is forbidden by CONVENTIONS §15. If you truly need
    to suppress the check (e.g. you're running a test of a
    Hamiltonian that *intentionally* saturates the truncation),
    pass a large positive tolerance like `fock_tolerance=1.0`.
    The library is deliberately opinionated here: no escape hatch
    that doesn't leave a trace on the result.

## Diagnosis recipe for `ConvergenceError`

When a solve raises `ConvergenceError`, the message already
contains everything you need:

```
Fock-truncation failure (CONVENTIONS.md §13, §15 Level 3):
top-level populations meet or exceed 10·ε = 1.000e-03 for one
or more modes [axial: p_top = 8.230e-03, radial_x: p_top = 2.11e-03].
Increase fock_truncations for the affected mode(s) and re-run.
```

Three pieces of information you'd otherwise be guessing at:

1. **Which modes failed.** A multi-mode system with 3 modes can
   have one mode saturating while the others are fine — widen
   only the failing one, not everything.
2. **By how much.** `p_top = 8.23e-3` vs threshold `10·ε = 1e-3`
   is a factor-of-8 overshoot; extrapolating the thermal tail
   (or the unitary Fock-ladder spreading) typically shows that
   doubling `N_Fock` on the affected mode drops `p_top` by
   orders of magnitude.
3. **Suggested remediation.** Widen `fock_truncations`. Other
   options exist (reduce `n̄` of the initial state, shorten the
   trajectory, tighten Ω·η — the phase-space excursion amplitude
   in MS-gate scenarios) but the most commonly-right move is the
   one the message names.

## Where to next

- [Tutorial 5](05_custom_observables.md) — `StorageMode.EAGER`
  and custom observable construction; complementary to the Fock
  check for post-hoc trajectory analysis.
- [Conventions §13 + §15](../conventions.md) — the binding spec
  for the Fock check and the three-level severity ladder.
- [`src/iontrap_dynamics/sequences.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/sequences.py)
  — reference implementation of the per-mode check, the
  `result.warnings` emission path, and the
  `ConvergenceError` raise logic.

---

## Licence

Sail material — adaptive guidance with specific parameter choices,
not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-dynamics/blob/main/docs/LICENCE).
