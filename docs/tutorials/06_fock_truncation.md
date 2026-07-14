# Tutorial 6 — Fock truncation diagnosis

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-dynamics/blob/main/docs/tutorials/notebooks/06_fock_truncation.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/build_tutorial_notebooks.py).

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

**Level.** `core` — assumes the basics (Tutorials 0–1).

**Prerequisites.** [Tutorial 1](01_first_rabi_readout.md) for the
`solve()` signature and the `result.expectations` access pattern.
CONVENTIONS [§13 (Fock truncation)](../conventions.md) and §15
(severity ladder) for the spec this tutorial walks you through.

---

!!! note "New here? Read this first"

    - A trapped-ion motional mode lives on a **finite ladder** of Fock states `|0⟩, |1⟩, … |N_Fock−1⟩` — you choose the height `N_Fock`.
    - If the state's population climbs to the top rung `|N_Fock−1⟩`, everything above it is dropped. The result is then **silently biased**, even though the state's norm stays exactly `1`.
    - Because the norm can't reveal this, the library grades every `solve` by the **top-rung population** `p_top` against a tolerance `ε`.
    - Four graded outcomes: **silent OK** → **Level 1** (a gentle convergence nudge) → **Level 2** (quality warning — result still returned) → **Level 3** (`ConvergenceError` — result refused).
    - You clear a flagged solve by raising `N_Fock` (more rungs); you can also *tighten* `ε` to demand a stricter envelope for publication-grade work.

    **In a hurry?** The four level sections (`N_Fock` = 5, 7, 11, 13) walk one scenario down the ladder; the sweep under *Reading `result.warnings`* ties measured `p_top` back to the thresholds.

**Symbols in this tutorial**

| Symbol | Plain meaning |
|--------|---------------|
| `N_Fock` | Fock truncation — how many ladder rungs the mode keeps, i.e. states `0 … N_Fock−1` (you set this). |
| `n̄` | mean phonon number of the motional state; sets how far up the ladder its population reaches. |
| `p_top` | top-rung population — the population of `|N_Fock−1⟩`, maximised over the trajectory; the number that gets graded. |
| `ε` | `FOCK_CONVERGENCE_TOLERANCE` — the tolerance `p_top` is compared against (library default `1e-4`). |
| Level 1 | `FockConvergenceWarning` — `ε/10 ≤ p_top < ε`; converged, but tighten for publication. |
| Level 2 | `FockQualityWarning` — `ε ≤ p_top < 10·ε`; result returned, quality flagged. |
| Level 3 | `ConvergenceError` — `p_top ≥ 10·ε`; result refused, nothing returned. |

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
import matplotlib.pyplot as plt
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

# House colours — match the project reference figures.
BLUE, RED, GREEN, PURPLE, GREY = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#444444"

N_BAR = 0.5
EPSILON = 1e-4  # library default FOCK_CONVERGENCE_TOLERANCE

def build_scenario(n_fock: int):
    # --- boilerplate below: ion + axial mode + Hilbert space. The one knob under
    # test is fock_truncations={"axial": n_fock} — the ladder height (a convergence
    # parameter, not physics: the converged result is invariant to it). ---
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

    # The physics that matters: a thermal state whose n̄=0.5 tail is exactly
    # what p_top measures against the top rung (Fock index n_fock-1).
    spin = qutip.ket2dm(spin_down())
    motion = qutip.thermal_dm(n_fock, N_BAR)
    rho_0 = qutip.tensor(spin, motion)
    return hilbert, hamiltonian, rho_0

# Theoretical top-Fock population for a thermal state:
# P_thermal(N_Fock-1) = (n̄/(1+n̄))^(N_Fock-1) / (1+n̄)
nfock_grid = np.arange(3, 22)
p_top_theory = (N_BAR / (1 + N_BAR)) ** (nfock_grid - 1) / (1 + N_BAR)

print("Setup — thermal tail p_top(N_Fock) for n̄ = 0.5:")
for nf, pt in zip(nfock_grid, p_top_theory):
    label = ("Level 3" if pt >= 10 * EPSILON else
             "Level 2" if pt >= EPSILON else
             "Level 1" if pt >= EPSILON / 10 else "OK")
    print(f"  N_Fock={nf:2d}: p_top = {pt:.3e}  →  {label}")

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.semilogy(nfock_grid, p_top_theory, color=GREY, marker="o", markersize=4, label=r"$p_\mathrm{top}$")
ax.axhline(10 * EPSILON, color=RED,    linestyle="--", linewidth=1.0, label=r"$10\varepsilon$ (Level 3 raise)")
ax.axhline(EPSILON,      color=PURPLE, linestyle="--", linewidth=1.0, label=r"$\varepsilon$ (Level 2)")
ax.axhline(EPSILON / 10, color=BLUE,   linestyle="--", linewidth=1.0, label=r"$\varepsilon/10$ (Level 1)")
ax.set_xlabel(r"Fock dimension $N_\mathrm{Fock}$")
ax.set_ylabel(r"top-Fock population $p_\mathrm{top}$")
ax.set_title(r"Thermal tail vs truncation ($\bar{n}=0.5$, $\varepsilon=10^{-4}$)")
ax.legend(frameon=False)
plt.show()
```

**Takeaway.** With the carrier held on resonance the Fock distribution never
moves, so `p_top` is a fixed, geometrically-falling function of `N_Fock` alone —
crossing each dashed line drops the solve one level down the ladder.

A thermal initial state is a **density matrix**, not a ket —
solvers will dispatch to `qutip.mesolve` rather than `sesolve`,
which is required for any mixed-state input.

## Level 3 — the hard failure (ConvergenceError)

Start with `N_Fock = 5` — deliberately tight. The top-Fock
population is `P_thermal(4) ≈ 8.23e-3`, well above `10·ε = 1e-3`:

```python
from iontrap_dynamics.exceptions import ConvergenceError

hilbert, hamiltonian, rho_0 = build_scenario(n_fock=5)
_p_top_level3 = (N_BAR / (1 + N_BAR)) ** (5 - 1) / (1 + N_BAR)
print(f"Level 3 — N_Fock=5: theoretical p_top = {_p_top_level3:.3e}  (threshold 10·ε = {10*EPSILON:.1e})")
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

!!! warning "Common confusion — a norm of 1 is *not* a convergence check"

    An under-truncated state stays perfectly normalised. The mode is
    represented on the finite ladder `|0⟩…|N_Fock−1⟩`; population that should
    sit above the top rung is simply absent, yet `Tr ρ = 1` holds throughout
    the trajectory. So the norm can never reveal a chopped tail — read
    `p_top` (the guard), never the norm, to judge truncation.

## Level 2 — `FockQualityWarning`

`N_Fock = 7` brings `p_top` into the `[ε, 10·ε)` band:

```python
import warnings
from iontrap_dynamics.exceptions import FockQualityWarning

hilbert, hamiltonian, rho_0 = build_scenario(n_fock=7)
_p_top_level2 = (N_BAR / (1 + N_BAR)) ** (7 - 1) / (1 + N_BAR)
print(f"Level 2 — N_Fock=7: theoretical p_top = {_p_top_level2:.3e}  (ε/10={EPSILON/10:.1e} ≤ p_top < 10·ε={10*EPSILON:.1e})")
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

!!! warning "Common confusion — a Warning is flagged, an Error is refused"

    Level 1 and Level 2 emit *warnings*: the `result` object is returned and
    usable — it just carries a `result.warnings` record you must read before
    publishing. Level 3 raises `ConvergenceError`, so there is **no result** to
    catch. "Warned" means usable-but-flagged; "raised" means the answer was
    withheld until you widen `N_Fock`.

## Level 1 — `FockConvergenceWarning`

`N_Fock = 11` drops `p_top` into `[ε/10, ε)`:

```python
from iontrap_dynamics.exceptions import FockConvergenceWarning

hilbert, hamiltonian, rho_0 = build_scenario(n_fock=11)
_p_top_level1 = (N_BAR / (1 + N_BAR)) ** (11 - 1) / (1 + N_BAR)
print(f"Level 1 — N_Fock=11: theoretical p_top = {_p_top_level1:.3e}  (ε/10={EPSILON/10:.1e} ≤ p_top < ε={EPSILON:.1e})")
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
_p_top_ok = (N_BAR / (1 + N_BAR)) ** (13 - 1) / (1 + N_BAR)
print(f"Silent OK — N_Fock=13: theoretical p_top = {_p_top_ok:.3e}  (< ε/10={EPSILON/10:.1e}); result.warnings = {result.warnings}")
assert result.warnings == (), (
    "Silent OK means p_top < ε/10, so the solver attaches no per-mode record: "
    "an empty result.warnings tuple is the positive all-clear, not evidence "
    "that the Fock check was skipped."
)
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

# Sweep N_Fock from 5 to 15 and collect the measured p_top_max from
# result.warnings (or 0 when warnings is empty).  Overlay the theoretical
# thermal tail and the three ε thresholds to confirm the ladder.
_sweep_nfock = [5, 7, 9, 11, 13, 15]
_measured_ptop = []
for _nf in _sweep_nfock:
    _h, _ham, _rho = build_scenario(n_fock=_nf)
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _r = solve(hilbert=_h, hamiltonian=_ham, initial_state=_rho,
                       times=np.linspace(0.0, 1e-6, 20),
                       observables=[spin_z(_h, 0)])
        _pt = max((w.diagnostics["p_top_max"] for w in _r.warnings), default=0.0)
    except ConvergenceError as _exc:
        # Level 3 — extract p_top from the exception message string
        import re as _re
        _m = _re.search(r"p_top\s*=\s*([0-9.e+-]+)", str(_exc))
        _pt = float(_m.group(1)) if _m else 10 * EPSILON
    _measured_ptop.append(_pt)

print("\nConvergence-ladder sweep (measured p_top_max vs N_Fock):")
for _nf, _pt in zip(_sweep_nfock, _measured_ptop):
    _label = ("Level 3" if _pt >= 10 * EPSILON else
              "Level 2" if _pt >= EPSILON else
              "Level 1" if _pt >= EPSILON / 10 else "OK")
    print(f"  N_Fock={_nf:2d}: p_top_max = {_pt:.3e}  →  {_label}")

_theory_sweep = np.array([(N_BAR / (1 + N_BAR)) ** (nf - 1) / (1 + N_BAR) for nf in _sweep_nfock])
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.semilogy(_sweep_nfock, _theory_sweep, color=GREY, linestyle="--", linewidth=1.0, label="theory")
ax.scatter(_sweep_nfock, [max(pt, 1e-10) for pt in _measured_ptop], color=BLUE, zorder=3, label="measured $p_{\\mathrm{top}}$")
ax.axhline(10 * EPSILON, color=RED,    linestyle=":", linewidth=1.2, label=r"$10\varepsilon$ raise")
ax.axhline(EPSILON,      color=PURPLE, linestyle=":", linewidth=1.2, label=r"$\varepsilon$ Lv 2")
ax.axhline(EPSILON / 10, color=GREEN,  linestyle=":", linewidth=1.2, label=r"$\varepsilon/10$ Lv 1")
ax.set_xlabel(r"$N_\mathrm{Fock}$")
ax.set_ylabel(r"$p_\mathrm{top}$")
ax.set_title("Convergence ladder: measured vs threshold")
ax.legend(frameon=False)
plt.show()
```

**Takeaway.** Where a rung is flagged, the reported `p_top_max` — from
`result.warnings` for Levels 1–2, from the `ConvergenceError` message for Level 3 —
matches the theoretical thermal tail and trips each level at its `ε` threshold. The
silent-OK rungs report no value (they sit at the plot floor): an empty
`result.warnings` is the all-clear, not a measured zero.

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
    ```text
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
_p_top_tight = (N_BAR / (1 + N_BAR)) ** (13 - 1) / (1 + N_BAR)
print(f"Tightened-ε — N_Fock=13: p_top ≈ {_p_top_tight:.3e};  default ε=1e-4 → silent,  tightened ε=1e-6 → Level 2/3")
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
