# Phase 1 Architecture

This page is the concrete reference for the `iontrap-dynamics` Phase 1 public
API surface. It complements [Project Framework](framework.md) (design rules)
and [CONVENTIONS.md](https://github.com/uwarring82/iontrap-dynamics/blob/main/CONVENTIONS.md)
(binding physics conventions) with the question "**where does each capability
live, and how do the pieces compose?**"

## Module map

```text
iontrap_dynamics/
├── conventions.py    CONVENTION_VERSION, FOCK_CONVERGENCE_TOLERANCE
├── exceptions.py     IonTrapError family + IonTrapWarning family
├── operators.py      canonical Pauli set (atomic-physics convention, §3)
├── species.py        IonSpecies, Transition, factories (mg25_plus, …)
├── drives.py         DriveConfig (k-vector, Rabi, phase, detuning)
├── modes.py          ModeConfig (eigenvector per ion, §11 normalisation)
├── system.py         IonSystem — composes species + drives + modes
├── hilbert.py        HilbertSpace — spins-then-modes tensor ordering (§2)
├── states.py         ground_state, compose_density, coherent/squeezed mode kets
├── analytic.py       closed-form references (§0.B analytic-regression tier)
├── observables.py    Observable + spin_x/y/z / number factories
├── hamiltonians.py   all builders — four symmetric families (§5–§10)
├── sequences.py      solve() — the single public solver entry point
├── results.py        TrajectoryResult, ResultMetadata, StorageMode, warnings
└── cache.py          hash-verified .npz + JSON manifest I/O
```

Dependencies flow strictly downward in this list — each row uses only the rows
above it. `sequences.py` is the top layer that stitches everything together,
and nothing inside the package imports from `sequences` or `cache`.

## Public API by module

### `conventions.py`

| Symbol                         | Type    | Purpose                                                      |
|--------------------------------|---------|--------------------------------------------------------------|
| `CONVENTION_VERSION`           | `str`   | Tag recorded on every `TrajectoryResult.metadata`.           |
| `FOCK_CONVERGENCE_TOLERANCE`   | `float` | Default ε for the §13 Fock-saturation ladder; override per-call. |

### `exceptions.py`

Two shallow, stable hierarchies — one for hard failures (§15 Level 3), one for
soft diagnostics (Level 1 / 2):

```text
Exception
└── IonTrapError
    ├── ConventionError   — §3/§10/§11 etc. violations; bad API usage
    ├── BackendError      — backend-internal failures, unsupported requests
    ├── IntegrityError    — cache hash mismatch, invariant violations
    └── ConvergenceError  — §13 Level 3, step-size failures

UserWarning
└── IonTrapWarning
    ├── FockConvergenceWarning — §13 Level 1 (ε/10 ≤ p_top < ε)
    └── FockQualityWarning     — §13 Level 2 (ε ≤ p_top < 10·ε)
```

### `operators.py`

Canonical Pauli set on a 2-dim atomic subspace. Built from first principles
(no `qutip.sigmaz`) — see CONVENTIONS.md §3 on why.

- `spin_down()`, `spin_up()` — basis kets with `|↓⟩ = qutip.basis(2, 0)`
- `sigma_plus_ion()`, `sigma_minus_ion()` — raising/lowering
- `sigma_z_ion()` — `σ_z|↓⟩ = −|↓⟩, σ_z|↑⟩ = +|↑⟩`
- `sigma_x_ion()`, `sigma_y_ion()` — built so `σ_y = −i(σ_+ − σ_−)`

### `species.py`, `drives.py`, `modes.py`, `system.py`

Frozen, keyword-only dataclasses — inputs to the Hilbert-space construction.

- `IonSpecies(name, mass_amu, transitions)` — factories `mg25_plus()`,
  `ca40_plus()`, `ca43_plus()`.
- `DriveConfig(k_vector_m_inv, carrier_rabi_frequency_rad_s, detuning_rad_s,
  phase_rad, …)` — a laser tone.
- `ModeConfig(label, frequency_rad_s, eigenvector_per_ion)` — §11
  normalisation enforced.
- `IonSystem(species_per_ion, drives, modes, convention_version)` — cross-
  validates dimensions; `.homogeneous(species, n_ions, modes)` classmethod.

### `hilbert.py`

`HilbertSpace(system, fock_truncations)` — the spin-then-mode tensor product
with spin subsystems first (§2). Exposes:

- `spin_dim`, `mode_dim(label)`, `total_dim`, `subsystem_dims`, `qutip_dims()`
- `spin_op_for_ion(op, ion_index)` / `mode_op_for(op, mode_label)` —
  embed a single-subsystem operator onto the full space.
- `annihilation_for_mode` / `creation_for_mode` / `number_for_mode` —
  pre-embedded motional operators.
- `identity()` — full-space `qeye`.

### `states.py`

- `ground_state(hilbert)` — `|↓⟩^⊗N ⊗ |0⟩^⊗M` (cold start).
- `compose_density(hilbert, spin_states_per_ion, mode_states_by_label)` —
  full-space density matrix from per-subsystem kets or `Qobj` dms.
- `coherent_mode(fock_dim, alpha)` — `D(α)|0⟩` with §7 conventions.
- `squeezed_vacuum_mode(fock_dim, z)` — `S(z)|0⟩` with §6 conventions.
- `squeezed_coherent_mode(fock_dim, *, z, alpha)` — `D(α)S(z)|0⟩`,
  displace-after-squeeze order.

### `analytic.py`

Closed-form reference formulas, each documenting the exact CONVENTIONS.md
clause it implements:

| Function                             | Formula                                |
|--------------------------------------|----------------------------------------|
| `lamb_dicke_parameter`               | `η = (k·b) √(ℏ/2mω)` (§10, 3D)         |
| `carrier_rabi_excited_population`    | `sin²(Ωt/2)`                           |
| `carrier_rabi_sigma_z`               | `−cos(Ωt)`                             |
| `generalized_rabi_frequency`         | `√(Ω² + δ²)`                           |
| `detuned_rabi_sigma_z`               | `−1 + 2(Ω/Ω_gen)² sin²(Ω_gen t/2)`     |
| `red_sideband_rabi_frequency`        | `|η| √n Ω` (leading order)             |
| `blue_sideband_rabi_frequency`       | `|η| √(n+1) Ω`                         |
| `coherent_state_mean_n`              | `|α|²`                                 |
| `ms_gate_phonon_number`              | MS δ=0 coherent-displacement `⟨n⟩(t)`  |
| `ms_gate_closing_detuning`           | `δ = 2|Ωη|√K` (MS Bell condition)      |
| `ms_gate_closing_time`               | `t_gate = π√K/|Ωη|`                    |

### `observables.py`

Factory functions returning `Observable(label, operator)`:

- `spin_x(hilbert, ion_index)`, `spin_y(...)`, `spin_z(...)` — auto-labelled
  `sigma_{x,y,z}_{i}`; override with `label=...`.
- `number(hilbert, mode_label)` — labelled `n_{mode_label}`.
- `expectations_over_time(states, observables)` — `{label → np.ndarray}`.
- `Observable(label=..., operator=...)` — frozen dataclass; compose custom
  observables directly (e.g. the Bell-population projectors in the MS demo).

### `hamiltonians.py`

A symmetric 4-family surface across time-independent Qobj vs. time-dependent
QuTiP list format:

|                | exact (time-indep. Qobj)   | detuned (list format)                |
|----------------|----------------------------|--------------------------------------|
| carrier        | `carrier_hamiltonian`      | `detuned_carrier_hamiltonian`        |
| red sideband   | `red_sideband_hamiltonian` | `detuned_red_sideband_hamiltonian`   |
| blue sideband  | `blue_sideband_hamiltonian`| `detuned_blue_sideband_hamiltonian`  |
| MS gate        | `ms_gate_hamiltonian`      | `detuned_ms_gate_hamiltonian`        |

Plus:

- `modulated_carrier_hamiltonian` — time-dependent envelope primitive for
  pulse shaping (Gaussian, Blackman, stroboscopic square-wave, …).
- `two_ion_red_sideband_hamiltonian` / `two_ion_blue_sideband_hamiltonian` —
  single-tone shared-mode drives on two ions (physically distinct from the
  bichromatic MS gate).
- **Full Lamb–Dicke flag** on the four sideband builders:
  `full_lamb_dicke: bool = False`. `True` replaces the leading-order `η·a` /
  `η·a†` with the all-orders `P_{Δn=±1}(e^{iη(a+a†)})` Wineland–Itano
  operator — Debye–Waller reduction and Laguerre-polynomial Rabi rates.

### `sequences.py` — the one public solver entry point

```python
def solve(
    *,
    hilbert: HilbertSpace,
    hamiltonian: qutip.Qobj | list[object],   # Qobj or QuTiP list format
    initial_state: qutip.Qobj,                # ket or density matrix
    times: np.ndarray,
    observables: Sequence[Observable] = (),
    request_hash: str = "",
    backend_name: str = "qutip-mesolve",
    storage_mode: StorageMode = StorageMode.OMITTED,
    provenance_tags: tuple[str, ...] = (),
    fock_tolerance: float | None = None,
) -> TrajectoryResult: ...
```

What `solve()` does on every call:

1. Rejects `StorageMode.LAZY` (mesolve materialises eagerly anyway).
2. Runs `qutip.mesolve`; computes `expectations` over the full trajectory.
3. Enforces the §13 Fock-saturation ladder per mode (Level 1/2 → warnings on
   both channels; Level 3 → `ConvergenceError`).
4. Assembles `TrajectoryResult` with propagated metadata (convention version,
   backend name/version, storage mode, Fock truncations, provenance tags,
   request hash) and returns a frozen object.

### `results.py`

Frozen dataclasses only:

- `StorageMode` — `EAGER` / `LAZY` / `OMITTED`.
- `WarningSeverity` — `CONVERGENCE` / `QUALITY`.
- `ResultWarning(severity, category, message, diagnostics)`.
- `ResultMetadata(convention_version, request_hash, backend_{name,version},
  storage_mode, fock_truncations, provenance_tags)`.
- `Result` (base) and `TrajectoryResult(times, expectations, warnings, states,
  states_loader, metadata)`.

### `cache.py`

- `compute_request_hash(parameters)` → SHA-256 hex.
- `save_trajectory(result, path, overwrite=False)` — writes
  `manifest.json` (schema-validated metadata + warnings +
  expectation-label list) plus `arrays.npz` (`times` in SI seconds,
  observables under `expectation__<label>`).
- `load_trajectory(path, *, expected_request_hash)` — validates
  schema and hash; any mismatch raises `IntegrityError`.

All demo tools under `tools/` use this as their canonical output
path; pedagogical narrative lives alongside in a separate
`demo_report.json` so the canonical layout stays tight.

## End-to-end pipeline

One `solve()` call drives every module in the right order:

```python
from iontrap_dynamics.drives import DriveConfig
from iontrap_dynamics.hamiltonians import detuned_ms_gate_hamiltonian
from iontrap_dynamics.hilbert import HilbertSpace
from iontrap_dynamics.modes import ModeConfig
from iontrap_dynamics.observables import Observable, number, spin_z
from iontrap_dynamics.operators import spin_down
from iontrap_dynamics.species import mg25_plus
from iontrap_dynamics.sequences import solve
from iontrap_dynamics.system import IonSystem
from iontrap_dynamics.analytic import (
    lamb_dicke_parameter, ms_gate_closing_detuning, ms_gate_closing_time,
)
import qutip, numpy as np

# 1. Configuration
mode = ModeConfig(
    label="com",
    frequency_rad_s=2 * np.pi * 1.5e6,
    eigenvector_per_ion=np.array([[0, 0, 1], [0, 0, 1]]) / np.sqrt(2),
)
system = IonSystem(species_per_ion=(mg25_plus(), mg25_plus()), modes=(mode,))
hilbert = HilbertSpace(system=system, fock_truncations={"com": 12})

drive = DriveConfig(
    k_vector_m_inv=[0.0, 0.0, 2 * np.pi / 280e-9],
    carrier_rabi_frequency_rad_s=2 * np.pi * 0.1e6,
)

# 2. Derive gate parameters from the analytic helpers
eta = lamb_dicke_parameter(
    k_vec=drive.k_vector_m_inv,
    mode_eigenvector=mode.eigenvector_at_ion(0),
    ion_mass=mg25_plus().mass_kg,
    mode_frequency=mode.frequency_rad_s,
)
delta  = ms_gate_closing_detuning(
    carrier_rabi_frequency=drive.carrier_rabi_frequency_rad_s,
    lamb_dicke_parameter=eta, loops=1,
)
t_gate = ms_gate_closing_time(
    carrier_rabi_frequency=drive.carrier_rabi_frequency_rad_s,
    lamb_dicke_parameter=eta, loops=1,
)

# 3. Builder (returns a QuTiP list-format Hamiltonian)
H = detuned_ms_gate_hamiltonian(
    hilbert, drive, "com", ion_indices=(0, 1), detuning_rad_s=delta,
)

# 4. Initial state + observables
psi_0 = qutip.tensor(spin_down(), spin_down(), qutip.basis(12, 0))
obs = [number(hilbert, "com"), spin_z(hilbert, 0), spin_z(hilbert, 1)]

# 5. Solve
result = solve(
    hilbert=hilbert, hamiltonian=H, initial_state=psi_0,
    times=np.linspace(0, t_gate, 500), observables=obs,
)
```

This is exactly the structure of [`tools/run_demo_ms_gate.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/tools/run_demo_ms_gate.py).

## Extension points

### Adding a new Hamiltonian builder

1. Pick the symmetric slot it belongs to (carrier / sideband / MS / other).
2. Return a `qutip.Qobj` for time-independent or `list[object]` for QuTiP
   time-dependent list format (see `detuned_*` builders for the decomposition
   into `[[A, cos_fn], [B, sin_fn]]` pattern).
3. Accept `hilbert, drive, [mode_label,] *, ion_index (or ion_indices)` —
   consistent with the existing signatures.
4. Raise `ConventionError` on unphysical input, propagate `IndexError` on bad
   ion index.
5. Add an entry to the `__all__` list, the module docstring summary table,
   and the architecture table above.

### Adding a new named observable

In `observables.py`:

```python
def my_observable(hilbert, ..., *, label=None) -> Observable:
    op = ...  # build Qobj on full Hilbert space
    return Observable(label=label or f"my_default_{...}", operator=op)
```

Ad-hoc observables (e.g. two-ion population projectors) can be constructed
inline: `Observable(label="p_dd", operator=...)`.

### Adding a new state-prep factory

In `states.py`. Return a single-mode ket (dim = `fock_dim`) that callers
tensor into a full-space state via `compose_density`, or add a full-space
helper alongside `ground_state` if the factory is spin-dependent.

### Adding a new Phase 0.F benchmark slot

In `tests/benchmarks/test_performance_smoke.py`: add a fourth threshold
constant + test function. Use `time.perf_counter()` around the `mesolve`
call; threshold must be the published WORKPLAN_v0.3 §0.F number.

## Result family vs. backend variety (D5)

`iontrap-dynamics` separates two axes that look similar from a distance but
belong to different governance layers:

- **Result family** (subclass axis). `Result` is abstract; `TrajectoryResult`
  and `MeasurementResult` are concrete siblings. New subclasses are justified
  only when the *semantic contract* of the output changes — e.g. a planned
  `StochasticTrajectoryResult` (a batch of unravellings, not a single
  trajectory) will land as its own subclass, because downstream code must
  handle it structurally differently from a deterministic trajectory.
- **Backend variety** (metadata axis). `mesolve`, `sesolve`, a future
  sparse-path dispatch, and a future JAX / Dynamiqs backend all produce the
  *same* `TrajectoryResult` object. They identify themselves through
  `ResultMetadata.backend_name` + `backend_version`. Downstream code never
  branches on backend type; Design Principle 5 ("one way to do it at the
  public API level") applies. If a future backend needs more provenance than
  those two strings carry, extend `ResultMetadata` with an additional
  free-form field (mirroring `provenance_tags`) rather than subclassing the
  result.

**Rule for contributors adding a new Phase 2 backend.** Reuse
`TrajectoryResult`. Set `backend_name` to a unique string (e.g.
`"qutip-mesolve-sparse"`, `"jax-dynamiqs"`); add the string to any test that
asserts on backend identity. Do **not** introduce a `SparseTrajectoryResult`
or `JaxTrajectoryResult` subclass — these would fracture the public API and
force every analysis tool to branch. If a new backend genuinely returns a
different in-memory state object (e.g. a JAX pytree instead of a tuple of
`Qobj`), keep that representation inside the backend module and convert it
to the canonical `TrajectoryResult` at the `sequences.solve` boundary.

**Rule for contributors adding a new semantic output.** New subclass of
`Result`. Document the distinguishing contract in the docstring ("what does
this carry that `TrajectoryResult` doesn't?"). Cache I/O may need a new
writer — treat this as a Phase 1-style dispatch with its own tests and
`CHANGELOG.md` entry.

This is decision **D5** from Phase 0 planning. It is already recorded in the
[`results.py`](https://github.com/uwarring82/iontrap-dynamics/blob/main/src/iontrap_dynamics/results.py)
module docstring; it is surfaced here so it is discoverable from the
architecture doc before the first Phase 2 backend dispatch lands.

## Non-goals for Phase 1

Explicitly deferred to Phase 2 or later:

- **Stochastic unravellings** (Monte-Carlo trajectories, quantum jumps).
  Planned: `StochasticTrajectoryResult` alongside `TrajectoryResult`.
- **Apparatus + measurement layers** — drifts, calibration errors, detector
  models, finite-shot sampling. Planned for a Phase 1.C / 1.D dispatch and a
  `MeasurementResult` sibling to `TrajectoryResult`.
- **JAX / Dynamiqs backend**. Current `BackendError` + `backend_name`
  metadata already anticipate the multi-backend registry; implementation is
  scoped for Phase 2 per WORKPLAN_v0.3 §4.
- **Convention-set object** beyond the current `CONVENTION_VERSION` string.
  Planned for a dedicated convention-enforcement dispatch once more than one
  competing convention set is in play.
- **Cross-platform bit-identity** (`10⁻¹⁶` tier of CONVENTIONS.md §14) —
  requires a pinned dependency lockfile and reference-platform CI, deferred.

## Status at this mark

- **497 tests passing, 3 skipped** (migration-tier scenarios 2/3/4 with
  specific probe-informed blockers documented in the test module).
- **Ruff lint + format, mypy strict on 17 source files, pa11y Level A
  report-only** — all green in CI.
- **Phase 0.F benchmarks**: all three active and well under threshold
  (<5 s, <30 s, <60 s on the canonical hardware).
