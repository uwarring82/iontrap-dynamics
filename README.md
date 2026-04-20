# iontrap-dynamics

Open-system quantum dynamics of trapped-ion spin-motion systems.

`iontrap-dynamics` is a domain-specific Python library for modelling trapped-ion
spin-motion physics with explicit, typed configuration objects for species,
drives, modes, and measurement conventions. The project is being built around a
hard separation between physics, apparatus, and observation layers, with QuTiP
as the Phase 0 reference backend.

## Status

Phase 0 and the Phase 1 *core* (configuration, builders, observables, state
preparation, diagnostics) have shipped on `main` and are part of the
`v0.1-alpha` cut per `WORKPLAN_v0.3.md` §5.0 (release-mapping amendment,
v0.3.1). The Phase 1 *measurement layer* opened at Dispatch H and is
in-flight toward `v0.2`; its conventions are staged, not yet frozen
(`CONVENTIONS.md` §17). End-to-end dynamics works
(`DriveConfig` → `carrier_hamiltonian` → `qutip.mesolve` → expected π-pulse
flip); end-to-end *measurement* works
(`TrajectoryResult` → `p_↑` → `BernoulliChannel` → `MeasurementResult`).

Phase 0 artefacts (all done):

- Public conventions locked in `CONVENTIONS.md` v0.1-draft; §17 (measurement
  layer) opened as staged rules.
- Three-layer regression harness populated: migration (5 / 5 scenarios with
  legacy `qc.py`-generated references, bit-identical across three runs;
  2 / 5 active comparisons, 3 / 5 skipped with probe-informed blockers),
  analytic (6 closed-form formulas), invariant (9 checks).
- Cache-integrity contract + the corresponding tests.
- CI with ruff, ruff-format, mypy strict, pytest, pa11y WCAG 2 Level A
  (hard gate, AA advisory per the v0.3.2 amendment).

Today the importable code surface covers:

**Foundation (Phase 0)**

- `iontrap_dynamics.exceptions` — canonical exception hierarchy
  (`IonTrapError`, `ConventionError`, `BackendError`, `IntegrityError`,
  `ConvergenceError`)
- `iontrap_dynamics.results` — frozen `TrajectoryResult` schema with
  storage-mode consistency enforcement
- `iontrap_dynamics.cache` — hash-verified `.npz` + JSON persistence
- `iontrap_dynamics.conventions` — `CONVENTION_VERSION` marker
- `iontrap_dynamics.invariants` — density-matrix / state-vector validators
- `iontrap_dynamics.analytic` — closed-form reference formulas (carrier
  Rabi, sideband rates, Lamb–Dicke parameter, coherent-state occupation)

**Configuration layer (Phase 1)**

- `iontrap_dynamics.operators` — single-ion Pauli set in the atomic-physics
  convention (`sigma_z_ion`, `sigma_plus_ion`, ...; see CONVENTIONS.md §3)
- `iontrap_dynamics.species` — `IonSpecies`, `Transition`, `TransitionType`
  and factories for ²⁵Mg⁺, ⁴⁰Ca⁺, ⁴³Ca⁺
- `iontrap_dynamics.drives` — `DriveConfig` (wavevector, Rabi, detuning, ...)
- `iontrap_dynamics.modes` — `ModeConfig` with CONVENTIONS.md §11
  normalisation enforced at construction
- `iontrap_dynamics.system` — `IonSystem` composition with cross-validation
- `iontrap_dynamics.hilbert` — `HilbertSpace` implementing the §2 tensor
  ordering, operator embedding helpers, motional primitives (a, a†, n̂)
- `iontrap_dynamics.states` — `ground_state` ket, `compose_density`, plus
  `coherent_mode` / `squeezed_vacuum_mode` / `squeezed_coherent_mode`
  factories (CONVENTIONS.md §6, §7)
- `iontrap_dynamics.observables` — named `Observable` records and
  `expectations_over_time(...)`; spin x/y/z plus mode number

**Dynamics (Phase 1, full builder family)**

The public Hamiltonian surface is symmetric across four families:

|                | exact (time-indep. Qobj)   | detuned (list format)                |
|----------------|----------------------------|--------------------------------------|
| carrier        | `carrier_hamiltonian`      | `detuned_carrier_hamiltonian`        |
| red sideband   | `red_sideband_hamiltonian` | `detuned_red_sideband_hamiltonian`   |
| blue sideband  | `blue_sideband_hamiltonian`| `detuned_blue_sideband_hamiltonian`  |
| MS gate        | `ms_gate_hamiltonian`      | `detuned_ms_gate_hamiltonian`        |

Plus `modulated_carrier_hamiltonian` (time-dependent envelope primitive),
`two_ion_{red,blue}_sideband_hamiltonian` (single-tone shared-mode), and a
`full_lamb_dicke: bool` flag on the sideband builders (Wineland–Itano
all-orders operator via matrix exponentiation). Solver entry point:
`iontrap_dynamics.sequences.solve(...)` — accepts both Qobj and QuTiP
list-format Hamiltonians, enforces the §13 Fock-saturation ladder on
every call.

**Measurement (Phase 1, v0.2 — staged)**

- `iontrap_dynamics.results.MeasurementResult` — `Result` sibling carrying
  the ideal / sampled dual-view mandated by WORKPLAN §5; enforces
  `shots ≥ 1` and `storage_mode = OMITTED` at construction.
- `iontrap_dynamics.measurement.BernoulliChannel` — per-shot Bernoulli
  sampler; returns `(shots, n_inputs)` bits with leading shot axis
  (CONVENTIONS.md §17.1).
- `iontrap_dynamics.measurement.BinomialChannel` — aggregated sampler;
  returns `(n_inputs,)` int64 counts (CONVENTIONS.md §17.7, shape
  classes). Distributionally equivalent to summing Bernoulli bits, not
  bit-identical under matched seed.
- `iontrap_dynamics.measurement.PoissonChannel` — per-shot photon-
  counting channel; consumes non-negative rates (not probabilities) and
  returns `(shots, n_inputs)` int64 counts via `rng.poisson(λ)`.
- `iontrap_dynamics.measurement.DetectorConfig` — detector-response
  parameters (efficiency `η`, dark-count rate `γ_d`, threshold `N̂`)
  with `apply(rate)`, `discriminate(counts)`, and
  `classification_fidelity(lambda_bright, lambda_dark)` methods.
  Composes explicitly with `PoissonChannel` via Poisson thinning plus
  additive background (exact; §17.8).
- `iontrap_dynamics.measurement.sample_outcome(channel, inputs=...,
  shots, seed, upstream)` — orchestrator that seeds the RNG, dispatches
  uniformly on channel type, stores ``inputs`` under
  ``ideal_outcome[channel.ideal_label]`` (``"probability"`` or
  ``"rate"``), and inherits upstream-trajectory metadata when supplied.

Dispatches M–P extend the layer with protocols (spin readout, parity
scan, sideband-flopping inference) and statistics; `CONVENTIONS.md`
§17 seals at v0.2 under a Convention Freeze gate.

**Demo tools** (`tools/run_*.py` with canonical `manifest.json` +
`arrays.npz` + `demo_report.json` artefacts under `benchmarks/data/`):
`run_benchmark_sideband`, `run_demo_carrier`, `run_demo_gaussian_pulse`,
`run_demo_ms_gate`, `run_demo_bernoulli_readout`, `run_demo_binomial_readout`,
`run_demo_poisson_readout`, `run_demo_detected_readout`.

Test suite: **569 passed, 3 skipped**. Skips are migration-tier
builder-comparison slots with probe-informed blockers (see `CHANGELOG.md`).

Docs site scaffold:

- `mkdocs.yml` configures the public-facing documentation build
- `docs/index.md` — welcome page
- `docs/getting-started.md` — install + first run
- `docs/framework.md` — high-level design rules
- `docs/conventions.md` — rendered live from root `CONVENTIONS.md`
  (single source of truth via `pymdownx.snippets`)
- `docs/phase-1-architecture.md` — concrete public-API reference
  (module map, per-module surface, extension points, non-goals)
- `docs/boundary-decision-tree.md` — contributor scope rules (closes D8)
- `docs/tutorials/index.md` — tutorials landing page (placeholder; content
  arrives in Phase 1.E)
- `docs/stylesheets/tokens.css` — vendored from `threehouse-plus-ec/cd-rules`

The authoritative project documents are:

- `WORKPLAN_v0.3.md` for scope, architecture, milestones, and governance
- `CONVENTIONS.md` for physical, numerical, and notational rules
- `LICENCE` for the repository split-licence declaration

## Scope

Planned capabilities include:

- Unitary and dissipative dynamics for coupled spin-motion systems
- Standard ion-trap Hamiltonians: carrier, sideband, Mølmer-Sørensen,
  parametric modulation, and stroboscopic drives
- Standard state preparations and observables for spins and motional modes
- Backend-agnostic architecture, with QuTiP first and JAX/Dynamiqs later

Explicitly out of scope:

- Trap geometry simulation
- Molecular dynamics of ion crystals
- Pulse-sequence compilers and hardware-control stacks
- Electromagnetic field modelling

## Development

Python 3.11+ is required.

Editable install:

```sh
python -m pip install -e ".[dev]"
```

Optional groups:

- `.[docs]` for documentation tooling
- `.[plot]` for plotting helpers used by examples and tutorials
- `.[jax]` for the future JAX/Dynamiqs backend track

Example:

```sh
python -m pip install -e ".[dev,docs]"
```

## Repository Layout

- `src/iontrap_dynamics/` — Python package (core + `measurement/` subpackage)
- `tests/` — `unit/`, `conventions/`, `regression/{analytic,invariants,migration}/`,
  `benchmarks/`
- `tools/` — maintenance scripts (asset fetch / checksum, SPDX check, pa11y
  config, migration-reference generator) and demo runners
- `benchmarks/data/` — canonical manifest + arrays + report artefacts per
  demo / benchmark
- `docs/` — mkdocs-material source for the documentation site
- `assets/` — design assets consumed from `threehouse-plus-ec/cd-rules`
- `legacy/` — pinned legacy `qc.py` used by migration-tier regression
- `WORKPLAN_v0.3.md` — project workplan (v0.3.2 amendments applied)
- `CONVENTIONS.md` — binding conventions document (§17 staged)
- `CHANGELOG.md` — Keep-a-Changelog log of dispatches on `main`

## Licence

The distributable Python package is MIT-licensed. The repository as a whole
uses a split-licence architecture declared in `LICENCE`; design documents and
tutorial material do not all share the same terms.

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
