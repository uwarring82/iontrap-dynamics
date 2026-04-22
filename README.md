# iontrap-dynamics

Open-system quantum dynamics of trapped-ion spin-motion systems.

`iontrap-dynamics` is a domain-specific Python library for modelling trapped-ion
spin-motion physics with explicit, typed configuration objects for species,
drives, modes, and measurement conventions. The project is being built around a
hard separation between physics, apparatus, and observation layers, with QuTiP
as the Phase 0 reference backend.

## Status

**v0.2.0 released 2026-04-21.** Phase 0, Phase 1 core (configuration,
builders, observables, state preparation, diagnostics), the Phase 1
measurement layer (channels, detector, protocols, statistics), the
Phase 1 systematics layer (jitter, drift, SPAM), and the registered
entanglement observables (concurrence, EoF, log-negativity) are all
shipped on `main`. `CONVENTIONS.md` is frozen at v0.2 — §1–16 from
the Phase 0 draft carry through unchanged; §17 (measurement) and §18
(systematics) are newly frozen.

End-to-end stacks work dynamics-through-statistics:

- `DriveConfig` → `carrier_hamiltonian` → `qutip.mesolve` → expected
  π-pulse flip (Phase 1 core).
- `TrajectoryResult` → `SpinReadout.run` → `MeasurementResult` →
  Wilson CI band on finite-shot estimator (Phase 1 measurement).
- Base `DriveConfig` → `perturb_carrier_rabi` → ensemble of solves
  → inhomogeneous-dephasing signature (Phase 1 systematics).

Phase 0 + Phase 1 artefacts delivered:

- Public conventions frozen in `CONVENTIONS.md` v0.2 (§1–18 complete).
- Three-layer regression harness populated: migration (5 / 5 scenarios
  with legacy `qc.py`-generated references, bit-identical across
  three runs; 3 / 5 active comparisons, 2 / 5 skipped with
  probe-informed blockers), analytic (6 closed-form formulas),
  invariant (9 checks).
- Cache-integrity contract + tests.
- CI with ruff, ruff-format, mypy strict, pytest, pa11y WCAG 2
  Level A (hard gate, AA advisory per v0.3.2 amendment).

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
  `expectations_over_time(...)`; spin x/y/z, multi-ion parity
  (`σ_z` product), plus mode number
- `iontrap_dynamics.entanglement` — nonlinear trajectory evaluators
  for `concurrence`, `entanglement_of_formation`, and
  `log_negativity` (with `partition="spins" | "modes"` for
  bipartite splits). Consume `TrajectoryResult.states` under
  `storage_mode=EAGER`
- `iontrap_dynamics.systematics` — dynamics-side noise models
  (§18 — frozen at v0.2). Jitter primitives `RabiJitter`
  (multiplicative on Ω), `DetuningJitter` (additive on δ),
  `PhaseJitter` (additive on φ) with `perturb_*` ensemble
  helpers. Parallel drift primitives `RabiDrift`, `DetuningDrift`,
  `PhaseDrift` (deterministic single-value offsets) with
  `apply_*_drift` composition helpers. SPAM primitives
  `SpinPreparationError`, `ThermalPreparationError` produce
  per-subsystem density matrices via `imperfect_spin_ground` /
  `imperfect_motional_ground` that compose into a full initial
  state via `states.compose_density`.

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
- `iontrap_dynamics.measurement.SpinReadout` — first protocol-layer
  composer. `.run(trajectory, shots, seed)` executes the projective-
  shot readout model (§17.9) and returns a `MeasurementResult` with
  per-shot counts / bits / bright-fraction plus the ideal `p_↑` and
  `TP · p_↑ + (1 − TN) · (1 − p_↑)` envelope.
- `iontrap_dynamics.measurement.ParityScan` — two-ion joint readout
  protocol (§17.10). Reconstructs `P(s_0, s_1)` from `⟨σ_z^i⟩`,
  `⟨σ_z^j⟩`, and `⟨σ_z^i σ_z^j⟩`, draws joint categorical samples so
  entangled-state correlations survive, and returns parity estimate
  + envelope shrunk by `(TP + TN − 1)²`. Requires the new
  `iontrap_dynamics.observables.parity` factory.
- `iontrap_dynamics.measurement.SidebandInference` — motional-state
  thermometry protocol (§17.11). Takes paired RSB / BSB trajectories
  and reports fidelity-corrected `n̄ = r / (1 − r)` via the
  short-time Leibfried–Wineland ratio; independent RNG streams
  per sideband; NaN propagates on singular ratios.
- `iontrap_dynamics.measurement.wilson_interval`,
  `clopper_pearson_interval`, and `binomial_summary` / the
  `BinomialSummary` dataclass (§17.12) — vectorised confidence
  intervals on binomial shot counts. Wilson is the recommended
  default; Clopper–Pearson is exact and conservative.

The measurement track is complete: `CONVENTIONS.md` §17.1–17.12
close the read-through, with §17 now frozen as the target for the
v0.2 Convention Freeze gate.

**Demo tools** (`tools/run_*.py` with canonical `manifest.json` +
`arrays.npz` + `demo_report.json` artefacts under `benchmarks/data/`):
`run_benchmark_sideband`, `run_demo_carrier`, `run_demo_gaussian_pulse`,
`run_demo_ms_gate`, `run_demo_bernoulli_readout`, `run_demo_binomial_readout`,
`run_demo_poisson_readout`, `run_demo_detected_readout`,
`run_demo_spin_readout`, `run_demo_parity_scan`,
`run_demo_sideband_inference`, `run_demo_wilson_ci`,
`run_demo_bell_entanglement`, `run_demo_rabi_jitter`,
`run_demo_detuning_jitter`, `run_demo_rabi_drift_scan`,
`run_demo_spam_prep`.

**Phase 2 benchmark tools** (`tools/run_benchmark_*.py` with
`report.json` + `plot.png` artefacts under `benchmarks/data/`):
`run_benchmark_sesolve_speedup` (Dispatch X — sesolve / mesolve
parity on QuTiP 5), `run_benchmark_ensemble_parallel` (Dispatch Y —
serial / loky / threading crossover), `run_benchmark_sparse_vs_dense`
(Dispatch OO — CSR / dense operator-dtype baseline; closes the
Phase 2 sparse-matrix-tuning open item).

Test suite: **795 passed, 2 skipped** (797 collected). Skips are
migration-tier builder-comparison slots (scenarios 3 and 4) with
probe-informed blockers (see `CHANGELOG.md`).

Docs site scaffold:

- `mkdocs.yml` configures the public-facing documentation build
- `docs/index.md` — welcome page
- `docs/getting-started.md` — install + first run
- `docs/framework.md` — high-level design rules
- `docs/conventions.md` — rendered live from root `CONVENTIONS.md`
  (single source of truth via `pymdownx.snippets`)
- `docs/phase-1-architecture.md` — concrete public-API reference
  (module map, per-module surface, extension points, non-goals)
- `docs/benchmarks.md` — honest performance baselines for every
  Phase 2 dispatch (closes the §5 Phase 2 `docs/benchmarks.md` item)
- `docs/boundary-decision-tree.md` — contributor scope rules (closes D8)
- `docs/tutorials/` — task-oriented walkthroughs. Twelve tutorials
  shipped (Tutorials 1–12 cover the full public-surface pipeline
  end-to-end, from carrier Rabi + Wilson CIs through two-ion
  Bell-state entanglement); see `docs/tutorials/index.md`
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
- Backend-agnostic architecture: QuTiP reference backend + JAX /
  Dynamiqs backend (opt-in via `backend="jax"` — see
  `docs/benchmarks.md` for when each is the right choice)

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
- `.[jax]` for the JAX / Dynamiqs backend (Phase 2 β.1–β.4 on
  `main`: opt in via `backend="jax"` on `solve` and on the
  time-dependent Hamiltonian builders)

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
- `WORKPLAN_v0.3.md` — project workplan (v0.3.4 amendments applied:
  §4.0 repo-hosting, §5.0 release-mapping, §5.1 v0.2 release,
  §5.2 post-v0.2 on-`main`)
- `CONVENTIONS.md` — binding conventions document (v0.2 frozen:
  §17 measurement and §18 systematics closed at the v0.2.0 release)
- `CHANGELOG.md` — Keep-a-Changelog log of dispatches on `main`

## Licence

The distributable Python package is MIT-licensed. The repository as a whole
uses a split-licence architecture declared in `LICENCE`; design documents and
tutorial material do not all share the same terms.

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
