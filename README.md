# iontrap-dynamics

Open-system quantum dynamics of trapped-ion spin-motion systems.

> **New here?** See [docs/overview.md](docs/overview.md) for an
> accessible introduction to what we do, how we work, and what we
> don't claim — written with three reading levels for students,
> collaborators, and curious visitors.

`iontrap-dynamics` is a domain-specific Python library for modelling trapped-ion
spin-motion physics with explicit, typed configuration objects for species,
drives, modes, and measurement conventions. The project is built around a hard
separation between physics, apparatus, and observation layers. QuTiP is the
reference backend; a JAX / Dynamiqs backend is available on the same public
solver surface via `backend="jax"`.

## Status

**v0.5.0 released 2026-06-03 — adds two application-agnostic service
surfaces (estimation / quantum Darwinism, and two-mode / motional
open-system primitives) plus a device-neutral JAX spectrum backend, all
under the v0.3 Convention Freeze.** Phase 0 foundations, Phase 1
(dynamics core + measurement layer + systematics layer + registered
entanglement observables), Phase 2 (performance track + JAX / Dynamiqs
backend end-to-end), the Clos 2016 integration (full-Lamb–Dicke carrier
dynamics, exact-diagonalization spectrum tools, N = 1 / 2 / 3 reproduction
inside declared tolerances), and the `v0.5.0` estimation + two-mode /
motional surfaces are all shipped on `main`. `CONVENTIONS.md` is frozen at
**v0.3** (§1–24; §19–24 — Fisher information / quantum Darwinism, two-mode
squeezing / SU(1,1), and motional CPTP channels — added at the v0.3
Convention Freeze, `CONVENTION_VERSION` 0.2 → 0.3). **Additive only:**
existing `v0.2.0` callers see no behaviour change — every new `backend=`
kwarg defaults to `"qutip"`, every `channels=` argument defaults to none,
every JAX entry is behind an opt-in `backend="jax"`.

Post-`v0.5.0` on `main` (carried in `CHANGELOG.md` `[Unreleased]`, awaiting
the next tag), the two-mode / motional service surface is **complete**:
interferometric fringe observables (MCD), Lamb–Dicke regime helpers —
Debye–Waller factor, a deep/intermediate/beyond classifier, and the
all-orders sideband-Rabi forms (MCE), a probe-QFI benchmark that consumes
the estimation primitive (MCF), and a motional `ModeFrequencyDrift`
systematic (MCG) — all additive under v0.3.

End-to-end stacks work dynamics-through-statistics, on either backend:

- `DriveConfig` → `carrier_hamiltonian` → `solve(...)` → expected
  π-pulse flip (Phase 1 core; QuTiP default, or `backend="jax"` for
  the Dynamiqs path — cross-backend agreement under 1e-3 across all
  builder families).
- `TrajectoryResult` → `SpinReadout.run` → `MeasurementResult` →
  Wilson CI band on finite-shot estimator (Phase 1 measurement).
- Base `DriveConfig` → `perturb_carrier_rabi` → ensemble of solves
  → inhomogeneous-dephasing signature (Phase 1 systematics).
- `detuned_carrier_hamiltonian(..., backend="jax")` → `solve(..., backend="jax")`
  → `TrajectoryResult(backend_name="jax-dynamiqs")` with `StorageMode.LAZY`
  per-index loader (Phase 2 JAX backend end-to-end).

Phase 0 + Phase 1 + Phase 2 artefacts delivered:

- Public conventions frozen in `CONVENTIONS.md` v0.3 (§1–24 complete;
  §19–24 added at the v0.3 Convention Freeze, post-`v0.4.0`).
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
  Rabi, sideband rates, Lamb–Dicke parameter, coherent-state occupation),
  plus the Lamb–Dicke regime helpers (`debye_waller_factor`,
  `lamb_dicke_regime` → `LambDickeRegime`, all-orders
  `{red,blue}_sideband_rabi_frequency_full_ld`)

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
- `iontrap_dynamics.states` — `ground_state` ket, `compose_density`,
  `coherent_mode` / `squeezed_vacuum_mode` / `squeezed_coherent_mode`
  factories (CONVENTIONS.md §6, §7), the two-mode `two_mode_squeezed_vacuum`
  factory (§23), and the `ghz_state` / `cat_mode` entangled-probe factories
- `iontrap_dynamics.observables` — named `Observable` records and
  `expectations_over_time(...)`; spin x/y/z, multi-ion parity
  (`σ_z` product), mode number, plus the interferometric fringe analysis
  `fringe_visibility` / `fit_fringe` → `FringeFit`
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
  `apply_*_drift` composition helpers — plus the motional
  `ModeFrequencyDrift` (`ω_m → ω_m·(1+δ)` on a `ModeConfig`) and the
  correlated `CommonModePhase` channel. SPAM primitives
  `SpinPreparationError`, `ThermalPreparationError` produce
  per-subsystem density matrices via `imperfect_spin_ground` /
  `imperfect_motional_ground` that compose into a full initial
  state via `states.compose_density`.
- `iontrap_dynamics.information` — the estimation surface (§19–20):
  SLD quantum Fisher information, classical Fisher information,
  Cramér–Rao bound and a linear-Gaussian oracle
  (`quantum_fisher_information_trajectory`, `classical_fisher_information`,
  `cramer_rao_bound`), plus quantum-Darwinism redundancy and
  Schumacher–Nielsen recoverability.

**Dynamics (Phase 1, full builder family)**

The public Hamiltonian surface is symmetric across four families:

|                | exact (time-indep. Qobj)   | detuned (list format)                |
|----------------|----------------------------|--------------------------------------|
| carrier        | `carrier_hamiltonian`      | `detuned_carrier_hamiltonian`        |
| red sideband   | `red_sideband_hamiltonian` | `detuned_red_sideband_hamiltonian`   |
| blue sideband  | `blue_sideband_hamiltonian`| `detuned_blue_sideband_hamiltonian`  |
| MS gate        | `ms_gate_hamiltonian`      | `detuned_ms_gate_hamiltonian`        |

Plus `modulated_carrier_hamiltonian` (time-dependent envelope primitive),
`two_ion_{red,blue}_sideband_hamiltonian` (single-tone shared-mode), the
two-mode `two_mode_squeezing_hamiltonian` / `beamsplitter_hamiltonian`
(SU(1,1) / SU(2), §23), and a `full_lamb_dicke: bool` flag on the sideband
builders (Wineland–Itano all-orders operator via matrix exponentiation).
Solver entry point: `iontrap_dynamics.sequences.solve(...)` — accepts both
Qobj and QuTiP list-format Hamiltonians, enforces the §13 Fock-saturation
ladder on every call, and accepts typed motional CPTP channels via
`solve(channels=…)` (`iontrap_dynamics.channels`: `AmplitudeDamping`,
`Heating`, `Dephasing`, with optional time windows; §24).

**Measurement (Phase 1, v0.2 — frozen)**

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
close the read-through, with §17 frozen at the `v0.2.0` Convention
Freeze gate.

**Phase 2 — performance and JAX backend (v0.3)**

- `sequences.solve(...)` gains a `backend: str = "qutip"` keyword-
  only parameter; `backend="jax"` dispatches to the Dynamiqs
  integrator via `iontrap_dynamics.backends.jax`. Solver / backend
  compatibility is validated (explicit `solver="sesolve"` or
  `"mesolve"` with `backend="jax"` raises `ConventionError`; only
  `solver="auto"` is accepted on the JAX path).
- `iontrap_dynamics.backends.jax.solve_via_jax(...)` — opt-in
  JAX-backend entry. Dispatches to `dynamiqs.sesolve` for ket
  inputs, `dynamiqs.mesolve` for density matrices. Honours all
  three `StorageMode` values (OMITTED / EAGER / LAZY — the LAZY
  loader closes over the Dynamiqs-returned JAX array, materialises
  one `Qobj` per index on demand, bounds-checks against JAX's
  silent-clamp behaviour). Forces JAX x64 at solve entry for
  complex128 arithmetic (CONVENTIONS.md §1 unit commitment).
- Results on the JAX backend are tagged with
  `ResultMetadata.backend_name="jax-dynamiqs"` and
  `backend_version=f"dynamiqs-{ver}+jax-{ver}"`. The
  `backend_name` string is a **schema-commitment tag**: a future
  integrator swap requires a new string (not a suffix), so users'
  cache manifests stay consistent. `convention_version` is read
  from `hilbert.system.convention_version`, honouring archival
  pins exactly like the QuTiP path.
- Every time-dependent Hamiltonian builder gains a `backend=`
  kwarg on the same pattern (carrier, RSB, BSB, MS gate,
  modulated carrier). The four structured detuning builders share
  `backends.jax._coefficients.timeqarray_cos_sin` for
  `dq.modulated(cos, H_static) + dq.modulated(sin, H_quadrature)`
  assembly; `modulated_carrier_hamiltonian` takes a user-supplied
  `envelope_jax` keyword (JAX-traceable mirror of `envelope`) for
  arbitrary pulse shapes. Missing `envelope_jax` on
  `backend="jax"` raises `ConventionError` — no silent
  translation.
- Cross-backend numeric equivalence validated at library-default
  integrator tolerances across all five time-dependent builders:
  worst-case 1.35e-5 absolute expectation delta, well under the
  1e-3 design-target tolerance. Honest performance null result
  at dim ≥ 100 / 5000 steps (QuTiP 5 is ~2.8× faster than
  Dynamiqs + JAX on CPU at current library scales; see
  `docs/benchmarks.md`). The JAX backend's value is positioning,
  cross-backend consistency checking, and forward-looking
  capability (autograd scaffolding ready via `envelope_jax`
  coefficients; GPU / TPU dispatch if the user installs a
  CUDA / Metal JAX build).

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
Phase 2 sparse-matrix-tuning open item),
`run_benchmark_jax_timedep` (Dispatch YY / β.4.5 — cross-backend
QuTiP-vs-JAX at dim ≥ 100 / 5000 steps across all five
time-dependent builders; needs the `[jax]` extras).

Test suite on current `main` (v0.5.0 + the post-v0.5.0 two-mode / motional
surface):

- **Base CI (no extras): 1159 passed, 3 skipped.** Two skipped
  tests are migration-tier builder-comparison slots (scenarios 3
  and 4) with probe-informed blockers (see `CHANGELOG.md`); one
  skipped module is the Dynamiqs-gated integration test file
  (`tests/unit/test_backends_jax_dynamiqs.py`, gated on
  `pytest.importorskip("dynamiqs")`).
- **With `[jax]` extras: 869 passed, 2 skipped.** Adds the 49
  Dynamiqs-gated integration tests covering cross-backend
  numeric equivalence, result metadata, storage modes,
  Fock-saturation check, time-dependent builders, user-envelope
  dual-callable contract, and `solve_ensemble` on JAX.

Docs site scaffold:

- `mkdocs.yml` configures the public-facing documentation build
- `docs/index.md` — welcome page
- `docs/getting-started.md` — install + first run
- `docs/framework.md` — high-level design rules
- `docs/conventions.md` — rendered live from root `CONVENTIONS.md`
  (single source of truth via `pymdownx.snippets`)
- `docs/phase-1-architecture.md` — concrete public-API reference
  (module map, per-module surface, extension points, non-goals;
  now also hosts the result-family vs backend-variety decision
  record from D5 / Dispatch NN)
- `docs/phase-2-jax-backend-design.md` — deliberation note for the
  Phase 2 JAX / Dynamiqs backend (design axes A-D, chosen Option
  β, ten open questions + their recorded answers)
- `docs/phase-2-jax-time-dep-design.md` — β.4 staging note for the
  time-dependent Hamiltonian track (Option X parallel JAX-native
  builders; 5-sub-dispatch plan; scope inventory)
- `docs/benchmarks.md` — honest performance baselines for every
  Phase 2 dispatch including the β.4.5 cross-backend benchmark at
  dim ≥ 100 / 5000 steps
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
- `WORKPLAN_v0.3.md` — project workplan (v0.3.5 amendments
  applied: §4.0 repo-hosting, §5.0 release-mapping, §5.1 v0.2
  release, §5.2 post-v0.2 on-`main`, §5.3 β.4 as v0.3.x
  follow-up)
- `CONVENTIONS.md` — binding conventions document (v0.3 frozen:
  §17 measurement and §18 systematics closed at the v0.2.0 release;
  §19–24 estimation/Darwinism + two-mode/motional closed at the v0.3
  Convention Freeze, post-`v0.4.0`)
- `CHANGELOG.md` — Keep-a-Changelog log of dispatches on `main`

## Licence

The distributable Python package is MIT-licensed. The repository as a whole
uses a split-licence architecture declared in `LICENCE`; design documents and
tutorial material do not all share the same terms.

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
