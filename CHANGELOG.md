# Changelog

All notable changes to `iontrap-dynamics` will be documented in this file.

The format follows Keep a Changelog. Semantic Versioning has been adopted
as of `v0.2.0` (tagged 2026-04-21); pre-`v0.2.0` versioning was
placeholder-only and did not follow semver.

## [Unreleased]

### Added

#### Phase 2 — JAX backend Dynamiqs integrator (Dispatch β.2 / SS)

Replaces the β.1 `NotImplementedError` stub at
`iontrap_dynamics.backends.jax._core.solve_via_jax` with a real
Dynamiqs `sesolve` / `mesolve` integrator. The JAX backend now
produces a canonical `TrajectoryResult` end-to-end; cross-backend
equivalence against the QuTiP reference is exercised on carrier
Rabi at dim 24, 4 Rabi periods, under library-default integrator
tolerances.

**Empirical cross-backend agreement** (Dynamiqs 0.3.4 + QuTiP 5.2.3,
library-default `Tsit5` vs scipy's `solve_ivp` defaults): ~2e-5
max absolute expectation delta. Both backends at tight tolerances
(atol=rtol=1e-12): ~5e-11 agreement, confirming the two
integrators are mathematically equivalent and the default-vs-
default gap is integrator precision, not translation bug. The
cross-backend test asserts 1e-3 as a conservative margin against
integrator-version drift — see
`docs/phase-2-jax-backend-design.md` §7 α.4 benchmark scale notes.

- `solve_via_jax` wired end-to-end:
  - x64 forced at solve entry via `jax.config.update("jax_enable_x64", True)`.
    Process-wide side effect; documented in the module docstring.
    The library's CONVENTIONS.md §1 unit commitment is double
    precision, and JAX defaults to float32 — without the flip,
    Dynamiqs would silently run in complex64 and systematics would
    drift below the §13 Fock-saturation check threshold.
  - Ket inputs → `dq.sesolve(H, psi0, tsave, exp_ops=...)`.
  - Density-matrix inputs → `dq.mesolve(H, [], rho0, tsave, exp_ops=...)`
    with empty jump-ops list (unitary Lindblad; matches the QuTiP
    path's choice).
  - QuTiP `Qobj` operators and states pass through directly —
    Dynamiqs accepts them via `QArrayLike` duck typing. No explicit
    `Qobj.full() → jnp.asarray` conversion at this layer.
  - Progress-meter disabled (`dq.Options(progress_meter=None)`);
    library calls are not interactive.
- `StorageMode` handling per the design note §2:
  - `OMITTED` — no state materialisation; expectations computed
    JAX-side via `exp_ops` and NumPy arrays cross the boundary.
  - `EAGER` — each time slice converted to `qutip.Qobj` with
    the original tensor dims preserved from `initial_state.dims`
    (ket-shape or operator-shape). Tested at dim 24 × 200 time
    samples; round-trips `initial_state` within 1e-10.
  - `LAZY` — raises `NotImplementedError` with a β.3-scope message.
    The JAX-array lifetime and loader-per-access conversion
    semantics need a design pass before wiring.
- Fock-saturation check without state materialisation: top-Fock
  projectors per mode are piggybacked onto the Dynamiqs `exp_ops`
  list, so `p_top_by_mode` is obtained directly from the solver
  output. Classification uses the shared
  `sequences._classify_fock_saturation` (refactored out of
  `_fock_saturation_warnings` in this dispatch), producing the
  same §15 warning ladder + `ConvergenceError` on Level 3 as the
  QuTiP path. Tested: saturated coherent state at Fock=4 raises
  `ConvergenceError` identically.
- Metadata:
  - `backend_name` defaults to `"jax-dynamiqs"` (schema-commitment
    string per design note §4.2 — written into cache manifests).
    User override via the existing `backend_name` kwarg still works.
  - `backend_version` is `f"dynamiqs-{dq.__version__}+jax-{jax.__version__}"`.
  - `convention_version` inherited from the library's
    `CONVENTION_VERSION` constant (`"0.2"` post-Dispatch QQ).
- Time-dependent Hamiltonians (QuTiP list-format with callables /
  piecewise arrays) raise `NotImplementedError` — Dynamiqs requires
  its own `TimeQArray` wrapping that is β.3 scope.
- `sequences._fock_saturation_warnings` split into a QuTiP-path
  wrapper + a shared `_classify_fock_saturation(hilbert,
  p_top_by_mode, tolerance)` classifier. Behaviour-preserving
  refactor; all 808 existing tests pass unchanged.
- `tests/unit/test_backends_jax_dynamiqs.py` — new 13-test module
  gated on `pytest.importorskip("dynamiqs")`. Covers:
  cross-backend expectation equivalence (sesolve path + DM path),
  metadata surface (backend_name, backend_version format,
  convention_version inheritance, user override), storage modes
  (OMITTED non-materialisation, EAGER round-trip of kets + DMs),
  Fock-saturation `ConvergenceError` fires on JAX too,
  `NotImplementedError` on time-dependent and LAZY inputs,
  `TrajectoryResult` return type.
- `tests/unit/test_backends_jax.py`: removed two β.1-stub tests
  (`test_beta2_stub_raised_when_extras_present`,
  `test_beta2_stub_reachable_through_sequences_solve`) that
  exercised the NotImplementedError stub β.2 replaced. The
  install-hint tests (via monkey-patched availability) stay — they
  cover the no-extras path independently of β.2.

Test-surface growth: 808 → 821 passing with `[jax]` extras (13
new in `test_backends_jax_dynamiqs.py`); 808 → 806 passing without
extras (2 β.1 stub tests removed; the 13 new tests skip via
`importorskip`). CI behaviour: tests pass in both the default
CI (no extras) and the `[jax]`-enabled CI (extras installed).

No `pyproject.toml` change — `[jax]` extras already declared.
No `backend=` kwarg change — β.1's validator surface unchanged.

#### Phase 2 — JAX backend skeleton (Dispatch β.1 / RR)

Opens the JAX-backend track per `docs/phase-2-jax-backend-design.md`.
Design β (Dynamiqs) is the selected path — it is default-aligned
with the existing `[jax] = jax + jaxlib + dynamiqs>=0.2` extras
block in `pyproject.toml:85` and realises the `WORKPLAN_v0.3.md` §1
Dynamiqs-as-future-backend-target commitment directly. This
dispatch ships the skeleton only; the Dynamiqs integrator wiring is
scoped for Dispatch β.2 per the design note's §7 staging.

- `sequences.solve` and `sequences.solve_ensemble` gain a
  `backend: str = "qutip"` kwarg. Default value preserves the
  existing QuTiP dispatch behaviour — no user-visible change unless
  `backend="jax"` is explicitly passed.
- `iontrap_dynamics.backends` subpackage created; houses the JAX
  backend at `iontrap_dynamics.backends.jax` and is the registration
  point for future alternative backends per D5 / Design Principle 5
  (one public entry point, backend is an implementation choice).
- `iontrap_dynamics.backends.jax._core` ships the availability check
  (`_is_jax_available()` — importable JAX + Dynamiqs) and the solve
  stub (`solve_via_jax`). When the `[jax]` extras are missing, the
  stub raises `BackendError` with an actionable install hint
  (`pip install iontrap-dynamics[jax]`). When the extras are present
  but the integrator is not yet wired (β.1 → β.2), the stub raises
  `NotImplementedError` pointing at the β.2 scope.
- `_validate_backend(backend, solver)` helper centralises the
  backend-kwarg validation surface. Unknown backend strings raise
  `ConventionError` listing the valid options (`{'qutip', 'jax'}`).
  `solver=` stays QuTiP-specific per `docs/phase-2-jax-backend-design.md`
  §4.1: passing `solver="sesolve"` or `solver="mesolve"` together
  with `backend="jax"` raises `ConventionError` because those are
  QuTiP solver identifiers, not backend-agnostic semantics. On
  `backend="jax"`, `solver="auto"` is the only accepted value —
  matching the design note's rule that backend choice changes the
  implementation, not the public kwarg's contract.
- `tests/unit/test_backends_jax.py` — 18 new tests covering the
  dispatch plumbing: default-backend-is-qutip regression,
  unknown-backend rejection, solver/backend compatibility matrix
  (explicit sesolve/mesolve rejected on JAX, auto passes
  validation, QuTiP backend's existing solver contract unchanged),
  JAX availability & stub behaviour (install hint present, β.2
  NotImplementedError, `_is_jax_available` returns bool), and
  `solve_ensemble` kwarg propagation.

No `pyproject.toml` change in this dispatch — the `[jax]` extras
block already declares the required dependencies (one of the
reasons β was the default-aligned choice over α / α′). `backend="qutip"`
remains the default; users opt into `backend="jax"` explicitly.

#### β.1 post-review tightening (Dispatch RR.1)

Three small cleanups + two tests added after an architect-stance
review of the β.1 skeleton. Same-track follow-up; no user-visible
contract change.

- `_validate_backend` checks solver vocabulary through
  `_QUTIP_SOLVER_VALUES` *before* the JAX / QuTiP-solver
  compatibility rule. Previously `backend="jax", solver="banana"`
  raised the "QuTiP-specific" `ConventionError`; it now raises
  "unknown solver" first, which matches the actual failure mode.
  Explicit `"sesolve"`/`"mesolve"` still surface the "QuTiP-
  specific" message. Also closes the dead-constant status of
  `_QUTIP_SOLVER_VALUES` (defined but previously unread).
- `solve_via_jax` signature tightened from `**kwargs: Any` to the
  explicit keyword-only parameter list that mirrors
  `sequences.solve` (minus the `backend=` discriminator). Any
  future `solve()` → `solve_via_jax` kwarg-name mismatch now
  surfaces as a `TypeError` at call time rather than silently
  landing in `**kwargs` and being dropped. β.2 consumes these
  parameters in the Dynamiqs integrator call; β.1 ignores them
  after the availability check (via a single `del` binding).
- New test
  `TestJaxAvailabilityAndStub::test_beta2_stub_reachable_through_sequences_solve`
  covers the end-to-end dispatch path — the existing
  `test_beta2_stub_raised_when_extras_present` only exercised
  `solve_via_jax()` directly. The new test routes through
  `sequences.solve(backend="jax")` with mocked availability and
  confirms the kwarg-forwarding matches `solve_via_jax`'s locked
  signature.
- New `TestDispatchOrdering` class (2 tests) locking in the
  validation order: `StorageMode.LAZY + backend="jax"` must reach
  the JAX dispatch (not the QuTiP-only LAZY guard) so β.2 is free
  to implement JAX-side lazy loaders per the design note §2. The
  sibling test confirms `LAZY + backend="qutip"` still raises as
  before — the ordering fix is not a general LAZY-support change.

Test-surface growth: 805 → 808 passing (3 new).

### Fixed

#### Post-v0.2.0 metadata drift (Dispatch QQ)

Three hygiene fixes that address stale version strings left behind
when `v0.2.0` was tagged. None are user-visible behaviour changes;
all three correct provenance metadata written into results / cache
manifests / the project header.

- `CONVENTION_VERSION` in `src/iontrap_dynamics/conventions.py`
  bumped from `"0.1-draft"` to `"0.2"` — matches the
  `CONVENTIONS.md` v0.2 Convention Freeze that happened at
  `v0.2.0` (Dispatch W on 2026-04-21) but was not reflected in the
  runtime constant. Every `TrajectoryResult` produced between
  Dispatch W and this dispatch was tagged with the draft version
  string; new results carry `"0.2"` as intended by the docstring's
  bump policy ("additions are free; changes require a minor-version
  bump and a CHANGELOG entry").
- `backend_version` fallback in
  `src/iontrap_dynamics/measurement/channels.py` no longer hard-
  codes `"0.1.0.dev0"` (the pre-`v0.2.0` `pyproject.toml` version
  string that was never updated). The fallback now resolves the
  installed distribution version dynamically via
  `importlib.metadata.version("iontrap-dynamics")`, with a
  `PackageNotFoundError` fallback to `"unknown"` for editable-
  install contexts where distribution metadata may be unresolved.
  This path is only taken when a measurement is constructed
  without an upstream `ResultMetadata` (i.e. a synthetic / test
  measurement not attached to a solver trajectory).
- `CHANGELOG.md` header no longer says semver adoption is future-
  tense; the statement now records that semver is in effect from
  `v0.2.0`.

No test-surface change: existing tests import `CONVENTION_VERSION`
as a symbol and assert against the imported value, so the constant
bump propagates without test edits.

### Added

#### Phase 2 — csr-vs-dense ops baseline (Dispatch OO)

- `tools/run_benchmark_sparse_vs_dense.py` — measures CSR-vs-dense
  operator-dtype wall-clock across the four canonical Hamiltonian
  builders at eight Hilbert-space sizes (single-ion carrier / RSB /
  BSB at fock 4–60, two-ion RSB at fock 15). At library-typical
  Hilbert sizes (dim ≤ 60) the two dtypes tie within ~5 %; CSR
  pulls ahead to 1.36–1.49× at single-ion fock=60 (dim 120). Mean
  dense/csr ratio across the scenario set is 1.11×.
- `docs/benchmarks.md` gains a new "`csr` vs `dense` operator dtype
  (Dispatch OO)" section with the measured table, an interpretation
  of why CSR never loses materially, and a note that QuTiP 5's
  default CSR dtype already discharges the "sparse-matrix tuning"
  Phase 2 open item. The headline paragraph is updated to reflect
  that Dispatches X / Y / OO have landed and only the JAX backend
  remains open on the Phase 2 track. The "Open Phase 2 items"
  subsection at the bottom retires the sparse-matrix entry.
- Artefact committed under `benchmarks/data/sparse_vs_dense/`
  (`report.json` with per-scenario wall-clock + environment metadata,
  `plot.png` with the dense/csr ratio bar chart).
- **No public API change.** `sequences.solve` does not gain a
  `matrix_format` kwarg. Design Principle 5 ("one way to do it at the
  public API level") applies — CSR is the one way because it never
  loses. Users needing a dense representation for debugging can
  still call `hamiltonian.to("dense")` manually.

#### Phase 2 — performance track opener (Dispatch X)

- `sequences.solve` gains a ``solver`` kwarg (``"auto"`` default,
  explicit ``"sesolve"``/``"mesolve"`` overrides). ``"auto"``
  dispatches pure kets to :func:`qutip.sesolve` and density
  matrices to :func:`qutip.mesolve`; ``"sesolve"`` on a density
  matrix raises :class:`ConventionError` (Schrödinger evolves pure
  states only). The selected solver is recorded on the result's
  ``backend_name`` metadata (``"qutip-sesolve"`` or
  ``"qutip-mesolve"``); the default value of ``backend_name`` is
  now ``None`` (auto-resolve) rather than a hard-coded string.
- `tools/run_benchmark_sesolve_speedup.py` — Phase 2 baseline.
  Measures mesolve-vs-sesolve wall-clock across three Hilbert-space
  sizes and two sideband builders. **Empirical finding:** on QuTiP
  5.2 at library-scale Hilbert spaces (dim ≤ 48), sesolve is *not*
  faster than mesolve — the folklore 2–3× advantage from the
  QuTiP 4.x era has largely been closed. The dispatch opts into
  sesolve on ket inputs for semantic correctness; measurable wins
  are left to later Phase 2 dispatches (sparse ops, JAX).
- 8 new unit-test cases in `tests/unit/test_sequences.py` covering
  the dispatch matrix: auto-dispatch on ket/density-matrix,
  explicit sesolve/mesolve selection, sesolve-on-density-matrix
  rejection, unknown-solver rejection, numerical-equivalence
  cross-check between the two paths, backend-name override still
  wins over auto.
- Module docstring of `sequences.py` updated: "v0.1 wraps mesolve
  directly" → dispatcher documentation covering the solver choice,
  backend-name auto-tagging, and the QuTiP-5 performance baseline.

#### Phase 2 — parallel sweeps via joblib (Dispatch Y)

- `sequences.solve_ensemble(hilbert, hamiltonians, ...)` — batch
  API over a sequence of Hamiltonians, wrapping
  :class:`joblib.Parallel`. Returns `tuple[TrajectoryResult, ...]`
  in the same order as the input hamiltonians. Shared solve-kwargs
  (``initial_state``, ``times``, ``observables``, ``storage_mode``,
  ``solver``, etc.) apply to every trial; per-trial variation is
  encoded in the Hamiltonian list.
- Default `n_jobs=1` runs serially in the main process with zero
  joblib overhead — measured by the benchmark tool below as faster
  than any parallel backend for the typical small/medium ion-trap
  scenarios. `n_jobs=-1` with `parallel_backend="loky"` becomes
  faster once single-solve cost exceeds ~15 ms and n_steps ≥ 2000.
- `joblib>=1.3` promoted from the `[legacy]` extra to base
  dependencies (was the qc.py generator's only caller; now core).
- `tests/unit/test_sequences.py`: 8 new cases covering batch-API
  shape, empty-list rejection, output-order preservation across
  varied Hamiltonians, serial / loky bit-equivalence, shared-kwarg
  propagation, default-n_jobs serial execution.
- `tools/run_benchmark_ensemble_parallel.py`: measures loky /
  threading / sequential backends across three scale regimes.
  **Empirical findings:** at small (fock=3, 2.7 ms single-solve)
  loky is 22× slower than serial; at medium (fock=12, 6 ms) serial
  and loky tie; at large (fock=24, 16 ms, 2000 steps) loky gives
  a 2.68× speedup over serial. Threading hurts at large scale
  (Python-level stepper overhead).

#### Tutorials — first end-to-end walkthrough (Dispatch AA)

Closes the Dispatch F `docs/tutorials/` placeholder with a real
entry. The first tutorial is the canonical "Hello world" for the
library post-v0.2: an end-to-end pipeline that exercises every
architectural layer introduced through v0.2.

- `docs/tutorials/01_first_rabi_readout.md` (new) — carrier Rabi
  flopping with finite-shot readout and 95 % Wilson CIs. ~10 min
  read, ~1 s runtime. Covers: four-step pattern (configure → build
  → solve → read out), public-surface use of `IonSystem`,
  `DriveConfig`, `ModeConfig`, `HilbertSpace`, `carrier_hamiltonian`,
  `sequences.solve`, `SpinReadout`, `DetectorConfig`,
  `binomial_summary`. Uses the same parameter values as
  `tools/run_demo_wilson_ci.py` so readers can diff prose against
  the working script. Includes an inline plot (served from the
  committed `benchmarks/data/wilson_ci_demo/plot.png`).
- `docs/tutorials/index.md` updated from placeholder to live page.
  Moves the planned-topics list from a single block to "Available"
  (Tutorial 1) + "Planned" (Tutorials 2–12). Adds three new
  planned topics covering Wilson / Clopper–Pearson statistics,
  jitter ensembles, and two-ion Bell-state entanglement — each
  paralleling a specific demo tool.
- `mkdocs.yml` nav gains an explicit Tutorial 1 entry nested under
  "Tutorials" so the first tutorial is one click from the docs
  landing page.
- README: updated docs-site scaffold listing notes that
  `docs/tutorials/` is no longer a pure placeholder.

#### Tutorials — two-ion Bell-state entanglement (Dispatch LL)

Twelfth and **final** entry in the tutorials track. Closes the
track originally planned in Dispatch F (placeholder) and opened
in Dispatch AA (Tutorial 1).

- `docs/tutorials/12_bell_entanglement.md` (new) — takes the
  Tutorial 4 MS-gate scenario and reads it out through two
  complementary measurement surfaces: (1) `ParityScan` protocol
  with explicit detector-classification envelope, showing the
  ideal `⟨σ_z σ_z⟩(t_gate) = +1.0` attenuate to
  `parity_envelope = +0.9928` under the 85%-efficiency /
  0.3-dark-count detector, with a 500-shot Wilson estimator
  tracking it at `+0.992`; (2) three registered trajectory
  evaluators (`concurrence_trajectory`,
  `entanglement_of_formation_trajectory`,
  `log_negativity_trajectory`). Three-witness narrative: at
  t=0 all three measures are 0 (product state); mid-gate
  (t≈13.6 μs) log-negativity peaks at 1.31 (spin-motion
  entanglement) while spin-spin concurrence sits at only 0.27
  (reduced state is mixed because entangled with phonons); at
  `t_gate` concurrence = EoF = 1.0 (Bell state) and
  log-negativity drops back to 0 (motion has disentangled).
  All nine quoted numbers spot-checked against an actual
  two-solve run. Also clarifies two easy-to-confuse API
  details: `concurrence_trajectory` uses `ion_indices=(i, j)`
  (two-qubit-specific) while `log_negativity_trajectory` uses
  `partition="spins" | "modes"` (bipartite-generic). Closes
  with pointers to remaining library surface not directly
  covered by the twelve tutorials (SPAM, drift, sideband
  inference, factory contributions).
- `docs/tutorials/index.md` — Tutorial 12 promoted to
  *Available*; the "Planned" section is removed and the
  introductory prose rewritten to reflect the completed track.
- `mkdocs.yml` nav adds an explicit Tutorial 12 entry beneath
  Tutorials 1–11.

#### Tutorials — systematics jitter ensembles (Dispatch KK)

Eleventh entry in the tutorials track. First systematics-layer
tutorial (opens the §18 surface that's been covered only
peripherally up to now).

- `docs/tutorials/11_jitter_ensembles.md` (new) — walks the
  full shot-to-shot Rabi-jitter pipeline: `RabiJitter(σ=0.03)`
  spec → `perturb_carrier_rabi(shots=200, seed=...)`
  materialise → 200 Hamiltonians → `solve_ensemble(n_jobs=1)`
  batch-solve → stack-and-aggregate. Verifies the ensemble
  mean against the analytic Gaussian-envelope dephasing
  prediction `⟨σ_z⟩ = −cos(Ω̄t)·exp(−(σΩ̄t)²/2)` at three
  time points (t = 1 μs → −0.986 / −0.982; t = 4 μs → −0.801
  / −0.753; t = 10 μs → −0.242 / −0.169), all numbers taken
  from an actual ensemble run. Distinguishes three error
  channels (ensemble mean / std / SEM); records the
  `n_jobs=1` default with a performance note tying back to
  Dispatch Y's benchmark; closes with a `DetuningJitter`
  variation that produces an asymmetric Lorentzian-style
  dephasing envelope instead of the Gaussian one. Opening
  comparison table enumerates all three jitter primitives
  (`RabiJitter` / `DetuningJitter` / `PhaseJitter`) with
  their physical-source attribution and typical σ ranges.
- `docs/tutorials/index.md` — Tutorial 11 moved from
  *Planned* to *Available*; the planned-topics list
  renumbers accordingly.
- `mkdocs.yml` nav adds an explicit Tutorial 11 entry beneath
  Tutorials 1–10.

#### Tutorials — finite-shot statistics (Dispatch JJ)

Tenth entry in the tutorials track. Expands Tutorial 1's
single-CI step into the full finite-shot reporting surface.

- `docs/tutorials/10_finite_shot_statistics.md` (new) — deep
  dive on the measurement layer's three statistics functions
  (`wilson_interval`, `clopper_pearson_interval`,
  `binomial_summary`) and the `BinomialSummary` dataclass.
  Anchor table covers Wilson vs Clopper–Pearson 95 % CIs at
  seven canonical `(k, n)` points — numbers verified against
  the actual library output (e.g. `5/10` → Wilson
  `[0.2366, 0.7634]`, CP `[0.1871, 0.8129]`; `50/100` → both
  `[0.40, 0.60]`). Vectorised `binomial_summary` across a full
  200-point carrier-Rabi trajectory with the correct
  `spin_readout_bits.sum(axis=0)` → bright-count path (earlier
  draft had a non-existent `spin_readout_bright_count` key —
  caught and corrected against actual
  `measurement.sampled_outcome` output). No-Wald rationale,
  Wilson vs CP decision tree, and a
  `n_required ≥ z²·p(1-p) / Δ²` shot-budget sizing formula.
- `docs/tutorials/index.md` — Tutorial 10 moved from *Planned*
  to *Available*; the planned-topics list renumbers
  accordingly.
- `mkdocs.yml` nav adds an explicit Tutorial 10 entry beneath
  Tutorials 1–9.

#### Tutorials — squeezed / coherent state prep (Dispatch II)

Ninth entry in the tutorials track. Covers the three named
single-mode state factories and their full-space composition —
the last major physics primitive in the state-prep surface that
hadn't been walked through in prose yet.

- `docs/tutorials/09_squeezed_coherent_prep.md` (new) —
  `coherent_mode`, `squeezed_vacuum_mode`,
  `squeezed_coherent_mode` plus `compose_density`. Each
  factory's ⟨n̂⟩ formula spot-checked (|α|² = 4 for α=2;
  sinh²(r) = 1.381 for r=1; |α|² + sinh²(|ξ|) = 5.381 for the
  combined case); squeezing-quadrature variances verified
  (e^(±2r)/2 asymmetry); CONVENTIONS §6 `ξ = r·e^(2iφ)` phase
  factor-of-2 convention recorded, and CONVENTIONS §7
  squeeze-then-displace ordering (matching qc.py) explained
  with a note on why it is keyword-only. `compose_density`
  covered as the multi-subsystem composition helper —
  spin-per-ion list + mode-label-keyed mode-state mapping, with
  the ConventionError guardrails for missing / extra modes.
  Closes with a collapse-and-revive red-sideband scenario from
  `|↓, α = 2⟩` running through `full_lamb_dicke=True`: σ_z
  oscillation amplitude decays to σ(σ_z) ≈ 0.26 over 8 periods
  at ⟨n̂⟩ = 4, the Rabi-rate-spread dephasing signature absent
  from a pure-Fock start. Two variations suggested
  (squeezed-vacuum drive with even-only Fock support;
  heterogeneous motional prep on a two-mode system via
  `compose_density`).
- `docs/tutorials/index.md` — Tutorial 9 moved from *Planned*
  to *Available*; the planned-topics list renumbers accordingly.
- `mkdocs.yml` nav adds an explicit Tutorial 9 entry beneath
  Tutorials 1–8.

#### Tutorials — full Lamb–Dicke for hot-ion regimes (Dispatch HH)

Eighth entry in the tutorials track. Back to physics-focused
material after the two architectural-layer tutorials (FF, GG).

- `docs/tutorials/08_full_lamb_dicke.md` (new) — when the
  `full_lamb_dicke=True` flag on the sideband builders matters,
  and when leading-order is enough. Covers the Wineland–Itano
  closed form `Ω_{n,n−1}^full = Ω·|η|·e^(−η²/2)·
  √((n−1)!/n!)·L_{n−1}^(1)(η²)`; the `η²·n ≳ 0.1`
  rule-of-thumb crossover; a quantitative three-scenario
  comparison over the Tutorial 2 scenario with `n = 1, 5, 10`
  showing 3 % → 16 % → 30 % rate shortfall (Debye–Waller +
  Laguerre); the cost of flipping the flag (one-time mode-level
  matrix exponentiation at build time, zero solve-time
  overhead); the uniform applicability across every sideband
  builder in the library (red / blue / detuned / two-ion /
  MS-gate composition). Closes with a five-branch when-to-flip
  decision tree (pure Fock vs thermal start, MS gate tuning,
  sideband cooling cascades, publication-grade runs). Rate
  numbers spot-checked against the actual
  `_full_ld_lowering_single_mode` engine via a one-shot run
  and `scipy.special.eval_genlaguerre` cross-verification.
- `docs/tutorials/index.md` — Tutorial 8 moved from *Planned*
  to *Available*; the planned-topics list renumbers
  accordingly.
- `mkdocs.yml` nav adds an explicit Tutorial 8 entry beneath
  Tutorials 1–7.

#### Tutorials — hash-verified cache round-trip (Dispatch GG)

Seventh entry in the tutorials track. Second architectural-layer
tutorial (complementing the Dispatch FF diagnostics tutorial);
covers the persistence contract that every committed
`benchmarks/data/<scenario>/` bundle in the repo rides on.

- `docs/tutorials/07_cache_round_trip.md` (new) — end-to-end
  walk through `compute_request_hash`, `save_trajectory`,
  `load_trajectory` applied to the Tutorial 2 RSB scenario.
  Contents: the parameter-dict → SHA-256 hex → embedded-manifest
  contract as a single-picture data-flow; what belongs in the
  hash payload vs the demo-report sidecar (primary inputs in,
  derived quantities out); why `StorageMode.OMITTED` is the only
  supported cache input (no-pickle policy, backend-agnostic
  round-trip); bit-identical round-trip assertion across times +
  expectations + metadata + warnings; four independent
  `IntegrityError` failure-mode walkthroughs (mismatched
  expected hash, missing file, tampered manifest hash, extra
  npz key) each with the exact diagnostic the library emits;
  three practical use patterns (notebook skip-recompute try /
  `IntegrityError` / solve, committed reference bundles for CI
  diff-check, cross-process sharing as a reproducibility
  contract); and the "don't commit 1000-trial sweep bundles"
  caveat. Hash values and manifest contents spot-checked against
  an actual run — the `d3c81eef…` hash and the `"0.1-draft"`
  `convention_version` in the quoted manifest match what a
  reader running the tutorial verbatim will see.
- `docs/tutorials/index.md` — Tutorial 7 moved from *Planned* to
  *Available*; the planned-topics list renumbers accordingly.
- `mkdocs.yml` nav adds an explicit Tutorial 7 entry beneath
  Tutorials 1–6.

#### Tutorials — Fock truncation diagnosis (Dispatch FF)

Sixth entry in the tutorials track. First diagnostic-layer
tutorial (no new physics scenario; teaches a single
architectural layer end-to-end).

- `docs/tutorials/06_fock_truncation.md` (new) — walks a single
  scenario (thermal initial state `n̄ = 0.5`, static carrier)
  through all four CONVENTIONS §15 statuses by varying `N_Fock`
  alone: silent OK (`N = 13`), Level 1
  `FockConvergenceWarning` (`N = 11`), Level 2
  `FockQualityWarning` (`N = 7, 9`), Level 3 `ConvergenceError`
  raise (`N = 5`). Numbers spot-checked against the actual solver
  output — the quoted `p_top` values match the run-time results
  to the digit. Covers: reading the Python-warnings channel with
  `warnings.catch_warnings`; reading the structured
  `result.warnings` tuple with its machine-readable `diagnostics`
  dict (preferred for CI gates); the
  `fock_tolerance` per-call override for
  publication-grade tightening; the
  `fock_tolerance=0` ConventionError trap; a cross-ensemble
  worst-case `p_top_max` aggregation recipe; and a diagnosis
  recipe for turning a `ConvergenceError` message into a
  remediation plan.
- `docs/tutorials/index.md` — Tutorial 6 moved from *Planned*
  to *Available*; the planned-topics list renumbers accordingly.
- `mkdocs.yml` nav adds an explicit Tutorial 6 entry beneath
  Tutorials 1–5.

#### Tutorials — custom observables (Dispatch EE)

Fifth entry in the tutorials track. First tutorial focused on a
single architectural hook rather than a new physics scenario —
it generalises the `Observable`-record foothold introduced in
Tutorial 4 into the full construction surface.

- `docs/tutorials/05_custom_observables.md` (new) — four patterns
  over the shared MS-gate scenario from Tutorial 4: multi-subsystem
  Bell-fidelity projector `|Φ⁻⟩⟨Φ⁻|` (coherent-superposition
  target, not a computational-basis sum); two-ion `⟨σ_x σ_x⟩`
  correlator via `HilbertSpace.spin_op_for_ion` (the MS-fringe
  σ_x-basis signature); mode Fock-state projector `|1⟩⟨1|` via
  `mode_op_for` (complements the built-in `number` / `⟨n̂⟩`
  factory with direct `P(n)` access); non-Hermitian virtual
  `|↓↓⟩⟨↑↑|` coherence diagnostic (with a warning about complex
  expectations and measurement-protocol incompatibility). Verifies
  all four against their expected Bell-gate targets at `t_gate`.
  Closes with factory-vs-inline guidance (pointing at
  `observables.py` as the factory template) and the
  `StorageMode.EAGER` post-hoc-analysis route through the
  registered `concurrence_trajectory` evaluator.
- `docs/tutorials/index.md` — Tutorial 5 moved from *Planned* to
  *Available*; the planned-topics list renumbers accordingly.
- `mkdocs.yml` nav adds an explicit Tutorial 5 entry beneath
  Tutorials 1–4.

#### Tutorials — Mølmer–Sørensen Bell gate (Dispatch DD)

Fourth entry in the tutorials track. First two-ion scenario.
Parallels `tools/run_demo_ms_gate.py` and embeds the scenario's
committed artefact plot.

- `docs/tutorials/04_ms_gate_bell.md` (new) — detuned MS gate on
  two ²⁵Mg⁺ ions sharing the axial COM mode. ~15 min read,
  ~10 ms runtime. First tutorial to scale past a single ion;
  introduces the direct `IonSystem(...)` constructor (vs
  `.homogeneous`) as the template for heterogeneous chains, the
  `ms_gate_closing_detuning` / `ms_gate_closing_time` analytic
  helpers (Bell-closing δ and gate time derived from Ω, η, K —
  not magic numbers), and the `Observable` record as the hook for
  custom population projectors (`P(|↓↓⟩)`, `P(|↑↑⟩)`,
  `P_flip`). Walks through four final-state invariants at
  `t_gate`: loop closure `⟨n̂⟩ → 0`, equal Bell populations
  `= 1/2`, odd-parity `P_flip ≡ 0`, and ion-exchange-symmetric
  `⟨σ_z⁽⁰⁾⟩ = ⟨σ_z⁽¹⁾⟩` to machine precision. Closes with three
  "physics you can probe next" extensions (higher-K multi-loop
  gates, thermal-state start with leading-order fidelity loss,
  detuning miscalibration sweep via `solve_ensemble`). Inline
  plot served from the committed
  `benchmarks/data/ms_gate_bell_demo/plot.png`.
- `docs/tutorials/index.md` — Tutorial 4 moved from *Planned* to
  *Available*; the planned-topics list renumbers accordingly.
- `mkdocs.yml` nav adds an explicit Tutorial 4 entry beneath
  Tutorials 1–3.

#### Tutorials — Gaussian π-pulse (Dispatch CC)

Third entry in the tutorials track. Parallels
`tools/run_demo_gaussian_pulse.py` and embeds the scenario's
committed artefact plot.

- `docs/tutorials/03_gaussian_pi_pulse.md` (new) — Gaussian-shaped
  π-rotation through `modulated_carrier_hamiltonian`. ~10 min
  read, ~1 s runtime. First tutorial to exercise the
  time-dependent list-format Hamiltonian path through
  `sequences.solve`; introduces the pulse-area normalisation
  `A = π / (Ω·σ·√(2π))` that turns a chosen Gaussian width into
  a clean π-rotation; three-observable Bloch trajectory
  (`spin_x` / `spin_y` / `spin_z`) with an analytic overlay built
  from the cumulative-trapezoidal integral
  `θ(t) = ∫₀^t Ω·f(t') dt'`; sub-ppm agreement demonstrated.
  Closes with three envelope extensions (Blackman-window π-pulse
  with `scipy.integrate.quad` normalisation, stroboscopic
  square-wave drive, adiabatic raised-cosine ramp). Inline plot
  served from the committed
  `benchmarks/data/gaussian_pi_pulse_demo/plot.png`.
- `docs/tutorials/index.md` — Tutorial 3 moved from *Planned* to
  *Available*; the planned-topics list renumbers accordingly.
- `mkdocs.yml` nav adds an explicit Tutorial 3 entry beneath
  Tutorials 1 and 2.

#### Tutorials — red-sideband flopping (Dispatch BB)

Second entry in the tutorials track. Parallels
`tools/run_benchmark_sideband.py` (Phase 0.F tripwire) and
embeds the scenario's committed artefact plot.

- `docs/tutorials/02_red_sideband_fock1.md` (new) — red-sideband
  flopping from `|↓, n = 1⟩` with spin-and-motion readout. ~10 min
  read, ~1 s runtime. Covers: four-step pattern inherited from
  Tutorial 1 with the Hamiltonian builder swapped to
  `red_sideband_hamiltonian`; first use of the
  `analytic.lamb_dicke_parameter` helper to derive η from laser
  wavevector, mode eigenvector, ion mass, and mode frequency;
  first use of the `observables.number` phonon-number factory;
  single-phonon-manifold conservation law
  `⟨σ_z⟩ + 2⟨n̂⟩ = 1` as an inline sanity check. Closes with
  three "physics you can probe next" extensions (detuned sideband
  via `detuned_red_sideband_hamiltonian`; higher-`n` starting
  Fock; full Lamb–Dicke Wineland–Itano coupling via
  `full_lamb_dicke=True`). Inline plot served from the committed
  `benchmarks/data/01_single_ion_sideband_flopping/plot.png`.
- `docs/tutorials/index.md` — Tutorial 2 moved from *Planned* to
  *Available*; the planned-topics list renumbers accordingly.
- `mkdocs.yml` nav adds an explicit Tutorial 2 entry beneath
  Tutorial 1 under the Tutorials section.

#### Phase 2 — benchmarks documentation (Dispatch Z)

- `docs/benchmarks.md` — consolidates every Phase 2 dispatch's
  measured baselines into one narrative page. Covers the Phase 0.F
  smoke-test thresholds (three tripwires, all comfortably green),
  the sesolve-vs-mesolve wall-clock comparison (Dispatch X) with
  the "no advantage at library scale — QuTiP 5.2 closed the
  QuTiP-4-era gap" finding, and the `solve_ensemble` parallelism
  crossover (Dispatch Y) with the per-regime table. Includes a
  when-to-opt-in decision guide table so users sizing up the
  library can match their scenario to recommended settings
  directly. Closes the §5 Phase 2 `docs/benchmarks.md` item.
- `mkdocs.yml` nav gains a **Benchmarks** entry between Phase 1
  Architecture and Tutorials — discoverable from the docs-site
  top-level nav.
- README `Docs site scaffold` listing updated to advertise the new
  page.

## [0.2.0] — 2026-04-21

First tagged release. Combines the Phase 0 + Phase 1 core surface that
was scoped to ``v0.1-alpha`` with the Phase 1 measurement, systematics,
and registered-entanglement additions that were scoped to ``v0.2``,
per the WORKPLAN §5.0 release-mapping amendment (Dispatch E). The
``v0.1-alpha`` milestone is therefore subsumed rather than
separately-tagged — the release bundle below names every dispatch
that contributed.

**Convention Freeze.** ``CONVENTIONS.md`` bumps from ``0.1-draft``
to ``0.2``. §1–16 carry through unchanged from Phase 0; §17
(measurement layer, Dispatch P) and §18 (systematics layer,
Dispatch U) are newly frozen. Post-freeze additions to any section
require a new version bump with an explicit Convention Freeze gate
per the Endorsement Marker.

**Release bundle — dispatches on ``main`` at v0.2.0:**

- Phase 0 scaffold (pre-dispatch period): conventions, split-licence,
  asset provenance, regression harness (analytic + invariant
  permanent tiers + migration tier with 5 qc.py references),
  cache-integrity, performance smoke, docs-site scaffold with
  cd-rules tokens.
- Phase 1 core (pre-dispatch period): configuration layer
  (operators, species, drives, modes, system, hilbert, states),
  full four-family Hamiltonian surface (carrier / red-sideband /
  blue-sideband / MS × exact / detuned), analytic helpers,
  observable factories, sequences.solve dispatcher, hash-verified
  cache I/O.
- Dispatches A–F: migration-test activation, doc alignment,
  tutorials placeholder.
- Dispatch G: v0.3.2 WORKPLAN amendments (§4.0 repo hosting, §5.0
  WCAG read-through).
- Dispatches H–P: measurement layer — ``MeasurementResult``,
  ``BernoulliChannel`` (H), README refresh (I), ``BinomialChannel``
  (J), ``PoissonChannel`` + ``inputs`` keyword rename (K),
  ``DetectorConfig`` (L), ``SpinReadout`` protocol (M),
  ``ParityScan`` + ``parity`` observable (N), ``SidebandInference``
  (O), Wilson + Clopper–Pearson CIs + §17 freeze (P).
- Dispatch Q: registered entanglement observables — concurrence,
  EoF, log-negativity trajectory evaluators.
- Dispatches R–U: systematics layer — ``RabiJitter`` + §18 opener
  (R), ``DetuningJitter`` + ``PhaseJitter`` (S), drift primitives
  ``RabiDrift`` / ``DetuningDrift`` / ``PhaseDrift`` (T), SPAM
  primitives ``SpinPreparationError`` + ``ThermalPreparationError``
  + §18 freeze (U).
- Dispatch V: migration-tier debt paydown — scenario 2 activated
  via invariant comparison; scenario 3 and 4 skip reasons refined
  with today's empirical findings.

**Health at release.** Test suite: 785 pass / 2 skipped (scenarios
3 and 4 — migration-tier blockers with documented activation
paths). Every demo under ``tools/run_demo_*.py`` runs to
completion and emits the canonical ``benchmarks/data/<scenario>/``
artefact bundle. CI green: ruff check, ruff format, mypy-strict
on 27 src files, pytest (including regression-migration +
benchmarks tiers), pa11y WCAG 2 Level A hard-gated with triaged
ignores.

### Added

#### Phase 0 scaffold

- Initial repository workplan and conventions documents.
- Split-licence declarations at repository root and in `assets/`.
- Asset provenance record plus fetch/hash maintenance scripts.
- `pyproject.toml` with hatchling build metadata and tool configuration.
- Package scaffold at `src/iontrap_dynamics/`.
- MkDocs landing site scaffold with a styled welcome page and first
  navigation layer under `docs/`.

#### Phase 1 — configuration layer

- `operators.py` — canonical atomic-physics Pauli set (`spin_down`,
  `spin_up`, `sigma_plus_ion`, `sigma_minus_ion`, `sigma_z_ion`,
  `sigma_x_ion`, `sigma_y_ion`) built from first principles, no
  `qutip.sigmaz` touch (CONVENTIONS.md §3).
- `species.py` — `IonSpecies` and `Transition` dataclasses plus
  `mg25_plus`, `ca40_plus`, `ca43_plus` factories.
- `drives.py` — `DriveConfig` (`k_vector_m_inv`, Rabi frequency,
  detuning, phase, polarisation, transition label) with `.wavenumber_m_inv`.
- `modes.py` — `ModeConfig` with CONVENTIONS.md §11 normalisation
  enforced at construction; per-ion-eigenvector copy via
  `.eigenvector_at_ion`.
- `system.py` — `IonSystem` composition layer (species tuple, drives,
  modes, convention version) with cross-validation and
  `.homogeneous` classmethod.
- `hilbert.py` — `HilbertSpace` with spin-then-mode tensor ordering
  (§2): `spin_op_for_ion`, `mode_op_for`,
  `annihilation_for_mode`/`creation_for_mode`/`number_for_mode`,
  `identity`, and dimension accessors.
- `states.py` — `ground_state` (|↓⟩^⊗N ⊗ |0⟩^⊗M); `compose_density`
  for full-space density matrices from per-subsystem inputs;
  `coherent_mode` (§7), `squeezed_vacuum_mode` (§6),
  `squeezed_coherent_mode` (displace-after-squeeze).

#### Phase 1 — Hamiltonian builders

A symmetric surface across four families (time-independent Qobj vs.
time-dependent QuTiP list format):

- **Carrier**: `carrier_hamiltonian` (δ = 0, Qobj),
  `detuned_carrier_hamiltonian` (δ ≠ 0, list format).
- **Red sideband**: `red_sideband_hamiltonian`,
  `detuned_red_sideband_hamiltonian`.
- **Blue sideband**: `blue_sideband_hamiltonian`,
  `detuned_blue_sideband_hamiltonian`.
- **Mølmer–Sørensen gate**: `ms_gate_hamiltonian` (δ = 0, bichromatic),
  `detuned_ms_gate_hamiltonian` (δ ≠ 0, gate-closing Bell form).
- **Two-ion single-tone sideband** (shared mode):
  `two_ion_red_sideband_hamiltonian`,
  `two_ion_blue_sideband_hamiltonian`.
- **Modulated carrier**: `modulated_carrier_hamiltonian` —
  time-dependent envelope primitive for pulse shaping (Gaussian,
  Blackman, stroboscopic square-wave, …).
- **Full Lamb–Dicke option**: `full_lamb_dicke: bool = False`
  keyword on `red_sideband_hamiltonian`, `blue_sideband_hamiltonian`,
  `two_ion_red_sideband_hamiltonian`, and
  `two_ion_blue_sideband_hamiltonian` — replaces the leading-order
  η·a / η·a† with the all-orders Wineland–Itano operator
  `P_{Δn=±1}(e^{iη(a+a†)})` via matrix exponentiation, recovering
  Debye–Waller Rabi-rate reduction and Laguerre-polynomial
  amplitude oscillations in hot-ion regimes.

#### Phase 1 — analytic-regression helpers

All in `iontrap_dynamics.analytic`:

- `lamb_dicke_parameter` — full 3D projection η = (k·b)·√(ℏ/(2mω))
  (§10), sign preserved.
- `carrier_rabi_excited_population`, `carrier_rabi_sigma_z` — on-
  resonance carrier formulas.
- `generalized_rabi_frequency`, `detuned_rabi_sigma_z` — off-
  resonance generalised-Rabi formulas.
- `red_sideband_rabi_frequency`, `blue_sideband_rabi_frequency` —
  leading-order sideband rates with √n / √(n+1) scaling.
- `coherent_state_mean_n` — ⟨n⟩ = |α|².
- `ms_gate_phonon_number` — MS δ = 0 coherent-displacement phonon
  number for σ_x product eigenstates.
- `ms_gate_closing_detuning`, `ms_gate_closing_time` — detuned MS
  Bell-state closing condition (δ = 2|Ωη|√K, t_gate = π√K/|Ωη|).

#### Phase 1 — observables and solver dispatcher

- `observables.py` — `Observable` frozen dataclass plus `spin_x`,
  `spin_y`, `spin_z` (per-ion factories with automatic `sigma_{x,y,z}_{i}`
  labels) and `number(hilbert, mode_label)`. `expectations_over_time`
  helper returns a `{label → np.ndarray}` dict over a trajectory.
- `sequences.py` — `solve(...)` entry point: builds the full
  `TrajectoryResult` from `(hilbert, hamiltonian, initial_state, times,
  observables, …)`, accepting both Qobj and QuTiP list-format
  Hamiltonians. Propagates convention version, backend name/version,
  Fock-truncation record, request-hash, and provenance tags through
  `ResultMetadata`.

#### Phase 1 — diagnostic infrastructure (CONVENTIONS.md §13, §15)

- `FOCK_CONVERGENCE_TOLERANCE = 1e-4` exposed in `conventions.py`.
- Warning hierarchy in `exceptions.py`: `IonTrapWarning` base
  (`UserWarning`), `FockConvergenceWarning` (Level 1),
  `FockQualityWarning` (Level 2).
- `sequences.solve(..., fock_tolerance=None)` enforces the §13/§15
  ladder per mode on every call: OK silent, Level 1/2 emit both to
  the Python `warnings` channel **and** append a `ResultWarning` to
  `TrajectoryResult.warnings`, Level 3 raises `ConvergenceError`.
  Silent degradation via the public API is no longer possible.

#### Phase 0.F — performance smoke benchmarks (`tests/benchmarks/`)

All three benchmark slots now active against the canonical-hardware
thresholds in WORKPLAN_v0.3 §0.F:

- **Benchmark 1** — single-ion red-sideband flopping (N_Fock = 30,
  200 steps, threshold 5 s).
- **Benchmark 2** — two-ion Mølmer–Sørensen gate (N_Fock = 15,
  500 steps, threshold 30 s).
- **Benchmark 3** — stroboscopic AC-π/2 drive with a
  `modulated_carrier_hamiltonian` envelope (N_Fock = 40, 1000 steps,
  threshold 60 s).

#### Phase 0 — regression-tier scaffolding

- Permanent `regression_invariant` tests for trace, Hermiticity,
  positivity, norm conservation, and swap symmetry.
- Permanent `regression_analytic` tests — closed-form physics checks
  feeding from `iontrap_dynamics.analytic`.
- Migration-regression tier (`regression_migration`, Phase 0 only).
  Two layers to keep straight:
    - **Reference generator** (`tools/generate_migration_references.py`)
      — all 5 canonical scenarios implemented; reference bundles
      committed under `tests/regression/migration/references/`.
      Regeneration requires the `[legacy]` extras. Scenario 4 is a
      Path-A compat duplicate of qc.py's `single_spin_and_mode_ACpi2`
      with a single spline-callable fix for QuTiP 5; the other four
      dispatch straight into qc.py methods.
    - **Builder comparison** (`test_migration_references.py`) — the
      Phase 1 builder output compared against each committed bundle.
      Uses the `_qc_to_iontrap_convention` translator (atomic-physics
      σ_z/σ_y sign flip). Active for scenario 1 (carrier-on-thermal
      at `atol = 5e-3`) and scenario 5 (carrier-on-squeezed-coherent
      at `atol = 2e-2`). Scenarios 2/3/4 remain skipped with specific,
      probe-informed blockers — those are comparison-tier blockers,
      not generator-tier blockers.

#### Phase 1 — demo tools (`tools/`, `benchmarks/data/`)

All four tools go through `sequences.solve(...)` + emit artefacts in
a split canonical / narrative layout:

```
benchmarks/data/<scenario>/
├── manifest.json        — canonical via cache.save_trajectory (§0.B §3)
├── arrays.npz           — canonical: 'times' in SI seconds,
│                          observables under 'expectation__<label>'
├── demo_report.json     — narrative (purpose, analytic formulas,
│                          elapsed, environment, derived finals)
├── analytic_overlay.npz — optional: analytic-comparison arrays
│                          (present for carrier + Gaussian demos)
└── plot.png             — human-readable figure
```

The canonical cache is hash-verified and round-trips through
`cache.load_trajectory(path, expected_request_hash=…)`. Narrative
extras live alongside so demos can carry pedagogical context without
bloating the typed schema.

- `tools/run_benchmark_sideband.py` — Phase 0.F benchmark 1 capture.
- `tools/run_demo_carrier.py` — static carrier analytic-Rabi overlay.
- `tools/run_demo_gaussian_pulse.py` — end-to-end
  `sequences.solve` + `modulated_carrier_hamiltonian` with a
  Gaussian π-pulse envelope; 4-panel figure.
- `tools/run_demo_ms_gate.py` — end-to-end two-ion list-format MS
  Bell gate via `detuned_ms_gate_hamiltonian`; three-panel figure.

#### Phase 1 — result layer

- Hash-verified cache I/O for `TrajectoryResult` (`.npz` + JSON
  manifest) via `iontrap_dynamics.cache`.
- `ResultWarning`, `WarningSeverity`, `ResultMetadata`, `Result`,
  `TrajectoryResult`, `StorageMode` (`OMITTED`/`EAGER`/`LAZY`)
  re-exported from the package root.

#### Phase 0 — exception hierarchy

- `IonTrapError` base plus four stable subclasses: `ConventionError`,
  `BackendError`, `IntegrityError`, `ConvergenceError`
  (`src/iontrap_dynamics/exceptions.py`).

#### Phase 1 — measurement layer (`measurement/`, Dispatches H / J / K / L / M / N / O / P)

- `MeasurementResult` — frozen / slotted / kw-only `Result` sibling
  carrying the ideal / sampled dual-view mandated by
  `WORKPLAN_v0.3.md` §5. Enforces `shots >= 1` and
  `storage_mode = OMITTED` at construction; `ConventionError` on
  violation (blanket-catchable via `IonTrapError`).
- `measurement/channels.py` — `BernoulliChannel` (per-shot bits in
  `{0, 1}`, leading shot axis per CONVENTIONS.md §17.1) plus the
  `sample_outcome(channel, probabilities, shots, seed, upstream)`
  orchestrator. Bit-reproducible given `(seed, probabilities, shots)`;
  inherits upstream-trajectory metadata and emits a `trajectory_hash`
  provenance link when `upstream` is supplied.
- `tools/run_demo_bernoulli_readout.py` — first end-to-end exercise of
  the measurement boundary: carrier Rabi → `⟨σ_z⟩` → `p_↑` →
  `BernoulliChannel` → shot-noisy estimate, overlaid against the
  extreme-value band `σ · √(2 log N)`.
- `BinomialChannel` (Dispatch J) — aggregated sampling path. Returns
  `(n_inputs,)` int64 counts per input probability via
  `rng.binomial(shots, p)`; shot axis is absorbed. Distributionally
  equivalent to summing `BernoulliChannel` bits along axis 0, but not
  bit-identical under matched seed (§17.7). `sample_outcome` dispatches
  uniformly on either channel type.
- `tools/run_demo_binomial_readout.py` — Binomial companion to the
  Bernoulli demo, overlaying the aggregate estimate `counts / shots`
  against the ideal `p_↑` with a normal-approximation CI band.
- `PoissonChannel` (Dispatch K) — per-shot photon-counting path.
  Returns `(shots, n_inputs)` int64 counts via `rng.poisson(λ)` for
  non-negative input rates. New `ideal_label` `ClassVar` on each
  channel advertises what it consumes (`"probability"` for
  Bernoulli / Binomial, `"rate"` for Poisson); `sample_outcome`
  renames its keyword `probabilities → inputs` (breaking change —
  §17 is explicitly staged; existing demos and tests updated).
- `tools/run_demo_poisson_readout.py` — photon-counting readout of
  the carrier Rabi trajectory, using the canonical rate model
  `λ(t) = λ_dark + (λ_bright − λ_dark) · p_↑(t)` with
  `λ_bright = 10`, `λ_dark = 0.5`. Overlays the shot-averaged count
  against `λ(t)` with the `±1σ` mean-of-Poisson band `sqrt(λ/N)`.
- `DetectorConfig` (Dispatch L) — frozen dataclass for detector
  response (efficiency `η`, dark-count rate `γ_d`, threshold `N̂`).
  `apply(rate)` returns `η · rate + γ_d` (exact Poisson thinning
  plus additive background); `discriminate(counts)` thresholds
  per-shot counts into bright / dark bits;
  `classification_fidelity(lambda_bright, lambda_dark)` returns the
  analytic TP / TN / F values from `scipy.stats.poisson.cdf`.
  `sample_outcome` stays detector-agnostic — detectors compose
  explicitly around the channel call.
- `tools/run_demo_detected_readout.py` — first composition of
  detector + Poisson channel. Runs the full
  `apply → PoissonChannel → discriminate` pipeline on the carrier
  Rabi trajectory with `η = 0.4`, `γ_d = 0.3`, `N̂ = 4`; overlays
  the shot-averaged bright fraction against the Poisson-tail
  envelope `P(count ≥ N̂ | λ_det)` and the ideal `p_↑(t)`.
- `SpinReadout` protocol (Dispatch M) — first protocol-layer
  composer. Frozen dataclass bundling `ion_index`, `DetectorConfig`,
  `lambda_bright`, `lambda_dark`, and a `label` prefix. Its
  `.run(trajectory, *, shots, seed)` executes the *projective-shot*
  model — each shot projects to bright / dark with probability
  `p_↑`, then Poisson-samples at the state-conditional rate, then
  thresholds — and returns a `MeasurementResult` with the dual-view
  payload (`p_up`, `bright_fraction_envelope` ideal; per-shot
  `counts`, `bits`, and `bright_fraction` sampled). Looks up
  `sigma_z_{ion_index}` on the trajectory and raises
  `ConventionError` if missing.
- `tools/run_demo_spin_readout.py` — first protocol demo. Runs
  `SpinReadout.run` on the carrier Rabi trajectory with the same
  detector parameters as the Dispatch L demo and overlays both
  envelopes (projective-linear vs rate-averaged-nonlinear) to
  visualise how far the two sampling models diverge at finite
  fidelity.
- `ParityScan` protocol (Dispatch N) — first multi-ion protocol.
  Reconstructs the joint readout distribution `P(s_0, s_1)` from
  `⟨σ_z^i⟩`, `⟨σ_z^j⟩`, and the new two-body observable
  `⟨σ_z^i σ_z^j⟩`, then draws one categorical sample per shot so
  entangled-state correlations survive. Returns a
  `MeasurementResult` with `ideal_outcome` = `{p_up_i, p_up_j,
  parity, parity_envelope, joint_probabilities}` and
  `sampled_outcome` carrying per-ion counts / bits, per-shot parity,
  and shot-averaged parity estimate.
- `iontrap_dynamics.observables.parity(hilbert, ion_indices)` —
  multi-ion σ_z product observable factory. Default label
  `"parity_{i0}_{i1}_…"`. Required input for `ParityScan`.
- `tools/run_demo_parity_scan.py` — Bell-state-formation demo. Runs
  the gate-closing MS Hamiltonian on two ions and reads joint
  parity at every step, overlaying the ideal `⟨σ_z σ_z⟩`, the
  projective fidelity-shrunk envelope, and the shot-averaged
  parity estimate.
- `SidebandInference` protocol (Dispatch O) — motional thermometry
  via the short-time Leibfried–Wineland ratio. Takes paired RSB
  and BSB :class:`TrajectoryResult`s driven on the same initial
  motional distribution, runs each through the projective-shot
  pipeline on independent RNG streams (`np.random.SeedSequence.spawn(2)`),
  inverts the detector's linear envelope to get fidelity-corrected
  probabilities, then reports `n̄ = r / (1 − r)` element-wise.
  Result carries both the corrected `nbar_estimate` and the
  uncorrected `nbar_from_raw_ratio` so the fidelity-correction
  visibility is explicit. NaN propagates wherever `p_up_bsb = 0`
  or `r ≥ 1` — callers mask with `np.nanmean` / `np.nanmedian`.
- `tools/run_demo_sideband_inference.py` — ground-state thermometry
  demo. Runs RSB + BSB Hamiltonians on `|↓, 0⟩`, showing that RSB
  stays pinned (no phonon to remove) while BSB flops normally; the
  inferred `n̄` has median ~0 with a shot-noise spread of ~0.14 at
  500 shots and F ≈ 0.97.
- `measurement/statistics.py` (Dispatch P) — binomial confidence-
  interval surface. `wilson_interval(successes, trials, confidence)`
  returns the Wilson-score `(lower, upper)` bounds — closed-form,
  well-behaved at `p̂ ∈ {0, 1}`, near-nominal coverage for modest
  `n`. `clopper_pearson_interval` returns the exact Beta-quantile
  bounds — conservative, guaranteed actual coverage ≥ nominal.
  Both are fully vectorised over `(k, n)`. The
  `binomial_summary(k, n, confidence, method)` convenience
  dispatches on method and wraps the result in a frozen
  `BinomialSummary` dataclass with matched-shape `successes`,
  `trials`, `point_estimate`, `lower`, `upper` fields. No Wald
  interval — its coverage collapses near `p̂ ∈ {0, 1}` extremes
  routine in ion-trap readout.
- `tools/run_demo_wilson_ci.py` — capstone demo for the
  measurement track. Runs `SpinReadout` on a carrier-Rabi
  trajectory at an intentionally modest shot budget (80 shots) and
  overlays Wilson 95 % CI band on the bright-fraction estimator;
  reports single-seed empirical coverage (~96 % at nominal 95 %).

#### Phase 1 — registered entanglement observables (Dispatch Q)

Closes the §5 Phase 1 gap "logarithmic negativity / concurrence /
EoF as registered observables". These are nonlinear in ρ so they do
not fit the ``operator → qutip.expect`` shape consumed by
``expectations_over_time``; they are exposed as trajectory
evaluators that post-process an ``EAGER``-mode
:class:`TrajectoryResult.states` into a ``(n_times,)`` scalar
trajectory.

- `entanglement.concurrence_trajectory(states, *, hilbert, ion_indices)`
  — Wootters concurrence of a two-ion reduced state. Traces over
  motion + any other ions, evaluates the 4 × 4 Hill–Wootters
  formula via :func:`qutip.concurrence`. Returns ``float64`` in
  ``[0, 1]``, ``1`` on Bell states.
- `entanglement.entanglement_of_formation_trajectory(...)` — closed-
  form EoF from concurrence via
  ``E_F = h((1 + √(1 − C²)) / 2)``. Exact for the two-qubit case.
- `entanglement.log_negativity_trajectory(states, *, hilbert,
  partition)` — ``log₂ ‖ρ^{T_A}‖₁`` via explicit partial transpose
  (so multi-subsystem partitions work). ``partition="spins"`` or
  ``"modes"``; symmetric in the bipartition choice. Raises on
  mode-less Hilbert spaces.
- `tools/run_demo_bell_entanglement.py` — MS-gate entanglement
  trajectory demo. Runs the gate-closing Hamiltonian on `|↓↓, 0⟩`
  with ``storage_mode=EAGER``, overlays concurrence, EoF, and
  spin-motion log-negativity on one plot. Textbook signature:
  concurrence grows monotonically to 1 at ``t_gate``; log-negativity
  loops (peaks at ~1.31) and returns to ~0 as motion disentangles.

#### Phase 1 — systematics layer (`systematics/`, Dispatches R / S / T / U)

Opens the §5 Phase 1 systematics surface. Jitter / drift / SPAM noise
classes (§18.1) are distinguished; Dispatch R ships the first jitter
primitive.

- `systematics.RabiJitter` — frozen dataclass, `sigma` parameter,
  zero-allowed for no-op tests. `.sample_multipliers(shots, rng)`
  returns `(shots,)` float64 array of `(1 + ε)` factors with
  `ε ~ Normal(0, σ)`.
- `systematics.perturb_carrier_rabi(drive, jitter, shots, seed)` —
  composition helper. Returns a `tuple[DriveConfig, ...]` of length
  `shots`, each with `carrier_rabi_frequency_rad_s = Ω₀ · (1 + ε_i)`.
  Non-Rabi `DriveConfig` fields pass through untouched.
  Bit-reproducible given `(drive, jitter, shots, seed)`.
- `CONVENTIONS.md §18` opened (staged, v0.2 Freeze target). §18.1
  noise taxonomy; §18.2 jitter composition pattern (per-shot
  ensemble of solves, NumPy aggregation); §18.3 RabiJitter
  semantics; §18.4 lists pending rules for Dispatches S–U.
- `tests/unit/test_systematics.py`: 19 cases across construction
  guards, multiplier-distribution contract (mean → 1, std → σ at
  50k shots), seed reproducibility, frozen-field preservation,
  zero-sigma no-op, zero-shots rejection.
- `tools/run_demo_rabi_jitter.py` — inhomogeneous-dephasing demo.
  Runs 200 carrier-Rabi trajectories with `σ_Ω = 3 %` and overlays
  the ideal `⟨σ_z⟩(t)`, ensemble mean (visibly damped), ±1σ
  shot-to-shot spread band, and the analytic
  `T₂* ≈ 1/(σ·Ω₀) ≈ 5.3 μs` marker.
- `systematics.DetuningJitter(sigma_rad_s)` and
  `systematics.PhaseJitter(sigma_rad)` (Dispatch S) — additive
  Gaussian jitter primitives on `DriveConfig.detuning_rad_s` and
  `DriveConfig.phase_rad` respectively. Each exposes a
  `.sample_offsets(shots, rng)` method returning the per-shot
  additive offsets. Parallel to `RabiJitter` but additive rather
  than multiplicative.
- `systematics.perturb_detuning(drive, jitter, shots, seed)` and
  `systematics.perturb_phase(drive, jitter, shots, seed)` (Dispatch
  S) — composition helpers matching `perturb_carrier_rabi`. Both
  bit-reproducible given `(drive, jitter, shots, seed)`.
- `CONVENTIONS.md §18.3` generalised to cover all three jitter
  primitives (Rabi multiplicative, Detuning / Phase additive),
  shared `σ ≥ 0` + zero-is-no-op + seed-reproducibility rules.
  §18.4 repoints to Dispatches T–U.
- `tools/run_demo_detuning_jitter.py` — off-resonance Rabi-mixing
  demo. Runs 200 carrier-Rabi trajectories with
  `σ_δ / 2π = 300 kHz` (`σ_δ / Ω₀ = 0.3`) and shows the combined
  dephasing + amplitude-reduction signature of off-resonance drive
  (each shot's effective Rabi rate depends on δ, and its amplitude
  is reduced by `Ω² / (Ω² + δ²)`). Max |ensemble mean − ideal| ≈
  0.72; terminal-offset-from-`−1` ≈ 0.46 at `t ≈ 5 T_Ω`.
- `systematics.RabiDrift(delta)`, `systematics.DetuningDrift(
  delta_rad_s)`, `systematics.PhaseDrift(delta_rad)` (Dispatch T)
  — deterministic single-value parameter offsets. Unlike jitter
  primitives, deltas are **unsigned** (either direction is
  physical: tuned-low or tuned-high). Each has a matching
  `systematics.apply_*_drift(drive, drift)` helper returning a
  single perturbed `DriveConfig` (no ensemble). Two `apply_*` calls
  with the same inputs are bit-identical.
- `CONVENTIONS.md §18.4` (new) — drift primitives rules (unsigned
  deltas, single-solve composition pattern, scan-by-comprehension
  idiom, `DriveConfig`-invariant interaction). §18.5 repoints to
  Dispatch U (SPAM).
- `tools/run_demo_rabi_drift_scan.py` — π-pulse miscalibration
  scan. Fixes pulse duration at nominal `t_π = π / Ω₀`, sweeps
  `RabiDrift.delta ∈ [−20 %, +20 %]`, plots final `⟨σ_z⟩` against
  the analytic `−cos((1+δ)π)` curve. Solver-vs-analytic max error
  is 9e-7. Reports `|δ| ≤ 6 %` as the 99 %-fidelity calibration
  tolerance window.
- `systematics.SpinPreparationError(p_up_prep)` and
  `systematics.ThermalPreparationError(n_bar_prep)` (Dispatch U)
  — state-preparation (SPAM) error primitives. Each has a matching
  `imperfect_spin_ground` / `imperfect_motional_ground` helper
  that returns a per-subsystem density matrix suitable for
  :func:`iontrap_dynamics.states.compose_density`. Deterministic
  (classical mixture → no RNG); `imperfect_motional_ground` guards
  against `n̄ ≥ fock_dim − 1` with a `ValueError`.
- `CONVENTIONS.md §18.5` (new): state-preparation rules.
  Density-matrix composition pattern, Fock-truncation guard,
  no-RNG determinism note. **§18 freeze** — the systematics-layer
  section is now a complete read-through for the v0.2 Convention
  Freeze gate, alongside §17 (measurement) sealed at Dispatch P.
- `tools/run_demo_spam_prep.py` — closing demo for the systematics
  track. Runs carrier Rabi on three initial states (ideal ket,
  `p_↑ = 3 %` spin-prep only, full SPAM with `n̄ = 0.1`) and
  verifies the analytic `(1 − 2p)` amplitude-reduction signature
  to solver-tolerance precision (3e-5 error).

#### Migration-tier debt paydown (Dispatch V)

Activates one of the three skipped migration-comparison tests that
have been carrying activation notes since the Phase 0 regression
harness. The other two are refined with today's empirical probe
data so the skip reasons name the exact physics gap rather than a
generic placeholder.

- **Scenario 2** (single-ion RSB from Fock |1⟩) **ACTIVATED**.
  Element-wise trajectory comparison is replaced with *invariant
  comparison* — the alternative path named in the original skip.
  Verifies σ_z swing amplitude (`|(max − min)/2| ≈ 1`) matches qc.py
  to within 0.03, and ⟨n̂⟩ minimum matches to within 0.02. Both
  invariants are phase/rate-independent and tolerate the 1.5×
  flop-rate gap between qc.py's lab-frame and our RWA-reduced
  dynamics. Reference module docstring updated.
- **Scenario 3** (two-ion single-tone blue sideband) — skip reason
  tightened with an empirical probe showing the discrepancy is
  ~2× on *amplitude* (σ_z peak 0.79 ours vs 0.27 qc.py; n_mode peak
  1.79 vs 1.13) rather than the original "1.5× on rate" hypothesis.
  The invariant-comparison fix (scenario 2's path) is therefore
  ruled out. New suggested activation paths: re-derive qc.py's
  two-spin coupling pre-factor for our `two_ion_blue_sideband_
  hamiltonian`, or retire in favour of an internal analytic
  reference.
- **Scenario 4** (stroboscopic AC-π/2) — skip reason clarified to
  note that this is a *missing physics feature* (full-exponential
  Lamb–Dicke operator) not a frame reconciliation, so the scenario-2
  invariant-comparison path does not apply. Activation path now
  suggests an `ld_order` parameter on `modulated_carrier_hamiltonian`
  or a dedicated frequency-modulated-carrier builder.

785 tests pass / 2 skipped (was 3).

### Changed

- CONVENTIONS.md §17 *(staged — v0.2 Convention Freeze target)*
  opened. Covers shot semantics, the ideal / sampled dual-view, RNG
  reproducibility, the `OMITTED` storage-mode tombstone for
  `MeasurementResult`, provenance chaining, and channel input
  semantics. Dispatch H shipped §17.1–17.6; Dispatch J added §17.7
  (per-shot vs aggregated output shape; distributional- not bit-
  equivalence across channel types); Dispatch K generalised §17.6
  to cover `probability` and `rate` input types via the
  `channel.ideal_label` ClassVar; Dispatch L added §17.8 (detector
  response: `η / γ_d / N̂` parameters, explicit rate-transform +
  threshold composition, exact Poisson thinning + additive
  background); Dispatch M added §17.9 (projective-shot readout
  model, linear fidelity envelope `TP·p_↑ + (1−TN)·(1−p_↑)`,
  per-protocol result layout); Dispatch N added §17.10 (multi-ion
  joint readout: joint-probability reconstruction from three ZZ-
  tomography components, why independent-Bernoulli sampling fails
  for entangled states, parity envelope
  `(TP + TN − 1)² · ⟨σ_z σ_z⟩ + (TP − TN)²` at zero marginals);
  Dispatch O added §17.11 (sideband inference: short-time ratio
  formula, fidelity correction before ratio, independent RNG
  streams via `SeedSequence.spawn`, NaN propagation); Dispatch P
  added §17.12 (binomial confidence intervals: Wilson + Clopper–
  Pearson formulas, non-nesting caveat, boundary snap, z / coverage
  convention). **§17 freeze** — the measurement-layer section is
  now a complete read-through for the v0.2 Convention Freeze gate.
- WORKPLAN v0.3.2: two amendments under Coastline authority.
  - §4.0 declares the interim `uwarring82/iontrap-dynamics` hosting
    and reconciles the §4 "Repository topology" clause and the
    Phase 0 exit criterion with the live `[project.urls]` and
    `mkdocs.yml` state.
  - §5.0 gains a "Consequence for WCAG clauses in §5 and §6"
    paragraph reconciling three stale "AA as gate" clauses with
    the actual "Level A gated, AA advisory" CI policy landed in
    `f370fe7`.

### Status at this mark

- **Test suite**: 497 passing, 3 skipped. The skips are all in the
  migration-tier **builder-comparison** layer (scenarios 2/3/4),
  which requires either reverse-engineering qc.py's frame/state
  choices or a full-exponential lab-frame builder. The
  reference-generator layer implements all 5 scenarios; the skips
  reflect unmatched comparisons, not missing references.
- **Gates**: ruff lint + format, mypy strict on 17 source files,
  pa11y Level A (report-only), CI green.
- **Phase 0.F benchmarks**: all three active and within thresholds.
