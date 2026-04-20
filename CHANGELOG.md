# Changelog

All notable changes to `iontrap-dynamics` will be documented in this file.

The format follows Keep a Changelog, and the project aims to follow Semantic
Versioning once the public package surface reaches its first alpha release.

## [Unreleased]

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

#### Phase 1 — measurement layer (`measurement/`, Dispatches H / J / K / L / M / N)

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
  `(TP + TN − 1)² · ⟨σ_z σ_z⟩ + (TP − TN)²` at zero marginals).
  §17.11 lists the rules still pending for Dispatches O–P.
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
