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
- Migration-regression tier (`regression_migration`, Phase 0 only):
  reference bundles under `tests/regression/migration/references/`,
  bundle-validation tests, `_qc_to_iontrap_convention` translator
  (atomic-physics σ_z/σ_y sign flip), and scenario-1 + scenario-5
  activated comparisons at physical-level tolerance (`atol = 5e-3`
  and `2e-2` respectively). Scenarios 2/3/4 remain skipped with
  specific, probe-informed blockers.

#### Phase 1 — demo tools (`tools/`, `benchmarks/data/`)

- `tools/run_benchmark_sideband.py` — Phase 0.F benchmark 1 capture
  tool with `arrays.npz`/`metadata.json`/`plot.png` output.
- `tools/run_demo_carrier.py` — static carrier analytic Rabi overlay.
- `tools/run_demo_gaussian_pulse.py` — end-to-end
  `sequences.solve` + `modulated_carrier_hamiltonian` with a
  Gaussian π-pulse envelope; 4-panel figure showing envelope +
  Bloch components.
- `tools/run_demo_ms_gate.py` — end-to-end two-ion list-format MS
  Bell gate via `detuned_ms_gate_hamiltonian`; three-panel figure
  showing phase-space loop closure, Bell populations converging to
  0.5/0.5, and ion-exchange symmetry.

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

### Status at this mark

- **Test suite**: 497 passing, 3 skipped (the three specific
  migration-tier scenarios 2/3/4 that require either reverse-
  engineering qc.py's frame/state choices or a full-exponential
  lab-frame builder).
- **Gates**: ruff lint + format, mypy strict on 17 source files,
  pa11y Level A (report-only), CI green.
- **Phase 0.F benchmarks**: all three active and within thresholds.
