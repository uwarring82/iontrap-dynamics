# CONVENTIONS

**Physical, numerical, and notational conventions for `iontrap-dynamics`**

Version 0.2 · Drafted 2026-04-17 · Frozen 2026-04-21 · Status: v0.2 Convention Freeze

**Classification:** Coastline (hard constraints per T(h)reehouse +EC CD 0.9).
**Licence:** CC BY-SA 4.0.
**Scope:** Conventions covering §1–18. Phase 0 (v0.1-alpha) shipped §1–16; the v0.2 Convention Freeze adds §17 (measurement layer — closed at Dispatch P) and §18 (systematics layer — closed at Dispatch U). Post-freeze additions require a CONVENTIONS.md version bump per the Endorsement Marker below.
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

This document is authoritative. Every `IonSystem` records the `CONVENTIONS.md` version it was built against; every `TrajectoryResult` carries that version in its metadata. When code and this document disagree, this document wins and the code is the bug.

---

## 1. Units

Interface units (what users see) are chosen for physical legibility; internal units (what solvers see) are SI.

| Quantity | Interface | Internal | Notes |
|---|---|---|---|
| Angular frequency | 2π · MHz | rad · s⁻¹ | User writes `omega = 2 * pi * 1.25` for 1.25 MHz; library stores 7.854 × 10⁶ rad/s |
| Time | μs | s | User writes `t = 50` for 50 μs; library stores 5 × 10⁻⁵ s |
| Mass | kg (SI) | kg (SI) | No conversion. `IonSpecies.mass` is SI throughout |
| Length | m (SI) | m (SI) | Wavevectors in m⁻¹, positions in m |
| Energy | — | J or ℏ · rad · s⁻¹ | Internal representation is always rad·s⁻¹ (i.e. H/ℏ) |

**Rule.** Users interact with 2π·MHz and μs. The library converts at the configuration-object boundary (`DriveConfig`, `ModeConfig`) and never re-converts downstream. Internal solver code operates in SI only.

**Planck's constant.** Internal Hamiltonians are represented as H/ℏ with units rad·s⁻¹. Factors of ℏ appear only when an expression crosses from frequency-space to energy-space (e.g. the Lamb–Dicke parameter; see §10).

---

## 2. Tensor ordering

For a composite system of N_s spins and N_m modes, the Hilbert space is:

```
H = H_spin_1 ⊗ H_spin_2 ⊗ ... ⊗ H_spin_{N_s} ⊗ H_mode_1 ⊗ H_mode_2 ⊗ ... ⊗ H_mode_{N_m}
```

**Rule.** Spins first, then modes. Within each group, ascending index left-to-right. This ordering is fixed; `IonSystem` constructs `HilbertSpace` in this order and every operator builder respects it.

**Test.** `tests/conventions/test_tensor_ordering.py` instantiates a 2-spin, 1-mode system and asserts that the reduced Hilbert dimensions multiply as `(2, 2, N_Fock)` in that order and that partial traces over mode indices recover the 2-spin reduced state.

---

## 3. Spin basis and Pauli convention

### Basis labelling

- `|↓⟩ ≡ basis(2, 0)` — ground state (lower energy)
- `|↑⟩ ≡ basis(2, 1)` — excited state (higher energy)

This is the Wineland/ion-trap convention: the energy eigenvalue of σ_z carries the physics sign, so that ground maps to −1 and excited to +1. Every textbook and thesis in the trapped-ion literature — Wineland, Leibfried, Schmidt-Kaler, the AG Schätz theses — uses this.

### Ladder action

```
σ_+ |↓⟩ = |↑⟩        σ_+ |↑⟩ = 0       (raises along the atomic energy ladder)
σ_− |↑⟩ = |↓⟩        σ_− |↓⟩ = 0       (lowers along the atomic energy ladder)
```

### Pauli operators in matrix form

In the ordered basis `(|↓⟩, |↑⟩)`:

```
σ_x = [[ 0, 1],     σ_y = [[ 0, −i],     σ_z = [[−1,  0],
       [ 1, 0]]            [ i,  0]]            [ 0, +1]]

σ_+ = [[ 0, 0],     σ_− = [[ 0, 1],
       [ 1, 0]]            [ 0, 0]]
```

Definitions:

```
σ_z ≡ |↑⟩⟨↑| − |↓⟩⟨↓|
σ_+ ≡ |↑⟩⟨↓|
σ_− ≡ |↓⟩⟨↑|
σ_x = σ_+ + σ_−
σ_y = −i(σ_+ − σ_−)
```

Eigenvalue statements:

```
σ_z |↓⟩ = −|↓⟩        σ_z |↑⟩ = +|↑⟩
```

### Why the library does not use `qutip.sigmaz`

QuTiP's `sigmaz()` returns `diag(+1, −1)` relative to `basis(2, 0)`. This is the quantum-information convention, where `basis(2, 0) = |0⟩` is the computational-basis *logical zero* with σ_z eigenvalue +1. In that convention, binding `|↓⟩ ≡ basis(2, 0)` would yield `σ_z |↓⟩ = +|↓⟩`, which contradicts every ion-trap paper a Clock-School student will read.

The library resolves this at the *operator* level, not the basis level: `|↓⟩ ≡ basis(2, 0)` is kept (it matches QuTiP's state-construction idioms), and the library exposes its own `sigma_z_ion` whose sign is flipped relative to `qutip.sigmaz()`. All Pauli operators enter solver code through the library's canonical operator module; `from qutip import sigmaz` (or any import aliasing it) is banned and flagged by convention-enforcement tests (see §0.D of the workplan).

This is Design Principle 2 ("no hidden laboratory assumptions") applied at the operator level: the sign of σ_z is a laboratory-visible convention and must not leak through a third-party default.

**Enforcement test (Phase 0.D):** `assert sigma_z_ion * ket_down == −ket_down` and `assert sigma_z_ion * ket_up == +ket_up`.

---

## 4. Detuning sign

```
δ ≡ ω_laser − ω_atom
```

- **Positive δ** → laser is blue-detuned (above resonance).
- **Negative δ** → laser is red-detuned (below resonance).

Every sideband Hamiltonian builder takes `detuning` in this sign convention. A red-sideband drive at the first motional sideband of mode m is `δ = −ω_m`; the corresponding blue sideband is `δ = +ω_m`.

---

## 5. Hamiltonian form and interaction picture

All builders return Hamiltonians in the **interaction picture of the atomic transition**: the free atomic term Σ_i (ω_atom / 2) σ_z^{(i)} is removed, and drives are written in the rotating frame at the atomic frequency.

**Rotating-wave approximation.** The RWA is applied by default; fast-rotating counter-rotating terms are dropped. Builders that support exact (non-RWA) evolution carry an explicit `rwa=False` flag and document the additional structure.

**Reference carrier Hamiltonian.** For a single ion driven on resonance (δ = 0) with Rabi frequency Ω and laser phase φ:

```
H_carrier = (ℏ Ω / 2) [σ_+ e^{iφ} + σ_− e^{−iφ}]
```

Off resonance, the same builder emits:

```
H_carrier(δ) = (ℏ Ω / 2) [σ_+ e^{iφ} e^{iδt} + σ_− e^{−iφ} e^{−iδt}]
```

Internally H/ℏ is stored in rad·s⁻¹ (see §1).

---

## 6. Squeezing parameter

```
z = r · exp(2iφ)     S(z) = exp[(z* a² − z a†²) / 2]
```

Matches QuTiP's `squeeze(N, z)` convention. The factor of 2 in the phase is deliberate: it makes z the natural variable for the squeezing *ellipse* (which has period π in φ), not the squeezing *operator* (which would double-count). Documented here because the factor varies across textbooks.

---

## 7. Displacement parameter

```
α = |α| · exp(iφ)     D(α) = exp(α a† − α* a)
```

Standard convention. `|α|²` is the mean phonon number of the resulting coherent state when displacing the vacuum.

---

## 8. Spin rotation Euler convention

Spin rotations composed of multiple axes are expressed in **extrinsic XYZ** order with **active** rotations:

```
R(α, β, γ) = R_z(γ) · R_y(β) · R_x(α)
```

- **Extrinsic** — each rotation is about the fixed laboratory axis, not the rotated-body axis.
- **Active** — rotations transform states (kets rotate), not coordinate systems.
- **Single-axis rotation** — `R_n̂(θ) = exp(−i (θ/2) n̂ · σ)` for unit vector n̂.

`tests/conventions/test_euler.py` verifies this by rotating a spin-coherent state through a known sequence and comparing Bloch-vector components against the analytic prediction.

---

## 9. Bell state convention

```
|Φ+⟩ = (|↓↓⟩ + |↑↑⟩) / √2
|Φ−⟩ = (|↓↓⟩ − |↑↑⟩) / √2
|Ψ+⟩ = (|↓↑⟩ + |↑↓⟩) / √2
|Ψ−⟩ = (|↓↑⟩ − |↑↓⟩) / √2
```

**Legacy divergence flagged.** The legacy `qc.py` uses `(|dd⟩ + i|uu⟩) / √2` as its MS-gate target state. This is a non-standard convention (differs from |Φ+⟩ by a relative phase of i on the |↑↑⟩ component) and is **not** adopted in `iontrap-dynamics`. Migration regressions that compare MS-gate outputs against `qc.py` reference arrays must apply the phase rotation explicitly and document it in the test metadata.

Fidelity against a named Bell state is computed against the definitions above, not against any phase-shifted variant.

---

## 10. Lamb–Dicke parameter

The Lamb–Dicke parameter of ion i with respect to mode m under drive with wavevector **k** is the full 3D dot product:

```
η_{i,m} = (k⃗ · b⃗_{i,m}) · √(ℏ / (2 · m_i · ω_m))
```

where:

- **k** — laser wavevector, 3-vector, units m⁻¹
- **b**_{i,m} — normal-mode eigenvector of mode m at ion i, 3-vector, dimensionless
- m_i — mass of ion i, kg
- ω_m — angular frequency of mode m, rad · s⁻¹
- ℏ — reduced Planck constant, J·s

**No 1D shortcut.** Even for axial drives on a linear mode, the library computes the dot product as a 3-vector operation. Convenience forms that accept scalars `k` and `b` are forbidden at the public API; internal builders always receive 3-vectors.

**Sign.** η can be negative. The sign is physical (it encodes the relative phase of the drive with respect to the mode displacement) and is preserved throughout the builder chain. Operators are built using complex exponentials that respect η's sign; taking |η| is reserved for derived quantities (Lamb–Dicke regime checks, Rabi-frequency rescaling of the carrier amplitude) and is done explicitly.

**Test.** `tests/conventions/test_lamb_dicke.py` checks:
1. **k** ∥ **b** case against the analytic 1D value.
2. **k** ⊥ **b** case returns exactly zero.
3. Oblique case matches the analytic projection for a specified angle.

---

## 11. Normal-mode eigenvector normalisation

For each mode m, the eigenvectors at all ions satisfy:

```
Σ_i |b⃗_{i,m}|² = 1
```

where |·|² is the squared Euclidean norm of the 3-vector and the sum runs over all ions in the crystal.

**Convention.** Mode eigenvectors are supplied by the user or by an external solver (e.g. `pylion`, `trical`) in this normalisation. `ModeConfig.__post_init__` verifies normalisation within 10⁻¹⁰ and raises if violated.

**Orthogonality.** Distinct modes are orthogonal under the same inner product: Σ_i b⃗_{i,m} · b⃗_{i,m'} = δ_{m,m'}. Checked at `IonSystem` construction for the set of supplied modes; violation raises a typed exception.

---

## 12. Trap frame

For linear Paul traps:

- Right-handed Cartesian coordinates.
- **z-axis** — trap symmetry axis (axial direction, typically the weakest confinement).
- **x, y** — radial directions.

For non-linear geometries (zigzag, 2D crystals, surface traps), the trap frame must be declared explicitly at `IonSystem` construction via `trap_frame=...`, which specifies the axis convention. No implicit default.

---

## 13. Fock truncation convergence

The motional Hilbert space of each mode is truncated at N_Fock. Convergence is monitored by the steady-state or peak population in the topmost Fock level, `p_top = max_t ⟨N_Fock − 1 | ρ_m(t) | N_Fock − 1⟩`.

**Default tolerance.** ε = 10⁻⁴.

**Status ladder** (wired into the warnings ladder of §15):

| Regime | p_top range | Status |
|---|---|---|
| Converged | p_top < ε / 10 | OK, silent |
| Slow convergence | ε / 10 ≤ p_top < ε | Convergence warning (§15 Level 1) |
| Degraded quality | ε ≤ p_top < 10 · ε | Degradation warning (§15 Level 2) |
| Truncation failure | p_top ≥ 10 · ε | Hard failure (§15 Level 3), typed exception |

Users may override ε per call; the default is set in `conventions.py` and recorded in `TrajectoryResult.metadata.conventions_version`.

---

## 14. Reproducibility layers

Per Design Principle 3, reproducibility is stratified:

| Layer | Condition | Expected equivalence |
|---|---|---|
| **Bit-exact** | Same platform, pinned dependency lockfile, same seed | Array equality (or ≤ 10⁻¹⁶ element-wise) against committed reference `.npz` |
| **Numerical** | Cross-platform or dependency drift within semver-compatible range | Element-wise tolerance 10⁻¹⁰ (absolute or relative, whichever is tighter) |
| **Metadata-exact** | Any environment | Parameter hashes, convention version, request hash — identical always |

**Rule.** CI runs the bit-exact tier on the reference platform (pinned macOS-ARM / Python 3.11 / QuTiP 5.0.x lockfile). The numerical tier is the public contract for users on other platforms. Metadata-exact is unconditional: a run that produces numerically-correct arrays but mismatched hashes is a bug.

---

## 15. Warnings and failure policy (three-level ladder)

Silent degradation is forbidden. Every numerical anomaly takes one of three forms:

### Level 1 — Convergence warning

- **Trigger:** solver converged but slowly. Includes: Fock top-population between ε/10 and ε (§13); adaptive-step solver reduced step size repeatedly; expectation-value time-series oscillates below but near tolerance.
- **Channel:** emitted to the Python `warnings` channel via a `ConvergenceWarning` subclass; also appended to `TrajectoryResult.warnings` as a structured record with category, message, and diagnostic snapshot.
- **Behaviour:** results are returned, trusted for coarse analysis, flagged for refinement.

### Level 2 — Numerical-quality degradation warning

- **Trigger:** non-convergence below the full tolerance, but above the hard-failure threshold. Includes: Fock top-population between ε and 10·ε; invariant violations (trace, norm) between 10⁻¹⁰ and 10⁻⁶; partial loss of physical property within recoverable range.
- **Channel:** Python `warnings` via a `QualityWarning` subclass; `TrajectoryResult.warnings` record with severity = `"degraded"`.
- **Behaviour:** results returned but consumers must consult `result.warnings` before publication-grade use. Analysis code that ignores this field and produces figures is a convention violation.

### Level 3 — Hard failure

- **Trigger:** parameter-hash mismatch on cache load; unsupported backend feature (e.g. stochastic solver requested on deterministic backend); physics-invariant violation beyond tolerance (trace deviation > 10⁻⁶, negative eigenvalue below −10⁻⁸, Hermiticity deviation > 10⁻¹⁰); Fock truncation failure (§13).
- **Channel:** typed exception, raised. Never a warning. Never silently continued.
- **Exception hierarchy.** `IonTrapError` is the base and inherits directly from `Exception` (not `RuntimeError` — many subclasses are validation or integrity errors, not runtime-logic errors). Four subclasses:

  - `ConventionError` — violations of this document. Raised e.g. when mode-eigenvector normalisation (§11) fails, when the trap frame is ambiguous (§12), when detuning units are outside the declared ranges (§1), or when a banned import (`qutip.sigmaz`, §3) is used in library code.
  - `BackendError` — backend-internal failures and unsupported-feature requests (e.g. stochastic solver requested on a deterministic-only backend, unavailable QuTiP option).
  - `IntegrityError` — parameter-hash mismatch on cache load, cache corruption, and physics-invariant violation beyond tolerance (trace deviation > 10⁻⁶, negative eigenvalue below −10⁻⁸, Hermiticity deviation > 10⁻¹⁰).
  - `ConvergenceError` — solver failures above tolerance, including Fock-truncation failure (§13) and non-terminating adaptive-step reductions.

  Downstream code may catch `IonTrapError` as a blanket or catch specific subclasses. The four-subclass grain is deliberate: fewer would force `except` clauses to over-catch; more would fracture natural families (e.g. separating "trace violated" from "positivity violated" when both are integrity failures). The set is locked at v0.1; additions in later versions must be justified as a new family, not a split of an existing one.

  Canonical location: `src/iontrap_dynamics/exceptions.py`.

**Rule.** Every solver exit path classifies its outcome into exactly one of {OK, Level 1, Level 2, Level 3}. The `TrajectoryResult.warnings` field is a list of structured records (possibly empty); the act of producing a result without classification is itself a bug.

---

## 16. Archival — tracked vs ephemeral artefacts

Design Principle 15 ("Deprecation, not deletion", per CD 0.8) applies to **tracked project artefacts**: authoritative documents, design assets, convention revisions, reference arrays that were once the right answer to a question a future reader might re-ask. These move to `archive/` with a dated note; they are not deleted.

The rule does **not** apply to ephemeral artefacts: generated caches (`__pycache__/`, `.pytest_cache/`, `.mypy_cache/`), build outputs (`dist/`, `build/`, `*.egg-info/`), notebook execution state, solver-level temporary files, or anything listed in `.gitignore`. These are expected to vanish between builds and carry no archival obligation.

**Decision rule for contributors.** Before deleting a tracked file, ask: *was this ever the authoritative answer to a question someone might later re-ask?* If yes → `archive/<date>-<reason>/` with a `README.md` stub. If no → delete freely.

---

## 17. Measurement layer *(staged — v0.2 Convention Freeze target)*

**Status:** opened at Dispatch H (`src/iontrap_dynamics/measurement/`). Rules below are staged, not frozen: additions across Dispatches I–O may tighten them, and the full section seals at the v0.2 release under a Convention Freeze gate per `WORKPLAN_v0.3.md` §5 Phase 1. Treat any call-site depending on details of §17 as provisional until the freeze.

### 17.1 Shot

A **shot** is one independent application of a measurement channel, producing one outcome sample at one setting. Shot count is a keyword-only argument `shots: int` (≥ 1) on every channel-facing API. The shot axis is always the **leading** axis of per-shot outputs — `(shots, n_settings)` or `(shots, n_times)`. Aggregated outputs (counts, estimators) place the setting / time axis first.

### 17.2 Ideal vs sampled outcomes (result dual-view)

Every measurement result carries two views, mandated by `WORKPLAN_v0.3.md` §5:

- `ideal_outcome: Mapping[str, NDArray]` — the noise-free input the channel was applied to (e.g. probability, expectation value, intensity).
- `sampled_outcome: Mapping[str, NDArray]` — the stochastic output (e.g. per-shot bits, photon counts).

Analytic-regression checks compare to the ideal view; statistics / estimators consume the sampled view. Library code MUST NOT silently overwrite one with the other.

### 17.3 RNG and reproducibility

The reference RNG is `numpy.random.default_rng`. Channel-facing APIs accept either a `seed: int | None` (convenience) or a caller-constructed `rng: np.random.Generator` (full control). When `seed` is supplied, the resulting `MeasurementResult.rng_seed` records it and the result is bit-reproducible given `(seed, probabilities, shots)`. When a pre-seeded generator is supplied, `rng_seed` is `None` and reproducibility is the caller's responsibility.

### 17.4 Storage-mode tombstone

`MeasurementResult` requires `metadata.storage_mode = StorageMode.OMITTED`. Measurement results never retain quantum states — the upstream `TrajectoryResult` does. Construction with any other storage mode raises `ConventionError`.

### 17.5 Provenance chaining

When a measurement is applied to an upstream `TrajectoryResult`, the measurement inherits that trajectory's `convention_version`, `backend_name`, `backend_version`, and `fock_truncations`; its `request_hash` is copied onto `MeasurementResult.trajectory_hash` so analysis code can rejoin a measurement to the dynamics that produced it. The measurement's `provenance_tags` start with the upstream tags and append `"measurement"` plus any caller-supplied extras.

### 17.6 Channel input semantics

Channels declare what they consume via a class-level `ideal_label`:

- `"probability"` — dimensionless, bounded to `[0, 1]`. `BernoulliChannel`, `BinomialChannel` consume probabilities. `.sample()` raises `ValueError` on out-of-range entries.
- `"rate"` — non-negative mean counts per shot. `PoissonChannel` consumes rates. `.sample()` raises `ValueError` on negative entries.

The orchestrator `sample_outcome(channel, inputs, shots, seed, …)` is input-neutral: it passes `inputs` through to `channel.sample()` and stores them under `MeasurementResult.ideal_outcome[channel.ideal_label]`. The keyword is deliberately `inputs` (not `probabilities` or `rates`) so channels consuming new input types in later dispatches can slot in without breaking callers.

Out-of-range violations are system-boundary input checks (`ValueError`), not convention violations — they indicate a bug in caller code (miscomputed probability / rate reduction), not a schema failure.

### 17.7 Per-shot vs aggregated output shape *(added in Dispatch J)*

Channels advertise their output shape by class, not by flag. The two shapes that v0.2 supports:

- **Per-shot** (Bernoulli, Poisson when per-click granularity matters): output shape `(shots, n_inputs)`, dtype ≥ `int8`, shot axis leading per §17.1. Callers that need aggregate counts reduce along `axis=0` explicitly.
- **Aggregated** (Binomial, Poisson when only totals matter): output shape `(n_inputs,)`, dtype ≥ `int64` to accommodate large shot budgets without per-call overflow checks. The shot axis is absorbed into the count.

Distributionally equivalent channels (Bernoulli-summed ≡ Binomial; per-click Poisson ≡ aggregated Poisson at matching rate) are **not** required to be bit-identical under a shared seed. Library implementations use the most efficient NumPy primitive (`rng.binomial`, `rng.poisson`, or threshold + aggregation), which consumes RNG bits differently depending on the path taken. Tests assert distributional — not bit — equivalence across channel types.

### 17.8 Detector response *(added in Dispatch L)*

A `DetectorConfig` carries three parameters:

- **Efficiency** `η ∈ [0, 1]` — combined collection and quantum efficiency. Thins the emitted Poisson rate multiplicatively.
- **Dark-count rate** `γ_d ≥ 0` — mean stray-light / detector-noise counts per shot. Adds an independent Poisson background.
- **Threshold** `N̂ ≥ 1` — bright / dark classification cut. A shot is classified *bright* when its count is at least `N̂`; otherwise *dark*.

Composition with `PoissonChannel` is **explicit**, not implicit: the orchestrator `sample_outcome` stays detector-agnostic, and callers transform the rate and threshold the counts either side of the channel call:

1. `detected_rate = detector.apply(emitted_rate)` — returns `η · emitted_rate + γ_d`.
2. `result = sample_outcome(channel=PoissonChannel(), inputs=detected_rate, shots=N, seed=…)`.
3. `bright_bits = detector.discriminate(result.sampled_outcome["poisson"])` — returns per-shot `{0, 1}` bits.

The thinning-plus-additive rate composition is exact for Poisson emission: a Poisson(`λ`) stream thinned by Bernoulli(`η`) and added to an independent Poisson(`γ_d`) background is Poisson(`η·λ + γ_d`), with no cross terms.

`DetectorConfig.classification_fidelity(lambda_bright=..., lambda_dark=...)` returns the analytic TP / TN rates and overall fidelity from `scipy.stats.poisson.cdf`. Fidelity is reported as the equal-prior mean `(TP + TN) / 2`; callers weighting by an a-priori probability compute the weighted sum themselves.

### 17.9 Projective-shot readout *(added in Dispatch M)*

Protocol-layer measurements in `measurement/protocols.py` use the *projective-shot* sampling model:

1. For each shot, project the qubit into bright with probability `p_↑` or dark with probability `1 − p_↑`.
2. Sample Poisson photon counts at the *state-conditional* effective rate — `η · λ_bright + γ_d` on the bright branch, `η · λ_dark + γ_d` on the dark branch.
3. Threshold each shot's count against `N̂` to produce a bright/dark bit.

This is the correct model for experimental ion-trap readout, where the detection laser optically pumps the qubit into a pinned bright or dark cycling transition for the duration of the detection window. The qubit "collapses" at the start of the window and emits photons at a single state-conditional rate for the remainder.

The infinite-shots envelope under the projective model is

    bright_fraction∞(t) = TP · p_↑(t) + (1 − TN) · (1 − p_↑(t))

with `TP` and `TN` from `DetectorConfig.classification_fidelity`. This is **linear** in `p_↑`, unlike the Poisson-tail envelope `P(count ≥ N̂ | η · (λ_b·p_↑ + λ_d·(1−p_↑)) + γ_d)` produced by the rate-averaged pipeline used in `tools/run_demo_detected_readout.py`. Protocol-layer code comparing dynamics predictions to experimental readout MUST use the projective model; callers needing the rate-averaged limit compose channel + detector explicitly instead.

Each protocol's result layout is dual-view, per §17.2:

- `ideal_outcome["p_up"]` — the `p_↑(t)` trajectory that drove the projection (exact, from the dynamics).
- `ideal_outcome["bright_fraction_envelope"]` — the `TP · p_↑ + (1 − TN) · (1 − p_↑)` limit.
- `sampled_outcome[f"{label}_counts"]` — `(shots, n_times)` int64 per-shot photon counts.
- `sampled_outcome[f"{label}_bits"]` — `(shots, n_times)` int8 bright/dark bits.
- `sampled_outcome[f"{label}_bright_fraction"]` — `(n_times,)` float64 shot-averaged estimate.

### 17.10 Multi-ion joint readout *(added in Dispatch N)*

Entangled-state measurements require joint sampling — each shot must project both (or all) ions on the same draw, so the correlations in the quantum state survive into the shot record. The `ParityScan` protocol (Dispatch N) implements the two-ion case by reconstructing the joint readout distribution from three expectations:

    P(↑↑) = (1 + ⟨σ_z^i⟩ + ⟨σ_z^j⟩ + ⟨σ_z^i σ_z^j⟩) / 4
    P(↑↓) = (1 + ⟨σ_z^i⟩ − ⟨σ_z^j⟩ − ⟨σ_z^i σ_z^j⟩) / 4
    P(↓↑) = (1 − ⟨σ_z^i⟩ + ⟨σ_z^j⟩ − ⟨σ_z^i σ_z^j⟩) / 4
    P(↓↓) = (1 − ⟨σ_z^i⟩ − ⟨σ_z^j⟩ + ⟨σ_z^i σ_z^j⟩) / 4

and drawing one categorical sample per shot instead of two independent Bernoullis. The trajectory must carry all three expectations — `sigma_z_{i}`, `sigma_z_{j}`, and `parity_{i}_{j}` (provided by `iontrap_dynamics.observables.parity`).

**Why independent Bernoullis fail for entanglement.** Sampling each ion with an independent Bernoulli at its marginal `p_↑^(k)` reproduces the correct single-ion statistics but **factorises** the joint distribution into `P(s_0, s_1) = p_↑^(0)·p_↑^(1) · …`. For a Bell state `|Φ+⟩ = (|↑↑⟩ + |↓↓⟩)/√2`, the marginals are `p_↑^(k) = 1/2` so the factorised joint gives `P(↑↑) = 1/4`, whereas the true joint gives `P(↑↑) = 1/2`. Parity estimates from an independent-Bernoulli pipeline underestimate Bell-state fidelity by a factor of up to 2.

**Projective envelope for the parity estimator.** Under a shared per-ion detector with `TP = P(bright | ↑)` and `TN = P(dark | ↓)` (from `DetectorConfig.classification_fidelity`), the infinite-shots parity estimator satisfies

    ⟨parity⟩∞ = 2 · Σ_s P(s) · P(bits agree | s)  −  1

which, expanded, reduces at zero marginals (`⟨σ_z^i⟩ = ⟨σ_z^j⟩ = 0`) to

    ⟨parity⟩∞ = (TP + TN − 1)² · ⟨σ_z^i σ_z^j⟩  +  (TP − TN)²

The first term is the entanglement-visibility shrinkage `contrast²`; the second is a detector-asymmetry offset that vanishes for symmetric detectors (`TP = TN`). Callers computing Bell-state fidelity from experimental parity records must divide out the contrast² factor.

### 17.11 Sideband inference *(added in Dispatch O)*

Motional-state thermometry on trapped ions uses the short-time Leibfried–Wineland ratio between red- and blue-sideband Rabi excitations:

    P↑_RSB(t) / P↑_BSB(t)  →  n̄ / (n̄ + 1)    as (2Ωη t)² → 0

inverting to `n̄ = r / (1 − r)`. The formula is exact in the short-time limit for *any* motional distribution — the $\sum_n p_n (2Ωη\sqrt{n+1} t)^2$ expansion weights by `n̄ + 1` on BSB and by `n̄` on RSB, independent of the shape of the distribution. `SidebandInference.run` evaluates the ratio element-wise.

**Fidelity correction before the ratio.** The detector's projective envelope `TP·p_↑ + (1−TN)·(1−p_↑)` (§17.9) is linearly invertible: `p̂_↑ = (bright_fraction − (1 − TN)) / (TP + TN − 1)`. The protocol applies this inversion to *both* sideband bright fractions before computing the ratio. Detector contrast `TP + TN − 1` must be strictly positive; a detector that can't distinguish bright from dark (`TP ≈ 1 − TN`) makes the inversion ill-defined and raises `ValueError`.

The protocol reports both the fidelity-corrected `n̄_estimate` (the principled one) and the raw-ratio `n̄_from_raw_ratio` (the naive one) so callers can see the size of the fidelity correction — at high-fidelity detectors the two converge; at low-contrast detectors the uncorrected estimate is visibly biased toward `(1 − TN) / TP` asymptotically.

**Independent RNG streams per sideband.** The two sideband readouts consume different shots and therefore different RNG streams. `SidebandInference.run(seed=K)` uses `np.random.SeedSequence(K).spawn(2)` to derive bit-reproducible but statistically independent generators — callers MUST NOT pre-split seeds themselves or reuse a single seed across sidebands.

**NaN propagation.** The ratio is NaN wherever `p_up_bsb ≤ 0` (indeterminate) or `r ≥ 1` (unphysical: RSB ≥ BSB leaves the short-time regime). `n̄` inherits the NaN. Callers mask with `np.nanmean` / `np.nanmedian` rather than expecting the protocol to regularise.

### 17.12 Binomial confidence intervals *(added in Dispatch P — §17 freeze target)*

Two estimators are supported for binomial proportion CIs on shot counts, both fully vectorised over ``(successes, trials)``:

- **Wilson score** (``wilson_interval``). The recommended default. Closed-form arithmetic, well-behaved at ``p̂ ∈ {0, 1}``, near-nominal coverage for modest ``n`` (typically 93–97 % at nominal 95 %). Formula:

        p̂ = k / n,   z = Φ⁻¹((1 + confidence) / 2)
        centre     = (p̂ + z² / (2n)) / (1 + z² / n)
        half_width = (z / (1 + z² / n)) · √(p̂(1 − p̂) / n + z² / (4n²))
        (lower, upper) = (centre − half_width, centre + half_width)

- **Clopper–Pearson** (``clopper_pearson_interval``). Exact binomial-quantile interval via the Beta distribution. Conservative — actual coverage ≥ nominal. Use for worst-case uncertainty reporting:

        α = 1 − confidence
        lower = Beta⁻¹(α / 2;     k,     n − k + 1)   (0 when k = 0)
        upper = Beta⁻¹(1 − α / 2; k + 1, n − k)       (1 when k = n)

No Wald interval is shipped: its coverage collapses near the ``p̂ ∈ {0, 1}`` extremes that arise routinely in ion-trap readout (ground-state RSB probes, high-fidelity detectors on |↓⟩ shots).

**Non-nesting caveat.** Wilson and Clopper–Pearson do **not** strictly nest — neither interval contains the other uniformly. CP is conservative in coverage, Wilson is near-nominal, but at specific ``(k, n)`` values CP can be slightly *narrower* than Wilson on one side. Callers comparing methods should not assume one bounds the other.

**z / confidence convention.** Two-sided intervals use ``z = Φ⁻¹((1 + confidence) / 2)`` via ``scipy.stats.norm.ppf``. ``confidence = 0.95`` therefore gives ``z ≈ 1.959963984…`` — the standard textbook value.

**Boundary snap.** ``wilson_interval`` snaps ``k = 0`` to ``lower = 0`` exactly and ``k = n`` to ``upper = 1`` exactly, overriding sub-epsilon floating-point noise. CP produces exact 0 / 1 by construction. Callers comparing against 0 / 1 with ``==`` succeed at boundaries.

**Input contract.** Both functions raise ``ValueError`` on ``confidence ∉ (0, 1)``, ``trials < 1``, ``successes < 0``, or ``successes > trials`` (element-wise). They do not silently clip or quietly skip — input-boundary errors are surfaced immediately.

---

**§17 freeze.** Dispatch P is the last staged extension to §17 under the v0.2 measurement-layer track. With the statistics surface in place, §17 is a complete read-through for the Convention Freeze gate at the v0.2 release. Post-freeze additions require a CONVENTIONS.md version bump per the Endorsement Marker below.

---

## 18. Systematics layer *(staged — v0.2 Convention Freeze target)*

**Status:** opened at Dispatch R (`src/iontrap_dynamics/systematics/`). Rules below are staged; additions across Dispatches S–U may tighten them, and the full section seals at the v0.2 release alongside §17. Treat any call-site depending on §18 details as provisional until the freeze.

### 18.1 Noise taxonomy

The systematics layer distinguishes three physically distinct noise classes, each with its own composition pattern:

- **Jitter** (this dispatch) — *stochastic* shot-to-shot parameter fluctuations. Each shot sees a different Hamiltonian drawn from a zero-mean distribution around the nominal value. Physical sources: laser intensity noise, AOM amplitude noise, detuning drift over a detection window, magnetic-field fluctuations. Ensemble mean of an observable dephases from the noise-free signal — the inhomogeneous-dephasing signature.
- **Drifts** (pending) — *systematic* bias shifts. A parameter sits at `p₀ + Δ` for the whole run, not `p₀`. Physical sources: calibration errors, uncorrected slow drifts, mis-set detuning. The signal acquires a constant offset; no ensemble averaging is needed, just a perturbation study.
- **SPAM** (pending) — State Preparation And Measurement errors. Imperfect ground-state preparation shifts the initial state; measurement infidelity shifts the reported outcome. Detection-side SPAM is already captured by :class:`DetectorConfig` (§17.8); preparation-side SPAM will be Dispatch U.

Each class composes differently with the solver: jitter via ensemble of solves (this dispatch), drifts via parameter perturbation (single solve), SPAM via initial-state perturbation (single solve from a noisy state). The three never get mixed in one call — callers stack them explicitly.

### 18.2 Jitter composition pattern *(added in Dispatch R)*

Jitter primitives are frozen dataclasses carrying a noise-amplitude parameter (e.g. `RabiJitter.sigma`). They expose a `.sample_multipliers` / `.sample_offsets` method that returns per-shot perturbation factors, and are materialised into perturbed input configs via module-level helpers (`perturb_carrier_rabi`, later `perturb_detuning`, etc.). The user-facing pattern is:

```python
drives = perturb_carrier_rabi(drive, jitter, shots=N, seed=0)
results = [solve(..., drive=d, ...) for d in drives]
expectations = np.stack(
    [r.expectations["sigma_z_0"] for r in results], axis=0
)  # shape (N, n_times)
ensemble_mean = expectations.mean(axis=0)
ensemble_std  = expectations.std(axis=0)
```

The `(N, n_times)` shape echoes the `(shots, n_times)` pattern from §17.1 — shot axis leading, time axis trailing — so jittered-dynamics aggregation looks identical to measurement-layer sampling from the caller's perspective. No new `Result` type is needed for Dispatch R; users compose with NumPy. A later dispatch may add an `EnsembleResult` wrapper once the pattern has been exercised by multiple jitter sources.

### 18.3 Jitter primitives *(opened in Dispatch R, extended in Dispatch S)*

Three shot-to-shot jitter primitives ship on `DriveConfig` fields, each a frozen dataclass carrying a single `sigma*` parameter plus an identifier `label`:

- **`RabiJitter(sigma=σ)`** — *multiplicative* Gaussian noise on `carrier_rabi_frequency_rad_s`. Each shot scales by `(1 + ε)` with `ε ~ Normal(0, σ)`. `σ` is dimensionless (a relative amplitude); realistic values `σ ≲ 0.05`. The drive sign can in principle flip at large `σ` via Gaussian tails; negligible at realistic values.
- **`DetuningJitter(sigma_rad_s=σ)`** — *additive* Gaussian noise on `detuning_rad_s`. Each shot adds `Δδ ~ Normal(0, σ)`. Units are rad·s⁻¹ (SI per §1). Typical values `σ / 2π ≈ 10 Hz – 1 kHz`. Composes via `perturb_detuning(drive, jitter, shots, seed)`.
- **`PhaseJitter(sigma_rad=σ)`** — *additive* Gaussian noise on `phase_rad`. Each shot adds `Δφ ~ Normal(0, σ)`. Units are radians. Phase is **not** wrapped — `DriveConfig.phase_rad` accepts any real value, and builders apply `exp(i φ)` internally. Single-pulse Rabi flopping is insensitive to a constant phase; `PhaseJitter` becomes visible only on multi-pulse sequences (Ramsey, MS) or interferometric observables.

All three share these rules:

- `σ` must be ``>= 0``; ``σ = 0`` is a valid no-op used for pipeline checks.
- Each primitive exposes a `.sample_multipliers(shots, rng)` (Rabi) or `.sample_offsets(shots, rng)` (Detuning / Phase) method returning a `(shots,)` `float64` array. Multipliers are multiplicative (`1 + ε`); offsets are additive (`Δ`).
- Module-level `perturb_*` helpers materialise a `tuple[DriveConfig, ...]` of length `shots` from the sampled perturbations; other `DriveConfig` fields pass through untouched via `dataclasses.replace`.
- Bit-reproducibility follows §17.3: supplying an integer `seed` to the composition helper makes the output tuple deterministic given `(drive, jitter, shots, seed)`. Callers wanting independent streams across multiple jitter sources should use `np.random.SeedSequence.spawn(n)` to derive per-source seeds (same discipline as §17.11).

### 18.4 Drift primitives *(added in Dispatch T)*

Drifts are *systematic* single-value offsets, not stochastic noise. Parallels to §18.3's jitter primitives, one per `DriveConfig` field:

- **`RabiDrift(delta)`** — multiplicative relative offset on `carrier_rabi_frequency_rad_s`. Applies `Ω → Ω · (1 + delta)` once. `delta` dimensionless, **unsigned** (unlike `RabiJitter`'s `σ ≥ 0`) — a drift models a directional mis-calibration (tuned low or high).
- **`DetuningDrift(delta_rad_s)`** — additive offset on `detuning_rad_s`. Units rad·s⁻¹ (SI per §1). Either sign.
- **`PhaseDrift(delta_rad)`** — additive offset on `phase_rad`. Units rad. Phase not wrapped per §18.3.

Shared rules:

- Drifts are **deterministic** — two `apply_*(drive, drift)` calls with the same inputs produce bit-identical outputs. No seed, no RNG.
- `apply_*` returns a single :class:`DriveConfig` (contrast with `perturb_*` which returns a tuple over shots). There is no ensemble.
- Non-drifted `DriveConfig` fields pass through via `dataclasses.replace`.
- Drift respects `DriveConfig` invariants: `apply_rabi_drift` with `delta ≤ −1` causes a non-positive Rabi frequency, which `DriveConfig.__post_init__` rejects with `ConventionError`. Callers sweeping through `delta = −1` must flip `phase_rad` by `π` instead of crossing the sign boundary.
- Canonical scan pattern (no dedicated helper — just a Python comprehension):

    ```python
    deltas = np.linspace(-0.1, 0.1, 21)
    results = [
        solve(..., drive=apply_rabi_drift(base_drive, RabiDrift(delta=d)))
        for d in deltas
    ]
    ```

### 18.5 State-preparation (SPAM) errors *(added in Dispatch U — §18 freeze target)*

Measurement-side SPAM is already covered by §17.8 (`DetectorConfig` — `TP / TN` rates from threshold + rate parameters). §18.5 covers the **preparation side**: imperfect initial-state construction. Two primitives ship:

- **`SpinPreparationError(p_up_prep)`** — probability `p ∈ [0, 1]` that the spin is pumped to `|↑⟩` instead of `|↓⟩`. Models incomplete optical pumping. The helper `imperfect_spin_ground(error)` returns the 2 × 2 classical-mixture density matrix `(1 − p) |↓⟩⟨↓| + p |↑⟩⟨↑|`. `p = 0` collapses to pure ground.
- **`ThermalPreparationError(n_bar_prep)`** — residual mean phonon occupation `n̄ ≥ 0` after cooling. Models sub-ideal sideband cooling. The helper `imperfect_motional_ground(error, fock_dim)` returns the Fock-truncated thermal density matrix via `qutip.thermal_dm`. `n̄ = 0` collapses to pure `|0⟩⟨0|`.

Composition rule (preparation side always lands a **density matrix**, never a ket):

    from iontrap_dynamics.states import compose_density
    rho_0 = compose_density(
        hilbert,
        spin_states_per_ion=[imperfect_spin_ground(spin_err)],
        mode_states_by_label={"axial": imperfect_motional_ground(thermal_err, fock_dim=N)},
    )
    result = solve(..., initial_state=rho_0, ...)

Because `compose_density` returns a density matrix, the downstream solver runs `mesolve` naturally — no additional wiring needed. The rest of the trajectory machinery is agnostic to whether the initial state is pure or mixed.

**Fock-truncation guard.** `imperfect_motional_ground` raises `ValueError` when `n̄_prep ≥ fock_dim − 1` — at that point the thermal distribution extends past the truncation and the prepared state is a poor approximation of the intended physics. Callers either increase `fock_dim` (recommended: `≥ 4·n̄ + 4`) or reduce `n̄`. This guard is *preparation-time* and is distinct from the §13 Fock-saturation ladder, which runs at solver time.

**No RNG, no seed.** SPAM errors produce deterministic mixed states — there is no sampling over shots. The classical mixture encoded in the density matrix already captures the ensemble average over preparation outcomes. Bit-reproducibility is automatic.

---

**§18 freeze.** Dispatch U is the last staged extension to §18 under the v0.2 systematics-layer track. With jitter (stochastic, §18.3), drift (systematic, §18.4), and SPAM (state preparation, §18.5) all documented, §18 is a complete read-through for the Convention Freeze gate at the v0.2 release, alongside §17 (measurement layer) which sealed at Dispatch P. Post-freeze additions to either section require a CONVENTIONS.md version bump per the Endorsement Marker below.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This document is a Coastline draft within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg). Conventions herein are binding within `iontrap-dynamics` at this version. §17 (measurement layer) and §18 (systematics layer) are closed under the v0.2 Convention Freeze; §1–16 carry forward from the Phase 0 draft unchanged. Post-freeze additions to any section require a new CONVENTIONS.md version bump with an explicit Convention Freeze gate.

**Convention version:** 0.2 · 2026-04-21 · v0.2 Convention Freeze.
**Workplan reference:** `WORKPLAN_v0.3.md` §0.A.
