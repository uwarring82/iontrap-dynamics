# CONVENTIONS

**Physical, numerical, and notational conventions for `iontrap-dynamics`**

Version 0.1-draft · Drafted 2026-04-17 · Status: Phase 0 working document

**Classification:** Coastline (hard constraints per T(h)reehouse +EC CD 0.9).
**Licence:** CC BY-SA 4.0.
**Scope:** Conventions sufficient for Phase 0 (v0.1-alpha). Phase 1 extensions (measurement channels, systematics parameterisations) will be staged in `CONVENTIONS.md` v0.2+ with explicit Convention Freeze gates at each library minor-version release. Convention **additions** are permitted without a freeze; convention **changes** require one.
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

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This document is a Coastline draft within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg). Conventions herein are binding within `iontrap-dynamics` at this version; extensions for Phase 1 measurement and systematics layers are staged and will carry explicit Convention Freeze gates.

**Convention version:** 0.1-draft · 2026-04-17 · Phase 0 working document.
**Workplan reference:** `WORKPLAN_v0.3.md` §0.A.
