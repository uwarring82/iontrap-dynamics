# Workplan: `iontrap-dynamics`

**Repository for open-system quantum dynamics of trapped-ion spin–motion systems**

Version 0.3.5 (amended §4.0 repo-hosting · §5.0 release-mapping · §5.1 v0.2 release · §5.2 post-v0.2 on-`main` · §5.3 β.4 as v0.3.x follow-up) · Drafted 2026-04-17 · Status: v0.2.0 tagged 2026-04-21; Phase 2 JAX-backend time-independent surface on `main`; β.4 time-dependent extension scoped as v0.3.x follow-up; see §5.3
Supersedes v0.2.1. Changes vs v0.2.1 summarised in Appendix B (CD integration).

**Classification:** Coastline (hard constraints per T(h)reehouse +EC CD 0.9).
**Licence:** CC BY-SA 4.0 (this document as coastline). See §4 for the split architecture governing the repository.
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (cd-rules v1.7.0, consumed via Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

---

## 1. Scope and non-scope *(Coastline)*

### Identity statement

> The library is **domain-specific** to trapped-ion spin–motion dynamics at the level of public identity, but **configuration-general** at the level of physical realisation. All physical parameters — species, isotopes, laser wavevectors, mode eigenvectors, trap-frame definitions, and mode-structure metadata relevant to spin–motion modelling — enter through typed configuration objects, never as hidden defaults in solver code.

This is the primary design rule. Every downstream decision in this workplan is derivable from it.

### In scope

- Unitary and dissipative dynamics of systems composed of N_s two-level spins and N_m harmonic motional modes
- Standard ion-trap Hamiltonians: carrier, red/blue sideband, Mølmer–Sørensen, parametric modulation, stroboscopic AC drives
- Standard state preparations: thermal, Fock, coherent, squeezed (single- and two-mode), arbitrary spin rotations
- Standard observables: populations, motional occupation numbers, spin expectations, Wigner functions, entropy, EoF, logarithmic negativity
- Typed measurement models: finite-shot sampling, detector imperfections, readout infidelities (Phase 1+)
- Typed systematics: frequency drifts, amplitude miscalibration, timing jitter, SPAM errors (Phase 1+)
- QuTiP as reference backend; architecture backend-agnostic for future JAX/Dynamiqs backend
- Laptop-runnable regression tests and worked examples

### In scope — configuration generality

The library must support, through configuration only, with no code changes:

- Arbitrary ion species and isotopes (²⁵Mg⁺, ⁴⁰Ca⁺, ⁴³Ca⁺, ⁸⁸Sr⁺, ¹⁷¹Yb⁺, …)
- Arbitrary laser geometries (axial, radial, Raman Δk, oblique addressing)
- Arbitrary normal-mode decompositions (linear chain, zigzag, 2D crystals), provided the normal-mode decomposition is supplied externally in the declared convention
- Arbitrary measurement protocols (direct spin readout, spin-mediated motional readout, parity scan, homodyne-equivalent quadrature sampling)

### Out of scope (explicitly)

- Trap geometry simulation — use `trical` or `pylion`
- Molecular dynamics of ion crystals — use `pylion`
- Pulse-sequence compilers / hardware control — use ARTIQ, OxfordControl, local stacks
- Electromagnetic field modelling
- **Full laboratory digital twins** including electrode-field simulation, pulse compiler emulation, hardware timing stack emulation, CAD/EM/control-system models

### Boundary Decision Tree (for contributors)

```
Does the feature modify the spin or motional quantum state?
├── YES → Is it a Hamiltonian / Lindbladian / initial state / observable?
│         ├── YES → In scope (core layer)
│         └── NO  → Is it a measurement channel or sampled observation?
│                   ├── YES → In scope (measurement layer, Phase 1+)
│                   └── NO  → Out of scope
└── NO  → Does it model apparatus imperfections (drifts, jitter, detector)?
          ├── YES → In scope (systematics layer, Phase 1+)
          └── NO  → Out of scope (trap geometry, control stack, EM, lab IT)
```

**Scope clarification for apparatus models.** Apparatus models are in scope only insofar as they modify Hamiltonian parameters, dissipative channels, or measurement outcomes at the level of the simulated quantum experiment. They are not a vehicle for importing lab infrastructure, control-software emulation, or non-quantum environmental modelling.

### Relationship to existing tools

| Tool | What it does | Our position |
|---|---|---|
| QuTiP | General open quantum systems | Dependency; we provide ion-trap-specific wrappers, conventions, measurement layers |
| pylion | Classical MD of ion crystals | Complementary; no overlap |
| trical | Trap electrode design | Complementary; no overlap |
| IonSim.jl | Ion-trap Hamiltonians in Julia | Parallel effort; our niche is Python/QuTiP and Clock-School pedagogy |
| Dynamiqs | GPU quantum dynamics (JAX) | Future backend target |
| `threehouse-plus-ec/cd-rules` | Corporate design blueprint | Upstream design authority; we consume via Model B |

---

## 2. Design principles *(Coastline)*

Binding. Every architectural choice must cite one.

1. **Conventions before code.** No feature implemented until its physical and numerical conventions are documented. Staged: `CONVENTIONS.md` v0.1 sufficient for Phase 0, with explicit Convention Freeze gates at each version milestone.

2. **No hidden laboratory assumptions.** Mass, wavelength, **k**-vector, mode orientation, normal-mode participation enter through named configuration objects — never through implicit defaults in solver code.

3. **Reproducibility is stratified, not absolute.**
   - *Same platform, pinned dependency stack, same seed* → bit-identical reference arrays required.
   - *Cross-platform or dependency drift* → numerical agreement within defined tolerances (default 10⁻¹⁰).
   - *Metadata and parameter hashes* → exact always.

4. **Pedagogical legibility outranks performance.** First reader is a Clock-School student.

5. **One way to do it — at the public API level.** Internal adapters may exist provided they normalise to canonical form.

6. **Positional return values are forbidden.** All state properties, simulation outputs, parameter bundles use dataclasses or typed dicts with named fields.

7. **Cache integrity non-negotiable.** No serialised file loaded without verifying parameter-hash match.

8. **Physical system is a parameter, not a constant.** No wavelength, mass, trap frequency hardcoded outside named configuration objects.

9. **Three-layer architecture: physics / apparatus / observation.** Dynamics core evolves states. Apparatus layer models systematics. Observation layer maps ideal observables to finite-shot sampled outcomes. Noise and readout never contaminate the Hamiltonian layer.

10. **Plotting is strictly downstream.** Plotting takes arrays, never quantum objects. No physics reconstruction inside plotting. No backend-specific object calls inside plotting.

11. **Lock–Key rule.** Core physics (lock) stable and versioned; user analyses (key) arbitrary. Library exposes stable locks.

12. **Corporate Design compliance.** The repository adopts the T(h)reehouse +EC Corporate Design blueprint (`cd-rules`) as a downstream consumer under **Model B** (distributed copy + SHA-256 checksum). All visual identity, colour tokens, typography, accessibility rules, folder conventions, and deprecation protocols flow from cd-rules at its tagged release. Drift detected by checksum mismatch; updates within one release cycle.

13. **Coastline / Sail labelling.** Documentation sections are labelled as Coastline (testable constraints) or Sail (adaptive guidance) per CD 0.9. This workplan is Coastline. `CONVENTIONS.md` is Coastline. Tutorials and pedagogical essays are Sail.

14. **No version numbers in filenames.** Versioning lives in git tags, not file paths (CD 0.6). Breaking asset changes use new paths, not v1/v2 suffixes.

15. **Deprecation, not deletion.** Superseded assets move to `archive/` with a dated note (CD 0.8). Git history is not a substitute for visible archival.

---

## 3. Architectural skeleton *(Coastline)*

```
iontrap-dynamics/
├── LICENCE                     # Split-licence declaration pointing to per-folder LICENCEs
├── README.md                   # Endorsement marker + stewardship + quick start
├── CONVENTIONS.md              # Coastline · CC BY-SA 4.0 · authoritative, versioned
├── CITATION.cff                # Zenodo DOI
├── CHANGELOG.md                # Keep-a-Changelog format
├── pyproject.toml
├── assets/                     # Design assets consumed from cd-rules via Model B
│   ├── LICENCE                 # MIT (as declared by cd-rules)
│   ├── SOURCE.md               # Origin repo + commit hash + SHA-256 of each file
│   ├── emblem-16.svg
│   ├── emblem-32.svg
│   ├── emblem-64.svg
│   ├── tokens.css
│   ├── wordmark-full.svg
│   └── wordmark-silent.svg
├── src/iontrap_dynamics/       # MIT
│   ├── __init__.py
│   ├── conventions.py          # ConventionSet object, versioned; constants; type aliases
│   ├── species.py              # IonSpecies: element, isotope, mass, transitions, wavelengths
│   ├── drives.py               # DriveConfig / LaserConfig: k-vector, frequency, phase, polarisation
│   ├── modes.py                # ModeConfig: frequency, eigenvector, label, ion participation
│   ├── system.py               # IonSystem: composes species + drives + modes + trap geometry
│   ├── hilbert.py              # HilbertSpace: cutoffs, tensor structure, operator cache
│   ├── states.py               # State preparation
│   ├── hamiltonians.py         # Builders: carrier, sideband, MS, parametric, stroboscopic
│   ├── sequences.py            # Dispatcher: composes builders + solver
│   ├── observables.py          # Named observable registry (ideal)
│   ├── results.py              # TrajectoryResult, MeasurementResult — canonical schemas
│   ├── measurement/            # Phase 1+ · observation layer
│   ├── systematics/            # Phase 1+ · apparatus layer
│   ├── backends/               # MIT
│   ├── io/                     # MIT
│   └── plotting/               # MIT
├── tests/                      # MIT
│   ├── regression/
│   │   ├── migration/
│   │   ├── analytic/
│   │   └── invariants/
│   ├── unit/
│   └── conventions/
├── examples/                   # MIT (code) · CC BY-SA 4.0 (accompanying READMEs)
│   └── 01_sideband_flopping/
├── docs/                       # Coastline docs: CC BY-SA 4.0; tutorials: CC BY-NC-SA 4.0
│   ├── LICENCE                 # Declares split
│   ├── index.md                # Coastline
│   ├── getting-started.md      # Coastline
│   ├── conventions.md          # Coastline — rendered CONVENTIONS.md
│   ├── boundary-decision-tree.md  # Coastline
│   └── tutorials/              # Sail · CC BY-NC-SA 4.0
├── archive/                    # Deprecation per CD 0.8
└── .github/
    ├── workflows/              # CI: tests, accessibility, CD hash check, docs build
    └── ISSUE_TEMPLATE/
```

### Key design decisions with justifications

**Configuration-layer modules (`species.py`, `drives.py`, `modes.py`).** Separate concerns that legacy `qc.py` fused. Changing from ²⁵Mg⁺ to ⁴⁰Ca⁺ is a configuration change, not a code change.

**`ConventionSet` object is versioned.** Every `IonSystem` carries a convention-version reference. Every `TrajectoryResult` records which version produced it.

**`IonSystem` vs `HilbertSpace` separation.** Physics vs numerics. Same `IonSystem` generates different Hilbert spaces at different truncations without re-specifying physics.

**Lamb–Dicke parameter 3D-native from day one:**

```python
eta = lamb_dicke_parameter(
    k_vec=drive.wavevector,                                  # shape (3,), m⁻¹
    mode_participation=mode.eigenvector_at_ion(ion_index),   # shape (3,)
    mass=species.mass,
    mode_frequency=mode.omega,
    conventions=system.conventions,
)
```

No 1D shortcut, not even as convenience.

**Operator cache** lazy, keyed by `(subsystem_index, operator_name, hilbert_signature)`.

**Hamiltonians are builders**, not pre-built objects. Solver call unified in `sequences.py`.

**Observables are a registry** via Strategy pattern.

**Backend contract richer than just `solve()`.** Includes operator construction, state construction, time-dependent encoding, expectation computation, partial-trace, output conversion.

**Results schema defined in Phase 0.** `TrajectoryResult` with `times`, `states` (or lazy handle), `expectations`, `metadata`, `request_hash`, `backend_info`, `convention_version`, `warnings`. Locked before any builder written.

**Measurement layer strictly downstream.** Finite-shot sampling, detectors, readout infidelities live in `measurement/`, never in `hamiltonians.py`.

**Archival format: `.npz` + JSON/YAML manifest.** Not pickle.

**CD-compliant folder structure.** `LICENCE` (split declaration), `assets/` (with `LICENCE` + `SOURCE.md`), `archive/` (deprecation per CD 0.8), `docs/` built against cd-rules `tokens.css` and typography.

---

## 4. Governance and licence *(Coastline)*

### 4.0 — Repository-hosting amendment (2026-04) *(Coastline, new in v0.3.2)*

Added after Phase 0 shipped on `main` under the maintainer's personal GitHub account rather than the neutral `open-iontrap/` org named in the "Repository topology" subsection below. The §4 topology clause and the Phase 0 exit criterion at §5 must be read through this amendment.

**Actual repository mapping (amends §4 below):**

| Asset | Originally planned | Actual |
|---|---|---|
| Primary code repo | `github.com/open-iontrap/iontrap-dynamics` | **`github.com/uwarring82/iontrap-dynamics`** |
| Design authority | `github.com/threehouse-plus-ec/cd-rules` | `github.com/threehouse-plus-ec/cd-rules` (unchanged) |
| Maintainer working fork | personal dev space | (collapsed into primary) |

**Why the merge happened.** The `open-iontrap/` org was a future-state positioning decision (CD 0.3's Model B neutrality), not a precondition for Phase 0. Creating the org before any code shipped would have required signalling a community-governance story Phase 0 did not need. The split-licence layout and Model B asset-propagation pipeline are intact under `uwarring82/`; the eventual org move is an address change, not a governance change.

**Consequence for §4 and §5 below.** The "Repository topology" subsection still names `open-iontrap/` as the primary repo for historical continuity. Read it through this amendment: the interim primary is `uwarring82/iontrap-dynamics`, and the Phase 0 exit criterion "Repository scaffolded under `open-iontrap/`, CI green" is satisfied under `uwarring82/iontrap-dynamics` with CI green. The `[project.urls]` block in `pyproject.toml` and the `site_url` / `repo_url` / `repo_name` entries in `mkdocs.yml` each carry a migration-aware comment listing the swap needed when the org move happens.

### Split licence architecture (per CD 0.3)

The repository adopts the T(h)reehouse +EC split architecture because it matches the library's epistemic structure: the workplan and CONVENTIONS are coastlines, tutorials are authored works, assets and code are infrastructure.

| Layer | Harbour term | Content | Licence | SPDX |
|---|---|---|---|---|
| Coastline docs | Coastline | This workplan, `CONVENTIONS.md`, architectural specs, Boundary Decision Tree, schema documents, result-schema specifications | CC BY-SA 4.0 | `CC-BY-SA-4.0` |
| Authored tutorials | Sail | `docs/tutorials/`, pedagogical essays, Clock-School teaching materials, worked-example narratives | CC BY-NC-SA 4.0 | `CC-BY-NC-SA-4.0` |
| Python source & tooling | Handbook | `src/`, `tests/`, `.github/workflows/`, build scripts, `pyproject.toml` | MIT | `MIT` |
| Design assets | Handbook | `assets/*` (emblems, wordmarks, `tokens.css`) | MIT | `MIT` (as declared by cd-rules) |
| External fonts | External | IBM Plex Mono, Crimson Pro | SIL OFL 1.1 | `OFL-1.1` |

**Per-folder LICENCE declarations.** Root `LICENCE` declares the split and points to per-folder files. `assets/LICENCE` = MIT. `docs/LICENCE` = split between Coastline and Sail. `src/` inherits MIT. When folder content is mixed, the more restrictive licence applies unless individual files declare otherwise.

### Asset propagation (Model B, per CD 0.10)

Design assets enter `iontrap-dynamics/assets/` as distributed copies from `threehouse-plus-ec/cd-rules` at a tagged commit. The record lives in `assets/SOURCE.md`:

```markdown
# Asset source record

Source repository: https://github.com/threehouse-plus-ec/cd-rules
Source tag:        cd-vX.Y.Z
Source commit:     <full SHA>
Copied on:         2026-04-XX

## Files and checksums

| File | SHA-256 | Size (bytes) |
|---|---|---|
| emblem-16.svg | <hash> | <size> |
| emblem-32.svg | <hash> | <size> |
| emblem-64.svg | <hash> | <size> |
| tokens.css | <hash> | <size> |
| wordmark-full.svg | <hash> | <size> |
| wordmark-silent.svg | <hash> | <size> |
```

CI performs periodic hash comparison against the declared `cd-vX.Y.Z` tag. Drift is flagged, not silently tolerated. Updates happen within one release cycle of upstream.

### Repository topology (two-repo model)

- **Design authority:** `github.com/threehouse-plus-ec/cd-rules` (upstream)
- **Primary code repo:** `github.com/open-iontrap/iontrap-dynamics` (neutral community org, consumes cd-rules via Model B)
- **Maintainer working fork:** personal dev space

Why not host `iontrap-dynamics` under `threehouse-plus-ec/`? The neutral `open-iontrap` positioning signals community-open software that adopts the CD layer by choice, not by ownership. This is the pattern CD 0.3 anticipates: "A collaborator at PTB or AIMS can embed the emblem in their page without consulting a lawyer" — precisely Model B.

### Governance

- **Citation:** CITATION.cff with Zenodo DOI from v0.1-alpha
- **Versioning:** Semantic versioning. Breaking changes below v1.0 require one-line CHANGELOG justification. **No version numbers in filenames** (CD 0.6).
- **Contribution model:** fork + PR. Every PR passes CI (tests, accessibility, CD hash check) and either preserves numerical output or documents the change with regression-test update.
- **Deprecation:** superseded artefacts move to `archive/` with dated notes per CD 0.8.
- **v1.0 gate (technical, not prestige):**
  - API stable for two consecutive minor versions
  - Test coverage ≥ 85%
  - Breaking-change policy frozen
  - All three regression layers fully populated
  - External adoption evidence is a separate *prestige* milestone
- **Endorsement Marker** on every document (CD 0.7)
- **Maintainer:** U. Warring, AG Schätz. Full governance document at v0.5.

---

## 5. Phased delivery

*(Phase milestones are Coastline; resource estimates are Sail.)*

### 5.0 — Release-mapping amendment (2026-04) *(Coastline, new in v0.3.1)*

Added after Phase 0 and the Phase 1 core builder layer both shipped back-to-back on `main`. The phase boundary collapsed in practice — by 2026-04-19 every Phase 0 exit criterion below plus the core Phase 1 builder / observable / state-prep / diagnostic surface are on `main` under a single `[Unreleased]` CHANGELOG block.

**Actual release mapping (amends §5 below):**

| Scope | Originally planned release | Actual release |
|---|---|---|
| Phase 0 foundations | `v0.1-alpha` | `v0.1-alpha` |
| Phase 1 — core builder / observable / state-prep / diagnostic surface | `v0.2` | **`v0.1-alpha` (combined)** |
| Phase 1 — measurement layer (`measurement/`) | `v0.2` | `v0.2` (unchanged) |
| Phase 1 — systematics layer (`systematics/`) | `v0.2` | `v0.2` (unchanged) |
| Phase 1 — logarithmic negativity / EoF as registered observables | `v0.2` | `v0.2` (unchanged) |
| Phase 2 — performance, JAX backend | `v0.3` | `v0.3` (unchanged) |

**Why the merge happened.** The Phase 0 regression-harness + schema + convention frame made the Phase 1 core builder family a thin shell on top of already-locked primitives. Landing one builder per commit behind a frozen result schema took less time than staging two releases would have cost. The measurement and systematics layers are genuinely new physics on top of the core and stay on their own `v0.2` line, as originally planned.

**Consequence for §5 below.** The Phase 0 and Phase 1 section headers still name their originally-planned release tags for historical continuity. Read them through this amendment: `v0.1-alpha` now covers the two rows marked above, and Phase 1's `v0.2` target now covers only the measurement + systematics additions plus any Phase 1 observables (logarithmic negativity, EoF) that were not already landed as part of the core surface.

**Shipped on `main` as of 2026-04-19, under `[Unreleased]`:**

- Configuration layer — `operators`, `species`, `drives`, `modes`, `system`, `hilbert`, `states` (with `coherent_mode` / `squeezed_vacuum_mode` / `squeezed_coherent_mode` factories).
- Full four-family Hamiltonian matrix (carrier / red-sideband / blue-sideband / MS) × (exact / detuned), plus `modulated_carrier_hamiltonian` and `two_ion_{red,blue}_sideband_hamiltonian`, plus a `full_lamb_dicke: bool` flag on the sideband builders.
- Analytic-regression helpers (lamb-Dicke parameter, carrier / sideband / generalised Rabi formulas, MS-gate closing condition).
- `observables` factory + `sequences.solve()` dispatcher with the §13 Fock-saturation ladder wired in.
- Hash-verified cache I/O (`cache.save_trajectory` / `load_trajectory`).
- All three Phase 0.F performance benchmarks active and under threshold.
- Four demo tools (`run_benchmark_sideband`, `run_demo_carrier`, `run_demo_gaussian_pulse`, `run_demo_ms_gate`) with canonical `manifest.json` + `arrays.npz` + narrative `demo_report.json` artefacts committed under `benchmarks/data/`.
- Three-tier regression harness: analytic + invariant permanent, migration tier with scenarios 1 and 5 active (2/3/4 skipped with probe-informed blockers).
- Accessibility CI gate at WCAG 2 Level A (hard fail), AA advisory.

The boundary-tree between "what's in `v0.1-alpha`" and "what's in `v0.2`" is: does it touch the measurement / systematics / extra-observable layers? If yes, `v0.2`. If no, already on `main` for `v0.1-alpha`.

**Consequence for WCAG clauses in §5 and §6 (amended in v0.3.2).** Three older clauses still name WCAG 2.2 AA as the CI gate:

- the Phase 0 exit criterion below ("Accessibility smoke check: docs site passes WCAG 2.2 AA rules A1–A29 from CD 12A");
- Phase 0.H step 5 ("Verify docs site against WCAG 2.2 AA rules A1–A29 (CD 12A) — this is a CI gate");
- the §6 risks-table row ("WCAG 2.2 AA rules A1–A29 as CI gate per CD 12A").

Read these through the WCAG bullet above: **WCAG 2 Level A is the hard CI gate (with the triaged theme-level rule-code ignores documented inline in `.github/workflows/ci.yml`); AA is advisory only.** The shift from "AA as gate" to "Level A gated, AA advisory" landed in commit `f370fe7` ahead of this amendment. The AA rule set is retained as a reporting target — re-promotion to a hard gate remains on the table once the triaged ignores can be retired.

### 5.1 — v0.2 release amendment (2026-04-21) *(Coastline, new in v0.3.3)*

**v0.2.0 shipped.** Every row in the §5.0 release-mapping table that was targeted at `v0.2` is now on `main`:

| §5.0 row | Delivered via |
|---|---|
| Phase 1 measurement layer (`measurement/`) | Dispatches H–P; §17 frozen |
| Phase 1 systematics layer (`systematics/`) | Dispatches R–U; §18 frozen |
| Phase 1 logarithmic negativity / EoF as registered observables | Dispatch Q (`entanglement.py`) |

`CONVENTIONS.md` freezes at version 0.2 at the same commit — §1–16 from the Phase 0 draft carry through unchanged; §17 (measurement layer) and §18 (systematics layer) are newly closed under Convention Freeze. Post-freeze additions to any section require a new CONVENTIONS.md version bump with an explicit freeze gate (§18 / §17 closing paragraphs).

**Consequence for §5 below.** The Phase 0 and Phase 1 section headers still read "target: v0.1-alpha" and "target: v0.2". Read them through both §5.0 and this amendment: the former collapsed `v0.1-alpha` into `v0.2.0`; this amendment records that `v0.2.0` is now tagged. Subsequent dispatches land under a new `[Unreleased]` block in `CHANGELOG.md` pending a v0.3 target.

**What's next on `main`.** Migration-scenario-3 coupling audit, migration-scenario-4 full-exponential Lamb–Dicke builder feature, and tutorial content are in the backlog but not v0.2-blocking. Phase 2 (performance — JAX backend, sparse-matrix tuning, parallel sweeps) is the next phase-level milestone, target `v0.3`.

### 5.2 — Post-v0.2.0 on-`main` amendment (2026-04-21) *(Coastline, new in v0.3.4)*

Added after `v0.2.0` was tagged (Dispatch W on 2026-04-21) to record subsequent work that has since landed on `main` under a fresh `[Unreleased]` block in `CHANGELOG.md`. No phase-level milestone is re-scoped here; the amendment exists so the §5.0 and §5.1 read-throughs stay honest against what is currently on `main`, and so the header status line plus the bottom Endorsement Marker can be updated in lock-step.

**Shipped on `main` after `v0.2.0` as of 2026-04-21:**

- **Migration-tier debt paydown** (Dispatch V, landed with `v0.2.0`). Migration-scenario 2 is active via the invariant tier (trace / Hermiticity / positivity checks on the reference trajectory). The §5.0 line *"scenarios 1 and 5 active (2/3/4 skipped)"* should be read as **scenarios 1, 2, 5 active; 3, 4 skipped** with the probe-informed blockers unchanged (coupling audit for scenario 3, full-exponential Lamb–Dicke builder for scenario 4).
- **Phase 2 opener** (Dispatches X, Y, Z). `sequences.solve` gains an `"auto"` / `"sesolve"` / `"mesolve"` dispatch with backend-name auto-tagging; `sequences.solve_ensemble` wraps `joblib.Parallel` for parallel sweeps (default `n_jobs=1` serial, `loky` wins once single-solve cost exceeds ~15 ms); `docs/benchmarks.md` records the `v0.2` performance baseline. Empirical finding: sesolve / mesolve parity on QuTiP 5.2 at library-scale Hilbert spaces (dim ≤ 48) — the folklore 2–3× advantage has closed, so measurable wins are deferred to later Phase 2 dispatches (sparse ops, JAX). Phase 2 remains the target for `v0.3`; these dispatches land Phase 2's opening deliverables against that target, not a re-scope.
- **Tutorial bundle** (Dispatches AA–LL, twelve entries). `docs/tutorials/01_first_rabi_readout.md` through `12_bell_entanglement.md` cover the public-surface pipeline end-to-end: carrier Rabi + Wilson CIs, red-sideband from |1⟩, Gaussian π-pulses, MS Bell gate, custom observables, Fock-truncation diagnosis, hash-verified cache round-trip, full Lamb–Dicke for hot-ion regimes, squeezed / coherent state preparation, finite-shot statistics, systematics jitter ensembles, two-ion Bell-state entanglement. This substantially discharges the Phase 3 deliverable *"≥5 worked examples; Clock-School tutorial bundle"* ahead of its `v0.5` target. Phase 3 is retained as the milestone for governance document, contribution guide with Boundary Decision Tree, and JOSS submission — the tutorial-count condition is effectively already met.

**Consequence for §5 below.** No re-scoping; §5.0 migration-tier line reads through to scenarios 1 / 2 / 5 active, §5.1 "what's next on `main`" reads through to Phase 2 opener and tutorial bundle now landed (with the three remaining non-v0.2-blocking items — scenarios 3 / 4, sparse / JAX performance — unchanged).

**Consequence for the Endorsement Marker.** The bottom Endorsement Marker is updated in the same commit as this amendment to bump the `CONVENTIONS.md` reference from v0.1 to v0.2 (frozen 2026-04-21 at the `v0.2.0` release, per §5.1) and to list §4.0, §5.0, §5.1, §5.2 in its workplan-version line.

### 5.3 — β.4 time-dependent Hamiltonian track as v0.3.x follow-up (2026-04-22) *(Coastline, new in v0.3.5)*

Added when Dispatch UU (β.4.1) landed on `main` —
`detuned_carrier_hamiltonian` gains a `backend=` kwarg; new
`src/iontrap_dynamics/backends/jax/_coefficients.py` module supplies
JAX-traceable coefficient factories. Records the scoping decision that the
full β.4 track — `detuned_{carrier, red_sideband, blue_sideband, ms_gate}_hamiltonian`
and `modulated_carrier_hamiltonian` extended to emit Dynamiqs
`TimeQArray` on `backend="jax"` — is a **v0.3.x follow-up**, not a
`v0.3` blocker.

**What the JAX backend covers at `v0.3`.** Time-independent Hamiltonians
across the four canonical families (carrier, RSB, BSB, MS), storage modes
OMITTED / EAGER / LAZY, Fock-saturation checks via exp_ops piggyback, and
cross-backend numeric equivalence on the carrier-Rabi exit criterion — per
Dispatches RR / RR.1 / SS / TT (β.1 skeleton, β.1 post-review tightening,
β.2 Dynamiqs integrator, β.3 LAZY storage). The time-dependent Hamiltonian
surface is extended incrementally under `v0.3.x` point releases.

**Rationale.** β.4 is additive — every affected builder keeps
`backend="qutip"` as the default, so existing callers observe no behaviour
change when a v0.3.x point release lands a new sub-dispatch. Shipping `v0.3`
with the time-independent half honest and complete, then extending under
`v0.3.x`, matches the workplan's semver commitment: `v0.3` closes the
Phase 2 milestone; point releases extend the deliverable surface without
re-opening the phase.

**On `main` toward β.4 at time of this amendment.** Dispatch UU = β.4.1
(`_coefficients.py` factory module + `detuned_carrier_hamiltonian`
`backend=` kwarg; cross-backend agreement 1.35e-5 at dim 8, 4 detuned-Rabi
periods, 1e-3 tolerance bound).

**Remaining β.4 sub-dispatches** (tracked for v0.3.x point releases, per
`docs/phase-2-jax-time-dep-design.md` §5 staging): β.4.2 detuned RSB / BSB,
β.4.3 detuned MS gate, β.4.4 `modulated_carrier_hamiltonian` with
user-supplied `envelope_jax`, β.4.5 cross-backend equivalence benchmark at
dim ≥ 100 / ≥ 5000 steps.

**Consequence for §5 above.** No re-scoping of Phase 2's target; `v0.3`
remains the Phase 2 milestone. The §5.2 "Phase 2 JAX backend — the
substantive remaining Phase 2 deliverable" bullet reads through this
amendment as: the time-independent portion satisfies `v0.3`; the
time-dependent portion is scheduled for `v0.3.x` points.

---

### Phase 0 — Foundations (target: v0.1-alpha, 4–6 weeks)

**Deliverable:** repository skeleton with conventions, regression harness, canonical result schema, corporate-design bootstrap, one end-to-end example.

**Exit criteria:**
- `CONVENTIONS.md` v0.1 complete and reviewed
- Legacy `qc.py` stability verified (three identical runs, identical outputs)
- Repository scaffolded under `open-iontrap/`, CI green
- Corporate design bootstrap complete (0.H below): assets copied, `SOURCE.md` with checksums committed, docs site renders with cd-rules tokens
- Accessibility smoke check: docs site passes WCAG 2.2 AA rules A1–A29 from CD 12A
- Result schema (`TrajectoryResult`) frozen and tested
- Three-layer regression harness operational: ≥5 migration references, ≥3 analytic limits, ≥5 invariant checks
- Cache-integrity test passes
- Performance smoke test defined and passing
- `examples/01_sideband_flopping/` runs on fresh install under 2 minutes
- Zenodo DOI reserved
- Per-folder LICENCE files in place; root LICENCE declares split

#### 0.A — `CONVENTIONS.md` v0.1 *(Coastline)*

Must specify unambiguously:

- **Units:** angular frequencies in 2π·MHz at interface, rad/s internally; times in μs at interface, SI internally; mass in kg SI throughout
- **Tensor ordering:** spins first, then modes, left-to-right by index
- **Spin basis ordering:** |↓⟩ = `basis(2,0)`, |↑⟩ = `basis(2,1)`; Pauli convention explicit
- **Detuning sign:** δ = ω_laser − ω_atom (positive δ = blue-detuned)
- **Hamiltonian form:** rotating frame of atomic transition; explicit interaction picture statement
- **Squeezing parameter:** z = r·exp(2iφ), QuTiP default, documented
- **Displacement parameter:** α = |α|·exp(iφ)
- **Spin rotation Euler convention:** extrinsic XYZ, active rotation
- **Bell state convention:** explicit |Φ±⟩, |Ψ±⟩ in terms of |↓⟩, |↑⟩. Legacy `qc.py` non-standard `(|dd⟩ + i|uu⟩)/√2` flagged and *not* adopted
- **Lamb–Dicke parameter:** full 3D form η_{i,m} = **k** · **b**_{i,m} · √(ℏ/2m·ω_m); no 1D shortcut
- **Normal-mode eigenvector normalisation:** Σᵢ |**b**_{i,m}|² = 1 across all ions for each mode
- **Trap frame:** right-handed, z-axis along trap symmetry for linear Paul traps; alternatives declared explicitly
- **Fock truncation convergence:** population in top state < ε; default ε = 10⁻⁴
- **Reproducibility layers:** as in Principle 3
- **Warnings and failure policy (three-level ladder):**
  - *Convergence warnings* — solver converged but slowly, top-Fock population between ε/10 and ε
  - *Numerical-quality degradation warnings* — non-convergence below tolerance but above threshold, recoverable parameter combinations
  - *Hard failures* — parameter-hash mismatch, unsupported backend feature, physics invariant violation beyond tolerance
  
  Warnings emitted to the result's `warnings` field and to the standard Python warnings channel; hard failures raise typed exceptions. Silent degradation is forbidden.

#### 0.B — Regression harness (three-layer) *(Coastline)*

**Migration regression** (Phase 0 only; retired after Phase 1):
1. Legacy-stability check: three identical `qc.py` runs → identical outputs. Gate for proceeding.
2. Freeze `qc.py` as pinned tag `qc-legacy-v1.0`
3. Canonical scenarios: single-ion carrier flopping (thermal); red-sideband flopping (Fock |1⟩); two-ion MS gate; single-ion stroboscopic AC-π/2; single-mode squeezing + displacement
4. References as `.npz` + JSON metadata in `tests/regression/migration/`
5. Comparison at tolerance 10⁻¹⁰ (same platform)

**Analytic regression** (permanent):
- Carrier Rabi closed-form vs numerical
- Vacuum-state red-sideband Rabi rate η·Ω
- Coherent-state displacement operator action
- Ideal MS gate Bell-state fidelity in Lamb–Dicke limit

**Invariant regression** (permanent):
- Trace preservation under dissipative evolution
- Hermiticity of density matrices
- Positivity within tolerance
- Norm conservation in closed-system evolution
- Symmetry checks where applicable

Migration is a *check*, not a *truth criterion*. Analytic and invariant tests are the permanent physics anchors.

#### 0.C — Repository scaffolding *(Coastline)*

- `pyproject.toml`, Python ≥ 3.11, QuTiP ≥ 5.0, NumPy, SciPy pinned
- `open-iontrap` org created; primary repo initialised there
- `.github/workflows/ci.yml`: ruff, mypy, pytest, mkdocs-material build, accessibility check, CD asset hash check
- `.github/workflows/release.yml`: tag-triggered sdist+wheel, Zenodo minting
- `pre-commit` hooks
- `CHANGELOG.md` (Keep-a-Changelog)

#### 0.D — Convention-enforcement tests *(Coastline)*

- All public functions type-annotated
- No hardcoded physical constants outside `conventions.py`, `species.py`
- All state-prep returns typed results with schema-validated metadata
- No `from qutip import *`
- Tensor-ordering test: 2-spin-1-mode dimension order matches CONVENTIONS.md
- Wigner-rotation test: active-Euler convention via known spin-coherent state rotation
- Lamb–Dicke projection test: 3D dot-product against analytic value for parallel and orthogonal **k**/**b**

#### 0.E — Result schema (`TrajectoryResult`) *(Coastline)*

Dataclass fields:
- `times`: np.ndarray
- `states`: list[Qobj] or lazy handle (see storage policy below)
- `expectations`: dict[str, np.ndarray]
- `metadata`: `ResultMetadata` (physical parameters, truncation choices, conventions version, backend name + version, request hash, provenance tags, warnings)

**State storage policy.** Public APIs declare whether state storage is *eager* (all states retained), *lazy* (accessed on demand), or *omitted* (expectation-only). Expectation-only trajectories are valid when explicitly requested. Downstream code must not assume both; consumers check the declared mode before accessing `states`.

Locked in Phase 0. Phase 1 builders consume it.

#### 0.F — Performance smoke test *(Coastline)*

Canonical laptop hardware (2023 MacBook Air M2 or equivalent):
- Single-ion sideband flopping (N_Fock=30, 200 steps): < 5 s
- Two-ion MS gate (N_Fock=15, 500 steps): < 30 s
- Stroboscopic AC-π/2 (N_Fock=40, 1000 steps): < 60 s

If breached, performance remediation reviewed immediately. Options include algorithmic optimisation, sparse-path refinement, targeted caching, or elevating JAX-backend work from Phase 2 to Phase 1. Integrator selects proportionate to root cause.

#### 0.G — First worked example: `01_sideband_flopping/` *(Coastline skeleton, Sail narrative)*

Minimal technical example (narrative in `docs/tutorials/`):
- `README.md` (CC BY-SA 4.0)
- `sideband_flopping.ipynb`: `IonSpecies` → `DriveConfig` → `ModeConfig` → `IonSystem` → `HilbertSpace` → state prep → Hamiltonian → sequence → observable → plot
- `parameters.yaml`
- `expected_outputs/`: frozen `.npz` + metadata
- `test_example.py`: headless run, reference comparison

Reference implementation of idiomatic library usage.

#### 0.H — Corporate design bootstrap *(Coastline, new in v0.3)*

1. Create `assets/` folder with per-folder `LICENCE` (MIT, as declared by cd-rules)
2. Copy the following from `threehouse-plus-ec/cd-rules@cd-v1.7.0`: `emblem-16.svg`, `emblem-32.svg`, `emblem-64.svg`, `tokens.css`, `wordmark-full.svg`, `wordmark-silent.svg`
3. Compute SHA-256 for each file; record source commit + hashes in `assets/SOURCE.md`
4. Configure `docs/` (mkdocs-material) to import `tokens.css` and the canonical Google Fonts URL from CD 6.1
5. Verify docs site against WCAG 2.2 AA rules A1–A29 (CD 12A) — this is a CI gate
6. Verify colour tokens applied correctly (K1–K8 from CD 5.3 / 5.4)
7. Verify typography rules (T1–T9 from CD 6.2 / 6.3)
8. Add root `LICENCE` file declaring split architecture with pointers to per-folder licences
9. Add Endorsement Marker to top of README, workplan, `CONVENTIONS.md`, and every docs page (CD 0.7)
10. CI job: periodic hash check of `assets/` against declared cd-rules tag; fail on silent drift

---

### Phase 1 — Core physics and measurement layer (target: v0.2, 8–10 weeks)

> **§5.0 amendment applies.** The *dynamics core* deliverable below has
> already shipped on `main` and is part of the `v0.1-alpha` cut (see the
> shipped-on-`main` list in §5.0). The `v0.2` release now covers only the
> measurement layer, the systematics layer, and the remaining registered
> observables (logarithmic negativity, EoF, any as-yet-unlanded coupled-
> ion normal-mode decomposition helpers).

**Dynamics core:** two-spin / two-mode systems, MS gate, parametric modulation, stroboscopic AC drives, coupled-ion normal-mode decomposition, logarithmic negativity / concurrence / EoF as registered observables.

**Measurement layer (`measurement/`):** channels (Bernoulli, binomial, Poisson), protocols (spin readout, parity scan, sideband-flopping inference), detectors (efficiency, dark counts, thresholding), statistics (estimators, CI).

**Systematics layer (`systematics/`):** drifts, jitter, SPAM.

**Result dual-view:** `MeasurementResult` exposes `ideal_outcome` and `sampled_outcome`.

**PR discipline:** one Hamiltonian or measurement channel per PR with tests and worked example.

---

### Phase 2 — Performance (target: v0.3, 4–6 weeks)

Sparse-matrix tuning; `sesolve` path; parallel sweeps via `joblib`; JAX backend with ≥1 worked example matching QuTiP within cross-platform tolerance; benchmarks in `docs/benchmarks.md`.

---

### Phase 3 — Community release (target: v0.5)

Governance document; contribution guide with Boundary Decision Tree; ≥5 worked examples; Clock-School tutorial bundle; JOSS submission.

---

### Phase 4 — Version 1.0

**Technical gate (blocking):** API stable for two minor versions; coverage ≥ 85%; breaking-change policy frozen; three regression layers fully populated.

**Prestige milestone (non-blocking):** external labs using library for published work.

---

## 6. Risks and mitigations *(Sail)*

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Scope creep into trap simulation | High | High | Non-scope list + Boundary Decision Tree + PR template |
| `qc.py` regression fails silently | Medium | Very high | 0.B legacy-stability check + three-layer harness |
| Convention ambiguity | Medium | High | Staged `CONVENTIONS.md`; enforcement tests in 0.D |
| QuTiP 5 API instability | Low | Medium | Pin versions; richer backend contract |
| Low external adoption | High | Medium | Clock-School positioning; neutral org signals durability |
| Performance below laptop threshold | Medium | High | 0.F smoke test; remediation reviewed |
| Maintainer bandwidth (single person) | High | High | MIT + neutral org + CITATION.cff = survives bus factor |
| Digital-twin scope creep | Medium | High | Explicit non-scope clause on electrode/control-stack modelling |
| Measurement layer entangles with Hamiltonian | Medium | High | Three-layer architecture enforced by module boundaries |
| Result-schema drift in Phase 1 | Low | High | Schema locked in 0.E before any builder written |
| **cd-rules drift** (local assets lag upstream) | Medium | Medium | `assets/SOURCE.md` + CI hash check; update within one release cycle per CD 0.10 |
| **Licence confusion** across split layers | Medium | Medium | Per-folder LICENCE files; root LICENCE declares split; licence matrix in §4 |
| **Accessibility regression** on docs site | Low | Medium | WCAG 2.2 AA rules A1–A29 as CI gate per CD 12A |

---

## 7. Open questions adjudicated *(Coastline)*

v0.1 left five for Council. All resolved:

1. **Repository name:** `iontrap-dynamics`.
2. **GitHub organisation:** Two-repo model. Primary under `open-iontrap/` (neutral community posture). Design authority under `threehouse-plus-ec/cd-rules` (consumed via Model B).
3. **Relationship to `single-25Mg-plus`:** Library is dependency; twin is configuration consumer.
4. **Pedagogical scope in v0.1:** Minimal technical example in `examples/`; guided physics in `docs/tutorials/`.
5. **Pre-announcement:** Silent development through Phase 0.

---

## 8. Immediate next actions (Week 1) *(Coastline)*

1. Create `open-iontrap` GitHub organisation
2. Tag legacy `qc.py` as `qc-legacy-v1.0` in current repository
3. Run legacy-stability check: three identical runs, verify identical outputs
4. Draft `CONVENTIONS.md` v0.1 — primary Week 1 writing task, no code
5. Scaffold empty repository with `pyproject.toml`, CI stubs, per-folder LICENCEs per §4 split
6. Execute Phase 0.H corporate-design bootstrap: copy cd-rules assets, write `assets/SOURCE.md` with commit hash + SHA-256 checksums, configure docs site against `tokens.css` and CD typography
7. Generate five migration-regression reference `.npz` files from legacy tag

Only then does implementation of `IonSpecies`, `DriveConfig`, `ModeConfig`, `IonSystem` begin.

---

## Appendix A — Changes from v0.1

(Retained from v0.2.1 for internal continuity; to be retired before public release.)

| Section | v0.1 | v0.2.1 |
|---|---|---|
| Scope | Dynamics only | Dynamics + measurement + systematics, three-layer architecture |
| Identity statement | Absent | Added |
| Boundary Decision Tree | Absent | Added with apparatus-scope clarification |
| Design principles | 8 | 11 |
| Reproducibility | "Byte-reproducible" | Stratified |
| Architecture | Single `system.py` | Four configuration modules: `species.py`, `drives.py`, `modes.py`, `conventions.py` |
| Lamb–Dicke | Implicit 3D | Explicit 3D-native, no 1D shortcut |
| Backend contract | Minimal | Richer |
| Regression | Single layer | Three layers |
| Archive format | Pickle | `.npz` + manifest |
| Result schema | Implied | Explicit `TrajectoryResult` with storage policy |
| Measurement / systematics layer | Absent | Phase 1 deliverable |
| Performance smoke test | Absent | Phase 0.F |
| Warnings policy | Absent | Three-level ladder |
| Convention staging | Single freeze | Staged gates |
| GitHub org | Open question | Two-repo model |
| v1.0 gate | Adoption-blocked | Technical-gated |

## Appendix B — Changes from v0.2.1 (CD integration)

| Section | v0.2.1 | v0.3 |
|---|---|---|
| Classification | Implicit | Explicit: Coastline per CD 0.9 |
| Licence architecture | "MIT for code, CC BY 4.0 for documentation" | Split architecture per CD 0.3: CC BY-SA 4.0 for coastlines, CC BY-NC-SA 4.0 for Sail, MIT for code + assets |
| Design principles | 11 | 15 (added: CD compliance, Coastline/Sail labelling, no filename versioning, deprecation not deletion) |
| Architecture skeleton | Core layout | Added: root `LICENCE` (split), `assets/` with `LICENCE` + `SOURCE.md`, `archive/` for deprecation |
| Asset propagation | Absent | Model B (distributed copy + SHA-256 checksum) per CD 0.10 |
| GitHub topology | Single neutral org | Two-repo model with design authority upstream |
| Phase 0 | 0.A–0.G | Added 0.H: corporate-design bootstrap |
| CI gates | Tests + docs | Added: accessibility (WCAG 2.2 AA rules A1–A29) + CD asset hash check |
| Risks | 10 entries | 13 entries (added: cd-rules drift, licence confusion, accessibility regression) |
| Endorsement marker | Footer only | Top + footer per CD 0.7 |
| Section labelling | Absent | Coastline / Sail labels throughout |
| Typography and visual identity | Undefined | Flows from cd-rules `tokens.css`, IBM Plex Mono + Crimson Pro |

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This workplan is a Coastline draft within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg) pending external contributor onboarding from v0.3. Lock–Key rule applies: this document specifies the stable locks of the library's architecture; individual analyses built on top are keys. The repository adopts T(h)reehouse +EC Corporate Design blueprint (`cd-rules`, consumed via Model B).

**Council status:** Guardian cleared (scope honest, split licence protects downstream). Architect approved (configuration-layer architecture, Lamb–Dicke 3D-native, richer backend contract, three-layer physics/apparatus/observation separation, CD structural compliance). Scout horizon signals addressed (two-repo topology resolves bus-factor; Model B prevents design drift). Integrator has sequenced Phase 0: conventions → legacy-stability → schema → scaffolding → regression → example → corporate-design bootstrap.

**Convention version:** references `CONVENTIONS.md` v0.2 (frozen 2026-04-21 at the `v0.2.0` release, closing §17 measurement layer and §18 systematics layer; §1–16 carry through unchanged from the v0.1 draft).
**Corporate design version:** **PROVISIONAL — no upstream tag yet** (decision D2). `cd-rules v1.7.0` is the intended target named throughout this document, but `threehouse-plus-ec/cd-rules` had no tagged releases at Phase 0 commencement (2026-04-17) and still does not today. Assets are pinned to a specific commit hash documented in [`assets/SOURCE.md`](assets/SOURCE.md), which carries the authoritative PROVISIONAL banner. Before any v0.1-alpha release the upstream first-tag action must complete, `SOURCE.md` must re-pin to the tagged commit and drop its banner, and only then does the CI hash-drift check activate as a permanent gate. Until then, every reference to `cd-v1.7.0` in this workplan reads as "the CD blueprint we will consume once tagged"; in-flight work continues against the pinned commit per D2.
**Workplan version:** 0.3.5 (amended §4.0 repo-hosting, §5.0 release-mapping 2026-04-19, §5.1 v0.2 release 2026-04-21, §5.2 post-v0.2 on-`main` 2026-04-21, §5.3 β.4 as v0.3.x follow-up 2026-04-22) · `v0.2.0` tagged 2026-04-21 covering Phase 0 foundations plus the full Phase 1 deliverable (dynamics core, measurement layer, systematics layer, registered entanglement observables); Phase 2 JAX-backend time-independent surface and twelve tutorials on `main` under `[Unreleased]`; β.4 time-dependent extension scoped as v0.3.x follow-up per §5.3.
