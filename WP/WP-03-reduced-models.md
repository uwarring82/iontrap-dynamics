# WP-03 — Reduced Light–Matter Models & Full-Ion Comparison Tutorial

**Executes the reduced-models task card: add abstract JC/AJC/QRM Hamiltonian builders, a model-vs-realisation comparison harness, and Tutorial 18 comparing reduced light–matter models against full trapped-ion dynamics.**

Version 0.2 · Drafted 2026-06-04 · **Ratified 2026-06-04** · Status: Ratified

**Classification:** Sail execution under Coastline gates (per T(h)reehouse +EC CD 0.9).
**Licence:** This WP document is CC BY-SA 4.0 (`WP/LICENCE`). Deliverables carry their layer's licence: code is MIT (`src/`, `tests/`, `.github/workflows/`); authored docs/tutorials are Sail / CC BY-NC-SA 4.0; the vendored hierarchy note and any `CONVENTIONS.md` edit are Coastline / CC BY-SA 4.0. See root `LICENCE`.
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (`cd-rules`, consumed via Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

---

## 0. Ratification decisions (2026-06-04) *(Coastline)*

Seven decisions resolved at ratification. They bind execution; the cross-referenced sections implement each one.

| # | Decision | Resolution | Binds |
|---|---|---|---|
| **R1** | Slug, branch & cadence | Slug `reduced-models`; execution branch `wp03-reduced-models` off `main`; one branch per WI lift with the WP-02 understand→implement→adversarial-verify cadence and a review pause per WI. | §4 |
| **R2** | **Conventions gate (binding)** | Land an additive **§25 "Reduced light–matter models"** *and* a **§5 scoping note** that re-scopes §5's interaction-picture mandate to builders *derived from an atomic transition* (the drive/apparatus builders, whose free atomic term §5 transforms away) — pure-motional §23/§24 have no atomic transition to transform — and defers reduced-model frames to §25. `CONVENTION_VERSION` **0.3 → 0.4**. Both are governed edits: **staged as a propose-don't-apply proposal, sealed by the maintainer in a single commit *before* WI-2 code** (conventions-before-code). Resolves the §5 ⊥ §25 contradiction the card surfaced. | §6, WI-1 |
| **R3** | Dispatch family | **`RL`** (RLA…RLG), minted in §4/§11. Collision grep over `CHANGELOG.md`, `WORKPLAN_v0.3.md`, `WP/LOGBOOK.md`, `docs/gpu-dispatch-design.md`, `src/`: GPU families AA–LL and WP-01/WP-02 families ED/MC are taken; `RL` is clear and is deliberately distinct from the card's RM0–RM6 *feature* labels. | §4, §11 |
| **R4** | Units | Reduced builders take **physical SI** angular frequencies/couplings (rad·s⁻¹), consistent with §1. **Tutorial 18 presents dimensionless ratios** (e.g. `g/ω0`) for pedagogy — a presentation choice, not a builder API. Recorded in §25. | WI-2, WI-6 |
| **R5** | `model_deviation` placement | Lands **in `reduced_models.py`** alongside the builders — and only if the deviation summary is not already expressible via existing observables / `qutip.fidelity`. Materialised-state requirement enforced; observable-RMS fallback explicit. Keeps the new surface to a single module. | WI-5 |
| **R6** | Case E | **Deferred** to a future WP behind a first-class two-tone sideband builder + effective-parameter convention. Tutorial 18 shows only the schematic deferred interaction (P2 boundary). | §1, WI-6 |
| **R7** | WI-4 external dependency | RM5 hierarchy-note vendoring carries an **external dependency**: `hierarchy.md` v0.4 + its source commit hash/DOI must be supplied before WI-4 lands. Gates WI-4 only; does **not** block WI-1/WI-2/WI-3. **Status 2026-06-04:** v0.4 note supplied → WI-4 (RLD) landed; the source commit/DOI is **still pending** (upstream note is a lock candidate), so R7 remains **partially open** (provenance-pending) and closes only when that field is recorded in `docs/models-hierarchy.md`. | WI-4 |

---

## 1. Purpose and the physics/apparatus boundary invariant *(Coastline)*

WP-03 executes the reduced-models tutorial card by adding **application-agnostic** library primitives for abstract qubit–oscillator models and by using the already-shipped trapped-ion sideband/full-Lamb-Dicke surface to compare those reduced models against physical realisations.

**The governing boundary invariant — quoted from the card §1, not paraphrased** (hard acceptance gate):

> **One boundary must hold (and it mirrors the library's architecture):** the *reduced models* are physics-layer objects (what the apparatus approximates); the *sideband Hamiltonians* are apparatus-layer objects (how a real ion realises them). The tutorial's whole point is to compare the two — Axis A (model containment) against Axis B (physical realisation) in the language of `hierarchy.md`. Keeping that separation is what makes the tutorial a *falsifiable demonstration* of the note rather than an illustration of it.

Concretely:

- reduced JC/AJC/QRM builders live in a new physics-layer module (`reduced_models.py`) and are not tied to a drive, species, wave-vector, or sideband selection;
- red/blue sideband and full-Lamb-Dicke builders remain apparatus-layer objects in `hamiltonians.py`;
- the comparison harness and Tutorial 18 make the mapping explicit instead of hiding it in a builder name;
- Case E (bichromatic simulated QRM) is **deferred** until a separate two-tone sideband convention/builder is scoped.

## 2. Reuse-first posture *(Coastline)*

The task card §3 lists the existing machinery that WP-03 must reuse, not rebuild:

| Already shipped (do not rebuild) | WP-03 builds on it |
|---|---|
| `operators.{sigma_z_ion, sigma_x_ion, sigma_plus_ion, sigma_minus_ion, spin_up, spin_down}` | reduced-model terms and LOCK-3 conventions test |
| `HilbertSpace` mode/spin embedding helpers | all reduced builders (`mode_label`, keyword-only `ion_index`) |
| `hamiltonians.{red_sideband_hamiltonian, blue_sideband_hamiltonian}` with `full_lamb_dicke=True` on red/blue sidebands | Cases B and D; no apparatus rebuild |
| `analytic.{red_sideband_rabi_frequency, blue_sideband_rabi_frequency, red_sideband_rabi_frequency_full_ld, blue_sideband_rabi_frequency_full_ld, debye_waller_factor, lamb_dicke_confinement, lamb_dicke_regime}` | analytic oracles and Tutorial 18 formulas |
| `solve(...)` / `solve_ensemble` with QuTiP default and `backend="jax"` | trajectory comparisons and backend parity |
| `spectrum.solve_spectrum` | Case A spectrum equality; Case C QRM ground-state and spectral deviation |
| tutorial/benchmark house style (`docs/tutorials/`, `tools/run_*`, `benchmarks/data/`) | Tutorial 18 and reproducible figure artefacts |

**New module expected:** `src/iontrap_dynamics/reduced_models.py`. Everything else extends existing surfaces or documentation.

## 3. Card linkage *(Sail)*

Executes **`task cards/TC-reduced-models-tutorial.md`** (**ID: TC-reduced-models-tutorial**, v0.2.1, 2026-06-04).

**Objective lifted from the card (one line):** deliver reduced JC/AJC/QRM physics-layer builders, a model-vs-realisation comparison harness, and Tutorial 18 with reproducible figures and oracle-backed comparisons against full trapped-ion dynamics.

**Governing gates from the card — quoted, not paraphrased**:

> **Oracle-first.** No case "passes" on a plot; each pass condition above is a runnable oracle check.

> **Conventions before code (RM0).** `CONVENTIONS.md` §25 and its conventions test land *before* the reduced-model builders.

> **Cross-backend.** Reduced-model solves agree QuTiP vs JAX under 1e-3 where a JAX path exists; builders themselves remain static QuTiP `Qobj` producers.

> **No claim beyond the note.** The tutorial introduces no physics statement absent from the locked `hierarchy.md`; it cites rungs.

**Out of scope (explicit).** Case E's bichromatic simulated-QRM bridge is deferred to a future WP unless a first-class two-tone sideband builder and effective-parameter convention land first. The WP does not redefine sideband builders, does not introduce consuming-application framing, and does not make hardware claims.

## 4. Work-item plan WI-1…WI-7 *(Sail)*

Dispatch codes are minted at Ratified (2026-06-04, R3) as family **`RL`** after the authoritative collision grep of `CHANGELOG.md`, `WORKPLAN_v0.3.md`, `docs/gpu-dispatch-design.md`, `WP/LOGBOOK.md`, and `src/` (GPU families AA–LL and WP-01/WP-02 families ED/MC taken; `RL` clean and distinct from the card's RM0–RM6 feature labels).

| WI | Dispatch | Card | Module / doc | Key contents | Reuse | Acceptance (oracle) | Conv. | Priority |
|---|---|---|---|---|---|---|---|---|
| **WI-1** | `RLA` | RM0 | `CONVENTIONS.md` §25 + §5 scoping note + `tests/conventions/test_reduced_models_conventions.py` | Reduced light–matter model convention: Schrödinger-picture bare terms, `H/ℏ` in rad·s⁻¹, `ω0` sign semantics, JC/AJC/QRM term selection, LOCK-3 identity; §5 scoping note re-scopes the interaction-picture mandate to builders derived from an atomic transition | §3 Pauli convention; §5 Hamiltonian picture contrast | LOCK-3 identity + symmetry/magnitude anchors hold on embedded operators (runnable, green now); `CONVENTION_VERSION` 0.3→0.4 bump + §25 text + §5 note present **at seal** | **new §25 + §5 note, staged, sealed before code (R2)** | P0 |
| **WI-2** | `RLB` | RM1 | `src/iontrap_dynamics/reduced_models.py` + exports | `jaynes_cummings_hamiltonian`, `anti_jaynes_cummings_hamiltonian`, `quantum_rabi_hamiltonian`; static `Qobj` builders with explicit `mode_label`, keyword-only `ion_index`; physical-SI inputs (R4); no builder-level `backend=` | `HilbertSpace`, `operators`, `solve`, `solve_spectrum` | Hermiticity/dims/API rejection; JC/AJC Rabi block rates; QRM weak-coupling reference; QuTiP/JAX solve parity < 1e-3 | per WI-1 | P0 |
| **WI-3** | `RLC` | RM6 core | `tests/regression/analytic/` references (benchmark deviation-curve artefacts → RLF) | Analytic oracles for Cases A–D, including `2g√(n±1)` sideband/RM coupling relation | existing `analytic.py`, `spectrum.py`, `sequences.py` | Case A identity/spectrum equality; Case C ground-state photon and reference-control points; Case D full-LD vs leading-order control points | none new | P0/P1 |
| **WI-4** † | `RLD` | RM5 | `docs/models-hierarchy.md` + `mkdocs.yml` | Vendor `ajc-provenance/docs/hierarchy.md` v0.4 with provenance header, commit hash/DOI slot, licence preserved | task-card source note; mkdocs nav pattern | links resolve; note renders; provenance recorded; tutorial can cite sections | none new (external dep, R7) | P1 |
| **WI-5** | `RLE` | RM2 | `reduced_models.py` (R5) | `model_deviation(...)` if needed: pinned `1 - qutip.fidelity(...)` convention for materialised states, observable/population RMS fallback | `TrajectoryResult`, `StorageMode`, observables, `qutip.fidelity` | deviation → 0 in common regimes; materialised-state requirement enforced; observable-only fallback explicit | none new | P1 |
| **WI-6** | `RLF` | RM3/RM4 | `docs/tutorials/18_reduced_models_vs_full_dynamics.md`; `tools/plot_reduced_models_comparison.py`; `benchmarks/data/` | Tutorial 18 walking Cases A–D; analytical-expression block from task card; dimensionless-ratio presentation (R4); deterministic figures for A–D with arrays/report metadata | tutorial/benchmark house style; docs nav | snippets execute; figures regenerate and are cited; pa11y WCAG A; no claim beyond hierarchy note | none new | P1/P2 |
| **WI-7** | `RLG` | release hygiene | `CHANGELOG.md`, `WP/LOGBOOK.md`, release docs | Dispatch-keyed changelog bullets, logbook decisions/deferrals, final WP status/release notes | existing WP/CHANGELOG pattern | all landed WI entries accounted for; CI green; release SemVer justification recorded if tagged | none new | P1 |

> **†  WI-4 (RLD) external dependency (R7).** Blocked by an external input: the vendored `hierarchy.md` v0.4 note *and* its source commit hash / DOI must be supplied before RLD lands. This gates **WI-4 only** — WI-1/WI-2/WI-3 proceed without it — and is why §25 forward-references the note rather than linking it live (§6).

**Sequencing.** WI-1 (RLA) is a hard gate for WI-2 (RLB) — conventions before code. After WI-2, WI-3/WI-4/WI-5 (RLC/RLD/RLE) parallelise; WI-6 (RLF) consumes them; WI-7 (RLG) closes. RLD's external dependency (above) is off the WI-1→WI-3 critical path.

## 5. Analytical-expression requirements for Tutorial 18 *(Sail)*

Tutorial 18 must display, at minimum, the task card's compact equations:

- QRM, JC, and AJC Hamiltonians and their symmetry contrast (`Z2` parity vs `U(1)` excitation-like numbers);
- the LOCK-3 identity `H_AJC(ω0)=σx H_JC(-ω0) σx`, with the negative-frequency caveat;
- the schematic full-ion Hamiltonian and first-order Lamb-Dicke red/blue sideband terms that explain red→JC and blue→AJC;
- the full-LD Debye–Waller / Laguerre sideband expression and its Lamb-Dicke limit `Ω|η|√(n±1)=2g√(n±1)`;
- the regime parameter `η²(2n+1)` / `η²(2 n̄+1)`;
- only a **schematic** deferred bichromatic retained interaction, with no committed effective-frequency map until the future Case-E convention lands.

This section is an execution checklist, not new physics; the authoritative conceptual source remains the vendored hierarchy note.

## 6. Convention plan *(Coastline)*

`CONVENTIONS.md` is frozen at v0.3 (§1–24). WP-03 requires a **two-part governed amendment** carried through `CONVENTION_VERSION` **0.3 → 0.4** **before WI-2** (R2). The reason is structural and was surfaced by the card: current **§5 declares *all* builders to be in the interaction picture** of the atomic transition (atomic term removed, RWA by default), while reduced JC/AJC/QRM builders intentionally carry **Schrödinger-picture bare `½ω0σz` terms** (and the QRM is non-RWA by construction). Left unamended, §5 would contradict §25 — so §5 must be *scoped*, not merely supplemented.

Planned amendment (both parts land together, staged as a single propose-don't-apply proposal, sealed by the maintainer before WI-2):

| Part | Edit | Content | Required before |
|---|---|---|---|
| **§5 scoping note** | short re-scope of §5's opening scope sentence | Re-scopes §5 from "All builders" to builders **derived from an atomic transition** (the drive/apparatus builders, whose free atomic term §5 transforms away); pure-motional objects (the §23 two-mode-squeezing builder, the §24 motional channels) lie outside by construction; reduced models in a different picture/RWA regime are governed by §25. **Canonical to-paste text: `WP/RL-conventions-proposal.md` §A.1.** | WI-2 |
| **new `## 25. Reduced light–matter models`** | additive section | Schrödinger-picture reduced-model Hamiltonians; physical-SI inputs in rad·s⁻¹ (R4); `ω0` sign semantics; JC/AJC/QRM term selection; LOCK-3 identity `H_AJC(ω0)=σx H_JC(−ω0)σx`; explicit non-RWA QRM caveat | WI-2 |

**Why the §5 note re-scopes the *predicate* (not a §23/§24 carve-out).** Grounding the wording against the live `CONVENTIONS.md` (5-agent verification, incl. an adversarial refuter) confirmed §5 literally opens "All builders return Hamiltonians in the interaction picture of the atomic transition" — genuinely overbroad — but **§25 is the *only* frame-specialising section**. §23/§24 are *not* counterexamples: §23's time-independent `iℏg(â†b̂† − âb̂)` form is already the rotating-frame/interaction-picture object (it fixes a *parameter* convention, `z = −gτe^{iφ}`, per-mode `sinh²|z|`, not a frame), and §24 is a dissipation-layer convention. They fall outside §5 only because, being pure-motional, they have **no atomic transition to transform away** — so the fix narrows §5's predicate from "all builders" to "builders derived from an atomic transition," which closes the gap without misattributing §23 as a frame departure.

The §25 text cites `CONVENTIONS.md` §3/§5 and **forward-references** the model-hierarchy companion note (vendored later under Dispatch RLD, §4) rather than linking it live — so §25 carries no dead internal link during the WI-1 → WI-4 interval. The conventions test (`tests/conventions/test_reduced_models_conventions.py`) is the first runnable gate, and it is **behavioural and green now**: the LOCK-3 identity on the embedded library operators, the U(1)/Z₂ symmetry contrast, and matrix-element anchors pinning the absolute ½/ω_f/g coefficients. It deliberately does **not** assert the `CONVENTION_VERSION` value or §25/§5 markdown presence — per the suite's house style those are validated at **seal time** (the `CONVENTION_VERSION` 0.3→0.4 pin lands in `test_convention_version.py` in the seal commit), not by this test.

**Propose-don't-apply.** Per the standing governance rule, this WP does **not** edit `CONVENTIONS.md` at ratification. WI-1 (RLA) stages the §25 + §5-note text as the **actual `CONVENTIONS.md` patch** (the exact diff on the branch, so review is grounded in text, not intent); the maintainer seals it (one commit, `CONVENTION_VERSION` 0.3→0.4) before any WI-2 code lands. Whether the seal carries a `FREEZE`-style side-car or amends in place is the maintainer's call; a single additive section + one re-scoped sentence is lighter than the v0.3 §19–24 batch and likely needs none.

## 7. Benchmark and test plan *(Coastline)*

| Case | Tool / test | Oracle |
|---|---|---|
| A | unit + conventions test; spectrum check | `H_AJC(+ω0) = σx H_JC(-ω0) σx`; eigenvalue sets equal |
| B | tutorial snippet + figure data | red sideband dark from `|↓,0⟩`; blue sideband bright with `blue_sideband_rabi_frequency` |
| C | spectrum/trajectory reference curve | weak-coupling JC≈QRM; QRM ground-state `⟨a†a⟩`; reference control points, not monotonicity |
| D | analytic + trajectory control points | full-LD sideband rates vs leading order; `2g√(n±1)` reduced limit; regime bands via `η²(2n+1)` |

Every benchmark artefact follows the existing compute/report style (`arrays.npz`, `report.json`, `plot.png` with alt text) where a figure is generated.

## 8. CHANGELOG + release plan *(Coastline gate)*

Target release: additive minor or patch depending on the surrounding unreleased surface. The current task card suggests an additive minor release theme: **reduced light–matter models + model-vs-realisation tutorial**.

Each landed dispatch gets a dispatch-keyed `[Unreleased]` bullet. A release cut follows the WP template's five-step procedure and records the explicit SemVer decision in `WP/LOGBOOK.md`.

## 9. Definition of Done *(Coastline)*

- [ ] `CONVENTIONS.md` §25 and `CONVENTION_VERSION` bump landed before reduced builders.
- [ ] `reduced_models.py` builders are static `Qobj` producers, explicit on `mode_label` and keyword-only `ion_index`, with no builder-level `backend=`.
- [ ] Cases A–D each have a runnable oracle; no plot-only acceptance.
- [ ] Case C/D reference bands/control points are committed before benchmark tests depend on them.
- [ ] Tutorial 18 includes the analytical-expression block and cites the vendored hierarchy note.
- [ ] Case E remains deferred unless a separate two-tone convention/builder lands first.
- [ ] SPDX headers on new code, docs licences preserved, CHANGELOG/logbook entries present, CI green.

## 10. Risks and mitigations *(Sail)*

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Reduced-model builder silently blurs physics and apparatus layers | Medium | High | §1 invariant; separate `reduced_models.py`; sidebands remain in `hamiltonians.py` |
| §25 convention delayed but code begins | Medium | High | WI-1 is first and blocks WI-2; conventions test required |
| Monotonicity accidentally reintroduced in Case C/D tests | Medium | Medium | use pre-committed control points/reference bands only |
| Fidelity comparison uses omitted states | Medium | Medium | RM2 helper requires materialised states or falls back to observable RMS |
| Case E scope creep blocks Tutorial 18 | Medium | High | explicit deferred status and future-WP prerequisite |
| Full-LD signed/magnitude convention confused in tutorial | Low | Medium | tutorial text must state signed matrix element vs helper magnitude |

## 11. Dispatch register *(Sail — family `RL` minted at Ratified)*

Family **`RL`** minted 2026-06-04 (R3) after the authoritative collision grep; mirrored into `WP/LOGBOOK.md`'s dispatch registry. Codes are atomic per WI; each lands one `[Unreleased]` CHANGELOG bullet.

| Dispatch | Maps to | CHANGELOG bullet | Status |
|---|---|---|---|
| `RLA` | WI-1 §25 convention + §5 scoping note + conventions test | `- **Dispatch RLA — conventions: reduced light–matter models (§25 + §5 scope, CONVENTION_VERSION 0.4).**` | **SEALED 2026-06-04** (`CONVENTIONS.md` §25 + §5 scope; `CONVENTION_VERSION` 0.4; test green) |
| `RLB` | WI-2 reduced-model builders | `- **Dispatch RLB — reduced_models: JC/AJC/QRM Hamiltonian builders.**` | **landed 2026-06-04** (`reduced_models.py` + package re-exports; 64 unit tests; mypy --strict clean; builders == §25 reference) |
| `RLC` | WI-3 analytic oracles | `- **Dispatch RLC — tests: reduced-model oracle suite (Cases A–D).**` | **landed 2026-06-04** (`test_reduced_models_oracles.py`, 36 oracle tests; Case A spectrum + JC dressed form, Case B red-dark/blue-bright, Case C QRM PT + committed bands, Case D 2g√(n±1) bridge + LD classifier) |
| `RLD` | WI-4 hierarchy note vendoring | `- **Dispatch RLD — docs: model hierarchy companion (vendored).**` | **landed 2026-06-04** (`docs/models-hierarchy.md` vendors `hierarchy.md` v0.4; provenance header + CC BY-SA 4.0 preserved; transit-encoding repaired, source-repo links neutralised; `mkdocs.yml` nav; Tutorial 18 §4/§5/§6/§8 refs now live links; mkdocs --strict clean. R7 note supplied; source commit/DOI pending upstream lock) |
| `RLE` | WI-5 comparison helper | `- **Dispatch RLE — reduced_models: model-deviation helper.**` | **landed 2026-06-04** (`model_deviation` + `ModelDeviation` in `reduced_models.py` + re-export; state-fidelity + observable-RMS paths; 22 unit tests; mypy --strict clean) |
| `RLF` | WI-6 Tutorial 18 + figures | `- **Dispatch RLF — tutorials: reduced models versus full dynamics.**` | **landed 2026-06-04** (Tutorial 18 Cases A–D, snippets execute; `tools/plot_reduced_models_comparison.py` + `benchmarks/data/reduced_models_comparison/` figures, oracle < 1e-3; nav + index; mkdocs --strict clean) |
| `RLG` | WI-7 release hygiene | `- **Dispatch RLG — release hygiene for reduced-model tutorial track.**` | planned |

## 12. Logbook hooks *(Sail)*

Entries this WP has generated in `WP/LOGBOOK.md` (dated):

- 2026-06-04 — WP-03 Drafted against `TC-reduced-models-tutorial`; task card staged as v0.2.1; dispatch codes unminted pending ratification.
- 2026-06-04 — WP-03 **Ratified** (v0.2). Decisions R1–R7 recorded (§0); dispatch family **`RL`** (RLA…RLG) minted after collision grep and registered; binding conventions gate fixed as §25 + §5 scoping note at `CONVENTION_VERSION` 0.4, staged propose-don't-apply, sealed before WI-2.

Expected future hooks: one per decision/deferral during execution, one at the §25/§5 convention seal (R2), one at release cut.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This Work-Plan is a Sail execution document within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg), under the Coastline gates of `WORKPLAN_v0.3.md` and `CONVENTIONS.md`. Lock–Key rule applies: this WP is a key built on the stable locks those documents specify. The repository adopts the T(h)reehouse +EC Corporate Design blueprint (`cd-rules`, consumed via Model B).

**Council status (at Ratified):** Guardian cleared — the conventions gate is routed *before* code as a staged, propose-don't-apply §25 + §5 scoping note (R2), and no Coastline gate is relaxed. Architect cleared — `reduced_models.py` is a single physics-layer module reusing `HilbertSpace`/operators/solvers; sidebands stay apparatus-layer in `hamiltonians.py` (§1 invariant). Scout horizon — Case E two-tone convention (R6) and the RM5 hierarchy-note provenance/external dependency (R7) remain on the horizon, not on the critical path. Integrator sequenced — WI-1 (RLA) → WI-2 (RLB) → WI-3/WI-4/WI-5 (RLC/RLD/RLE) → WI-6 (RLF) → WI-7 (RLG).

**Convention version:** references `CONVENTIONS.md` v0.3 (frozen 2026-06-03). This WP introduces a new §25 and a §5 scoping note through WI-1, bumping `CONVENTION_VERSION` **0.3 → 0.4**, staged propose-don't-apply and sealed before WI-2 code (R2).
**Corporate design version:** `cd-v1.7.1` (consumed via Model B).
**Workplan reference:** `WORKPLAN_v0.3.md`. Pasting this WP's track into `WORKPLAN_v0.3.md` as an append-only `§5.x` amendment is a **separate future maintainer act** — not performed at ratification (propose-don't-apply); the WP is Ratified without it.
