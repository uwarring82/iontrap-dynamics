# WP-06 — Tutorial-track accessibility: a newcomer rail across the docs

**Roll the proven T19/T20 newcomer rail across the whole tutorial track, add an on-ramp (Tutorial 0) and a glossary, and give the index a navigable difficulty map — Sail-only, no convention touched.**

Version 0.1 · Drafted 2026-07-14 · Status: Drafted

**Classification:** Sail execution under Coastline gates (per T(h)reehouse +EC CD 0.9).
**Licence:** This WP document itself is CC BY-SA 4.0 (`WP/LICENCE`). Deliverables carry their layer's licence: the tutorials, glossary, diagram, and index prose are Sail / CC BY-NC-SA 4.0; the one *optional* code item (a notebook-builder `???` extension under TA7, and any helper) is MIT (`tools/`, `src/`). No `CONVENTIONS.md` / spec edit — this WP introduces no convention.
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (`cd-rules`, consumed via Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

---

## 1. Card linkage *(Sail)*

Executes `task cards/TC-tutorial-accessibility.md` (**ID: TC-tutorial-accessibility**, v0.2, 2026-07-14 — decisions ratified).

**Objective lifted from the card (one line):** make the tutorial track approachable to a newcomer without trapped-ion / QuTiP / Gaussian-state fluency, by adding scaffolding *around* the physics — never simplifying the physics itself.

**Governing invariants from the card — quoted, not paraphrased** (hard acceptance gates):

> Additive-only. No physics claim, oracle, or `assert` is removed or loosened; the rail is scaffolding *around* the demonstration. A tutorial with its rail stripped is byte-for-byte the same falsifiable script.

> Simplifications are physics claims. Every newcomer-facing sentence is adversarially accuracy-reviewed.

> No convention touched. `CONVENTIONS.md` and `CONVENTION_VERSION` (0.5) are untouched.

The card's three ratified decisions (carried into §4): **TA4 taxonomy = `intro / core / advanced`**; **TA7 = collapsed setup blocks (Option B), no public `tutorial_helpers` module**; **TA8 = defer the T20 split** (annotate first; never renumber on the SQ6 branch).

## 2. WORKPLAN linkage *(Coastline gate — pending maintainer seal)*

This WP feeds a new dispatch track to be recorded in `WORKPLAN_v0.3.md` as the next free amendment **§5.9** (`*(Coastline, new in v0.3.11)*`).

- **Amendment §:** §5.9 — *tutorial-track accessibility* (to be pasted at ratification).
- **Doc version bump:** v0.3.10 → v0.3.11 (header line, footer `**Workplan version:**`, Endorsement Marker in lock-step).
- **Stub follows the §5.3 template:** anchoring sentence, scope (blocks/does not block which release), Rationale, "On `main` toward …" status, "Remaining sub-dispatches", "Consequence for §5 above".
- **Propose-don't-apply:** the §5.9 paste and the version bump are **maintainer acts at ratification** — this WP does not edit `WORKPLAN_v0.3.md`. **No `CONVENTION_VERSION` bump** (Sail-only track; contrast WP-01/03/05, which each sealed a convention).

## 3. Objective and scope *(Sail)*

**Objective (execution terms):** every tutorial carries a self-consistent newcomer rail (mental-model box, symbols table, common-confusion callouts, figure takeaways, orienting comments, explanatory `assert` messages); a new Tutorial 0 gives a 30-second mental model with one runnable spin-flop; a glossary defines the recurring vocabulary once; and the index offers a navigable difficulty map. "Done" = the three tutorial CI guards stay green and no oracle is weakened, across the whole track.

**In scope.** TA1 rail across tutorials 01–18 (T19/T20 shipped as the pilot); TA2 Tutorial 0; TA3 glossary; TA4 index "First time?" path + `intro/core/advanced` tags; TA5 getting-started runnable example; TA6 pipeline diagram; TA7 collapsed setup blocks (+ the notebook-builder `???` prerequisite).

**Out of scope (explicitly, from the card).** A public `tutorial_helpers` module (TA7 Option A, rejected); splitting T20 into 20a/20b (TA8, deferred — revisit only post-SQ6-merge if T20 still draws complaints); any physics simplification; any convention edit.

## 4. Work items *(Sail)*

| WI | Deliverable (proposed) | Key contents | Reuse | Acceptance | Dispatch | Status |
|---|---|---|---|---|---|---|
| **TA1** | `docs/tutorials/0*.md`, `1*.md` | the §4-card rail on tutorials 01–18, tuned per page | the shipped T19/T20 rail; `!!!` admonitions | `pytest -m tutorial` green; notebooks fresh; `mkdocs --strict`; every simplification accuracy-reviewed; no `assert` weakened | `TA1` | open |
| **TA2** | `docs/tutorials/00_mental_model.md` | 30-second physics‖code spine + one runnable spin-flop | `carrier_hamiltonian`, `spin_z`, `solve` (Tutorial 1 pattern) | the single cell runs from empty cwd; matches `[0-9][0-9]_*.md`; notebook builds | `TA2` | **draft landed** |
| **TA3** | `docs/glossary.md` | one definition per recurring term, per-term anchors, cross-links | `CONVENTIONS.md` as the authority | anchors resolve; nav-registered; linked from the tutorials that introduce each term | `TA3` | **draft landed** |
| **TA4** | `docs/tutorials/index.md`, per-tutorial badge | "First time?" path + `intro/core/advanced` tags | existing index bullets + Prerequisites lines | tags consistent with the real prerequisite chain; path stands alone in the shared namespace | `TA4` | open |
| **TA5** | `docs/getting-started.md` | a ~8-line runnable carrier-Rabi example beside the hand-built one | Tutorial 1 scenario | executes from a clean install; `mkdocs --strict` | `TA5` | open |
| **TA6** | pipeline diagram | one ASCII `IonSystem → … → readout` figure, reused | Tutorial 0's diagram (single source) | renders in site + notebooks; no CD-asset entanglement | `TA6` | open |
| **TA7** | `tools/build_tutorial_notebooks.py`; setup blocks | collapsed `??? note "Setup"` blocks; **prerequisite:** extend `transform_admonitions` to convert `???`/`???+` (it currently matches `!!!` only, so `???` leaks to Colab as literal text) | the builder's admonition transform | builder unit-checks `???` → blockquote; site renders (pymdownx.details is enabled); no new public API | `TA7` | open |

## 5. Sequencing and gates *(Coastline gate)*

**Order (per the card, maintainer-steered subset-first):** TA2 + TA3 (this slice — shared scaffolding) → TA7 prerequisite (`???` builder support) → TA1 on the **newcomer-critical subset 0, 1, 2, 6, 9** → TA4 path/tags → TA1 across the remaining 03–05, 07–08, 10–18 → TA5 → TA6 folded in with TA2/TA5. TA8 (T20 split) deferred, and only after the SQ6 branch merges to `main`.

**Blockers:** TA7 collapsed blocks depend on the `???` builder fix (logged). TA1 across the track and TA4 tags operate on `main`'s base tutorials; this WP branches off `main` (`wp-06-tutorial-accessibility`), independent of the in-flight SQ6 branch — but TA1 must be re-based over the SQ6 tutorial content once that merges (T19/T20 already carry the rail as the pilot).

**Coastline gates every WI must clear before it counts as landed:**

- [ ] **CONVENTIONS freeze respected** — no new convention; `CONVENTION_VERSION` stays 0.5.
- [ ] **Additive-only** — no oracle/`assert` weakened; rail is strippable to the original script.
- [ ] **The three tutorial CI guards** — `pytest -m tutorial` (execution), `build_tutorial_notebooks.py --check` (freshness), `mkdocs build --strict` (links/nav); admonitions render in **both** site and notebook.
- [ ] **Accuracy review** — every simplified newcomer-facing claim adversarially reviewed (the pilot's three slips are the precedent).
- [ ] **SPDX + CHANGELOG** — only if code changes (TA7 builder edit); docs-only WIs need no SPDX. A dispatch-keyed `[Unreleased]` bullet per landed dispatch.

## 6. Dispatch register *(Sail — proposed, not minted)*

Family **`TA`** (Tutorial Accessibility) — collision-checked clear against `WP/LOGBOOK.md`, `WORKPLAN_v0.3.md`, and `CHANGELOG.md` (taken families: single/double letters, `ED*`, `MC*`, `RL*`, `ND*`, `SQ*`, `AAG/AAH`; `TA*` has zero hits). **Codes mint at ratification** (maintainer act), then mirror into the `WP/LOGBOOK.md` registry.

| Dispatch | Maps to | CHANGELOG bullet | Status |
|---|---|---|---|
| `TA1` | WI-TA1 | `- **Dispatch TA1 — newcomer rail across tutorials 01–18.**` | proposed |
| `TA2` | WI-TA2 | `- **Dispatch TA2 — Tutorial 0: the 30-second mental model.**` | proposed |
| `TA3` | WI-TA3 | `- **Dispatch TA3 — docs glossary.**` | proposed |
| `TA4` | WI-TA4 | `- **Dispatch TA4 — index "First time?" path + difficulty tags.**` | proposed |
| `TA5` | WI-TA5 | `- **Dispatch TA5 — runnable getting-started example.**` | proposed |
| `TA6` | WI-TA6 | `- **Dispatch TA6 — pipeline diagram.**` | proposed |
| `TA7` | WI-TA7 | `- **Dispatch TA7 — collapsed setup blocks + notebook-builder `???` support.**` | proposed |

## 7. Release plan *(Coastline gate)*

Target: a **docs/accessibility minor or patch** — nothing here changes a default, a public API (TA7 stays Option B), or a convention. If TA7 ships the notebook-builder `???` extension, that is the only code change and rides as an additive tooling fix. SemVer per the repo convention; the release theme is *tutorial-track accessibility*. No convention seal, so no freeze coordination.

## 8. Logbook hooks *(Sail)*

Entries this WP has generated / will generate in `WP/LOGBOOK.md`:

- 2026-07-14 — WP-06 drafted against TC-tutorial-accessibility (v0.2, decisions ratified); `TA` family proposed (collision-checked, not minted); TA2 + TA3 drafts landed on `wp-06-tutorial-accessibility` off `main`.
- (at ratification) — mint `TA1–TA7`, paste WORKPLAN §5.9, bump v0.3.11.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This Work-Plan is a Sail execution document within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg), under the Coastline gates of `WORKPLAN_v0.3.md` and `CONVENTIONS.md`. Lock–Key rule applies: this WP is a key built on the stable locks those documents specify; it opens no new lock (Sail-only, no convention).

**Council status:** Guardian <pending: confirm no oracle/`assert` weakened, no convention touched, every simplification accuracy-reviewed>. Architect <pending: confirm TA7 stays Option B (no new public API), TA8 deferred, TA6 diagram single-sourced>. Scout <horizon: rebase TA1 over the SQ6 tutorial content once that merges; whether the rail should reach `benchmarks/` demo tools>. Integrator <sequenced subset-first per §5; branch off `main`, separate from the SQ6 PR; release is a docs/accessibility minor>.

**Convention version:** references `CONVENTIONS.md` v0.5 (frozen, §26 sealed). This WP introduces no convention and bumps no `CONVENTION_VERSION`.
**Corporate design version:** `cd-v1.7.1` (consumed via Model B).
**Workplan reference:** `WORKPLAN_v0.3.md` v0.3.10; this WP's track lands as amendment §5.9 (`new in v0.3.11`), pending maintainer seal.
