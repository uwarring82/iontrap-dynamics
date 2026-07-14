# Task Card — Tutorial-track accessibility: a newcomer rail across the docs

**Authored from:** the `iontrap-dynamics` maintainer's review of the SQ6 tutorial weaving (T19/T20), which landed a per-tutorial newcomer rail as a **pilot** and prompted a broader, track-wide accessibility pass.
**Upstream target:** `uwarring82/iontrap-dynamics` @ `main` (post-v0.6.0 unreleased; `CONVENTIONS.md` v0.5 frozen, §26 sealed).
**ID:** TC-tutorial-accessibility · **Status:** v0.2 — the three embedded decisions **ratified by the maintainer 2026-07-14** (TA4 taxonomy = `intro/core/advanced`; TA7 = collapsed setup blocks, no public module; TA8 = defer the T20 split). Lifted to `WP/WP-06-tutorial-accessibility.md` (Drafted). First slices TA2 (Tutorial 0) + TA3 (glossary) drafted on `wp-06-tutorial-accessibility` off `main`.
**Licence:** this card is CC BY-SA 4.0 (spec/Coastline-adjacent). Deliverables carry their layer's licence: the tutorials, glossary, diagram, and index prose = **Sail / CC BY-NC-SA 4.0** (adaptive guidance, not coastline); the optional `tutorial_helpers` module (if adopted, TA7) = **MIT** (`src/`). **No `CONVENTIONS.md` edit and no `CONVENTION_VERSION` bump** — the whole card is Sail plus one *optional* additive code module; it defines no physics convention.

---

## 1. Verdict (fit assessment)

The tutorial track is **technically strong but assumes fluency** the target newcomer does not have: trapped-ion notation, QuTiP `tensor` structure, and Gaussian-state language appear without on-ramps. The maintainer's guidance is explicit and correct: **add a newcomer rail, do not simplify the main physics.** The precision is an asset; the barrier is the *missing scaffolding around* the precision, not the precision itself.

The gap is **entirely additive and Sail-layer**: overview scaffolding, orienting callouts, a glossary, and a difficulty map — none of it redefines a coastline or a convention. A single **pilot already shipped** on T19/T20 (New-here box, symbols table, Common-confusion callouts, figure Takeaways, boilerplate/physics comments, explanatory `assert` messages), executes green, and renders in both the site and the Colab notebooks. This card **generalises that proven pattern across the track** and resolves the three structural decisions the pilot deliberately left open.

**One boundary must hold:** the rail is *scaffolding around* the physics, never *inside* it. No tutorial's physics claim, oracle, or `assert` is weakened; the newcomer material is strictly additive (boxes, tables, captions, comments, messages). A tutorial with its rail removed must still be the same falsifiable demonstration it is today.

**Non-apparatus:** every item is docs/library work — unblocked, parallelisable, no hardware dependency.

## 2. Ownership / layer boundary

| Capability | Layer / licence | Owner |
|---|---|---|
| Per-tutorial newcomer rail (New-here box, symbols table, Common-confusion, Takeaways, explanatory asserts) | Sail docs / CC BY-NC-SA 4.0 | library |
| Tutorial 0 — the 30-second mental model page | Sail docs | library |
| `docs/glossary.md` + cross-links | Sail docs | library |
| Index "First time?" path + difficulty tags | Sail docs | library |
| `getting-started.md` runnable physics example | Sail docs | library |
| Pipeline diagram (reused across index / Tutorial 0 / getting-started) | Sail docs | library |
| **[DECISION]** `tutorial_helpers` setup module | **code / MIT** (`src/`) *if adopted* | library |
| **[DECISION]** Splitting heavy tutorials (e.g. T20 → 20a/20b) | Sail docs (renumber) | library |

## 3. Already present — do not rebuild

Confirmed in the current tree:

- **Admonition support** — `!!! note / tip / warning` are house style across tutorials 01–12; `tools/build_tutorial_notebooks.py` (`transform_admonitions`) converts them to labelled blockquotes (📝/💡/⚠️) for Colab. **Reuse; do not invent a new callout syntax.**
- **Notebook build + freshness guard** — `build_tutorial_notebooks.py` (single source of truth = Markdown; `--check` CI freshness). A Colab "Setup" `pip install` cell is auto-prepended, so any helper must ship *inside the installed package*, not as a repo-relative import (bears directly on TA7).
- **Tutorial-execution oracle** — `tests/docs/test_tutorials_execute.py` (`pytest -m tutorial`) execs every `[0-9][0-9]_*.md`'s python blocks in one namespace from an **empty cwd**. A `00_*.md` (Tutorial 0) is automatically covered by the two-digit glob.
- **Rewrite-links / rendering** — the builder rewrites relative `*.md` links to the published site; `mkdocs --strict` gates cross-links and nav. `docs/glossary.md` anchors (`glossary.md#adiabatic`) are covered by this once registered in nav.
- **The pilot** — T19/T20 rails (this card's exemplar), staged on `sq6-forced-displacement-echo` alongside the SQ6 engine (`bbf837b`). **TA1 lifts this exact pattern; it is not re-derived.**
- **Getting-started + index** — `docs/getting-started.md`, `docs/tutorials/index.md`, `mkdocs.yml` nav all exist and are edited in place.

## 4. The newcomer-rail pattern (the Sail spine)

The pilot fixed a **reusable rail** every tutorial can carry. TA1 rolls it across the track; the other dispatches add track-level scaffolding. The rail per tutorial:

- **New-here box** (`!!! note "New here? Read this first"`) — a plain-English mental model in ≤6 bullets, ending with an **"In a hurry?"** minimal-run-path (verified to stand alone in the shared top-to-bottom namespace).
- **Symbols table** — the ≤7 symbols that tutorial uses, one line each.
- **Common-confusion callouts** (`!!! warning`) — 1–2 per tutorial, each a crisp actionable rule for a real trap (e.g. one-way ≠ cyclic; the top Fock level lies; squeezing ≠ displacement).
- **Figure Takeaways** — a one-sentence `**Takeaway.**` after the key figures.
- **Orienting comments** — separate *boilerplate* (Hilbert-space setup, inert spectator spin) from *the one function to focus on*.
- **Explanatory `assert` messages** — the instructive checks carry a `", …"` message so a failing Colab cell teaches instead of just aborting.
- **Optional "You should see"** expected-output callout after key cells (batch-2 item #5) — a `!!! tip` with the target numbers + a one-line self-diagnosis ("if `ν` ≫ 1, your Fock cutoff is too small").

**Accuracy discipline (learned from the pilot):** a focused adversarial review of the pilot caught three real accuracy slips a simplification introduced (`ν` mislabeled "purity"; "untouched, in fact stronger"; "slow → no squeezing" vs. parametric modulation). **Every rail addition is adversarially reviewed for newcomer-facing accuracy** — a simplified sentence is a physics claim and is gated as one.

## 5. Proposed dispatches (family `TA`)

Priorities: P0 = pattern/decisions the rest depend on; P1 = core scaffolding; P2 = polish. All Sail unless marked. **None carry a conventions gate.**

### TA1 — Roll the newcomer rail across the track *(the core deliverable)*
- **Serves:** every tutorial.
- **Add:** the §4 rail to the remaining tutorials (01–18), tuned per tutorial (each gets its own symbols/confusions; not boilerplate-stamped). T19/T20 are the shipped exemplar.
- **Acceptance:** each rail addition executes green (`pytest -m tutorial`), notebooks regenerate fresh, `mkdocs --strict` clean, admonitions render in site + notebook; every simplified claim adversarially accuracy-reviewed; no `assert`/oracle weakened.
- **Priority:** P1 (per-tutorial, parallelisable). **Owner:** library (Sail).

### TA2 — Tutorial 0: the 30-second mental model
- **Serves:** the whole track's on-ramp.
- **Add:** `docs/tutorials/00_*.md` — the one-sentence physics loop beside the one-sentence code loop (`IonSystem → HilbertSpace → hamiltonian → solve → readout`), and **one** minimal runnable cell that flops a spin and prints `⟨σ_z⟩` — an early win before `StorageMode`/`ResultWarning`/Fock truncation. Register in nav + index.
- **Acceptance:** the single cell executes from empty cwd; ≤1 screen; links onward to Tutorial 1; notebook builds.
- **Priority:** P1. **Owner:** library (Sail).

### TA3 — Glossary
- **Serves:** TA1, TA2, and every tutorial.
- **Add:** `docs/glossary.md` defining each recurring term once (coastline, sail, Fock truncation, Lamb–Dicke, RWA, adiabatic, WKB phase, symplectic eigenvalue, covariance matrix, quench, …), linked as `[adiabatic](../glossary.md#adiabatic)`. Tutorials keep their compact per-page symbols table (TA1); the glossary is the deep reference.
- **Acceptance:** anchors resolve under `mkdocs --strict`; registered in nav; terms cross-linked from ≥ the tutorials that introduce them.
- **Priority:** P1. **Owner:** library (Sail).

### TA4 — Index "First time?" path + difficulty tags **[RATIFIED: `intro/core/advanced`]**
- **Serves:** newcomer navigation.
- **Add:** a "First time?" paragraph on `docs/tutorials/index.md` (the guided path 0 → 1 → 2 → 6, "stop when you can build a Hamiltonian and diagnose Fock truncation"), plus a per-tutorial difficulty tag.
- **RATIFIED (2026-07-14):** vocabulary = **`intro` / `core` / `advanced`**, rendered as a **bracketed prefix in the index bullet** *and* a **one-line badge at the top of each tutorial**. (Rejected: two-tier `core`/`advanced`; prerequisites-only.)
- **Acceptance:** path and tags consistent across index + tutorials; no tag drift vs. the actual prerequisite chain.
- **Priority:** P1. **Owner:** library (Sail).

### TA5 — Getting-started runs physics
- **Serves:** first-contact users who bounce off an abstract example.
- **Add:** a second `docs/getting-started.md` example (~8 lines) that runs the Tutorial-1 carrier-Rabi simulation and prints a result — the package visibly *doing something* — beside the existing hand-built `TrajectoryResult`.
- **Acceptance:** the snippet executes against the public API from a clean install; `mkdocs --strict` clean.
- **Priority:** P2. **Owner:** library (Sail).

### TA6 — Pipeline diagram
- **Serves:** TA2, TA5, index.
- **Add:** one portable pipeline diagram (`IonSystem → HilbertSpace → Hamiltonian → solve → readout`, with the sub-labels), reused across Tutorial 0 / getting-started / index. **Prefer a fenced ASCII diagram** (renders identically in mkdocs and Colab, no asset-integrity/CD-token entanglement); an SVG only if a richer figure is wanted (then it must clear the corporate-design asset gate).
- **Acceptance:** renders in site + notebooks; single source, not duplicated divergently.
- **Priority:** P2. **Owner:** library (Sail).

### TA7 — Collapsed setup blocks **[RATIFIED: Option B — no public module]**
- **Serves:** reducing repeated `ModeConfig`/`IonSystem`/`HilbertSpace` boilerplate that hides the new idea (batch-2 item #2).
- **RATIFIED (2026-07-14): Option B — collapsed setup blocks.** Keep the ~10-line setup inline but wrap it in a `??? note "Setup (click to expand)"` collapsible + the TA1 "boilerplate" comment, so newcomers can fold it away **with no new public API**. Rejected: **Option A** (ship `iontrap_dynamics.tutorial_helpers`) — a new supported public surface + cross-tutorial coupling is overkill for docs scaffolding; **Option C** (status-quo comments only) — leaves the noise.
- **TA7a — prerequisite (confirmed defect, not an open question):** the notebook builder's `transform_admonitions` matches **`!!!` only**, so `???`/`???+` collapsibles would leak to Colab as literal text (the site is fine — `pymdownx.details` is enabled). **Extend `tools/build_tutorial_notebooks.py` to convert `???`/`???+` to a labelled blockquote** (an MIT tooling change) **before** any tutorial uses a collapsible. Unit-test the transform.
- **Acceptance:** no repo-relative import; tutorials still execute from empty cwd; `???` renders in site (details) and in notebook (blockquote after TA7a); no new public API.
- **Priority:** TA7a P1 (unblocks the collapsibles) → setup-block conversion P2, folded into TA1. **Owner:** library.

### TA8 — Heavy-tutorial split **[RATIFIED: defer]**
- **Serves:** cognitive load on the longest tutorials (T20 now spans Pₙ, dynamical pairs, truncation guard, displacement, echo, cosmology).
- **RATIFIED (2026-07-14): defer.** Annotate first — the TA4 `advanced` tag + the TA1 "In a hurry?" path — and split T20 into **20a / 20b** only if it still draws complaints after TA1/TA4 land, **and only after the SQ6 branch merges to `main`** (a split renumbers and collides with the SQ6 Step 4 echo; **never split on the SQ6 branch**). Splitting is the highest-churn, lowest-certainty item, so it stays out until evidence demands it.
- **Priority:** P2 (deferred, post-merge, evidence-gated). **Owner:** library (Sail).

## 6. Acceptance gates / decision rules (Guardian)

- **Additive-only.** No physics claim, oracle, or `assert` is removed or loosened; the rail is scaffolding *around* the demonstration. A tutorial with its rail stripped is byte-for-byte the same falsifiable script.
- **Simplifications are physics claims.** Every newcomer-facing sentence is adversarially accuracy-reviewed (the pilot's three slips are the precedent for why).
- **The three CI guards stay green** on every landed item: `pytest -m tutorial` (execution), `build_tutorial_notebooks.py --check` (freshness), `mkdocs --strict` (links/nav). Admonitions must render in **both** site and notebook.
- **No repo-relative reads** (empty-cwd Colab constraint) — governs TA5/TA7 especially.
- **No convention touched.** `CONVENTIONS.md` and `CONVENTION_VERSION` (0.5) are untouched; if TA7 Option A ships, it is an additive MIT module with the standard code gates (ruff/format/mypy/SPDX/CHANGELOG), not a convention.
- **Decisions before their dependents.** TA4's taxonomy, TA7's surface choice, and TA8's split/defer are ratified **before** the work they gate.

## 7. Sequencing and open questions

**Order (decisions ratified 2026-07-14, maintainer-steered subset-first):** TA2 Tutorial 0 + TA3 glossary (**this slice — drafted**; TA6 diagram single-sourced from TA2) → TA7a `???` builder fix → TA1 on the **newcomer-critical subset 0, 1, 2, 6, 9** → TA4 path/tags → TA1 across the remaining 03–05, 07–08, 10–18 → TA5 getting-started → TA8 deferred (post-SQ6-merge, evidence-gated). TA1 on T19/T20 is the shipped pilot.

**Resolved at ratification (2026-07-14):** TA4 → `intro/core/advanced` (bracketed index prefix + top-of-tutorial badge); TA7 → Option B (collapsed blocks, with the `???` builder fix promoted to prerequisite **TA7a**); TA8 → defer; TA1 scope → **subset-first** (0,1,2,6,9) then the rest; execution → **WP-06** (a full WP, not a docs-only mini-track).

**Still open (for the breakout, non-blocking):**
- **Expected-output callouts (TA1 sub-item):** every key cell, or only the ones with a common failure mode?
- **Rail reach:** should the rail extend to the `benchmarks/` demo tools and `getting-started`, or stay within the tutorial track?

## 8. Execution

This card is the **spec**. Execution is **`WP/WP-06-tutorial-accessibility.md` (Drafted 2026-07-14)**: it lifts the §6 gates as quoted invariants, proposes the `TA` family (collision-checked clear, **minted at ratification**, registered in `WP/LOGBOOK.md`), flags the WORKPLAN §5.9 amendment (v0.3.10 → v0.3.11) as a pending maintainer seal, and lands per-tutorial (each rail is its own reviewable slice). Release theme = *tutorial-track accessibility* (docs/accessibility minor); nothing bumps a convention or changes a default.

---

## Endorsement Marker

**Local candidate framework under active stewardship. No parity implied with externally validated laws.** This Task Card is a deliberation spec for `uwarring82/iontrap-dynamics`, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg), under the Coastline gates of `WORKPLAN` and `CONVENTIONS.md` v0.5. Lock–Key rule applies: this card is a key built on the stable locks those documents specify; it opens no new lock (Sail-only, plus one optional additive MIT module).

**Council status:** Guardian — pending: confirm no oracle/`assert` is weakened, no convention is touched, and every simplification is accuracy-reviewed (per-slice). Architect — **cleared on the two structural calls (2026-07-14): TA7 = Option B (no new public API); TA8 = deferred (no renumber on the SQ6 branch).** Scout — horizon: rebase TA1 over the SQ6 tutorial content once it merges; whether the rail should reach `benchmarks/` demo tools and `getting-started`. Integrator — sequenced subset-first per §7; every item additive and independently landable; the three tutorial CI guards are the merge gate.

**Convention version:** references `CONVENTIONS.md` v0.5 (frozen, §26 sealed). **This card specifies no convention and no `CONVENTION_VERSION` bump.**
**Pilot:** T19/T20 newcomer rail, staged on `sq6-forced-displacement-echo` with the SQ6 engine (`bbf837b`), 2026-07-13 — the shipped exemplar TA1 generalises.

## Version history

| Version | Date | Change |
|---|---|---|
| 0.2 | 2026-07-14 | Maintainer ratified the three decisions: **TA4** = `intro/core/advanced` (bracketed index prefix + top-of-tutorial badge); **TA7** = Option B collapsed setup blocks (no public module), with the `???` notebook-builder fix promoted to prerequisite **TA7a**; **TA8** = defer the T20 split. Sequencing set subset-first (0,1,2,6,9). Lifted to `WP/WP-06-tutorial-accessibility.md` (Drafted); `TA` family collision-checked. First slices TA2 (Tutorial 0) + TA3 (glossary) drafted. Architect cleared on TA7/TA8. |
| 0.1 | 2026-07-13 | Initial deliberation draft. Fit verdict, ownership/layer boundary, do-not-rebuild inventory, the §4 newcomer-rail pattern (from the T19/T20 pilot), dispatches TA1–TA8 with the three embedded decisions (TA4 taxonomy, TA7 helper surface, TA8 split), Guardian gates, sequencing, and a WP-06 execution hook. Sail-only; no convention gate. Not endorsed. |
