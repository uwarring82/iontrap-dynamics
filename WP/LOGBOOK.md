# WP — Development Logbook

**Dated, append-only narrative of decisions, dispatches, dead-ends, and rationale for `iontrap-dynamics`**

Version 0.1 · Drafted 2026-06-02 · Status: live; first entry 2026-06-02

**Classification:** Sail (narrative) under Coastline gates (per T(h)reehouse +EC CD 0.9).
**Licence:** CC BY-SA 4.0 (Coastline-adjacent; see root `LICENCE`).
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (`cd-rules`, consumed via Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

---

## Purpose *(Sail)*

This is the **single dated stream** of how the project actually went: the decisions, the rationale, the options rejected, the dead-ends, the deferrals, the dispatch-code minting, and the release-cut events. It exists because, before it, that narrative was reconstructed from three incompatible places — WORKPLAN `§`-amendments, design-note `§`-additions, and git commit bodies — and the dead-ends and null results were buried in prose or lost entirely.

**What the logbook captures that nothing else does:**

- **Decision rationale and alternatives rejected,** chronologically, in one place.
- **Open-question lifecycle** across the whole repo (question ID, opened-on, status, resolution, where the spec changed) — not just inside one design note.
- **Dead-ends, deferrals, and null results** — the things `CHANGELOG.md` structurally cannot hold, because it is *Added / Changed / Fixed* only. (e.g. "post-α GPU open-system branch deferred indefinitely under the Q6 resolution"; "jax-metal 0.1.1 incompatible with jax 0.10.0 — local M1 cannot serve as the GPU reference" — these belong here.)
- **The release-cut procedure and its per-release SemVer justification,** so it stops being archaeology recoverable only from commit bodies.
- **The dispatch-code registry** — code → title → WP → status → carrier — so codes do not collide and "what shipped when" is answerable in one place.
- **Cross-repo decisions** referencing `cd-rules`, the `single-25Mg-plus` twin, or the two-repo hosting model (`§4.0`).

### Reconciliation with CHANGELOG — read this once *(Coastline gate)*

- **CHANGELOG = what shipped.** Keep-a-Changelog, dispatch-indexed, *Added / Changed / Fixed*, per-release summary + test counts. Outcome only. Binding surface record.
- **LOGBOOK = why it went that way.** Narrative, dated, append-only. Holds rationale, rejected options, Q-resolutions, dead-ends, deferrals, release-cut events, code minting — none of which the CHANGELOG can or should hold.

If a line says "function X added, tests pass", it is a CHANGELOG line. If a line says "we chose X over Y because Z, and Y is deferred until a real case lands", it is a logbook line. **Never duplicate the CHANGELOG here**; link to it.

## Entry format *(Sail)*

Append entries newest-last under `## Entries`. Each entry:

```
### YYYY-MM-DD — <short title>

- **Refs:** WP-NN · Dispatch <CODE> · TC-<ID> · §5.x · Q<n>   (whichever apply)
- **Stance:** Guardian | Architect | Scout | Integrator   (only if a stance framed the call; omit otherwise)
- **What:** one or two sentences — the decision or event.
- **Why:** the rationale, and the options rejected (name them).
- **Outcome:** landed / deferred / null result / open; where the spec changed.
- **Links:** CHANGELOG bullet, WORKPLAN amendment, design note §, commit, card.
```

Keep entries short and honest. A null result is a first-class entry — label it as one. Originally-recorded entries are never rewritten; a later correction is a *new* entry that reads back through the old one (honesty discipline, carried from the WORKPLAN).

---

## Entries

### 2026-06-02 — WP management system created; TC-ITD-ESTDARW-01 received; FAIR initiative opened

- **Refs:** WP-01 (to be drafted) · TC-ITD-ESTDARW-01
- **Stance:** Integrator (sequencing the working surfaces) with a Guardian check (no Coastline gate relaxed).
- **What:** Stood up a lightweight, governed Work-Plan framework under `WP/` — `WP/README.md` (what a WP is and is not, the five-carrier separation, the Drafted → Ratified → In-flight → Released → Archived lifecycle, `WP-NN-slug.md` naming), `WP/TEMPLATE.md` (reusable governed skeleton), and this logbook. Received task card **TC-ITD-ESTDARW-01** ("`iontrap_dynamics` Service Upgrade: Estimation & Darwinism", v0.1, 2026-06-01): add four application-agnostic capabilities — classical/quantum Fisher information + Cramér–Rao bounds, quantum-Darwinism redundancy/recoverability, GHZ/cat state factories, and a correlated common-mode channel — each grounded in a targeted literature review and proven by a generic benchmark, landed under the Convention Freeze and shipped in a tagged release. Opened the **FAIR initiative** (Findable / Accessible / Interoperable / Reusable) as the framing goal: the WP framework and this logbook are the first FAIR deliverable, making the project's decision history findable and reusable rather than reconstructed from commit bodies.
- **Why:** Before today the project had a strategic Coastline roadmap (`WORKPLAN_v0.3.md`, v0.3.5) and incoming `task cards/`, but **no execution-plan layer and no logbook**. Decisions lived in three incompatible carriers (WORKPLAN `§`-amendments, design-note `§`-additions, git commit bodies); dead-ends and deferrals (e.g. the post-α GPU open-system branch and the β-as-benchmark option both "deferred indefinitely" under the Q6 resolution `α → C → β-thin → B → γ`; the jax-metal/jax incompatibility that ruled out the local M1 as a GPU reference) were buried in design-note prose or in memory and never surfaced in one dated stream. A new card arriving — and the maintainer expecting more — made the gap actionable. **Rejected options:** (a) a formal ADR directory — too heavy for a solo maintainer and redundant with the existing `§N decision recorded` design-note habit; (b) widening the CHANGELOG to hold rationale — breaks Keep-a-Changelog's *Added/Changed/Fixed* contract and conflates *what shipped* with *why*; (c) putting execution plans directly into WORKPLAN amendments — would force append-only edits on a living plan and bloat the Coastline roadmap with WI-level detail. Chosen instead: a thin `WP/` Sail layer plus this append-only narrative logbook, each with one job, none duplicating the CHANGELOG or the roadmap.
- **Outcome:** Landed (framework + first logbook entry). **Next:** draft `WP-01-estimation-darwinism.md` against TC-ITD-ESTDARW-01 (Status: Drafted → Ratified), mint its dispatch codes, and add the new dispatch-track stub to `WORKPLAN_v0.3.md` as the next free `§5.x` amendment (`new in v0.3.6`), with the header version line, footer `**Workplan version:**`, and Endorsement Marker bumped in lock-step. The card's governing invariant — **application-agnostic; no consuming-application framing in any library symbol** — carries into WP-01 verbatim and is a hard acceptance gate.
- **Links:** `task cards/task-card-iontrap-dynamics-service-upgrade.md` (TC-ITD-ESTDARW-01) · `WP/README.md` · `WP/TEMPLATE.md` · CHANGELOG `[Unreleased]` (no entry yet — nothing shipped) · WORKPLAN `§5` (amendment stub pending).

### 2026-06-02 — WP-01 subpackage naming ratified: `information/` umbrella

- **Refs:** WP-01 · TC-ITD-ESTDARW-01 · §3
- **Stance:** Architect (public-surface shape) with a Guardian check (No-TMC honesty of the name).
- **What:** With `WP-01-estimation-darwinism.md` drafted (Status: Drafted), ratified its §3 subpackage-naming decision in favour of **Option B — a single `information/` umbrella** (`information/fisher.py`, `information/redundancy.py`, `information/recoverability.py`), over Option A (the card's `estimation/` + `darwinism/` split). GHZ/cat factories still extend `states.py`; the common-mode channel still extends `systematics/`. The `<info>/` placeholder throughout WP-01 §4 and §7 is now resolved to `information`, and WP-01 §3 is marked ratified.
- **Why:** `information` is the most application-agnostic root — a downstream reader cannot mis-read it as application framing, where "darwinism" risked exactly that; and CFI, QFI and fragment mutual information genuinely share one nonlinear-in-ρ helper layer (`_ensure_density`, `_binary_entropy`, ptrace masks) that a single sub-package gives a home, avoiding a helper with no owner. **Rejected:** Option A (the card's literal split) — cleaner one-to-one topic-to-name mapping, but two top-level sub-packages and a homeless shared-helper layer. The decision is a one-way door (the downstream TMC application pins the import path), so it was ratified before WI-1 opened.
- **Outcome:** Landed (decision). WP-01 §3 ratified; header `Status:` line updated. **Next:** WI-1 (`information/fisher.py` + the keystone QFI-scaling benchmark) may open on the maintainer's go; the §13 WORKPLAN §5.4 dispatch-track stub becomes paste-ready when WI-1 lands on `main`.
- **Links:** `WP/WP-01-estimation-darwinism.md` §3 · TC-ITD-ESTDARW-01 §4 (WI table) · CHANGELOG `[Unreleased]` (no entry yet).

### 2026-06-02 — WP framework review applied; lifecycle and licences clarified

- **Refs:** WP-01 · `WP/REVIEW_LOG.md` · §3 · §4.4
- **Stance:** Guardian (a review found the seed WP contradicting the framework's own rules) into Architect follow-through (resolve, don't paper over).
- **What:** Applied a document / cross-reference review (`WP/REVIEW_LOG.md`) of the untracked `WP/` framework. Resolutions: (1) **WP-01 now instantiates the template spine** — reclassified document-level to *Sail execution under Coastline gates* (was bare *Coastline*) and added the missing spine sections §14 Sequencing-and-gates, §15 Dispatch-register, §16 Logbook-hooks; the README now admits *constraint-heavy* WPs explicitly. (2) **Lifecycle disambiguated** — only the §3 naming *sub-decision* is ratified; WP-01 as a whole stays **Drafted**, so dispatch codes are unminted and WI-1 is blocked until Ratified; the §13 stub is relabelled a *template stub*, not paste-ready. (3) **WI-4 API fixed** — `CommonModePhase` gains a `correlation ∈ [0, 1]` field with `offset_i = √c·ξ_shared + √(1−c)·ξ_i`, so the card's *reduces-to-independent-at-zero-correlation* acceptance is now representable (it was not). (4) **Registry widened** (see below). (5) **Card path made literal** in WP-01, and the README pattern relaxed from `TC-*.md` to `*.md` carrying an internal `TC-…` ID (the real cards do not match `TC-*.md`). (6) **Licences reconciled** — the whole `WP/` folder is CC BY-SA 4.0 governance material (`WP/LICENCE` added; FAIR side-car aligned), and the literature-review note is recommended Coastline / CC BY-SA (cited as authoritative). The regressed `<info>/` placeholder wording in WP-01 §3 was also restored.
- **Why:** A framework whose own seed example violates its central rule is not yet trustworthy, and the review caught exactly that — a Coastline-classified WP missing the spine, an un-satisfiable WI-4 acceptance, and a registry that would still collide. Fixing the seed is cheaper now than after WP-02. **Rejected:** relaxing the framework merely to *admit* a constraint-heavy exception without making WP-01 conform — that would hide the un-spined WP rather than fix it; instead WP-01 was made to carry the spine **and** the README now names the constraint-heavy case.
- **Outcome:** Landed (framework revision). WP-01, README, this logbook, and FAIR updated; `WP/LICENCE` added; the root `LICENCE` `WP/` row is *proposed, not applied* (governed file). Two open calls remain for the maintainer: the literature-review note licence, and ratifying the root-`LICENCE` `WP/` row.
- **Links:** `WP/REVIEW_LOG.md` · `WP/WP-01-estimation-darwinism.md` §14–16 · `WP/LICENCE` · `WP/README.md` §4 / §6.

### 2026-06-02 — Second task card received (undetected-modes) → WP-02 pending; QFI + v0.3-freeze coordination

- **Refs:** WP-02 (to be drafted) · `task cards/TC-iontrap-dynamics.md`
- **Stance:** Scout (a second consumer appears on the horizon) into Integrator (sequence it against WP-01).
- **What:** Received the second task card, `task cards/TC-iontrap-dynamics.md` — `iontrap-dynamics` as a **service module for *Undetected Modes*** (WP1, r5), authored from the `undetected-modes` side. It requests F1 two-mode squeezing (SU(1,1)) Hamiltonian, F2 two-mode squeezed-vacuum factory, F3 **typed motional CPTP channels + `c_ops` exposure** (pivotal — `solve()` hard-wires `c_ops=[]` today), F4 interferometric observables (visibility, fringe phase), F5 Lamb–Dicke helpers, **F6 a general QFI primitive (optional)**, and F7 identifiability support (a small new `ModeFrequencyDrift` systematic). It will become **WP-02** (serial minted at its ratification).
- **Why:** Two genuine cross-WP coordination points, recorded now so they are not rediscovered late. **(a) QFI overlap** — card-2 F6 (general QFI) is the *same primitive* as WP-01 WI-1 (`information/fisher.py`); the library should host **one** QFI implementation that serves both, with each programme's resource-constraint / identifiability logic staying downstream. **(b) Shared v0.3 Convention Freeze** — card-2 F1 (two-mode squeezing phase/sign/ordering) and F3 (channel parameterisation) require a CONVENTIONS v0.3 freeze, the *same bump* WP-01 §6 opens for §19–22. Whether one combined v0.3 freeze covers both WPs (card-2 §7 Q2 recommends combined) or two is a roadmap-level call for `WORKPLAN_v0.3.md`, not for either WP alone.
- **Outcome:** Open. WP-02 not yet drafted; flagged in WP-01 §14 Blockers. **Next (maintainer's call):** draft WP-02 from the template; decide F6/WI-1 QFI ownership (recommend one library primitive) and the v0.3-freeze cadence before either WP opens its QFI or two-mode-squeezing conventions.
- **Links:** `task cards/TC-iontrap-dynamics.md` · `WP/WP-01-estimation-darwinism.md` §14 · card-2 §7 (open questions for U).

### 2026-06-02 — Round-2 review approved; literature-review note licence ratified Coastline / CC BY-SA

- **Refs:** WP-01 · `WP/REVIEW_LOG.md` (Round 2) · §5 · §9
- **Stance:** Guardian (licence call closed cleanly) into Integrator (small consistency fixes folded in).
- **What:** The Round-2 review (`WP/REVIEW_LOG.md`) marked the framework **approved for ratification** — every Round-1 finding resolved. The maintainer **ratified the literature-review note (`docs/estimation-darwinism-review.md`) as Coastline / CC BY-SA 4.0** (it carries a `## Endorsement Marker`, not the Sail `## Licence` footer), because CONVENTIONS §19–22 cite it as authoritative — closing the last open licence call. Folded in three non-blocking WP-01 fixes from the review: §7 benchmark 5 renamed `run_demo_ghz_cat.py` → `run_benchmark_ghz_cat.py` (pattern consistency); §10 now headlines the `states.py` top-level re-export as the largest API-surface change; §14 states explicitly why WI-3 precedes WI-2 (the keystone QFI benchmark needs `ghz_state`). Logged three forward watch-items: collapse `per-file-ignores` to an `information/*` glob at 2–3 modules; compare WP-02's freeze plan against WP-01 §6 before either goes In-flight; mint all of C1–C6 into the registry at Ratification.
- **Why:** The note defines the conventions the library will freeze; treating it as binding Coastline rather than interpretive Sail keeps the definition-of-record and its licence aligned with `CONVENTIONS.md`. The root-`LICENCE` `WP/` row remains the one outstanding governed edit (proposed in `WP/LICENCE`, not applied).
- **Outcome:** Landed. WP-01 §5 / header / §9 updated to *ratified Coastline*; §7 / §10 / §14 fixes applied. WP-01 is review-approved and ready for the maintainer to move Drafted → Ratified. **Open (separately):** whether to consolidate TC-ITD-ESTDARW-01 and TC-iontrap-dynamics into a common WP, or keep WP-01 + WP-02 with a shared v0.3-freeze coordination — under deliberation; to be logged when decided.
- **Links:** `WP/REVIEW_LOG.md` (Round 2) · `WP/WP-01-estimation-darwinism.md` §5 / §7 / §10 / §14 · `WP/LICENCE` (root-row proposal).

### 2026-06-02 — Structure ratified: two WPs + shared v0.3 Convention Freeze

- **Refs:** WP-01 · WP-02 (pending) · `WP/FREEZE-v0.3.md`
- **Stance:** Architect (a structural choice spanning two cards) with a Guardian check (one-card-one-WP preserved).
- **What:** Resolved the open structural question from the previous entry. The two task cards are executed as **two separate WPs** — WP-01 (estimation/Darwinism) and WP-02 (undetected-modes) — **not** merged into one programme-WP. The two genuine overlaps are handled *above* the WPs by a new side-car, `WP/FREEZE-v0.3.md`: (a) it owns the single `CONVENTION_VERSION` 0.2 → 0.3 bump and the section allocation (WP-01 §19–22; WP-02 §23–24), so neither WP bumps alone; (b) it records that QFI is **one** primitive — WP-01 WI-1 delivers `information/fisher.py`, WP-02 F6 consumes it. WP-01 §6 / §10 / §14 / §15 / footer reframed to *feed* the shared freeze rather than own it; `WP/README.md` §4 now names shared-concern side-cars explicitly.
- **Why:** A Convention Freeze is a repo-wide, version-gated event — never a per-WP concern — so two WPs each "bumping to v0.3" is incoherent. Lifting the freeze to a shared side-car gives card-2 the single combined freeze its §7 Q2 asks for **without** merging two cards into one document. **Rejected:** (B) one consolidated programme-WP — breaks one-card-one-WP and puts the No-TMC and undetected-modes boundaries under one master; (C) separate v0.3 / v0.4 freezes — clean but two bumps/releases and ignores card-2's combined-freeze preference. Option (A) is the only one that keeps both the framework rule and the combined freeze.
- **Outcome:** Landed. `WP/FREEZE-v0.3.md` created; it carries the §4 **timeline-coupling decision** (combined seal vs WP-01-first) deferred to WP-02 ratification. **Next:** draft WP-02 from the template (workflow-scale, against card-2 F1–F7), then take the `FREEZE-v0.3.md` §4 call.
- **Links:** `WP/FREEZE-v0.3.md` · `WP/WP-01-estimation-darwinism.md` §6 / §14 · `task cards/TC-iontrap-dynamics.md` (card 2).

### 2026-06-02 — WP-01 Ratified; dispatch family `ED` minted; WI-1 opened

- **Refs:** WP-01 · EDA–EDF · `WP/FREEZE-v0.3.md`
- **Stance:** Integrator (planning → execution) with a Guardian check (codes collision-checked before minting).
- **What:** Moved **WP-01 Drafted → Ratified** (review-approved, Round 2). Minted its six dispatch codes as the fresh **`ED`** family — `EDA` (WI-1 estimation + keystone QFI benchmark), `EDB` (WI-2 Darwinism), `EDC` (WI-3 GHZ/cat), `EDD` (WI-4 common-mode), `EDE` (five benchmarks), `EDF` (review note + §19–22 staged) — recorded in the registry below and in WP-01 §15. **Opened WI-1** (`information/fisher.py` + the keystone QFI-scaling benchmark). Per the maintainer's steer, **WP-02 is held** until WI-1 is moving (to be drafted in parallel while WI-1 is in flight), since `information/fisher.py` is self-contained and not blocked by WP-02's two-mode-squeezing / `c_ops` surface.
- **Why:** The `ED` root was chosen after grepping `CHANGELOG.md`, `WORKPLAN_v0.3.md`, and `docs/gpu-dispatch-design.md` — single `A`–`Z`, doubles `AA`–`ZZ`, `BBA`–`BBE`, `RR.1`, `P.*`, Greek `β.1`–`β.3` / `δ.2` are taken; `EDA`–`EDF` had zero hits and `ED` is mnemonic (Estimation/Darwinism). Execution bias: the shared-freeze side-car already de-risks the cross-WP boundary (QFI ownership settled, bump de-duplicated, §4 timeline fallback explicit), so WI-1 need not wait for WP-02.
- **Outcome:** Landed (ratification + minting). WP-01 lifecycle → Ratified; registry populated. **Next:** implement `information/fisher.py` + tests to the strict gates (ruff / mypy --strict / pytest), commit the `WP/` scaffolding on a branch, then the keystone benchmark (which needs `ghz_state` from WI-3, pulled forward).
- **Links:** `WP/WP-01-estimation-darwinism.md` §15 (register) · registry below · `WP/REVIEW_LOG.md` (Round 2 approval).

### 2026-06-02 — Dispatch EDA complete: keystone QFI-scaling benchmark landed

- **Refs:** WP-01 · EDA · WI-1 / WI-3 · DoD-5
- **Stance:** Integrator (closing the dispatch) with a Guardian check (the decoupling proof is the No-TMC gate's evidence).
- **What:** Landed the keystone QFI-scaling benchmark — `tools/run_benchmark_qfi_scaling.py` writing `benchmarks/data/qfi_scaling/` (`report.json` + `arrays.npz` + the log–log figure), and the binding oracle `tests/regression/analytic/test_qfi_scaling.py`. The GHZ probe reaches the Heisenberg limit N² and the product probe the standard quantum limit N, reproduced to **max error 1.4e-14**. **This completes Dispatch EDA** (module + benchmark) and establishes the decoupling proof for DoD-5 (application-agnostic; textbook oracle only, zero application framing).
- **Why (the one decision worth recording):** the binding oracle was placed in a **new sibling file** `tests/regression/analytic/test_qfi_scaling.py` rather than in `test_analytic.py` as WP-01 §7 originally specified, because `test_analytic.py` is deliberately **QuTiP-free** ("they ARE the solver truth") and the QFI oracle must construct quantum states. Both sit in the `regression_analytic` tier; WP-01 §7 / §8 were updated to point at the new file. The compute-only artifact shape (`report.json` + `arrays.npz` + `plot.png`, **no** solve-based `manifest.json`) follows the `sparse_vs_dense` precedent — no cache-framework redesign, per the maintainer's scope constraint.
- **Outcome:** Landed; gates green (ruff / ruff-format / mypy --strict / 38 regression-analytic + full suite). **Consequence:** the §13 WORKPLAN dispatch-track stub is now **ready to paste**, but that edits the governed `WORKPLAN_v0.3.md`, so it awaits maintainer action. **Next:** WI-2 (EDB) Darwinism, WI-4 (EDD) common-mode, then EDF (review note + §19–22 staged into the shared v0.3 freeze).
- **Links:** `benchmarks/data/qfi_scaling/` · `tools/run_benchmark_qfi_scaling.py` · `tests/regression/analytic/test_qfi_scaling.py` · `WP/WP-01-estimation-darwinism.md` §7 / §13 / §15 · CHANGELOG `[Unreleased]` (EDA bullet).

### 2026-06-02 — WI-2 redundancy landed; recoverability deferred pending its measure-convention

- **Refs:** WP-01 · EDB · WI-2 · §5 / §20 / EDF
- **Stance:** Guardian (Design Principle 1 — conventions before code) with Architect follow-through (split EDB along the convention-certainty line).
- **What:** Landed the **redundancy** half of WI-2 — `information/redundancy.py` (`fragment_mutual_information`, `partial_information_plot`, `redundancy` R_δ), verified on the GHZ-cascade Darwinism plateau (`I(S:F) = H_S`; `R_δ = N`). **Deferred** the **recoverability** half (`recoverability.py`) until its measure-convention is ratified.
- **Why:** Redundancy rests on *standard, unambiguous* conventions (von Neumann mutual information; Zurek's R_δ deficit) — safe to code now. Recoverability does **not**: WP-01 §5 cites **five** candidate definitions (Knill–Laflamme; Schumacher–Nielsen coherent information; Petz recovery map; Fawzi–Renner; Bény–Oreshkov), and the repo's Design Principle 1 is *conventions before code*. Guessing a measure now risks rework when the §5 review note (EDF) pins §20. **Recommendation (for ratification):** the **Schumacher–Nielsen coherent information** `I_c(S⟩A) = S(ρ_A) − S(ρ_{S∪A})` as the recoverability/residual measure (clamped to ≥ 0 for the "recoverable quantum information" reading), giving exact erasure/dephasing endpoints (perfect recovery → H_S; full decoherence → 0; monotone between). Heavier alternative: a Petz / Fawzi–Renner recovery-*map* fidelity.
- **Outcome:** redundancy landed (gates green; 869 passed). recoverability open — awaiting the maintainer's measure choice (or EDF). **Next:** WI-4 (EDD) common-mode channel is convention-light and unblocked, or ratify the recoverability measure to finish EDB.
- **Links:** `src/iontrap_dynamics/information/redundancy.py` · `tests/unit/test_redundancy.py` · `WP/WP-01-estimation-darwinism.md` §4.2 / §5 · CHANGELOG `[Unreleased]` (EDB part).

### 2026-06-02 — Recoverability measure ratified (coherent information); EDB complete

- **Refs:** WP-01 · EDB · WI-2 · §20
- **Stance:** Integrator (closing EDB) with a Guardian check (convention pinned *before* the code landed — Principle 1 honoured).
- **What:** The maintainer ratified the **Schumacher–Nielsen coherent information** as the §20 recoverability measure (the recommendation from the previous entry). Implemented `information/recoverability.py`: `recoverability(...) = max(0, S(ρ_A) − S(ρ_{S∪A}))`. Factored the shared nonlinear-in-ρ helpers (`_ensure_density`, `_von_neumann_entropy_bits`, `_validate_indices`) into `information/_common.py` and refactored `fisher.py` + `redundancy.py` onto it — realising the shared-helper layer that motivated the `information/` umbrella (WP-01 §3). **This completes Dispatch EDB** (redundancy + recoverability).
- **Why:** Coherent information is the standard "recoverable quantum information" measure, lightweight (entropies only), with exact endpoints — chosen over the heavier Petz / Fawzi–Renner recovery-map fidelity. The deferral from the previous entry is now closed.
- **Outcome:** Landed; gates green (ruff / ruff-format / mypy --strict / 873 passed). EDB complete. The measure is staged for §20 and seals with the shared v0.3 freeze; it may be re-confirmed against the EDF review note's citation. **Next:** WI-4 (EDD) common-mode channel (convention-light); then EDF (review note + §19–22 / §20 staged into the freeze).
- **Links:** `src/iontrap_dynamics/information/recoverability.py` · `src/iontrap_dynamics/information/_common.py` · `tests/unit/test_recoverability.py` · CHANGELOG `[Unreleased]` (EDB) · `WP/WP-01-estimation-darwinism.md` §15.

### 2026-06-02 — Capability-surface review applied (pre-EDE stabilisation)

- **Refs:** WP-01 · WI-1…WI-4 · `WP/REVIEW_LOG.md`
- **Stance:** Guardian (stabilise the public surface before EDE multiplies it across five benchmarks) into Architect follow-through.
- **What:** Ran an adversarial review of the whole `information` / `states` / common-mode surface (four reviewers — API, naming/conventions, validation/numerics, test-hygiene; 27 findings) and applied the API / validation / hygiene fixes. **Validation:** a state-dimension check (vs `hilbert.total_dim`) in `fragment_mutual_information` / `partial_information_plot` / `redundancy` / `recoverability` (they previously mis-traced a wrong-dimension state silently — finding [2], high); non-finite rejection in `cramer_rao_bound`, `classical_fisher_information` (also empty), `linear_gaussian_fisher`, `CommonModePhase`, `cat_mode`, and `_von_neumann_entropy_bits` (NaN slipped past the `< 0` / `<= 0` guards). **Hygiene:** removed the `redundancy` double ket→ρ conversion via a private `_partial_information_from_density`; de-duplicated the `_spin_hilbert` / `_collective_jz` / `_product_plus` test helpers into `tests/_helpers.py` (added `tests` to the pytest `pythonpath`) across five files; collapsed the four `information/*.py` ruff `per-file-ignores` to one glob. **Docs:** `information/__init__` to present tense; WP-01 §3/§4.2 `_binary_entropy` → `_von_neumann_entropy_bits`; ghz/cat docstrings note the qc.py legacy convention is not adopted. Added guard tests (dim-mismatch, non-finite).
- **Why:** EDE replicates the public API and conventions across five benchmark scripts + data dirs + regression anchors, so hardening the surface first is cheaper than fixing it five-fold afterwards. Most of the 27 findings were no-ops (reviewers were told to be adversarial); the actionable ones were validation gaps where a malformed input returned a silent wrong answer instead of a clear error. **Judged no-change (correct per spec):** the QFI trajectory-evaluator signature (plural `states`) vs the single-`state` Darwinism measures, and the recoverability §5 cross-reference.
- **Outcome:** Landed; gates green (ruff / ruff-format / mypy --strict / **955 passed**). Capability surface stabilised; ready for EDE from a clean base.
- **Links:** `WP/REVIEW_LOG.md` · `tests/_helpers.py` · `src/iontrap_dynamics/information/_common.py` (`_validate_state_dim`).

### 2026-06-02 — EDE: the five remaining benchmarks landed (orchestrated)

- **Refs:** WP-01 · EDE · §7 · DoD-3
- **Stance:** Integrator (close the validation layer) with a Guardian check (decoupling grep + oracle reproduction).
- **What:** Built dispatch EDE — the five remaining §7 benchmarks — via a parallel workflow (five agents, one per benchmark, each writing its tool + `benchmarks/data/<name>/` + a `regression_analytic` anchor and self-verifying its oracle): `cfi_linear_gaussian`, `darwinism_redundancy`, `recoverability`, `ghz_cat`, `common_mode`. All five reproduce their textbook oracles (max error ≤ 1.4e-4 for the sampled common-mode variance; ≤ 1e-15 for the rest). With the keystone (EDA), all six §7 benchmarks are now landed. The shared integration was done in the main loop: ruff `per-file-ignores` for the new tools + anchors, the CHANGELOG EDE bullet, §7 / §15 status, and the gates (ruff / mypy --strict / **990 passed**).
- **Why (the one decision worth recording):** the **No-TMC decoupling grep** (§11, DoD-5) was sharpened — `arm` → `arms` (the arm-A/C/F concept), because `arm64` legitimately appears in every `report.json` platform-provenance string and would otherwise be a false positive; and two benchmark docstrings (keystone + CFI) were reworded to state the *absence* of application framing without using the literal word "TMC", so the concept-word grep returns genuinely zero hits. The compute-only artifact shape (`report.json` + `arrays.npz` + `plot.png`, no solve-based manifest) follows the `sparse_vs_dense` precedent set by the keystone.
- **Outcome:** Landed; gates green (990 passed; decoupling grep clean). WP-01 now has the full surface (WI-1…WI-4) **and** the validation evidence (six benchmarks). **Next:** EDF — the literature-review note + staging CONVENTIONS §19–22 / §20 / §22 into the shared v0.3 freeze — then the governed seal + the WORKPLAN §5.4 stub paste.
- **Links:** `tools/run_benchmark_{cfi_linear_gaussian,darwinism_redundancy,recoverability,ghz_cat,common_mode}.py` · `benchmarks/data/` · `tests/regression/analytic/` · `WP/WP-01-estimation-darwinism.md` §7 / §11 / §15.

---

### 2026-06-02 — EDE Round-7 cleanup; EDF (review note + CONVENTIONS/nav proposal) drafted

- **Refs:** WP-01 · EDE/EDF · §5 · §6 · §7 · §13
- **Stance:** Guardian (close two integrity/labelling gaps before they ossify) then Architect (stage the conventions without touching the locks).
- **What:** (1) **EDE Round-7 cleanup** — `run_benchmark_common_mode` dropped the `c = 1` overwrite that hard-set the measured difference variance to 0.0 before writing the artefact (it now reports the *measured* `c1_difference_variance_measured` and the error metric spans all `c`, so the artefact itself proves the rejection); `run_benchmark_recoverability` relabelled the Werner mixture from "dephasing" to the depolarizing-noise (Werner) family and recorded `max_error_scope` (endpoint-exact, monotone interior). WP-01 §7 rows 2–6 / §8 / §13 refreshed from planning placeholders (`test_analytic.py`, "to follow", EDA-only stub) to the as-built anchors and the landed EDA–EDE range. (2) **EDF option (a)** — drafted the additive literature-review note `docs/estimation-darwinism-review.md` (Coastline / CC BY-SA 4.0, Endorsement Marker, 18 cited sources with DOIs where available, source matrix, every definition anchored; renders under the real theme) and the maintainer-ready proposal `WP/EDF-conventions-nav-proposal.md` (the four staged CONVENTIONS §19–22 sections, the `mkdocs.yml` nav line, the seal-time header/marker/footer edits).
- **Why (the one decision worth recording):** EDF was executed as **option (a)** on the user's steer — draft the additive artefacts, **propose** the two governed edits (`CONVENTIONS.md`, `mkdocs.yml`), but **do not apply** them. The seal, the single `CONVENTION_VERSION` 0.2 → 0.3 bump, and the `WORKPLAN_v0.3.md` §5.4 paste stay maintainer-governed acts, owned by `WP/FREEZE-v0.3.md` and gated on the §4 combined-vs-WP-01-first timeline decision (taken at WP-02 ratification). The review note is the cited authority for §19–22, closing the WP-01 §5 binding rule (every convention section cites the note; every definition cites a primary source).
- **Outcome:** Additive artefacts landed; the two governed edits staged, not applied. EDF status moves *minted → review note + proposal landed; seal pending maintainer*. **Next:** maintainer seals the v0.3 freeze (bump + §19–22 + nav line + WORKPLAN §5.4), coordinated with WP-02; then the release tag.
- **Links:** `docs/estimation-darwinism-review.md` · `WP/EDF-conventions-nav-proposal.md` · `WP/FREEZE-v0.3.md` §2 / §6 · `WP/WP-01-estimation-darwinism.md` §5 / §6 / §13.

---

### 2026-06-02 — WP-02 Drafted against card `TC-iontrap-dynamics` (two-mode / motional open-system)

- **Refs:** WP-02 · serial 02 minted · card `TC-iontrap-dynamics`
- **Stance:** Architect (decompose the second card into reuse-first work items under the standing boundary discipline).
- **What:** Drafted `WP/WP-02-two-mode-motional.md` (Status: Drafted) decomposing the undetected-modes service card into seven work items WI-1…WI-7 (card F1–F7): two-mode SU(1,1) Hamiltonian (`hamiltonians.py`), two-mode squeezed-vacuum factory (`states.py`), **the pivotal** typed motional CPTP channels + `solve(channels=…)` (one new module `channels.py` + `sequences.solve` wiring), interferometric observables (`observables.py`), Lamb–Dicke regime helpers (`analytic.py`), a **consumed** probe-QFI (WP-01 `information/fisher.py`, not re-implemented), and a `ModeFrequencyDrift` systematic (`systematics/drift.py` on `modes.ModeConfig`). Only `channels.py` is new; everything else extends an existing module. Conventions §23 (two-mode squeezing / SU(1,1)) and §24 (motional CPTP channels) staged toward the shared `WP/FREEZE-v0.3.md` freeze; F2/F4/F5/F6/F7 additive under v0.2.
- **Why (the decisions worth recording):** the card §2 **service-module boundary** is lifted as WP-02's governing invariant (the No-TMC analogue) — the library owns the forward model + general primitives; the analogy map, the engineered-channel choice, native-subtraction, the resource-constrained benchmark, and claim discipline stay programme-side, enforced by a decoupling grep. The **FREEZE-v0.3 §4 timeline decision** (combined §19–24 vs WP-01-first) is surfaced as a ratification decision with a **Combined (bounded to F1+F2+F3)** recommendation per card §7 Q2. QFI is **one primitive, consumed** (FREEZE-v0.3 §5; card Q1). The `MC` dispatch family is the *proposed* leading candidate — **not minted** (codes mint at ratification, after the authoritative grep). Capability-framed slug `two-mode-motional` chosen over the application name `undetected-modes`, parallel to WP-01.
- **Outcome:** Drafted, awaiting ratification. **Decisions for ratification (§17):** F6 QFI boundary (Q1), freeze cadence / §4 timeline (Q2), upstream-vs-fork (Q3), and the slug. No dispatch codes minted, no WORKPLAN paste, no convention sealed — all ratification/maintainer acts. **Next:** maintainer ratifies → mint `MC` family, take the §4 decision, paste the WORKPLAN §5.5 stub.
- **Links:** `WP/WP-02-two-mode-motional.md` · `task cards/TC-iontrap-dynamics.md` · `WP/FREEZE-v0.3.md` §2 / §4 / §5 / §6 · `WP/WP-01-estimation-darwinism.md` (exemplar).

---

### 2026-06-02 — WP-02 Ratified; `MC` family minted; FREEZE-v0.3 §4 resolved (Combined, bounded to P0)

- **Refs:** WP-02 · `MC` family minted · FREEZE-v0.3 §4 · card `TC-iontrap-dynamics` §7 Q1–Q3
- **Stance:** Scout (the maintainer ratified all five §17 decisions and took the cross-WP timeline call).
- **What:** WP-02 moves Drafted → **Ratified** (then **In-flight**, opening with WI-3). All five §17 decisions ratified: (1) F6 QFI **consumed** from WP-01 `information/fisher.py`, no second implementation; (2) freeze cadence = **Combined v0.3, bounded to WP-02 P0** (F1+F2+F3 / WI-1–WI-3); (3) **branch-first** (`wp02-two-mode-motional` off the WP-01 branch), upstream issues after the freeze shape is stable; (4) slug `two-mode-motional`; (5) dispatch family **`MC`** (Motional Channels) minted `MCA`–`MCG` after a clear collision grep (`mcsolve` is unused in `src/`, not a code collision). Execution opens on `wp02-two-mode-motional` with **WI-3** (the `channels.py` + `solve(channels=…)` hinge), preceded by a short `solve()`-surface mapping pass per the maintainer's churn-risk caveat.
- **Why (the decision worth recording):** the **FREEZE-v0.3 §4** timeline-coupling call — the one genuine risk the shared-freeze structure carried — is resolved **Combined, bounded to P0**: §19–24 seal in one v0.3 freeze when WP-02's P0 subset lands, but P1/P2 (F4–F7) are additive and do **not** gate the seal, so they cannot hold WP-01 hostage. Rejected: WP-01-first (would decouple timing at the cost of a second v0.4 freeze) — the maintainer judged WP-01 materially complete and the seal a formality, so the cleaner one-bump/one-example/one-release path wins. WP-01's §19–22 + bump + WORKPLAN §5.4, and WP-02's §23–24 + §5.5, are all **held in escrow** for a single maintainer seal commit when P0 lands.
- **Outcome:** WP-02 In-flight; `MC` minted and registered; FREEZE-v0.3 §4 box checked. **Next:** map the `solve()` integration surface, then land WI-3 (Dispatch MCC), after which WI-1/WI-2 (MCA/MCB) populate the new module in parallel.
- **Links:** `WP/WP-02-two-mode-motional.md` §6 / §15 / §17 · `WP/FREEZE-v0.3.md` §4 · `WP/WP-01-estimation-darwinism.md` (seal-pending on the Combined track).

---

### 2026-06-02 — WI-3a landed (Dispatch MCC, part): typed motional channels + `solve(channels=…)`

- **Refs:** WP-02 · WI-3a · MCC · card F3 · **deferral: `Depolarising`**
- **Stance:** Integrator (land the pivotal solver hinge cleanly) with a Guardian check (the empty-default path must stay byte-for-byte unchanged).
- **What:** After the maintainer-requested `solve()`-surface mapping pass (clean: the `HilbertSpace` mode-operator API + QuTiP's `[op, coeff]` time-dependent precedent cover everything), landed WI-3a — `src/iontrap_dynamics/channels.py` with `AmplitudeDamping`, `Heating`, `Dephasing` typed Lindblad dissipators, and a keyword-only `channels=` argument on `sequences.solve` that builds their collapse operators and forces the `mesolve` path. Three analytic oracles reproduce (`tests/regression/analytic/test_motional_channels.py`): `⟨n̂⟩ = n₀e^{−κt}`, `⟨n̂⟩ → n̄`, dephasing coherence `⟨X̂⟩ = X₀e^{−γt/2}`. Gates green: ruff, mypy --strict (42 files), **1014 passed** (+24).
- **Why (decisions worth recording):** (1) **`channels=` empty default leaves the solver byte-for-byte unchanged** (kets still take the sesolve fast path); dissipation forces mesolve; `backend='jax'` and `solver='sesolve'` with channels raise `ConventionError` (jax does not route collapse operators; sesolve cannot carry them). (2) **Deferral — `Depolarising`:** the card F3 lists it, but a depolarising channel is canonical for finite-dimensional systems, not a single bosonic mode; rather than invent an arbitrary truncated-Fock definition, it is deferred (re-open if the programme needs a qubit-subspace depolariser). The three Lindblad dissipators cover the native-heating + dephasing models the surface needs. (3) The CPTP test surfaced the §13 Fock-convergence guard correctly tripping on heating-to-n̄ in a too-small truncation — sized the oracle Fock dims to physics.
- **Outcome:** WI-3a landed on `wp02-two-mode-motional`; §24 parameterisation referenced in docstrings (formal staging held for the combined seal). **Next:** WI-3b — optional per-channel time window via the `[op, coeff]` format + the R8 non-commuting-order test (the card F3 acceptance not yet covered by WI-3a, since constant whole-solve channels have no ordering).
- **Links:** `src/iontrap_dynamics/channels.py` · `src/iontrap_dynamics/sequences.py` (`solve`) · `tests/unit/test_channels.py` · `tests/regression/analytic/test_motional_channels.py` · `WP/WP-02-two-mode-motional.md` §5 / §15.

---

### 2026-06-02 — WI-3b landed (Dispatch MCC complete): time-windowed channels + the R8 test

- **Refs:** WP-02 · WI-3b · MCC · card F3 (R8) · **two QuTiP integrator footguns**
- **Stance:** Guardian (the R8 boundary is the card's hard requirement; get the window mechanism numerically honest, not just green).
- **What:** Each channel gained an optional `window=(t0, t1)` (half-open `[t0, t1)`), realised via QuTiP's `[op, coeff]` time-dependent format — no custom solver path. The R8 regression test (`heating@[0,T/2]` then `damping@[T/2,T]` vs swapped) proves order-dependence: the two orderings reach `⟨n̂⟩ ≈ 0.14` vs `≈ 1.73`, each matching its own closed form, separation ≈ 1.6. WI-3 (MCC) is complete; **1025 tests**.
- **Why (the two footguns worth recording — both Scout-flagged):** (1) **QuTiP keys the coefficient calling convention on the parameter NAME `args`.** A coeff written `f(t, _args)` is called with one argument and raises; it must be `f(t, args)` (matching the `cos_coeff`/`sin_coeff` house pattern in `hamiltonians.py`). (2) **The adaptive integrator skips a window that turns on after a quiescent interval.** Heating windowed to `[T/2, T]` from the ground state gave `⟨n̂⟩ ≡ 0` — across the dead first half the stepper takes one giant step and never samples the turn-on. Fix: when any channel is windowed, `solve` caps `mesolve`'s `max_step` at the output-grid resolution `min(diff(times))` (verified to catch even off-grid boundaries); the constant-channel and no-channel paths are untouched. This was a genuine silent-wrong-results footgun, not a test artefact — fixed in the library, not papered over in the test.
- **Outcome:** WI-3 complete on `wp02-two-mode-motional`. **Next (maintainer review point per the steer):** WI-3a + WI-3b together are the pivotal F3 surface — pause for review before WI-1/WI-2 (the §23 two-mode squeezing pillar).
- **Links:** `src/iontrap_dynamics/channels.py` (`_window_coefficient`, `_apply_window`) · `src/iontrap_dynamics/sequences.py` (`solve` `max_step` branch) · `tests/regression/analytic/test_motional_channels.py` (`test_r8_non_commuting_channels_are_order_dependent`).

---

### 2026-06-02 — WI-3 review-round fix: short-window skip (High) + R8 wording

- **Refs:** WP-02 · WI-3 · MCC · maintainer review (1 High, 1 Medium, 2 Low)
- **Stance:** Guardian (a silent-wrong-results bug in a solver abstraction is the worst kind; fix it properly).
- **What:** (High) The first `max_step` fix used `min(diff(times))`, which still **skipped a short window lying entirely between two output points** (maintainer probe: heating on `[0.53T, 0.54T)` with output spacing `0.1T` → `⟨n̂⟩ ≡ 0`, analytic 0.0784). Replaced with `windowed_max_step(channels, times)` (new `channels.py` helper, unit-tested): the cap is the smallest gap in the **union of the output times and every window endpoint**, so the step is bounded below the shortest boundary interval and no window can fall between steps. Added the binding regression `test_short_window_between_output_points_is_not_skipped`. (Medium) Tightened the R8 wording everywhere (`channels.py`, CHANGELOG, WP-02 §5): the test proves **temporal-schedule** order-dependence (which channel is active *when*), **not** `channels`-list order — channels sharing a window are simultaneous Lindblad terms whose list order is irrelevant. (Low ×2) `solve()` docstring now describes the landed `window=` contract (was "a later dispatch"); WP-02 §5 WI-3 row drops `Depolarising` from the deliverable (matches the logged deferral).
- **Why:** the maintainer empirically falsified the "cannot skip a window's turn-on" claim. The union-gap rule is the maintainer's suggested option (b), verified against the exact probe (0.0784) and the long-window R8 (unchanged). Augmenting the tlist alone (option a) was tried and **failed** — the integrator's step-size adaptation lags after a quiescent interval even when forced to stop at the boundary, so the step must actually be *bounded*, not merely punctuated.
- **Outcome:** fixed; 1028 tests. WI-3 (MCC) holds at complete. **Next:** WI-1/WI-2 (§23 two-mode squeezing) per the approved steer.
- **Links:** `src/iontrap_dynamics/channels.py` (`windowed_max_step`) · `src/iontrap_dynamics/sequences.py` (`solve`) · `tests/regression/analytic/test_motional_channels.py` · `tests/unit/test_channels.py`.

---

### 2026-06-03 — WI-1 + WI-2 landed (MCA, MCB): two-mode SU(1,1) squeezing; **P0 subset complete**

- **Refs:** WP-02 · WI-1/WI-2 · MCA/MCB · §23 · card F1/F2
- **Stance:** Architect (the §23 squeezing pillar) with a Guardian check (the convention must be self-consistent: factory state == Hamiltonian-evolved vacuum).
- **What:** `hamiltonians.two_mode_squeezing_hamiltonian` (`H/ℏ = ig(e^{iφ}â†b̂† − e^{−iφ}âb̂)`) and `beamsplitter_hamiltonian` (SU(2) partner), both **label-based** embedded via the `HilbertSpace` mode-operator API (Scout's steer — same discipline as `channels.py`); `states.two_mode_squeezed_vacuum(fock_dims, z) = exp(z*âb̂ − zâ†b̂†)|0,0⟩`, re-exported from the package root. Oracles green: factory per-mode `n̄ = sinh²|z|`; the Hamiltonian evolves the vacuum to `n̄ = sinh²(gτ)` with the su(1,1) difference number conserved and a state-fidelity match to the factory (`|z| = gτ`); the beamsplitter conserves total occupation. 1051 tests.
- **Why (decisions worth recording):** (1) The two-mode squeeze is **defined explicitly**, *not* delegated to `qutip.squeezing` — the latter carries a factor of ½ (`sinh²(|z|/2)`), which would contradict the card's `sinh²|z|` acceptance and the §6 single-mode parameterisation. §23 fixes the no-½ convention (`exp(z*âb̂ − zâ†b̂†)`), consistent with the Hamiltonian's `r = gτ`. (2) qutip 5 reports a two-mode ket's dims as `[[N_a, N_b], [1]]` (trivial column flattened) — the factory output is already canonical; an initial attempt to force `[[…],[1,1]]` was reverted as wrong (the test expectation was the bug, not the factory).
- **Outcome:** **The P0 freeze subset F1+F2+F3 (WI-1+WI-2+WI-3) is complete** — the FREEZE-v0.3 §4 trigger for the combined seal. **Next (conventions-before-seal):** draft the WP-02 review note `docs/two-mode-motional-review.md` (§23 + §24) and stage the §23/§24 CONVENTIONS text, then the maintainer's single combined seal (§19–24 + bump + WORKPLAN §5.4/§5.5).
- **Links:** `src/iontrap_dynamics/hamiltonians.py` (`two_mode_squeezing_hamiltonian`, `beamsplitter_hamiltonian`) · `src/iontrap_dynamics/states.py` (`two_mode_squeezed_vacuum`) · `tests/unit/test_two_mode.py` · `tests/regression/analytic/test_two_mode_squeezing.py`.

---

### 2026-06-03 — WI-1/WI-2 review-round fix: distinct-label + finite guards; phase map spelled out

- **Refs:** WP-02 · WI-1/WI-2 · MCA/MCB · maintainer review (2 Medium, 2 Low)
- **What:** (Medium) The two-mode builders accepted `mode_labels=("a","a")` (silent single-mode/number-operator physics for a *two-mode* API) and non-finite `g`/`coupling`/`phase` (Qobjs with non-finite entries that fail deep in the solver). Added a shared `_validate_two_mode_inputs` guard: identical labels → `ConventionError`, non-finite scalar → `ValueError`. Negative `g`/`coupling` is **allowed** (a legitimate π-phase-shift sign convention; documented + tested Hermitian). (Low) Spelled out the signed phase map `z = −gτ·e^{iφ}` (not just `|z| = gτ`) in the §23 docstrings + CHANGELOG, and added a **nonzero-phase** consistency regression (evolve at φ=0.7, fidelity-match the factory at `z = −gτe^{iφ}`). (Low) Corrected the factory return-dims docstring to qutip 5's `[[N_a, N_b], [1]]`.
- **Outcome:** fixed; 1058 tests. P0 subset holds at complete. **Next:** the conventions-before-seal package (§23/§24 review note + staged-conventions proposal).
- **Links:** `src/iontrap_dynamics/hamiltonians.py` (`_validate_two_mode_inputs`) · `tests/unit/test_two_mode.py` · `tests/regression/analytic/test_two_mode_squeezing.py`.

---

### 2026-06-03 — WP-02 conventions-before-seal package drafted (§23/§24 review note + proposal)

- **Refs:** WP-02 · §23/§24 · review note · combined seal (FREEZE-v0.3 §4)
- **Stance:** Architect (the EDF-equivalent for WP-02, now that the P0 subset has landed) with a Guardian check (citation audit + boundary grep before staging governed text).
- **What:** Drafted `docs/two-mode-motional-review.md` (Coastline / CC BY-SA, 13 references) anchoring §23 (explicit two-mode squeeze `S₂(z) = exp(z*âb̂ − zâ†b̂†)`, per-mode `sinh²|z|`, `z = −gτe^{iφ}`, su(1,1) Casimir, SU(2) beamsplitter) and §24 (Lindblad dissipators, windowing, temporal-schedule R8), and `WP/MC-conventions-nav-proposal.md` (staged §23/§24 text + nav line + seal edits, framed to apply **with** the EDF package in one combined seal). FREEZE-v0.3 §2/§6 and WP-02 §7 cross-links updated (TBD → real files).
- **Why (verification worth recording):** a background **citation-audit workflow** web-verified all 13 references (venue/volume/DOI/attribution) — all confirmed correct; one Low refinement applied (the explicit two-mode squeeze *operator* is Part II's contribution, Schumaker & Caves 1985 PRA 31 3093, while Part I introduces the squeezed states). The boundary-decoupling grep is clean on both the note and the proposal (no `undetected`/`analogy`/`native-subtraction`/… application framing in the to-be-sealed text). Conventions-fidelity: every note/proposal claim traces to the landed implementation.
- **Outcome:** the WP-02 conventions package is **ready for the combined maintainer seal** — §19–24 + the single `CONVENTION_VERSION` 0.2→0.3 bump + both nav lines + WORKPLAN §5.4/§5.5, in one commit. Everything governed is staged-not-applied. **Remaining WP-02 scope:** F4–F7 (P1/P2), additive and non-gating.
- **Links:** `docs/two-mode-motional-review.md` · `WP/MC-conventions-nav-proposal.md` · `WP/FREEZE-v0.3.md` §2/§4/§6 · `WP/EDF-conventions-nav-proposal.md` (the WP-01 half of the combined seal).

---

### 2026-06-03 — **v0.3 Convention Freeze SEALED** (commit `fdcd20f`): the combined §19–24 lock turn

- **Refs:** FREEZE-v0.3 · CONVENTIONS §19–24 · `CONVENTION_VERSION` 0.2 → 0.3 · WP-01 + WP-02 (combined)
- **Stance:** Guardian (the single repo-wide lock-turning event; executed only after both reviewers' line-by-line review of the prepared patch).
- **What:** The maintainer's combined seal — one commit (`fdcd20f`, 8 governed/public files): `CONVENTIONS.md` §19–22 (estimation/Darwinism) + §23–24 (two-mode/motional) pasted and frozen (freeze lines, H3 subsections, anchored to the two review notes); `CONVENTION_VERSION` "0.2" → "0.3" (`conventions.py`); the `tests/conventions/test_convention_version.py` guard; `mkdocs.yml` both review-note nav lines; `WORKPLAN_v0.3.md` §5.4 + §5.5 amendments + header/footer 0.3.5 → 0.3.7; `CHANGELOG.md` `### Changed`; README + `docs/index.md` v0.2 → v0.3. Gates green: ruff, mypy --strict, mkdocs --strict, **1059 passed**.
- **Why (the review-round decisions worth recording):** the prepared patch went through two reviewers. The maintainer found and I fixed, **in the seal commit**: (High) the §13 doc field `conventions_version` → `convention_version` (the long-standing §13/§17.5 inconsistency, FREEZE-v0.3 §3); (High) a missing `CHANGELOG ### Changed` entry for the bump; (Medium) the WORKPLAN header/footer lock-step (header was still 0.3.5); (Medium) README/`docs/index.md` still claiming v0.2. **Judgment call (staged tags):** the two reviewers split — the maintainer said **drop** the `*(staged …)*` tags from §19–24, Scout said keep (the §17/§18 precedent). The maintainer's call governs: tags dropped from §19–24, §17/§18 left untouched (a later consistency cleanup if ever).
- **Outcome:** **§19–24 frozen; `CONVENTION_VERSION` = "0.3".** WP-01 → **Released-pending** (only the release tag remains, §10); WP-02 **P0 subset SEALED**, P1/P2 (F4–F7) additive and non-gating. The FREEZE-v0.3 side-car has done its single job.
- **Links:** commit `fdcd20f` · `CONVENTIONS.md` §19–24 · `WP/FREEZE-v0.3.md` (now Sealed) · `WP/WP-01-estimation-darwinism.md` (Released-pending) · `WP/WP-02-two-mode-motional.md` (P0 sealed).

---

### 2026-06-03 — **`v0.5.0` released** (release cut) — with a tag-placement incident, corrected

- **Refs:** release v0.5.0 · WP-01 §10 release cut · `main` = `c294b5b`
- **Stance:** Integrator (cut the release) with a Guardian recovery (a mis-placed tag, caught and corrected).
- **What:** The 5-step release cut for `v0.5.0` (additive minor; package `0.4.0` → `0.5.0`): `[Unreleased]` backfill was already complete (BBA/BBB GPU + EDA–EDF + MCA–MCC + the `### Changed` bump); `pyproject` bumped; `CHANGELOG` rolled to `[0.5.0] — 2026-06-03` with a Release summary + `Test surface: 1059 passed / 3 skipped`; **Release commit `c294b5b`** (SemVer + compatibility justification); `main` fast-forwarded to it and pushed; annotated tag `v0.5.0`. Ships the estimation/Darwinism + two-mode/motional service surfaces and the GPU spectrum backend, under the v0.3 Convention Freeze. **WP-01 → Released; WP-02 P0 sealed on `main`.**
- **Why (the incident worth recording — Guardian honesty):** the annotated tag `v0.5.0` was first created **on the wrong commit** — `git tag -a` was run while checked out on `main` at its *old* pre-WP HEAD (`beb7e07`), before the fast-forward, so the published tag pointed at content with `CONVENTION_VERSION "0.2"` and none of the release. Caught by verifying `v0.5.0^{commit}` against the release commit before recording closure (the trees differed). **Root cause:** tagging before fast-forwarding `main`. **Fix (maintainer-approved):** `git tag -d v0.5.0` → recreate `git tag -a v0.5.0 c294b5b` → `git push --force origin refs/tags/v0.5.0` (low blast-radius: just-pushed tag, interim-primary repo). Verified: local + remote `v0.5.0` now deref to `c294b5b` (CONVENTION_VERSION "0.3", `pyproject` "0.5.0"); `main` == `origin/main` == `v0.5.0` == `c294b5b`. **Lesson for the next release:** fast-forward `main` first, then tag the merge point (never tag from a stale `main`).
- **Outcome:** `v0.5.0` release coherent on `main` + origin. **Next:** WP-02 F4–F7 (P1/P2, additive/non-gating) on a fresh branch off `main`.
- **Links:** tag `v0.5.0` → `c294b5b` · `CHANGELOG.md` `[0.5.0]` · `WP/WP-01-estimation-darwinism.md` (Released) · `pyproject.toml` (`0.5.0`).

---

### 2026-06-03 — WI-7 landed (Dispatch MCG): ModeFrequencyDrift — WP-02's surface complete

- **Refs:** WP-02 · WI-7 · MCG · card F7 · §18.4 · branch `wp02-f7-mode-drift` (off `main` = `e11ccf5`)
- **Stance:** Integrator (the last additive systematic) with a Guardian conventions-before-code gate (settled the §18 taxonomy placement with the maintainer before coding).
- **What:** `systematics.ModeFrequencyDrift` (frozen `delta: float`) + `apply_mode_frequency_drift(mode, drift)` → multiplicative `ω_m → ω_m·(1 + delta)` on `ModeConfig` via `dataclasses.replace`; the motional complement to the drive drifts (`RabiDrift`/`DetuningDrift`/`PhaseDrift`). Positivity (`ω_m > 0`) delegated to `ModeConfig.__post_init__` (`delta ≤ −1` → `ConventionError`, no phase-flip escape). Re-exported at all three levels (drift.py, `systematics/__init__`, package top-level). Gates green: ruff, ruff format, mypy --strict (`src`), **1077 passed** (+18), 3 skipped; §1 decoupling grep clean.
- **Conventions-before-code (the gate):** the understand fan-out confirmed §18 is **frozen** (sealed 2026-06-03) but `ModeFrequencyDrift` is a new INSTANCE of the existing §18.4 drift contract → **no new convention**, proceed. Two understand-agents split on the parameterisation; surfaced to the maintainer, who chose **multiplicative** (per the magnitude-vs-offset logic: ω_m is a positive magnitude like Ω, not an offset like a detuning — so it joins RabiDrift's multiplicative family, not DetuningDrift's additive one; trap-frequency drift is also conventionally fractional).
- **§18.4 doc-mention — DECLINED by the maintainer (governance call):** on review the maintainer ruled that `CONVENTIONS.md` §18 is frozen and post-freeze additions require a version bump (per the §18 freeze note + the Endorsement Marker), not a freestanding v0.3 doc-mention. Since `ModeFrequencyDrift` is just an instance of the existing §18.4 contract, the explanatory text lives in this LOGBOOK entry + the CHANGELOG + the `drift.py`/`systematics/__init__` docstrings; **`CONVENTIONS.md` stays untouched** (queued for a future convention bump only if it is ever judged to warrant a §18.4 entry). Good outcome of propose-don't-apply.
- **Physics:** `η = (k⃗·b⃗)·√(ℏ/(2 m ω_m)) ∝ ω_m^{−1/2}`, so a drift δ rescales η to `η₀/√(1+δ)` (magnitude only; sign `k⃗·b⃗` preserved). Sub-linear: a fractional ω drift δ gives a fractional η change ≈ −δ/2, not −δ — documented so callers don't misread δ as the η drift. Tests: zero-drift identity, the √-law (parametrised), drift∘inverse round-trip (`δ_inv = −δ/(1+δ)`), positivity raise, determinism, label/eigenvector invariance.
- **Built with two workflows:** an **understand** fan-out (drift surface, §18 verdict, ModeConfig mechanics, η-ω physics, test idioms) + a **three-lens adversarial-verify** (physics+mutation, API/contract, §18-contract/§1-boundary).
- **Review-round fixes (maintainer):** (1) **Medium — non-finite ω_m**: `ModeConfig.__post_init__` checked only `≤ 0` (NaN/inf slip past, since NaN comparisons are False), so both `ModeConfig(frequency_rad_s=nan/inf)` and a non-finite-δ drift could fabricate an invalid mode. Fixed at the **root** (`modes.py`): the invariant now requires `np.isfinite(ω_m) and ω_m > 0`, globally true for every construction path; added `test_non_finite_frequency_rejected` (ModeConfig) + `test_non_finite_delta_raises` (drift). (2) **Low** — the stale `systematics/__init__` module docstring ("drift/SPAM all in place", §18 freezes at v0.2) updated to record the MCG motional drift and the v0.3 freeze.
- **Outcome:** MCG landed + review-clean on `wp02-f7-mode-drift`. **WP-02's full WI-1…WI-7 surface is now complete** (P0 sealed in v0.5.0; P1/P2 = MCD/MCE/MCF/MCG on independent branches off `main`). **Next:** the maintainer-directed merge/release sequencing for the four P1/P2 branches into `main`.
- **Links:** `src/iontrap_dynamics/systematics/drift.py` (`ModeFrequencyDrift`, `apply_mode_frequency_drift`) · `tests/unit/test_systematics.py` (`TestModeFrequencyDrift`) · `modes.ModeConfig` · `analytic.lamb_dicke_parameter`.

---

## Dispatch-code registry *(Sail)*

The **forward** registry of dispatch codes minted under this framework (from 2026-06-02), so codes never collide and "what shipped when" is answerable in one place. A code is minted when its WP reaches **Ratified**, recorded here, and carried into a `CHANGELOG.md` `[Unreleased]` bullet at landing (the CHANGELOG remains the binding shipped record; this table is the forward index).

| Code | Title | WP | CHANGELOG / WORKPLAN carrier | Status |
|---|---|---|---|---|
| **EDA** | WI-1 estimation `information/fisher.py` + keystone QFI-scaling benchmark | WP-01 | CHANGELOG `[Unreleased]` · WORKPLAN §5.4 | **landed 2026-06-02 — EDA complete** (module + benchmark) |
| **EDB** | WI-2 Darwinism redundancy + recoverability | WP-01 | CHANGELOG `[Unreleased]` | **landed 2026-06-02 — EDB complete** |
| **EDC** | WI-3 `states.ghz_state` + `cat_mode` | WP-01 | CHANGELOG `[Unreleased]` | landed 2026-06-02 |
| **EDD** | WI-4 `systematics/common_mode.py` | WP-01 | CHANGELOG `[Unreleased]` | landed 2026-06-02 |
| **EDE** | five generic benchmarks under `benchmarks/data/` | WP-01 | CHANGELOG `[Unreleased]` | **landed 2026-06-02** |
| **EDF** | review note + CONVENTIONS §19–22 staged for the shared v0.3 freeze | WP-01 | CHANGELOG `[Unreleased]` + `WP/FREEZE-v0.3.md` | **SEALED 2026-06-03** (commit `fdcd20f`): `docs/estimation-darwinism-review.md` + §19–22 frozen, `CONVENTION_VERSION` 0.2 → 0.3 (combined with WP-02 §23–24) |
| **MCA** | WI-1 two-mode SU(1,1) squeezing Hamiltonian (`hamiltonians.py`) | WP-02 | CHANGELOG `[Unreleased]` · WORKPLAN §5.5 | **landed 2026-06-03** (n̄ = sinh²(gτ); Casimir; + beamsplitter) |
| **MCB** | WI-2 `states.two_mode_squeezed_vacuum` | WP-02 | CHANGELOG `[Unreleased]` | **landed 2026-06-03** (per-mode n̄ = sinh²\|z\|; Schmidt) |
| **MCC** | WI-3 typed motional CPTP channels + `solve(channels=…)` (new `channels.py`) | WP-02 | CHANGELOG `[Unreleased]` | **WI-3 complete 2026-06-02** (WI-3a dissipators + solver wiring; WI-3b time windows + R8 test + `windowed_max_step` union-gap fix; review-round short-window-skip fix; 1028 tests) |
| **MCD** | WI-4 interferometric observables (visibility, fringe phase) | WP-02 | CHANGELOG `[Unreleased]` | minted 2026-06-02; P1, open |
| **MCE** | WI-5 Lamb–Dicke regime helpers (`analytic.py`) | WP-02 | CHANGELOG `[Unreleased]` | minted 2026-06-02; P1, open |
| **MCF** | WI-6 probe-QFI benchmark (**consumes** WP-01 `fisher.py`) | WP-02 | CHANGELOG `[Unreleased]` | minted 2026-06-02; P1, open |
| **MCG** | WI-7 `ModeFrequencyDrift` (`systematics/drift.py`) | WP-02 | CHANGELOG `[Unreleased]` | **landed 2026-06-03** (`wp02-f7-mode-drift`; multiplicative ω-drift on `ModeConfig`, §18.4 contract; η₀/√(1+δ); finite+positive `ω_m` enforced; §18.4 doc-mention declined—CONVENTIONS frozen) |

**This registry is forward-only; it does not catalogue history — that is the CHANGELOG's job.** Many families are already taken by pre-framework dispatches and reservations, and a new code must avoid all of them. **Minting rule:** before minting, grep `CHANGELOG.md`, `WORKPLAN_v0.3.md`, and `docs/gpu-dispatch-design.md` for the candidate family. Known-taken / reserved as of 2026-06-02 (**non-exhaustive** — the grep is authoritative): single letters `A`–`Z`; doubles `AA`–`WW` (incl. tutorial `AA`–`LL`, Phase-2 `OO`, `QQ`–`ZZ`); triples `BBA` / `BBB` landed and `BBC`–`BBE` reserved for the GPU track (`docs/gpu-dispatch-design.md`); sub-coded `RR.1`, `P.*`; Greek `β.1`–`β.4`; **WP families `ED` (WP-01, `EDA`–`EDF`) and `MC` (WP-02, `MCA`–`MCG`, minted 2026-06-02)**. A new WP mints from a clearly-fresh family chosen *after* that grep — not merely "anything but `BBA` / `BBB`".

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This logbook is a Sail narrative within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg), kept under the Coastline gates of `WORKPLAN_v0.3.md` and `CONVENTIONS.md`. Lock–Key rule applies: the logbook records keys (decisions, executions) built on the stable locks specified in the Coastline docs. The repository adopts the T(h)reehouse +EC Corporate Design blueprint (`cd-rules`, consumed via Model B).

**Council status:** Guardian cleared (logbook records, never relaxes, gates; CHANGELOG remains the binding shipped record). Architect approved (logbook = dated narrative carrier, distinct from CHANGELOG outcome record, WORKPLAN spec, and per-card WP). Scout horizon signals addressed (dead-ends, deferrals, and null results now have a home; dispatch-code registry prevents collision). Integrator has sequenced the entry hooks across the WP lifecycle.

**Convention version:** references `CONVENTIONS.md` v0.2 (frozen 2026-04-21).
**Corporate design version:** `cd-v1.7.1` (consumed via Model B).
**Workplan reference:** `WORKPLAN_v0.3.md` v0.3.5; new tracks land as append-only `§5.x` amendments.
