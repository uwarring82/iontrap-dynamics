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

---

## Dispatch-code registry *(Sail)*

The **forward** registry of dispatch codes minted under this framework (from 2026-06-02), so codes never collide and "what shipped when" is answerable in one place. A code is minted when its WP reaches **Ratified**, recorded here, and carried into a `CHANGELOG.md` `[Unreleased]` bullet at landing (the CHANGELOG remains the binding shipped record; this table is the forward index).

| Code | Title | WP | CHANGELOG / WORKPLAN carrier | Status |
|---|---|---|---|---|
| **EDA** | WI-1 estimation `information/fisher.py` + keystone QFI-scaling benchmark | WP-01 | CHANGELOG `[Unreleased]` (at landing) · WORKPLAN §5.4 | minted · WI-1 open |
| **EDB** | WI-2 Darwinism redundancy + recoverability | WP-01 | CHANGELOG `[Unreleased]` (at landing) | minted |
| **EDC** | WI-3 `states.ghz_state` + `cat_mode` | WP-01 | CHANGELOG `[Unreleased]` (at landing) | minted |
| **EDD** | WI-4 `systematics/common_mode.py` | WP-01 | CHANGELOG `[Unreleased]` (at landing) | minted |
| **EDE** | five generic benchmarks under `benchmarks/data/` | WP-01 | CHANGELOG `[Unreleased]` (at landing) | minted |
| **EDF** | review note + CONVENTIONS §19–22 staged for the shared v0.3 freeze | WP-01 | CHANGELOG `[Unreleased]` + `WP/FREEZE-v0.3.md` | minted |

**This registry is forward-only; it does not catalogue history — that is the CHANGELOG's job.** Many families are already taken by pre-framework dispatches and reservations, and a new code must avoid all of them. **Minting rule:** before minting, grep `CHANGELOG.md`, `WORKPLAN_v0.3.md`, and `docs/gpu-dispatch-design.md` for the candidate family. Known-taken / reserved as of 2026-06-02 (**non-exhaustive** — the grep is authoritative): single letters `A`–`Z`; doubles `AA`–`WW` (incl. tutorial `AA`–`LL`, Phase-2 `OO`, `QQ`–`ZZ`); triples `BBA` / `BBB` landed and `BBC`–`BBE` reserved for the GPU track (`docs/gpu-dispatch-design.md`); sub-coded `RR.1`, `P.*`; Greek `β.1`–`β.4`. A new WP mints from a clearly-fresh family chosen *after* that grep — not merely "anything but `BBA` / `BBB`".

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This logbook is a Sail narrative within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg), kept under the Coastline gates of `WORKPLAN_v0.3.md` and `CONVENTIONS.md`. Lock–Key rule applies: the logbook records keys (decisions, executions) built on the stable locks specified in the Coastline docs. The repository adopts the T(h)reehouse +EC Corporate Design blueprint (`cd-rules`, consumed via Model B).

**Council status:** Guardian cleared (logbook records, never relaxes, gates; CHANGELOG remains the binding shipped record). Architect approved (logbook = dated narrative carrier, distinct from CHANGELOG outcome record, WORKPLAN spec, and per-card WP). Scout horizon signals addressed (dead-ends, deferrals, and null results now have a home; dispatch-code registry prevents collision). Integrator has sequenced the entry hooks across the WP lifecycle.

**Convention version:** references `CONVENTIONS.md` v0.2 (frozen 2026-04-21).
**Corporate design version:** `cd-v1.7.1` (consumed via Model B).
**Workplan reference:** `WORKPLAN_v0.3.md` v0.3.5; new tracks land as append-only `§5.x` amendments.
