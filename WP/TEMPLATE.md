<!--
  WP/TEMPLATE.md — reusable skeleton for a Work-Plan.
  HOW TO USE:
    1. Copy this file to WP/WP-NN-slug.md (mint the next free two-digit serial NN; record it in WP/LOGBOOK.md).
    2. Fill every <…> placeholder and delete the guidance comments before ratifying.
    3. Keep section class tags *(Sail)* / *(Coastline gate)* exactly — a reader uses them to see what is revisable.
    4. A WP is Sail (authored execution) under Coastline gates it may NOT relax. See WP/README.md §1, §6.
    5. British (Oxford) spelling throughout: Licence, behaviour, normalise, artefacts, -ise not -ize.
-->

# WP-NN — <Work-Plan title>

**<bold subtitle: one line on what this WP executes>**

Version 0.1 · Drafted YYYY-MM-DD · Status: Drafted
<!-- Status field carries the lifecycle state and is the single source of truth for it:
     Drafted → Ratified (date) → In-flight → Released (vX.Y.Z, date) → Archived (date).
     On each transition, edit this line AND append a WP/LOGBOOK.md entry. -->

**Classification:** Sail execution under Coastline gates (per T(h)reehouse +EC CD 0.9).
**Licence:** This WP document itself is CC BY-SA 4.0 (`WP/LICENCE`). The deliverables it plans carry their own layer's licence: code is MIT (`src/`, `tests/`, `.github/workflows/`); authored docs/tutorials are Sail / CC BY-NC-SA 4.0; any spec/schema/CONVENTIONS edit is Coastline / CC BY-SA 4.0. See root `LICENCE`.
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (`cd-rules`, consumed via Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

---

## 1. Card linkage *(Sail)*

Executes `task cards/<card-file>.md` (**ID: TC-<…>**, v<…>, <date>).

**Objective lifted from the card (one line):** <…>

**Governing invariants from the card — quoted, not paraphrased** (these are hard acceptance gates):

> <e.g. "application-agnostic: no consuming-application framing in any library symbol">

<!-- Lift every invariant the card declares binding. If the card lists explicit OUT-of-scope items, copy them to §3 verbatim. -->

## 2. WORKPLAN linkage *(Coastline gate)*

This WP feeds a new dispatch track recorded in `WORKPLAN_v0.3.md` as the next free amendment **§5.x** (`*(Coastline, new in v0.3.N)*`).

- **Amendment §:** <§5.x — to be assigned>
- **Doc version bump:** v0.3.<N-1> → v0.3.<N> (one patch per amendment subsection)
- **Stub follows the §5.3 template:** anchoring sentence ("Added when Dispatch <X> landed on `main` —"), scope statement (blocks / does not block which release), **Rationale**, **"On `main` toward <track>"** status line, **"Remaining sub-dispatches"** list, **"Consequence for §5 above"** read-through.
- **Lock-step:** the WORKPLAN header version line, footer `**Workplan version:**`, and Endorsement Marker are bumped in the same commit as the amendment.

<!-- The WORKPLAN owns the strategic stub; THIS WP owns the execution detail. Do not duplicate work-item tables into the amendment. -->

## 3. Objective and scope *(Sail)*

**Objective (execution terms):** <one paragraph — what gets built and what "done" looks like>

**In scope.** <the capabilities this WP delivers>

**Out of scope (explicitly).** <copied from the card; name what belongs to a consuming application or a different repo>

## 4. Work items *(Sail)*

<!-- One row per atomic, separately-acceptable unit. "Reuse" cites existing modules (Design Principle: reuse before adding). "Acceptance" must be a runnable test or a reproduced oracle. "Dispatch" is minted at Ratified, recorded in the logbook registry. -->

| WI | Module (proposed) | Key contents | Reuse | Acceptance | Dispatch | Status |
|---|---|---|---|---|---|---|
| **WI-1** | `<path/module.py>` | <…> | `<existing module>` | <runnable test / reproduced result> | `<CODE>` | open |
| **WI-2** | `<…>` | <…> | <…> | <…> | `<CODE>` | open |

<!-- Status values: open / in-flight / landed / deferred. A deferred WI gets a WP/LOGBOOK.md entry stating why and what re-opens it — deferrals never vanish silently. -->

## 5. Sequencing and gates *(Coastline gate)*

**Order:** <WI-1 → WI-2 → … with dependencies stated>

**Blockers:** <anything not yet resolved; link the logbook entry tracking it>

**Coastline gates every WI must clear before it counts as landed:**

- [ ] **CONVENTIONS freeze respected** — no new convention introduced; if one is needed, stop and route a `CONVENTIONS.md` bump (log the blocker).
- [ ] **SPDX header** on every new module; licence matches the split (code = MIT).
- [ ] **Tests** — unit + regression; benchmark reproduces its stated oracle within declared tolerance.
- [ ] **CHANGELOG** — a dispatch-keyed `[Unreleased]` bullet (`- **Dispatch <CODE> — <title>.**`).
- [ ] **CI green** — ruff, ruff-format, mypy strict, pytest, accessibility gate where docs change.

## 6. Dispatch register *(Sail)*

<!-- Mint codes at Ratified. Mirror them into WP/LOGBOOK.md's dispatch-code registry to prevent collision across WPs.
     Codes are opaque letter families (e.g. CCA, CCB …) or Greek sub-tracks (e.g. δ.1) — assigned here, not derived. -->

| Dispatch | Maps to | CHANGELOG bullet | Status |
|---|---|---|---|
| `<CODE>` | WI-<n> | `- **Dispatch <CODE> — <title>.**` | planned |

## 7. Release plan *(Coastline gate)*

Target release tag: **v<X.Y.Z>** — <theme>. SemVer per the repo's adopted convention (minor for additive capability, patch for fixes; nothing below v1.0 is removed without a one-line CHANGELOG justification).

**Release-cut (5-step, logged in WP/LOGBOOK.md):**

1. Backfill `[Unreleased]` so every shipped dispatch has an entry.
2. Bump `pyproject.toml` `version = "…"`.
3. Roll `[Unreleased]` into `[X.Y.Z] — DATE` with a **Release summary.** paragraph + **Test surface at `vX`:** line.
4. Commit `Release vX.Y.Z — <theme>`; body records the explicit SemVer decision + justification and an "Unchanged from vPrev" compatibility statement.
5. Annotated git tag.

## 8. Logbook hooks *(Sail)*

Entries this WP has generated in `WP/LOGBOOK.md` (dated):

- YYYY-MM-DD — <ratification / decision / dead-end / deferral / release-cut> — <one line>

<!-- At minimum: one entry at Ratified, one per decision-with-rejected-options, one per dead-end/deferral, one at release-cut. -->

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This Work-Plan is a Sail execution document within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg), under the Coastline gates of `WORKPLAN_v0.3.md` and `CONVENTIONS.md`. Lock–Key rule applies: this WP is a key built on the stable locks those documents specify. The repository adopts the T(h)reehouse +EC Corporate Design blueprint (`cd-rules`, consumed via Model B).

**Council status:** Guardian <cleared / pending: confirm no Coastline gate is relaxed, no convention altered, no relicensing>. Architect <approved / pending: confirm work items reuse existing machinery and respect the three-layer architecture>. Scout <horizon signals: list any deferrals / cross-repo touchpoints>. Integrator <sequenced: state the WI order and the release target>.

**Convention version:** references `CONVENTIONS.md` v0.2 (frozen 2026-04-21). This WP introduces no convention.
**Corporate design version:** `cd-v1.7.1` (consumed via Model B).
**Workplan reference:** `WORKPLAN_v0.3.md` v0.3.5; this WP's track lands as amendment §5.<x> (`new in v0.3.<N>`).
