# WP — Work-Plan management

**Per-card execution plans for `iontrap-dynamics`, governed under Coastline gates**

Version 0.1 · Drafted 2026-06-02 · Status: live; seeded with WP-01 against task card TC-ITD-ESTDARW-01 (Estimation & Darwinism service upgrade)

**Classification:** Sail execution under Coastline gates (per T(h)reehouse +EC CD 0.9). This README and the WP framework it describes are *Sail* (adaptive guidance); the gates a WP must pass — CONVENTIONS freeze, CHANGELOG, licence split — are *Coastline* and are not negotiable from inside a WP.
**Licence:** CC BY-SA 4.0 — `WP/` planning material is project-governance on the Coastline track; the per-folder grant is declared in `WP/LICENCE`, consistent with root `LICENCE` (the root split table should gain a `WP/` row at maintainer ratification).
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (`cd-rules`, consumed via Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

---

## 1. What a WP is — and what it is not *(Sail)*

A **Work-Plan (WP)** is a single, self-contained **execution plan** for one piece of work: how a task card gets turned into landed dispatches. One task card in → one WP → one or more dispatches out → CHANGELOG entries → a release tag. The WP is the working surface where scope is decomposed into work items, acceptance criteria are pinned, and progress is tracked while the work is in flight.

A WP **is**:

- a **decomposition** of one incoming brief into ordered work items (WI-1, WI-2, …), each with an acceptance test;
- the **execution-level home** for sequencing, blockers, and the dispatch-code minting for that work;
- a **live document** that is edited in place while in flight (unlike WORKPLAN amendments, which are append-only — see §4);
- **Sail**: authored, adaptive, revisable. It is guidance for getting the work done, not a binding spec.

A WP **is not**:

- **not the strategic roadmap.** `WORKPLAN_v0.3.md` (the *Coastline roadmap*) owns phase milestones, release-mapping, and architectural constraints. A WP never re-scopes a phase, never changes a convention, never relicenses anything. When a WP and the WORKPLAN disagree, the WORKPLAN wins and the WP is wrong.
- **not the incoming brief.** The brief lives in `task cards/`. The card states *what* is wanted and *why*; the WP states *how* it will be built. A WP is downstream of exactly one card.
- **not the shipped-change record.** `CHANGELOG.md` (Keep-a-Changelog, dispatch-indexed) owns *what shipped*. A WP plans; the CHANGELOG records the surface delta after the fact.
- **not the development logbook.** `WP/LOGBOOK.md` owns the dated *narrative*: decisions, rejected options, dead-ends, deferrals, release-cut events. A WP holds the current plan; the logbook holds the chronology of why it changed. See §5.
- **not a convention.** Conventions live in `CONVENTIONS.md` (frozen at v0.2). A WP that needs a new convention must stop and route that through a CONVENTIONS bump, outside the WP.

The one-line test: **if it is binding and testable, it belongs in a Coastline doc (WORKPLAN / CONVENTIONS / CHANGELOG); if it is "how I am going to do this card", it belongs in a WP; if it is "what I decided and why, on which date", it belongs in the logbook.**

## 2. Separation of concerns *(Sail)*

The five carriers and their non-overlapping jobs:

| Carrier | Class | Owns | Mutability | Answers |
|---|---|---|---|---|
| `WORKPLAN_v0.3.md` | Coastline | Strategic roadmap: phases, release-mapping, architectural constraints, dispatch-track stubs | Append-only amendments (`§5.x`, `new in v0.3.N`) | "What is the plan of record?" |
| `task cards/*.md` (ID `TC-…`) | brief (external) | The incoming request: context, objective, scope in/out, acceptance | Frozen on receipt (the card is a handoff) | "What was asked, and why?" |
| `WP/WP-NN-slug.md` | Sail | Per-card execution plan: work items, sequencing, acceptance, dispatch minting | Live; edited in flight | "How is this card being built?" |
| `CHANGELOG.md` | Coastline | Shipped surface deltas, dispatch-indexed, per-release summary + test counts | Append under `[Unreleased]`, rolled at tag | "What shipped, and when?" |
| `WP/LOGBOOK.md` | Sail (narrative) | Dated decisions, rationale, rejected options, dead-ends, deferrals, release-cut events, dispatch-code registry | Append-only, chronological | "Why did it go the way it did?" |

The discipline: **no carrier duplicates another.** A WP that starts narrating decisions chronologically has drifted into logbook territory; move it. A logbook entry that lists every shipped function has drifted into CHANGELOG territory; trim it to the *why*. The WORKPLAN that starts listing work items has absorbed a WP; pull them back out.

## 3. WP lifecycle *(Sail)*

A WP moves through five states. The current state is recorded in the WP's own version line (the `Status:` field).

```
Drafted ──▶ Ratified ──▶ In-flight ──▶ Released ──▶ Archived
```

1. **Drafted.** The WP exists, decomposes the card into work items, but is not yet committed to. Scope may still move. No dispatch codes minted yet.
2. **Ratified.** The maintainer has accepted the decomposition. Work items, acceptance criteria, and the dispatch-track stub in the WORKPLAN are locked enough to begin. A logbook entry records ratification (date, WP ref). Dispatch codes may now be minted.
3. **In-flight.** Dispatches are landing on `main`. Each landed dispatch gets a CHANGELOG `[Unreleased]` bullet and, if it carried a decision/dead-end/deferral, a logbook entry. The WP's work-item table tracks per-WI status (open / landed / deferred).
4. **Released.** All work items are landed or explicitly deferred, and the work has been rolled into a tagged release. The release-cut event (5-step procedure, SemVer justification) is logged in `WP/LOGBOOK.md`. The WP `Status:` records the release tag.
5. **Archived.** The WP is complete and historical. It moves to `WP/archive/` with a dated closing note (deprecation-not-deletion, CD 0.8). It stays readable; git history is not a substitute for visible archival.

A WP is never deleted and never silently rewritten once **Ratified** — scope changes after ratification are append-only notes inside the WP plus a logbook entry, mirroring the WORKPLAN's honesty discipline (originally-planned text is read through, not overwritten).

## 4. Naming and structure *(Sail)*

**File name:** `WP/WP-NN-slug.md`, where

- `NN` is a **zero-padded two-digit serial** assigned in order of ratification (`WP-01`, `WP-02`, …). Serials are minted here and recorded in the logbook to prevent collisions; they are not derived from the card ID.
- `slug` is a short, hyphenated, lower-case mnemonic (`WP-01-estimation-darwinism`). British spelling in slugs.

**No version numbers anywhere else in the filename** (CD 0.6 / Principle 14): the internal `Status:` line and the logbook carry the lifecycle state; the filename is stable for the life of the WP.

**Permitted `WP/` documents.** The folder holds: per-card execution plans `WP-NN-slug.md`; this `README.md`; `TEMPLATE.md`; the `LOGBOOK.md`; the folder `LICENCE`; and **thematic planning side-cars** — cross-cutting plans not tied to a single card (e.g. `FAIR.md`, the FAIR4RS action plan; `FREEZE-v0.3.md`, the shared Convention-Freeze coordination that more than one WP feeds). A side-car carries the same governed header but is *not* a WP (no card, no work items, no dispatch register); it is governance/planning material under the same CC BY-SA 4.0 (`WP/LICENCE`). A side-car is the right home for any concern that is **shared across WPs** (a repo-wide freeze, a metadata initiative) and therefore cannot belong to one card's WP.

**Section skeleton** (every WP mirrors this; see `WP/TEMPLATE.md` for the annotated form):

1. Header block (governed; see §6)
2. **Card linkage** — which `task cards/*.md` (internal ID `TC-…`) this WP executes, named by literal path, and the governing invariants lifted from it *(Sail)*
3. **WORKPLAN linkage** — the dispatch-track stub in `WORKPLAN_v0.3.md` this WP feeds, the next free `§5.x` *(Coastline gate)*
4. **Objective and scope (in / out)** — execution-terms objective; scope inherited from the card and sharpened for execution *(Sail)*
5. **Work items** — table: WI-N · module · key contents · reuse · acceptance · dispatch code(s) · status *(Sail)*
6. **Sequencing and gates** — order, blockers, the Coastline gates each WI must clear (CONVENTIONS freeze, SPDX, tests, CHANGELOG, CI) *(Coastline gate)*
7. **Dispatch register** — codes minted by this WP, mapped to WI and CHANGELOG bullet *(Sail)*
8. **Release plan** — target tag, SemVer justification, the 5-step release cut *(Coastline gate)*
9. **Logbook hooks** — which `WP/LOGBOOK.md` entries this WP has generated *(Sail)*
10. **Footer Endorsement Marker** (governed; see §6)

These eight middle sections (2–9) are the **minimum every WP carries, in this order** — the spine of `WP/TEMPLATE.md`. A WP may add sections and may be *constraint-heavy* (most of its sections tagged `*(Coastline)*` / `*(Coastline gate)*` rather than `*(Sail)*`); that does **not** change its document-level class, which is always **Sail execution under Coastline gates** (§6) — a WP states and clears gates, it never relaxes them. It must still carry the spine. WP-01 is the worked example: it expands the spine to sixteen sections (adding reuse-posture, the naming decision, separate literature-review / convention-freeze / benchmark / test / docs sections, and the WORKPLAN stub), and carries the spine's Sequencing / Dispatch-register / Logbook-hooks as its §14 / §15 / §16. `WP/TEMPLATE.md` is the annotated canonical form to copy.

## 5. Cross-linking *(Sail)*

Every WP threads three explicit links, so the five carriers stay navigable:

- **WP → task card.** The WP names the card by ID **and literal path**: *"Executes `task cards/task-card-iontrap-dynamics-service-upgrade.md` (ID `TC-ITD-ESTDARW-01`)."* Governing invariants from the card (e.g. *application-agnostic: no consuming-application framing in library symbols*) are quoted, not paraphrased.
- **WP → WORKPLAN dispatch-track stub.** A new track lands in the roadmap as a **new numbered amendment under `§5`** (the next free `§5.x`, bumping the doc to the next patch, e.g. `new in v0.3.6`), following the `§5.3` template: header with date + `*(Coastline, new in v0.3.N)*`, an anchoring sentence ("Added when Dispatch X landed on `main` —"), a scope statement, a **Rationale**, an **"On `main` toward <track>"** status line, a **"Remaining sub-dispatches"** list, and a **"Consequence for §5 above"** read-through. The WP §3 points at that stub; the stub points back at the WP. Lock-step: the WORKPLAN header version line, footer `**Workplan version:**`, and Endorsement Marker are bumped in the same commit as the amendment.
- **WP → logbook.** Every ratification, decision, dead-end, deferral, dispatch-code minting, and release-cut for this WP gets a dated `WP/LOGBOOK.md` entry tagged with the WP ref (`WP-01`) and dispatch code where relevant. The WP §9 lists those entry dates.

Backlinks are reciprocal: the logbook entry names the WP; the WP names the entry; the WORKPLAN stub names both.

## 6. Classification and governed header *(Coastline gate)*

A WP is **Sail** (authored execution) but it operates **under Coastline gates** that it cannot relax:

- **CONVENTIONS freeze.** No WP introduces a convention. New physical/numerical conventions require a `CONVENTIONS.md` version bump, decided outside the WP. A WP that hits a missing convention stops and logs the blocker.
- **Split-licence architecture** (CD 0.3). Code and tooling a WP lands are **MIT** (`src/`, `tests/`, `.github/workflows/`); authored docs/tutorials are **Sail / CC BY-NC-SA 4.0**; specs, schemas, and CONVENTIONS edits are **Coastline / CC BY-SA 4.0**. The `WP/` planning documents themselves (plans, this README, TEMPLATE, LOGBOOK, side-cars) are **CC BY-SA 4.0** governance material per `WP/LICENCE` — their in-document `*(Sail)*` / `*(Coastline)*` tags mark which *lines* are revisable vs binding, and do not change the folder licence. Note British spelling **`LICENCE`** throughout. Every new module carries its SPDX header.
- **CHANGELOG discipline.** Every landed dispatch gets a Keep-a-Changelog `[Unreleased]` bullet, dispatch-keyed.
- **Endorsement Marker** top and foot (CD 0.7).

Each WP section heading carries an inline italic class tag exactly as the WORKPLAN does, so a reader can see at a glance what each section is: `*(Sail)*` revisable guidance · `*(Coastline)*` a binding constraint stated in the WP · `*(Coastline gate)*` a hard external gate (CI, the CONVENTIONS freeze, the licence split). A constraint-heavy WP carries many of the latter two; it still cannot relax any of them.

## 7. Receiving a new card — the quick path *(Sail)*

The maintainer expects more cards. The pragmatic loop:

1. Card lands in `task cards/*.md` (each carrying an internal `TC-…` ID). Read it; lift the governing invariants.
2. Copy `WP/TEMPLATE.md` to `WP/WP-NN-slug.md`, mint the next serial, fill card + WORKPLAN linkage. **Status: Drafted.** Log the draft.
3. Decompose into work items with acceptance tests. Ratify → **Status: Ratified.** Log it. Mint dispatch codes. Add the `§5.x` dispatch-track stub to `WORKPLAN_v0.3.md`.
4. Land dispatches → **In-flight.** CHANGELOG `[Unreleased]` per dispatch; logbook per decision/dead-end.
5. Cut the release (5-step procedure, logged) → **Released**, then **Archived**.

Keep it light. The framework exists to stop decisions becoming archaeology, not to add ceremony.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This Work-Plan framework is a Sail execution layer within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg), operating under the Coastline gates of `WORKPLAN_v0.3.md` and `CONVENTIONS.md`. Lock–Key rule applies: the WORKPLAN and CONVENTIONS specify the stable locks; a WP is a key built on top of them. The repository adopts the T(h)reehouse +EC Corporate Design blueprint (`cd-rules`, consumed via Model B).

**Council status:** Guardian cleared (a WP cannot relax a Coastline gate, relicense, or alter a convention; honesty discipline carried over from the WORKPLAN). Architect approved (five-carrier separation of concerns; WP = Sail execution, logbook = dated narrative, CHANGELOG = shipped delta, no duplication). Scout horizon signals addressed (more cards expected; template + serial registry prevent code/serial collision; archival is visible, not git-only). Integrator has sequenced the WP lifecycle: Drafted → Ratified → In-flight → Released → Archived, with logbook hooks at each transition.

**Convention version:** references `CONVENTIONS.md` v0.2 (frozen 2026-04-21 at the `v0.2.0` release). A WP does not alter this; new conventions require a separate bump.
**Corporate design version:** `cd-v1.7.1` (decision D2 closed 2026-04-23, consumed via Model B).
**Workplan reference:** `WORKPLAN_v0.3.md` v0.3.5; new dispatch tracks land as append-only `§5.x` amendments. This WP framework is Sail and does not amend the WORKPLAN's version.
