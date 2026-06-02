# WP Framework Review Log

## Round 1 — 2026-06-02

Scope reviewed: untracked `WP/` framework files, untracked `task cards/` files, and the existing repo files they reference (`WORKPLAN_v0.3.md`, `CHANGELOG.md`, `LICENCE`, `pyproject.toml`, package exports, systematics/state helpers).

### Findings

#### High — WP-01 does not yet instantiate the framework it is meant to exercise

`WP/README.md:43`, `WP/README.md:74`, and `WP/README.md:87` define a WP as a Sail execution plan that mirrors the `WP/TEMPLATE.md` spine. `WP/TEMPLATE.md:27` through `WP/TEMPLATE.md:108` then requires explicit card linkage, WORKPLAN linkage, work items, gates, dispatch register, release plan, and logbook hooks.

`WP/WP-01-estimation-darwinism.md:7` instead classifies the whole WP as Coastline, and its section layout does not carry the template sections in order. In particular, WP-01 has no explicit dispatch-register table and no logbook-hooks section; card linkage and WORKPLAN linkage are mostly in the header and the §13 paste block.

**Comment:** either make WP-01 conform to the Sail-under-Coastline-gates template, or change `WP/README.md` / `WP/TEMPLATE.md` to admit "constraint-heavy WP" exceptions. As written, the seeded WP contradicts the framework's central rule.

#### High — Lifecycle state is ambiguous before WI-1 opens

`WP/README.md:57` through `WP/README.md:59` says dispatch codes are minted after the WP is Ratified, and `WP/README.md:115` through `WP/README.md:116` says the quick path is Drafted -> Ratified -> mint codes -> add the WORKPLAN stub. `WP/LOGBOOK.md:77` through `WP/LOGBOOK.md:81` repeats that codes are minted when the WP reaches Ratified.

`WP/WP-01-estimation-darwinism.md:5` still says `Status: drafted, pre-execution`, but also says §3 naming is ratified and WP-01 is ready for WI-1. `WP/LOGBOOK.md:70` says WI-1 may open, while the dispatch registry still has no real codes. The §13 stub then still carries `Dispatch <C1>` placeholders at `WP/WP-01-estimation-darwinism.md:297` and `WP/WP-01-estimation-darwinism.md:317`.

**Comment:** decide whether only the naming decision is ratified, or the whole WP is Ratified. If the whole WP is not Ratified, WI-1 should probably remain blocked. If it is Ratified, mint the dispatch codes and fill the registry before calling the stub paste-ready.

#### High — WI-4 API cannot satisfy its own correlation acceptance criteria

`WP/WP-01-estimation-darwinism.md:108` through `WP/WP-01-estimation-darwinism.md:110` proposes `CommonModePhase` with `sigma_rad` and `perturb_common_mode(...)`. But `WP/WP-01-estimation-darwinism.md:111` introduces a correlation parameter, and `WP/WP-01-estimation-darwinism.md:112` requires reduction to independent `PhaseDrift` / `PhaseJitter` at zero correlation plus full-correlation common-mode cancellation.

**Comment:** the proposed API has no `correlation` / `rho` / `shared_fraction` field, so zero-correlation behaviour cannot be represented. Either narrow WI-4 to a fully shared common-mode channel only, or add the correlation parameter to the spec and tests.

#### Medium — Task-card path linkage is still not concrete in WP-01

`WP/README.md:93` says every WP names the card by ID and path. The actual received card is `task cards/task-card-iontrap-dynamics-service-upgrade.md` with ID `TC-ITD-ESTDARW-01` at `task cards/task-card-iontrap-dynamics-service-upgrade.md:3`.

`WP/WP-01-estimation-darwinism.md:11` names the ID and stream, but not the actual card path. Separately, `WP/README.md:42` and `WP/README.md:93` imply `task cards/TC-*.md`, which does not match this card's filename.

**Comment:** add the literal path to WP-01, and either relax the README to "task cards/*.md with an internal TC-* ID" or rename the task card to match the framework.

#### Medium — Dispatch registry is called global, but is seeded too narrowly

`WP/LOGBOOK.md:75` through `WP/LOGBOOK.md:83` defines the registry as the single collision-prevention table across all WPs, but seeds only a placeholder row and names `AAA`-`AAI`, `BBA`, and `BBB` as avoided pre-registry families.

Existing history already uses many other families in `CHANGELOG.md` and `WORKPLAN_v0.3.md`: single-letter `A`-`Z`, tutorial `AA`-`LL`, Phase 2 `OO`, `QQ`-`ZZ`, `RR.1`, Greek `β.1`-`β.4`, and active GPU planning reserves `BBC`-`BBE` in `docs/gpu-dispatch-design.md`. A new WP choosing only "not AAA-AAI/BBA/BBB" can still collide or confuse readers.

**Comment:** either mark the registry explicitly incomplete until backfilled, or seed all known taken/reserved families with `pre-registry` status.

#### Medium — `WP/` is not represented in the root licence split

`WP/README.md:7` classifies the framework as Sail guidance, while `WP/README.md:8` licenses it as CC BY-SA because it is Coastline-adjacent. `WP/LOGBOOK.md:8` also uses CC BY-SA. `WP/FAIR.md:10` instead marks the FAIR plan as Sail material under CC BY-NC-SA. Root `LICENCE:7` through `LICENCE:14` does not list `WP/` at all.

**Comment:** add `WP/` and `task cards/` to the root split, or put a local `WP/LICENCE` in place. This also resolves whether WP planning/logbook material is CC BY-SA by governance exception or CC BY-NC-SA by Sail default.

#### Low — Placeholder wording regressed in the ratified naming note

`WP/WP-01-estimation-darwinism.md:57` says "`information/` now replaces the former `information/` placeholder". The logbook has the intended wording at `WP/LOGBOOK.md:68`: the former placeholder was `<info>/`.

**Comment:** change the WP-01 note to say "`information/` now replaces the former `<info>/` placeholder".

#### Low — "Paste-ready" is overstated while dispatch placeholders remain

`WP/WP-01-estimation-darwinism.md:290` calls §13 paste-ready, but the block still includes `<C1>` at `WP/WP-01-estimation-darwinism.md:297` and `WP/WP-01-estimation-darwinism.md:317`.

**Comment:** until codes are minted, label §13 as a template stub rather than paste-ready. Once WI-1 lands, fill the real dispatch code and then call it paste-ready.

### Open Calls (Round 1)

- The literature-review note licence call remains real. WP-01 currently treats `docs/estimation-darwinism-review.md` as Sail / CC BY-NC-SA (`WP/WP-01-estimation-darwinism.md:117` through `WP/WP-01-estimation-darwinism.md:121`), but the note is also the source cited by future CONVENTIONS sections (`WP/WP-01-estimation-darwinism.md:119` and `WP/WP-01-estimation-darwinism.md:259`). If it becomes authoritative definitions rather than interpretive review, Coastline / CC BY-SA is cleaner.
- `WP/FAIR.md` sits inside `WP/` but is neither a `WP-NN-*` plan nor the logbook/template/README. If it remains there, the README should name FAIR plans as a permitted side-car document type; otherwise move it under a broader planning/docs location.

### Checks Run (Round 1)

- `git status --short`
- Static reads of `WP/README.md`, `WP/TEMPLATE.md`, `WP/LOGBOOK.md`, `WP/WP-01-estimation-darwinism.md`, `WP/FAIR.md`, both task cards, `WORKPLAN_v0.3.md`, `CHANGELOG.md`, `LICENCE`, `docs/LICENCE`, `pyproject.toml`, `src/iontrap_dynamics/__init__.py`, `states.py`, `entanglement.py`, and `systematics/{drift,jitter}.py`
- Targeted `rg` checks for placeholders, dispatch references, licence/classification references, and WORKPLAN §5.4 references

No test suite was run; this was a document and cross-reference review.

---

## Round 2 — 2026-06-02 (refinement-pass response)

Scope reviewed: `WP/WP-01-estimation-darwinism.md` (v0.1, post-refinement), `WP/README.md`, `WP/TEMPLATE.md`, `WP/LOGBOOK.md`, `task cards/task-card-iontrap-dynamics-service-upgrade.md`, root `LICENCE`, `CONVENTIONS.md` (header only).

### Overall assessment

**Approved for ratification.** The refinement pass resolves every High and Medium finding from Round 1. The framework is now internally consistent: WP-01 instantiates the TEMPLATE spine (§14–16 added), the lifecycle is unambiguous (only the §3 naming sub-decision is ratified; the WP stays Drafted), the WI-4 API carries the `correlation` field, card paths are literal, the dispatch-code registry is seeded with pre-registry families, and `WP/LICENCE` is in place. The README↔TEMPLATE skeleton reconciliation is clean — one canonical 8-section spine, with WP-01 expanding it to 13 without violating the contract. This is exactly what "refine the framework first" meant.

### Resolution of Round 1 findings

| Round 1 finding | Status in Round 2 | Evidence |
|---|---|---|
| High — WP-01 does not instantiate the framework | **Resolved** | WP-01 now carries the full spine (§14 Sequencing, §15 Dispatch register, §16 Logbook hooks); README §4 names the *constraint-heavy* expansion pattern explicitly. |
| High — Lifecycle ambiguous | **Resolved** | Header `Status:` line disambiguates: only the §3 sub-decision is ratified; WP-01 as a whole is Drafted; WI-1 blocked until Ratified. |
| High — WI-4 API unsatisfiable | **Resolved** | `CommonModePhase` now carries `correlation: float = 1.0` with the `√c·ξ_shared + √(1−c)·ξ_i` formula; acceptance criteria are representable. |
| Medium — Task-card path not concrete | **Resolved** | WP-01 header names literal path `task cards/task-card-iontrap-dynamics-service-upgrade.md`; README relaxed from `TC-*.md` to `*.md` carrying internal `TC-…` ID. |
| Medium — Registry seeded too narrowly | **Resolved** | `WP/LOGBOOK.md` dispatch-code registry now lists all known pre-registry taken/reserved families (single `A`–`Z`, doubles `AA`–`WW`, `BBA`–`BBE`, Greek `β.1`–`β.4`, etc.) with a grep-is-authoritative rule. |
| Medium — `WP/` not in root licence split | **Resolved** | `WP/LICENCE` added (CC BY-SA 4.0); root `LICENCE` `WP/` row is proposed (not yet applied, maintainer call). |
| Low — Placeholder wording regressed | **Resolved** | WP-01 §3 note now reads "`<info>/` placeholder" (verified). |
| Low — "Paste-ready" overstated | **Resolved** | §13 relabelled *template stub, not yet paste-ready*; paste-ready status gated on Ratification + WI-1 landing. |

### Steer on open calls

#### 1. Literature-review note licence — ratify to Coastline / CC BY-SA 4.0

**Recommendation: Coastline.** The note is not interpretive review; it is a normative definitions document that CONVENTIONS §19–22 will cite as authoritative. The repo's split-licence architecture already has a bucket for this: "specs, schemas, and CONVENTIONS edits are Coastline / CC BY-SA 4.0" (`LICENCE`, Coastline/Sail semantics section). A document that fixes canonical definitions (SLD-QFI convention, R_δ deficit convention, recoverability measure) and is cited by binding conventions is spec material, not Sail pedagogy.

**Consequences if accepted:**
- WP-01 §5 licence paragraph flips from "recommended Sail / CC BY-NC-SA" to **"Coastline / CC BY-SA 4.0 — ratified"**.
- WP-01 §9 docs-plan table updates: `docs/estimation-darwinism-review.md` footer becomes `## Endorsement Marker` (Coastline style), not `## Licence` (Sail style).
- The note itself, when authored, carries the governed header block (Classification, Licence, Stewardship, Endorsement Marker) consistent with `CONVENTIONS.md` and `WORKPLAN_v0.3.md`.
- No change to the root `LICENCE` split table is needed: `docs/` Coastline material is already CC BY-SA 4.0.

If you later decide the note should carry interpretive commentary that goes beyond definitions, the Coastline licence still fits — CC BY-SA does not forbid interpretation, it merely removes the non-commercial barrier that would prevent downstream packages (including any future commercial fork or teaching context) from citing the definitions.

#### 2. "Is this the refinement you meant?" — Yes.

"Refine the framework first" meant: make the WP system itself coherent before using it. You did exactly the right things: tightened the README↔TEMPLATE spine sync, seeded the dispatch-code registry, fixed the WORKPLAN slot reference, resolved the licence ambiguity for `WP/`, and corrected the inconsistencies that would have propagated into WP-02 and beyond. Nothing here needs re-doing.

### Specific observations on WP-01 (non-blocking)

#### §4.3 — `states.py` re-export is the largest hidden surface change

The plan correctly notes that `states.py` is **not** currently exported from `src/iontrap_dynamics/__init__.py`. Adding it exposes ~7 existing symbols (`ground_state`, `coherent_mode`, `squeezed_coherent_mode`, `squeezed_vacuum_mode`, `compose_density`, plus the two new factories) to the top-level public API. This is architecturally correct — users expect factory access — but it is a **minor API expansion**, not merely a new-module dispatch. When the release notes are authored (§10, step 4), I recommend an explicit call-out: *"Package `__init__.py` now re-exports `states.py` factories; previously these were module-internal only."* This prevents a downstream consumer from being surprised by newly visible symbols.

#### §7 — Benchmark tool naming inconsistency (benchmark 5)

Benchmarks 1–4 and 6 follow the `run_benchmark_<name>.py` pattern. Benchmark 5 is `tools/run_demo_ghz_cat.py`. The `run_demo_` prefix is inherited from the existing `run_demo_bell_entanglement.py`, but it breaks the convention established in §7's opening paragraph: *"a `tools/run_benchmark_<name>.py` writing `benchmarks/data/<name>/`"*. Either:
- **Rename to `run_benchmark_ghz_cat.py`** for consistency (preferred), or
- **Explicitly note the exception** in §7: "Benchmark 5 uses the `run_demo_` prefix because it mirrors the existing bell-entanglement demo pattern and does not emit a canonical `manifest.json` cache pair."

This is a one-line doc fix, not a blocker.

#### §14 — WI-3 before WI-2 sequencing is deliberate and correct

The dependency chain `WI-1 → WI-3 → WI-2` is well-motivated: the keystone QFI-scaling benchmark needs `ghz_state`. I confirm this is the right order. One minor suggestion: in §14, add an explicit one-sentence rationale for the WI-3 → WI-2 ordering so a future reader does not assume it is a typo:
> "WI-3 (GHZ/cat factories) precedes WI-2 (Darwinism) because the keystone benchmark (benchmark 1, bundled with C1) requires `ghz_state`; Darwinism measures have no dependency on GHZ states."

#### §6 — `convention_version` vs `conventions_version` doc inconsistency

The plan correctly flags and resolves this. Good.

#### §8 — `tests/conventions/test_convention_version.py`

The target directory `tests/conventions/` exists and already holds convention-enforcement tests. The new test is well-placed.

### Things to watch during execution

#### RUF00x ignore-list growth

Each new Unicode-bearing `src/` module self-registers `per-file-ignores = ["RUF001","RUF002","RUF003"]` in `pyproject.toml`. With three `information/` modules (`fisher.py`, `redundancy.py`, `recoverability.py`) plus `systematics/common_mode.py`, that's up to four new entries. Ruff supports glob patterns in `per-file-ignores` (e.g., `"iontrap_dynamics/information/*.py"`). Consider collapsing to a directory-level ignore when the second or third module lands, to keep `pyproject.toml` tidy. Not a blocker for WI-1.

#### WP-02 coordination on the shared v0.3 Convention Freeze

The logbook entry for WP-02 (undetected-modes) correctly flags that both WPs may need the same `CONVENTION_VERSION` 0.2 → 0.3 bump. WP-01 §14 Blockers also records this. **Watch point:** when WP-02 is drafted, explicitly compare its convention-freeze plan against WP-01 §6. If WP-02 needs §23+ (two-mode squeezing conventions, `c_ops` exposure conventions), the maintainer must decide whether to:
- **Combine** into one v0.3 freeze covering §19–23+ (simpler, one bump, one test pin), or
- **Separate** into v0.3 (WP-01) and v0.4 (WP-02) bumps (cleaner isolation, more changelog noise).

My recommendation: combine if WP-02's convention sections are ready before WP-01 releases; separate if WP-02 is still Drafted when WP-01 tags. Record the decision in `WP/LOGBOOK.md` as a roadmap-level call.

#### Dispatch registry population at Ratification

When WP-01 moves Drafted → Ratified, populate the LOGBOOK registry with **all six** planned codes (C1–C6), not just C1. The registry's value is as a forward collision-prevention table; a half-populated registry fails that purpose. The table row can carry Status: `planned` for C2–C6 and `open` for C1 once WI-1 begins.

### Readiness for next moves

The framework is coherent. WP-01 is ratifiable. My recommended next actions, in order:

1. **Accept or reject the Coastline licence steer** for the literature-review note (§5, §9). If accepted, update WP-01 §5 and §9 in the same commit.
2. **Ratify WP-01** (Drafted → Ratified). This unblocks:
   - Mint dispatch codes C1–C6 and populate `WP/LOGBOOK.md` registry.
   - Mark §13 stub paste-ready (still not actually pasted until WI-1 lands).
3. **Branch and commit the WP/ scaffolding** (`WP-01`, `README`, `TEMPLATE`, `LOGBOOK`, `LICENCE`, `FAIR` as applicable) on a feature branch.
4. **Begin FAIR Phase A** (additive files, no governed-file edits) in parallel with WI-1 planning.
5. **Open WI-1** (`information/fisher.py` + keystone QFI-scaling benchmark) once the branch is clean and the registry is populated.

No further framework refinement is needed before execution begins.
