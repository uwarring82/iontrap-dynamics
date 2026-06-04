# Proposal — `WORKPLAN_v0.3.md` §5.6 (reduced light–matter models) + version bump

**Status:** **APPLIED 2026-06-04** — folded into the RLG commit on the maintainer's explicit
go-ahead. The §5.6 block (now at `WORKPLAN_v0.3.md` §5.6) and the header / Endorsement-Marker
version bumps (workplan 0.3.7 → 0.3.8; convention reference → v0.4) are in `WORKPLAN_v0.3.md`.
Retained as the proposal-of-record per the RLA precedent (`WP/RL-conventions-proposal.md` was
likewise kept after its §25 seal applied); the edits below are the as-applied text.

**Dispatch:** WP-03 WI-7 / RLG · **Drafted:** 2026-06-04 · **Author stance:** Integrator.

**Why a proposal and not an edit.** `WORKPLAN_v0.3.md` carries the project's stable
release-mapping locks; per the governance rule (and the WP-02 precedent where a frozen-section
doc-mention was declined in favour of a version bump), the workplan is amended only by the
maintainer, in lock-step with its header version line and bottom Endorsement Marker. RLG
therefore *proposes* §5.6 rather than applying it. The non-governed release artefacts
(`pyproject` version, `CHANGELOG` `[0.6.0]`, WP-03 status, LOGBOOK) are applied by RLG; this
§5.6 amendment and the two version-string bumps below are the maintainer's to apply.

---

## Edit 1 — splice the §5.6 amendment

**Where:** in `WORKPLAN_v0.3.md`, immediately after the §5.5 block — i.e. after its closing
`---` separator and before `### Phase 0 — Foundations (target: v0.1-alpha, 4–6 weeks)`.

**Insert verbatim:**

```markdown
### 5.6 — Reduced light–matter models as v0.3.x follow-up (2026-06-04) *(Coastline, new in v0.3.8)*

Added as the reduced light–matter models service surface (card: abstract JC / AJC / QRM
Hamiltonians + a model-vs-realisation comparison harness + a full-ion comparison tutorial)
landed on `wp03-reduced-models`, staged for `v0.6.0`. Records the scoping decision that this
surface is a **v0.3.x follow-up**, additive and orthogonal to Phase 2 (§5.3) and to the
estimation/Darwinism (§5.4) and two-mode/motional (§5.5) tracks. Executed via
`WP/WP-03-reduced-models.md`.

**Scope.** Physics-layer only: the abstract reduced-model builders
(`reduced_models.jaynes_cummings_hamiltonian` / `anti_jaynes_cummings_hamiltonian` /
`quantum_rabi_hamiltonian`, WI-2/RLB); an analytic oracle suite for the four falsifiable
hierarchy cases (WI-3/RLC); a `model_deviation` comparison helper measuring the rotating-wave
breakdown (WI-5/RLE); Tutorial 18 + its deterministic comparison benchmark (WI-6/RLF); and the
**vendored** model-hierarchy companion note (WI-4/RLD). The physics/apparatus layer boundary —
the *reduced models* are what the apparatus approximates; the *sideband Hamiltonians* are how a
real ion realises them — is the gating invariant; the tutorial is a falsifiable demonstration of
that boundary, not an illustration of it.

**Rationale.** Additive — every new symbol is well-defined on generic spin-motion inputs (a
Hilbert space, a mode label, the three model frequencies), and existing callers observe no
behaviour change. Unlike §5.4/§5.5 (whose CONVENTIONS §19–24 sealed together under the single
v0.3 Combined Freeze, `CONVENTION_VERSION` 0.2 → 0.3), WP-03's conventions are a **separate
seal**: CONVENTIONS **§25** (reduced light–matter models — the three bare-term Hamiltonians, the
LOCK-3 identity `H_AJC(ω₀) = σ_x H_JC(−ω₀) σ_x`, the effective-`ω₀`-sign semantics) plus a §5
scope note, sealed at Dispatch RLA with `CONVENTION_VERSION` bumped **0.3 → 0.4**.

**On `wp03-reduced-models` (as landed).** Dispatches **RLA–RLF + RLD all landed**: the
`reduced_models` module (RLB) + re-exports, the analytic oracle suite (RLC), `model_deviation`
(RLE), Tutorial 18 + `tools/plot_reduced_models_comparison.py` + `benchmarks/data/reduced_models_comparison/`
(RLF), the vendored `docs/models-hierarchy.md` companion note (RLD), and **CONVENTIONS §25 + the
§5 scope note** sealed at RLA. WI-7 (RLG) stages the `v0.6.0` release (this amendment).

**External-dependency note (R7).** WI-4 (RLD) carried an external dependency — the `hierarchy.md`
v0.4 note *and* its source commit/DOI. The v0.4 note was supplied and vendored; the source
commit/DOI is **still pending** (the upstream note is a lock candidate), so **R7 is partially
discharged, not fully closed**, and `docs/models-hierarchy.md` records that field as pending.

**Consequence for §5 above.** No re-scoping of Phase 2's target; lands additively toward the
`v0.6.0` minor release, alongside the **WP-02 P1/P2** sub-dispatches (MCD–MCG) that completed on
the same `[Unreleased]` block. The `v0.6.0` tag is a maintainer release act (ff-merge
`wp03-reduced-models` → `main` first, then tag).

---
```

## Edit 2 — header version line (line 5)

**Replace** the leading `Version 0.3.7 (amended … §5.5 two-mode/motional)` clause so the
amendment list and status read:

> Version 0.3.8 (amended §4.0 repo-hosting · §5.0 release-mapping · §5.1 v0.2 release · §5.2 post-v0.2 on-`main` · §5.3 β.4 as v0.3.x follow-up · §5.4 estimation/Darwinism · §5.5 two-mode/motional · **§5.6 reduced light–matter models**) · Drafted 2026-04-17 · Status: v0.2.0 tagged 2026-04-21; Phase 2 JAX-backend time-independent surface on `main`; the **v0.3 Convention Freeze** seals CONVENTIONS §19–24 (§5.4/§5.5) and bumps `CONVENTION_VERSION` 0.2 → 0.3; **WP-03 reduced models seal CONVENTIONS §25 + §5 scope and bump `CONVENTION_VERSION` 0.3 → 0.4 (§5.6), staged for `v0.6.0`**; β.4 time-dependent extension scoped as v0.3.x follow-up; see §5.3

## Edit 3 — bottom Endorsement Marker

**(3a) Convention version line** — append the §25 seal:

> **Convention version:** references `CONVENTIONS.md` **v0.4** (the v0.2 freeze 2026-04-21 closed §17 measurement layer and §18 systematics layer; the **v0.3 freeze 2026-06-03** closes §19–22 estimation/Darwinism and §23–24 two-mode/motional, per §5.4/§5.5; **WP-03 seals §25 reduced light–matter models + the §5 scope note 2026-06-04, bumping `CONVENTION_VERSION` 0.3 → 0.4 per §5.6**; §1–16 carry through unchanged from the v0.1 draft).

**(3b) Workplan version line** — bump 0.3.7 → 0.3.8 and add the §5.6 clause:

> **Workplan version:** 0.3.8 (amended §4.0 repo-hosting, §5.0 release-mapping 2026-04-19, §5.1 v0.2 release 2026-04-21, §5.2 post-v0.2 on-`main` 2026-04-21, §5.3 β.4 as v0.3.x follow-up 2026-04-22, §5.4 estimation/Darwinism 2026-06-02, §5.5 two-mode/motional 2026-06-03, **§5.6 reduced light–matter models 2026-06-04**) · `v0.2.0` tagged 2026-04-21 covering Phase 0 foundations plus the full Phase 1 deliverable; the **v0.3 Convention Freeze** seals CONVENTIONS §19–24 and bumps `CONVENTION_VERSION` 0.2 → 0.3; **WP-03 seals §25 + §5 scope, bumping `CONVENTION_VERSION` 0.3 → 0.4, staged for `v0.6.0` (§5.6)**; Phase 2 JAX-backend time-independent surface and tutorials on `main` under `[Unreleased]`; β.4 time-dependent extension scoped as v0.3.x follow-up per §5.3.

---

## Application checklist (maintainer)

1. Apply Edit 1 (splice §5.6 after §5.5, before Phase 0).
2. Apply Edit 2 (header version line 0.3.7 → 0.3.8).
3. Apply Edits 3a/3b (Endorsement Marker convention + workplan version lines).
4. Commit on `wp03-reduced-models` (or fold into the RLG commit).
5. Release: ff-merge `wp03-reduced-models` → `main`, then `git tag v0.6.0`.
6. Once the upstream model-hierarchy note locks, record its commit/DOI in
   `docs/models-hierarchy.md` to fully close **R7**, and refresh the sealed-§25 forward-reference
   wording at the next convention bump.

After this proposal is applied, this file may be deleted (its content lives in `WORKPLAN_v0.3.md`).
