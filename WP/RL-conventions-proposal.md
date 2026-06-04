# RL — Proposed CONVENTIONS §25 + §5 scope note *(maintainer-applied)*

**Status:** WP-03 conventions-before-code (Dispatch RLA) — proposal only. The text
below is **ready to paste** but edits two governed files (`CONVENTIONS.md` and
`src/iontrap_dynamics/conventions.py`); per the WP governance rule those are
**maintainer-governed acts**, applied at the v0.4 seal, not by the implementing
agent. Nothing here is auto-applied.

**Licence:** CC BY-SA 4.0 (`WP/` governance material). Permitted side-car, not a WP.

**Post-v0.3 (v0.4 target).** Unlike the EDF/MC proposals (which rode the v0.3
Convention Freeze), this amendment lands **after** the sealed v0.3 freeze
(`FREEZE-v0.3.md`, commit `fdcd20f`, `CONVENTION_VERSION` already `0.3`). The bump
is **0.3 → 0.4**, owned by the v0.4 seal — a `WP/FREEZE-v0.4.md` side-car or an
in-place amendment, the maintainer's call (WP-03 §0 R2) — **not** by the spent
`FREEZE-v0.3.md` §3.

**This proposal touches a frozen section — read this.** §25 is a brand-new
additive section (the EDF/MC pattern). The **§5 scope note is different**: §5 is
*frozen* (≤ v0.3), and the project's standing rule is that a frozen section is
amended **only via a version bump, never a no-bump doc-mention** (cf. the declined
§18.4 doc-mention, 2026-06-03). This proposal therefore carries the §5 re-scope
**inside** the `CONVENTION_VERSION` 0.3 → 0.4 bump that legitimises it — it is *not*
a doc-mention. The note is a **normative re-scope** (it narrows §5's universal "All
builders" predicate). The maintainer rules on whether to (a) seal it as written,
(b) reword, or (c) reject it and instead make §25 self-contained without touching
§5. The conventions test (`tests/conventions/test_reduced_models_conventions.py`,
already green under RLA) anchors the LOCK-3 **algebra behaviourally** and never
reads the `CONVENTIONS.md` markdown, so it stays green regardless of that ruling.

The §25 definition is anchored by the vendored model-hierarchy companion note
(vendored separately at Dispatch RLD; forward-referenced until it lands) and cites
`CONVENTIONS.md` §3/§5 (WP-03 §6 binding rule).

---

## A. Staged CONVENTIONS edits

### A.1 §5 scope note (edit to the **frozen** §5 — paste after the opening paragraph)

`CONVENTIONS.md` §5 currently opens (line 119): *"All builders return Hamiltonians
in the **interaction picture of the atomic transition**: the free atomic term
Σ_i (ω_atom / 2) σ_z^{(i)} is removed, and drives are written in the rotating frame
at the atomic frequency."* Insert the following **new paragraph immediately after
that opening sentence**, before the blank line and the **Rotating-wave
approximation.** paragraph:

> **Scope.** The interaction-picture discipline above governs builders **derived
> from an atomic (spin) transition**, whose lab-frame free atomic term
> Σ_i (ω_atom/2) σ_z^{(i)} has been transformed away — the apparatus/drive builders
> (carrier, sidebands, Mølmer–Sørensen, full-Lamb–Dicke). Pure-motional objects —
> the §23 two-mode-squeezing builder and the §24 motional channels — have no atomic
> transition to transform and lie outside it by construction. A builder family
> deliberately placed in a **different picture or RWA regime** — notably the
> Schrödinger-picture, non-RWA reduced light–matter models of **§25** — is governed
> by that section instead.

This narrows the literal "All builders" claim (which is overbroad: the atomic-term
removal is only meaningful for builders derived from an atomic transition) without
disturbing the RWA paragraph or the `H_carrier` reference example below it.

### A.2 New section (paste after §24)

### `## 25. Reduced light–matter models *(staged — v0.4 target; depends on the frozen-§5 edit A.1)*`

**Status:** §25 **staged at Dispatch RLA** (this work item, conventions-before-code);
the JC/AJC/QRM builders that *consume* it land at WI-2 (Dispatch RLB,
`src/iontrap_dynamics/reduced_models.py`) only after this seal. Staged, not sealed;
seals under a post-v0.3 amendment (`CONVENTION_VERSION` 0.3 → 0.4 — `FREEZE-v0.3.md`
is closed). Definitions anchored by the vendored model-hierarchy companion note
(Dispatch RLD; forward-referenced until it lands).
Conventions test `tests/conventions/test_reduced_models_conventions.py` is already
green (Dispatch RLA); it exercises the LOCK-3 algebra at dimensionless O(1) scale
(the identity is scale-free) while builder inputs remain SI rad·s⁻¹ (WP-03 R4),
and it pins the absolute ½/ω_f/g coefficients via matrix-element anchors.

Reduced light–matter models are **physics-layer** abstract qubit–oscillator
Hamiltonians: what the apparatus approximates, distinct from the §5 apparatus/drive
builders that realise them. They are written in the **Schrödinger picture** with a
**bare** atomic term — NOT the §5 interaction picture (see the §5 scope note A.1) —
and returned as static, Hermitian `Qobj` in `H/ℏ` units of **rad·s⁻¹** on a
one-spin + one-mode embedding (§2 order, §3 spin basis).

#### 25.1 Term selection

For atomic frequency ω₀, oscillator/field frequency ω_f (builder kwarg `omega_f`;
the physical motional mode ω_m of §10/§11 in an apparatus realisation), coupling g
(all rad·s⁻¹), with σ_z = |↑⟩⟨↑| − |↓⟩⟨↓| and σ_+ = |↑⟩⟨↓| (§3) and â the mode
annihilation operator:

- **Jaynes–Cummings (JC):** ``H_JC/ℏ = ½ω₀ σ_z + ω_f â†â + g(â σ_+ + â† σ_−)`` —
  co-rotating; conserves the excitation number ``N̂ = â†â + |↑⟩⟨↑| = â†â + σ_+σ_−``
  (a U(1) symmetry); couples ``|↑,n⟩ ↔ |↓,n+1⟩``, leaving ``|↓,0⟩`` dark.
- **Anti-Jaynes–Cummings (AJC):** ``H_AJC/ℏ = ½ω₀ σ_z + ω_f â†â + g(â† σ_+ + â σ_−)``
  — counter-rotating; conserves the difference number ``Ĉ = â†â − |↑⟩⟨↑| =
  â†â − σ_+σ_−``; couples ``|↓,n⟩ ↔ |↑,n+1⟩`` (its dark state is ``|↑,0⟩``).
- **Quantum Rabi (QRM):** ``H_QRM/ℏ = ½ω₀ σ_z + ω_f â†â + g σ_x(â + â†)`` — full
  dipole coupling, **non-RWA** (JC + AJC); conserves neither ``N̂`` nor ``Ĉ``, only
  the **Z₂ parity** ``P = exp(iπ N̂) = −σ_z(−1)^{â†â}`` (the companion note / card
  write ``Π = σ_z(−1)^{â†â}``; the two differ by an immaterial global sign — same
  ±1 eigenspaces).

#### 25.2 ω₀ sign semantics

``ω₀`` is an **effective model bare splitting**, not a physical ion transition
frequency. It may be taken **negative** as a reduced-model parameter (e.g.
red-sideband selection maps to an effective ``−ω₀`` frame). Physically ``ω₀ > 0``
puts ``|↑⟩`` above ``|↓⟩`` (§3); the negative argument in the LOCK-3 identity below
is a **model sign**, distinct from the §4 drive detuning ``δ = ω_laser − ω_atom``.

#### 25.3 LOCK-3 identity

``H_AJC(ω₀) = σ_x H_JC(−ω₀) σ_x``.

Holds exactly under §3 (``σ_x σ_z σ_x = −σ_z``, ``σ_x σ_± σ_x = σ_∓``; ``σ_x`` is
the identity on the motional factor). The negative argument ``−ω₀`` is **essential
for ω₀ ≠ 0**: the ``σ_x`` conjugation flips the ``σ_z`` sign, so the input
frequency must be pre-flipped to recover the physical ``+½ω₀ σ_z`` AJC term (at
``ω₀ = 0`` the sign is immaterial). The QRM coupling ``σ_x(â + â†)`` is
``σ_x``-invariant, so ``σ_x H_QRM(−ω₀) σ_x = H_QRM(ω₀)`` (only the bare sign moves;
the JC↔AJC coupling swap is absent). Symmetry contrast: JC/AJC carry a U(1)
excitation-like number, the QRM only a Z₂ parity.

**Convention.** Reduced-model builders return Schrödinger-picture, bare-term,
static Hermitian `Qobj` per 25.1; physical-SI inputs (rad·s⁻¹); ``ω₀`` sign per
25.2; the LOCK-3 identity 25.3 is the conventions-test gate.
**Cross-refs.** §2 (tensor order), §3 (spin basis), §5 scope note A.1 (why these
are exempt from the interaction-picture / RWA default), §4 (drive detuning — a
distinct sign), §10 (Lamb–Dicke — the apparatus realisation), §23/§24 (other
physics-layer / dissipation conventions).
**Test.** `tests/conventions/test_reduced_models_conventions.py` (LOCK-3 identity +
JC/AJC/QRM symmetry contrast); reduced-model builder behaviour under Dispatch
RLB/RLC.

---

## B. `mkdocs.yml` nav — none required

§25 is a CONVENTIONS *section* (not a new doc page), and the §5 note edits an
existing section, so **no new `mkdocs.yml` nav line** is needed here. (Contrast the
EDF/MC proposals, whose §B added a nav line for a new `docs/*-review.md` page.) The
model-hierarchy companion note that anchors §25 is a **separate** deliverable at
Dispatch RLD (WI-4); its nav line is staged with that dispatch, not here.

---

## C. Seal-time edits — at the v0.4 seal only (maintainer-governed)

§25 + the §5 scope note are a **post-v0.3 amendment**: `FREEZE-v0.3.md` is sealed
(`fdcd20f`, `CONVENTION_VERSION` already `0.3`), so they cannot ride it. Whether
the maintainer spins up a `WP/FREEZE-v0.4.md` side-car or amends in place is the
maintainer's call (WP-03 §0 R2). Either way the following land **once, in a single
seal commit**:

1. **Insert the §5 scope note** (A.1) after the §5 opening paragraph
   (`CONVENTIONS.md` line 119), before the **Rotating-wave approximation.**
   paragraph. *This edits the frozen §5 — sanctioned only by the version bump in
   step 3, not as a no-bump doc-mention (see the carve-out above).*
2. **Append §25** (A.2) after §24; drop the `*(staged …)*` tag from its heading and
   append a freeze line: `**§25 freeze.** Sections 25.1–25.3 received a complete
   read-through for the v0.4 convention gate. Post-v0.4 additions require a further
   version bump.`
3. **Bump** `CONVENTION_VERSION` 0.3 → 0.4 in `src/iontrap_dynamics/conventions.py`,
   **and** update the pinned literal in `tests/conventions/test_convention_version.py`
   (0.3 → 0.4) in the same commit (mirrors FREEZE-v0.3 §3's bump-and-guard pairing).
   Optionally add a one-line inline comment by the `CONVENTION_VERSION = "0.4"` line
   recording the date + §25/§5 context (the existing `0.3` literal carries no
   provenance), to keep the version history grep-friendly.
4. **Header block** → `**Scope:** Conventions covering §1–25`; freeze narrative
   names v0.4 and the added §25 + the §5 re-scope.
5. **Endorsement Marker** → restate: §17–18 under v0.2; §19–24 under the v0.3
   freeze; **§25 + the §5 scope note under v0.4**; §1–16 carry forward (§5
   re-scoped, not reopened).
6. **Footer** → `**Convention version:** 0.4 · 2026-06-XX · reduced light–matter
   models (§25) + §5 scope.`
7. **`WORKPLAN_v0.3.md`** → the WP-03 dispatch-track stub (the next free §5.x),
   header/footer/Endorsement bumped in lock-step — a separate maintainer act
   (WP-03 footer), not performed at ratification.
8. **Verify after the seal** — run `git grep CONVENTION_VERSION` to confirm no
   stale `0.3` literal remains outside the updated `test_convention_version.py`
   (catch hardcoded references in docs/comments), and re-run the conventions tier.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with
externally validated laws. This is a Coastline proposal side-car within the
Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-
Universität Freiburg). It stages — does not apply — edits to the `CONVENTIONS.md`
and `src/iontrap_dynamics/conventions.py` locks; the seal and the
`CONVENTION_VERSION` 0.3 → 0.4 bump are maintainer-governed acts. Licensed under
**CC BY-SA 4.0**.

**Convention version:** stages `CONVENTIONS.md` §25 + a §5 scope note for a v0.4
amendment (post-v0.3).
**Workplan reference:** `WP/WP-03-reduced-models.md` §0 (R2), §6; conventions test
Dispatch RLA.
