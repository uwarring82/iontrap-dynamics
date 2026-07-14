# Proposal — `WORKPLAN_v0.3.md` §5.10 (multimode Gaussian toolbox) + version bump

**Status:** WP-07 ratification bookkeeping — the §5.10 amendment and the two version-string bumps
below are **applied to `WORKPLAN_v0.3.md`** in the WP-07 ratification commit (per the maintainer's
staged-patch instruction), and this side-car is retained as the **proposal-of-record** (the
`WP/RL-workplan-5.6-proposal.md` precedent). The **CONVENTIONS §27 paste**, the
`CONVENTION_VERSION` 0.5 → 0.6 bump in `src/iontrap_dynamics/conventions.py`, and the
`tests/conventions/test_convention_version.py` pin remain the **maintainer-governed lock-turns**,
applied in a separate seal commit.

**Dispatch:** WP-07 ratification (GT family) · **Drafted:** 2026-07-14 · **Author stance:** Integrator.

**Why a proposal-of-record.** `WORKPLAN_v0.3.md` carries the project's release-mapping and
convention-version narrative; it is amended only in lock-step with its header version line and
footer Endorsement Marker. This side-car records the exact as-applied text so the change is
auditable alongside the seal it accompanies.

---

## Edit 1 — splice the §5.10 amendment

**Where:** in `WORKPLAN_v0.3.md`, immediately after the §5.9 block's closing `---` separator and
before `### Phase 0 — Foundations`.

**Insert verbatim:**

```markdown
### 5.10 — Multimode Gaussian toolbox (2026-07-14) *(Coastline, new in v0.3.12)*

Added as an **application-independent Gaussian-state / covariance toolbox** (card:
`TC-gaussian-entanglement-toolbox.md` — covariance `V`, the Williamson symplectic
spectrum, symplectic congruence, arbitrary-cut log-negativity, and effective
temperature `T_eff`), reusable by the existing two-mode squeezing machinery and by
the future two-ion WP-SQ Phase B. Ratified and executed via
`WP/WP-07-gaussian-toolbox.md`; dispatch family `GT`.

**Scope.** A new multimode `gaussian.py` surface generalising WP-05's single-mode
covariance core; the ion-specific normal→local `S`-adapter is kept separate (near the
sibling `iontrap-structure`). The work splits into GT1–GT6:
- **GT1** — §27 seal + the low-level convention primitives (`symplectic_form`,
  multimode `covariance_matrix`, the `V + iΩ ≥ 0` PSD guard, `symplectic_eigenvalues`,
  `partial_transpose`);
- **GT2** — purity + von-Neumann entropy (from GT1's symplectic spectrum);
- **GT3** — the generic symplectic congruence `S V Sᵀ` + the separate ion `S`-adapter;
- **GT4** — arbitrary-cut Gaussian log-negativity + cut semantics (`1×N` separability
  certification; `M×N` NPT-witness with the PPT-bound-entangled caveat surfaced);
- **GT5** — effective temperature (neutral `T_eff`, first-moment-aware `n̄`);
- **GT6** — (optional) the locally-symmetric Gaussian `E_F`; generic `E_F` deferred
  (2019 PRL Supplemental-gated).

**Convention bump (pending seal).** Introduces a new **CONVENTIONS §27** (multimode
quadrature ordering, symplectic form `Ω = ⊕J`, partial-transpose sign map) — a brand-new
additive section that **reuses** frozen §26.2 (the vacuum-variance-1 single-mode
normalisation) and **cross-refs** frozen §23 (the TMSV oracle), with no frozen-section
edit — bumping `CONVENTION_VERSION` **0.5 → 0.6**. §27 is **staged, not yet sealed**, in
`WP/GT-conventions-proposal.md` (conventions-before-code); the maintainer applies the seal
after a green `tests/conventions/test_gaussian_conventions.py`.

**Seal posture.** WP-07 ratified 2026-07-14 (dispatch family `GT`, `GT1`–`GT6` minted,
collision-clear via the five-source grep); the CONVENTIONS §27 seal + `CONVENTION_VERSION`
0.5 → 0.6 bump remain the maintainer's separate lock-turn.

**Consequence for §5 above.** No re-scoping of Phase 2's target or of the
WP-02/WP-03/WP-04/WP-05/WP-06 surfaces; lands additively toward a future
multimode-Gaussian minor release.
```

Then append the closing `---` after the block.

## Edit 2 — bump the header version line

**Where:** `WORKPLAN_v0.3.md` line 5 (`Version 0.3.11 (amended …)`).

- `Version 0.3.11` → `Version 0.3.12`.
- Extend the amendment list: append `· **§5.10 multimode Gaussian toolbox**` after
  `**§5.9 tutorial-track accessibility**`.
- Extend the freeze narrative: after the WP-06 clause, add `**WP-07 multimode Gaussian toolbox
  seals CONVENTIONS §27 + `CONVENTION_VERSION` 0.5 → 0.6 (§5.10), 2026-07-14**`.

## Edit 3 — bump the footer Workplan-version line

**Where:** `WORKPLAN_v0.3.md` footer (`**Workplan version:** 0.3.11 (amended …)`).

- `0.3.11` → `0.3.12`; append `**§5.10 multimode Gaussian toolbox 2026-07-14**` to the amendment
  list; add the WP-07 §27 / `CONVENTION_VERSION` 0.5 → 0.6 clause to the narrative, mirroring the
  header.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally
validated laws. This side-car records a Coastline amendment to `WORKPLAN_v0.3.md` within the
Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg).
The §5.10 amendment and the two version-string bumps above accompany the WP-07 ratification; the
`CONVENTIONS.md` §27 paste and the `CONVENTION_VERSION` 0.5 → 0.6 bump remain maintainer-governed
lock-turns. Licensed under **CC BY-SA 4.0**.

**Workplan reference:** `WORKPLAN_v0.3.md` §5.10 (`new in v0.3.12`); `WP/WP-07-gaussian-toolbox.md`.
