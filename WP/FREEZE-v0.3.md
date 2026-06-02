# WP — Shared CONVENTIONS v0.3 Convention Freeze

**The single owner of the `CONVENTIONS.md` v0.2 → v0.3 bump, which more than one WP feeds**

Version 0.1 · Drafted 2026-06-02 · Status: open; sections staged, none sealed; `CONVENTION_VERSION` still `"0.2"`

**Classification:** Coastline gate (per T(h)reehouse +EC CD 0.9) — a binding coordination surface, not revisable guidance. The version bump, section-number allocation, and seal-at-release procedure below are hard gates.
**Licence:** CC BY-SA 4.0 (`WP/` governance material, `WP/LICENCE`). Permitted **side-car** document (`WP/README.md` §4), not a WP.
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (`cd-rules` v1.7.1, consumed via Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

---

## 1. Why this side-car exists *(Coastline gate)*

A CONVENTIONS freeze is a **single, repo-wide, version-gated event**: `CONVENTION_VERSION`
moves `0.2 → 0.3` exactly once, and the provenance schema stamps that one value
(`convention_version` in every `manifest.json`). It was never a per-WP concern.

Two WPs now need new conventions at the same freeze:

- **WP-01** (estimation/Darwinism, card `TC-ITD-ESTDARW-01`) — §§19–22.
- **WP-02** (undetected-modes service module, card `TC-iontrap-dynamics`) — §§23–24 (its F1 two-mode squeezing and F3 channel parameterisation are the only freeze-gated items; F4/F5/F6 are additive under v0.2 per that card's §6).

Letting each WP independently "bump to v0.3 and open a freeze gate" would be incoherent
(two different v0.3 freezes, two bumps of one constant). **This side-car owns the bump.**
No WP bumps `CONVENTION_VERSION` or seals a freeze on its own; each WP *contributes
sections* and points here. This is the structure ratified 2026-06-02 ("two WPs + shared
v0.3 freeze"); card-2 §7 Q2 independently recommends one combined freeze.

## 2. Section allocation *(Coastline gate)*

New CONVENTIONS sections continue the integer sequence (current max §18). Each section is
**staged** (`*(staged — v0.3 Convention Freeze target)*`) when its WP drafts it, and
**sealed** together at the v0.3 release (§4).

| New § | Title | Owning WP | Card item | Cites review note |
|---|---|---|---|---|
| §19 | Quantum and classical Fisher information (SLD-QFI, CFI, Cramér–Rao) | WP-01 | WI-1 | `docs/estimation-darwinism-review.md` |
| §20 | Quantum Darwinism — redundancy and recoverability | WP-01 | WI-2 | `docs/estimation-darwinism-review.md` |
| §21 | GHZ and cat state conventions | WP-01 | WI-3 | `docs/estimation-darwinism-review.md` |
| §22 | Common-mode (shared-latent) phase channel | WP-01 | WI-4 | `docs/estimation-darwinism-review.md` |
| §23 | Two-mode squeezing / SU(1,1) (phase, sign, ordering on the mode pair) | WP-02 | F1, F2 | WP-02 review note (TBD) |
| §24 | Motional CPTP channels (rates, bath occupation, Kraus vs Lindblad; sequence-aware application) | WP-02 | F3 | WP-02 review note (TBD) |

§23–24 are **provisional** until WP-02 is drafted and ratified; the titles and exact count
may move, but they occupy the next integers after WP-01's §19–22.

## 3. The single version bump *(Coastline gate)*

Owned here, executed once (de-duplicated from WP-01 §6 and WP-02's freeze plan):

- Constant `CONVENTION_VERSION: str = "0.2"` in `src/iontrap_dynamics/conventions.py`
  (module-level, in `__all__`). **Bump `"0.2"` → `"0.3"` exactly once**, at the seal (§4).
- Add `tests/conventions/test_convention_version.py` asserting `CONVENTION_VERSION == "0.3"`
  (marker `convention`) — the guard against a silent bump-skip; no such literal pin exists
  today. **This test is added by whichever WP seals first; the other WP does not re-add it.**
- Resolve the `convention_version` (§17.5) vs `conventions_version` (§13) doc inconsistency
  in the same commit (see WP-01 §6).
- A CHANGELOG `### Changed` entry records the bump, attributed to the sealing dispatch.

## 4. Seal-at-release procedure and the timeline-coupling decision *(Coastline gate)*

Sections are drafted early and **sealed together** at the v0.3 release, following the
`CONVENTIONS.md` §17/§18 staged → frozen pattern (the detailed 8-step procedure lives in
WP-01 §6 and applies to every contributing section).

**Live coordination decision — when do the two WPs' timelines meet?**

- [ ] **Combined (card-2 §7 Q2 preference).** Hold the v0.3 release until **both** WP-01
      (§19–22) and WP-02 (§23–24) sections are complete; seal §19–24 together; one bump,
      one worked example, one release. Couples the two WPs' release timing.
- [ ] **WP-01-first fallback.** If WP-02 is not ready when WP-01 is, seal **§19–22 as
      v0.3** and move WP-02's §23–24 to a later **v0.4** freeze (its own bump). Decouples
      timing at the cost of two freezes — the Option-3 structure for WP-02 only.

This box is the one genuine timeline risk in the "two WPs + shared freeze" structure. It is
**decided at WP-02 ratification**, not before, and logged in `WP/LOGBOOK.md`. Until then,
WP-01 may execute its WIs and stage §19–22, but **must not seal or bump** without this call.

## 5. QFI — one primitive, not two *(Coastline gate)*

The other shared surface (recorded here so it is not rediscovered): WP-01 WI-1 and WP-02 F6
are the **same QFI primitive**. The library hosts **one** implementation —
`information/fisher.py`, delivered by **WP-01 WI-1**. WP-02 F6 **consumes** it (card-2 marks
F6 "optional… a *bare* QFI primitive"); WP-02 does not re-implement QFI. Each programme's
resource-constraint / identifiability logic stays in its own downstream repo. This is a
dependency, not a freeze item (the QFI *convention* is §19, owned by WP-01).

## 6. Cross-links *(Coastline gate)*

- **WP-01 §6** feeds §19–22 here; it no longer owns the bump.
- **[`WP/EDF-conventions-nav-proposal.md`](EDF-conventions-nav-proposal.md)** — the ready-to-paste §19–22 staged text, the `mkdocs.yml` nav line for the review note, and the seal-time `CONVENTIONS.md` header/marker/footer edits (Dispatch EDF, drafted 2026-06-02; maintainer applies at the seal). The bump and `test_convention_version.py` stay owned by §3 here.
- **[`WP/WP-02-two-mode-motional.md`](WP-02-two-mode-motional.md) §6** feeds §23–24 here (Drafted 2026-06-02; §23 two-mode squeezing / SU(1,1) ← F1/F2, §24 motional CPTP channels ← F3). Its §6/§17 take the §4 timeline-coupling decision at WP-02 ratification.
- **`CONVENTIONS.md`** is the destination; this side-car is the staging/coordination plan.
- **`WORKPLAN_v0.3.md`** records the v0.3 freeze at roadmap level when sealed (the WP
  dispatch-track stubs reference it).

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally
validated laws. This freeze-coordination side-car is a Coastline gate within the
Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität
Freiburg), under the Coastline locks of `WORKPLAN_v0.3.md` and `CONVENTIONS.md`. Lock–Key
rule applies: this document coordinates a single change to the `CONVENTIONS.md` lock; the
WPs are keys built on it. The repository adopts the T(h)reehouse +EC Corporate Design
blueprint (`cd-rules`, consumed via Model B).

**Council status:** Guardian cleared (one bump, one seal; no WP relaxes or duplicates the
gate). Architect approved (the freeze is lifted above both WPs to its proper repo-wide
altitude; one-card-one-WP preserved). Scout horizon signals addressed (the timeline-coupling
risk in §4 is surfaced as an explicit decision, not left implicit). Integrator has sequenced
the seal to the release where the contributing WPs' sections are complete.

**Convention version:** references `CONVENTIONS.md` v0.2 (frozen 2026-04-21); this side-car
owns the staged bump to v0.3.
**Corporate design version:** `cd-v1.7.1` (consumed via Model B).
**Workplan reference:** `WORKPLAN_v0.3.md` v0.3.5; the v0.3 freeze lands at roadmap level via the WP dispatch-track stubs (WP-01 §13 / §5.4).
