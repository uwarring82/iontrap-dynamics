# WP — FAIR-for-Research-Software Assessment & Action Plan

> **Classification.** Sail improvement plan, governed by Coastline gates.
> This document is *adaptive guidance* (a planning record), not a binding constraint;
> it does not amend `CONVENTIONS.md`, `WORKPLAN_v0.3.md`, or any frozen spec.
> Where an action edits a **governed/frozen** file, that action is flagged and
> deferred to maintainer ratification (Phase B). Nothing here ships without
> a CHANGELOG entry and, for governed edits, a migration note.
>
> **Licence (planning doc):** `CC BY-SA 4.0` — `WP/` governance/planning material (`WP/LICENCE`). A FAIR *improvement plan* reads as Sail in content, but the whole `WP/` folder is licensed on the Coastline track for consistency; see `WP/LICENCE`. It is a permitted **side-car** document (`WP/README.md` §4), not a WP.
> **Date:** 2026-06-02. **Author:** U. Warring (solo maintainer).
> **Spelling:** Oxford British English throughout.
> **Endorsement Marker:** local candidate framework; no external endorsement implied.

---

## 1. Why FAIR4RS here — and what is already done

FAIR-for-Research-Software (FAIR4RS) extends the FAIR principles
(Findable, Accessible, Interoperable, Reusable) to software as a first-class
research output. For `iontrap-dynamics` the motivating events are concrete:

1. an imminent migration of the repository to the **`open-iontrap`** GitHub org
   (inventoried in `pyproject` `[project.urls]` comments and `LICENCE`), which
   will break every GitHub URL we currently rely on for findability; and
2. the desire for the package to be **citable** as a research artefact in the
   `single-25Mg-plus` twin and downstream applications.

**This plan is deliberately proportionate.** The repository is already strong on
the two FAIR axes that solo physics packages most often miss — rich metadata and
reproducible provenance. We must *not* over-engineer. The following are treated
as **already satisfied** and are out of scope for re-work:

- **Reusable provenance schema (R, strong).** Benchmark outputs carry verbatim,
  machine-readable, hash-verified provenance:
  `request_hash`, `convention_version`, `backend_name`/`backend_version`,
  `cache_format_version`, `storage_mode`, `provenance_tags`
  (`benchmarks/data/ms_gate_bell_demo/manifest.json`), and
  `schema_version`, `canonical_request_hash`, `generated_at` (ISO-8601 UTC),
  `environment` (`python/numpy/scipy/qutip/platform/machine`), `workplan_reference`
  (`demo_report.json`). This is seed/environment-stamped and better than most
  comparable packages.
- **Pinned dependencies** with floors *and* ceilings
  (`qutip>=5.0,<6`, `numpy>=1.24,<3`, `scipy>=1.10,<2`).
- **SemVer + annotated git tags** (`v0.2.0`/`v0.3.0`/`v0.4.0`), consistent across
  `pyproject.toml`, `CITATION.cff`, and `README.md` (all `0.4.0`);
  `CHANGELOG.md` follows Keep a Changelog.
- **`CITATION.cff`** (CFF 1.2.0) — machine-readable citation vocabulary present.
- **MIT licence, clear and SPDX-tagged** (`LICENCE-MIT`), with the split-licence
  per-layer SPDX table documented in `LICENCE`. Every `.py` carries
  `# SPDX-License-Identifier: MIT` on line 1.
- **Rich `pyproject` metadata** (name/version/authors/maintainers/keywords/
  classifiers/5+ urls, `Typing :: Typed`).
- **mkdocs-material docs site** with `mkdocstrings` available; strict build in CI.

> **Net effect:** F-metadata-richness and R-provenance/versioning are *done*.
> The real gaps are a **persistent identifier**, an **archival path**, the single
> **interoperability vocabulary** (`codemeta.json`), and **community/governance
> scaffolding** for the org transition. Everything below targets only those.

---

## 2. FAIR scorecard

Granularity follows the FAIR4RS sub-principles at a level that is actionable.
"Touches governed?" = yes if the action edits a frozen/governed file
(`CITATION.cff`, `README.md`, `mkdocs.yml`/`docs/`) requiring a migration
note + maintainer sign-off.

| Sub-principle | Current state | Gap | Recommended action | Risk / owner | Touches governed? |
|---|---|---|---|---|---|
| **F1 — global, persistent ID** | No DOI anywhere; `CITATION.cff` has no `doi:`/`identifiers:`; no DOI badge | No persistent identifier | Enable Zenodo–GitHub integration; cut a GitHub Release on `v0.4.0` to mint concept-DOI + version-DOI | Low / maintainer | Wiring **no**; DOI write-back **yes** (Phase B) |
| **F2 — rich metadata** | Rich `pyproject` + `CITATION.cff` (keywords, classifiers) | None of substance | None — keep current discipline | — / maintainer | No |
| **F3 — metadata include ID** | Metadata present, ID absent | DOI not yet in metadata | After F1: add `doi:`/`identifiers:` to `CITATION.cff` | Low / maintainer | **Yes** (Phase B) |
| **F4 — indexed/searchable** | GitHub-indexable only; no PyPI (`Pre-Alpha`) | No index beyond GitHub | Defer PyPI to post-org; Zenodo DOI satisfies F4 for now | Low / maintainer | No |
| **A1 — retrievable by ID (open protocol)** | Public repo by URL; MIT; in-repo metadata | No ID-based retrieval; URLs break on `open-iontrap` rename | Zenodo DOI gives a GitHub-independent persistence layer (highest leverage) | Low / maintainer | No (additive snapshot) |
| **A2 — metadata persist beyond software** | Lives only in the live repo | No frozen snapshot if repo moves | Zenodo archival snapshot (same action as A1/F1) | Low / maintainer | No |
| **A-aux — security contact** | No `SECURITY.md` | Minor for A; tidiness | Minimal `SECURITY.md` (contact = maintainer email in `pyproject`) | Low / maintainer | No (new file) |
| **I1 — formal, shared vocabulary** | `CITATION.cff` (CFF 1.2.0) present; `.npz`+`.json` data; typed API | No `codemeta.json` (schema.org/CodeMeta crosswalk) | Generate `codemeta.json` from `pyproject` + `CITATION.cff` | Low / maintainer | No (new file) |
| **I2 — qualified references** | Bespoke `schema_version`/`cache_format_version` ints; internally consistent | Provenance JSON has no `$schema`/`@context` linked to a published schema | *Optional* — publish a JSON Schema for manifest/report under `docs/`, reference it | Low / maintainer | No (new file; defer) |
| **I-aux — JSON-LD on docs site** | mkdocs-material, no `SoftwareSourceCode` injected | No schema.org on site | Defer — `codemeta.json` covers the I-vocabulary gap | Low / maintainer | (deferred) |
| **R1 — clear licence** | MIT (`LICENCE-MIT`, SPDX), split-licence documented | None | None | — / maintainer | No |
| **R1.1 — provenance** | `request_hash`/`convention_version`/`backend_version`/`environment`/`generated_at`/`canonical_request_hash` | None — strong | None (do not over-engineer) | — / maintainer | No |
| **R1.2 — community standards** | `CHANGELOG.md`, `CONVENTIONS.md`, `docs/boundary-decision-tree.md`; **no** `CONTRIBUTING.md`, CoC, issue/PR templates | No contribution norms or community health files | Add `CONTRIBUTING.md`, CoC, issue/PR templates, `GOVERNANCE.md` stub | Low / maintainer | No (new files) |
| **R-aux — dependencies/versioning** | Pinned floors+ceilings; SemVer+tags; Keep-a-Changelog | None | None | — / maintainer | No |
| **R-aux — archival path in CI** | `ci.yml`+`docs-deploy.yml` only; no `tags:`/`release:` trigger | No release-archival automation feeding Zenodo | Add tag-triggered `release.yml` (build sdist/wheel; attach to Release Zenodo watches) | Low / maintainer | No (new workflow) |

---

## 3. Prioritised, phased action plan

Ordering principle: **highest FAIR leverage, lowest risk, additive-first.**
Each item is checkable. Phase A touches no governed file. Phase B is gated on
maintainer ratification (and, for most items, the `open-iontrap` decision).
Phase C is the steady-state routine.

### Phase A — purely additive, low-risk new files (no governed edits)

These can land immediately and independently of WP-01 and of the org migration.

- [ ] **`codemeta.json`** — derive from existing `pyproject.toml` + `CITATION.cff`
      (e.g. `cffconvert` / `codemetar`). Closes the sole I-vocabulary gap.
      *Acceptance:* file validates; authors/licence/version match `pyproject`/`CITATION.cff`.
- [ ] **`.zenodo.json`** *(optional)* — only if we want to override Zenodo's
      auto-derivation from `CITATION.cff`. Skip unless the auto-derived record is wrong.
- [ ] **`CONTRIBUTING.md`** — point to existing `docs/boundary-decision-tree.md`,
      `CONVENTIONS.md`, the pre-commit/CI flow, and the dispatch + CHANGELOG discipline.
      Record the **5-step release-cut procedure** here so it stops being commit-body
      archaeology (currently encoded only in the `v0.4.0` release commit body).
      *Acceptance:* a new contributor can reproduce a CHANGELOG entry + run the gates from this file alone.
- [ ] **`CODE_OF_CONDUCT.md`** — Contributor Covenant; contact = maintainer email.
- [ ] **`GOVERNANCE.md`** *(stub only)* — solo-maintainer → org-steward model;
      defer detail until `open-iontrap` exists. One short page.
- [ ] **`SECURITY.md`** — minimal; private-disclosure contact = maintainer email.
- [ ] **`.github/ISSUE_TEMPLATE/bug_report.yml`** + **`feature_request.yml`** —
      lightweight; the `.github/` tree is already MIT per the `LICENCE` table.
- [ ] **`.github/PULL_REQUEST_TEMPLATE.md`** — checklist:
      CHANGELOG entry · convention/`CONVENTIONS.md` impact · tests added/updated ·
      SPDX header on new `.py` · dispatch code (if applicable).
- [ ] **`.github/workflows/release.yml`** — **tag-triggered** (`on: push: tags: 'v*'`
      / `release:`); builds sdist+wheel via hatchling, attaches artefacts to the
      GitHub Release that Zenodo watches. **Does not touch `ci.yml` logic.**
      *Acceptance:* dry-run on a throwaway tag produces a wheel+sdist; Zenodo deposit appears.
- [ ] **API reference autodoc in mkdocs** — add a `mkdocstrings`-driven reference
      page (the plugin is already an installed `docs` extra). Build a `docs/reference.md`
      (or expand `phase-1-architecture.md`) using `::: iontrap_dynamics...` autodoc blocks.
      > **Caveat (load-bearing):** any **new** page must be added to `mkdocs.yml` `nav`
      > — `mkdocs build --strict` (CI `docs` job) promotes "page not in nav" to a hard
      > failure; there is no `not_in_nav` escape hatch. Adding the nav entry is a
      > **governed edit** to `mkdocs.yml`, so the *nav line* itself is technically
      > Phase B; the page content authoring is Phase A. Any embedded figure needs
      > **descriptive alt text** to clear the Level-A `pa11y` (`WCAG2A`) gate without
      > new ignore codes.

### Phase B — edits to governed/frozen files (require maintainer ratification)

Gate: **wait for the `open-iontrap` migration decision** before minting the DOI
write-back, so the persistent record points at the final home. Each item needs a
CHANGELOG entry + migration note + sign-off.

- [ ] **Mint the Zenodo DOI** — enable Zenodo–GitHub integration; cut a GitHub
      Release on the existing `v0.4.0` tag to obtain a **concept-DOI** (version-agnostic,
      cite-this-software) + a **version-DOI**. *(The wiring is additive; only the
      write-backs below are governed.)*
      **This single action satisfies F1 + A1 + A2 and survives the org rename — highest leverage.**
- [ ] **`CITATION.cff`** *(GOVERNED)* — add `doi:` (concept-DOI) and `identifiers:`
      (version-DOI); add `date-released`; consider `commit:` and `preferred-citation:`.
- [ ] **`README.md`** *(GOVERNED)* — add a **DOI badge** and a **"How to cite"**
      section (README currently has zero badges and no citation heading).
- [ ] **`mkdocs.yml`** *(GOVERNED)* — add the `nav:` entry for the new API reference
      page (mandatory under `--strict`; see Phase A caveat).
- [ ] **`mkdocs.yml`/`docs/` schema.org JSON-LD** *(GOVERNED, OPTIONAL)* — defer;
      `codemeta.json` already closes the I-vocabulary gap.

### Phase C — ongoing routine (steady state)

- [ ] **Release-archival routine.** On each future tag: `release.yml` builds and
      attaches artefacts; Zenodo auto-mints the new version-DOI under the concept-DOI;
      update `CITATION.cff` `version`/`date-released`/`identifiers` as part of the
      existing 5-step release cut (fold into `CONTRIBUTING.md`).
- [ ] **Metadata-drift upkeep.** Keep `codemeta.json` consistent with
      `pyproject.toml`/`CITATION.cff`. *(Future option: a CI drift check; additive,
      defer until it has earned its keep.)*
- [ ] **Community-health upkeep.** Maintain `CONTRIBUTING.md`/CoC/templates;
      flesh out `GOVERNANCE.md` once `open-iontrap` and its steward model are real.

---

## 4. Do NOT do / out of scope (keep it proportionate)

The existing provenance and versioning discipline is strong; the following would
be over-engineering for the solo / pre-org phase and are explicitly excluded:

- **No PyPI publishing yet.** Status is `Development Status :: 2 - Pre-Alpha`.
  Defer to post-org-migration; the Zenodo DOI satisfies findability now.
- **No premature DOI** if the maintainer prefers to wait — the DOI write-backs
  (Phase B) are **gated on the `open-iontrap` migration decision** so the concept
  record resolves to the final home, not a soon-to-be-renamed URL.
- **No heavyweight `GOVERNANCE.md`.** A stub now; detail only when the org exists.
- **No schema.org/JSON-LD injection** into the docs site — `codemeta.json` covers
  the I-vocabulary gap; site-level JSON-LD is deferred indefinitely.
- **No re-engineering of the provenance schema.** Publishing an external JSON
  Schema with `$schema`/`@context` for the manifest/report (I2) is *nice-to-have*,
  not required; do it only if a consumer needs it.
- **No new metadata frameworks** beyond `codemeta.json` (e.g. no CodeMeta v3
  tooling sprawl, no DataCite-direct flows — Zenodo handles DataCite for us).
- **No edits to `ci.yml` logic.** Archival lives in a *separate* `release.yml`.
- **No touching `CONVENTIONS.md`/`WORKPLAN`/design-note specs** — this is a Sail
  plan; it records intent and does not amend binding spec.

---

## 5. Sequencing vs WP-01 (the service upgrade)

**FAIR work and the WP-01 service upgrade are orthogonal.** WP-01 concerns the
library's runtime/service surface; this plan concerns metadata, identifiers, and
community scaffolding. They share no code path and no governed-file contention
(WP-01 does not touch `CITATION.cff`/`README` citation metadata or `codemeta.json`).

Recommended sequencing:

- **Phase A runs alongside WP-01, now.** It is purely additive, touches no frozen
  file, and adds no CI-logic risk to the service work. The highest-value Phase A
  items are `codemeta.json` (one I-gap, fully derivable) and `CONTRIBUTING.md`
  (captures the otherwise-undocumented release procedure).
- **Phase B waits for the `open-iontrap` migration decision.** Minting the
  concept-DOI and writing it into `CITATION.cff`/`README.md` should happen *after*
  the org home is fixed, so the persistent identifier resolves to the final URL and
  the governed edits are made once, not twice. This is the only hard dependency in
  the plan, and it is a *decision* dependency, not a code one.
- **Phase C** begins the first time a tag is cut after `release.yml` lands.

> **One-line recommendation.** Start **Phase A immediately, in parallel with WP-01**
> (lead with `codemeta.json` + `CONTRIBUTING.md`); **hold Phase B** — especially
> the DOI mint and the governed `CITATION.cff`/`README.md` write-backs — until the
> `open-iontrap` migration is decided, then execute it as a single ratified batch.

---

*End of plan. `WP/` governance/planning material under `CC BY-SA 4.0`; see `WP/LICENCE`.
No external endorsement implied.*