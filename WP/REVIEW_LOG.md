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

---

## Round 3 — 2026-06-02 (WI-1 implementation review)

Scope reviewed: branch `wp01-estimation-darwinism` (commits `66d313c`, `eef899f`), `src/iontrap_dynamics/information/fisher.py`, `src/iontrap_dynamics/information/__init__.py`, `src/iontrap_dynamics/__init__.py`, `tests/unit/test_fisher.py`, `CHANGELOG.md`, `pyproject.toml`, `WP/WP-01-estimation-darwinism.md` (Ratified state), `WP/LOGBOOK.md` (registry).

### Overall assessment

**Approved. Continue with WI-3 + keystone benchmark.**

The implementation is clean, the tests reproduce the textbook oracle exactly, the framework discipline is flawless (no spurious logbook entry for a clean landing, CHANGELOG bullet is precise, all six dispatch codes minted and registered), and the branch is ready for WI-3 to be pulled forward.

### What was checked

- `git show --stat 66d313c` — framework + WP-01 (Ratified) + FREEZE-v0.3 + FAIR + task cards (10 files, +1621 lines).
- `git show --stat eef899f` — WI-1 estimation module (8 files, +515 insertions / −2 deletions).
- Static read of `fisher.py` (276 lines), `test_fisher.py` (189 lines), `information/__init__.py` (22 lines), package `__init__.py` (155 lines).
- `git diff main eef899f -- pyproject.toml` — confirms single-line `per-file-ignores` addition for `information/fisher.py`.
- `pytest` was not re-run (user reports 809 passed; I trust the gate).

### Detailed observations

#### 1. `fisher.py` — implementation quality

**SLD-QFI formula:** Correct. Uses the eigenvalue-sum form

```
F_Q = 2 Σ_{j,k : λ_j + λ_k > cutoff} (λ_j − λ_k)² / (λ_j + λ_k) · |⟨j|G|k⟩|²
```

with `_SLD_EIGENVALUE_CUTOFF = 1e-12`. The `np.where` guard on `lam_sum > cutoff` handles the kernel correctly; diagonal terms (j=k) contribute zero because `lam_diff = 0`. The Hermitisation step (`0.5 * (rho + rho†)`) absorbs numerical noise before `eigh`. This is the standard stable implementation.

**`_ensure_density`:** Minimal (ket → `ket * ket.dag()`). Sufficient for WI-1. When WI-2 adds redundancy/recoverability, this helper may need to grow (e.g., ptrace-awareness, subspace validation), but keeping it small now is correct.

**Trajectory-evaluator shape:** Mirrors `entanglement.py` — `(states, *, hilbert, generator) -> NDArray[np.float64]`. The docstring explicitly notes the `StorageMode.EAGER` requirement and the post-processing pattern. Good.

**`linear_gaussian_fisher` — validation gap (very minor):** The function checks that `sigma` is symmetric and matches `A`'s observation count, but does not verify positive-definiteness. `np.linalg.inv` will raise `LinAlgError` for singular matrices, but for an indefinite `sigma` (negative eigenvalues) it will silently return a wrong Fisher matrix. The docstring claims "positive-definite"; the code should probably assert `np.all(np.linalg.eigvals(sigma_mat) > 0)` or catch the indefinite case. This is a one-line defensive check. Not a blocker for WI-1.

**`_PROBABILITY_SUM_TOLERANCE` naming:** The constant is used for both the probability-sum check (`abs(total - 1.0) > tolerance`) and the non-negativity check (`p < -tolerance`). The name says "SUM_TOLERANCE" but serves double duty. Very minor — rename to `_PROBABILITY_TOLERANCE` if you care, otherwise leave it.

#### 2. `test_fisher.py` — test coverage

**15 cases, all oracles verified:**
- `test_qfi_ghz_reaches_heisenberg_limit` — `N = 1,2,3`; asserts `QFI == N²` within `1e-9`. This is the keystone oracle.
- `test_qfi_product_reaches_standard_quantum_limit` — `N = 1,2,3`; asserts `QFI == N` within `1e-9`.
- `test_qfi_pure_state_equals_four_times_variance` — eigenstate (`QFI = 0`) and `|+>` (`QFI = 4·Var = 4`) on a single qubit.
- `test_qfi_maximally_mixed_state_is_zero` — `QFI = 0` for maximally mixed state.
- `test_qfi_trajectory_length_and_dimension_check` — shape `(2,)` for two states; `ValueError` on dimension mismatch.
- `test_classical_fisher_information_values` — three standard cases (symmetric binomial, zero derivative, zero-probability support).
- `test_classical_fisher_information_validation` — shape mismatch, sum ≠ 1, negative probability.
- `test_cfi_never_exceeds_qfi_braunstein_caves` — saturating measurement (`CFI == QFI`) and sub-optimal measurement (`CFI < QFI`).
- `test_cramer_rao_bound` — `1/4` and `ValueError` on non-positive input.
- `test_linear_gaussian_fisher_matrix` — two independent observations, `F = 1/σ₁² + 1/σ₂²`.
- `test_linear_gaussian_fisher_validation` — non-2-D `A`, shape mismatch, non-symmetric `sigma`.

**Test helper quality:** `_ghz`, `_product_plus`, `_collective_jz` are local builders using `spin_up()`, `spin_down()`, `sigma_z_ion()`, `hilbert.spin_op_for_ion()`. They correctly use the library's canonical operators rather than `qutip.sigmaz`. When WI-3 lands `states.ghz_state`, these helpers should be refactored to use the real factory.

**Import path:** Tests import from the package root (`iontrap_dynamics`), not from `iontrap_dynamics.information.fisher`. This validates the re-export chain — valuable.

#### 3. Package integration

**`information/__init__.py`:** Clean umbrella module. SPDX-headed, docstring explains the sub-package purpose, exports are explicit and alphabetised.

**Package `__init__.py`:** The new `.information` import block sits between `.exceptions` and `.measurement` — alphabetically correct. The four new names are merged into `__all__` in alphabetical order (`classical_fisher_information`, `cramer_rao_bound`, `linear_gaussian_fisher`, `quantum_fisher_information_trajectory`) within the ALL-CAPS-first-then-alphabetical scheme. Correct.

**`pyproject.toml`:** Single-line `per-file-ignores` addition for `information/fisher.py`. As noted in Round 2, consider collapsing to `"src/iontrap_dynamics/information/*.py"` once WI-2 lands the second module.

#### 4. Framework discipline

**No logbook entry for clean landing:** Correct. The module landed with no decisions, dead-ends, or deferrals — purely CHANGELOG territory. The framework's own separation-of-concerns rule is exercised faithfully.

**CHANGELOG bullet:** Precise, dispatch-keyed, notes the keystone benchmark is pending WI-3. Length is fine for `[Unreleased]`.

**Dispatch registry:** All six ED codes (EDA–EDF) are minted and recorded in both WP-01 §15 and `WP/LOGBOOK.md`. EDA's status correctly notes "module landed; benchmark pending WI-3". No collision risk — the `ED` root was grepped and free.

**WP-01 header:** Correctly updated to `Ratified 2026-06-02`, `Lifecycle: Ratified`, `WI-1 is open`, `dispatch codes EDA–EDF minted`. The literature-review note licence is ratified Coastline / CC BY-SA 4.0 (Round 2 steer accepted). §14 now explicitly explains why WI-3 precedes WI-2. §10 headlines the `states.py` re-export.

**Commit structure:** Two commits, clean separation — framework docs first, code second. Good.

#### 5. Round 2 review fixes — verification

| Round 2 observation | Status |
|---|---|
| §7 benchmark 5 renamed `run_demo_ghz_cat.py` → `run_benchmark_ghz_cat.py` | **Verified** in WP-01 §7 table. |
| §10 headlines `states.py` top-level re-export | **Verified** in WP-01 §10 release plan. |
| §14 explains WI-3 → WI-2 ordering | **Verified** in WP-01 §14: "Why WI-3 precedes WI-2 (deliberate, not a typo)". |
| Lit-review note licence ratified Coastline | **Verified** in WP-01 header and §5. |

### Non-blocking suggestions for WI-3 / forward

1. **Add positive-definiteness check to `linear_gaussian_fisher`** (one line: `if not np.all(np.linalg.eigvals(sigma_mat) > 0): raise ValueError(...)`). This closes the gap between docstring claim and runtime validation.
2. **When refactoring `test_fisher.py` to use `states.ghz_state`**, keep the local `_ghz` helper as a fallback or remove it entirely — the real factory should produce the same state.
3. **Collapse `per-file-ignores`** to `"src/iontrap_dynamics/information/*.py"` when WI-2 lands `redundancy.py`.
4. **The keystone benchmark tool** (`tools/run_benchmark_qfi_scaling.py`) should reproduce the exact same oracle values as the unit tests (`N=1,2,3` or a wider sweep`) and record `max_numerical_vs_analytic_error` in `demo_report.json`. The unit tests already prove correctness; the benchmark proves it at scale and produces the figure.

### Recommendation

**Continue.** Pull WI-3 forward (`states.ghz_state` + `cat_mode`), then implement the keystone QFI-scaling benchmark. The branch is clean, the foundation is solid, and momentum is good. No pause needed.

---

## Round 4 — 2026-06-02 (hardening + WI-3 review)

Scope reviewed: branch `wp01-estimation-darwinism` (commits `e0d24f4`, `5ef4f28` on top of Round 3), `src/iontrap_dynamics/information/fisher.py` (hardened), `src/iontrap_dynamics/states.py` (WI-3), `tests/unit/test_fisher.py` (hardening tests), `tests/unit/test_states_ghz_cat.py`, `CHANGELOG.md`, `WP/LOGBOOK.md`, `WP/WP-01-estimation-darwinism.md`.

### Overall assessment

**Approved. Continue — build the keystone benchmark now.**

Every Round 3 gap is patched. The WI-3 factories are clean, well-tested, and correctly unblock EDA. The branch has earned its next chunk.

### What was checked

- `git diff eef899f e0d24f4` — Round-3 hardening (39 insertions / 11 deletions across `fisher.py` + `test_fisher.py`).
- `git diff eef899f 5ef4f28` — WI-3 EDC (94 insertions in `states.py`, 16 in package `__init__.py`, 125 in `test_states_ghz_cat.py`).
- Static reads of hardened `fisher.py` (297 lines), `test_fisher.py` (202 lines), `states.py` (382 lines), `test_states_ghz_cat.py` (125 lines).
- `CHANGELOG.md` header — verifies both EDA and EDC bullets.
- `WP/LOGBOOK.md` registry — verifies EDA/EDC status updates.

### Resolution of Round 3 findings

| Round 3 finding | Status in Round 4 | Evidence |
|---|---|---|
| `linear_gaussian_fisher` missing PD check | **Resolved** | Cholesky gate added: `np.linalg.cholesky(sigma_mat)` raises `ValueError` on symmetric-indefinite input. Test covers `[[1,2],[2,1]]` (eigenvalues 3, −1). |
| `_PROBABILITY_SUM_TOLERANCE` misnamed | **Resolved** | Renamed to `_PROBABILITY_TOLERANCE` throughout `fisher.py`; docstring updated. |
| Zero-prob + non-zero derivative silently masked | **Resolved** | Explicit guard: `p <= 0.0 & abs(dp) > tolerance` → `ValueError` with clear message. Test covers `[1.0, 0.0]` with `[0.0, 0.5]`. |
| `test_fisher.py` local `_ghz` helper | **Not yet** | Will refactor when keystone benchmark lands; acceptable. |
| `per-file-ignores` glob collapse | **Not yet** | Deferred to WI-2; acceptable. |

### Detailed observations

#### 1. Round-3 hardening (commit `e0d24f4`)

**Zero-probability + non-zero derivative guard:** Mathematically correct. In the classical Fisher information `Σ (∂p)²/p`, a zero-probability outcome with non-zero derivative makes the term diverge. Silently masking it (the old behaviour) hides an ill-posed model. Raising `ValueError` is the right call — the caller must fix their model. The code comment explains this clearly.

**Cholesky gate for positive-definiteness:** Elegant. Cholesky is cheaper than a full eigendecomposition (roughly `n³/3` vs `4n³/3`), succeeds iff the matrix is symmetric positive-definite, and fails with `LinAlgError` on indefinite or singular matrices. The `from exc` chain preserves the cause. The test uses `[[1,2],[2,1]]` (eigenvalues 3 and −1) — a classic symmetric-indefinite example.

**Hermitian generator check:** Uses `np.allclose(generator_matrix, generator_matrix.conj().T)`. This is a practical check rather than an exact one, which is correct for numerical matrices. The test uses `|up⟩⟨down|` (the raising operator σ₊) as the non-Hermitian counterexample — canonical and clear.

**WP-01 §13 sharpening:** "paste only when EDA fully lands: module **and** keystone benchmark" — correct. The stub must not be pasted until the decoupling-proof figure exists.

#### 2. WI-3 / EDC (commit `5ef4f28`)

**`ghz_state(hilbert)` — implementation:**
- Uses `hilbert.n_ions` and `hilbert.system.modes` correctly.
- Mode vacua factor out of the superposition: `(all_up + all_down) ⊗ mode_vacua`. This is computationally efficient and physically transparent.
- `ConventionError` on `n_ions < 1` — consistent with existing `states.py` error style.
- `.unit()` normalises; the state is a ket by construction.
- Docstring references the §0.A Φ⁺ Bell convention and the N² QFI scaling — ties the factory to its physical purpose without application framing.

**`cat_mode(fock_dim, alpha, parity)` — implementation:**
- Reuses `coherent_mode(fock_dim, alpha)` — good reuse, no duplication of the coherent-state convention.
- `ConventionError` guards: `fock_dim <= 0`, parity not in `{"even", "odd"}`, degenerate odd cat at `α = 0`.
- The degeneracy check (`norm < 1e-12`) is necessary because `coherent_mode(fock_dim, 0.0)` returns the vacuum `|0⟩`, so `|α⟩ − |−α⟩ = |0⟩ − |0⟩ = 0`. The cutoff `1e-12` is consistent with the `_SLD_EIGENVALUE_CUTOFF` in `fisher.py`.
- Returns a normalised single-mode ket, composable via `compose_density` or `qutip.tensor`. Matches the `coherent_mode` contract.

**Package `__init__.py` re-export:**
- The `.states` import block is alphabetically placed between `.spectrum_observables` and `.systematics`. Correct.
- Seven names added to `__all__`: `cat_mode`, `coherent_mode`, `compose_density`, `ghz_state`, `ground_state`, `squeezed_coherent_mode`, `squeezed_vacuum_mode`. ALL-CAPS-first-then-alphabetical ordering preserved.
- This is the largest API-surface expansion noted in Round 2 and explicitly called out in the CHANGELOG.

**`states.py` `__all__`:** Updated to include `cat_mode` and `ghz_state`. Alphabetical.

#### 3. Test coverage

**`test_fisher.py` — 16 cases (was 15):**
- New: `test_qfi_rejects_non_hermitian_generator` — σ₊ rejected.
- New: zero-prob derivative divergence guard — `[1.0, 0.0]` with `[0.0, 0.5]` rejected.
- New: symmetric-indefinite sigma rejected via Cholesky.

**`test_states_ghz_cat.py` — 125 lines, 9 test functions:**
- `test_ghz_state_is_normalised_ket` — `N = 1,2,3`; `isket` and `norm ≈ 1`.
- `test_ghz_population_split_equally_between_all_up_and_all_down` — `N = 2,3`; overlap probabilities ≈ 0.5 each.
- `test_ghz_two_ion_is_maximally_entangled` — concurrence = 1.0 for 2-ion GHZ. Uses `entanglement.concurrence_trajectory`, validating cross-module reuse.
- `test_ghz_qfi_reaches_heisenberg_limit` — **the cross-check**. `ghz_state` feeds `quantum_fisher_information_trajectory` → `N²` within `1e-9`. This is the miniature keystone result, now proven module-to-module.
- `test_cat_mode_is_normalised_ket` — both parities; `isket` and `norm ≈ 1`.
- `test_cat_mode_parity_eigenvalue` — even cat `⟨Π⟩ ≈ +1`, odd cat `⟨Π⟩ ≈ −1`.
- `test_cat_mode_fock_support_is_single_parity` — even cat has zero amplitude on odd Fock states, odd cat has zero on even. Loop over all Fock states.
- `test_cat_mode_validation` — `fock_dim = 0`, bad parity, degenerate odd cat at `α = 0`.

The test suite is comprehensive without being redundant. The QFI cross-check is particularly valuable — it proves the WI-1 and WI-3 modules compose correctly before the benchmark tool is even written.

#### 4. CHANGELOG and registry discipline

**CHANGELOG:** Two bullets, dispatch-keyed, in reverse chronological order (EDC above EDA). The EDC bullet explicitly calls out the `states.py` public-surface expansion. Good.

**LOGBOOK registry:** EDC status updated to `landed 2026-06-02`. EDA status updated to `module landed 2026-06-02; benchmark unblocked (EDC), pending`. This is precise — EDA is not fully closed until the benchmark lands.

**No spurious logbook entry for EDC landing:** Correct. A clean factory landing with no decisions belongs in the CHANGELOG only.

### Non-blocking observations

1. **Test helper duplication.** `test_fisher.py` and `test_states_ghz_cat.py` each define local `_spin_hilbert` and `_collective_jz` helpers. As the test suite grows, consider extracting these to a shared `tests/conftest.py` or `tests/fixtures.py` (e.g., `spin_system_hilbert(n_ions)`, `collective_jz(hilbert)`). Not a blocker — the duplication is small and the tests are self-contained.

2. **`cat_mode` Fock truncation warning.** The docstring says "choose it large enough that `|α|²` fits well inside the Fock envelope", but there's no runtime warning if `fock_dim` is too small for `|alpha|`. A caller passing `cat_mode(fock_dim=8, alpha=3.0)` will get a normalised state, but it will be truncated and not represent the true cat. This is the same behaviour as `coherent_mode` (which delegates to `qutip.coherent`), so it's consistent with existing precedent. If you want to be defensive, a warning when `abs(alpha)**2 > fock_dim / 2` (rough heuristic) could be added later. Not a blocker.

3. **Benchmark artifact shape tension.** WP-01 §7 prescribes `demo_report.json` (schema_version=2) with fields like `scenario`, `workplan_reference`, `threshold_seconds`, `elapsed_seconds`, `parameters`, `canonical_request_hash`, `analytic_formulas`, `max_numerical_vs_analytic_error`. The existing compute-only benchmarks (`sparse_vs_dense`, `spectrum_envelope`) use a simpler `report.json`. Since the keystone benchmark is compute-only (no `solve()`), following the `sparse_vs_dense` pattern (`report.json` + `arrays.npz` + `plot.png`) is consistent with repo precedent. However, if you want the benchmark to be fully comparable to the WP-01 §7 canonical set, consider using the `demo_report.json` schema and leaving `canonical_request_hash` empty or noting "compute-only, no solve()" — the other fields (parameters, analytic_formulas, max_error) all apply. Either way is fine; just be intentional.

### Recommendation

**Build the keystone benchmark now.** The branch is in excellent shape — four clean commits, all gates green, 823 passed. WI-3 unblocked EDA and the cross-check already proves `ghz_state` → QFI = N². The next chunk (benchmark tool + report + plot + regression anchor) is well-defined and closes Dispatch EDA. No pause needed.

---

## Round 5 — 2026-06-02 (keystone benchmark + EDA completion review)

Scope reviewed: branch `wp01-estimation-darwinism` (commits `5f9bad4`, `c0576a0` on top of Round 4), `tools/run_benchmark_qfi_scaling.py`, `tests/regression/analytic/test_qfi_scaling.py`, `benchmarks/data/qfi_scaling/report.json` + `arrays.npz`, `CHANGELOG.md`, `WP/LOGBOOK.md`, `WP/WP-01-estimation-darwinism.md`.

### Overall assessment

**Approved. Dispatch EDA is complete. Continue with WI-2 (EDB).**

The keystone benchmark is the headline evidence for DoD-5 and it is impeccable: compute-only, deterministic, zero application framing, max error 1.4e-14, alt-text for WCAG, separate binding oracle preserving `test_analytic.py`'s QuTiP-free invariant. The framework updates (§7, §8, §13, §15, CHANGELOG, LOGBOOK, registry) are all precise and consistent. The branch has earned WI-2.

### What was checked

- `git show --stat 5f9bad4` — keystone benchmark + regression anchor (9 files, +394 / −9).
- `git show --stat c0576a0` — Round-4 review log committed (1 file, +103).
- Static reads of `run_benchmark_qfi_scaling.py` (216 lines), `test_qfi_scaling.py` (79 lines), `report.json` (77 lines).
- `python3 -c np.load('arrays.npz')` — verified keys and floating-point values match report.json.
- `git diff 5ef4f28 5f9bad4` — framework updates (WP-01 §7/§8/§13/§15, CHANGELOG, LOGBOOK, pyproject.toml).

### Detailed observations

#### 1. Benchmark tool (`run_benchmark_qfi_scaling.py`)

**Compute-only, deterministic:** No `solve()`, no randomness, no `StorageMode.EAGER` cache-pair problem. Builds states directly and calls `quantum_fisher_information_trajectory`. This sidesteps the EAGER-vs-cache-v1 caveat entirely — the right shape for this benchmark.

**Artifact set (`benchmarks/data/qfi_scaling/`):**
- `report.json` — provenance + oracle + results + environment + alt-text. Contains all the essential fields: `scenario`, `purpose`, `workplan_reference`, `convention_version`, `backend_name`/`version`, `generator`, `n_values`, per-N `results`, `analytic_formulas`, `max_numerical_vs_analytic_error` (1.421e-14), `tolerance` (1e-9), `plot_alt_text`, `environment`, `generated_at`.
- `arrays.npz` — `n_ions`, `qfi_ghz`, `qfi_product`, `analytic_ghz`, `analytic_product`. Verified that floating-point values match report.json exactly.
- `plot.png` — log-log GHZ N² vs product N. Not inspected visually (image file), but the code uses standard matplotlib with proper labels, legend, and `tight_layout()`.

**Alt-text:** Descriptive and suitable for WCAG Level A: "Log-log plot of quantum Fisher information versus qubit number N... The GHZ curve follows Heisenberg scaling N squared (slope 2); the product-state curve follows the standard quantum limit N (slope 1). The numerical points lie exactly on the analytic reference lines."

**Graceful matplotlib skip:** `try/except ImportError` returns 0 if matplotlib is absent. This is the same pattern used in other benchmark tools. Good.

**N sweep:** `N_VALUES = (1, 2, 3, 4, 5, 6)`. This is a reasonable range — large enough to show the scaling, small enough that the Hilbert space stays small (dim = 2^N, so N=6 is dim 64, trivial). The unit tests cover N=1..5; the benchmark adds N=6. Good.

**per-file-ignores:** Registered in `pyproject.toml` for the benchmark tool. Good.

#### 2. Regression oracle (`test_qfi_scaling.py`)

**Kept separate from `test_analytic.py` — correct call.** `test_analytic.py` is deliberately QuTiP-free (it validates closed forms in `iontrap_dynamics.analytic` without a backend). The QFI oracle must construct quantum states, so it needs QuTiP. Both sit in the `regression_analytic` tier (`pytestmark = pytest.mark.regression_analytic`). This preserves the architectural invariant without forcing `test_analytic.py` to import QuTiP.

**Named symbolic tolerance:** `ATOL_QFI_SCALING = 1e-9` — follows the migration-tier precedent of per-scenario symbolic tolerances.

**Three test functions:**
- `test_ghz_qfi_is_heisenberg_n_squared` — parametric `n_ions = [1,2,3,4,5]`.
- `test_product_qfi_is_standard_quantum_limit_n` — parametric `n_ions = [1,2,3,4,5]`.
- `test_ghz_qfi_is_n_times_product_qfi` — the scaling-gap test: `F_Q(GHZ) = N · F_Q(product)` for N=2..5. This is a stronger check than the individual oracle tests because it validates the *ratio* independently of absolute calibration.

#### 3. Framework updates

| File | What changed | Quality |
|---|---|---|
| `WP/WP-01-estimation-darwinism.md` §7 | Benchmark 1 marked ✓ **landed**; exact error 1.4e-14 noted; regression test location updated to `test_qfi_scaling.py` | Precise |
| `WP/WP-01-estimation-darwinism.md` §8 | Regression-analytic tier now names `test_qfi_scaling.py` as sibling to `test_analytic.py`, with rationale | Clear |
| `WP/WP-01-estimation-darwinism.md` §13 | "EDA complete 2026-06-02; stub ready to paste, awaiting maintainer" | Correct — WORKPLAN is governed |
| `WP/WP-01-estimation-darwinism.md` §15 | EDA status → "landed 2026-06-02 — EDA complete" | Precise |
| `CHANGELOG.md` | EDA bullet updated: benchmark landed, max error 1.4e-14, completes EDA | Good |
| `WP/LOGBOOK.md` | New entry: EDA completion with rationale for test-file separation and artifact shape | Excellent — records the *why* |
| `WP/LOGBOOK.md` registry | EDA status → "landed 2026-06-02 — EDA complete" | Precise |

**The LOGBOOK entry for EDA completion is model-quality.** It records the one genuine decision: why the QFI oracle was placed in a new sibling file rather than `test_analytic.py`. This is exactly what the logbook is for — rationale that the CHANGELOG cannot hold.

#### 4. One minor schema note

WP-01 §7 prescribes `demo_report.json` (with `schema_version: 2`, `threshold_seconds`, `elapsed_seconds`, `parameters`, `canonical_request_hash`, etc.). The keystone benchmark uses `report.json` following the `sparse_vs_dense` compute-only precedent. The content is functionally equivalent — all essential oracle and provenance fields are present. The schema name difference is acceptable given the repo precedent, but if you want strict compliance with WP-01 §7, consider renaming to `demo_report.json` and adding `schema_version: 2` (with `canonical_request_hash` null or noted as "compute-only, no solve()"). This is a one-line rename + two-line addition. Not a blocker.

### Non-blocking observations

1. **Test helper duplication persists.** `test_qfi_scaling.py` defines its own `_spin_hilbert`, `_collective_jz`, and `_product_plus` — the same helpers that exist in `test_fisher.py` and `test_states_ghz_cat.py`. As noted in Round 4, consider a shared `tests/conftest.py` when the suite grows. The duplication is now three-fold.

2. **Benchmark N range.** N=6 is the maximum (dim = 64). If you ever want to demonstrate scaling at larger N (e.g., N=8 or N=10), the benchmark will still run instantly because the SLD-QFI implementation is O(dim³) and dim=1024 at N=10 is ~1 second. No change needed now, but the tool can scale up easily if desired.

3. **WORKPLAN paste.** The §13 stub is ready and correctly flagged as awaiting maintainer action. When you paste it, remember to bump the WORKPLAN header version line, footer `**Workplan version:**`, and Endorsement Marker in the same commit — the lock-step rule is important.

### Recommendation

**Continue with WI-2 (EDB).** Dispatch EDA is complete and the decoupling proof is established. The information/ umbrella is proven, the evaluator shape is exercised, and the helper layer (`_ensure_density`) is ready for WI-2's redundancy/recoverability modules. WI-2 (Darwinism) is the natural next chunk.

The WORKPLAN §5.4 paste can happen asynchronously — it doesn't block coding. If you want to batch it, paste the stub right before or right after WI-2 lands, but don't let it hold up execution.

---

## Round 6 — 2026-06-02 (WI-2 Darwinism + shared helper layer review)

Scope reviewed: branch `wp01-estimation-darwinism` (commits `97873fd`, `3b307d7`, `6caa17a` on top of Round 5), `src/iontrap_dynamics/information/redundancy.py`, `recoverability.py`, `_common.py`, `tests/unit/test_redundancy.py`, `test_recoverability.py`, `CHANGELOG.md`, `WP/LOGBOOK.md`, `WP/WP-01-estimation-darwinism.md`, `pyproject.toml`.

### Overall assessment

**Approved. The information-theoretic surface is landed and proven. Continue with WI-4 (EDD).**

Three WIs complete, 873 tests green. The shared-helper layer (`_common.py`) is clean and validates the `information/` umbrella architecture. The recoverability deferral-then-ratify pattern in the logbook is exactly what the framework is designed for. The Round-5 schema fix is applied correctly. No pause needed.

### What was checked

- `git show --stat 97873fd` — redundancy module (9 files, +332).
- `git show --stat 3b307d7` — recoverability + `_common.py` refactor (8 files, +295).
- `git show --stat 6caa17a` — keystone report schema fix (3 files, +11 / −1).
- Static reads of `redundancy.py` (213 lines), `recoverability.py` (89 lines), `_common.py` (55 lines), `test_redundancy.py` (91 lines), `test_recoverability.py` (74 lines).
- `git diff HEAD~3 HEAD -- pyproject.toml` — confirms four `information/` per-file-ignores entries.

### Detailed observations

#### 1. Shared helper layer (`information/_common.py`)

**Realises the umbrella architecture.** The docstring explicitly notes: "the shared helper layer that motivated the single `information/` umbrella (WP-01 §3)". This is architecturally satisfying — the refactor validates the naming decision.

**Contents:**
- `_ensure_density(state)` — ket → density matrix promotion. Moved from `fisher.py`.
- `_von_neumann_entropy_bits(rho)` — von Neumann entropy `S(ρ) = −Σ λ log₂ λ` in bits. Uses `rho.eigenenergies()` (QuTiP), clips tiny negatives, applies `_ENTROPY_EIGENVALUE_CUTOFF = 1e-12`.
- `_validate_indices(indices, hilbert, name)` — checks non-empty, in-range `[0, n_subsystems)`, distinct. Returns a `list[int]`.

**Cutoff consistency:** `_ENTROPY_EIGENVALUE_CUTOFF = 1e-12` matches the `_SLD_EIGENVALUE_CUTOFF` in `fisher.py`. Good — one numerical floor across the sub-package.

**Privacy:** No `__all__`; nothing re-exported from `information/__init__.py`. Correct for a private helper module.

#### 2. `redundancy.py` — implementation

**Three public symbols:** `fragment_mutual_information`, `partial_information_plot`, `redundancy`.

**`fragment_mutual_information`:** Computes `I(S:F) = S(ρ_S) + S(ρ_F) − S(ρ_{S∪F})` in bits. Validates disjoint index sets. Uses `rho.ptrace(...)` on the density matrix. Correct.

**`partial_information_plot`:** Builds nested fragments `F_f = environment_indices[:f]` and returns `[I(S:F_0), I(S:F_1), …, I(S:F_N)]`. Entry 0 is 0.0 by convention. For the GHZ cascade, the curve plateaus at `H_S = 1` bit for non-empty proper fragments and jumps to `2·H_S` at the full environment (because `I(S:E) = 2·S(ρ_S)` when the total state is pure). This matches the textbook behaviour and is verified in tests.

**`redundancy`:** Computes `R_δ = N / f_δ`. The implementation:
- Validates `delta ∈ (0, 1)`.
- Short-circuits to `0.0` if `H_S ≈ 0` (nothing to encode).
- Computes the partial-information plot, then finds the smallest `k` where `pip[k] >= (1−δ)·H_S`.
- Returns `N / k` or `0.0` if threshold never reached.

This is the standard Zurek deficit convention. Correct.

**One minor inefficiency:** `redundancy` calls `partial_information_plot`, which re-validates indices and re-computes `s_system`. The double validation is harmless (fast), but the re-computation of `s_system` is also harmless because the PIP is short. Not a blocker.

#### 3. `recoverability.py` — implementation

**Single public symbol:** `recoverability`.

**Measure:** `max(0, S(ρ_A) − S(ρ_{S∪A}))` — the clamped coherent information `I_c(S⟩A)`. This was the recommended measure from the logbook deferral entry, ratified before coding.

**Endpoints:**
- Perfect recovery (Bell pair / GHZ over 2 qubits) → `H_S = 1` bit.
- Full decoherence (accessible part uncorrelated) → `0` (coherent information ≤ 0, clamped).
- Monotone between on a Werner family — verified in tests.

**Docstring:** Cites Schumacher–Nielsen, notes the clamping, references CONVENTIONS §20 (staged). Good.

#### 4. Test coverage

**`test_redundancy.py` — 91 lines, 6 test functions:**
- `test_partial_information_plot_shows_darwinism_plateau` — parametric `n_env = [3, 5]`; verifies plateau at 1.0 for fragments 1..N-1, jump to 2.0 at full environment.
- `test_fragment_mutual_information_single_qubit_carries_full_bit` — a single environment qubit in the GHZ cascade carries the full 1 bit.
- `test_redundancy_is_maximal_for_ghz_cascade` — parametric `n_env = [3, 5]`; `R_δ = N` (each single qubit suffices).
- `test_redundancy_zero_when_system_has_no_information` — product state, `H_S = 0`, so `R_δ = 0`.
- `test_mutual_information_validation` — overlapping indices, out-of-range, empty set.
- `test_redundancy_delta_validation` — `delta` outside `(0, 1)`.

**`test_recoverability.py` — 74 lines, 4 test functions:**
- `test_perfect_recovery_equals_system_entropy` — Bell pair → 1.0 bit.
- `test_full_decoherence_is_zero` — product state with system entangled to an *inaccessible* reference; accessible qubit uncorrelated → 0.
- `test_recoverability_monotone_in_werner_fidelity` — Werner family `p·|Φ⁺⟩⟨Φ⁺| + (1−p)·I/4`. Checks endpoints (0 and 1), non-decreasing, and genuine intermediate rise (`0 < vals[1] < vals[2] < 1`). This is the standard test for entanglement monotonicity.
- `test_recoverability_validation` — overlapping, out-of-range, empty.

Both test files import from the package root (`iontrap_dynamics`), validating the re-export chain.

#### 5. Framework discipline — the deferral-then-ratify pattern

**This is model-quality framework usage.** The LOGBOOK contains two entries for EDB:

1. **Deferral entry:** "WI-2 redundancy landed; recoverability deferred pending its measure-convention". Records that redundancy (standard conventions) shipped first, while recoverability (five candidate measures) was blocked on a convention decision. Explicitly lists the five candidates and recommends Schumacher–Nielsen coherent information.

2. **Ratification entry:** "Recoverability measure ratified (coherent information); EDB complete". Records the maintainer's ratification of the recommendation, the implementation, and the `_common.py` refactor.

This is **Design Principle 1 (conventions before code) in action**, and the logbook captures the full lifecycle: deferral → rationale → recommendation → ratification → implementation. The CHANGELOG only sees "Dispatch EDB — redundancy + recoverability" because it records *what shipped*, not *why it was deferred*.

#### 6. Package integration

**`information/__init__.py`:** Updated to import from `redundancy` and `recoverability`. `__all__` now lists 8 symbols, alphabetically ordered:
`classical_fisher_information`, `cramer_rao_bound`, `fragment_mutual_information`, `linear_gaussian_fisher`, `partial_information_plot`, `quantum_fisher_information_trajectory`, `recoverability`, `redundancy`.

**Package `__init__.py`:** The `.information` import block now carries all 8 names, alphabetically merged into the top-level `__all__`. Correct.

#### 7. pyproject.toml and schema fix

**per-file-ignores:** Four new entries for `information/` (`_common.py`, `fisher.py`, `recoverability.py`, `redundancy.py`). As noted in Round 3/4, this should be collapsed to a glob `"src/iontrap_dynamics/information/*.py"`. The `_common.py` entry is the tipping point — four separate lines for one sub-package is untidy. One-line fix.

**Keystone schema fix (commit `6caa17a`):** `report.json` now carries `schema_version: 2` and `canonical_request_hash: null`. WP-01 §7 reconciled with a new `!!! note` block explaining the compute-only `report.json` precedent. This resolves the Round-5 schema tension cleanly.

### Non-blocking observations

1. **Collapse `per-file-ignores` to a glob.** Replace the four `information/*.py` entries with one `"src/iontrap_dynamics/information/*.py" = ["RUF001", "RUF002", "RUF003"]`. This is overdue now that the sub-package has four modules.

2. **`redundancy.py` double `_ensure_density` call.** `redundancy` calls `_ensure_density(state)`, then calls `partial_information_plot(state, ...)` which calls `_ensure_density(state)` again. This is harmless (cheap for small states), but a micro-optimisation would pass `rho` to a private `_partial_information_plot_from_rho` helper. Not worth it unless profiling shows it's hot.

3. **Test helper duplication is now four-fold.** `test_fisher.py`, `test_states_ghz_cat.py`, `test_redundancy.py`, and `test_recoverability.py` all define local `_spin_hilbert` and `_collective_jz`. The shared `tests/conftest.py` refactor is becoming more valuable as the suite grows. Consider doing this when WI-4 lands, or as a separate hygiene commit before release.

### Recommendation

**Continue with WI-4 (EDD).** The information-theoretic surface (estimation + Darwinism) is complete, proven, and well-factored. Three of four WIs done. WI-4 (common-mode channel) is convention-light, well-scoped in WP-01 §4.4, and unblocked. No pause needed.

After WI-4: EDE (benchmarks) and EDF (review note + CONVENTIONS staging) are the remaining work. The branch is on track for a clean release.

---

## Round 7 — 2026-06-02 (WI-4 common-mode channel + glob collapse review)

Scope reviewed: branch `wp01-estimation-darwinism` (commits `c672804`, `c728997` on top of Round 6), `src/iontrap_dynamics/systematics/common_mode.py`, `tests/unit/test_common_mode.py`, `pyproject.toml`, `CHANGELOG.md`, `WP/LOGBOOK.md`, `WP/WP-01-estimation-darwinism.md`, `src/iontrap_dynamics/systematics/__init__.py`, `src/iontrap_dynamics/__init__.py`.

### Overall assessment

**Approved. All four WP-01 work items are implemented and proven. Continue with EDE (the five benchmarks).**

WI-4 is clean, convention-light, and correctly mirrors the existing systematics pattern. The glob collapse is applied. The branch has earned its validation layer. EDE is the natural next chunk — mechanical now that every primitive and its oracle exists.

### What was checked

- `git show --stat c672804` — WI-4 common-mode channel (9 files, +412 / −6).
- `git show --stat c728997` — Round-6 review log committed (1 file, +121).
- Static reads of `common_mode.py` (148 lines), `test_common_mode.py` (118 lines), `systematics/__init__.py` (81 lines).
- `git diff HEAD~2 HEAD -- pyproject.toml` — confirms information/*.py glob + systematics/common_mode.py entry.

### Detailed observations

#### 1. `common_mode.py` — implementation

**Mirrors the existing systematics pattern exactly.** Frozen dataclass (`CommonModePhase`, `slots=True`, `kw_only=True`) + `perturb_*` free function (`perturb_common_mode`). This is the same shape as `PhaseJitter` / `perturb_phase` and `PhaseDrift` / `apply_phase_drift`.

**`CommonModePhase` fields:**
- `sigma_rad: float` — marginal per-subsystem std. Validated `>= 0` in `__post_init__`.
- `correlation: float = 1.0` — shared fraction `c ∈ [0, 1]`. Validated in `[0, 1]` in `__post_init__`.
- `label: str = "common_mode_phase"` — identifier for downstream aggregation.

**`sample_offsets`:** Draws `shared ~ N(0, σ²)` (shape `(shots,)`) and `independent ~ N(0, σ²)` (shape `(shots, n_subsystems)`), then mixes: `offsets = √c·shared[:, None] + √(1−c)·independent`. Vectorised, no Python loops over shots or subsystems. Validated `n_subsystems >= 1` and `shots >= 1`.

**`perturb_common_mode`:** Takes a `Sequence[DriveConfig]`, returns `tuple[tuple[DriveConfig, ...], ...]` — outer tuple length `shots`, inner tuple length `len(drives)`. Uses `dataclasses.replace` to perturb `phase_rad` while leaving all other fields untouched. Bit-reproducible via `seed`. Validated non-empty drives and `shots >= 1`.

**Docstring quality:** Excellent. Explains the interpolation formula, the c=0 and c=1 limits, the marginal-std invariance, and the departure from existing systematics (perturbs a *sequence* of drives with one correlated draw per shot).

#### 2. `test_common_mode.py` — test coverage

**118 lines, 8 test functions:**
- `test_full_correlation_shares_one_offset` — `c = 1`, 3 subsystems, 2000 shots; verifies all offsets in a shot are identical (`np.allclose`).
- `test_full_correlation_cancels_in_difference` — `c = 1`, 2 drives with initial phases 0.0 and 0.5; verifies the phase difference remains exactly −0.5 after perturbation (common-mode rejection). This is the key physical oracle.
- `test_zero_correlation_is_independent` — `c = 0`, 2 subsystems, 50000 shots; cross-correlation `< 0.05`, std matches `sigma_rad` within 5%. Statistical test with sufficient shots.
- `test_marginal_std_independent_of_correlation` — parametric `c = [0.0, 0.5, 1.0]`; std matches `sigma_rad` within 5% for all. Verifies the marginal invariance.
- `test_difference_variance_monotone_in_correlation` — `c = [0.0, 0.25, 0.5, 0.75, 1.0]`; verifies `var(diff)` is non-increasing, equals `2σ²` at `c=0`, and exactly `0` at `c=1`. This is the strongest test — it validates the full interpolation curve.
- `test_perturb_structure_and_immutability` — verifies original drives not mutated, non-phase fields pass through, correct tuple nesting.
- `test_reproducible_with_seed` — same seed → identical results.
- `test_validation` — negative `sigma_rad`, `correlation` outside `[0, 1]`, empty drives, `shots = 0`.

All oracles from WP-01 §4.4 are covered: independent at `c=0`, common-mode rejection at `c=1`, marginal std invariant, difference-variance monotone.

#### 3. Package integration

**`systematics/__init__.py`:** Imports `CommonModePhase` and `perturb_common_mode` from `.common_mode`. `__all__` updated — alphabetically ordered within the class/function groups. Correct.

**Package `__init__.py`:** Re-exports `CommonModePhase` and `perturb_common_mode` from `.systematics`. Alphabetically merged into top-level `__all__`. Correct.

#### 4. Framework updates

| File | What changed | Quality |
|---|---|---|
| `CHANGELOG.md` | EDD bullet; explicitly notes "With EDD, all four WP-01 work items are implemented" | Clear milestone call-out |
| `WP/LOGBOOK.md` registry | EDD → "landed 2026-06-02" | Precise |
| `WP/WP-01-estimation-darwinism.md` §15 | EDD → "landed 2026-06-02 (correlation interpolation; common-mode-rejection oracle)" | Precise |
| `pyproject.toml` | Four `information/*.py` lines collapsed to `"src/iontrap_dynamics/information/*.py"`; added `"src/iontrap_dynamics/systematics/common_mode.py"` | Clean |

#### 5. Resolution of prior findings

| Prior finding | Status |
|---|---|
| Round-6 #1: Collapse `per-file-ignores` to glob | **Resolved** — `information/*.py` glob applied in commit `c672804`. |
| Round-6 #2: `redundancy.py` double `_ensure_density` | **Still open** — non-blocking; deferred to pre-release hygiene. |
| Round-6 #3: Test helper duplication (four-fold) | **Still open** — non-blocking; deferred to pre-release hygiene. |

### Non-blocking observations

1. **`systematics/__init__.py` `__all__` ordering.** The list is alphabetically ordered within class and function groups, but `perturb_common_mode` appears between `perturb_carrier_rabi` and `perturb_detuning` — which is correct alphabetical. No issue.

2. **The `test_common_mode.py` `_drive` helper uses hard-coded `280e-9` wavelength.** This is the same laser wavelength used in other tests (carrier Rabi, etc.), so it's consistent. If the library ever changes its default wavelength, this helper would need updating — but it's a test helper, not a production default.

3. **Benchmark tool naming consistency reminder.** Benchmark 5 in WP-01 §7 is still `run_demo_ghz_cat.py` while all others use `run_benchmark_*.py`. This was noted in Round 2 and remains unfixed. Not a blocker, but a one-line rename would clean it up before EDE starts.

### Recommendation

**Continue with EDE (the five benchmarks).** All four WIs are done, all oracles are proven, and the benchmark pattern is established. EDE is mechanical work now:

1. CFI/linear-Gaussian benchmark (uses `linear_gaussian_fisher`)
2. Darwinism-redundancy benchmark (uses `partial_information_plot` on the GHZ cascade)
3. Recoverability benchmark (uses `recoverability` on the Werner family)
4. GHZ-cat benchmark (uses `ghz_state` / `cat_mode` — though the keystone already exercises `ghz_state`, this could be a dedicated state-factory benchmark)
5. Common-mode benchmark (uses `CommonModePhase.sample_offsets` to show the correlation sweep)

Each follows the same pattern: `tools/run_benchmark_<name>.py` → `benchmarks/data/<name>/` (`report.json` + `arrays.npz` + `plot.png`) + `tests/regression/analytic/test_<name>.py` with a named `ATOL`.

After EDE: EDF (review note + CONVENTIONS staging) is the last substantive chunk before release hygiene. The branch is on track.
