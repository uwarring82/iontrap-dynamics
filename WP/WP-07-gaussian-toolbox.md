# WP-07 — Multimode Gaussian-state / covariance toolbox

**Build the application-independent Gaussian covariance/symplectic layer — covariance `V`, Williamson spectrum, symplectic congruence, arbitrary-cut log-negativity, effective temperature — behind a single sealed multimode convention (§27), reusing WP-05's §26.2 quadrature normalisation.**

Version 0.1 · Drafted 2026-07-14 · Ratified and sealed 2026-07-14 · Status: **Ratified** (GT1 landed + §27 sealed at `CONVENTION_VERSION` 0.6, 2026-07-14; GT2 purity/entropy + GT4 log-negativity/separability landed 2026-07-15; GT3/GT5/GT6 open)

**Classification:** Sail execution under Coastline gates (per T(h)reehouse +EC CD 0.9).
**Licence:** This WP document itself is CC BY-SA 4.0 (`WP/LICENCE`). Deliverables carry their layer's licence: code = MIT (`src/`, `tests/`, `tools/`); the §27 CONVENTIONS edit = Coastline / CC BY-SA 4.0 (staged in `WP/GT-conventions-proposal.md`, maintainer-sealed).
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (`cd-rules`, consumed via Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

---

## 1. Card linkage *(Sail)*

Executes `task cards/TC-gaussian-entanglement-toolbox.md` (**ID: TC-gaussian-entanglement-toolbox**, v0.2, 2026-07-09 — adversarially verified + maintainer v0.2 review; all §7 open questions carry provisional decisions).

**Objective lifted from the card (one line):** ship an application-agnostic Gaussian covariance/symplectic toolbox (covariance `V`, symplectic spectrum, congruence, log-negativity, `T_eff`) reusable by the existing two-mode squeezing machinery **and** the future two-ion WP-SQ Phase B, sealing the multimode convention **once** as §27.

**Governing invariants from the card — quoted, not paraphrased** (hard acceptance gates):

> The toolbox carries **no** consuming-application framing (no "Wittemer", no "Hawking", no "cosmology") — enforced by the decoupling grep.

> The PPT / `ν̃_k ≥ 1` criterion certifies separability **only for `1×N` Gaussian cuts**. For general `M×N` cuts, PPT-bound-entangled Gaussian states exist, so `E_N = 0` does **not** certify separability — scope the separability-certification API to `1×N`; for `M×N` expose `E_N` as an NPT witness with the caveat surfaced.

> `V` **alone cannot certify Gaussianity** — the toolbox must state the precondition and guard it.

The card's provisional decisions (carried into §4): per-mode ordering `R = (x̂₁,p̂₁,…)`; symplectic eigenvalues via `|eig(iΩV)|` deduplicated (not SVD); log-negativity `Σ_k max(0,−log₂ ν̃_k)`; `n̄ = (tr V + dᵀd − 2)/4` (not `(ν−1)/2`); the `N=1` core lives in `gaussian.py` (created by WP-05 A2), this WP **generalises it to multimode** (extension, not fork); the ion `S`-adapter is a separate application-specific piece; generic Gaussian `E_F` deferred (Supplemental-gated).

## 2. WORKPLAN linkage *(Coastline gate — pending maintainer seal)*

This WP feeds a new dispatch track to be recorded in `WORKPLAN_v0.3.md` as the next free amendment **§5.10** (`*(Coastline, new in v0.3.12)*`).

- **Amendment §:** §5.10 — *multimode Gaussian toolbox* (pasted at ratification).
- **Doc version bump:** v0.3.11 → v0.3.12 (header line, footer `**Workplan version:**`, Endorsement Marker in lock-step).
- **Convention seal:** **§27** (multimode ordering + `Ω` + partial transpose) is staged in `WP/GT-conventions-proposal.md` (propose-don't-apply); the seal bumps **`CONVENTION_VERSION` 0.5 → 0.6**. §27 **reuses** frozen §26.2 (does not redefine the single-mode normalisation) and cross-refs frozen §23 (the TMSV oracle) — no frozen-section edit. All maintainer acts at ratification, after a green conventions test.

## 3. Objective and scope *(Sail)*

**Objective (execution terms):** a `gaussian.py` covariance/symplectic layer generalising WP-05's single-mode core to `N` modes — `V` + first moments, the Williamson spectrum (purity/entropy), the generic symplectic congruence `S V Sᵀ`, arbitrary-cut Gaussian log-negativity, and a neutral effective-temperature helper — validated against the existing squeezed-state factories as exact oracles, behind one sealed §27 convention. "Done" = every functional reproduces its closed-form oracle, the conventions test is green, and the decoupling grep is clean.

**In scope.** GT1 §27 seal + the convention primitives (symplectic form, multimode covariance, physicality guard, symplectic spectrum, partial transpose); GT2 purity + entropy; GT3 congruence + ion `S`-adapter; GT4 log-negativity (`1×N` separability / `M×N` NPT-witness) + cut semantics; GT5 effective temperature; GT6 (optional) locally-symmetric `E_F`.

**Out of scope (explicitly, from the card).** Generic (non-symmetric) Gaussian `E_F` (optimisation; Supplemental-gated — deferred); any `T_H`/"Hawking" symbol (consuming-application framing, owned by WP-SQ Phase B); the normal-mode spectrum itself (`M_ij` / eigenvectors — supplied by sibling `iontrap-structure`); re-implementing the single-mode `gaussian.py` core (this WP extends it).

## 4. Work items *(Sail)*

| WI | Module (proposed) | Key contents | Reuse | Acceptance | Dispatch | Status |
|---|---|---|---|---|---|---|
| **GT1** | `gaussian.py` + `WP/GT-conventions-proposal.md` | **§27 conventions gate + the low-level convention primitives:** `symplectic_form(N)=⊕J`; multimode `covariance_matrix(state)` → `V` (2N×2N) + first moments `d` (per-mode ordering; `N` inferred from `state.dims`; **the existing N=1 call/return contract preserved**); `is_physical(V)` (the `V+iΩ≥0` Hermitian-PSD guard via `eigvalsh`); `symplectic_eigenvalues(V)` = `|eig(iΩV)|`, one per mode (positive half; **no tolerance-dedup that could merge degenerate modes**); `partial_transpose(V, mode_indices)` (flips `p̂_B`) | extends the `N=1` `gaussian.py` core; may wrap `qutip.continuous_variables.covariance_matrix` with §26.2 basis ops | vacuum `V=𝟙_{2N}` and `V+iΩ≥0`; thermal `ν=2n̄+1`; TMSV PT `ν̃₋=e^{−2r}`; the indefinite `\|eig(iΩV)\|≥1` counterexample is rejected by the PSD guard; **conventions test green** (the §27 seal follows) | `GT1` | **landed 2026-07-14** |
| **GT2** | `gaussian.py` | `purity(V)` = `∏ 1/ν_i` = `1/√det V`; `gaussian_entropy_bits(V)` = `Σ g(ν_i)`, `g(ν)=(ν+1)/2·log₂((ν+1)/2) − (ν−1)/2·log₂((ν−1)/2)`, `g(1)=0` — **consumes GT1's `symplectic_eigenvalues`** (multiplicity preserved) | `information/_common._von_neumann_entropy_bits` (bits convention) | vacuum purity 1 / entropy 0; thermal entropy agrees with `_von_neumann_entropy_bits` on truncated ρ; squeezing/displacement leave `μ`,`S` invariant; degenerate ν kept; **unphysical/malformed `V` rejected** (`V+iΩ≱0`, non-real/finite/symmetric) | `GT2` | **landed 2026-07-15** |
| **GT3** | `gaussian.py` + ion `S`-adapter (separate) | generic `congruence(S, V)` with `SΩSᵀ=Ω` guard; **ion-specific `S`** (normal→local from a complete basis + cross-mode orthogonality + masses + local frequencies) in a separate adapter near `iontrap-structure` | `ModeConfig.eigenvector_per_ion` (necessary-not-sufficient) | **squeezed** normal-mode product → correct ion-cut `E_N`; `S⁻¹` round-trip; non-orthogonal (COM vs stretch) case; `SΩSᵀ=Ω` verified | `GT3` | open |
| **GT4** | `gaussian.py` | `log_negativity(V, mode_indices)` = `Σ_k max(0,−log₂ ν̃_k)` (**full-sum**, bits) + `is_separable(V, mode_indices)` (the `1×N`-scoped cut semantics) — **consumes GT1's `partial_transpose` + `symplectic_eigenvalues`** | Fock `qutip.negativity(logarithmic)` (exact cross-check; QuTiP's covariance `logarithmic_negativity` is broken for this convention, **not** used) | TMSV → `E_N=2r/ln2` (+ exact Fock cross-check); separable/thermal → 0; symmetric in the cut; multimode `1×2`; **`1×N` separability certified**, `M×N (≥2)` → `is_separable` **raises** (PPT-bound-entangled caveat surfaced) | `GT4` | **landed 2026-07-15** |
| **GT5** | `gaussian.py` | `effective_temperature(nbar, omega_loc)` with `n̄=(tr V_red + dᵀd − 2)/4` (first-moment-aware); explicit `ω_loc`; `T_eff(0)=0`, reject `n̄<0`; neutral symbol | `qutip.thermal_dm` (round-trip oracle) | thermal round-trip `n̄→T_eff→n̄`; a **squeezed and/or displaced** marginal does **not** give `T_eff=0`; energy-equivalent framing documented | `GT5` | open |
| **GT6** | `gaussian.py` | *(optional)* locally-symmetric Gaussian `E_F` with the **`ν̃₋<1` gate** and a guarded local-symmetry precondition | — | symmetric TMSV → entropy of entanglement; separable → 0. *Generic `E_F` deferred.* | `GT6` | open |

## 5. Sequencing and gates *(Coastline gate)*

**Order:** ratify → mint `GT`, stage §27 seal → **GT1** (§27 seal + the convention primitives: covariance, symplectic form, physicality, symplectic spectrum, PT — conventions-before-code, nothing consumes `V` before the seal) → GT2 (purity/entropy) → GT4 (log-negativity, the headline consumer functional) → GT5 (`T_eff`) → GT3 (congruence + ion `S`-adapter — the two-ion Phase B dependency) → GT6 (optional `E_F`). GT2/GT4/GT5 are pure `gaussian.py` linear algebra over GT1's spectrum + PT and can parallelise once GT1 lands. **GT1 landed and §27 sealed 2026-07-14** (conventions test green; `CONVENTION_VERSION` 0.5 → 0.6); **GT2 (purity/entropy) landed 2026-07-15**; **GT4 (log-negativity + `is_separable`) landed 2026-07-15** (full-sum `E_N`, exact Fock cross-check, `1×N`-scoped separability; GT1 hardening gave it a trustworthy PT spectrum); GT5/GT3/GT6 remain open.

**Blockers:** the §27 seal gates GT1's covenant (conventions-before-code — Design Principle 1). Generic `E_F` is blocked on the 2019 PRL Supplemental (deferred, not in this WP). The ion `S`-adapter (GT3) couples to `iontrap-structure`'s `ModeConfig` — the first cross-repo dynamics consumer.

**Coastline gates every WI must clear before it counts as landed:**

- [x] **Conventions-before-code** — §27 sealed (via `WP/GT-conventions-proposal.md`) with a green `tests/conventions/test_gaussian_conventions.py` **before** GT2+ consume `V`. *(Satisfied 2026-07-14: §27 sealed at `CONVENTION_VERSION` 0.6; GT2 landed 2026-07-15 over the sealed spectrum.)*
- [ ] **Application-agnostic** — the decoupling grep finds no consuming-application framing (no "Wittemer"/"Hawking"/"cosmology"/"T_H") in any `gaussian.py` symbol or docstring.
- [ ] **Oracle-first** — every functional reproduces a closed-form oracle (vacuum `V=𝟙`; thermal `ν=2n̄+1`; TMSV `E_N=2r/ln2`), not a plot.
- [ ] **Gaussianity precondition guarded** and the `1×N` vs `M×N` separability scoping surfaced.
- [ ] **SPDX** on new modules; **CHANGELOG** dispatch-keyed `[Unreleased]` bullet per landed dispatch.
- [ ] **CI green** — ruff, mypy strict, pytest (unit + regression + conventions).

## 6. Dispatch register *(Sail — minted 2026-07-14 at ratification)*

Family **`GT`** (Gaussian Toolbox) — collision-checked clear against `WP/LOGBOOK.md`, `WORKPLAN_v0.3.md`, `CHANGELOG.md`, and `src/` (taken families: single/double letters, `ED*`, `MC*`, `RL*`, `ND*`, `SQ*`, `TA*`, `AAG/AAH`; `GT*` had zero hits). **Minted at ratification (2026-07-14)** and mirrored into the `WP/LOGBOOK.md` registry.

| Dispatch | Maps to | CHANGELOG bullet | Status |
|---|---|---|---|
| `GT1` | WI-GT1 | `- **Dispatch GT1 — §27 convention primitives: symplectic form, multimode covariance, physicality guard, symplectic spectrum, partial transpose.**` | **landed + sealed 2026-07-14** (conventions test green; §27 sealed, `CONVENTION_VERSION` 0.6) |
| `GT2` | WI-GT2 | `- **Dispatch GT2 — purity + von-Neumann entropy from the symplectic spectrum.**` | **landed 2026-07-15** |
| `GT3` | WI-GT3 | `- **Dispatch GT3 — symplectic congruence + ion S-adapter.**` | minted, open |
| `GT4` | WI-GT4 | `- **Dispatch GT4 — arbitrary-cut Gaussian log-negativity + cut semantics.**` | **landed 2026-07-15** |
| `GT5` | WI-GT5 | `- **Dispatch GT5 — effective temperature (neutral T_eff).**` | minted, open |
| `GT6` | WI-GT6 | `- **Dispatch GT6 — locally-symmetric Gaussian E_F (optional).**` | minted, open |

## 7. Release plan *(Coastline gate)*

Target: an **additive minor** — a new `gaussian.py` multimode surface + the §27 seal (`CONVENTION_VERSION` 0.5 → 0.6). No default changes, no removals; the single-mode `gaussian.py` API is a strict `N=1` special case. Release theme: *multimode Gaussian toolbox*. Coordinated with the §27 seal (like the §26/v0.5 pairing).

## 8. Logbook hooks *(Sail)*

Entries this WP has generated / will generate in `WP/LOGBOOK.md`:

- 2026-07-14 — WP-07 drafted against TC-gaussian-entanglement-toolbox (v0.2); `GT` family proposed (collision-checked, not minted); §27 staged in `WP/GT-conventions-proposal.md`.
- 2026-07-14 — WP-07 **ratified**; `GT1–GT6` minted in the `WP/LOGBOOK.md` registry; the WI boundary moved the symplectic spectrum + partial transpose into GT1 (the low-level convention primitives); **GT1 landed** (`gaussian.py` §27 primitives, conventions test green); WORKPLAN §5.10 applied (v0.3.12, seal-pending wording).
- (maintainer lock-turn) — seal CONVENTIONS §27 + bump `CONVENTION_VERSION` 0.5 → 0.6 + pin `tests/conventions/test_convention_version.py`.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This Work-Plan is a Sail execution document within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg), under the Coastline gates of `WORKPLAN_v0.3.md` and `CONVENTIONS.md`. Lock–Key rule applies: this WP is a key built on the stable locks those documents specify; §27 opens one new lock (staged, maintainer-sealed).

**Council status:** Guardian <pending: confirm §27 reuses (not redefines) §26.2, cross-refs (not edits) frozen §23, and that the decoupling grep is clean>. Architect <pending: confirm the `N=1` `gaussian.py` core is generalised (not forked), and the ion `S`-adapter is kept out of the generic congruence>. Scout <horizon: the 2019 PRL Supplemental unblocks generic `E_F`; GT3 is the first `iontrap-structure` cross-repo dynamics consumer>. Integrator <sequenced conventions-before-code per §5; §27 seal + `CONVENTION_VERSION` 0.5 → 0.6 paired with the WP-07 minor release>.

**Convention version:** `CONVENTIONS.md` v0.6 (§27 sealed 2026-07-14 through GT1; proposal-of-record `WP/GT-conventions-proposal.md`).
**Corporate design version:** `cd-v1.7.1` (consumed via Model B).
**Workplan reference:** `WORKPLAN_v0.3.md` v0.3.11; this WP's track lands as amendment §5.10 (`new in v0.3.12`), pending maintainer seal.
