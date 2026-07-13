# Task Card — Gaussian-state / covariance-matrix toolbox (symplectic spectrum, log-negativity, effective temperature)

**Authored from:** the `iontrap-dynamics` side, as a scoping / build deliberation.
**Topic:** an **application-independent Gaussian-state toolbox** built on the phase-space covariance matrix `V`:
(A) `V` assembly from a solved motional state; (B) the **symplectic spectrum** (Williamson eigenvalues `ν_i`,
purity, von-Neumann entropy); (C) the **normal-mode → local-ion symplectic congruence** `V ↦ S V Sᵀ`;
(D) **Gaussian logarithmic negativity** for arbitrary mode/ion cuts; (E, deferred) **Gaussian
entanglement-of-formation**; (F) a **reduced-state → effective-temperature** (`n̄ → T_eff`) prescription.
**Why a separate card:** these are reusable for *any* Gaussian motional state, with **two genuine consumers** —
(1) the **existing two-mode squeezing** machinery (`states.two_mode_squeezed_vacuum` +
`hamiltonians.two_mode_squeezing_hamiltonian`, §23), and (2) the **future two-ion non-adiabatic squeezing +
Hawking analog** (WP-SQ Phase B, `task cards/TC-nonadiabatic-squeezing-dynamics.md` §3.6/§5). Splitting the toolbox
out keeps WP-SQ **Phase-A-only** and mirrors the WP-04/ND precedent (reusable primitives get their own
deliberation space before a physics-target WP consumes them).
**Scope boundary (per the maintainer review):** extend WP-SQ's A2 single-mode covariance convention to **multimode**
Gaussian states; fix quadrature ordering, the symplectic form, covariance normalisation, reductions, partial
transpose, and symplectic spectra; provide the **normal-mode → local-coordinate** transform; ship **Gaussian
log-negativity**; **defer generic Gaussian `E_F`** until the (APS-gated) 2019 PRL Supplemental pins the applicable
formula/domain; keep **`Hawking`/`T_H` out of public symbols** (a neutral `T_eff` helper takes an *explicit* local
Hamiltonian; the two-ion application owns `T_H`).
**Status:** **v0.2 deliberation record. Not endorsed; no WP ratified.** The first slice (A/B/C/D/F) is fully
specified from open literature and **not source-blocked**; only **(E) generic Gaussian `E_F` is source-blocked**
(pending the 2019 PRL Supplemental, ref [10]). Sibling to `task cards/TC-nonadiabatic-squeezing-dynamics.md`
(WP-SQ) and `task cards/TC-non-markovianity-spectral-density.md` (WP-04). *Formulas below were adversarially
verified against the vacuum-variance-1 convention (workflow `gaussian-toolbox-card-research`, 2026-07-09); ⚠ marks
a correction from that pass or the maintainer's v0.2 review.*

**Revision log. v0.2 (2026-07-09)** — folds in the maintainer's v0.2 review (7 corrections): (1) **arbitrary-cut
log-negativity** `E_N = Σ_k max(0, −log₂ ν̃_k)` (min-only form is two-mode/`1×N` only) + **PPT certifies
separability only for `1×N`** (`M×N ≥ 2` has PPT-bound-entangled Gaussian states, Werner–Wolf); (2) **occupation
includes first moments** `n̄ = (tr V + dᵀd − 2)/4`, `T_eff(0)=0`/reject `n̄<0`, **energy-equivalent**; (3) avoid
`|iΩV|` notation → **moduli of `eig(iΩV)`**, not SVD; (4) **`ModeConfig` alone is insufficient** for `S` (needs a
complete basis + cross-mode orthogonality + masses + local frequencies → separate ion adapter); (5) **not
"entirely greenfield"** — QuTiP's `continuous_variables` provides `covariance_matrix`/`wigner_covariance_matrix`/
two-mode `logarithmic_negativity` (partial reuse/cross-check); (6) **Gaussianity precondition** stated
(purity/entropy/log-neg are true quantities only for Gaussian states); (7) **module ownership** — `N=1` core in
`gaussian.py` immediately, `phase_space.py` = Wigner/readout façades. Also: locally-symmetric `E_F` (broader than
literal `A=B`). **v0.1 (2026-07-09)** — first draft from the adversarially-verified formula sheet.

---

## 1. Verdict

**Mostly greenfield, high reuse-value, low convention risk beyond the quadrature pin — ship A/B/C/D/F, defer E.**
The Gaussian/covariance layer does **not exist in the repo** (whole-tree grep: zero `covarianc`/`symplectic`/
`williamson` hits). QuTiP's `continuous_variables` **does** supply covariance assembly and a two-mode
log-negativity (partial reuse / cross-check — §4), but the **symplectic/Williamson spectrum, the arbitrary-cut
log-negativity, the local-mode congruence, the Gaussian `E_F`, and `n̄ → T_eff` are genuinely new** (QuTiP has no
Williamson). It is **self-contained linear algebra on a 2N×2N matrix**, with the existing squeezed-state factories
as exact
regression oracles. It shares exactly one convention surface with WP-SQ (the vacuum-variance-1 **quadrature
normalisation**), which must be sealed **once** and referenced by both. Recommended first slice: covariance core
(A) + symplectic spectrum (B) + local-mode transform (C) + log-negativity (D) + effective temperature (F). **Defer
(E) generic Gaussian `E_F`** — the general (non-symmetric) two-mode `E_F` is an optimisation, not a closed form,
and the paper's exact method is Supplemental-gated; the symmetric-case closed form is optional (guarded `A=B`).

## 2. Ownership boundary & consumers

| Capability | Owner | Rationale |
|---|---|---|
| Quadrature ops `x̂=â+â†`, `p̂=i(â†−â)`; covariance `V` (single- and multi-mode) | **library (new `gaussian.py`)** | phase-space state functionals; generalises WP-SQ A2's single-mode `V` |
| Symplectic spectrum — Williamson `ν_i`, purity, von-Neumann entropy | **library** | standard Gaussian invariants |
| Generic symplectic congruence `V ↦ S V Sᵀ` (given a symplectic `S`) | **library (`gaussian.py`)** | the transform the ion-cut needs; convention-independent linear algebra |
| Constructing the ion `S` (normal→local) from modes + masses + local frequencies | **ion-specific adapter (separate; near the two-ion consumer / `iontrap-structure`)** | needs a complete mode basis + cross-mode orthogonality + species masses + local reference frequencies — more than a single `ModeConfig` |
| Gaussian log-negativity (arbitrary mode/ion cut, closed form) | **library** | covariance-based; the mode-A/B cut the Fock `log_negativity_trajectory` cannot do |
| Effective temperature `T_eff(n̄, ω_loc)` | **library (neutral symbol)** | takes an *explicit* local `ω_loc`; thermal-equivalence, not thermodynamic |
| Generic Gaussian `E_F` (non-symmetric two-mode) | **deferred** | optimisation-based; method Supplemental-gated |
| `T_H` / "Hawking temperature" framing | **downstream two-ion card (WP-SQ Phase B)** | consuming-application framing — never a toolbox symbol |
| Normal-mode spectrum (`M_ij`, COM/rocking eigenvectors) | **sibling `iontrap-structure`** | supplies `ModeConfig`; the toolbox consumes it for `S` |

**Two consumers, one invariant.** The toolbox carries **no** consuming-application framing (no "Wittemer", no
"Hawking", no "cosmology") — enforced by the decoupling grep. Consumer (1) is the existing two-mode squeezing
(the **validation oracle**); consumer (2) is WP-SQ Phase B (which supplies `ModeConfig` + owns `T_H`).

## 3. What the toolbox computes — verified formula sheet (vacuum-variance-1)

Quadrature convention (shared with WP-SQ §26): `x̂ = â + â†`, `p̂ = i(â† − â)`, **vacuum variance 1**, so
`[x̂, p̂] = 2i`.

⚠ **Gaussianity precondition.** `V` and the first moments `d` are well-defined for **any** state, but the
**interpretation** of the symplectic spectrum — purity, entropy (B), and the covariance-based log-negativity (D) —
as the true *quantum-state* quantities holds **only for Gaussian states**. For a non-Gaussian solved state these are
the *second-moment (Gaussian-equivalent)* values, and `V` **alone cannot certify Gaussianity** (higher moments are
needed). The WP-SQ consumer is safe — a quadratic `ω(t)` Hamiltonian preserves Gaussianity — but the toolbox must
**state the precondition and guard it** (e.g. an optional symplectic-purity vs Fock-purity cross-check that raises
on a large mismatch).

**A. Covariance matrix, symplectic form, physicality.** With `R = (x̂₁, p̂₁, …, x̂_N, p̂_N)`,
`V_ij = ½⟨{ΔR_i, ΔR_j}⟩`. Single-mode **vacuum `V = 𝟙₂`**. CCR `[R_i, R_j] = 2i Ω_ij` with `Ω = ⊕_k J`,
`J = [[0, 1], [−1, 0]]`. Bona-fide (Robertson–Schrödinger) physicality: **`V + iΩ ≥ 0`** (vacuum saturates:
`eig(V + iΩ) = {0, 2}`); vacuum symplectic eigenvalue `ν = 1`.

**B. Symplectic spectrum (Williamson).** `V = S D Sᵀ`, `S` symplectic (`SΩSᵀ = Ω`),
`D = diag(ν₁, ν₁, …, ν_N, ν_N)`. ⚠ The **symplectic eigenvalues `ν_i` are the moduli of the eigenvalues of `iΩV`**
— which are real and occur in `±ν_i` pairs — counted **once per pair**. *Compute as `|eig(iΩV)|` deduplicated to
pairs; **not** the singular values / matrix absolute value of `iΩV` (`√(M†M)`, SVD), which in general differ from
the symplectic eigenvalues.* Physicality ⇔ `ν_i ≥ 1`. **Purity** `μ = Tr(ρ²) = ∏_i (1/ν_i) = 1/√(det V)`.
**Von-Neumann entropy** (bits, matching `information/_common._von_neumann_entropy_bits`) `S = Σ_i g(ν_i)`,
`g(ν) = (ν+1)/2·log₂((ν+1)/2) − (ν−1)/2·log₂((ν−1)/2)` for `ν ≥ 1` (`g(1) = 0`; thermal `ν = 2n̄+1`).

**C. Normal-mode → local-ion symplectic congruence.** A linear canonical change `R_local = S R_normal` acts on
covariances by **`V_local = S V_normal Sᵀ`**, with `S` symplectic (`SΩSᵀ = Ω`, `det S = +1`). Entanglement is
**partition-dependent**: the COM/stretch normal-mode cut is *not* the ion-A/ion-B cut, so ion–ion entanglement
must be read from `V` in **local** coordinates. ⚠ **Two corrections from the verify pass:** (i) *nonclassicality is
required for beamsplitter entanglement* — a passive 50:50 map cannot entangle **classical** (thermal/coherent,
positive-P) normal modes; they stay **separable** across the ion cut. The minimal entangling example is a product
of **squeezed** normal modes (Kim–Son–Bužek–Knight 2002), not thermal ones. (ii) The transform is the *orthogonal*
50:50 rotation **only** when local and normal quadratures share one common normalisation frequency; with per-mode
dimensionless quadratures (COM vs stretch frequencies differ — `√3` axially for two ions) `S` stays **symplectic
but not orthogonal**.

**D. Gaussian logarithmic negativity.** ⚠ General multimode form: **`E_N = Σ_k max(0, −log₂ ν̃_k)`**
(`= −Σ_{ν̃_k < 1} log₂ ν̃_k`, base-2 → ebits), summed over **all** symplectic eigenvalues `ν̃_k` of the **partial
transpose** `Ṽ = T_B V T_B` (`T_B : p̂_B → −p̂_B`). The **smallest-only form `E_N = max(0, −log₂ ν̃₋)` holds only for
two-mode and `1×N` cuts** (a single PT eigenvalue drops below 1). Two-mode closed form with `V = [[A, C], [Cᵀ, B]]`:
`Δ̃ = det A + det B − 2 det C`, `ν̃₋² = (Δ̃ − √(Δ̃² − 4 det V)) / 2`. **Oracle:** two-mode squeezed vacuum →
`ν̃₋ = e^{−2r}`, `E_N = 2r/ln 2`.
⚠ **Separability caveat (verify pass + QuTiP audit).** The PPT / `ν̃_k ≥ 1` criterion certifies **separability only
for `1×N` Gaussian cuts** (Simon; Werner–Wolf). For general `M×N` cuts with `M, N ≥ 2`, **PPT-bound-entangled
Gaussian states exist**, so `E_N = 0` does **not** certify separability — log-negativity is only an **NPT witness**
there. **Scope the separability-certification API to `1×N`;** for `M×N` expose `E_N` as an NPT witness with the
caveat surfaced, never as a separability test.

**E. Gaussian entanglement-of-formation (DEFER generic).** ⚠ **Locally-symmetric** two-mode (equal local
reductions; `A = B` is the strict special case, the exact-formula domain is somewhat broader) closed form:
`E_F = c₊ log₂ c₊ − c₋ log₂ c₋` **for `ν̃₋ < 1`** (and **`E_F = 0` for `ν̃₋ ≥ 1`**), `c_± = (ν̃₋ ± 1)² / (4 ν̃₋)`.
The gate is the **domain condition `ν̃₋ < 1`, not `max(0, ·)`** — because `c_±` are invariant under `ν̃₋ ↔ 1/ν̃₋`,
the raw expression is *positive* for separable states too. Oracle: symmetric TMSV → `c₊ = cosh²r`, `c₋ = sinh²r`.
The **general (non-symmetric) two-mode `E_F` is an optimisation** (Tserkis–Ralph 2017; Akbari), and the paper's
exact formula/domain needs the 2019 PRL Supplemental → **defer**. Ship log-negativity first; the locally-symmetric
`E_F` is optional with a guarded local-symmetry precondition.

**F. Reduced-state → effective temperature.** ⚠ **Physical occupation (general, includes first moments):**
`n̄ = Tr(ρ_red â†â) = (tr V_red + dᵀd − 2)/4`, with `d = (⟨x̂⟩, ⟨p̂⟩)` the first moments (from
`â†â = (x̂² + p̂² − 2)/4`; the `(V₁₁ + V₂₂ − 2)/4` form is the **centered** occupation, valid only for `d = 0` — if
displacement is deliberately excluded, name it *centered occupation*). Then `T_eff = ℏω_loc / (k_B · ln(1 + 1/n̄))`,
given an **explicit** local-Hamiltonian frequency `ω_loc`; define `T_eff(n̄ = 0) = 0` by continuity and **reject only
`n̄ < 0`**. This is an **energy-equivalent** temperature (the thermal state with the same mean occupation/energy at
`ω_loc`) — for a squeezed/non-thermal marginal it is **not** a state- or entropy-equivalence, and not a
thermodynamic claim; public symbol **`T_eff`** (neutral). ⚠ **Do not use `n̄ = (ν−1)/2`** (`ν = √det V_red`): that
equals `Tr(ρ_red â†â)` *only* for an unsqueezed, undisplaced (isotropic, centered) marginal — a reduced marginal of
a two-mode Gaussian state can be **squeezed and/or displaced**, where `(ν−1)/2` misses both the squeezing and the
first-moment energy and would spuriously give `T_eff = 0`. `(ν−1)/2` is instead the occupation of the
*symplectically-equivalent thermal core* (after stripping local squeezing) — a different quantity.

## 4. Capability map — reuse vs gap

⚠ **Partial reuse — QuTiP's CV layer (correction: not "entirely greenfield").** QuTiP **5.2.3** already ships
`qutip.continuous_variables.covariance_matrix(basis, rho, symmetrized=True)` (`V_ij = ½⟨{a_i,a_j}⟩ − ⟨a_i⟩⟨a_j⟩`
over a **user-supplied operator basis** — **directly reusable for (A)** if the basis is our vacuum-variance-1
`x̂,p̂`), plus `wigner_covariance_matrix` and a **two-mode-only** `logarithmic_negativity(V, g=√2)`. But QuTiP's
log-negativity is **not a drop-in**: it is hard-wired two-mode, uses **natural log** (nats, not `log₂`/ebits), the
`g=√2` default is the **½-convention** (`ħ=1`, vacuum variance ½), and it hard-codes the **smallest-`ν̃₋`** form —
i.e. it inherits the very `1×N`/two-mode restriction of §3.D. So it is a **cross-check** for the two-mode case
(after matching `g` and the log base), not the arbitrary-cut API.

**Genuinely absent (build new):** the **symplectic/Williamson** layer (symplectic eigenvalues, purity/entropy from
the spectrum — B), the **symplectic congruence `S V Sᵀ`** + the ion-specific `S`-construction (C), **arbitrary-cut**
Gaussian log-negativity with the general `Σ_k` form (D), the **locally-symmetric `E_F`** (E), and **`n̄ → T_eff`**
(F). Whole-tree grep confirms **zero** `covarianc`/`symplectic`/`williamson` hits in `src/`, **no quadrature
operators** (only `HilbertSpace.annihilation_for_mode`/`creation_for_mode`/`number_for_mode`,
[hilbert.py:228–236](src/iontrap_dynamics/hilbert.py#L228)), and **QuTiP has no Williamson decomposition**.

**Reuse — validation oracles & convention anchors:**

| Asset | Existing API | Role |
|---|---|---|
| QuTiP CV covariance assembly | `qutip.continuous_variables.covariance_matrix` (basis-agnostic, symmetrised) | **direct reuse for (A)** with vacuum-variance-1 basis ops |
| QuTiP two-mode log-negativity | `qutip.continuous_variables.logarithmic_negativity(V, g)` | **two-mode cross-check** only (nats, ½-conv, min-`ν̃₋`) — not the arbitrary-cut API |
| Two-mode squeezed vacuum (`⟨n̂⟩ = sinh²\|z\|`, no ½; §23.1) | `states.two_mode_squeezed_vacuum` ([states.py:285](src/iontrap_dynamics/states.py#L285)) + `hamiltonians.two_mode_squeezing_hamiltonian` ([hamiltonians.py:1601](src/iontrap_dynamics/hamiltonians.py#L1601)) | **the** regression oracle for `V`, `ν_i`, `E_N`, `E_F` |
| Single-mode coherent/squeezed factories (§6/§7) | `coherent_mode`, `squeezed_vacuum_mode`, `squeezed_coherent_mode` ([states.py:198–245](src/iontrap_dynamics/states.py#L198)) | single-mode `V` oracles |
| Fock-space log-negativity (all-spins-vs-all-modes cut only) | `log_negativity_trajectory` ([entanglement.py:134](src/iontrap_dynamics/entanglement.py#L134)) | **cross-check** in the truncation-converged limit; **not** reusable — its `partition` can't do mode-A/B |
| Spin-only Wootters `E_F` | `entanglement_of_formation_trajectory` ([entanglement.py:109](src/iontrap_dynamics/entanglement.py#L109)) | precedent only — **not** Gaussian |
| von-Neumann entropy in **bits** | `information/_common._von_neumann_entropy_bits` ([_common.py:31](src/iontrap_dynamics/information/_common.py#L31)) | fixes the `log₂`/bits convention; cross-check on truncated states |
| Thermal factory (`n̄` as input) | `qutip.thermal_dm` via `compose_density` ([states.py:116](src/iontrap_dynamics/states.py#L116)) | oracle for `n̄ → T_eff`; confirms `n̄→T` is the missing inverse |
| ⚠ Normal-mode eigenvectors — **necessary but not sufficient** for `S` | `ModeConfig.eigenvector_per_ion` ([modes.py:42](src/iontrap_dynamics/modes.py#L42)) | one mode's vector only; building `S` also needs a **complete mode basis** for the subspace, **cross-mode orthogonality (mass-metric) validation** (the repo checks only per-mode norm, §11), **species masses** + the eigenvector weighting convention, and **explicit local reference frequencies** — an **ion-specific adapter**, kept out of the generic `gaussian.py` congruence |

## 5. Conventions & governance

- **One shared convention surface with WP-SQ — seal it once.** Both cards pin the vacuum-variance-1 **quadrature
  normalisation** (`x̂ = â+â†`, `p̂ = i(â†−â)`). WP-SQ's **§26** seals the single-mode normalisation; the toolbox
  **reuses §26** and **adds** the multimode pieces: (i) quadrature **ordering** `R = (x̂₁,p̂₁,…,x̂_N,p̂_N)` (per-mode
  `J`-blocks, Adesso/Serafini — vs the Simon `(x₁…x_N,p₁…p_N)` layout); (ii) the **symplectic form** `Ω = ⊕J` and
  its `2i` scaling; (iii) the **partial-transpose sign map** `p̂_B → −p̂_B`. These are new → the **next free section
  after WP-SQ's §26, i.e. §27** (confirm both at ratification via the 5-source collision grep). **Sequencing
  dependency:** if WP-SQ ratifies first (recommended), §26 seals the quadrature normalisation and the toolbox's §27
  references it; if the toolbox lands first, it seals both. Either way the normalisation must not be defined twice.
- **`CONVENTION_VERSION` bump.** The new Gaussian section is additive and **pure-motional** (like the frozen §23
  two-mode squeezing / §24 CPTP), fitting the established "pure-motional object" precedent — but any sealed section
  needs a version bump. Route via the RLA/§25 path: review note → side-car (propose-don't-apply) → green
  conventions-test → **maintainer seals**. **§23 is frozen** (the TMSV convention the toolbox validates against) —
  cross-ref, don't edit. Cross-ref also **§2** (tensor ordering, spins-then-modes), **§11** (eigenvector
  normalisation — the `S` transform depends on it), **§6/§7** (single-mode squeeze/displacement).
- **Module placement — resolve at WP-SQ ratification, no fork.** ⚠ The `N = 1` covariance/symplectic core goes into
  **`gaussian.py` immediately** — created by WP-SQ A2 itself (WP-SQ does **not** wait on this toolbox card; it just
  puts the single-mode core in the right file). **`phase_space.py` owns only the Wigner/readout façades** over
  `gaussian.py`; **no `phase_space.py`-native symplectic arithmetic** and no covariance re-implementation. This
  toolbox then **generalises `gaussian.py` to multimode** — a pure extension, never a refactor of a parallel path.
  *(Actionable rule for the WP-SQ A2 implementer: single-mode readout = the `N = 1` limit of `gaussian.py` APIs.)*
- **Governance.** Lean deliberation card now; a future WP would mint its own dispatch family (candidate **`GT`**,
  Gaussian toolbox — confirm collision-free at ratification). Current `CONVENTION_VERSION = "0.5"`.

## 6. Proposed first slice — WI sketch

1. **Quadratures + covariance** (`gaussian.py`): `x̂`, `p̂` (named vacuum-variance-1), `covariance_matrix(state,
   mode_labels)` → 2N×2N `V` + first moments `d` (incl. cross `C_xp` and inter-mode blocks); may wrap
   `qutip.continuous_variables.covariance_matrix` with vacuum-variance-1 basis ops. Acceptance: vacuum `V = 𝟙`;
   TMSV `V` matches the analytic block form; `V + iΩ ≥ 0` holds (physicality guard raises otherwise);
   **Gaussianity guard** (symplectic-purity vs Fock-purity) available.
2. **Symplectic spectrum** (`gaussian.py`): `symplectic_eigenvalues(V)` = **moduli of `eig(iΩV)`, one per `±ν`
   pair** ⚠ (not SVD/matrix-abs); `purity`, `gaussian_entropy_bits`. Acceptance: vacuum `ν = 1`; thermal
   `ν = 2n̄+1`; entropy agrees with `_von_neumann_entropy_bits` on truncated states; `g(1) = 0`.
3. **Symplectic congruence + ion `S`-adapter** (`gaussian.py` + separate adapter ⚠): generic `congruence(S, V)`
   with an `SΩSᵀ = Ω` guard in `gaussian.py`; the **ion-specific `S`** (normal→local from a complete mode basis +
   cross-mode orthogonality + species masses + explicit local frequencies) in a separate adapter. Acceptance:
   **squeezed** (not thermal ⚠) normal-mode product → correct ion-cut `E_N`; round-trip `S⁻¹`; the non-orthogonal
   (unequal-frequency, COM vs stretch) case handled; `SΩSᵀ = Ω` verified.
4. **Gaussian log-negativity** (`gaussian.py`): `log_negativity(V, cut)` = **`Σ_k max(0, −log₂ ν̃_k)`** ⚠ over all
   PT symplectic eigenvalues, arbitrary mode/ion bipartition. Acceptance: TMSV → `E_N = 2r/ln 2`; separable/thermal
   → `0`; agrees with the Fock `log_negativity_trajectory` in the truncation-converged limit; **separability
   certification exposed for `1×N` only**, `M×N (≥2)` returns `E_N` as an NPT witness with the PPT-bound-entangled
   caveat surfaced ⚠.
5. **Effective temperature** (`gaussian.py`): `effective_temperature(nbar, omega_loc)` with `n̄` from
   **`(tr V_red + dᵀd − 2)/4`** ⚠ (first-moment-aware; not `(ν−1)/2`); explicit `ω_loc`; **`T_eff(0) = 0`, reject
   `n̄ < 0`** ⚠; neutral `T_eff`. Acceptance: thermal-state round-trip (`n̄ → T_eff → n̄`); a **squeezed and/or
   displaced** marginal does **not** give `T_eff = 0`; energy-equivalent framing documented.
6. **(optional) Locally-symmetric `E_F`** (`gaussian.py`): the closed form with the **`ν̃₋ < 1` gate** ⚠ and a
   guarded **local-symmetry** precondition (broader than literal `A = B`). Acceptance: symmetric TMSV → entropy of
   entanglement; separable → `0`. *Generic (non-symmetric) `E_F` deferred pending the 2019 PRL Supplemental.*

## 7. Open questions & provisional decisions

1. **Quadrature ordering** — **PROVISIONAL: `(x̂₁,p̂₁,…,x̂_N,p̂_N)` per-mode `J`-blocks** (Adesso/Serafini), sealed
   in §27 alongside `Ω` and the PT sign. Pin once; both cards reference it.
2. **Module** — **PROVISIONAL: the `N=1` covariance/symplectic core goes into `gaussian.py` immediately (created by
   WP-SQ A2); `phase_space.py` owns only Wigner/readout façades — no native symplectic arithmetic.** The toolbox
   generalises `gaussian.py` to multimode. WP-SQ does not wait on this card.
3. **Log-negativity API** — **PROVISIONAL: covariance-based `log_negativity(V, cut) = Σ_k max(0, −log₂ ν̃_k)` for
   arbitrary cuts**; **separability certification for `1×N` only** (`M×N ≥ 2` → NPT witness + PPT-bound-entangled
   caveat). QuTiP's two-mode `logarithmic_negativity` is a cross-check (nats/½-conv), not the API; the Fock
   `log_negativity_trajectory` a truncation cross-check.
4. **Gaussian `E_F`** — **PROVISIONAL: defer generic; ship the locally-symmetric closed form optionally with the
   `ν̃₋ < 1` gate** (guarded local-symmetry, broader than literal `A=B`). Revisit when the 2019 PRL Supplemental
   pins the paper's method (symmetric vs general).
5. **Effective temperature** — **PROVISIONAL: `n̄ = (tr V_red + dᵀd − 2)/4` (first-moment-aware),
   `T_eff = ℏω_loc/(k_B ln(1+1/n̄))`, explicit `ω_loc`, `T_eff(0)=0`, reject `n̄<0`, neutral `T_eff`; energy-
   equivalent framing; `(ν−1)/2` documented as the thermal-core occupation only.**
6. **Occupation-of-reduced / Gaussianity guard** — **PROVISIONAL: covariance is assembled for any state, but
   purity/entropy/covariance-log-neg are the true quantities only for Gaussian states; ship an optional
   Gaussianity guard (symplectic-purity vs Fock-purity) and document the precondition.**
7. **Symplectic-eigenvalue numerics** — **PROVISIONAL: `|eig(iΩV)|` deduplicated to `±ν` pairs, NOT SVD/matrix-abs.**
8. **Section numbering / sequencing** — **PROVISIONAL: WP-SQ §26 seals the quadrature normalisation first; toolbox
   §27 extends it (ordering + `Ω` + PT).** Confirm both at ratification (collision grep).
9. **Consumer coupling & `S`-adapter** — the two-ion card (WP-SQ Phase B) consumes this toolbox **plus** an
   **ion-specific `S`-adapter** (built from `iontrap-structure`'s modes + masses + local frequencies, not a single
   `ModeConfig`); the toolbox's generic congruence and the whole toolbox stay application-agnostic (no `T_H`).

## 8. Rooting sources

| Role | Source | Local |
|---|---|---|
| Covariance / symplectic formalism, log-negativity, `E_F` domain | Weedbrook et al., **RMP 84, 621 (2012)**; Adesso, Ragy, Lee, **Open Syst. Inf. Dyn. 21, 1440001 (2014)**; Serafini, *Quantum Continuous Variables* (2017) | ✗ open literature (fetchable) |
| Symplectic eigenvalues / Williamson | J. Williamson, **Am. J. Math. 58, 141 (1936)**; Adesso, Serafini, Illuminati, **PRA 70, 022318 (2004)** | ✗ open |
| Log-negativity / PPT (`1×N` sufficiency; `M×N` bound entanglement) | Vidal & Werner, **PRA 65, 032314 (2002)**; Simon, **PRL 84, 2726 (2000)**; Werner & Wolf, **PRL 86, 3658 (2001)** | ✗ open |
| QuTiP CV layer (partial reuse / cross-check) | `qutip.continuous_variables` (`covariance_matrix`, `wigner_covariance_matrix`, two-mode `logarithmic_negativity`), QuTiP **5.2.3** | ✓ installed dependency |
| Gaussian `E_F` (symmetric closed form / general) | Giedke, Wolf, Krüger, Werner, Cirac, **PRL 91, 107901 (2003)**; Adesso & Illuminati, **J. Phys. A 40, 7821 (2007)**; Tserkis & Ralph, **PRA 96, 062338 (2017)** | ✗ open |
| Beamsplitter needs nonclassical input (the ⚠ C correction) | Kim, Son, Bužek, Knight, **PRA 65, 032323 (2002)** | ✗ open |
| Ion-trap normal modes (`M_ij` → local coords) | D. F. V. James, **Appl. Phys. B 66, 181 (1998)** | ✓ (sibling `iontrap-structure` validation ref) |
| The paper's exact two-ion `E_F` method/domain | Wittemer 2019 **PRL 123 180502 Supplemental** (ref [10]) | ✗ **APS-gated — blocks generic `E_F` (E) only** |

## 9. References

- G. Adesso, S. Ragy, A. R. Lee, *Continuous Variable Quantum Information: Gaussian States and Beyond*, **Open Syst. Inf. Dyn. 21, 1440001 (2014)**.
- C. Weedbrook et al., *Gaussian quantum information*, **Rev. Mod. Phys. 84, 621 (2012)**.
- A. Serafini, *Quantum Continuous Variables: A Primer of Theoretical Methods* (CRC Press, 2017).
- G. Vidal, R. F. Werner, *Computable measure of entanglement*, **PRA 65, 032314 (2002)**.
- R. F. Werner, M. M. Wolf, *Bound Entangled Gaussian States*, **PRL 86, 3658 (2001)** (PPT ⇏ separable for `M×N ≥ 2`).
- G. Giedke, M. M. Wolf, O. Krüger, R. F. Werner, J. I. Cirac, *Entanglement of Formation for Symmetric Gaussian States*, **PRL 91, 107901 (2003)**.
- G. Adesso, F. Illuminati, *Entanglement in continuous-variable systems*, **J. Phys. A 40, 7821 (2007)**.
- M. S. Kim, W. Son, V. Bužek, P. L. Knight, *Entanglement by a beam splitter: Nonclassicality as a prerequisite for entanglement*, **PRA 65, 032323 (2002)**.
- D. F. V. James, *Quantum dynamics of cold trapped ions…*, **Appl. Phys. B 66, 181 (1998)**.
- M. Wittemer et al., **PRL 123, 180502 (2019)** — the downstream two-ion consumer (`task cards/TC-nonadiabatic-squeezing-dynamics.md` Phase B).
