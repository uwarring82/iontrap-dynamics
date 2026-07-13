# WP-05 — Non-adiabatic squeezing dynamics (Phase A)

**Executes Phase A of the non-adiabatic-squeezing task card: a centred time-dependent-frequency squeezing
Hamiltonian (`ω(t)` quench + parametric modulation), a phase-space / Gaussian readout (quadratures, covariance,
Wigner, `r`, `ν`, `α`, direct `Pₙ`), a compute-only Wittemer-2020 single-ion reproduction benchmark, and two
tutorials — sealing a new §26 convention for the squeezing generator + single-mode quadrature normalisation. The
forced-displacement + echo extension (A3) is an optional Phase-A tail; the two-ion entanglement / Hawking analog
(Phase B) is out of scope.**

Version 0.1 · Drafted 2026-07-10 · **Sealed 2026-07-10** · Status: **Ratified** (task card v0.2.1 approved; SQ family minted, §26 sealed, CONVENTION_VERSION 0.4 → 0.5)

**Origin / rooting:** the deliberation & survey record `task cards/TC-nonadiabatic-squeezing-dynamics.md`
(v0.2.1; **not** duplicated here) — it carries the verdict, the codebase/conventions analysis, the A1/A2/A3
sub-slice split, the rooting sources (Wittemer 2019 PRL 123 180502 + 2020 Phil Trans R Soc A 378 20190230 +
Silveri 2015 + Burd 2019 + Fey 2018, staged under `sources/pdf/`), and the §9 provisional decisions. This WP is
the **execution layer** for Phase A only. The reusable **Gaussian toolbox** the readout builds on has its own card
(`task cards/TC-gaussian-entanglement-toolbox.md`, §27, future WP); WP-05 creates only the `N=1` core it needs.

**Classification:** Sail execution under Coastline gates (per T(h)reehouse +EC CD 0.9).
**Licence:** This WP document is CC BY-SA 4.0 (`WP/LICENCE`). Deliverables carry their layer's licence: code is MIT
(`src/`, `tests/`); authored tutorials are Sail / CC BY-NC-SA 4.0. See root `LICENCE`.
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (`cd-rules`, Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

---

## 0. Ratification decisions (proposed 2026-07-10) *(Coastline — awaiting maintainer seal)*

| # | Decision | Resolution | Binds |
|---|---|---|---|
| **R1** | Slug, sub-slices & cadence | Slug `nonadiabatic-squeezing-dynamics`. Phase A splits into **A1** (centred quadratic `ω(t)` squeezing), **A2** (phase-space / Gaussian readout, observable-only), **A3** (forced displacement + echo, **optional tail**). Cadence: understand→implement→**adversarial-verify** per WI, external review pause per WI. A2/A3 land trunk-based on `main`; **A1's §26 seal follows the RLA/§25 side-car path** (propose-don't-apply → green conventions-test → **maintainer seals**), code staged behind the seal. | §1, §2 |
| **R2** | **Conventions gate — §26 to be sealed (bump `0.4 → 0.5`)** | Unlike WP-04 (no bump), A1 introduces the **first single-mode time-dependent-frequency squeezing generator** `H_sq/ℏ = −(i/4)(d ln ω/dt)(â†² − â²)` (Hermitian time-list basis `−i(â†² − â²)` with **real** coefficient `¼·d ln ω/dt`) in a **fixed `ω(0)` basis** (Silveri 2015) **plus** the vacuum-variance-1 **quadrature normalisation** (`x̂=â+â†`, vacuum variance 1) and the **Wigner `g`-pin**. These are a **new §26** (frozen §6 is the squeeze *parameter/ellipse*, not a generator — no in-place edit). Readout functionals (`ν`, `r`, `n̄_sq`, `α`, direct `Pₙ`) stay **observable-only** (compose §26; MCF/ND precedent). §27 (multimode ordering + `Ω` + PT) is the **toolbox's**, out of scope here. The seal itself (the CONVENTIONS.md §26 text + the `CONVENTION_VERSION 0.4→0.5` bump in `conventions.py`) is the **maintainer's act** after a green conventions-test; WP-05 ships the **side-car proposal** (`WP/SQ-conventions-proposal.md`), not the edit. | A1, A2 |
| **R3** | Dispatch family | **`SQ`** (squeezing dynamics), minted at ratification. Authoritative 5-source **pre-ratification** collision grep for **`SQ<n>` dispatch IDs** (`\bSQ[0-9]+\b`, dispatch-code-scoped — *not* a literal `SQ` prefix, which would catch "standard quantum limit / SQL" prose in `fisher.py`/docs) over `CHANGELOG.md`, `WORKPLAN_v0.3.md`, `WP/LOGBOOK.md`, `docs/`, `src/` returns **zero pre-existing** hits before adding WP-05 / §5.8. Distinct from `ED`/`MC`/`RM`/`ND`. | §2 |
| **R4** | Placement & **no-fork** | A1: `hamiltonians.py` (`nonadiabatic_squeezing_hamiltonian`) + `waveforms.py` (`FrequencyWaveform`, named shapes). A2: **the `N=1` covariance/symplectic core is created in `gaussian.py` here**; `phase_space.py` holds **only** Wigner/readout façades — **no `phase_space.py`-native symplectic arithmetic**. The future toolbox WP generalises `gaussian.py` to multimode (pure extension, never a refactor). A3: `hamiltonians.py` (linear force term). | A1, A2, A3 |
| **R5** | Waveform contract | A1 takes a **`FrequencyWaveform` object** exposing both `ω(t)` **and** the **analytic** `d ln ω/dt` (+ JAX variants), **not** a bare callable — the generator coefficient is the log-derivative and a black-box callable cannot supply it safely. Validates **finite, strictly positive `ω(t)`** and **finite `d ln ω/dt`**; paired-callable consistency is the caller's responsibility (build-time spot-checks) — **never runtime numerical differentiation**. Fallback: paired callables `(omega_of_t, d_ln_omega_dt)`. | A1 |
| **R6** | Readout form | `r` from the **covariance-eigenvalue** form `r = ¼ ln(λ_max/λ_min)`, `ν = √(det V)` — **not** `tr V` (which conflates squeezing with thermal width); `α = (⟨x̂⟩+i⟨p̂⟩)/2`, `\|α\| = ½√(⟨x̂⟩²+⟨p̂⟩²)` (frozen §7-consistent). Quadratures ship named vacuum-variance-1. | A2 |
| **R7** | **A3 is an optional tail** | The centred quadratic A1 Hamiltonian is **parity-preserving** (`⟨â⟩=0` from vacuum) and **cannot** produce the parasitic coherent displacement / echo `δp` — that needs A3's linear force term. A3 ships only if the displacement/echo story is wanted; **A1/A2 alone reproduce the displacement-free 2020 results** (esp. the parametric arm). | A3 |
| **R8** | **Phase B out of scope** | Two-ion motional entanglement + the Hawking/cosmology analog — needing the **Gaussian toolbox** (`TC-gaussian-entanglement-toolbox.md`) + `iontrap-structure`'s `ModeConfig` + a normal-mode→local-ion transform + a `T_H` readout — are **deferred to a future consumer WP**. Generic Gaussian `E_F` is additionally **Supplemental-gated** (2019 PRL ref [10]). | future WP |

---

## 1. Scope (Phase A)

A **single** motional mode with a time-dependent trap frequency `ω(t)`, evolved as a closed-system unitary under a
centred quadratic Hamiltonian (`H(t) = ℏω(t)(â†â+½) − (iℏ/4)(d ln ω/dt)(â†²−â²)`) — fully solvable with the existing
solver (`solve` already accepts `[[H, f(t)]]` / `TimeQArray`), no new solver, no open-system machinery. Two extreme
regimes: **quench** (Gaussian `ω(t)` pulse) and **parametric modulation** (`ω_mod = 2ω_ini`,
`n̄_sq = sinh²(2π g T_mod)`). Reads out via the phase-space covariance matrix. Phase B (two-ion / Hawking) is
explicitly **out of scope** (R8).

## 2. Dispatch decomposition (`SQ`)

| Code | Sub-slice | Work item | Deliverable |
|---|---|---|---|
| **SQ1** | A1 | `ω(t)` squeezing Hamiltonian + §26 seal | `nonadiabatic_squeezing_hamiltonian` (`hamiltonians.py`) + `FrequencyWaveform` + named shapes (`waveforms.py`) + `ω̇/ω²` diagnostic (`analytic.py`); **§26 side-car** (`0.4→0.5`). Seal gates: analytic squeeze-kick / narrowing smooth ramp → `r = ½\|ln(ω_f/ω_i)\|`; **cyclic** adiabatic → `r→0`; cross-backend 1e-3. |
| **SQ2** | A2 | Quadratures + covariance core + Wigner | `gaussian.py` `N=1` covariance core (`x̂,p̂` named vacuum-variance-1, 2×2 `V` incl. `C_xp`) + `phase_space.py` façade + Wigner (**`g`-pinned**). Acceptance: vacuum `V=I`; `squeezed_vacuum_mode(r)` → `λ={e^{-2r},e^{2r}}`. |
| **SQ3** | A2 | Squeezing / displacement readout (observable-only) | `ν=√(det V)`, `r=¼ln(λ_max/λ_min)`, `n̄_sq=sinh²r`, `α`, `n_dsp`. Acceptance: round-trips `squeezed_coherent_mode`; **thermal-squeezed → `r` invariant under `n̄_th`** (the `tr V` form fails this). |
| **SQ4** | A2 | Direct `Pₙ` + truncation gate | `Pₙ = ⟨n\|ρ\|n⟩` + pure-squeezed-vacuum oracle (`P_odd=0`). Acceptance: parity signature; **parity-aware tail-window / cross-cutoff** §13 convergence gate (the top-level guard is parity-blind for even-dim squeezed vacuum). |
| **SQ5** | benchmark | Wittemer-2020 single-ion reproduction | Compute-only (`tools/run_benchmark_*.py` → `benchmarks/data/`): the **parametric** arm `n̄_sq = sinh²(2π g T_mod)` (A1+A2, displacement-free) + quench `n̄_sq` vs amplitude; analytic squeeze-kick / cyclic-adiabatic oracle. |
| **SQ6** | A3 *(optional)* | Forced displacement + echo | `H_force ∝ f(t)(â+â†)` (`hamiltonians.py`) + two-pulse echo protocol. Acceptance: static offset seeds `α≠0`; echo suppresses it (`δp`), qualitative 2020 Fig 2b. |
| **SQ7** | tutorials | Tutorials A & B | A ("Squeezing by quenching `ω(t)`") + B ("Phonon-pair creation & readout") — print+plot + Colab, `docs/tutorials/19_…`, `20_…`, `tools/build_tutorial_notebooks.py`, `notebooks`/`tutorials` CI guards. |

## 3. Acceptance gates

- **SQ1:** sudden (squeeze-kick / narrowing ramp) → `r = ½|ln(ω_f/ω_i)|`; **cyclic** adiabatic → `r→0`; `FrequencyWaveform` rejects non-positive/non-finite `ω`, non-finite `d ln ω/dt`; qutip↔jax agree at 1e-3. **§26 conventions-test green before seal.**
- **SQ2/SQ3:** vacuum `V=I`; squeezed-vacuum principal variances `e^{∓2r}`; **thermal-squeezed `r` invariant under `n̄_th`** (covariance-eigenvalue vs `tr V` regression); `α` matches §7; Wigner vacuum isotropic at the pinned width.
- **SQ4:** `P_odd=0` for pure squeezed vacuum; parity-aware truncation convergence in both `n̄_sq` and the `Pₙ` tail; §15 raises on under-truncation (no silent low-bias).
- **SQ5:** qualitative reproduction of 2020 Fig 3 (parametric `sinh²(2π g T_mod)`) + Fig 2 quench trend; analytic-limit oracle.
- **SQ6 (optional):** displacement seeded and echo-suppressed (`δp` trend).
- **SQ7:** notebooks execute under the `pytest -m tutorial` guard.

## 4. Governance notes

- **§26 seal is the pivotal act** (R2): squeezing generator `H_sq/ℏ = −(i/4)(d ln ω/dt)(â†² − â²)` (fixed `ω(0)` basis) + vacuum-variance-1 quadrature normalisation + Wigner `g`-pin → `CONVENTION_VERSION 0.4→0.5`, via **side-car (`WP/SQ-conventions-proposal.md`) → maintainer seals** (propose-don't-apply; frozen §6/§7 untouched). Readout functionals stay observable-only.
- **§27 is reserved, not claimed here** — the multimode Gaussian ordering + `Ω` + PT sign belong to the toolbox card's future WP; WP-05 only establishes the single-mode normalisation §27 will extend. **No divergence:** the `N=1` covariance core lives in `gaussian.py` from SQ2 (R4).
- **Pending at seal:** an append-only **WORKPLAN §5.8 dispatch-track stub** (mirroring WP-04's §5.7) recording WP-05 as a v0.3.x follow-up, in lock-step with the header/footer version lines.
- Phase B (R8) requires its own card→WP and consumes the Gaussian toolbox + `iontrap-structure` `ModeConfig`.
