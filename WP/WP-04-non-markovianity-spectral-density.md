# WP-04 — Non-Markovianity Measure & Spectral Density (Phase A)

**Executes Phase A of the non-Markovianity + spectral-density task card: trace-distance + BLP non-Markovianity primitives, the quantum-projection-noise (QPN) bias estimator, a single-mode Wittemer-2018 reproduction benchmark, and two tutorials — all observable-only.**

Version 0.1 · Drafted 2026-06-17 · **Ratified 2026-06-17** · Status: **In-flight** (ND1, ND3 landed on `main`; ND4, ND5 remaining)

**Origin / rooting:** the frozen deliberation & survey record `task cards/TC-non-markovianity-spectral-density.md` (v0.5+, **not** duplicated here) — it carries the verdict, the codebase/conventions analysis, the two-phase scope, the rooting sources (PRA 97 020102 + BLP 2009 + Porras 2008 + Clos–Breuer 2012, staged under `sources/pdf/`), and the §9 open decisions. This WP is the **execution layer** for Phase A only.

**Classification:** Sail execution under Coastline gates (per T(h)reehouse +EC CD 0.9).
**Licence:** This WP document is CC BY-SA 4.0 (`WP/LICENCE`). Deliverables carry their layer's licence: code is MIT (`src/`, `tests/`); authored tutorials are Sail / CC BY-NC-SA 4.0. See root `LICENCE`.
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (`cd-rules`, Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

---

## 0. Ratification decisions (2026-06-17) *(Coastline)*

| # | Decision | Resolution | Binds |
|---|---|---|---|
| **R1** | Slug & cadence | Slug `non-markovianity-spectral-density`. Phase-A WIs are **observable-only additive slices** landed **directly on `main`** (trunk-based, per the session's established flow), with the understand→implement→adversarial-verify cadence and an external review pause per WI. | §4 |
| **R2** | **Conventions gate** | **No conventions edit, no `CONVENTION_VERSION` bump.** `D(t)`, `𝒩`, `ℬ` are standard derived observables (the MCF probe-QFI precedent): cite the frozen sections, edit nothing. Frozen §17 (measurement) and §24 (motional CPTP) are **untouched**. | all WIs |
| **R3** | Dispatch family | **`ND`** (non-markovian dynamics / distance), minted at ratification. Authoritative 5-source collision grep (`CHANGELOG.md`, `WORKPLAN_v0.3.md`, `WP/LOGBOOK.md`, `docs/gpu-dispatch-design.md`, `src/`) returns no `ND*` dispatch code (only `NDArray` typing). Distinct from the ED/MC/RL families. | §4, §6 |
| **R4** | Placement | Primitives live in the `information/` umbrella (`distinguishability.py`, `qpn_bias.py`) alongside `fisher`/`redundancy`/`recoverability` — derived, application-agnostic observables. **Not** in the frozen `measurement/` (§17) package. | ND1, ND3 |
| **R5** | Spin `Γ_dec` | **Benchmark/tutorial-local `c_ops`** (T2 `σ_z`, T1 `σ₋`) in Phase A — **no public channel symbol**, no convention. A public `SpinDephasing`/`SpinDecay` channel is a convention-gated promotion, **deferred** (task card §9). | ND4 |
| **R6** | Tomography approach | Finite-shot **per-axis projective sampling** (binomial) + Bloch reconstruction, chosen over an **analytic sampling-covariance** propagation (recorded as a **deferred alternative for later** — LOGBOOK 2026-06-17). The bias is a rectification effect a leading-order covariance only approximates. | ND3, ND4 |
| **R7** | Measure form | The **fixed-pair** BLP measure (PRA 97 020102 Eq. 1). The original sup-over-pairs Monte-Carlo optimisation is **deferred** (task card §9). | ND1 |
| **R8** | **Phase B is out of scope** | The spectral density `J(ω)` + a non-Markovian (memory-kernel / structured-bath) solver — touching the frozen §24 Markovian boundary and a possible new sealed `J(ω)` convention (§26) — are **deferred to a future WP**, triggered by a concrete structured-bath use case (task card §5/§7.2/§9). | future WP |

---

## 1. Scope (Phase A)

A single spin coherently coupled to **one** motional mode (the PRA 97 020102 realisation): evolve the closed `S+E` system (optionally with a small spin `Γ_dec`), trace out the mode, and quantify information back-flow via the trace distance and the BLP measure — fully solvable with the existing solver + `qutip.ptrace`; **no `J(ω)`, no new solver**. Phase B (continuum spectral density) is explicitly **out of scope** (R8).

## 2. Dispatch decomposition

| Code | Work item | Deliverable | Status |
|---|---|---|---|
| **ND1** | WI-1 — trace distance + BLP `𝒩` | `information/distinguishability.py` (`trace_distance`, `trace_distance_trajectory`, `blp_non_markovianity`) + BLP-2009 Eq. 14 oracle + contract tests | **landed** `996e428` |
| **ND3** | WI-3 — QPN-bias estimator | `information/qpn_bias.py` (`non_markovianity_qpn_bias`, `QpnBiasResult`) + decomposition oracle | **landed** `60265d2` (+ Bloch-norm fix `38c4293`) |
| **ND4** | WI-4 (+ WI-2) — single-mode benchmark | compute-only reproduction of Wittemer Fig 2/3: `clos2016_spin_boson_hamiltonian` (`dimensionless_mode_frequencies=[1]`) → `solve()` with `⟨σ_{x,y,z}⟩` e_ops on `|↑⟩`,`|↓⟩` → reduced-spin Bloch → ND1/ND3 primitives; spin `Γ_dec` as **benchmark-local `c_ops`** (R5). Report + arrays + plot; analytic-limit oracle. | **next** |
| **ND5** | WI-5 — tutorials | Tutorial A (effective coupling `Ω′` + spectral-density concept under added decoherence) + Tutorial B (`𝒩` + QPN bias, local probing), print+plot + Colab | reserved |

*WI-2 (spin `Γ_dec`) ships no public symbol (R5) and is folded into ND4's benchmark rather than carrying its own dispatch code.*

## 3. Acceptance gates

- **ND1:** `D = ½‖ρ₁−ρ₂‖₁` matches BLP-2009 Eq. 14 (`√(a²+f²|b|²)`); `𝒩=0` for monotone (Markovian) `D`, `𝒩>0` for revivals; §15 raises. ✅
- **ND3:** `ℬ_QPN > 0` and **→ 0 as `r → ∞`**; coarse-`γ` sampling bias `< 0`; `ℬ_QPN + ℬ_sampling = ℬ_total`. ✅
- **ND4:** qualitative reproduction of Fig 2/3 features (the resonance `𝒩(ω_z)`, the QPN-bias sign/`r`-trend) + an analytic dephasing-limit cross-check.
- **ND5:** notebooks execute under the `pytest -m tutorial` guard.

## 4. Governance notes

- Phase A adds **no convention symbol** and **no dispatch-family collision** (R2/R3). `CONVENTION_VERSION` stays `0.4`.
- **WORKPLAN §5.7 dispatch-track stub: pending** — to be added as a governed append-only amendment (header/footer version bump in lock-step) on the maintainer's go-ahead; flagged, not silently applied.
- Phase B (R8) requires its own card→WP and, if `J(ω)` becomes public, a sealed convention — out of scope here.
