# WP-02 — Two-Mode Squeezing & Motional Open-System Service Surface

**Executes the undetected-modes service card: two-mode SU(1,1) physics, typed motional CPTP channels exposed through `solve()`, interferometric observables, Lamb–Dicke regime helpers, a consumed QFI primitive, and a motional mode-frequency drift — all application-agnostic library primitives.**

Version 0.2 · Drafted 2026-06-02 · Ratified 2026-06-02 · Status: In-flight (WI-3 first; FREEZE-v0.3 §4 resolved — **Combined v0.3, bounded to WP-02 P0**)

**Classification:** Sail execution under Coastline gates (per T(h)reehouse +EC CD 0.9).
**Licence:** This WP document is CC BY-SA 4.0 (`WP/LICENCE`). Deliverables carry their layer's licence: code is MIT (`src/`, `tests/`, `.github/workflows/`); the literature-review note (§5) is Coastline / CC BY-SA 4.0; CONVENTIONS edits are Coastline / CC BY-SA 4.0. See root `LICENCE`.
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (`cd-rules`, consumed via Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.

---

## 1. Purpose and the service-module boundary invariant *(Coastline)*

WP-02 executes the undetected-modes service card by adding **application-agnostic** library primitives: a two-mode parametric (SU(1,1)) Hamiltonian, a two-mode squeezed-vacuum factory, typed motional CPTP channels wired into `solve()`, interferometric observables, Lamb–Dicke regime helpers, a QFI primitive (consumed from WP-01), and a motional mode-frequency drift. Every symbol is defined on generic spin-motion inputs — a mode pair, a channel, a probe state, a regime point — with **no reference to the consuming programme**.

**The governing boundary invariant — quoted from the card §2, not paraphrased** (hard acceptance gate):

> **One boundary must hold:** the library owns the *forward model and general primitives*; `undetected-modes` owns the *analogy, channel choices, native-subtraction, resource-constrained benchmark, and claim discipline*. Keeping this line is what preserves the library's reusability.

This is WP-02's analogue of WP-01's No-TMC invariant. Concretely, the following stay **out of** every library symbol, benchmark, and CONVENTIONS section (they belong to the consuming programme):

- the **analogy map** (what physical quantity corresponds to what in the application);
- the **specific engineered channel ε** chosen by the programme (the library ships the *typed channel family*, not a privileged choice);
- **native-subtraction / composition logic** `ε_total` (the R8 boundary);
- the **resource-constrained SU(1,1)-vs-coherent benchmark** (fixed mean excitation / interrogation time);
- the **identifiability-vs-native-drift** check and the **claim-calibration register**.

`SU(1,1)`, `two-mode squeezing`, `amplitude damping`, `Debye–Waller`, `QFI` etc. are **general physics** and are in scope — they are not application framing.

**Decoupling proof (DoD gate, §11).** A whole-word grep over the new library surface, benchmarks, and CONVENTIONS §23–24 returns **zero** hits for the application concepts: `undetected`, `analogy`, `native-subtraction` / `native subtraction`, `ε_total` / `epsilon_total`, `claim-calibration` / `claim register`, `resource-constrained benchmark`, `identifiability` (as the programme check). *(As in WP-01 §11, benign collisions are sharpened, not suppressed: e.g. `native` only as the application's native-subtraction concept, not the everyday adjective.)*

## 2. Reuse-first posture *(Coastline)*

Design Principle 1 (reuse before adding). The card §3 enumerates what is **already present — do not rebuild**, and every WI below is sited against an existing module:

| Already shipped (do not rebuild) | WP-02 builds on it |
|---|---|
| Multi-mode `HilbertSpace` (modes *a*, *b* coexist; per-mode Fock truncation) — `hilbert.py`, `modes.py` (`ModeConfig`) | WI-1, WI-2, WI-7 |
| Single-mode factories `coherent_mode` / `squeezed_vacuum_mode` / `squeezed_coherent_mode` — `states.py` | WI-2 (two-mode complement) |
| Hamiltonian builders (`carrier_hamiltonian_full_ld`, red/blue sideband) returning the `mesolve` Qobj/list surface — `hamiltonians.py` | WI-1 |
| `solve()` dispatch to `sesolve`/`mesolve` — `sequences.py:289` (today `c_ops` is not caller-exposed) | WI-3 (the pivotal `channels=` wiring) |
| `observables.py` + the finite-shot measurement layer (`SpinReadout`, Wilson CI) | WI-4 |
| `analytic.lamb_dicke_parameter`, `red/blue_sideband_rabi_frequency` (∝ η√n) — `analytic.py` | WI-5 |
| **WP-01 `information/fisher.py`** — the SLD-QFI / CFI / Cramér–Rao primitive | WI-6 (**consumes**, does not re-implement — §8) |
| Drive systematics `RabiDrift` / `DetuningDrift` / `PhaseDrift` — `systematics/drift.py` | WI-7 (`ModeFrequencyDrift` complement) |

**The one genuine gap (card §7 Q4, resolved):** `drift.py` drifts *drive* parameters only; it carries **no motional mode-frequency drift**, and since η depends on ω_mode the η-drift point needs a small new `ModeFrequencyDrift` acting on `ModeConfig` (WI-7).

## 3. Module-placement and the WP slug — proposed, ratify in §17 *(Coastline)*

Where each feature lands (reuse-first; one genuinely new module):

| Card feature | Module (proposed) | New or additive-to-existing |
|---|---|---|
| F1 two-mode SU(1,1) Hamiltonian (+ optional beamsplitter) | `hamiltonians.py` | additive to existing builders |
| F2 two-mode squeezed-vacuum factory | `states.py` | additive to existing factories |
| F3 typed motional CPTP channels + `solve(channels=…)` | **new `src/iontrap_dynamics/channels.py`** + `sequences.solve` wiring | **one new module** + a solver argument |
| F4 interferometric observables (visibility, fringe phase) | `observables.py` | additive |
| F5 Lamb–Dicke regime helpers | `analytic.py` | additive |
| F6 QFI primitive / channel-QFI | `information/fisher.py` (WP-01) — **consume**; thin channel-QFI helper only if needed | additive (§8) |
| F7 `ModeFrequencyDrift` systematic | `systematics/drift.py` (on `modes.ModeConfig`) | additive |

**Only `channels.py` is a new module.** All other features extend modules that already host their kind. This mirrors WP-01's reuse posture (it added the `information/` umbrella; everything else extended existing files).

**WP slug (ratification decision).** This WP is slugged **`two-mode-motional`** — capability-framed, parallel to WP-01's `estimation-darwinism`, and consistent with the §1 boundary discipline (the consuming application is not in the library-facing identifier). The alternative is the application-named `undetected-modes` (as the FREEZE-v0.3 side-car and the project memory currently refer to it informally). Recommendation: keep the capability slug; the card linkage (§below) makes the served programme unambiguous. **Ratify in §17.**

## 4. Card linkage *(Sail)*

Executes **`task cards/TC-iontrap-dynamics.md`** (**ID: TC-iontrap-dynamics**, v0.1 draft, *"`iontrap-dynamics` as a service module for Undetected Modes"*).

**Objective lifted from the card (one line):** add the targeted, additive forward-model and general-primitive capabilities the undetected-modes programme depends on (T1.4 numerical components, D1.4b, the MS1b theory/numerics gate), **without** redefining the library's coastline.

**Governing rules from the card — quoted, not paraphrased** (hard gates):

> Each card follows the library's own rule — *conventions before code* — so a conventions entry is part of every acceptance set.

> the library must **not** silently assume the two [channels] commute (this is exactly the R8 boundary the WP refuses to assume away).

> **Requires a documented CONVENTIONS v0.3 freeze gate:** F1 (two-mode squeezing phase/sign/ordering) and F3 (channel parameterisation; sequence-aware application). … they must be written before the code.

**Out of scope (explicitly, from card §2).** The analogy map; the choice of engineered channel ε; native-subtraction / `ε_total` composition; the resource-constrained SU(1,1)-vs-coherent benchmark; the identifiability-vs-native-drift check; the claim-calibration register. These are `undetected-modes`' own; the library ships only the general primitives they consume.

## 5. Work-item plan WI-1…WI-7 *(Sail)*

One row per atomic, separately-acceptable unit. "Acceptance" is a runnable test or a reproduced analytic oracle (the card's own acceptance criteria). Dispatch codes are **minted at ratification** (§15), not now.

| WI | Card | Module (proposed) | Key contents | Reuse | Acceptance (oracle) | Conv. | Priority |
|---|---|---|---|---|---|---|---|
| **WI-1** | F1 | `hamiltonians.py` | `two_mode_squeezing_hamiltonian(system, mode_labels=("a","b"), *, g, phase=0.0, backend="qutip")` → `H_TMS = iℏg(e^{iφ}â†b̂† − e^{−iφ}âb̂)`; optional `beamsplitter_hamiltonian(... J, φ)` → `ℏJ(e^{iφ}â†b̂ + h.c.)` | existing builder surface (Qobj/list for `mesolve`) | TMSV generated by `H_TMS` has per-mode n̄ = sinh²(gτ); su(1,1) Casimir invariant; cross-backend < 1e-3 | **§23** (freeze) | P0 |
| **WI-2** | F2 | `states.py` | `two_mode_squeezed_vacuum(fock_dims, z)` | single-mode factories | analytic Schmidt structure / per-mode n̄ = sinh²\|z\|; trace + positivity invariants | **§23** (shares F1) | P0 |
| **WI-3** | F3 | **new `channels.py`** + `sequences.solve` | typed dissipators `AmplitudeDamping(mode, rate)`, `Heating(mode, n_bar_bath, rate)`, `Dephasing(mode, rate)` → QuTiP collapse ops; optional `window=(t0, t1)` time-windowed (sequence-aware) application via QuTiP's `[op, coeff]` format; `solve(…, channels=[…])` routed to the `mesolve` path. **`Depolarising` deferred** — no canonical single-mode bosonic form (logbook 2026-06-02; re-open for a qubit-subspace depolariser if needed) | `sequences.solve` (`mesolve` path), `hilbert` mode-operator API | analytic ⟨n̂⟩ decay (damping); coherence decay (dephasing); n̄→n̄_bath steady state (heating); CPTP/trace invariants; **R8 test: temporal schedule order-dependent** (heat-then-damp ≠ reverse) | **§24** (freeze) | **P0 (pivotal)** |
| **WI-4** | F4 | `observables.py` | fringe **visibility** `V = (P_max−P_min)/(P_max+P_min)` and fitted phase shift Δφ over a parameter scan, on the spin-readout signal | `observables.py`, measurement layer | ideal SU(1,1) sequence → known V; an F3 channel degrades V as the closed form predicts | additive v0.2 | P1 |
| **WI-5** | F5 | `analytic.py` | Debye–Waller factor `∝ e^{−η²(2n̄+1)/2}`; regime **classifier** deep/intermediate/beyond from η²(2n̄+1); intermediate-order correction to the leading-order sideband-Rabi analytic | `analytic.lamb_dicke_parameter`, sideband-Rabi | Debye–Waller matches the series; classifier thresholds documented + tested; continuity with the shipped full-LD carrier numerics | additive v0.2 | P1 |
| **WI-6** | F6 | `information/fisher.py` (WP-01) | **consume** the SLD-QFI primitive for a parametrised family ρ(θ); add only a thin **channel-QFI-through-output-state** helper if a real contract is missing (§8) | **WP-01 `information/fisher.py`** | QFI of a coherent/squeezed probe matches closed forms; saturates CFI for the optimal measurement on a test case | additive v0.2 | P1 |
| **WI-7** | F7 | `systematics/drift.py` (on `modes.ModeConfig`) | `ModeFrequencyDrift` — the motional mode-frequency drift `drift.py` lacks; the rest of F7 (identifiability assembly) is **programme-side**, out of scope | `RabiDrift`/`DetuningDrift`/`PhaseDrift` pattern; `ModeConfig` | drifts ω_mode per the existing systematics contract; η-dependence propagates; invariant/round-trip test | additive v0.2 | P2 |

*Status column omitted while Drafted (all WIs `open`). The card's F7 is **mostly** programme-side; WP-02 owns only the `ModeFrequencyDrift` library addition.*

## 6. Convention-Freeze plan *(Coastline)*

`CONVENTIONS.md` is frozen at v0.2 (§1–18). WP-02 contributes **two** new sections to the **shared v0.3 Convention Freeze** coordinated in [`WP/FREEZE-v0.3.md`](FREEZE-v0.3.md) (which owns the single `CONVENTION_VERSION` 0.2 → 0.3 bump and the seal). Per the card §6, exactly F1 and F3 are freeze-gated; F2/F4/F5/F6/F7 are additive under v0.2.

| New § | Title | Cites review (§5) | Backs |
|---|---|---|---|
| `## 23. Two-mode squeezing / SU(1,1)` *(staged — v0.3 Convention Freeze target)* | phase, sign, operator ordering on the labelled mode pair; squeezing-parameter convention | two-mode-squeezing / SU(1,1) literature | WI-1, WI-2 |
| `## 24. Motional CPTP channels` *(staged …)* | rates, bath occupation, Kraus vs Lindblad; **sequence-aware (non-commuting) application** | open-quantum-systems / Lindblad literature | WI-3 |

These are exactly the §23–24 the FREEZE-v0.3 §2 allocation reserves for WP-02. WP-02 **stages** them following the §17/§18 staged→frozen procedure (the shared 8-step in WP-01 §6); it **does not** bump or seal — the side-car does that once, at the release.

**The timeline-coupling decision (FREEZE-v0.3 §4) — taken at WP-02 ratification.** This is the genuine cross-WP decision the side-car defers to here:

- **Combined (card §7 Q2 preference; recommended).** Seal §19–24 together in one v0.3 freeze, bounded to WP-02's freeze-gated subset **F1 + F2 + F3** (the card §5 minimum first milestone — "F3 + F1 + F2 on one branch, with the v0.3 conventions freeze covering both"). One bump, one worked example, one release. F4/F5/F6/F7 (additive) follow under v0.3.x without touching the freeze. **Cost:** WP-01's already-ready §19–22 wait for WP-02's F1+F2+F3 to land.
- **WP-01-first (fallback).** Seal §19–22 as the v0.3 freeze now; WP-02's §23–24 move to a later **v0.4** convention freeze (its own bump). Decouples timing; costs a second freeze gate. WP-01 is seal-ready today, so this unblocks its release immediately.

**Recommendation:** Combined, bounded to F1+F2+F3 — it matches the card Q2 preference and yields one worked example for the whole v0.3 convention surface, at the cost of holding WP-01's release until the WP-02 P0 subset lands. If WP-01's release is time-critical, WP-01-first is the honest decoupling.

**Decision (ratified 2026-06-02): Combined v0.3, bounded to WP-02 P0 (F1+F2+F3 / WI-1–WI-3).** §19–24 seal together in one v0.3 freeze once WP-02's P0 subset lands; F4/F5/F6/F7 (P1/P2) do **not** gate the seal. WP-01 stays Ratified, seal-pending: its staged artefacts (`WP/EDF-conventions-nav-proposal.md`, the single `CONVENTION_VERSION` 0.2 → 0.3 bump, the `WORKPLAN_v0.3.md` §5.4 stub) are **held in escrow** and applied together with WP-02's §23–24 in one maintainer seal commit when WP-02 P0 lands. `FREEZE-v0.3.md` §4 is updated to match.

## 7. Literature-review note plan *(Coastline plan; note is Coastline / CC BY-SA 4.0)*

Mirroring WP-01 §5: the two freeze-gated sections must be **anchored to primary sources before the code** (conventions-before-code). Deliverable: **`docs/two-mode-motional-review.md`** (the "WP-02 review note (TBD)" the FREEZE-v0.3 §2 table reserves), bounded — it fixes the §23/§24 conventions and names the analytic oracles, it is not a survey. Same house style as the WP-01 note (H1 first line, no front matter, Unicode maths, pipe-table source matrix, `## Endorsement Marker`, CC BY-SA 4.0, nav entry).

| Topic | Seed references (confirm exact citations at execution) | Feeds |
|---|---|---|
| Two-mode squeezing / SU(1,1) | Caves & Schumaker (1985) two-mode squeezed states; Yurke, McCall & Klauder (1986) SU(1,1) interferometers; Barnett & Radmore (textbook, ordering) | §23 phase/sign/ordering, WI-1/WI-2 oracles (n̄ = sinh²) |
| Open-system motional channels | Breuer & Petruccione (Lindblad master equation); Nielsen & Chuang (Kraus / CPTP, amplitude-damping); a standard ion-heating reference for the thermal-`â`/`â†` (anomalous-heating) model | §24 rates/bath/Kraus-vs-Lindblad, WI-3 oracles |
| Lamb–Dicke regime | Wineland et al. (1998) NIST review (Debye–Waller, sideband regimes); Leibfried et al. (2003) RMP | WI-5 helpers (additive, no freeze) |
| QFI | re-use WP-01's `docs/estimation-darwinism-review.md` §2 (one QFI convention, §19) | WI-6 (consumes; §8) |

WI-6 does **not** introduce a new QFI convention — it cites WP-01's §19 review note. WP-02's note covers §23–24 only.

## 8. QFI — one primitive, consumed not re-implemented *(Coastline)*

Per FREEZE-v0.3 §5 and card §7 Q1: the library hosts **one** QFI implementation — WP-01's `information/fisher.py` (SLD-QFI, §19 convention). **WI-6 consumes it.** The card marks F6 "optional… a *bare* QFI primitive", and that bare primitive already shipped with WP-01 (Dispatch EDA). WP-02 therefore:

- **Re-uses** `quantum_fisher_information_trajectory` / `classical_fisher_information` / `cramer_rao_bound` directly for the probe-QFI acceptance;
- adds **only** a thin *channel-QFI-through-the-output-state* helper **if** the existing trajectory API cannot already express it (decide during WI-6; if it can, WI-6 is a benchmark-and-docs-only dispatch with no new public symbol);
- keeps the **resource-constrained benchmark** and **identifiability-vs-native-drift** check **out** of the library (card §2 boundary; they are programme-side).

This is a dependency, not a freeze item — the QFI *convention* is §19, owned by WP-01.

## 9. Benchmark and test plan *(Coastline)*

Each freeze-gated and P0/P1 WI gets a generic benchmark following the repo harness (compute-only `report.json` schema-v2 + `arrays.npz` + `plot.png` with alt text, or a solve-based `manifest.json`/`arrays.npz` cache pair where a trajectory is run), validated against an analytic oracle with a **named symbolic `ATOL_*` in the test** (never read from the artefact), containing **zero application context** (§1).

| # | Tool (proposed) | Oracle | Test (regression_analytic) |
|---|---|---|---|
| 1 | `run_benchmark_two_mode_squeezing.py` | per-mode n̄ = sinh²(gτ); su(1,1) Casimir conserved | `test_two_mode_squeezing.py` |
| 2 | `run_benchmark_tmsv_state.py` | Schmidt spectrum / per-mode n̄ = sinh²\|z\|; trace + positivity | `test_tmsv_state.py` |
| 3 | `run_benchmark_motional_channels.py` | ⟨n̂⟩ decay (damping); coherence decay (dephasing); n̄→n̄_bath (heating); **non-commuting channels order-dependent (R8)** | `test_motional_channels.py` |
| 4 | `run_benchmark_interferometric_visibility.py` | ideal V; F3-channel degradation matches the closed form | `test_interferometric_visibility.py` |
| 5 | `run_benchmark_lamb_dicke_regime.py` | Debye–Waller series match; classifier thresholds | `test_lamb_dicke_regime.py` |
| 6 | `run_benchmark_probe_qfi.py` | coherent/squeezed-probe QFI closed form; CFI saturation | `test_probe_qfi.py` (consumes WP-01 fisher) |

**Three-tier tests** mirror WP-01 §8: unit (`tests/unit/test_*.py`), conventions (`tests/conventions/` — the new `src/` modules auto-scanned for the qutip-import / `sigmaz` bans; `test_convention_version.py` is owned by whichever WP seals first, per FREEZE-v0.3 §3), and analytic-regression (the six oracles above). `tests/regression/analytic/test_analytic.py` stays QuTiP-free; each oracle is a sibling file with a globally-unique basename.

## 10. CHANGELOG + release plan *(Coastline gate)*

Each landed dispatch gets a dispatch-keyed `[Unreleased]` bullet. **Target release tag: `v0.5.0`** — additive minor (the package is at `v0.4.0`; nothing is removed). SemVer: minor for additive capability. If the freeze is **Combined**, `v0.5.0` carries §19–24 and the single `CONVENTION_VERSION` 0.3 bump (WP-01 + WP-02 share the release); if **WP-01-first**, WP-01 ships its own earlier tag and WP-02 lands at `v0.5.0`/`v0.6.0` with a v0.4 convention freeze. The 5-step release cut (backfill → `pyproject` bump → roll `[Unreleased]` → commit with SemVer justification → annotated tag) is logged in `WP/LOGBOOK.md`.

## 11. Definition of Done *(Coastline)*

- [ ] Every WI's acceptance oracle reproduced within its named `ATOL_*`; the **R8 non-commuting-channels** test present and passing (card F3 hard requirement).
- [ ] `solve(channels=…)` routes typed dissipators to the `mesolve` path with **segment-bound** application; the old `c_ops`-hardwired behaviour is unchanged when `channels` is omitted (no caller breakage).
- [ ] §23 + §24 staged in `CONVENTIONS.md` (not sealed) and anchored to the §7 review note; F2/F4/F5/F6/F7 confirmed additive (no new convention).
- [ ] **Boundary decoupling grep (§1) clean** — zero application-concept hits across new `src/`, `tools/run_*`, `benchmarks/data/`, the review note, and CONVENTIONS §23–24.
- [ ] SPDX `MIT` header on the new `channels.py`; every new public symbol specified on generic spin-motion inputs.
- [ ] `ruff` + `ruff format --check` + `mypy --strict` + `pytest` + `mkdocs build --strict` + WCAG-A green.

## 12. Risks and mitigations *(Sail)*

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Application framing leaks into a channel choice or benchmark (boundary breach) | Medium | Very high | §1 decoupling grep gate; ship the typed *family*, never a privileged ε |
| `solve(channels=…)` silently assumes channels commute (R8 breach) | Medium | High | Segment-bound application + an explicit order-dependence test (card F3) |
| Two-mode squeezing phase/sign/ordering diverges from field convention | Medium | High | Anchor §23 to Caves–Schumaker / Yurke–McCall–Klauder in §7 before coding; Casimir-invariant oracle |
| QFI re-implemented instead of consumed (duplicate primitive) | Low | High | §8 binding: consume WP-01 `fisher.py`; new symbol only on a proven missing contract |
| Combined freeze blocks WP-01's ready release indefinitely | Medium | Medium | §6 bounds the wait to F1+F2+F3; §17 offers WP-01-first as the honest decoupling |
| `solve()` signature change breaks existing callers | Low | High | `channels` is keyword-only with an empty default; regression-test the unchanged path |

## 13. Dispatch-track stub for `WORKPLAN_v0.3.md` *(Coastline gate — drafted; held in escrow for the combined seal)*

WP-02 feeds the next free amendment after WP-01's §5.4 → **§5.5** (`*(Coastline, new in v0.3.7)*`). Drafted here per the §5.3 template. Under the ratified **Combined** decision (§6), this stub is **held in escrow** alongside WP-01's §5.4 stub and pasted by the maintainer in the **single seal commit** when WP-02 P0 lands — not at ratification, so the governed `WORKPLAN_v0.3.md` is touched exactly once for the whole v0.3 surface. Sketch:

> **### 5.5 — Two-mode & motional open-system service surface as v0.3.x follow-up *(Coastline, new in v0.3.7)*** — Added when Dispatch `MCx` lands on `main`. Records that the undetected-modes service surface (card `TC-iontrap-dynamics`) is a v0.3.x follow-up: additive two-mode SU(1,1) + motional CPTP channels + observables/regime/QFI/mode-drift, orthogonal to Phase 2 (§5.3). **Conventions §23–24** join the shared v0.3 freeze (`WP/FREEZE-v0.3.md`); F2/F4/F5/F6/F7 additive. **Remaining sub-dispatches:** the WI-1…WI-7 set. **Consequence for §5:** no re-scoping of Phase 2; lands additively toward `v0.5.0`.

## 14. Sequencing and gates *(Coastline gate)*

**Order (card §5/§7 — F3 first, it unblocks everything dissipative):**

WI-3 (F3 channels + `solve` wiring) → WI-1 (F1 SU(1,1) Hamiltonian) → WI-2 (F2 TMSV factory) → WI-6 (F6 QFI, consumes WP-01) → WI-4 (F4 observables) + WI-5 (F5 regime helpers) → WI-7 (F7 `ModeFrequencyDrift`).

*Rationale:* F3 is the pivotal enabling change (exposes `c_ops`); F1 generator and F2 factory share the §23 convention (F2 needs F1's squeezing convention); F4's full acceptance needs an F3 channel to degrade visibility; F6 only needs WP-01's already-landed `fisher.py`; F7 is independent and lowest priority.

**Blockers:** (1) the **FREEZE-v0.3 §4 combined-vs-WP-01-first** decision — **resolved at ratification 2026-06-02: Combined v0.3, bounded to WP-02 P0** (§6); WP-02 stages §23–24, the seal/bump remain a single maintainer commit when the P0 subset lands. (2) Cross-WP QFI dependency on WP-01 `information/fisher.py` — **resolved** (landed, Dispatch EDA). No open blockers.

**Coastline gates every WI clears** (as WP-01 §14): CONVENTIONS freeze respected (no convention outside §23–24); SPDX header; unit + analytic-regression green with named `ATOL_*`; dispatch-keyed `[Unreleased]` bullet; CI green incl. `mkdocs build --strict` + WCAG-A.

## 15. Dispatch register *(Sail — `MC` family minted at ratification 2026-06-02)*

Fresh family **`MC`** (Motional Channels — capturing the pivotal F3 and the open-system theme), `MCA`–`MCG`, **minted 2026-06-02** at ratification after the authoritative grep of `CHANGELOG.md`, `WORKPLAN_v0.3.md`, `docs/gpu-dispatch-design.md`, and the `WP/LOGBOOK.md` registry — **clear**, no collision (WP-01's `ED` family and the known-taken ranges are avoided; QuTiP's `mcsolve` is unused in `src/` and is not a code collision). Recorded here and in the `WP/LOGBOOK.md` dispatch-code registry.

| Proposed | Maps to | CHANGELOG bullet (at landing) | Status |
|---|---|---|---|
| `MCA` | WI-1 two-mode SU(1,1) Hamiltonian | `- **Dispatch MCA — hamiltonians: two-mode squeezing (SU(1,1)) generator.**` | minted |
| `MCB` | WI-2 `two_mode_squeezed_vacuum` | `- **Dispatch MCB — states: two-mode squeezed-vacuum factory.**` | minted |
| `MCC` | WI-3 motional CPTP channels + `solve(channels=…)` | `- **Dispatch MCC — channels: typed motional CPTP channels exposed in solve().**` | **WI-3 complete 2026-06-02** (WI-3a: 3 dissipators + `solve(channels=…)`; WI-3b: time windows + R8 non-commuting test + integrator `max_step` fix; oracles green, 1025 tests) |
| `MCD` | WI-4 interferometric observables | `- **Dispatch MCD — observables: fringe visibility + phase.**` | minted |
| `MCE` | WI-5 Lamb–Dicke regime helpers | `- **Dispatch MCE — analytic: Debye–Waller + regime classifier.**` | minted |
| `MCF` | WI-6 probe-QFI (consumes WP-01) | `- **Dispatch MCF — information: probe-QFI benchmark (consumes fisher).**` | minted |
| `MCG` | WI-7 `ModeFrequencyDrift` | `- **Dispatch MCG — systematics: motional mode-frequency drift.**` | minted |

## 16. Logbook hooks *(Sail)*

Entries this WP will generate in `WP/LOGBOOK.md` (dated):

- 2026-06-02 — WP-02 **Drafted** against card `TC-iontrap-dynamics`.
- 2026-06-02 — WP-02 **Ratified**; `MC` family minted (`MCA`–`MCG`); FREEZE-v0.3 §4 = **Combined, bounded to WP-02 P0**; all five §17 decisions resolved; execution opens on `wp02-two-mode-motional` with WI-3.
- *(per WI)* — one entry per decision-with-rejected-options / dead-end / deferral during execution.
- *(at release)* — the 5-step release cut, SemVer justification (the combined v0.3 / `v0.5.0` seal).

## 17. Decisions taken at ratification *(Coastline gate — all five ratified 2026-06-02)*

The card §7 left three open questions (Q4 resolved); WP-02 added the slug and the freeze-timeline call. **All five ratified 2026-06-02:**

1. **F6 QFI boundary (card Q1) — RATIFIED: consume.** Host the **bare** QFI in the library (WP-01 §19/`fisher.py`); WI-6 **consumes** it, no second implementation or wrapper; the resource-constraint + identifiability stay programme-side (§8).
2. **Conventions cadence / FREEZE-v0.3 §4 (card Q2) — RATIFIED: Combined, bounded to WP-02 P0.** Seal §19–24 together under the single v0.3 freeze once WP-02's **F1+F2+F3 (WI-1–WI-3)** land; P1/P2 (F4–F7) do not gate the seal. WP-01 stays Ratified, seal-pending; its staged artefacts are held in escrow and sealed in one maintainer commit when P0 lands (§6).
3. **Upstream vs fork (card Q3) — RATIFIED: branch-first.** Execute on `wp02-two-mode-motional` (branched off the WP-01 branch); mirror as upstream issues on `uwarring82/iontrap-dynamics` only after the freeze shape is stable.
4. **WP slug (§3) — RATIFIED: `two-mode-motional`.** Capability-framed, parallel to WP-01.
5. **Dispatch family (§15) — RATIFIED: `MC`** (Motional Channels), `MCA`–`MCG`, minted 2026-06-02 after a clear collision grep.

**Execution steer (ratified):** move to **In-flight starting with WI-3** (F3 — the `channels.py` + `solve(channels=…)` hinge that WI-1/WI-2 populate and the R8 test exercises); WI-1 and WI-2 follow. Per the maintainer's sequencing caveat, **map the `solve()` integration surface first** if it is not already well understood, since `solve(channels=…)` is the pivotal API and the easiest place to create churn.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This Work-Plan is a Sail execution document within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg), under the Coastline gates of `WORKPLAN_v0.3.md` and `CONVENTIONS.md`. Lock–Key rule applies: this WP is a key built on the stable locks those documents specify. The repository adopts the T(h)reehouse +EC Corporate Design blueprint (`cd-rules`, consumed via Model B).

**Council status (ratified 2026-06-02):** Guardian cleared — WP-02 introduces no convention outside the staged §23–24, relaxes no Coastline gate, relicenses nothing; it stages but does not seal the freeze (the seal is the held-in-escrow maintainer commit). Architect approved — every WI reuses an existing module (one new module, `channels.py`); the three-layer architecture and the §1 boundary are respected; QFI is consumed, not duplicated. Scout — horizon signals addressed: the FREEZE-v0.3 §4 timeline decision is taken (Combined, bounded to P0, §6/§17), the cross-WP QFI dependency resolved, and the F7 programme-side remainder surfaced. Integrator — sequenced F3-first (§14), target `v0.5.0` under the combined v0.3 freeze.

**Convention version:** references `CONVENTIONS.md` v0.2 (frozen 2026-04-21); WP-02 stages §23–24 toward the shared v0.3 freeze (`WP/FREEZE-v0.3.md`), introducing no convention on its own.
**Corporate design version:** `cd-v1.7.1` (consumed via Model B).
**Workplan reference:** `WORKPLAN_v0.3.md` v0.3.5; this WP's track lands as amendment §5.5 (`new in v0.3.7`).
