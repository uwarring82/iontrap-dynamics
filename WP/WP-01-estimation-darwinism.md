# WP-01 — Estimation & Darwinism Service Upgrade

**Execution work-plan for adding generic estimation, Darwinism, state-factory and common-mode-channel primitives to `iontrap-dynamics`**

WP-01 v0.2 · ID WP-01-estimation-darwinism · Drafted 2026-06-02 · Ratified 2026-06-02 · Oxford British English · **Lifecycle: Released-pending** (per `WP/README.md` §3). All WIs (WI-1…WI-4) + the six §7 benchmarks landed (dispatches **EDA–EDF**); CONVENTIONS **§19–22 sealed** at the v0.3 Convention Freeze 2026-06-03 (commit `fdcd20f`, combined with WP-02 §23–24, `CONVENTION_VERSION` 0.2 → 0.3). **Remaining:** the release tag (the §10 release cut).

**Classification:** Sail execution under Coastline gates (per T(h)reehouse +EC CD 0.9), per `WP/README.md` §1/§6 — an authored, revisable execution plan. It is *constraint-heavy*: most sections state binding gates (work items, acceptance criteria, the convention-freeze procedure, benchmark oracles), tagged `*(Coastline)*` / `*(Coastline gate)*`; the risk register (§12) and the reuse/narrative passages are `*(Sail)*`. No section relaxes a Coastline gate.
**Licence:** CC BY-SA 4.0 (WP/ planning material, per `WP/LICENCE`). The *code* it plans inherits MIT per the §4 split-licence architecture of `WORKPLAN_v0.3.md`; the *literature-review note* it plans (§5) is **Coastline / CC BY-SA 4.0** (ratified 2026-06-02) because CONVENTIONS §19–22 cite it as authoritative — see §5.
**Stewardship:** U. Warring, AG Schätz. Under T(h)reehouse +EC corporate design (cd-rules v1.7.1, consumed via Model B).
**Endorsement Marker:** Local candidate framework. No external endorsement implied.
**Maps to:** Task Card `task cards/task-card-iontrap-dynamics-service-upgrade.md` (ID TC-ITD-ESTDARW-01 v0.1, 2026-06-01) · Stream L of Work Programme TMC-WP-0 v0.3. Library-only, application-agnostic.
**Dispatch track:** lands in `WORKPLAN_v0.3.md` as a new numbered amendment subsection §5.4 (`*(Coastline, new in v0.3.6)*`), following the §5.3 template. The §13 stub carries the minted code **EDA** and is **paste-ready when WI-1 (EDA) lands** on `main`.

---

## 1. Purpose and the No-TMC governing invariant *(Coastline)*

This work-plan executes TC-ITD-ESTDARW-01: add four general capabilities — estimation (CFI/QFI/Cramér–Rao), quantum Darwinism (redundancy and recoverability), GHZ/cat state factories, and a correlated common-mode channel — to `iontrap-dynamics` as a reusable, application-agnostic service surface, then hand off a tagged release.

**The gating constraint — No-TMC invariant.** The library must stay free of any application framing. **No TMC content, no "temporal" interpretation, no record-model / arms / discriminants — anywhere in this work.** Every new symbol must be well-defined on **generic inputs** (a state, a channel, a partition, a generator). This invariant is binding and overrides convenience: a symbol that cannot be specified without naming an arm, a record, a discriminant, a falsifier, or a Ledger entry does not belong in this library and must be pushed downstream to the consuming application.

The generic benchmarks of §7 are the **proof** of this invariant: each reproduces a standard textbook result with no application context. The single most informative proof is the QFI-scaling benchmark (GHZ N² vs product N) — see §7, marked the keystone.

**Out of scope (restated, binding).** Record model, arms A/C/F, discriminants D1–D3, the falsifier, Ledger entries; any hardware; Phase-1 Hamiltonian embedding of the new channels. The sole `"Ledger"` token in this repo lives in the task card and is explicitly out of scope. These belong to the consuming application (`broadcast-protection`, provisional), not the library.

---

## 2. Reuse-first posture *(Coastline)*

Per Design Principle 1 ("Conventions before code") and Principle 5 ("One way to do it"), every WI below reuses an existing module's shape rather than inventing a parallel one. The ground-truth hooks, verified against `src/`:

| New capability | Mirrors existing shape | Ground-truth module |
|---|---|---|
| QFI / CFI trajectory measures | `entanglement.py` trajectory-evaluator signature `(states, *, hilbert, …) -> NDArray[np.float64]` | `src/iontrap_dynamics/entanglement.py` |
| Estimation analytic oracles | `analytic.py` pure closed-form, keyword-only, SI | `src/iontrap_dynamics/analytic.py` |
| Darwinism mutual-information measures | `entanglement.py` (nonlinear-in-ρ evaluator) + `rho.ptrace([...])` against §2 indices | `entanglement.py`, `hilbert.py` |
| GHZ / cat factories | `states.py` factories (`ground_state`, `coherent_mode`, `squeezed_*`) | `src/iontrap_dynamics/states.py` |
| Common-mode channel | `systematics/drift.py` + `jitter.py` frozen-dataclass spec + `apply_*`/`perturb_*` free function | `src/iontrap_dynamics/systematics/{drift,jitter}.py` |
| Parity / generators | `observables.parity(...)`, `hilbert.spin_op_for_ion(...)` | `observables.py`, `hilbert.py` |

Subsystem ordering everywhere is CONVENTIONS.md §2: spins first (ion-index order), modes after (order in `system.modes`). `HilbertSpace` is the ground-truth for tensor layout. All operators/states are QuTiP `Qobj`. There is **no existing QFI implementation anywhere in `src/`** — it must be authored new, but mirroring the `entanglement.py` evaluator signature.

---

## 3. Subpackage-naming decision — ratified *(Coastline — ratified 2026-06-02)*

The task card marks subpackage naming **provisional**: `estimation/` + `darwinism/` versus a single `information/` umbrella. This is a one-way door (it sets the public import path the downstream application pins), so it is recorded here as an explicit decision the maintainer ratifies **before WI-1 opens**.

| Option | Public surface | For | Against |
|---|---|---|---|
| **A — split** `estimation/fisher.py`, `darwinism/redundancy.py`, `darwinism/recoverability.py` | `from iontrap_dynamics.estimation import …`; `from iontrap_dynamics.darwinism import …` | Matches the card's WI table verbatim; mirrors existing `systematics/` and `measurement/` sub-package precedent (two-tier re-export already exists); each name maps 1:1 to a literature topic and a CONVENTIONS section | Two new top-level sub-packages; QFI/CFI and redundancy share helpers (binary entropy, ptrace masks) that would live in neither |
| **B — umbrella** `information/fisher.py`, `information/redundancy.py`, `information/recoverability.py` | `from iontrap_dynamics.information import …` | One import root; shared helpers (`_von_neumann_entropy_bits`, ptrace-mask builders) sit naturally together; "information" is the honest generic super-category for Fisher information *and* mutual information, with zero application framing | Loses the literal `estimation`/`darwinism` topic split; one CONVENTIONS section becomes broader; departs from the card's named modules |

**Recommendation: Option B — `information/` umbrella.** Rationale: (a) it is the most No-TMC-honest name — "information" is the generic mathematical category, whereas a downstream reader might mis-read "darwinism" as application framing; (b) CFI, QFI and fragment mutual information genuinely share the same nonlinear-in-ρ helper layer (`_ensure_density`, `_von_neumann_entropy_bits`, ptrace masks) that `entanglement.py` already factors, so one sub-package avoids a helper home with no owner; (c) it still maps cleanly onto distinct modules (`fisher.py`, `redundancy.py`, `recoverability.py`) and distinct CONVENTIONS sections. The partition "library-general only" holds regardless of the name chosen.

!!! note "Decision status — ratified 2026-06-02"
    **Ratified: Option B — `information/` umbrella.** The maintainer ratified the recommended root on 2026-06-02 (logged in `WP/LOGBOOK.md`); `information/` now replaces the former `<info>/` placeholder throughout §4 and §7. The WP as a whole is now **Ratified** (2026-06-02; see header), and dispatch codes EDA–EDF are minted (§15). The §13 stub (code EDA) is paste-ready once WI-1 lands on `main`.

    - [ ] Option A — `estimation/` + `darwinism/` split
    - [x] Option B — `information/` umbrella (ratified)

Below, `information/` denotes the ratified root; GHZ/cat go to `states.py` and the common-mode channel goes to `systematics/` regardless of choice.

---

## 4. Work-item plan WI-1…WI-4 *(Coastline)*

Each WI adds typed, SPDX-headed modules (`# SPDX-License-Identifier: MIT` as line 1, then the module docstring, then `from __future__ import annotations`). Each new Unicode-bearing `src/` module must self-register a `per-file-ignores` entry `["RUF001","RUF002","RUF003"]` in `pyproject.toml` to use Greek glyphs as its siblings do (**forward note:** once `information/` holds two or three Unicode-bearing modules, collapse these per-module entries into one directory glob `information/*` in `pyproject.toml` rather than repeating them); `mypy --strict` and `N802` apply to all `src/` modules; `py.typed` is inherited package-wide. Hatchling packages the whole `src/iontrap_dynamics` tree — no `pyproject` `packages` edit is needed for a new module or sub-package.

### 4.1 WI-1 — Estimation (CFI / QFI / Cramér–Rao) *(Coastline, new module)*

| Field | Value |
|---|---|
| Module path | `src/iontrap_dynamics/information/fisher.py` |
| Public symbols | `quantum_fisher_information_trajectory(states, *, hilbert, generator) -> NDArray[np.float64]` (SLD-QFI, pure- and mixed-state); `classical_fisher_information(probabilities, *, parameter_derivative) -> float`; `cramer_rao_bound(fisher: float) -> float`; `linear_gaussian_fisher(*, A, sigma) -> NDArray[np.float64]` (the `F = AᵀΣ⁻¹A` helper) |
| Reuse hooks (by name) | `analytic.py` generators (`coherent_state_mean_n`, `generalized_rabi_frequency`) as oracle line; `observables.parity`, `hilbert.spin_op_for_ion(sigma_z_ion(), i)` to build the collective generator `J_z = Σ σ_z` (same primitive `parity` uses); `entanglement._ensure_density` pattern for ket→dm |
| Signature shape | QFI is nonlinear in ρ → mirror the `entanglement.py` evaluator signature `(states, *, hilbert, <selector>) -> NDArray[np.float64]`, **not** the `Observable`/`expectations_over_time` shape; requires `storage_mode=StorageMode.EAGER` and runs as post-processing |
| Convention introduced | SLD-QFI definition (`F_Q = Tr[ρ L²]`, `L` the symmetric logarithmic derivative), the CFI definition, the CRB; SLD eigenvalue cutoff for the mixed-state form documented |
| Acceptance (from card) | Reproduces the §7 estimation oracles; **CFI ≤ QFI holds numerically** (Braunstein–Caves); QFI_GHZ = N², QFI_product = N within tolerance |
| `__init__.py` export | add each symbol to `information/__init__.py` (own grouped import + `__all__`), then re-export from package `__init__.py` `from .information import (...)` block in module-alphabetical position; add each name to top-level `__all__` (ALL-CAPS first, then alphabetical) |

### 4.2 WI-2 — Darwinism (redundancy + recoverability) *(Coastline, new module(s))*

| Field | Value |
|---|---|
| Module paths | `src/iontrap_dynamics/information/redundancy.py`, `src/iontrap_dynamics/information/recoverability.py` |
| Public symbols | `fragment_mutual_information(state, *, hilbert, system_indices, fragment_indices) -> float`; `partial_information_plot(state, *, hilbert, system_indices, environment_indices) -> NDArray[np.float64]` (the I(S:F) vs fragment-size curve); `redundancy(state, *, hilbert, system_indices, environment_indices, delta=0.1) -> float` (R_δ = N/N_δ = 1/f_δ); `recoverability(state, *, hilbert, system_indices, accessible_indices) -> float` (residual-information after conditioning) |
| Reuse hooks (by name) | `entanglement.py` trajectory-evaluator shape + `_ensure_density`, `_von_neumann_entropy_bits`; `rho.ptrace([...])` against §2 ordering for system/fragment partitions; `states.compose_density` to assemble the canonical system+environment test states |
| Convention introduced | mutual-information convention I(S:F) = S(ρ_S)+S(ρ_F)−S(ρ_SF); the **R_δ deficit convention** (information deficit δ, fragment fraction f_δ); the recoverability measure (residual information after conditioning on accessible fragments) |
| Acceptance (from card) | Reproduces the §7 Darwinism **plateau** (I(S:F) ≈ H_S) and **recoverability endpoints** (perfect recovery → residual = full; full decoherence → residual = 0; monotone between) |
| `__init__.py` export | as WI-1: per-module `__all__` → `information/__init__.py` → package `__init__.py` re-export + top-level `__all__` |

### 4.3 WI-3 — GHZ and cat state factories *(Coastline, extends `states.py`)*

| Field | Value |
|---|---|
| Module path | extend `src/iontrap_dynamics/states.py` (no new module) |
| Public symbols | `ghz_state(hilbert: HilbertSpace) -> qutip.Qobj` (full-space ket, mirrors `ground_state(hilbert)` — the one existing factory that takes `hilbert` and returns full-space output); `cat_mode(fock_dim: int, alpha: complex, *, parity: str = "even") -> qutip.Qobj` (single-mode, mirrors `coherent_mode(fock_dim, alpha)` exactly — bare `fock_dim`, normalised ket, `ConventionError` on `fock_dim <= 0`) |
| Reuse hooks (by name) | `states.ground_state` (full-space template for GHZ); `states.coherent_mode` / `squeezed_*` (single-mode template for cat); `states.compose_density` for composing the cat into a full system; `hilbert.spin_op_for_ion` for the spin-only GHZ ket |
| Convention introduced | GHZ phase/sign convention in terms of |↓⟩/|↑⟩ (consistent with CONVENTIONS §0.A Bell-state convention; the legacy `qc.py` non-standard cat phase is *not* adopted); cat-state parity convention (even/odd superposition of |α⟩, |−α⟩) |
| Acceptance (from card) | State properties match analytic — GHZ **parity oscillates at N·φ** (checked via `observables.parity`); entanglement (log-negativity / EoF via `entanglement.log_negativity_trajectory`, `entanglement_of_formation_trajectory`) as expected; `coherent_state_mean_n(alpha)` consistency for the cat |
| `__init__.py` export | **`states.py` is currently NOT imported in package `__init__.py`** — exposing `ghz_state`/`cat_mode` requires adding a **brand-new** `from .states import (...)` block (between `.spectrum_observables` and `.systematics`) listing `cat_mode, coherent_mode, compose_density, ghz_state, ground_state, squeezed_coherent_mode, squeezed_vacuum_mode`, plus each new name into top-level `__all__`, plus the names into `states.__all__` |

### 4.4 WI-4 — Common-mode channel *(Coastline, extends `systematics/`)*

| Field | Value |
|---|---|
| Module path | `src/iontrap_dynamics/systematics/common_mode.py` |
| Public symbols | `@dataclass(frozen=True, slots=True, kw_only=True) class CommonModePhase` (fields `sigma_rad: float`; **`correlation: float = 1.0`** — 0 → independent per subsystem, 1 → one shared latent across all; `label: str = "common_mode_phase"`; `__post_init__` raising `ValueError` for `sigma_rad < 0` or `correlation ∉ [0, 1]`; `sample_offsets(*, n_subsystems: int, shots: int, rng) -> np.ndarray` of shape `(shots, n_subsystems)`); `perturb_common_mode(drives: Sequence[DriveConfig], spec: CommonModePhase, *, shots: int, seed: int | None = None) -> tuple[tuple[DriveConfig, ...], ...]` |
| Reuse hooks (by name) | mirror `systematics/jitter.py` (`PhaseJitter`, `perturb_phase`) and `drift.py` (`PhaseDrift`) — frozen dataclass + `dataclasses.replace` + `np.random.default_rng(seed)`; **KEY DEPARTURE:** existing `perturb_*` draw `shots` *independent* offsets for *one* drive. Common-mode draws, per shot, a shared latent `ξ_shared ~ N(0, σ²)` **and** per-subsystem `ξ_i ~ N(0, σ²)`, returning `offset_i = √c·ξ_shared + √(1−c)·ξ_i` with `c = correlation` (see convention row). At `c = 1` one offset is broadcast across all subsystems; at `c = 0` it reduces to the independent loop. Implement the general `c`, **not** only the two limits — do not copy the independent-sampling loop wholesale |
| Convention introduced | common-mode (shared-latent) channel definition: per shot, subsystem *i* gets `offset_i = √c·ξ_shared + √(1−c)·ξ_i` with `ξ_shared, ξ_i ~ N(0, σ_rad²)` and `c = correlation ∈ [0, 1]` — `c = 0` ⇒ independent (reduces to `PhaseJitter`/`PhaseDrift`), `c = 1` ⇒ one shared latent (full common mode); `σ_rad` is the marginal per-subsystem standard deviation at every `c` |
| Acceptance (from card) | **At `correlation = 0`** the per-subsystem offset distribution is statistically identical to independent `PhaseJitter`/`PhaseDrift` (KS / variance test); **at `correlation = 1`** the shared offset cancels in the difference observable (common-mode rejection); the difference-observable variance is monotone in `correlation` between the limits |
| `__init__.py` export | add to `systematics/__init__.py` (grouped import + `__all__`), then package `__init__.py` re-exports from `.systematics` (the two-tier sub-package re-export template already exists) |

---

## 5. Literature-review note plan *(Coastline plan; note is Coastline / CC BY-SA 4.0, ratified 2026-06-02)*

**Deliverable.** `docs/estimation-darwinism-review.md` (currently absent). It records the chosen definitions (SLD-QFI convention, R_δ deficit convention, recoverability measure), each anchored to a cited source. **Rule (binding): every new CONVENTIONS section added in §6 cites this note, and every definition in the note cites a primary source.** The review is bounded, not open-ended (task card §5): it exists only to (a) fix canonical definitions and conventions and (b) identify the §7 analytic oracles.

**Page placement and styling** (docs conventions): H1 first line, **no front matter**; `##`/`###` sections; pipe tables for the source matrix; relative `.md` cross-links to `framework.md`, `benchmarks.md`; GitHub-blob URLs for external refs; `!!! note`/`!!! tip` admonitions for caveats. **Licence (ratified 2026-06-02 — Coastline / CC BY-SA 4.0).** Because CONVENTIONS §19–22 cite this note as *authoritative for their definitions*, it is **Coastline / CC BY-SA 4.0** and closes with a `## Endorsement Marker` (as the Coastline docs do), **not** the Sail tutorials' CC BY-NC-SA `## Licence` footer. Any embedded figure carries descriptive alt text to clear the WCAG Level-A `accessibility` gate without new ignore codes.

**Seed references by topic** (confirm exact citations and years in session):

| Topic | Seed references | Feeds |
|---|---|---|
| QFI & estimation | Helstrom (quantum detection & estimation); Braunstein & Caves (statistical distance, ~1994); Paris ("Quantum estimation for quantum technology", ~2009); Tóth & Apellaniz (metrology review, ~2014) | WI-1 SLD-QFI convention, CFI ≤ QFI oracle |
| Heisenberg-limit / GHZ metrology | Giovannetti, Lloyd & Maccone; Bollinger, Itano, Wineland & Heinzen (~1996) | WI-1/WI-3 QFI_GHZ = N² oracle |
| Quantum Darwinism & redundancy | Zurek (Nat. Phys., ~2009); Ollivier, Poulin & Zurek (~2004–05); Blume-Kohout & Zurek; Riedel, Zurek & Zwolak | WI-2 R_δ deficit convention, plateau oracle |
| Recoverability / QEC information | Knill & Laflamme (~1997); Schumacher & Nielsen (coherent information, ~1996); Petz (recovery map); Fawzi & Renner (~2015); Bény & Oreshkov (~2010); Wilde (textbook) | WI-2 recoverability measure, endpoint oracles |
| Common-mode / correlated noise | correlated-dephasing / common-mode-rejection treatments in precision/clock spectroscopy (cited only for the generic channel definition) | WI-4 channel convention |
| Numerics | Johansson, Nation & Nori (QuTiP) | all WIs (backend reference) |

---

## 6. Convention-Freeze plan *(Coastline)*

CONVENTIONS.md is **frozen at v0.2 (§1–18)**. Post-freeze additions to frozen sections are **not free** — they require a minor bump and a CHANGELOG entry (per the docstring on `CONVENTION_VERSION` and the §17/§18 closing paragraphs). **WP-01 does not bump the version on its own:** it *contributes* sections §19–22 to the **shared v0.3 Convention Freeze** coordinated in [`WP/FREEZE-v0.3.md`](FREEZE-v0.3.md), which owns the single `CONVENTION_VERSION` 0.2 → 0.3 bump, the section-number allocation (WP-01 §19–22; WP-02 §23–24), the `test_convention_version.py` guard, and the seal-at-release (including the timeline-coupling decision in `FREEZE-v0.3.md` §4). The mechanics below are reproduced for WP-01's contribution; the **bump and seal are executed once, by the side-car**, not per WP.

**Version-bump mechanics** (owned by `WP/FREEZE-v0.3.md` §3 — executed once for the whole v0.3 freeze, not per WP; reproduced here for WP-01's reference):

- Constant: `CONVENTION_VERSION: str = "0.2"` in `src/iontrap_dynamics/conventions.py` (module-level, in `__all__` alongside `FOCK_CONVERGENCE_TOLERANCE`). **Bump `"0.2"` → `"0.3"`.**
- Verify the stamped metadata key against `results.py` before relying on it: docs use `convention_version` (§17.5) and `conventions_version` (§13) inconsistently. The provenance stamp the benchmarks read is `convention_version` in `manifest.json` `metadata` (verified in real artefacts). Resolve the doc inconsistency in the same commit.
- No existing test pins the **literal value** of `CONVENTION_VERSION`: the seven `tests/unit/` references import it as a symbol to stamp or compare metadata (`convention_version=CONVENTION_VERSION`, `assert system.convention_version == CONVENTION_VERSION`) — none asserts it equals a frozen literal, so a silent bump-skip is currently undetected. **Add** `tests/conventions/test_convention_version.py` asserting `CONVENTION_VERSION == "0.3"` so a future freeze cannot silently skip the bump. Marker `convention`.

**New CONVENTIONS sections (continue the integer sequence; current max is §18):**

| New § | Title | Cites review (§5) | Backs |
|---|---|---|---|
| `## 19. Quantum and classical Fisher information *(staged — v0.3 Convention Freeze target)*` | SLD-QFI, CFI, Cramér–Rao | Braunstein–Caves; Paris | WI-1 |
| `## 20. Quantum Darwinism — redundancy and recoverability *(staged …)*` | I(S:F), R_δ deficit, recoverability | Zurek; Ollivier–Poulin–Zurek; Petz; Fawzi–Renner | WI-2 |
| `## 21. GHZ and cat state conventions *(staged …)*` | GHZ phase, cat parity | Bollinger et al.; GLM | WI-3 |
| `## 22. Common-mode (shared-latent) channel *(staged …)*` | shared draw across subsystems | correlated-noise refs | WI-4 |

*WP-02 contributes §23–24 (two-mode squeezing; motional CPTP channels) to the **same** v0.3 freeze — see `WP/FREEZE-v0.3.md` §2. WP-01 stages §19–22 only.*

**Draft-early-freeze-at-release procedure** (lifecycle staged → frozen, from §17/§18; this procedure is shared — `WP/FREEZE-v0.3.md` applies it to every contributing section, §19–24):

1. Open each section with a `**Status:**` line naming the opening Dispatch and source path; tag the H2 `*(staged — v0.3 Convention Freeze target)*`. Tag incremental subsections `### N.M … *(added in Dispatch …)*`.
2. Each section's shape: definition block → `**Rule.**`/`**Convention.**`/`**Status:**` → cross-refs (`§N.M`) → `**Test.**` pointer to `tests/conventions/test_*.py` and/or `tests/regression/analytic/test_*.py`.
3. At the release commit, **seal** each: `**§N freeze.** … a complete read-through for the Convention Freeze gate at the v0.3 release. Post-freeze additions require a CONVENTIONS.md version bump.`
4. Update the header block (`Status:` line, `**Scope:** Conventions covering §1–22`, the freeze narrative naming the new version and which §§ it adds).
5. Update the Endorsement Marker to list the newly closed §19–22 and restate §1–18 carry-forward.
6. Update the footer: `**Convention version:** 0.3 · 2026-06-XX · v0.3 Convention Freeze.` and `**Workplan reference:**`.
7. Any new `src/` module under the new conventions is AST-scanned automatically by `tests/conventions/test_static_conventions.py`: **no `from qutip import *`, no `from qutip import sigmaz`, no `qutip.sigmaz`** — route all Pauli use through `iontrap_dynamics.operators`.
8. CHANGELOG entry (mandated by the constant's docstring for any version-changing edit).

---

## 7. Generic-benchmark plan *(Coastline)*

Each feature gets a generic benchmark following the repository harness: a `tools/run_benchmark_<name>.py` writing `benchmarks/data/<name>/` with the canonical artefact set, validated against an analytic oracle within a stated tolerance, containing **zero application context**. The canonical write sequence inside `main()` is fixed: build `parameters` → `request_hash = compute_request_hash(parameters)` → `result = solve(..., request_hash=request_hash, storage_mode=StorageMode.OMITTED, provenance_tags=("benchmark", "<scenario>"))` → `save_trajectory(result, OUTPUT_DIR, overwrite=True)` (the sole writer of `manifest.json` + `arrays.npz`) → hand-write `demo_report.json` → optional `np.savez(analytic_overlay.npz, …)` → optional matplotlib `plot.png`.

**Canonical artefact set + provenance schema (cite verbatim):**

- `manifest.json` (from `save_trajectory`, `storage_mode=OMITTED` only): `cache_format_version` (int, =1); `metadata` = `{convention_version, request_hash, backend_name, backend_version, storage_mode, fock_truncations, provenance_tags}`; `expectation_labels`; `warnings`.
- `arrays.npz`: `_TIMES_KEY = "times"` (SI seconds) + `_EXPECTATION_PREFIX = "expectation__"` + `<label>` per observable; the loader rejects any missing/extra key.
- `demo_report.json` (hand-written, `schema_version = 2`): `scenario, workplan_reference, threshold_seconds, elapsed_seconds, parameters, arrays_schema_note, canonical_request_hash, environment, generated_at, schema_version`, plus oracle keys `analytic_formulas` and `max_numerical_vs_analytic_error`.
- `plot.png` (optional, graceful skip; **descriptive alt text required** wherever embedded in docs).
- `analytic_overlay.npz` (only if an oracle exists; kept SEPARATE from `arrays.npz`).

!!! note "Two schema namespaces — do not conflate"
    `cache_format_version: 1` (manifest, in `cache.py`) is distinct from `schema_version: 2` (demo_report, tool-local) and from the migration-reference `schema_version == 1`. The provenance schema has **no `tolerance` or `oracle` field** — tolerance/oracle live only in `demo_report.json` (`max_numerical_vs_analytic_error`, `analytic_formulas`) and in the binding `tests/` assertion. There is no shared demo-report/env/plot helper: `_environment()` and the `demo_report` dict shape are replicated by copy across tools — a new benchmark duplicates them.

!!! note "Compute-only vs solve-based report file (landed precedent)"
    A benchmark that runs **no `solve()` trajectory** (the keystone QFI-scaling benchmark is the first such here — it constructs states and computes QFI, no dynamics) emits **`report.json`**, not `demo_report.json`, following the `sparse_vs_dense` / `spectrum_envelope` precedent. It still carries `schema_version: 2`, `convention_version`, backend provenance, `analytic_formulas`, `max_numerical_vs_analytic_error`, `tolerance`, and `plot_alt_text`, plus `canonical_request_hash: null` (no solve → no request hash). No solve-based `manifest.json` / `arrays.npz` cache pair is produced (`arrays.npz` is still written for the numeric data). This is the resolution of the EAGER-vs-cache caveat for the non-dynamical benchmarks.

!!! tip "EAGER caveat for the nonlinear benchmarks"
    QFI, Darwinism and entanglement measures are nonlinear in ρ → they need `storage_mode=StorageMode.EAGER` (states retained). But `save_trajectory` raises `ConventionError` for any non-`OMITTED` mode, so the **canonical cache pair cannot be emitted** for a full-state run (exactly the bell-demo situation: "no canonical `manifest.json` emitted — cache v1 rejects `storage_mode=EAGER`"). For these, run an `OMITTED` pass to emit the canonical pair for any linear observable (e.g. parity), and emit the nonlinear measure into a scenario-specific extra (`information.npz` / `entanglement.npz`) plus the `demo_report.json` physics scalars — mirroring `run_demo_bell_entanglement.py`.

**The six benchmarks:**

| # | Tool | Generic setup | Analytic oracle | Tolerance | Regression-test location |
|---|---|---|---|---|---|
| **1 — keystone** ✓ **landed** | `tools/run_benchmark_qfi_scaling.py` → `benchmarks/data/qfi_scaling/` | GHZ vs product probe, vs qubit number N, under `J_z = ½ Σ σ_z` | QFI_GHZ = N² (Heisenberg); QFI_product = N (SQL) | exact (max error 1.4e-14) | `tests/regression/analytic/test_qfi_scaling.py` (`ATOL_QFI_SCALING`, marker `regression_analytic`) |
| **2** ✓ **landed** | `tools/run_benchmark_cfi_linear_gaussian.py` → `benchmarks/data/cfi_linear_gaussian/` | Linear-Gaussian model, known A, Σ; single-qubit phase readout | `F = AᵀΣ⁻¹A`; CRB = F⁻¹; CFI ≤ QFI (Braunstein–Caves) | exact (max error 6.7e-16) | `tests/regression/analytic/test_cfi_linear_gaussian.py` |
| **3** ✓ **landed** | `tools/run_benchmark_darwinism_redundancy.py` → `benchmarks/data/darwinism_redundancy/` | GHZ-cascade: 1 system + N environment qubits | mutual-information **plateau** I(S:F) = H_S; redundancy R_δ = N | exact (max error 0) | `tests/regression/analytic/test_darwinism_redundancy.py` |
| **4** ✓ **landed** | `tools/run_benchmark_recoverability.py` → `benchmarks/data/recoverability/` | Werner depolarizing-noise family ρ(p), p ∈ [0, 1] | perfect recovery → H_S; full decoherence → 0; monotone between | endpoints exact (6.7e-16); monotone | `tests/regression/analytic/test_recoverability_channel.py` |
| **5** ✓ **landed** | `tools/run_benchmark_ghz_cat.py` → `benchmarks/data/ghz_cat/` | GHZ parity fringe under `e^{-iφ J_z}`; cat parity | `⟨X^⊗N⟩ = cos(Nφ)`; cat parity ±1 | exact (max error 4.4e-16) | `tests/regression/analytic/test_ghz_cat_properties.py` |
| **6** ✓ **landed** | `tools/run_benchmark_common_mode.py` → `benchmarks/data/common_mode/` | Two-subsystem correlated dephasing; sweep correlation 0 → 1 | difference variance `2σ²(1 − c)`; `c = 0` independent, `c = 1` exact rejection | `c = 1` exact (measured, not enforced); sampled ≤ 1.4e-4 | `tests/regression/analytic/test_common_mode_rejection.py` |

*Naming: all six tools use the `run_benchmark_` prefix for consistency with the section's opening contract. Benchmark 5 (GHZ/cat) validates **state properties** rather than runtime — a "demo" in the repo's `run_demo_*` parlance — but is named `run_benchmark_` here because it is a closed-form-oracle validation, and it emits the same canonical artefact set.*

**Keystone — benchmark 1 (QFI scaling).** This is the single decoupling proof: it exercises the QFI implementation (WI-1), the GHZ factory (WI-3) and the SQL/Heisenberg distinction in one figure, **with no application framing**. It is the headline evidence for DoD point 5 ("zero TMC content"). The figure shows QFI_GHZ = N² above QFI_product = N over a sweep of N, reproduced from textbook closed forms only.

**Oracle/tolerance discipline (binding).** No mechanism re-runs a tool and diffs `demo_report.json` against a stored oracle. The tool *records* `max_numerical_vs_analytic_error` + `analytic_formulas`; the **binding assertion** is authored separately in `tests/`, with the **tolerance as a named symbolic constant in the test** (e.g. `ATOL_QFI_SCALING = 1e-3`), never a global value and never read from the artefact — following the migration-tier precedent of per-scenario symbolic tolerances.

**EDE landed (2026-06-02).** All six §7 benchmarks now exist. Benchmarks 2–6 each ship as `tools/run_benchmark_<name>.py` → `benchmarks/data/<name>/`, with a dedicated binding anchor in `tests/regression/analytic/`: `test_cfi_linear_gaussian`, `test_darwinism_redundancy`, `test_recoverability_channel`, `test_ghz_cat_properties`, `test_common_mode_rejection` (the `_channel` / `_properties` / `_rejection` suffixes keep basenames globally unique — `tests/` has no `__init__.py`). Each carries a named `ATOL_*` and reproduces its oracle (≤ 1.4e-4 for the sampled common-mode variance, ≤ 1e-15 for the rest). The "regression-test location" column above named the planning placeholder `test_analytic.py`; these per-benchmark anchors are the as-built reality.

---

## 8. Test plan across the three tiers *(Coastline)*

`pytest` config is `--strict-markers --strict-config`; any `@pytest.mark.X` must be one of the declared markers (`regression_migration`, `regression_analytic`, `regression_invariant`, `regression_reproduction`, `convention`, `benchmark`, `slow`, `gpu`).

| Tier | Location | Marker | New tests |
|---|---|---|---|
| **unit** | `tests/unit/test_*.py` | none | per-module: `test_fisher.py` (CFI/QFI/CRB/linear-Gaussian numerics, CFI ≤ QFI), `test_redundancy.py`, `test_recoverability.py` (endpoints + monotonicity), `test_states_ghz_cat.py` (norm, parity, ConventionError on `fock_dim <= 0`), `test_common_mode.py` (corr=0 reduces to independent draw; `dataclasses.replace` not mutation; `ValueError` on `shots < 1`) |
| **conventions** | `tests/conventions/test_*.py` | `convention` | `test_convention_version.py` asserting `CONVENTION_VERSION == "0.3"`; new `src/` modules auto-scanned by `test_static_conventions.py` for the qutip-import / `sigmaz` bans (no action beyond compliance) |
| **regression — analytic** | one sibling file per benchmark in `tests/regression/analytic/`, kept separate from the QuTiP-free `test_analytic.py` | `regression_analytic` | **all six benchmark oracles landed**, each with a named symbolic `atol`: `test_qfi_scaling.py` (QFI_GHZ = N², QFI_product = N), `test_cfi_linear_gaussian.py` (`F = AᵀΣ⁻¹A`, CFI ≤ QFI), `test_darwinism_redundancy.py` (plateau, R_δ = N), `test_recoverability_channel.py` (endpoints + monotonicity), `test_ghz_cat_properties.py` (GHZ `cos(Nφ)` fringe, cat parity), `test_common_mode_rejection.py` (`2σ²(1 − c)`, exact rejection) |

Out of scope for this tier set: migration tier (Phase-0 only, retiring); invariant tier may optionally gain trace/positivity checks on produced GHZ/cat states but is not required by the card. No `--cov-fail-under` gate exists; coverage is reported only.

---

## 9. Docs plan *(Coastline plan)*

Two required edits per new page (no `not_in_nav` escape hatch; `mkdocs build --strict` is a hard CI gate that promotes off-nav pages to failures): create the `.md` (H1 first line, no front matter) **and** add a `nav:` entry in `mkdocs.yml`.

| Page | Nav entry (`mkdocs.yml`) | Layer / footer |
|---|---|---|
| `docs/estimation-darwinism-review.md` | `- Literature Review — Estimation & Darwinism: estimation-darwinism-review.md` (top level, between `Benchmarks:` and `Boundary Decision Tree:`) | **Coastline** → `## Endorsement Marker` (CC BY-SA 4.0), ratified 2026-06-02 (§5) |
| New module reference (estimation/Darwinism API) | add under the API surface, or extend `phase-1-architecture.md` | Coastline → `## Endorsement Marker` |
| Benchmark write-ups | extend `docs/benchmarks.md` — one `##` section per benchmark (headline → table → ` ```bash ` `Run with:`), plots as raw-GitHub PNG with **alt text** | Coastline |

**WCAG Level-A gate (hard, `accessibility` job in `ci.yml`).** Every embedded figure carries descriptive `![alt](…)`; headings well-nested (no skips); link text non-empty. New pages must not introduce content-level Level-A violations and must **not** add new codes to `WCAG2A_IGNORES` without a justification comment (theme-level only). AA is advisory (`continue-on-error`).

---

## 10. CHANGELOG + release plan *(Coastline)*

CHANGELOG is Keep-a-Changelog + SemVer; landed work accumulates under `## [Unreleased]` keyed by Dispatch.

**`[Unreleased]` entries to add (one bullet per dispatch, `- **Dispatch <CODE> — <title>.**`):**

The six dispatches map exactly to the §15 register (`EDA` carries WI-1 **plus** the keystone QFI benchmark as the decoupling proof; `EDE` is the remaining five benchmarks; `EDF` carries the review note **and** the §19–22 sections staged for the shared freeze):

- `### Added` — `Dispatch EDA — information: CFI/QFI/Cramér–Rao + QFI-scaling benchmark (information/fisher.py)`; `Dispatch EDB — information: redundancy + recoverability`; `Dispatch EDC — states: ghz_state + cat_mode`; `Dispatch EDD — systematics: common-mode channel`; `Dispatch EDE — five generic benchmarks (CFI / Darwinism / recoverability / GHZ–cat / common-mode) under benchmarks/data/`; `Dispatch EDF — literature-review note docs/estimation-darwinism-review.md`.
- `### Changed` — (at the **shared** v0.3 seal, per `WP/FREEZE-v0.3.md` §3) `CONVENTION_VERSION bumped 0.2 → 0.3; CONVENTIONS §19–22 (WP-01) added under the v0.3 Convention Freeze.` The bump is executed **once by the side-car**, not by WP-01 alone — if WP-02's §23–24 seal in the same v0.3, this entry also covers them.

**Release-note headline (binding).** The largest surface change is **WI-3's new `from .states import (…)` block in the package `__init__.py`** (§4.3): `states.py` is *not* currently re-exported at top level, so `ghz_state`, `cat_mode`, **and the existing `coherent_mode` / `squeezed_*` / `ground_state` / `compose_density` factories** all become importable from `iontrap_dynamics` for the first time. This is additive (no existing import breaks) but materially widens the public API surface — headline it in the release **summary**, not only as a one-line dispatch bullet.

Dispatch codes are minted when this WP is **Ratified** (§15), from a clearly-fresh family chosen after grepping `CHANGELOG.md` / `WORKPLAN_v0.3.md` / `docs/gpu-dispatch-design.md` (see the `WP/LOGBOOK.md` registry minting rule — single `A`–`Z`, doubles `AA`–`WW`, `BBA`–`BBE`, Greek `β.1`–`β.4` are taken). Each commit body ends with the `Co-Authored-By` trailer.

**Release cut (5-step, encoded only in commit bodies — there is no `RELEASING.md`):**

1. Backfill `[Unreleased]` so every shipped dispatch has an entry.
2. Bump `pyproject.toml` `version` (currently `"0.4.0"`) → the new tag.
3. Roll `[Unreleased]` into `## [X.Y.Z] — YYYY-MM-DD` with a bold **Release summary.** paragraph + **Test surface at `vX`:** pass/skip line.
4. Commit `Release vX.Y.Z — Estimation & Darwinism service surface`; body records the explicit **SemVer decision + justification** (minor, not patch, because new public API is added; not v1.0 because nothing is removed/broken) and an "Unchanged from vPrev" compatibility statement.
5. Annotated git tag.

**Handoff artefact:** the tag. The TMC application repo (`broadcast-protection`, provisional) pins `iontrap-dynamics>=<this release>` and consumes the primitives in Stream A. This work must not block the `open-iontrap` org migration and is orthogonal to the package's Phase 2 (JAX).

---

## 11. Definition of Done *(Coastline)*

Reproducing the task-card 6-point DoD, plus the decoupling proof:

- [ ] **DoD-1.** WI-1…WI-4 implemented, typed, documented, SPDX-headed; `ruff check` + `ruff format --check` + `mypy --strict` + `pytest` green in CI (3.11 and 3.12).
- [ ] **DoD-2.** The literature-review note `docs/estimation-darwinism-review.md` is committed, and **every new CONVENTIONS §19–22 definition cites it**.
- [ ] **DoD-3.** Each feature has a generic benchmark reproducing its analytic oracle within its named tolerance, with canonical `benchmarks/data/<name>/` artefacts (`manifest.json` + `arrays.npz` + `demo_report.json` where `OMITTED` is feasible; scenario-extra `.npz` + `demo_report.json` for EAGER nonlinear measures).
- [ ] **DoD-4.** New CONVENTIONS §19–22 frozen at the v0.3 Convention Freeze; `CONVENTION_VERSION` bumped `0.2 → 0.3` in `src/iontrap_dynamics/conventions.py`; CHANGELOG updated.
- [ ] **DoD-5.** Demos/benchmarks contain **zero TMC content** — the decoupling proof (see below).
- [ ] **DoD-6.** A release is tagged; the tag is the handoff artefact the application will pin.

**Decoupling proof (DoD-5, made checkable):**

- [ ] whole-word grep across `src/iontrap_dynamics/information/`, `states.py`, `systematics/common_mode.py`, `tools/run_*`, `benchmarks/data/`, `docs/estimation-darwinism-review.md`, and CONVENTIONS §19–22 returns **zero** hits for the application concepts: `TMC`, `temporal`, **`record` as the concept noun** (the TMC *record model* / *measurement record* — *not* the everyday verb "records", which appears benignly in `states.py:235` "a named alias that records the CONVENTIONS.md §6 choice" and in out-of-scope GPU tools recording VRAM, exactly the `arm64`-style collision), **`arms`** (the arm-A/C/F concept — *not* `arm`, which collides with the benign `arm64` architecture string in `report.json` provenance), `discriminant`, `falsifier`, `Ledger`, `broadcast`. Verified clean for EDA + EDE + EDF on 2026-06-02 (the review note reworded the GHZ-cascade description from "broadcast to" → "redundantly copied onto"; benchmark docstrings state absence of application framing without using the literal concept words).
- [ ] every new public symbol is specified purely on generic inputs (a state, a channel, a partition, a generator) — verified by the unit tests, none of which import or name an application concept.
- [ ] the keystone QFI-scaling benchmark (benchmark 1) reproduces QFI_GHZ = N² vs QFI_product = N from textbook closed forms with no application framing — the single headline figure of the decoupling.

---

## 12. Risks and mitigations *(Sail)*

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Application framing leaks into a symbol name or docstring (No-TMC breach) | Medium | Very high | DoD-5 grep gate in CI/review; generic-input specification rule (§2); keystone benchmark as positive proof |
| QFI authored from scratch (no precedent in `src/`) diverges from field convention | Medium | High | Anchor SLD-QFI to Braunstein–Caves in §5 note before coding; CFI ≤ QFI numerical guard; analytic-tier oracle |
| EAGER vs `save_trajectory` `ConventionError` blocks canonical cache pair for nonlinear measures | High | Medium | Documented `OMITTED`-pass + scenario-extra `.npz` pattern (§7), mirroring `run_demo_bell_entanglement.py`; do not force a cache pair where cache v1 rejects it |
| Common-mode helper copies the independent-sampling loop and silently breaks correlation | Medium | High | Explicit "KEY DEPARTURE" note (§4.4); unit test that corr=0 reduces to independent `PhaseDrift` and corr=1 cancels in the difference observable |
| Subpackage name churned after downstream pins it | Low | High | Ratify §3 decision **before** WI-1 opens; the chosen import root is one-way |
| Off-nav docs page fails `mkdocs build --strict` | Medium | Medium | Mandatory `nav:` entry per §9; do not emulate the four off-nav design notes |
| Forgotten `CONVENTION_VERSION` bump | Medium | Medium | New `tests/conventions/test_convention_version.py` pinning `"0.3"` (no such guard exists today) |
| Unicode-bearing new `src/` module fails ruff RUF00x | Medium | Low | self-register `per-file-ignores` entry per module (§4) |

**Review-cycle termination rule (binding).** One internal review of the upgrade **before** the release tag. Re-opening is warranted **only** if a benchmark fails to match its oracle. No other condition re-opens the cycle; on a clean review, tag immediately.

---

## 13. Dispatch-track stub for `WORKPLAN_v0.3.md` *(Coastline gate — full library surface + all six benchmarks landed 2026-06-02; stub ready to paste, awaiting maintainer)*

WP-01 is Ratified, dispatch codes **EDA–EDF** are minted, and **EDA–EDE are now fully landed** on `main` (2026-06-02): the entire library surface (WI-1 Fisher/CRB, WI-2 Darwinism redundancy + recoverability, WI-3 GHZ/cat factories, WI-4 common-mode channel) and **all six §7 benchmarks** with their binding `regression_analytic` anchors. Only **EDF** (the additive literature-review note + the CONVENTIONS §19–22 freeze and `CONVENTION_VERSION` bump) and the release tag remain. The stub below is therefore **ready to paste** — but pasting edits `WORKPLAN_v0.3.md`, a governed Coastline file, so it **awaits maintainer action** (not done unilaterally). To paste: insert it as the next free amendment subsection under §5 (after §5.3, current free slot §5.4 per the v0.3.x patch sequence) and update the header version line, the footer `**Workplan version:**` line, and the Endorsement Marker in lock-step in the same commit.

```markdown
### 5.4 — Estimation & Darwinism service surface as v0.3.x follow-up (2026-06-02) *(Coastline, new in v0.3.6)*

Added as the estimation/Darwinism service surface landed on `main` —
`iontrap_dynamics.information` supplies the generic estimation and quantum-
Darwinism primitives (CFI / SLD-QFI / Cramér–Rao; fragment mutual information,
redundancy, recoverability). Records the scoping decision that the
four-capability service surface (TC-ITD-ESTDARW-01, Stream L of TMC-WP-0 v0.3)
is a **v0.3.x follow-up**, not a `v0.3` blocker, and is orthogonal to the
Phase 2 JAX track (§5.3) and to the `open-iontrap` org migration (§4.0).

**Scope.** Application-agnostic only: classical/quantum Fisher information +
Cramér–Rao (WI-1); quantum-Darwinism redundancy + recoverability (WI-2);
GHZ / cat state factories (WI-3); a correlated common-mode channel (WI-4).
No TMC content, no record-model / arms / discriminants — the six generic
benchmarks (textbook oracles) are the decoupling proof.

**Rationale.** Additive — every new symbol is well-defined on generic inputs
(a state, a channel, a partition, a generator) and existing callers observe no
behaviour change. CONVENTIONS §19–22 are drafted early and frozen at the
release, with `CONVENTION_VERSION` bumped 0.2 → 0.3; this matches the
workplan's Convention-Freeze commitment without re-opening a phase.

**On `main` toward the service surface at time of this amendment.**
Dispatches **EDA–EDE landed**: the full library surface — WI-1 estimation module
(`information/fisher.py`), WI-2 Darwinism (`information/redundancy.py`,
`information/recoverability.py`), WI-3 GHZ / cat factories (`states.py`),
WI-4 common-mode channel (`systematics/common_mode.py`) — plus **all six §7
benchmarks** (QFI-scaling keystone, CFI linear-Gaussian, Darwinism redundancy,
recoverability, GHZ/cat, common-mode), each with a binding `regression_analytic`
anchor reproducing its textbook oracle (max error ≤ 1.4e-4 sampled, ≤ 1e-15 the
rest).

**Remaining sub-dispatches** (tracked for v0.3.x point releases, per
`WP/WP-01-estimation-darwinism.md`): **EDF** — the additive literature-review
note `docs/estimation-darwinism-review.md` and the CONVENTIONS §19–22 freeze
with the `CONVENTION_VERSION` 0.2 → 0.3 bump (staged through the shared
`WP/FREEZE-v0.3.md` side-car, sealed in coordination with WP-02); then the
release tag (the handoff artefact the TMC application pins).

**Consequence for §5 above.** No re-scoping of Phase 2's target; `v0.3`
remains the Phase 2 milestone. This track lands under `[Unreleased]` and is
cut as its own tagged release per the §10 release plan of WP-01, decoupled
from the JAX time-dependent track of §5.3.
```

---

## 14. Sequencing and gates *(Coastline gate)*

**Order (one-way dependencies):** WP-01 ratified (2026-06-02) → **WI-1** `information/fisher.py` + keystone QFI-scaling benchmark → WI-3 GHZ/cat factories (the keystone benchmark needs `ghz_state`) → WI-2 Darwinism → WI-4 common-mode channel → literature-review note + CONVENTIONS §19–22 **staged into the shared v0.3 freeze** (`WP/FREEZE-v0.3.md`) → one internal review → release tag. The `CONVENTION_VERSION` 0.2 → 0.3 bump and the seal are coordinated by the side-car, not done by WP-01 alone.

**Why WI-3 precedes WI-2 (deliberate, not a typo):** the keystone QFI-scaling benchmark is bundled with WI-1 and compares a GHZ state against a product state, so it needs `ghz_state` from WI-3 *before* it can run. WI-2 (Darwinism) has no such dependency on the keystone, so it follows. The order is QFI → GHZ factory → Darwinism, not the WI-number order.

**Blockers (current):** (1) *resolved* — WP-01 Ratified 2026-06-02, WI-1 open; (2) *resolved* — dispatch codes EDA–EDF minted (§15); (3) coordination with **WP-02** (undetected-modes): the *shared QFI primitive* (WP-02 F6 **consumes** this WP's WI-1 `information/fisher.py`) and the *shared v0.3 Convention Freeze* (`WP/FREEZE-v0.3.md`) — including its §4 timeline-coupling decision (combined seal vs WP-01-first), taken at WP-02 ratification. WP-01 may stage §19–22 but **must not seal or bump** before that call.

**Coastline gates every WI clears before it counts as landed** (cross-referenced to their detailed sections):

- [ ] No new convention introduced outside the §6 freeze plan; `CONVENTION_VERSION` bumped exactly once (§6).
- [ ] SPDX `# SPDX-License-Identifier: MIT` header on every new module (§4).
- [ ] Unit + analytic-regression tests green; each benchmark reproduces its oracle within the named `atol` (§7, §8).
- [ ] Dispatch-keyed `[Unreleased]` CHANGELOG bullet (§10).
- [ ] `ruff` + `ruff format --check` + `mypy --strict` + `pytest` + docs `mkdocs build --strict` + WCAG-A green in CI (§8, §9).

## 15. Dispatch register *(Sail)*

Dispatch codes for WP-01 were **minted at Ratification (2026-06-02)** as the fresh **`ED`** family (Estimation/Darwinism) — `EDA`–`EDF` — chosen after grepping `CHANGELOG.md`, `WORKPLAN_v0.3.md`, and `docs/gpu-dispatch-design.md` (taken/reserved: single `A`–`Z`, doubles `AA`–`ZZ` incl. `OO` / `QQ`–`ZZ`, `BBA`–`BBE` GPU track, `RR.1`, `P.*`, Greek `β.1`–`β.3` / `δ.2`; the `ED` root was free). Recorded here and in the global registry in `WP/LOGBOOK.md`.

| Dispatch | Maps to | CHANGELOG bullet (at landing) | Status |
|---|---|---|---|
| **EDA** | WI-1 estimation `information/fisher.py` (+ keystone QFI-scaling benchmark, bundled as the decoupling proof) | `- **Dispatch EDA — information: CFI/QFI/Cramér–Rao + QFI-scaling benchmark.**` | **landed 2026-06-02 — Dispatch EDA complete** (module + keystone benchmark `benchmarks/data/qfi_scaling/`, max error 1.4e-14) |
| **EDB** | WI-2 Darwinism `information/redundancy.py` + `recoverability.py` | `- **Dispatch EDB — information: redundancy + recoverability.**` | **landed 2026-06-02 — EDB complete** (redundancy plateau oracle; recoverability = coherent information, §20 convention ratified) |
| **EDC** | WI-3 `states.ghz_state` + `cat_mode` | `- **Dispatch EDC — states: ghz_state + cat_mode.**` | landed 2026-06-02 (tests green; `states` now public) |
| **EDD** | WI-4 `systematics/common_mode.py` | `- **Dispatch EDD — systematics: common-mode channel.**` | **landed 2026-06-02** (correlation interpolation; common-mode-rejection oracle) |
| **EDE** | the remaining five generic benchmarks under `benchmarks/data/` | `- **Dispatch EDE — five generic benchmarks (CFI / Darwinism / recoverability / GHZ–cat / common-mode).**` | **landed 2026-06-02** — all five reproduce their oracles (≤ 1.4e-4 sampled / ≤ 1e-15); anchors in `tests/regression/analytic/` |
| **EDF** | literature-review note + CONVENTIONS §19–22 **staged** into the shared v0.3 freeze (bump/seal owned by `WP/FREEZE-v0.3.md`) | `- **Dispatch EDF — CONVENTIONS §19–22 staged for the shared v0.3 freeze; review note.**` | **review note + proposal landed 2026-06-02** (`docs/estimation-darwinism-review.md` Coastline/CC BY-SA; `WP/EDF-conventions-nav-proposal.md` carries the §19–22 staged text + nav line + seal edits); **seal/bump/WORKPLAN paste pending maintainer** |

The §13 stub now carries the landed range `EDA–EDE` (full library surface + all six benchmarks), with only `EDF` (review note + §19–22 freeze) and the release tag outstanding. (This mirrors the §10 CHANGELOG plan; the keystone QFI benchmark ships with `EDA` as the decoupling proof, the other five benchmarks as `EDE`.)

## 16. Logbook hooks *(Sail)*

Entries this WP has generated in `WP/LOGBOOK.md` (dated, append-only):

- **2026-06-02** — WP management system created; TC-ITD-ESTDARW-01 received; FAIR initiative opened (framework creation).
- **2026-06-02** — WP-01 subpackage naming ratified: `information/` umbrella (the §3 sub-decision).
- **2026-06-02** — WP framework review applied; lifecycle clarified (WP-01 stays Drafted; only the naming sub-decision is ratified).
- **2026-06-02** — Round-2 review approved; literature-review note licence ratified Coastline / CC BY-SA (§5).
- **2026-06-02** — Structure ratified: two WPs + shared v0.3 freeze (`WP/FREEZE-v0.3.md`); WP-01 §6 reframed to feed it, not own the bump.
- **2026-06-02** — WP-01 **Ratified**; dispatch family `ED` minted (EDA–EDF); WI-1 opened for execution.

**Future hooks (expected):** one entry at WP-01 **Ratified** (with dispatch-code minting); one per decision-with-rejected-options during WI execution; one per dead-end / deferral; one at the release-cut (5-step, SemVer justification).

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally validated laws. This work-plan is a Sail execution plan under Coastline gates within the Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg) pending external contributor onboarding. Lock–Key rule applies: this document specifies the stable locks of the planned service surface; individual analyses built on top are keys. The repository adopts the T(h)reehouse +EC Corporate Design blueprint (`cd-rules`, consumed via Model B). The No-TMC invariant (§1) is binding: this library upgrade carries zero application framing, and the §7 generic benchmarks are its proof.

**Council status:** Guardian cleared (No-TMC invariant restated as the gating constraint; decoupling proof reproduced as a checkable DoD item). Architect approved (every WI mirrors an existing module shape — `entanglement.py` evaluator signature for QFI/Darwinism, `states.py` factory pattern for GHZ/cat, `systematics/` frozen-dataclass + `perturb_*` pattern for common-mode; QFI authored new but to the established signature). Scout horizon signals addressed (subpackage-naming one-way door surfaced for ratification before WI-1; EAGER-vs-cache-v1 caveat documented; forgotten-bump guard added). Integrator has sequenced the track: ratify naming → WI-1 + keystone benchmark → WI-2/3/4 → review-note + CONVENTIONS §19–22 freeze + `CONVENTION_VERSION` bump → one internal review → release tag.

**Convention version:** references `CONVENTIONS.md` v0.2 (frozen 2026-04-21); WP-01 contributes §19–22 to the **shared** v0.3 Convention Freeze (`WP/FREEZE-v0.3.md`, which owns the single 0.2 → 0.3 bump); §1–18 carry through unchanged.
**Corporate design version:** `cd-v1.7.1`.
**Workplan version:** WP-01 v0.1 · maps to Task Card TC-ITD-ESTDARW-01 v0.1 / Stream L of TMC-WP-0 v0.3 · lands in `WORKPLAN_v0.3.md` as amendment §5.4 (new in v0.3.6) per §13.
