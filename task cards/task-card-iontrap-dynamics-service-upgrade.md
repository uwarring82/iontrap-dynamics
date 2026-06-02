# Task Card — `iontrap_dynamics` Service Upgrade: Estimation & Darwinism

**ID:** TC-ITD-ESTDARW-01 · **v0.1** · 2026-06-01 · Oxford British English
**Repository:** `uwarring82/iontrap-dynamics` (upstream feature branch — U is maintainer)
**Maps to:** Stream L of Work Programme TMC-WP-0 v0.3; lands as a new dispatch track in `WORKPLAN_v0.3.md`.
**Mode:** dedicated breakout session. On completion, return to the TMC programme (Stream A) to consume the released primitives.

---

## 1. Context (for handoff)

The TMC programme needs general estimation and information-theoretic primitives. Rather than embed them in an application, they are added **upstream to `iontrap_dynamics`** as a reusable service surface, then consumed by a separate study repo. This card covers **only the library upgrade**.

**Governing invariant — application-agnostic.** The library must stay free of any application framing. **No TMC content, no "temporal" interpretation, no record-model / arms / discriminants — anywhere in this work.** Every new symbol must be well-defined on **generic inputs** (a state, a channel, a partition, a generator). The generic benchmarks (§6) are the proof of this: they reproduce standard textbook results with no application context.

---

## 2. Objective

Add four general capabilities to `iontrap_dynamics`, each **grounded in a targeted literature review** (§5) and **validated by a generic benchmark** (§6), landed under the repository's Convention Freeze, and shipped in a tagged release.

The new capabilities (none exist in the package today):
1. **Estimation** — classical and quantum Fisher information, Cramér–Rao bounds.
2. **Darwinism** — quantum-Darwinism redundancy and recoverability / residual-information measures.
3. **State factories** — GHZ and cat states.
4. **Common-mode channel** — a correlated (shared latent) channel across subsystems.

---

## 3. Scope

**In.** The four capabilities above; their literature review and convention definitions; unit + regression tests; generic benchmarks with canonical artifacts; mkdocs documentation; CHANGELOG entries; a release tag.

**Out.** Anything TMC (record model, arms A/C/F, discriminants D1–D3, the falsifier, Ledger entries); any hardware; Phase-1 Hamiltonian embedding. These belong to the consuming application, not the library.

---

## 4. Work items

Each work item reuses existing package machinery where possible and adds typed, SPDX-headed modules with docstrings.

| WI | Module (proposed) | Key contents | Reuse | Acceptance |
|---|---|---|---|---|
| **WI-1** | `estimation/fisher.py` | Classical Fisher information; **quantum Fisher information (QFI)** via the SLD (pure-state and mixed-state forms); Cramér–Rao bounds; a linear-Gaussian helper `F = AᵀΣ⁻¹A` | `analytic` (generators), `observables` | Reproduces the §6 estimation oracles; CFI ≤ QFI holds numerically |
| **WI-2** | `darwinism/redundancy.py`, `darwinism/recoverability.py` | Fragment mutual information I(S:F); partial-information plot; redundancy R_δ = N/f_δ; recoverability / residual-information measure (after conditioning on accessible fragments) | `entanglement`, `states` | Reproduces the §6 Darwinism plateau and recoverability endpoints |
| **WI-3** | `states.ghz_state`, `states.cat_state` | GHZ and cat-state factories, alongside the existing `coherent_mode` / `squeezed_*` factories | `states`, `hilbert` | State properties match analytic (parity, entanglement via `entanglement`) |
| **WI-4** | `systematics` common-mode | A correlated channel applying a shared latent variable across subsystems, alongside `PhaseDrift` / `DetuningDrift` | `systematics/drift`, `drives` | Reduces to independent drift at zero correlation; common-mode signatures at full correlation |

**Subpackage naming is provisional.** `estimation/` and `darwinism/` are proposals; a single `information/` umbrella is an acceptable alternative. The partition (library-general only) holds regardless of the names chosen.

---

## 5. Literature review (targeted)

**Purpose.** Bounded, not open-ended. The review exists to (a) **fix canonical definitions and conventions** for each new symbol so the library matches the field, and (b) **identify the analytic oracles** the §6 benchmarks must reproduce. Every new convention definition must be traceable to a cited source.

**Seed references by topic** *(confirm exact citations and years in session; expand as needed).*
- *Quantum Fisher information & estimation* — Helstrom (quantum detection & estimation theory); Braunstein & Caves (statistical distance, ~1994); Paris ("Quantum estimation for quantum technology", ~2009); Tóth & Apellaniz (quantum-metrology review, ~2014).
- *Heisenberg-limit / GHZ metrology* — Giovannetti, Lloyd & Maccone (quantum-enhanced measurement); Bollinger, Itano, Wineland & Heinzen (maximally correlated states for frequency standards, ~1996).
- *Quantum Darwinism & redundancy* — Zurek (Nature Physics, ~2009); Ollivier, Poulin & Zurek (objectivity from subjective states, ~2004–05); Blume-Kohout & Zurek; Riedel, Zurek & Zwolak (spin environments).
- *Recoverability / QEC information theory* — Knill & Laflamme (theory of QEC, ~1997); Schumacher & Nielsen (coherent information, ~1996); Petz (recovery map / sufficiency); Fawzi & Renner (conditional mutual information and approximate recovery, ~2015); Bény & Oreshkov (approximate QEC, ~2010); Wilde (quantum information textbook) as a consolidated reference.
- *Common-mode / correlated noise* — correlated-dephasing and common-mode-rejection treatments in precision comparison/clock spectroscopy (relevant downstream to the CACE correlation-access constraint, but cited here only for the generic channel definition).
- *Numerics* — Johansson, Nation & Nori (QuTiP), since the package is QuTiP-based.

**Deliverable.** A short **literature-review note** committed under `docs/` (e.g., `docs/estimation-darwinism-review.md`), from which the new CONVENTIONS sections cite. The note records the chosen definitions (SLD-QFI convention, R_δ deficit convention, recoverability measure), each anchored to a reference.

---

## 6. Generic benchmarking (no TMC content)

Each new feature gets a **generic benchmark** following the repository harness: a `tools/run_benchmark_<name>.py` (or `run_demo_<name>.py`) writing `benchmarks/data/<name>/` with the canonical `{manifest.json, report.json, arrays.npz, plot.png}` and the existing provenance schema (`request_hash`, `convention_version`, `backend_name`/`version`, `provenance_tags`). Each benchmark validates against an **analytic oracle** within a stated **tolerance**. None contains application context.

| Benchmark | Generic setup | Analytic oracle | Tolerance |
|---|---|---|---|
| **QFI scaling** | Phase estimation, GHZ vs product/coherent-spin state, vs qubit number N | QFI_GHZ = N² (Heisenberg); QFI_product = N (standard quantum limit) | scaling exponent + closed-form values within tol |
| **CFI / linear-Gaussian** | Linear-Gaussian model with known A, Σ; single-qubit phase readout | F = AᵀΣ⁻¹A; CRB = F⁻¹; CFI ≤ QFI (Braunstein–Caves) | exact to numerical precision |
| **Darwinism redundancy** | Canonical system-qubit + N environment-qubit decoherence model | Mutual-information **plateau** ≈ H_S; classical-plateau R_δ shape | plateau height & onset within tol |
| **Recoverability** | Erasure / dephasing channel of known strength | perfect recovery → residual = full; full decoherence → residual = 0; monotone between | endpoints exact; monotonicity holds |
| **GHZ / cat factory** | Parity and entanglement of the produced state | GHZ parity oscillates at N·φ; entanglement (log-negativity / EoF via `entanglement`) as expected | matches analytic |
| **Common-mode channel** | Two-subsystem correlated dephasing; sweep correlation 0 → 1 | corr = 0 reduces to independent `PhaseDrift`; corr = 1 cancels in the difference observable (common-mode rejection) | limiting cases exact |

**Note.** The QFI-scaling benchmark (GHZ N² vs product N) is the single most informative check — it exercises the QFI implementation, the GHZ factory, and the SQL/Heisenberg distinction in one figure, with no application framing.

---

## 7. Conventions and constraints (repository discipline)

- **No-TMC invariant** (§1) — the gating constraint; the generic benchmarks are its proof.
- **Convention Freeze** — `CONVENTIONS.md` is frozen at v0.2 (§1–18). The new estimation and darwinism conventions land as **new sections under a freeze gate** (drafted early, frozen at release), with a `convention_version` bump. No post-freeze edits without a further bump.
- **SPDX headers** on every new file (`tools/check_spdx.py` is a CI gate).
- **Three-tier regression harness** — unit tests; analytic-regression tests (the closed-form oracles); reproduction tests with seed-stamped reference fixtures.
- **Typing & lint** — `py.typed` package; `ruff` + `mypy` clean; Python ≥3.11; QuTiP ≥5 / NumPy / SciPy (already core deps).
- **Docs** — mkdocs pages for the new modules + the literature-review note; docs site passes the existing WCAG Level A CI gate.
- **CHANGELOG** — entries under a new `[Unreleased]` block; a release tag at completion.
- **Licence** — additions inherit the MIT package licence and the split-licence repo architecture.

---

## 8. Definition of done

1. WI-1…WI-4 implemented, typed, documented, SPDX-headed; `ruff`/`mypy`/tests green in CI.
2. The **literature-review note** is committed and every new convention definition cites it.
3. Each feature has a **generic benchmark** reproducing its analytic oracle within tolerance, with canonical `benchmarks/data/` artifacts.
4. New **CONVENTIONS sections frozen**; `convention_version` bumped; CHANGELOG updated.
5. **Demos/benchmarks contain zero TMC content** — the decoupling proof.
6. A **release is tagged**; the tag is the handoff artifact the application will pin.

**Review-cycle termination.** One internal review of the upgrade before the release tag. Re-opening is warranted only if a benchmark fails to match its oracle.

---

## 9. Dependencies and handoff

- **Branch** off `iontrap-dynamics` `main`; this work is orthogonal to the package's own Phase 2 (JAX) and must not block the planned `open-iontrap` org migration.
- **No external dependency** beyond the existing toolchain; QuTiP/NumPy/SciPy suffice.
- **Handoff out:** the tagged release. The TMC application repo (`broadcast-protection`, provisional) will pin `iontrap-dynamics>=<this release>` and consume the new primitives in Stream A.
- **Return point:** once released, resume the TMC programme (Work Programme TMC-WP-0 v0.3, Stream A) — the arms, discriminants, and falsifier are built there, on top of these primitives.

---

*Task Card TC-ITD-ESTDARW-01 v0.1. Library-only, application-agnostic; literature-grounded and benchmark-validated. For execution in a dedicated breakout session, then return to TMC-WP-0 v0.3.*
