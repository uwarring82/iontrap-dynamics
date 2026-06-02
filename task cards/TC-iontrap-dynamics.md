# Task Card — `iontrap-dynamics` as a service module for *Undetected Modes*

**Authored from:** the `undetected-modes` side, as a dependency / integration spec
**Upstream target:** `uwarring82/iontrap-dynamics` @ v0.4.0 (`CONVENTIONS.md` frozen v0.2)
**Serves:** WP1 (r5) — esp. T1.4 numerical components, D1.4b, and the MS1b theory/numerics gate
**Status:** v0.1 draft. **Not endorsed.** Intended to be mirrored as upstream issues once agreed. Authority derives from use (Lock–Key Rule).

---

## 1. Verdict (fit assessment)

`iontrap-dynamics` is a **good fit** as a service module. It is mature, typed, well-tested (three-layer regression harness, mypy-strict CI), multi-species and η-parametrised (matching the WP's platform-agnostic stance), and it already reproduces Clos 2016. The gaps are **targeted and within the library's declared scope** — the workplan already lists *two-mode* squeezed states as in scope. No gap requires redefining the library's coastline; the work is additive capability plus one conventions amendment.

**One boundary must hold:** the library owns the *forward model and general primitives*; `undetected-modes` owns the *analogy, channel choices, native-subtraction, resource-constrained benchmark, and claim discipline* (§2). Keeping this line is what preserves the library's reusability.

**Non-apparatus:** every item below is simulation/library work, so it is **unblocked under the r5 lock** and can proceed in parallel with T1.0 and T1.3a.

## 2. Service-module boundary (ownership)

| Capability | Owner | Rationale |
|---|---|---|
| Two-mode parametric (SU(1,1)) Hamiltonian, beamsplitter | **library** | General spin-motion / motional physics |
| Two-mode squeezed-vacuum state prep | **library** | Already in declared scope (single + two-mode) |
| Typed motional CPTP channels + `c_ops` exposure | **library** | General open-system capability |
| Interferometric observables (visibility, fringe phase) | **library** | General readout observables |
| Lamb–Dicke regime helpers (Debye–Waller, classifier) | **library** | General to the spin-motion interface |
| General QFI primitive (state / channel) | **library (optional)** | Reusable; see F6 boundary note |
| **The analogy map** (what corresponds to what) | undetected-modes | Programme-specific (CONVENTIONS §1 there) |
| **Choice of the engineered channel ε** | undetected-modes | Programme-specific |
| **Native-subtraction / composition logic** ε_total | undetected-modes | Programme-specific; R8 boundary |
| **Resource-constrained SU(1,1)-vs-coherent benchmark** | undetected-modes | Programme-specific (T1.4) |
| **Identifiability vs native drift + technical noise** | undetected-modes | Programme-specific (MS1b) |
| **Claim-calibration register** | undetected-modes | Programme-specific (T1.6) |

## 3. Already present — do not rebuild

Multi-mode `HilbertSpace` (modes *a*, *b* coexist; per-mode Fock truncation) · single-mode `coherent_mode` / `squeezed_vacuum_mode` / `squeezed_coherent_mode` · `carrier_hamiltonian_full_ld` and red/blue sideband builders · `analytic.lamb_dicke_parameter`, leading-order sideband Rabi (∝ η√n) · finite-shot measurement layer (`SpinReadout`, Wilson CI) · classical readout channels (Bernoulli/Binomial/Poisson) · systematics: **timing/phase jitter** and **frequency drift** (directly useful for the MS1b identifiability check and the §1A η-drift point) · SPAM (thermal-prep, readout) · QuTiP + JAX/Dynamiqs backends.

## 4. Required feature updates

Each card follows the library's own rule — *conventions before code* — so a conventions entry is part of every acceptance set.

### F1 — Two-mode parametric (SU(1,1)) Hamiltonian, with optional beamsplitter
- **Serves:** the analogue interferometer (CONVENTIONS §1, structural row); T1.4 forward model.
- **Add:** a builder for the two-mode squeezing generator on a labelled mode pair,
  `H_TMS(g, φ) = iℏg( e^{iφ} â†b̂† − e^{−iφ} â b̂ )`, plus (optional, secondary) the SU(2) beamsplitter `H_BS(J, φ) = ℏJ( e^{iφ} â†b̂ + h.c. )`.
- **Interface (sketch):** `two_mode_squeezing_hamiltonian(system, mode_labels=("a","b"), g, phase=0.0, backend="qutip")` returning the same Qobj/list surface as existing builders.
- **Acceptance:** generated two-mode squeezed vacuum has per-mode n̄ = sinh²(gτ); su(1,1) Casimir invariant test; cross-backend agreement < 1e-3; regression entry.
- **Conventions:** phase/sign/ordering on the two-mode subspace → **CONVENTIONS v0.3 freeze gate**.
- **Priority:** P0. **Owner:** library.

### F2 — Two-mode squeezed-vacuum state factory
- **Serves:** SU(1,1) probe preparation; QFI benchmark (F6).
- **Add:** `two_mode_squeezed_vacuum(fock_dims, z)` (complements the existing single-mode factories already shipped).
- **Acceptance:** matches analytic Schmidt structure / per-mode n̄ = sinh²|z|; invariant (trace, positivity).
- **Conventions:** squeezing-parameter convention shared with F1.
- **Priority:** P0. **Owner:** library.

### F3 — Typed motional CPTP channels + `c_ops` exposure *(pivotal)*
- **Serves:** the engineered channel ε on mode *b* (CONVENTIONS §1 central row); the native channel (T1.3); R8.
- **Add:** a `channels` module of typed motional dissipators — amplitude damping (zero-T, `â` at rate κ), **thermal amplitude / heating** (`â`, `â†` with bath n̄ — this *is* the native anomalous-heating model), pure dephasing (`n̂` at rate γ_φ), depolarising — each yielding either QuTiP collapse operators or Kraus maps. **Expose them in `solve()`** (today `c_ops=[]` is hardwired): add a typed `channels=`/`dissipators=` argument routed to the existing `mesolve` path.
- **Sequence-aware application:** allow a channel to act on a chosen *segment* of a pulse sequence, so ε_native and ε_engineered compose **according to the actual sequence** — i.e. the library must **not** silently assume the two commute (this is exactly the R8 boundary the WP refuses to assume away).
- **Interface (sketch):** `AmplitudeDamping(mode="b", rate=κ)`, `Heating(mode="b", n_bar_bath=…, rate=…)`, `Dephasing(mode="b", rate=γ_φ)`; `solve(..., channels=[…])`; segment binding via the sequence API.
- **Acceptance:** amplitude damping reproduces analytic ⟨n̂⟩ decay; dephasing reproduces coherence decay; heating reaches n̄_bath at steady state; CPTP/trace-preservation invariants; regression entries; an explicit test that two non-commuting channels do **not** give order-independent results (guards R8).
- **Conventions:** channel parameterisation (rates, bath occupation, Kraus vs Lindblad) → **CONVENTIONS v0.3 freeze gate**.
- **Priority:** P0 (the enabling change). **Owner:** library.

### F4 — Interferometric observables: visibility and fringe phase
- **Serves:** V(θ), Δφ(θ) (CONVENTIONS §3); D1.5 worked example.
- **Add:** observable builders that, over a phase/parameter scan, return fringe visibility `V = (P_max − P_min)/(P_max + P_min)` and the fitted phase shift Δφ from the spin-readout signal, layered on the existing `observables` + measurement surface.
- **Acceptance:** ideal SU(1,1) sequence → known V; adding an F3 channel degrades V as predicted by the closed form; definitions match `undetected-modes/CONVENTIONS §3`.
- **Conventions:** visibility/phase definitions → additive under v0.2 (new `Observable` records), **no schema break**.
- **Priority:** P1. **Owner:** library.

### F5 — Lamb–Dicke regime helpers
- **Serves:** D1.4b η-regime performance map; the §1A regime boundary and its dynamic validity.
- **Add:** a Debye–Waller factor helper (∝ e^{−η²(2n̄+1)/2}), a **regime classifier** returning deep / intermediate / beyond from (η, n̄) via η²(2n̄+1), and an intermediate-order correction option on the existing leading-order sideband-Rabi analytic.
- **Acceptance:** Debye–Waller matches series expansion; classifier thresholds documented and tested; continuity with the full-LD carrier numerics already shipped.
- **Conventions:** additive under v0.2 (new `analytic` functions), **no schema break**.
- **Priority:** P1. **Owner:** library.

### F6 — General QFI primitive *(optional in the library — boundary call)*
- **Serves:** T1.4 QFI-vs-benchmark.
- **Add:** a `fisher` module — state QFI via the symmetric logarithmic derivative for a parametrised family ρ(θ), and a channel-QFI helper through the output state.
- **Boundary note:** QFI is a *general* quantity and is reasonable to host in the library. But the **resource-constrained benchmark** (fixed mean probe excitation and interrogation time) and the **identifiability check against native drift / technical phase noise** are programme-specific and **stay in `undetected-modes`**. If the library prefers to remain strictly forward-model, host QFI in `undetected-modes` instead — **decide before implementing** (§7).
- **Acceptance:** QFI of a coherent/squeezed probe matches known closed forms; saturates the classical Fisher information for the optimal measurement on a test case.
- **Conventions:** additive (new module), **no schema break**.
- **Priority:** P1 (or relocated). **Owner:** library *or* undetected-modes (open).

### F7 — Identifiability support (mostly assembly, not new physics)
- **Serves:** MS1b model-discrimination check.
- **Note:** the technical-noise ingredients largely **already exist** — phase/timing **jitter** and frequency **drift** systematics, plus the F3 channels. F7 is the *programme-side* assembly in `undetected-modes` that varies engineered strength (≥2 points, per the T1.5 minimum tier) and tests whether the inferred native contribution moves. **Confirmed library gap:** `drift.py` drifts *drive* parameters (`RabiDrift`, `DetuningDrift`, `PhaseDrift`) but carries **no motional mode-frequency drift**; since η depends on ω_mode, the §1A η-drift point needs a small new `ModeFrequencyDrift` systematic acting on `ModeConfig`. This is the one genuine library addition under F7.
- **Priority:** P2. **Owner:** undetected-modes (with a possible small library check).

## 5. Sequencing against the WP gates

- **Before MS1b / for T1.4:** F1, F2, F3, and F6 (QFI) are the critical set — they constitute the forward model and the metrological-advantage calculation. **F3 first** (it unblocks everything dissipative).
- **For D1.4b:** F5 (regime helpers).
- **For D1.5 / T1.5:** F4 (observables) and the F3 sequence-aware composition (R8 at ≥2 strengths).
- **For MS1b identifiability:** F7 assembly (consumes F3 + existing jitter/drift).

A natural first upstream milestone: **F3 + F1 + F2 on one branch**, with the v0.3 conventions freeze covering both — that is the minimum that lets `undetected-modes` build a first numerical forward model for one Markovian channel.

## 6. Conventions & versioning

The library freezes conventions at version milestones and forbids code ahead of documented conventions. Mapping:

- **Additive under the frozen v0.2 (callers unaffected, like the v0.4 additions):** F4 (observables), F5 (regime helpers), F6 (QFI module), F2 (state factory, sharing F1's squeezing convention).
- **Requires a documented CONVENTIONS v0.3 freeze gate:** F1 (two-mode squeezing phase/sign/ordering) and F3 (channel parameterisation; sequence-aware application). These define *new physical conventions*, so they must be written before the code.

Licence: the library is code (split-licence, MIT track); `undetected-modes` consumes it as a dependency. No conflict with the CC BY 4.0 docs licence.

## 7. Open questions for U

1. **F6 boundary:** QFI in the library, or in `undetected-modes`? (Recommend: a *bare* QFI primitive in the library; the resource-constraint and identifiability stay programme-side.)
2. **Conventions cadence:** one combined v0.3 freeze covering F1+F3, or two smaller freezes? (Recommend: one combined freeze — fewer gates, one worked example.)
3. **Upstream vs fork:** open these as issues on `uwarring82/iontrap-dynamics`, or stage on a branch first? (Affects how the v0.3 freeze is governed.)
4. **Resolved:** `drift.py` does **not** cover motional mode-frequency drift (it drifts drive parameters only), so a small `ModeFrequencyDrift` systematic is needed for the η-drift point (folded into F7).

---

## Provenance

Subordinate to WP1 (r5) and the parent concept note. Reflects `iontrap-dynamics` @ v0.4.0 as inspected; verify against the repository state of record before opening issues. This card adds structure and a boundary, not claim authority.
