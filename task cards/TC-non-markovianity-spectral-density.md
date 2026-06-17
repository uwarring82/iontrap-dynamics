# Task Card — Spectral density & a non-Markovianity measure (Wittemer 2018) + tutorials

**Authored from:** the `iontrap-dynamics` side, as a scoping / build deliberation
**Topic:** add (1) a **spectral-density** concept `J(ω)` and a tutorial showing how the
**effective spin–mode coupling changes when decoherence is added**; and (2) a **non-Markovianity
measure** (BLP trace-distance) with the **quantum-projection-noise (QPN) bias** and a tutorial
reproducing **Wittemer et al., PRA 97, 020102(R) (2018)** — a paper co-authored by the maintainer.
**Rooting sources (staged, untracked):** `sources/pdf/PhysRevA.97.020102.pdf` (target);
`sources/pdf/BLP-2009-…-PRL103-210401.pdf` (the measure); `sources/pdf/Porras-2008-…-trapped-ions.pdf`
(spin-boson + `J(ω)`); `sources/pdf/Clos-Breuer-2012-memory-effects-spin-boson.pdf` (PRA **86**, 012115,
the continuum precursor). **PENDING:** the PRA 97 020102 **Supplemental Material** (ref [47]) is APS-gated
— needs a maintainer institutional download into `sources/pdf/` (tomography + simulation methods).
**Status:** v0.1 deliberation record. **Not endorsed; no WP ratified.** Frames the build-vs-scope
question before lifting into a WP.

---

## 1. Verdict

**Strong fit, mostly additive — but split it into two regimes.** `iontrap-dynamics` already holds
nearly all the machinery for the paper's **single-mode** realization; the **continuum spectral-density**
generalization is a genuinely new (and frozen-boundary-touching) capability. Recommended staging:

- **Phase A — single-mode, ship now, observable-only.** The PRA 97 020102 non-Markovianity comes from
  *coherent* coupling of a spin to **one** motional mode: evolve the closed `S+E` system (optionally with
  a small spin `Γ_dec`) and `ptrace` the mode → `D(t)` → `𝒩` → the **QPN bias** `ℬ`. This needs only a
  **trace-distance observable**, a **BLP-𝒩 accumulator**, a **QPN-bias estimator**
  (over the existing finite-shot layer, extended to spin-state tomography), and a
  **spin decoherence term**. The observables `D(t)`, `𝒩`, `ℬ` need no new convention
  symbol — the MCF probe-QFI precedent (benchmark-only, cite the frozen section,
  edit nothing). To keep **Phase A convention-free**, inject `Γ_dec` as benchmark/tutorial-local
  `c_ops` (qutip collapse operators); promoting it to a *public* `SpinDephasing`/`SpinDecay` channel is a
  separate, convention-gated step (frozen §24 is motional-only — side-car → seal, never an in-place edit).
- **Phase B — spectral density `J(ω)` (deliberate, later).** A genuine structured/continuum bath
  (Porras 2008; Clos–Breuer 2012) introduces `J(ω)` — which has **real normalization ambiguity** (one-
  vs two-sided, the `2π`) and, if it implies a memory kernel, lies **outside frozen §24's Markovian GKSL
  form and outside the `mesolve` path**. Phase B is where a sealed convention (likely a new **§26**) and a
  solver decision (many-mode unitary vs qutip HEOM/Bloch-Redfield) are needed. (Any sealed `J(ω)`
  convention would take the next free section — §26 as of 2026-06-17, but not pre-claimed.)

This is **quantum open-system dynamics — squarely `iontrap-dynamics`**, not the classical `iontrap-structure`
sibling (which could, separately, *supply* the mode spectrum that builds `J(ω)`).

## 2. Ownership boundary

| Capability | Owner | Rationale |
|---|---|---|
| Trace distance `D(t)`, BLP `𝒩`, QPN bias `ℬ` (general observables) | **library** | information-theoretic observables, application-agnostic |
| Spin `Γ_dec` decoherence | **benchmark-local (Phase A); library-public later** | Phase A: local `c_ops`; a public `SpinDephasing`/`SpinDecay` is a new convention surface (frozen §24 is motional-only) |
| Spectral density `J(ω)` representation + trap-preset → `J(ω)` map | **library** | general spin-boson primitive |
| Effective-coupling `Ω′` helper (`√(Ω²+δω_z²)`) | **library (already exists)** | `analytic.generalized_rabi_frequency` — reuse, don't re-implement |
| The *choice* of "reproduce Wittemer 2018" benchmark + its parameters | **benchmark/tutorial** | programme-specific; not a library symbol |
| Non-Markovian (memory-kernel / structured-bath) **solver** | **deliberate (Phase B)** | beyond the frozen §24 Markovian `mesolve` path |

**Application-agnostic invariant** (lifted verbatim from WP-01/02/03): library symbols carry **no
consuming-application framing**; "Wittemer 2018" lives only in a benchmark tool + regression oracle, never
in a public symbol. Enforced by a decoupling grep.

## 3. What the paper needs — calculations to enable

System (Eq. 2): `H = (ℏω_z/2)σ_z + ℏω_E a†a + (ℏΩ/2)[σ⁺ e^{iη(a†+a)} + h.c.]` — one spin (S), one motional
mode (E), full-exponential spin-boson coupling.

1. **Trace distance** `D(t) = ½‖ρ_S^1(t) − ρ_S^2(t)‖₁` between the reduced spin states of two initial
   conditions (PRA: the fixed orthogonal pair `|↑⟩,|↓⟩`), each `⊗` a thermal mode state `n̄`. The PRA
   protocol reconstructs `ρ_S(t)` from `⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩` (spin-state tomography) [47]; the QPN
   enters through finite `r` per axis.
2. **BLP non-Markovianity** `𝒩`. The paper's discretized, **fixed-pair** form `𝒩 = Σ_t [D(t) − D(t−Δt)]_{>0}`
   (Eq. 1), with `γ = 1/Δt`, up to `t_max`. *Note the deliberation point* (§9): the **original BLP `𝒩`**
   (BLP 2009 Eq. 11/12; Clos–Breuer 2012 Eq. 3) is a **maximum over initial pairs** (a Monte-Carlo
   optimization, maximizer on the state-space boundary), which the paper deliberately simplifies to one pair.
3. **Bias** `ℬ ≡ 𝒩(γ,r) − 𝒩_true`, with `𝒩_true ≡ lim_{γ,r→∞} 𝒩` (zero noise **and** infinite sampling).
   Keep the two sources separate (do **not** conflate them): **(i) QPN bias** `ℬ_QPN = 𝒩(γ,r) − 𝒩(γ,∞)` —
   because `𝒩` sums only *positive* increments, the `1/√r` projection noise rectifies into a **positive**
   systematic bias that **vanishes as `r→∞`** at fixed `γ` (the *total* bias at `(γ₀,r₀)` is `ℬ/𝒩_true ≈ +17%`, QPN-dominated there; `|ℬ/𝒩_true|` up to 45% over the
   `n̄` sweep); **(ii) sampling bias** `ℬ_sampling = 𝒩(γ,∞) − 𝒩_true` — a finite `γ` misses fast features of
   `D(t)`, a **negative** bias for `γτ < 8` that does **not** vanish with `r`. Headline (precise form): at the
   paper's adequately-high `γ₀τ ≈ 15` the QPN part dominates and increasing `r` recovers `𝒩 → 𝒩_true`; at low
   `γ` a residual underestimation persists regardless of `r`.
4. **Local quantum probing.** (a) Vary `ω_z` near `ω_E`: resonance in `𝒩`, effective coupling
   `Ω′ ∝ √(Ω²+δω_z²)`, `D`-amplitudes `∝ Ω²/Ω′²`. (b) Vary thermal `n̄`: `𝒩(n̄)` slope flips with `t_max`
   (more Fock states → richer, harder-to-resolve `D(t)`).
5. **Residual decoherence hierarchy** `Γ_dec ≪ 1/t_max < Ω` (the paper keeps `Γ_dec` negligible; the
   *first* requested tutorial deliberately turns it **up**).

Parameter mapping to the existing builder: the `clos2016_spin_boson_hamiltonian` argument `detuning_rad_s` corresponds to `δω_z ≡ ω_z − ω_E`; single mode = `dimensionless_mode_frequencies=[1]`; `Ω′` is then `analytic.generalized_rabi_frequency(carrier_rabi_frequency=Ω, detuning_rad_s=δω_z)`. Note the builder's generator `e^{η(a−a†)}` is unitarily equivalent to the paper's `e^{iη(a†+a)}` (via `a → i a`), so it reproduces the same `D(t)`/`𝒩` — the thermal mode state is invariant under the phase-space rotation `e^{iθ a†a}`, so the *reduced* spin evolution is preserved, not merely the spectrum.

Parameters (quote in the benchmark): `ω_E/2π = 1.920(3) MHz`, `Ω/2π ≈ 100 kHz`, `η ≈ 0.32`, `τ = 2π/Ω`,
`t_max ∈ {2,5,9}τ`, `r₀ = 500`, `γ₀ ≈ 15 τ⁻¹`, `n̄ = 1.0(1)` (Fig 2, inherited by Fig 3) / `0.09(2)` (Fig 4a) / →1.4 (Fig 4b).

## 4. Capability map — reusable vs gaps

**Reusable (already in the package):**

| Ingredient | Existing API | Fit |
|---|---|---|
| Full-exponential spin-boson `H` (single mode = length-1 `dimensionless_mode_frequencies`) | `clos2016_spin_boson_hamiltonian` (`clos2016.py`) | adaptable |
| Full-LD sideband engine (arbitrary Δn) | `_full_ld_*` builders (`hamiltonians.py`) | adaptable |
| Thermal mode prep, Bloch spin prep | `clos2016_initial_state` / SPAM | direct |
| Motional decoherence (`Γ` on the mode) | `Heating`/`Dephasing`/`AmplitudeDamping` (`channels.py`); routed via `solve(channels=…)` (`sequences.py`) | direct |
| **Finite-shot QPN** (the paper's `r`-repetition noise) | `SpinReadout(shots=r)` — Bernoulli/Poisson sampling **and** the `r→∞` `ideal_outcome` envelope; Wilson/CP CIs (`measurement/`). Captures `σ_z`-basis QPN; `σ_x`/`σ_y` tomography needs final rotation pulses (or a tomography wrapper). | **direct, with tomography note** |
| Trajectory-observable + `ptrace`-to-subsystem pattern | `information/fisher.py` (trajectory shape); `redundancy.py`/`recoverability.py` + `_common.py` (`ptrace`, dim validation) | direct (template) |
| Effective coupling `Ω′ = √(Ω²+δ²)` | `analytic.generalized_rabi_frequency` (not yet re-exported at package top level — one-line public-surface add) | **already exists** |
| Clos-2016 reproduction (sibling paper, ref [37]) | `clos2016.py` + references + tests | spectral/IPR only — **no** time-domain `D(t)` yet |

**Gaps (new code):**

1. **Trace-distance observable** `D(t)` → new `information/distinguishability.py` (mirrors the QFI trajectory signature; `ptrace` to the spin subsystem). 
2. **BLP-`𝒩` accumulator** (thin post-processor over `D(t)`; optional Monte-Carlo pair-optimization for the full BLP form) → same module.
3. **QPN-bias estimator** — sweep `r`, simulate finite-shot spin-state tomography (`x`/`y`/`z`) or propagate the sampling covariance through the Bloch-vector `→ D(t) → 𝒩` pipeline, report `ℬ` + CI vs the `r→∞` envelope → `measurement/bias.py` (consumes `ideal_outcome` vs sampled).
4. **Spin decoherence** for scalar `Γ_dec` (today `channels.py` is **mode-keyed only**). *Phase A:* pass `Γ_dec` as benchmark-local `c_ops` (T2 `σ_z` dephasing, T1 `σ₋` decay) — no public symbol, no convention. *Promotion:* a public `SpinDephasing`/`SpinDecay` channel + widened union routed through `solve(channels=…)` is convention-gated (side-car section; frozen §24 is motional-only).
5. **Spectral density `J(ω)`** representation (Phase B) — `J(ω)=π Σ_n|λ_n|²δ(ω−ω_n)` and its continuum power-law limits → new module (e.g. `spectral_density.py`), **kept separate** from the eigenvalue-oriented `spectrum.py`.

## 5. The central scope fork — single mode vs spectral density

- **Single mode (Phase A) is fully solvable today.** The "environment" is one coherently-coupled mode;
  information backflow is recovered by evolving `S+E` (closed, or with spin `Γ_dec`) and tracing out the
  mode — `mesolve` + `qutip.ptrace` + the new `D(t)` observable. **No memory kernel, no `J(ω)`, no new
  solver.** This reproduces the paper.
- **Continuum spectral density (Phase B) hits two frozen boundaries at once.** A structured `J(ω)`
  (Porras 2008: `J(ω) ∝ ω^s`, Ohmic `s=1`, tunable by trap geometry; mesoscopic revival time `τ_rev=2π/δω`)
  needs either **(i)** explicit many-mode unitary evolution (Hilbert-space blow-up) or **(ii)** a genuine
  non-Markovian solver. The current `solve()` path is `mesolve` (GKSL/**Markovian**) and **cannot carry a
  memory kernel**; qutip's **HEOM / Bloch–Redfield** solvers can, but adding one is a backend decision.
  Clos–Breuer 2012 took the **TCL2** route (no Markov/RWA) — another option. This is the deliberate part.

## 6. Conventions & governance implications

- **`D(t)`, `𝒩`, `ℬ` → benchmark/observable-only, no new symbol** (MCF probe-QFI precedent: ship a
  compute-only benchmark tool + a regression-analytic oracle against a closed form; **cite** the frozen
  section, edit nothing). Trace distance is a standard functional; `𝒩` is a fixed functional of `D(t)`.
  **Caveat for `ℬ`:** observable-only *iff* it merely composes the §17 shot axis + §17.12 CIs; a *named
  debiasing formula* would be a §17.x convention needing a `0.4→0.5` bump.
- **Spin channels are a convention surface too.** `SpinDephasing`/`SpinDecay` are standard Pauli
  channels, but they are not motional and therefore not an in-place edit of frozen §24. They need a
  small new convention section or sub-section (side-car → seal) and must route through
  `solve(channels=…)` without perturbing the existing motional path.
- **`J(ω)` is the convention-sensitive one.** Observable/input-only if it merely *supplies* §24 Lindblad
  rates (schematically `rate ~ J(ω_m)`, `n̄` from `J` — the exact `2π` / one- vs two-sided normalization is
  itself the ambiguity to seal). But a **public, downstream-pinned `J(ω)` API** commits to a
  normalization/sign/one-vs-two-sided convention (real cross-textbook ambiguity — the class §6 squeezing
  and §10 Lamb–Dicke were written to nail) → a **new sealed section** (next free integer, §26 as of
  2026-06-17), not an in-place §24 edit. A
  memory-kernel/coloured bath is also **outside frozen §24** (Markovian GKSL).
- **Frozen-section hazard.** §17/§24 are frozen; **no in-place edits** (the §18.4 doc-mention was *declined*
  on exactly this rule). Route any genuine new convention via the proven RLA/§25 path: review note →
  side-car proposal (propose-don't-apply) → green conventions-test → **maintainer seals** the bump.
- **§13/§15 hygiene.** `D(t)` from solved states inherits Fock-truncation status (§13) and must classify
  into the §15 ladder — unphysical/non-finite `D` must **raise**, not NaN-propagate (the gap RLE closed).
- **Governance.** New **WP-04** (one-card→one-WP); fresh dispatch family minted at ratification after the
  five-source collision grep (candidate mnemonics e.g. `ND` non-markov/distance, `SD` spectral density).
  Current `CONVENTION_VERSION = "0.4"`.

## 7. Proposed tutorials

- **Tutorial A — "Effective spin–mode coupling & spectral density under decoherence."** Build the single-mode
  spin-boson `H`; show the effective coupling `Ω′ = √(Ω²+δω_z²)` and the resonance in information backflow
  vs detuning; then **turn up the spin `Γ_dec`** (new channel) and show `D(t)`/`𝒩` and the apparent coupling
  degrade as decoherence is added (the user's stated goal). Introduce `J(ω)` conceptually: single mode = a
  `δ`-peak; sketch the structured-bath generalization (Phase B) and `τ_rev`.
- **Tutorial B — "A non-Markovianity measure and its measurement limit (Wittemer 2018)."** Reproduce `D(t)`,
  `𝒩`, and the **bias** `ℬ(γ,r)`; show the local-probing resonance `𝒩(ω_z)` and the `𝒩(n̄)` slope flip;
  make the headline point: more repetitions kill the **QPN bias** (`ℬ_QPN→0`), but a finite-`γ`
  **sampling bias** remains — faithful `𝒩_true` needs both `r→∞` *and* `γ→∞`.

## 8. Proposed first slice (Phase A) — WI sketch

1. `information/distinguishability.py`: `trace_distance_trajectory(...)` + `blp_measure(D, times)` (+ optional
   pair-optimization), validated against **closed forms**: damped-JC `σ=−γ(t)e^{−Γ(t)}` (BLP 2009 Eq. 13) and
   the central-spin `D=√(a²+f²|b|²)` (Eq. 14).
2. Spin `Γ_dec` as benchmark-local `c_ops` (T2 `σ_z` dephasing and/or T1 `σ₋` decay); acceptance: analytic coherence/population decay — pin the `σ_z` pure-dephasing rate normalization (`1/T2` vs qutip's √-rate convention) against that oracle. (Public `SpinDephasing`/`SpinDecay` channel deferred — convention-gated.)
3. `measurement/bias.py`: bias estimator over `(γ, r)` (accounting for `x`/`y`/`z` tomographic sampling);
   acceptance: `ℬ_QPN → 0` as `r→∞` at fixed `γ` (positive sign); a finite-`γ` **sampling** bias persists
   for low `γ` (negative sign) — both limits/signs matching Fig 3.
4. **Benchmark + oracle**: reproduce Wittemer 2018 Fig 2/3 qualitative features (single-mode, fixed pair),
   compute-only (`report.json`+`arrays.npz`+`plot.png`), plus a regression test on `D(t)` analytic limits.
5. Tutorials A & B (print+plot + Colab, per the existing tutorial-notebook track).

## 9. Open questions (for deliberation)

1. **Fixed-pair vs full BLP `𝒩`** — implement the paper's fixed orthogonal pair, the full sup-over-pairs
   (Monte-Carlo), or both (the latter is the honest general measure; the former reproduces the paper)?
2. **`ℬ` scope** — pure observable (compose §17 shots + tomography) vs a named debiasing convention
   (→ §17.x bump)?
3. **Spin-state tomography in the finite-shot layer** — add a `SpinTomographyReadout` wrapper that performs
   `x`/`y`/`z` basis readouts via final rotations, or keep it analytic (sampling covariance on the Bloch
   vector)?
4. **Phase B trigger** — do we commit to `J(ω)` + a non-Markovian solver now, or defer until a concrete
   structured-bath use case (and decide many-mode-unitary vs HEOM/Bloch-Redfield vs TCL2)?
5. **`J(ω)` convention** — if/when public, seal a new section (next free integer, §26 as of 2026-06-17): normalization, one- vs two-sided, `2π`.
6. **Sibling cross-link** — should `iontrap-structure`'s mode spectrum *feed* `J(ω)` (it produces the
   `λ_n, ω_n`), making this the first real consumer of the classical sibling (one-way data feed, not a
   hard dependency)?
7. **Spin-channel promotion** — keep `Γ_dec` as benchmark-local `c_ops` (Phase A), or promote to a public
   `SpinDephasing`/`SpinDecay` channel now (which triggers a convention side-car, frozen §24 being
   motional-only)?

## 10. Rooting sources

| Role | Source | Local |
|---|---|---|
| Target (the measure + QPN bias + probing) | Wittemer, Clos, Breuer, Warring, Schaetz, **PRA 97, 020102(R) (2018)** | ✓ `sources/pdf/` |
| The measure (`D`, `σ`, `𝒩`, divisibility) | **Breuer, Laine, Piilo, PRL 103, 210401 (2009)** | ✓ staged |
| Spin-boson + `J(ω)` from ion modes | **Porras, Marquardt, von Delft, Cirac, PRA 78, 010101(R) (2008)** | ✓ staged |
| Continuum precursor (`𝒩` in spin-boson, `J_eff`, resonance) | **Clos, Breuer, PRA 86, 012115 (2012)** | ✓ staged |
| Methods (tomography, simulation) | PRA 97 020102 **Supplemental Material** (ref [47]) | ✗ **APS-gated — pending maintainer download** |
| Already reproduced (sibling) | Clos, Porras, Warring, Schaetz, PRL 117, 170401 (2016) — `clos2016.py` | ✓ in `src/` |

Deeper lineage available locally (Mendeley) for the implementation phase: Breuer 2016 RMP review;
Rivas-Huelga-Plenio 2010 / Chruściński-Maniscalco 2014 (other measures); Lemmer 2018, Wilhelm-Kleff-von Delft
2004, Nevado-Porras 2013 (structured-environment `J(ω)`); Borrelli 2013 / Borrelli-Maniscalco 2014
(Coulomb-crystal `J(ω)` + non-Markovianity); Haikka-McEndoo-Maniscalco 2013 (local probing); Clos 2017 (thesis).

## 11. References

- M. Wittemer, G. Clos, H.-P. Breuer, U. Warring, T. Schaetz, *Measurement of quantum memory effects and its fundamental limitations*, **PRA 97, 020102(R) (2018)**.
- H.-P. Breuer, E.-M. Laine, J. Piilo, *Measure for the degree of non-Markovian behavior…*, **PRL 103, 210401 (2009)**.
- D. Porras, F. Marquardt, J. von Delft, J. I. Cirac, *Mesoscopic spin-boson models of trapped ions*, **PRA 78, 010101(R) (2008)**.
- G. Clos, H.-P. Breuer, *Quantification of memory effects in the spin-boson model*, **PRA 86, 012115 (2012)**.
- G. Clos, D. Porras, U. Warring, T. Schaetz, *Time-Resolved Observation of Thermalization in an Isolated Quantum System*, **PRL 117, 170401 (2016)** (already reproduced).
