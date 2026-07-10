# Task Card — Non-adiabatic squeezing dynamics: time-dependent trap frequency, phonon-pair creation & motional entanglement (Wittemer 2019/2020) + tutorials

**Authored from:** the `iontrap-dynamics` side, as a scoping / build deliberation.
**Topic:** add the engine parts to build on **Wittemer et al., PRL 123, 180502 (2019)** ("Phonon Pair Creation
by Inflating Quantum Fluctuations in an Ion Trap") and **Wittemer et al., Phil. Trans. R. Soc. A 378, 20190230
(2020)** ("Trapped-ion toolkit for studies of quantum harmonic oscillators under extreme conditions") — both
co-authored by the maintainer. The physics core is **a single (or two-ion) quantum harmonic oscillator with a
time-dependent trap frequency `ω(t)` that generates squeezing non-adiabatically**: (1) a **centred
time-dependent-frequency squeezing Hamiltonian** (quench + parametric-modulation regimes); (2) **phase-space /
Gaussian readout** — quadratures, the full 2×2 covariance matrix, Wigner function, squeezing parameter `r`,
symplectic eigenvalue `ν`, coherent displacement `α`, phonon-number diagonals `Pₙ`; (3, optional) a **forced
displacement** term + echo/two-pulse purification; and (4, deferred) **two-ion motional entanglement** (Gaussian
entanglement-of-formation across the **ion**-cut, reduced-mode entropy → effective Hawking temperature) — the
cosmological-particle-creation / Hawking analog.
**Rooting sources (staged, untracked):** `sources/pdf/Wittemer-Phonon Pair Creation…-2019-Physical Review Letters.pdf`
(PRL 123 180502, target B); `sources/pdf/Wittemer-Trapped-ion toolkit…-2020-Philosophical Transactions…pdf`
(Phil Trans R Soc A 378 20190230, target A — carries the explicit Hamiltonian Eq. 2.1). **PENDING:** the PRL 123
180502 **Supplemental Material** (ref [10]) is APS-gated — needs a maintainer institutional download into
`sources/pdf/` (numerical-simulation method, the `Pₙ^par(r,α,n̄_th)` parametrisation, the Gaussian entanglement
quantification). The 2020 **electronic supplementary material is open** on figshare
(doi:10.6084/m9.figshare.c.5011307) — **fetch now** for benchmark parameters. Methods lineage: **Silveri et al.,
Rep. Prog. Phys. 80, 056002 (2015)** ("Quantum systems under frequency modulation" — the Eq. 2.1 form, ref [21]);
**Burd et al., Science 364, 1163 (2019)** (parametric coupling `g`, ref [22]); **Fey, Schaetz, Schützhold, PRA 98,
033407 (2018)** (the two-ion analog theory, ref [19/49]).
**Status:** **v0.2.1 deliberation record — LIFTED to `WP/WP-05-nonadiabatic-squeezing.md` and SEALED
2026-07-10 (Phase A: A1 + A2; A3 optional tail; SQ dispatch family, §26 sealed, `CONVENTION_VERSION`
0.4 → 0.5). Ratified via WP-05.** Frames the build-vs-scope question behind the WP. Sibling to
`task cards/TC-non-markovianity-spectral-density.md` (the WP-04 / `ND` precedent this card structurally follows)
and to `task cards/TC-gaussian-entanglement-toolbox.md` (the reusable Gaussian toolbox that Phase B will consume).

**Revision log.** **v0.2.1 (2026-07-09)** — pins the last pre-WP interface issue: the A1 builder takes a
**`FrequencyWaveform` object** exposing both `ω(t)` and the **analytic** `d ln ω/dt` (+ JAX variants), not a bare
`omega_of_t` callable (a generic callable cannot supply its derivative safely across the sharp quench); paired
callables are the fallback (§4 gap 1, §8 WI-1, §9.4). Also links the sibling `TC-gaussian-entanglement-toolbox.md`.
**v0.2 (2026-07-09)** — folds in the maintainer's physics-consistency review. Substantive
corrections vs v0.1: (a) **Phase A split into three sub-slices** A1 (centred quadratic squeezing) / A2 (covariance
+ Wigner + direct `Pₙ` readout) / A3 (**optional** forced-displacement + echo); (b) **displacement normalisation
corrected** to `α = (⟨x̂⟩+i⟨p̂⟩)/2`, `|α| = ½√(⟨x̂⟩²+⟨p̂⟩²)` — v0.1 copied the paper's `√(x²+p²)` and was 2× high
(4× in `n_dsp`) against frozen §7; (c) **`r` readout moved from `tr V` to the covariance-eigenvalue form**
`r = ¼ ln(λ_max/λ_min)`, `ν = √(det V)` — `tr V` cannot separate squeezing from thermal width, which matters
because the card carries `n̄_th`; (d) **echo/`δp` cannot follow from the parity-preserving quadratic Hamiltonian**
(needs a linear force term) → moved to the optional A3; (e) **`Pₙ` disambiguated** into three distinct
capabilities, first slice ships direct `diag(ρ)` + the pure-squeezed-vacuum oracle only; (f) **oracles qualified**
— sudden limit via a narrowing smooth ramp / analytic squeeze kick (a literal step has a δ-function `d ln ω/dt`),
adiabatic `r→0` only for a **cyclic** waveform (or `r` relative to the instantaneous vacuum); (g) **Fock guard is
parity-blind** for even-dim squeezed vacuum → tail-window/cross-cutoff test (may need a §13 amendment); (h)
**Wigner is convention surface** (qutip's default vacuum variance ½ ≠ the paper's 1 — pin `g`); (i) **Phase B
mode-cut ≠ ion-cut** — the ion-A/ion-B entanglement needs a normal-mode→local-ion covariance transform, and `T_H`
needs a local Hamiltonian + thermal prescription; (j) architecture — **a dedicated phase-space/Gaussian module**,
not the linear-operator `Observable` record.

---

## 1. Verdict

**Strong fit, but genuinely new dynamics — phase it, single-ion first, and split Phase A.** `iontrap-dynamics`
already holds the *static* squeezing machinery (squeezed/squeezed-coherent state factories, the two-mode SU(1,1)
Hamiltonian, the sideband engine) and — critically — a **solver that already accepts a genuinely time-dependent
Hamiltonian** (`sequences.solve` takes `[[H, f(t)]]` lists and Dynamiqs `TimeQArray`,
[sequences.py:142](src/iontrap_dynamics/sequences.py#L142), [sequences.py:290](src/iontrap_dynamics/sequences.py#L290)).
What is **absent** is exactly the paper's engine: a **centred time-dependent-frequency squeezing generator**
(`H(t) = ℏω(t)(a†a+½) − (iℏ/4)(d ln ω/dt)(a†²−a²)`), a **phase-space / Gaussian readout** (covariance matrix,
Wigner, `r`, `ν`, `α`, `Pₙ` — as *state functionals*, not linear observables), and — only for the parasitic
displacement/echo — a **linear force term**. Recommended staging:

- **Phase A — single-ion non-adiabatic squeezing engine (Wittemer 2020 single-ion results), ship first, in three
  sub-slices.**
  - **A1 — centred quadratic squeezing.** The `ω(t)` squeezing Hamiltonian (Eq. 2.1), a **parity-preserving**
    quadratic generator that keeps `⟨â⟩ = 0` from vacuum. Two extreme regimes: **(i) quench** — a fast Gaussian
    pulse of `ω(t)` (width `δτ`, amplitude `Δω`, `ω̇/ω² ≈ 5`); **(ii) parametric modulation** — sinusoidal `ω(t)`
    at `ω_mod = 2ω_ini`, `n̄_sq = sinh²(2π g T_mod)`. The one convention-touching item (new **§26**).
  - **A2 — phase-space / Gaussian readout (observable-only).** Quadratures, the full 2×2 covariance matrix `V`,
    `ν = √(det V)`, `r = ¼ ln(λ_max/λ_min)`, `n̄_sq = sinh²r`, `α`, Wigner, and **direct** `Pₙ = ⟨n|ρ|n⟩`.
  - **A3 — forced displacement + echo (optional, Phase-A tail).** A time-dependent **linear force** term
    (`∝ â+â†`) to seed the parasitic coherent displacement the paper attributes to a static offset from the RF
    null, plus the two-pulse echo that cancels it (`δp` reproduction). *Only A3 can reproduce the displacement/echo
    results — A1 alone cannot.*

  A1+A2 already reproduce the **displacement-free** 2020 results — decisively the **parametric-modulation** arm
  (Fig 3, "all data consistent with `n_dsp = 0`") and the 2019 single-mode squeezing. The quench+echo arm (Fig 2b,
  `δp`) needs A3.
- **Phase B — two-ion motional entanglement & the Hawking/cosmology analog (Wittemer 2019 PRL, deliberate later).**
  Two ions' coupled radial modes (in-phase COM `ω₁`, out-of-phase rocking `ω₂`) under a shared `ω_rad(t)` ramp →
  **entanglement between the two ions' spatial d.o.f.** This needs: **two-ion normal modes** (the Coulomb matrix
  `M_ij` diagonalisation — owned by the classical sibling `iontrap-structure`, which already emits `ModeConfig`), a
  **Gaussian-state toolbox** (covariance matrix, a **normal-mode → local-ion coordinate transform**, Gaussian
  entanglement-of-formation `E_F` across the **ion**-cut, mode-A/mode-B log-negativity), and a
  **reduced-ion-state-entropy → effective (Hawking) temperature `T_H`** readout (which additionally needs a local
  Hamiltonian + a thermal-equivalent prescription). Heavier, cross-repo, and touches new convention surface —
  defer to its own card→WP (possibly a standalone "Gaussian entanglement toolbox" card first).

This is **closed-system unitary squeezing dynamics — squarely `iontrap-dynamics`**. The classical two-ion mode
spectrum (`M_ij`, COM/rocking eigenvectors) is the sibling `iontrap-structure`'s job; Phase B *consumes* its
`ModeConfig`, it does not re-derive it.

## 2. Ownership boundary

| Capability | Owner | Rationale |
|---|---|---|
| Centred `ω(t)` squeezing Hamiltonian `H(t) = ℏω(t)(a†a+½) − (iℏ/4)(d ln ω/dt)(a†²−a²)` | **library (A1)** | general single-mode parametric-drive primitive; convention-gated (new §26) |
| Time-dependent **linear force** term `∝ f(t)(â+â†)` (seeds coherent displacement) | **library (A3, optional)** | the only route to the paper's parasitic `α`; separate primitive from A1 |
| Quadratures `x̂=â+â†`, `p̂=i(â†−â)`; 2×2 covariance `V` (incl. `xp` term); `ν=√(det V)` | **library (A2)** | Gaussian state functionals; **not** the linear-operator `Observable` record → dedicated module |
| Readout `r=¼ln(λ_max/λ_min)`, `n̄_sq=sinh²r`, `α=(⟨x̂⟩+i⟨p̂⟩)/2`, `n_dsp=\|α\|²`, Wigner `W(x,p)` | **library (A2, observable-only)** | derived functionals; Wigner pins the normalisation (`g`) |
| Direct phonon-number diagonals `Pₙ = ⟨n\|ρ\|n⟩` | **library (A2)** | trivial state functional |
| Parametrised Gaussian-state `Pₙ^par(r,α,n̄_th)` (phase/ordering-dependent) | **deferred** (needs 2019 Supplemental) | named forward model; ship after the supplement pins phase/ordering |
| Sideband **inversion** of measured flopping → `Pₙ` | **deferred / separate** | the experimental thermometry inverse; distinct from `diag(ρ)` |
| Adiabaticity parameter `ω̇/ω²` | **library** | small general diagnostic |
| The **quench** and **parametric-modulation** `ω(t)` waveforms + the echo sequence | **benchmark/tutorial** | protocols composed from the primitives |
| The *choice* of "reproduce Wittemer 2019/2020" + cosmology/Hawking framing (`a²(t)`, e-foldings, WKB phase `φ`, `T_H`) | **benchmark/tutorial** | consuming-application framing — never a library symbol |
| Two-ion normal modes (Coulomb `M_ij`, COM/rocking eigenvectors) | **sibling `iontrap-structure`** | classical equilibrium+modes → `ModeConfig`; Phase B consumes it |
| Normal-mode → **local-ion** covariance transform; Gaussian `E_F` (ion-cut); `T_H` readout | **library (Phase B)** | new Gaussian-entanglement capability; mode-cut ≠ ion-cut |
| ML waveform pre-compensation (transfer function `T`, PyTorch) | **out of scope** | experimental hardware control, not a simulation primitive |

**Application-agnostic invariant** (lifted verbatim from WP-01/02/03/04): library symbols carry **no
consuming-application framing**; "Wittemer 2019/2020", "cosmological particle creation", and "Hawking temperature"
live only in benchmark tools + regression oracles + tutorials, never in a public symbol. Enforced by a decoupling
grep.

## 3. What the papers need — calculations to enable

**Core system (2020 Eq. 2.1)** — a single mode with time-dependent eigenfrequency, in a **fixed** operator basis
defined at `ω_ini = ω(0)`:

```
Ĥ(t) = ℏω(t)(â†â + ½) − (iℏ/4)·(d ln ω(t)/dt)·(â†² − â²)
```

The operator strings `â, â†` are the **lab-fixed** ladder operators at `ω(0)`; the second term is the **squeezing
generator** (`∝ â†² − â²`), switched on whenever `ω(t)` changes, with coefficient `−(i/4)·d ln ω/dt` on the
anti-Hermitian generator (equivalently real `¼·d ln ω/dt` on the Hermitian basis `−i(â†²−â²)`, Silveri 2015,
ref [21]). *A user wanting an instantaneous-basis picture applies the Bogoliubov transform themselves — in
the instantaneous basis the generator's sign/coefficient change.* This is a **centred, parity-preserving quadratic**
generator: from vacuum it keeps `⟨â⟩ = 0` (no coherent displacement — see §3.3). Adiabatic (`ω̇/ω² ≪ 1`) → the
state tracks the instantaneous ground state; non-adiabatic (`ω̇ ≳ ω²`) → an excited squeezed state that persists
after the ramp stops.

### 3.1 Phase-space evolution & readout (via the covariance matrix)

Dimensionless quadratures `x̂ = â + â†`, `p̂ = i(â† − â)` (vacuum variance `1`); means `⟨x̂⟩, ⟨p̂⟩`. Assemble the
**2×2 covariance matrix**

```
V = [[ (Δx)²,        C_xp ],
     [ C_xp,        (Δp)² ]],     C_xp = ½⟨{x̂,p̂}⟩ − ⟨x̂⟩⟨p̂⟩
```

and read the invariants:

- **Purity / thermal content** `ν = √(det V)` (symplectic eigenvalue): `ν = 1` pure, `ν = 2n̄_th + 1` thermal.
  `det V` is squeezing-invariant (squeezing is symplectic, det 1), so `ν` isolates the *thermal* part.
- **Squeezing** `r = ¼ ln(λ_max/λ_min)`, where `λ_{max,min}` are the eigenvalues (principal variances) of `V`.
  The eigenvalue **ratio** isolates squeezing (invariant under thermal scaling). `n̄_sq = sinh²(r)` is the
  **pure-squeezing** content — **not** the state's centred occupation when `n̄_th > 0`.
- **Coherent displacement** — consistent with frozen §7 (`⟨â⟩ = α`): `α = (⟨x̂⟩ + i⟨p̂⟩)/2`,
  `|α| = ½√(⟨x̂⟩² + ⟨p̂⟩²)`, `n_dsp = |α|²`.  *(The 2020 paper writes `|α| = √(x²+p²)` under an implicit
  normalisation; the repo-consistent form carries the ½.)*
- **Centred occupation** `n̄_centred = ¼ tr V − ½` (mixes thermal + squeezing) — for cross-checks only.

The naïve pure-state form `r = ½ arccosh[½ tr V]` agrees with the eigenvalue form **only for pure states**
(`det V = 1`); it spuriously reports `r > 0` for an unsqueezed thermal state. Use the covariance eigenvalue form.

**Wigner function** `W(x,p)` — the phase-space snapshot; pin the normalisation to vacuum-variance-1 (§6/§9).

### 3.2 Two extreme regimes

**(i) Quench** — Gaussian `ω(t)` pulse, width `δτ ≈ 0.5·2π/ω(0)`, amplitude `Δω`; the 2020 single-ion sweep reaches
`n̄_sq ≤ 0.2` (5 dB) up to `r = 1.1(3)` (6.3 dB, two-pulse). **(ii) Parametric modulation** —
`ω(t) = ω + δω_mod·sin(ω_mod t)`, `ω_mod = 2ω_ini`, `δω_mod ≈ 2π·8 kHz`, duration `T_mod`;
`n̄_sq = sinh²(2π g T_mod)` with parametric coupling `g/2π = 4.64(6) kHz` (2020 Fig 3, **displacement-free**).
`n̄_sq = sinh²(gτ)` is **the same functional form** the existing `two_mode_squeezing_hamiltonian` already uses
([hamiltonians.py:1623](src/iontrap_dynamics/hamiltonians.py#L1623)) — the parametric limit is the single-mode
degenerate partner of that SU(1,1) builder.

### 3.3 Forced displacement & echo / two-pulse purification (A3, optional)

**The centred quadratic Hamiltonian of §3 cannot produce a coherent displacement.** It preserves parity and keeps
`⟨â⟩ = 0` from vacuum, so `α ≡ 0` under A1 alone. The paper's parasitic displacement arises from the ion sitting
**off the RF null** — a *linear* force `∝ (â + â†)` with a time-dependent coefficient (equivalently, a moving
equilibrium position). Reproducing the displacement, and hence the echo suppression `δp = n_dsp,1/n_dsp,2 = 51(2)`
(2020) / `r=0.83(8), |α|=0.29(15)` (2019), **requires A3's force primitive** — it is not derivable from A1. The
echo geometry: the displacement rotates at `ω` while squeezing accumulates at `2ω`; at `t_free = π/ω` (odd
multiples) the second pulse cancels the residual `α` while amplifying `r`. Composed from the primitives via
piecewise `ω(t)` + `f(t)` — a **benchmark protocol**, not a new sequence primitive.

### 3.4 Phonon-number distribution — three distinct capabilities

The card must not conflate:

1. **Direct diagonals** `Pₙ = ⟨n|ρ_mode|n⟩` from the solved reduced state — trivial, **A2 ships this**.
2. **Analytic forward model** `Pₙ^par(r, α, n̄_th)` (squeezed + displaced + thermal) — depends on the
   squeezing/displacement **phases** and **operator ordering** (`D(α)S(ξ)` vs `S(ξ)D(α)`). **Deferred** until the
   2019 Supplemental pins the parametrisation; ship the **pure squeezed-vacuum** closed form as the A2 oracle only.
3. **Sideband inversion** — reconstructing `Pₙ` from measured BSB/RSB flopping (the experimental thermometry
   inverse; distinct from `diag(ρ)`). **Separate/deferred.**

Pure squeezed **vacuum** → **even-`n` only** (`P_odd ≈ 0`), the phonon-**pair** signature. **With displacement or
thermal content present, all `n` are populated** — the tutorials must not overstate "even-`n` only".

### 3.5 Adiabaticity diagnostic

`ω̇/ω²` (the paper's `≈ 5` figure of merit for "non-adiabatic"); adiabatic is `ω̇/ω² ≪ 1`.

### 3.6 (Phase B) Two-ion coupled modes, ion-cut entanglement, and `T_H`

Coupled EOM `δq̈_i + ω²_rad(t)δq_i + Σ_j M_ij δq_j = 0` (2019 Eq. 4); diagonalise `M_ij` → eigenmodes (COM `ω₁`,
rocking `ω₂`). A shared `ω_rad(t)` ramp squeezes a common mode and **entangles the two ions' spatial d.o.f.**
**The COM/rocking normal-mode cut is *not* the ion-A/ion-B cut** — the paper's entanglement is between the two
**ions**. Quantifying it requires transforming the normal-mode covariance matrix into **local-ion canonical
coordinates**, then evaluating the bipartite Gaussian measure across the ion cut: **Gaussian
entanglement-of-formation** `E_F` (2019: `E_F ≈ 0.41` vs vacuum `≈ 10⁻⁵`, Serafini/Adesso Gaussian criteria) and
mode-A/mode-B log-negativity. The **reduced state of one ion** is a mixed state; mapping it to an **effective
Hawking temperature `T_H`** needs a **local Hamiltonian + a thermal-equivalent prescription** (`n̄ = 1/(e^{ℏω/k_BT}−1)`
inversion at the ion's local frequency), not entropy alone.

Parameters to quote in the benchmark: single ion `ω_ini/2π ≈ 2.8 MHz` (2020) / two-ion radial `ω₂/2π` spanning
`2.50(1)→0.50(1) MHz` (2019, 1.6 e-foldings); `t_ramp = 1 μs`; `ω̇/ω² ≈ 5`; quench `δτ ≈ 0.18 μs`; parametric
`δω_mod ≈ 2π·8 kHz`, `g/2π = 4.64(6) kHz`, `ω_mod = 1.979(1)·2ω`; `n̄_ini < 0.1`. Anchor on `²⁵Mg⁺` (the paper's
ion, the `single-25Mg-plus` twin's species).

## 4. Capability map — reusable vs gaps

**Reusable (already in the package):**

| Ingredient | Existing API | Fit |
|---|---|---|
| Genuinely time-dependent `H(t)` in the solver (`[[H,f(t)]]`, `TimeQArray`) | `sequences.solve` ([sequences.py:142](src/iontrap_dynamics/sequences.py#L142), [:290](src/iontrap_dynamics/sequences.py#L290)) | **direct** — accepts any coefficient list; no ω(t)/force builder emits one yet |
| Arbitrary user waveform `f(t)` (+ `f_jax`) pattern | `modulated_carrier_hamiltonian(envelope=…, envelope_jax=…)` ([hamiltonians.py:1407](src/iontrap_dynamics/hamiltonians.py#L1407)) | **template** — modulates carrier *amplitude*, not mode frequency/force |
| Two-mode SU(1,1) squeezing Hamiltonian (constant `g`, `n̄=sinh²(gτ)`) | `two_mode_squeezing_hamiltonian` ([hamiltonians.py:1601](src/iontrap_dynamics/hamiltonians.py#L1601)); §23 | **anchor** — parametric-modulation & Phase-B partner |
| Squeezed / squeezed-coherent / thermal / two-mode-squeezed factories (oracles + initial states) | `squeezed_vacuum_mode`, `squeezed_coherent_mode`, `coherent_mode`, `two_mode_squeezed_vacuum` ([states.py:226–333](src/iontrap_dynamics/states.py#L226)); §6/§7/§23.1 | **direct** |
| Sideband engine (rates + Hamiltonians) | `analytic.{red,blue}_sideband_rabi_frequency[_full_ld]` ([analytic.py:200–540](src/iontrap_dynamics/analytic.py#L200)); `{red,blue}_sideband_hamiltonian` ([hamiltonians.py:503,608](src/iontrap_dynamics/hamiltonians.py#L503)) | reuse for the deferred `Pₙ` inversion |
| Mean-`n̄` sideband thermometry | `SidebandInference` ([measurement/protocols.py:634](src/iontrap_dynamics/measurement/protocols.py#L634)); §17.11 | partial — gives `n̄`, not full `Pₙ` |
| `n̂ = a†a` observable | `observables.number` ([observables.py:152](src/iontrap_dynamics/observables.py#L152)) | direct (for `Pₙ` diagonals) |
| Reduced-state entropy (`ptrace` + von-Neumann in bits) | `information/_common._von_neumann_entropy_bits` ([_common.py:31](src/iontrap_dynamics/information/_common.py#L31)) + `qutip.ptrace` | **template** (Phase B; entropy alone ≠ `T_H`) |
| Log-negativity across a bipartition | `log_negativity_trajectory` ([entanglement.py:134](src/iontrap_dynamics/entanglement.py#L134)) | partial — only `"spins"`/`"modes"` cut, **no mode-A/mode-B, no ion-cut** |
| Finite-shot readout + binomial CIs (for `Pₙ` sampling) | `measurement/` §17.9/§17.12 | direct |
| `ModeConfig` record (frequency + eigenvector-per-ion) | `ModeConfig` ([modes.py:42](src/iontrap_dynamics/modes.py#L42)) | Phase-B carrier of the sibling's normal modes |

**Gaps (new code):**

1. **(A1) Centred `ω(t)` squeezing Hamiltonian** — `nonadiabatic_squeezing_hamiltonian(hilbert, mode_label,
   waveform, …)` where `waveform` is a **`FrequencyWaveform` object exposing both `omega(t)` and `d_ln_omega_dt(t)`
   (+ JAX variants)** — *not* a bare `ω(t)` callable, since the squeezing coefficient is the **analytic**
   log-derivative and a black-box callable cannot supply its derivative safely (§9.4). Returns the
   `[[H_free, ω(t)], [−i(â†²−â²), ¼·d ln ω/dt]]` list — Hermitian basis `−i(â†²−â²)` with **real** coefficient
   `¼·d ln ω/dt`, i.e. `H_sq/ℏ = −(i/4)(d ln ω/dt)(â†²−â²)` (solver already accepts it). The **only single-mode
   parametric-drive** primitive; no single-mode `(a†²−a²)` generator exists today. **Convention-gated (§26).**
   → `hamiltonians.py` (+ named waveforms in `waveforms.py`).
2. **(A3, optional) Time-dependent linear force** — `H_force ∝ f(t)(â+â†)`; the only route to the parasitic
   coherent displacement. → `hamiltonians.py`.
3. **(A2) Phase-space / Gaussian module** — quadratures `x̂,p̂`, the 2×2 covariance `V` (incl. `C_xp`), `ν`, `r`,
   `n̄_sq`, `α`, and direct `Pₙ`. **These are state functionals, not linear operators** — they do *not* fit the
   `observables.py` `Observable` record (which holds spin_x/number-style linear ops,
   [observables.py:72–152](src/iontrap_dynamics/observables.py#L72)). → **new `phase_space.py`** (or `gaussian.py`).
4. **(A2) Wigner passthrough** — thin wrapper over `qutip.wigner` **with the scaling pinned** (qutip's default
   vacuum variance is ½, not the paper's 1). Convention-touching. → `phase_space.py`.
5. **Adiabaticity parameter** `ω̇/ω²` — small diagnostic. → `analytic.py`.
6. **(deferred) `Pₙ^par(r,α,n̄_th)`** named forward model — phase/ordering-dependent; ship after the 2019
   Supplemental. → future `phonon_distribution.py`.
7. **(Phase B) Gaussian-entanglement toolbox** — covariance matrix, **normal-mode → local-ion transform**,
   Gaussian `E_F` (ion-cut), mode-A/mode-B negativity, reduced-ion `T_H`. Absent
   (`entanglement_of_formation_trajectory` is **spin-only Wootters**, [entanglement.py:109](src/iontrap_dynamics/entanglement.py#L109)).
8. **(Phase B) Two-ion normal modes** — the Coulomb `M_ij` diagonalisation is **out-of-package** (owned by
   `iontrap-structure`; `ModeConfig` carries the result). → consume, don't re-derive.

## 5. The central scope fork — single-ion engine vs two-ion entanglement analog

- **Phase A (single-ion) is fully solvable today** once A1's `ω(t)` squeezing builder lands. The evolution is a
  closed-system unitary under a time-dependent **quadratic** Hamiltonian — `solve()` already carries `[[H,f(t)]]`.
  No new solver, no open-system machinery (contrast the WP-04 Phase-B `J(ω)` fork, which *did* need a non-Markovian
  solver). The convention-touching items are narrow: A1's generator (§26) and the quadrature/Wigner normalisation.
  A1+A2 reproduce the displacement-free 2020 results (esp. the parametric arm) and the 2019 single-mode squeezing;
  A3 adds the quench+echo displacement story.
- **Phase B (two-ion) crosses three surfaces at once.** (i) It needs the **classical two-ion mode spectrum**
  (`M_ij`, COM/rocking) — the `iontrap-structure` sibling's product, making Phase B the **first real dynamics
  consumer of the classical sibling** (a one-way `ModeConfig` data feed, exactly the cross-link flagged in the
  WP-04 card §9.6 and the structural-dynamics survey). (ii) It needs a **Gaussian-state toolbox** (covariance +
  local-ion transform + Gaussian `E_F` + mode/mode negativity) that does not exist and is independent enough to
  warrant **its own small deliberation card even before the two-ion application**. (iii) The **Hawking-temperature**
  reduced-state readout is new (needs a local Hamiltonian + thermal prescription, not entropy alone). None of this
  is blocked by a frozen boundary, but it is a materially larger build and is the deliberate part.

## 6. Conventions & governance implications

- **A1's squeezing Hamiltonian is the one convention-sensitive primitive.** It commits to: the sign/coefficient of
  the `(â†²−â²)` generator (coefficient `−(i/4)·d ln ω/dt`, Silveri 2015); the choice of a **fixed** operator basis at `ω(0)`
  (documented explicitly — an instantaneous-basis picture is the user's Bogoliubov transform); and the
  time-dependent-list decomposition. §6 (single-mode squeeze *parameter*) is **frozen** and is about the squeeze
  *operator/ellipse*, **not** a time-dependent generator — so this is a **new section, not an in-place §6 edit.**
  Route via the proven RLA/§25 path: review note → **side-car** proposal (propose-don't-apply) → green
  conventions-test → **maintainer seals** the bump. **The new section is §26** (the live sections run through §25;
  confirm with the 5-source collision grep at ratification), with a **`CONVENTION_VERSION` bump `0.4 → 0.5`**.
  **§6 and §7 remain untouched.**
- **Quadrature normalisation is a real pin — ship it named, don't assume it.** The paper's `x̂=â+â†`, `p̂=i(â†−â)`
  (vacuum variance `1`) is **not** the quantum-optics-standard `x=(a+a†)/√2` (variance ½). Ship public observables
  with an explicit name — `quadrature_x` / `quadrature_p` documented "Wittemer/Silveri convention, vacuum variance
  1" — and expose a `quadrature_normalization` constant/enum; the `r`/`α` readout is hard-coded to (or takes) that
  convention. Pin against two closed limits (vacuum → `r=0`; pure squeezed vacuum → `n̄_sq=sinh²r`) as **unit
  tests** in `tests/unit`.
- **The readout functionals are observable-only, no new symbol** (the MCF probe-QFI / ND `𝒩`,`ℬ` precedent):
  `r`, `ν`, `n̄_sq`, `|α|`, `n_dsp`, direct `Pₙ` are standard derived functionals — ship compute-only + regression
  oracles against closed forms (pure-squeezed `Pₙ`, analytic squeeze-kick `r`), cite the frozen sections, edit
  nothing. **But Wigner is *not* purely observable-only:** `qutip.wigner`'s default scaling puts the vacuum variance
  at ½, so the wrapper must **pin `g`** and document the vacuum-variance-1 convention — that pin lives on the §26
  convention surface.
- **§13/§15 hygiene — and a parity hazard the frozen §13 guard does not catch.** `r`/`n̄_sq`/`Pₙ` from a solved
  state inherit Fock-truncation status (§13); a squeezed state needs a **generous** truncation (the pair-creation
  tail). **The existing §13 guard monitors only the highest Fock level — which is parity-blind for a squeezed
  vacuum:** in an even-dimension Fock space the top level is *odd* and carries exactly zero population even when the
  even-`n` tail is badly truncated, so the guard falsely reports "converged". A **tail-window (sum over the last few
  levels) or cross-cutoff convergence test** is required — convergence in **both** `n̄_sq` **and** the `Pₙ` parity
  tail — and this may itself need a **§13 amendment** (frozen — side-car → seal, not an in-place edit).
- **(Phase B) Gaussian entanglement is new convention surface.** A covariance-matrix / symplectic toolbox, the
  normal-mode→local-ion transform, and a Gaussian `E_F` definition (Serafini/Adesso) would seal their own section;
  the mode-A/mode-B negativity is a small extension of the existing `partition` argument. Deferred with Phase B.
- **Governance.** New **WP** (one-card→one-WP), fresh dispatch family minted at ratification after the five-source
  collision grep. **Mnemonic: `SQ`** (squeezing dynamics), collision-free vs `ED`/`MC`/`RM`/`ND`; fallback `NA`
  (non-adiabatic) only on a future collision. (`PC` rejected — overlaps "particle creation" framing.) Current
  `CONVENTION_VERSION = "0.5"`.

## 7. Proposed tutorials

- **Tutorial A — "Squeezing a trapped ion by quenching its trap frequency."** Build A1's single-mode `ω(t)`
  squeezing Hamiltonian; run a **quench** and a **parametric modulation**; assemble the covariance matrix and watch
  the Wigner ellipse squeeze; extract `r` (from the eigenvalue ratio), `ν`, `n̄_sq`; show the **adiabatic →
  non-adiabatic** crossover via `ω̇/ω²`. If A3 lands, add the **echo/two-pulse** purification (displacement at `ω`,
  squeezing at `2ω`, cancellation at `t_free=π/ω`) — **flagging that the displacement needs the force term**, not
  A1 alone.
- **Tutorial B — "Phonon-pair creation and reading it out (Wittemer 2019/2020)."** Show the phonon-number
  diagonals `Pₙ` of a squeezed state — the **even-`n`-only pair signature for pure squeezed vacuum** (and note
  explicitly that displacement/thermal content fills in the odd `n`); connect `n̄_sq = sinh²(2π g T_mod)` for the
  parametric regime; sketch (framing only) the cosmological-particle-creation / Hawking analog (`a²(t)=ω²(t)/ω²(0)`,
  e-foldings, WKB phase `φ`) and the two-ion **ion-cut** entanglement → `T_H` outlook (Phase B).

## 8. Proposed first slice (Phase A) — WI sketch

**A1 — centred quadratic squeezing**
1. **`nonadiabatic_squeezing_hamiltonian`** (`hamiltonians.py`): the time-dependent-list `ω(t)` squeezing builder,
   `−(i/4)(d ln ω/dt)(â†²−â²)` (Hermitian basis `−i(â†²−â²)`, real coeff `¼·d ln ω/dt`) in a fixed `ω(0)` basis;
   takes a **`FrequencyWaveform` object** exposing `omega(t)`
   **and** `d_ln_omega_dt(t)` (+ JAX variants, `TimeQArray`-ready), per the proven `envelope`/`envelope_jax`
   pattern — named shapes (`gaussian_quench`, `sinusoidal_modulation`) in `waveforms.py` supply **analytic**
   derivatives (§9.4). **Convention side-car (§26, `0.4→0.5`).** Acceptance (as
   **convention-seal gates**, not just benchmark checks): **sudden** limit via a **narrowing smooth ramp** or an
   explicit **analytic squeeze kick** → `r = ½|ln(ω_f/ω_i)|` (locks the sign; a literal step has a δ-function
   `d ln ω/dt`); **adiabatic** limit → `r → 0` **for a cyclic waveform** returning to `ω_ini` (or `r` defined
   relative to the instantaneous vacuum); cross-backend (qutip vs jax) agreement at 1e-3.

**A2 — phase-space / Gaussian readout (observable-only)**
> **No-fork rule (from the Gaussian-toolbox card).** The **`N=1` covariance/symplectic core lives in `gaussian.py`**
> (created here — WP-SQ does **not** wait on the toolbox card), and **`phase_space.py` holds only the Wigner/readout
> façades** over it. **No `phase_space.py`-native symplectic arithmetic** — the single-mode readout is the `N=1`
> limit of `gaussian.py` APIs, which the toolbox later generalises to multimode. See
> `task cards/TC-gaussian-entanglement-toolbox.md` §5.
2. **Quadratures + covariance** (`gaussian.py`, façade in `phase_space.py`): `x̂=â+â†`, `p̂=i(â†−â)` with a named
   `quadrature_normalization` (vacuum variance 1); full 2×2 `V` (incl. `C_xp`). Acceptance: vacuum `V = I`;
   `squeezed_vacuum_mode(r)` → `λ = {e^{-2r}, e^{2r}}`.
3. **Readout** (`phase_space.py`): `ν=√(det V)`, `r=¼ln(λ_max/λ_min)`, `n̄_sq=sinh²r`, `α=(⟨x̂⟩+i⟨p̂⟩)/2`, `n_dsp`.
   Acceptance: round-trips `squeezed_coherent_mode(z,α)`; **thermal-squeezed** state → `r` invariant under `n̄_th`
   (the `tr V` form fails this — the key regression); displacement matches §7.
4. **Wigner** (`phase_space.py`): `qutip.wigner` wrapper with **`g` pinned** to the vacuum-variance-1 convention,
   documented. Acceptance: vacuum `W` isotropic with the pinned width.
5. **Direct `Pₙ`** (`phase_space.py`): `Pₙ = ⟨n|ρ_mode|n⟩` trajectory + the **pure squeezed-vacuum** closed-form
   oracle (`P_odd = 0`, even-`n` pair signature). Acceptance: parity structure; **§13/§15 truncation gate = a
   tail-window / cross-cutoff convergence test** (parity-aware — the top-level guard is fooled), converging in both
   `n̄_sq` and the `Pₙ` tail. *(The named `Pₙ^par(r,α,n̄_th)` forward model is deferred to `phonon_distribution.py`
   pending the 2019 Supplemental — phase/ordering-dependent.)*
6. **Adiabaticity diagnostic** `ω̇/ω²` (`analytic.py`) — acceptance against the paper's `≈5`.

**A3 — forced displacement + echo (optional Phase-A tail)**
7. **`H_force ∝ f(t)(â+â†)`** (`hamiltonians.py`) + the two-pulse echo protocol. Acceptance: a static offset seeds
   `α ≠ 0`; the echo suppresses it (`δp`), reproducing the qualitative 2020 Fig 2b trend.

**Benchmark + tutorials**
8. **Benchmark + oracle** (`tools/run_benchmark_*.py` → `benchmarks/data/`): reproduce Wittemer 2020 — the
   **parametric** arm `n̄_sq = sinh²(2π g T_mod)` (A1+A2, displacement-free); the quench `n̄_sq` vs amplitude and the
   echo `δp` (needs A3) — compute-only (`report.json`+`arrays.npz`+`plot.png`) + the analytic squeeze-kick / cyclic
   adiabatic regression oracle.
9. Tutorials A & B (print+plot + Colab; `docs/tutorials/19_…`, `20_…` + `tools/build_tutorial_notebooks.py` + the
   `notebooks`/`tutorials` CI guards).

## 9. Open questions & provisional decisions (for deliberation)

*Provisional decisions recorded from the v0.2 review; confirm at ratification.*

1. **Squeezing-generator convention (§9.1)** — **PROVISIONAL: fixed `ω(0)` basis** (Eq. 2.1), generator
   `−(i/4)(d ln ω/dt)(â†²−â²)`, sealed as **§26** with `CONVENTION_VERSION 0.4→0.5`; **§6/§7 untouched**. Sudden-quench
   (squeeze-kick) and **cyclic** adiabatic limits are the seal gates. *Confirm the sign against Silveri 2015.*
2. **Quadrature normalisation (§9.2)** — **PROVISIONAL: public named observables** `quadrature_x/p` with a
   documented vacuum-variance-1 `quadrature_normalization` constant + closed-limit unit tests; readout hard-coded to
   that convention. Wigner `g` pinned to match.
3. **`Pₙ` scope (§9.3)** — **PROVISIONAL: ship direct `diag(ρ)` + the pure-squeezed-vacuum oracle** in A2's
   `phase_space.py`; **defer** the named `Pₙ^par(r,α,n̄_th)` forward model (and the sideband inversion) to a later
   `phonon_distribution.py`, pending the 2019 Supplemental (phase/ordering).
4. **Waveform interface (§9.4)** — **PROVISIONAL: a `FrequencyWaveform` object exposing both `omega(t)` and
   `d_ln_omega_dt(t)` (+ JAX variants)** — *not* a bare `omega_of_t` callable, because the squeezing coefficient is
   the **analytic** log-derivative `d ln ω/dt` (entering `H_sq` as `−(i/4)·d ln ω/dt`) and a generic callable cannot supply its own derivative safely
   (numerical differentiation is unstable across the sharp quench, and the `H_free`/`H_sq` coefficients must stay
   mutually consistent). Named shapes (`gaussian_quench`, `sinusoidal_modulation`) in a small `waveforms.py`
   implement it with **analytic** derivatives; **paired callables `(omega_of_t, d_ln_omega_dt)` are the
   low-ceremony fallback**. Follows the `modulated_carrier_hamiltonian` `envelope`/`envelope_jax` precedent.
   **Contract:** `FrequencyWaveform` validates **finite, strictly positive `ω(t)`** and **finite `d ln ω/dt`**;
   paired-callable mutual consistency stays the **caller's responsibility** (build-time diagnostic spot-checks) —
   **never runtime numerical differentiation**.
5. **Echo / displacement (§9.5)** — **PROVISIONAL: A3 is an optional Phase-A tail** — a time-dependent linear
   **force primitive** (`∝ â+â†`) + a **benchmark-level** piecewise echo protocol (not a first-class sequence
   primitive). A1 alone cannot and must not claim the `δp` reproduction.
6. **`r` readout robustness (§9.new)** — **PROVISIONAL: covariance-eigenvalue form** (`ν`, `λ_max/λ_min`), not
   `tr V`; the thermal-squeezed invariance test is a required regression.
7. **Fock-truncation guard (§9.new)** — **PROVISIONAL: A2 adds a parity-aware tail-window / cross-cutoff
   convergence test**; assess whether it needs a **§13 amendment** (frozen — side-car → seal).
8. **Phase-B trigger (§9.6)** — **PROVISIONAL: defer** until Phase A ships **and** the `iontrap-structure`
   `ModeConfig` cross-repo feed is validated end-to-end. Consider a **standalone "Gaussian entanglement toolbox"
   card** (covariance + local-ion transform + Gaussian `E_F` + `T_H`) *before* the two-ion application, since the
   toolbox is application-independent. `T_H` needs a local Hamiltonian + thermal prescription, not entropy alone.
9. **Sibling cross-link (§9.7)** — Phase B is the natural **first dynamics consumer of `iontrap-structure`** (its
   `M_ij`→COM/rocking `ModeConfig`). One-way data feed (not a hard dependency), consistent with the WP-04 §9.6 note.
10. **Species/parameter grounding (§9.8)** — **PROVISIONAL: anchor on `²⁵Mg⁺`** at `ω_ini/2π ≈ 2.8 MHz`; **fetch
    the open 2020 figshare supplement now** for benchmark-oracle parameters.

## 10. Rooting sources

| Role | Source | Local |
|---|---|---|
| Target A (the toolkit + Eq. 2.1 + quench/parametric regimes) | Wittemer, Schröder, Hakelberg, Kiefer, Fey, Schützhold, Warring, Schaetz, **Phil. Trans. R. Soc. A 378, 20190230 (2020)** | ✓ `sources/pdf/` |
| Target B (phonon-pair creation, two-ion entanglement, cosmology analog) | Wittemer, Hakelberg, Kiefer, Schröder, Fey, Schützhold, Warring, Schaetz, **PRL 123, 180502 (2019)** | ✓ `sources/pdf/` |
| Methods: the Eq. 2.1 modulated-oscillator Hamiltonian (sign/coefficient) | **Silveri, Tuorila, Thuneberg, Paraoanu, Rep. Prog. Phys. 80, 056002 (2015)** | ✗ pending (ref [21]) — **needed to seal §26** |
| Parametric coupling `g`, `n̄_sq=sinh²(gT)` | **Burd et al., Science 364, 1163 (2019)** | ✗ pending (ref [22]) |
| Two-ion analog theory (ion-cut entanglement, `T_H`) | **Fey, Schaetz, Schützhold, PRA 98, 033407 (2018)** | ✗ pending (ref [19/49]) — **Phase-B blocker** |
| Numerical method + `Pₙ^par` + Gaussian `E_F` | PRL 123 180502 **Supplemental Material** (ref [10]) | ✗ **APS-gated — pending maintainer download** (blocks §3.4 model 2 + Phase B) |
| 2020 electronic supplementary material (benchmark parameters) | figshare **doi:10.6084/m9.figshare.c.5011307** | **open — fetch now** |
| Two-ion normal modes (`M_ij` → `ModeConfig`) | sibling `uwarring82/iontrap-structure` (equilibrium+modes, James-1998-validated) | ✓ separate repo |

**Phase-A rooting is sufficient today** (open 2020 + Silveri 2015 + Burd 2019 lock the Hamiltonian and the
parametric limit). **Phase B must not advance** until the 2019 Supplemental **and** Fey et al. PRA 98 033407 are
in hand.

## 11. References

- M. Wittemer, F. Hakelberg, P. Kiefer, J.-P. Schröder, C. Fey, R. Schützhold, U. Warring, T. Schaetz, *Phonon Pair Creation by Inflating Quantum Fluctuations in an Ion Trap*, **PRL 123, 180502 (2019)**.
- M. Wittemer, J.-P. Schröder, F. Hakelberg, P. Kiefer, C. Fey, R. Schützhold, U. Warring, T. Schaetz, *Trapped-ion toolkit for studies of quantum harmonic oscillators under extreme conditions*, **Phil. Trans. R. Soc. A 378, 20190230 (2020)**.
- M. P. Silveri, J. A. Tuorila, E. V. Thuneberg, G. S. Paraoanu, *Quantum systems under frequency modulation*, **Rep. Prog. Phys. 80, 056002 (2015)**.
- S. C. Burd et al., *Quantum amplification of mechanical oscillator motion*, **Science 364, 1163 (2019)**.
- C. Fey, T. Schaetz, R. Schützhold, *Ion-trap analog of particle creation in cosmology*, **PRA 98, 033407 (2018)**.
- M. Wittemer, G. Clos, H.-P. Breuer, U. Warring, T. Schaetz, **PRA 97, 020102(R) (2018)** — the sibling WP-04 target (`task cards/TC-non-markovianity-spectral-density.md`).
