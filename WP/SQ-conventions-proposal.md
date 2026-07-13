# SQ — Proposed CONVENTIONS §26 (non-adiabatic squeezing) *(maintainer-applied)*

**Status:** WP-05 conventions-before-code (Dispatch SQ1, A1) — **proposal only**. The text
below is **ready to paste** but edits two governed files (`CONVENTIONS.md` and
`src/iontrap_dynamics/conventions.py`); per the WP governance rule those are
**maintainer-governed acts**, applied at the seal after a green conventions test, **not**
by the implementing agent. Nothing here is auto-applied.

**Licence:** CC BY-SA 4.0 (`WP/` governance material). Permitted side-car, not a WP.

**Post-v0.4 (v0.x target).** This amendment lands **after** the sealed v0.3 freeze
(`FREEZE-v0.3.md`, `fdcd20f`) and the WP-03 v0.4 seal (`CONVENTION_VERSION` already `0.4`,
§25). The bump is **0.4 → 0.5**, owned by the seal — a `WP/FREEZE-v0.5.md` side-car or an
in-place amendment, the maintainer's call. Nothing here rides a spent freeze.

**This proposal does *not* touch a frozen section — read this.** §26 is a **brand-new
additive section** (the §23/§24/§25 pattern). The squeezing generator is a **pure-motional
object** (it acts only on the mode `â, â†`; no atomic/spin transition), so — exactly like
the §23 two-mode-squeezing builder and the §24 motional channels — it lies **outside the §5
interaction-picture mandate by construction** and is already covered by the sealed §5 scope
note (v0.4, RL side-car A.1). **No frozen-§5 edit is required** (contrast the RL proposal).
Frozen **§6** (the squeeze *parameter / ellipse*, a static operator) and **§7** (displacement)
are **untouched**: §26 defines a *time-dependent generator* and a *quadrature normalisation*,
which §6/§7 do not.

**§26 numbering — supersedes a non-binding forward-guess.** The §5.7 WORKPLAN note (WP-04)
tentatively wrote that the *deferred* non-Markovianity Phase B (`J(ω)` / structured bath)
"would require a new CONVENTIONS section (**likely §26**)" — but explicitly "**not
pre-claimed**", and that Phase B has **no ratified WP**. WP-05 is the **first to actively
seal**, so it takes **§26**. The Gaussian-toolbox card (`TC-gaussian-entanglement-toolbox.md`)
holds **§27** (multimode ordering + `Ω` + partial-transpose). A future non-Markovianity
Phase B convention takes the **next free section after those** (§28+). Confirm free at seal
via `grep -nE "^## 2[6-9]\." CONVENTIONS.md`.

---

## A. Staged CONVENTIONS edit

### A.1 New section (paste after §25)

### `## 26. Non-adiabatic squeezing: time-dependent-frequency generator, quadrature normalisation, Wigner scaling *(staged — v0.5 target)*`

**Status:** §26 **staged at Dispatch SQ1** (conventions-before-code); the
`nonadiabatic_squeezing_hamiltonian` builder and the phase-space readout that *consume* it
(Dispatches SQ1–SQ4, `hamiltonians.py` / `waveforms.py` / `gaussian.py` / `phase_space.py`)
land only after this seal. Staged, not sealed; seals under a post-v0.4 amendment
(`CONVENTION_VERSION` 0.4 → 0.5). Conventions test
`tests/conventions/test_squeezing_conventions.py` anchors the coefficients/limits
behaviourally (below) and never reads the `CONVENTIONS.md` markdown, so it stays green
regardless of wording.

Non-adiabatic squeezing objects are **pure-motional** (Schrödinger-picture, mode-only): a
single harmonic oscillator whose trap frequency `ω(t)` is varied in time, generating
squeezing. Returned as time-dependent Hermitian `H/ℏ` in **rad·s⁻¹** on a single-mode
embedding (§2 order), outside the §5 interaction picture (see the §5 scope note).

#### 26.1 Time-dependent-frequency squeezing generator

For a mode of instantaneous frequency `ω(t)`, with **time-independent** ladder operators
`â, â†` defined at the **initial** frequency `ω_ini = ω(0)` (a **fixed** basis, not the
instantaneous one), the generator is (Silveri, Tuorila, Thuneberg, Paraoanu, *Rep. Prog.
Phys.* **80**, 056002 (2015)):

    H(t)/ℏ = ω(t)(â†â + ½) − (i/4)·(d ln ω/dt)·(â†² − â²)

- The **squeezing term coefficient is `−(i/4)·d ln ω/dt`** (imaginary; the `i` is
  mandatory). Because `â†² − â²` is **anti-Hermitian**, `−i(â†² − â²)` **is Hermitian**, so
  in a time-dependent-list decomposition the Hermitian basis operator is
  `H_sq = −i(â†² − â²)` carried with the **real** coefficient `¼·d ln ω/dt`:
  `H(t) = [[H_free = â†â + ½, ω(t)], [H_sq = −i(â†² − â²), ¼·d ln ω/dt]]`.
- **Fixed `ω(0)` basis.** Operator strings are the lab-fixed ladder operators at `ω(0)`. An
  instantaneous-basis picture is the user's Bogoliubov transform (the generator sign/coefficient
  change there); the convention is the **fixed** basis.
- **Limits (conventions-test gates).** *Sudden* (analytic squeeze kick / narrowing smooth
  ramp — **not** a literal step, whose `d ln ω/dt` is a δ-function): squeezing
  `r = ½|ln(ω_f/ω_i)|`. *Adiabatic* (`ω̇/ω² ≪ 1`): `r → 0` **for a cyclic waveform** returning
  to `ω_ini` (or `r` defined relative to the instantaneous vacuum).

#### 26.2 Quadrature normalisation (vacuum variance 1)

Dimensionless quadratures are

    x̂ = â + â†,     p̂ = i(â† − â),

so the **vacuum quadrature variance is 1** and `[x̂, p̂] = 2i`. This is **not** the
quantum-optics `x = (a + a†)/√2` convention (vacuum variance ½, `[x, p] = i`); it is fixed
here because the downstream squeezing readout (§26.4) and the future multimode Gaussian
toolbox (§27) depend on it. The single-mode covariance is `V_ij = ½⟨{ΔR_i, ΔR_j}⟩`,
`R = (x̂, p̂)`; **vacuum `V = 𝟙₂`**.

#### 26.3 Wigner scaling

Any Wigner-function helper pins its scaling to §26.2 so the **vacuum Wigner has variance 1**
(consistent with `x̂ = â + â†`). QuTiP's `qutip.wigner(…, g=…)` defaults to `g = √2`
(vacuum variance ½); the wrapper sets `g` to the value giving vacuum variance 1 (`g = 1` for
`x̂ = â + â†`) and **documents it**. The scaling is part of the sealed convention, not a
free display parameter.

#### 26.4 Readout functionals are observable-only (no new symbol)

The squeezing/displacement readout derived from the covariance matrix — the symplectic
eigenvalue `ν = √(det V)`, the squeezing `r = ¼·ln(λ_max/λ_min)` (eigenvalues of `V`; **not**
`tr V`, which conflates squeezing with thermal width), `n̄_sq = sinh²r`, the coherent
displacement `α = (⟨x̂⟩ + i⟨p̂⟩)/2` with `|α| = ½√(⟨x̂⟩² + ⟨p̂⟩²)` (§7-consistent), and the
direct phonon diagonals `Pₙ = ⟨n|ρ|n⟩` — are **standard derived functionals**, not new
convention symbols (the MCF probe-QFI / ND `𝒩`,`ℬ` precedent). They **compose** §26.2 and
are shipped compute-only with closed-form oracles; §26 seals only the generator (26.1),
the quadrature normalisation (26.2), and the Wigner scaling (26.3).

**Convention.** Non-adiabatic-squeezing builders return time-dependent Hermitian `Qobj`
lists per 26.1 (fixed `ω(0)` basis, real `¼·d ln ω/dt` on the Hermitian `−i(â†²−â²)`);
quadratures per 26.2 (vacuum variance 1); Wigner scaling per 26.3; readout functionals per
26.4 are observable-only.
**Cross-refs.** §2 (tensor order), §6 (the squeeze *parameter/ellipse* — a distinct, static
object), §7 (displacement `α`), §10 (Lamb–Dicke — an apparatus realisation), §23/§24 (other
pure-motional physics-layer conventions; the §5-exemption precedent), §25 §5-scope note (why
pure-motional objects are exempt from the interaction-picture default), **§27** (the multimode
Gaussian ordering / `Ω` / partial-transpose extension — reserved for the Gaussian-toolbox card,
**not** claimed here).
**Test.** `tests/conventions/test_squeezing_conventions.py` — sudden-quench
`r = ½|ln(ω_f/ω_i)|`; cyclic-adiabatic `r → 0`; vacuum `V = 𝟙`; squeezed-vacuum principal
variances `e^{∓2r}`; thermal-squeezed `r` invariant under `n̄_th` (the covariance-eigenvalue
vs `tr V` regression); `FrequencyWaveform` rejects non-positive/non-finite `ω` and non-finite
`d ln ω/dt`.

---

## B. `mkdocs.yml` nav — none required

§26 is a CONVENTIONS *section* (not a new doc page) and edits no existing section, so **no new
`mkdocs.yml` nav line** is needed here.

---

## C. Seal-time edits — at the v0.5 seal only (maintainer-governed)

§26 is a **post-v0.4 amendment** (`FREEZE-v0.3.md` sealed; WP-03 v0.4 seal applied). Whether
the maintainer spins up a `WP/FREEZE-v0.5.md` side-car or amends in place is the maintainer's
call. Either way the following land **once, in a single seal commit**:

1. **Confirm §26/§27 free** — `grep -nE "^## 2[6-9]\." CONVENTIONS.md` returns nothing (as of
   2026-07-10, CONVENTIONS.md ends at §25). If a competing claim appears, renumber §26 → the
   next free integer and update WP-05 + the toolbox card in lock-step.
2. **Append §26** (A.1) after §25; drop the `*(staged …)*` tag from its heading and append a
   freeze line: `**§26 freeze.** Sections 26.1–26.4 received a complete read-through for the
   v0.5 convention gate. Post-v0.5 additions require a further version bump.`
3. **Bump** `CONVENTION_VERSION` 0.4 → 0.5 in `src/iontrap_dynamics/conventions.py`, **and**
   update the pinned literal in `tests/conventions/test_convention_version.py` (0.4 → 0.5) in
   the same commit (the FREEZE-v0.3 §3 / RL §C bump-and-guard pairing). Optionally add a
   one-line inline comment recording the date + §26 context.
4. **Header block** → `**Scope:** Conventions covering §1–26`; freeze narrative names v0.5 and
   the added §26.
5. **Endorsement Marker** → restate: §17–18 under v0.2; §19–24 under the v0.3 freeze; §25 under
   v0.4; **§26 under v0.5**; §1–16 carry forward.
6. **Footer** → `**Convention version:** 0.5 · 2026-XX-XX · non-adiabatic squeezing (§26).`
7. **`WORKPLAN_v0.3.md`** → the WP-05 dispatch-track stub **§5.8** (staged with this WP; header/
   footer/version bumped v0.3.9 → v0.3.10 in lock-step) — a separate maintainer act at seal.
8. **Verify after the seal** — `git grep CONVENTION_VERSION` shows no stale `0.4` outside the
   updated `test_convention_version.py`; re-run the conventions tier.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally
validated laws. This is a Coastline proposal side-car within the Open-Science Harbour,
stewarded by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg). It stages — does not
apply — edits to the `CONVENTIONS.md` and `src/iontrap_dynamics/conventions.py` locks; the seal
and the `CONVENTION_VERSION` 0.4 → 0.5 bump are maintainer-governed acts. Licensed under
**CC BY-SA 4.0**.

**Convention version:** stages `CONVENTIONS.md` §26 for a v0.5 amendment (post-v0.4).
**Workplan reference:** `WP/WP-05-nonadiabatic-squeezing.md` §0 (R2), §2 (SQ1); conventions test
`tests/conventions/test_squeezing_conventions.py`.
