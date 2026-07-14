# GT — Proposed CONVENTIONS §27 (multimode Gaussian toolbox) *(maintainer-applied)*

**Status:** WP-07 conventions-before-code (Dispatch GT1) — **proposal only**. The text
below is **ready to paste** but edits two governed files (`CONVENTIONS.md` and
`src/iontrap_dynamics/conventions.py`); per the WP governance rule those are
**maintainer-governed acts**, applied at the seal after a green conventions test, **not**
by the implementing agent. Nothing here is auto-applied.

**Licence:** CC BY-SA 4.0 (`WP/` governance material). Permitted side-car, not a WP.

**Post-v0.5 (v0.x target).** This amendment lands **after** the sealed §26 v0.5 seal
(`CONVENTION_VERSION` already `0.5`, WP-05). The bump is **0.5 → 0.6**, owned by the seal —
a `WP/FREEZE-v0.6.md` side-car or an in-place amendment, the maintainer's call. Nothing here
rides a spent freeze.

**This proposal does *not* touch a frozen section — read this.** §27 is a **brand-new
additive section** (the §23/§24/§25/§26 pattern). The Gaussian covariance/symplectic toolbox
is a **pure-motional object** (it acts only on the modes' quadratures `x̂, p̂`; no atomic/spin
transition), so — exactly like the §23 two-mode-squeezing builder, the §24 motional channels,
and the §26 squeezing generator — it lies **outside the §5 interaction-picture mandate by
construction** and is already covered by the sealed §5 scope note (v0.4, RL side-car A.1). **No
frozen-§5 edit is required.** §27 **reuses** the frozen §26.2 vacuum-variance-1 quadrature
normalisation and **extends it to the multimode case** (ordering + symplectic form + partial
transpose); it does **not** redefine the single-mode normalisation (contrast: §26.2 is
referenced, not edited). Frozen **§23** (the TMSV occupation `⟨n̂⟩ = sinh²|z|`, the toolbox's
regression oracle) is **cross-referenced, not edited**.

**§27 numbering — pre-reserved.** §26.4's cross-refs already reserve **§27** for "the multimode
Gaussian ordering / `Ω` / partial-transpose extension — reserved for the Gaussian-toolbox card,
**not** claimed here." A future non-Markovianity Phase B convention (`J(ω)`) takes the next free
section after §27 (§28+). Confirm free at seal via `grep -nE "^## 2[7-9]\." CONVENTIONS.md`.

---

## A. Staged CONVENTIONS edit

### A.1 New section (paste after §26)

### `## 27. Multimode Gaussian states: quadrature ordering, symplectic form, partial transpose *(staged — v0.6 target)*`

**Status:** §27 **staged at Dispatch GT1** (conventions-before-code); the covariance/symplectic
toolbox that *consumes* it (Dispatches GT1–GT6, `gaussian.py` + the ion `S`-adapter) lands only
after this seal. Staged, not sealed; seals under a post-v0.5 amendment (`CONVENTION_VERSION`
0.5 → 0.6). Conventions test `tests/conventions/test_gaussian_conventions.py` anchors the
ordering / symplectic spectra / partial-transpose behaviourally (below) and never reads the
`CONVENTIONS.md` markdown, so it stays green regardless of wording.

Multimode Gaussian objects are **pure-motional** (mode-only phase-space functionals on `N`
motional modes): a `2N × 2N` covariance matrix `V` and its symplectic invariants, outside the
§5 interaction picture (see the §5 scope note). §27 **reuses §26.2** (the vacuum-variance-1
single-mode quadratures `x̂ = â + â†`, `p̂ = i(â† − â)`, `[x̂, p̂] = 2i`) and fixes the multimode
extension: the **ordering**, the **symplectic form**, and the **partial-transpose sign map**.

#### 27.1 Quadrature ordering and covariance

For `N` modes with §26.2 quadratures, the phase-space vector is ordered **per mode**:

    R = (x̂₁, p̂₁, x̂₂, p̂₂, …, x̂_N, p̂_N)          (per-mode (x̂,p̂) blocks; Adesso/Serafini)

**not** the Simon `(x̂₁ … x̂_N, p̂₁ … p̂_N)` (position-block) layout. The covariance matrix and
first moments are

    V_ij = ½⟨{ΔR_i, ΔR_j}⟩,     d_i = ⟨R_i⟩,     ΔR = R − d,

so that **vacuum `V = 𝟙_{2N}`** (each single-mode block is §26.2's `𝟙₂`). This is the multimode
generalisation of §26.2/§26.4's single-mode `V`; the single-mode covariance is the `N = 1` limit
(a shared `gaussian.py` surface — the §27 toolbox generalises the §26.4 core, never forks a
parallel path).

#### 27.2 Symplectic form and physicality

The canonical commutation relations are `[R_i, R_j] = 2i Ω_ij` (the `2i` is the §26.2
vacuum-variance-1 scaling), with the **symplectic form**

    Ω = ⊕_{k=1}^{N} J,     J = [[0, 1], [−1, 0]]          (per-mode J-blocks, matching 27.1)

Bona-fide (Robertson–Schrödinger) physicality is **`V + iΩ ≥ 0`** — the Hermitian matrix
`V + iΩ` is positive semidefinite (vacuum saturates: `eig(V + iΩ) = {0, 2}`). This is the
**physicality guard**: a *direct Hermitian PSD test* on `V + iΩ`, **not** a symplectic-eigenvalue
check. The **symplectic (Williamson) eigenvalues** `ν_i` are the moduli of the eigenvalues of
`iΩV` — real, occurring in `±ν_i` pairs, counted **once per pair**; **compute as `|eig(iΩV)|`
deduplicated to pairs, not the singular values / matrix absolute value `√(M†M)`** of `iΩV`, which
in general differ. For a **real, symmetric, positive-semidefinite** covariance candidate,
physicality ⇔ `ν_i ≥ 1`; but `ν_i ≥ 1` is **not sufficient in general** — an *indefinite* `V` can
have `|eig(iΩV)| ≥ 1` while failing `V + iΩ ≥ 0` — so the implementation guards physicality with
the `V + iΩ ≥ 0` PSD test, never with a bare `ν_i ≥ 1` check. Vacuum `ν = 1`; thermal `ν = 2n̄+1`.

#### 27.3 Partial transpose (sign map)

For a bipartition `A | B` (a subset `B` of modes), the partial transpose acts on `V` by flipping
the **momenta** of the `B` modes:

    T_B : p̂_b → −p̂_b  for b ∈ B  (positions unchanged);     Ṽ = T_B V T_B

The PT symplectic eigenvalues `ν̃_k` are the `ν_i` of `Ṽ` (via 27.2 on `Ṽ`). (Flipping the `B`
positions instead is an equivalent convention; §27 pins the **momentum** flip.)

#### 27.4 Derived Gaussian functionals are observable-only (pinned formulas, no new symbol)

The functionals built from 27.1–27.3 are **standard derived quantities**, not new convention
symbols (the §26.4 / MCF / ND observable-only precedent). §27 pins their **formulas** so both
consumers agree; the toolbox ships them compute-only with closed-form oracles.

- **Purity** `μ = Tr(ρ²) = ∏_i (1/ν_i) = 1/√(det V)`. **Von-Neumann entropy** (bits, matching
  `information/_common._von_neumann_entropy_bits`) `S = Σ_i g(ν_i)`,
  `g(ν) = (ν+1)/2·log₂((ν+1)/2) − (ν−1)/2·log₂((ν−1)/2)` for `ν ≥ 1` (`g(1) = 0`).
- **Logarithmic negativity** (base-2, ebits) `E_N = Σ_k max(0, −log₂ ν̃_k)`, summed over **all**
  PT symplectic eigenvalues (27.3). The smallest-only form `E_N = max(0, −log₂ ν̃₋)` holds **only
  for two-mode and `1×N` cuts**. **PPT / `ν̃_k ≥ 1` certifies separability only for `1×N` Gaussian
  cuts** (Simon; Werner–Wolf); for `M×N` with `M, N ≥ 2`, PPT-bound-entangled Gaussian states
  exist, so `E_N = 0` is **not** a separability certificate — `E_N` is then only an **NPT
  witness**. Oracle: two-mode squeezed vacuum → `ν̃₋ = e^{−2r}`, `E_N = 2r/ln 2`.
- **Occupation / effective temperature** (first-moment-aware) `n̄ = (tr V_red + dᵀd − 2)/4`
  (from `â†â = (x̂² + p̂² − 2)/4`; the centered `(V₁₁+V₂₂−2)/4` form is valid only for `d = 0`),
  then `T_eff = ℏω_loc / (k_B · ln(1 + 1/n̄))` given an **explicit** local frequency `ω_loc`;
  `T_eff(n̄ = 0) = 0` by continuity, reject `n̄ < 0`. Energy-equivalent (neutral symbol `T_eff`);
  **not** `n̄ = (ν−1)/2` (that is the symplectically-equivalent thermal-core occupation, missing
  squeezing and first-moment energy). The `Hawking`/`T_H` framing is a consuming-application
  symbol (WP-SQ Phase B), never a toolbox symbol.
- **Locally-symmetric Gaussian `E_F`** (optional, guarded): closed form `E_F = c₊ log₂ c₊ −
  c₋ log₂ c₋` for `ν̃₋ < 1` (`E_F = 0` for `ν̃₋ ≥ 1`; the gate is the **domain condition
  `ν̃₋ < 1`**, not `max(0, ·)`), `c_± = (ν̃₋ ± 1)²/(4 ν̃₋)`, under a local-symmetry precondition.
  The **general (non-symmetric) two-mode `E_F` is deferred** (optimisation; the exact
  method/domain is Supplemental-gated).
- **Gaussianity precondition.** `V`/`d` are defined for **any** state, but purity, entropy, and
  the covariance log-negativity are the true *quantum-state* quantities **only for Gaussian
  states** (a quadratic `ω(t)` Hamiltonian preserves Gaussianity — the consumers are safe); for a
  non-Gaussian solved state these are the second-moment (Gaussian-equivalent) values, and `V`
  alone cannot certify Gaussianity. The toolbox states this and offers an optional
  symplectic-purity vs Fock-purity guard.
- **Symplectic congruence.** A linear canonical map `R_local = S R_normal` (`S` symplectic,
  `SΩSᵀ = Ω`) acts by `V_local = S V Sᵀ`. §27 fixes the generic congruence (convention-independent
  linear algebra); the **ion-specific `S`** (normal→local from a complete mode basis + cross-mode
  orthogonality + species masses + explicit local frequencies) is an application adapter, not a
  §27 symbol.

**Convention.** Multimode Gaussian objects use the per-mode ordering `R = (x̂₁,p̂₁,…,x̂_N,p̂_N)`
(27.1), the symplectic form `Ω = ⊕J` with `[R_i,R_j] = 2iΩ_ij` (27.2), and the partial-transpose
momentum-flip `p̂_B → −p̂_B` (27.3); derived functionals (27.4) are observable-only with the
pinned formulas above and hold only for Gaussian states. `x̂, p̂` and the vacuum-variance-1
normalisation are §26.2 (reused, not redefined).
**Cross-refs.** §26.2 (the single-mode quadrature normalisation this extends), §2 (tensor order,
spins-then-modes), §6/§7 (single-mode squeeze parameter / displacement `α`), §11 (eigenvector
normalisation — the ion `S`-adapter depends on it), §23 (TMSV `⟨n̂⟩ = sinh²|z|`, the regression
oracle — **cross-ref, not edited**), §25 §5-scope note (pure-motional exemption).
**Test.** `tests/conventions/test_gaussian_conventions.py` — vacuum `V = 𝟙_{2N}` and
`V + iΩ ≥ 0`; thermal `ν = 2n̄+1`; TMSV `ν̃₋ = e^{−2r}` and `E_N = 2r/ln 2`; entropy agrees with
`_von_neumann_entropy_bits` on truncated states; `1×N` separability certified vs `M×N` NPT-witness
scoping; `n̄ = (tr V + dᵀd − 2)/4` recovers a squeezed/displaced marginal's occupation (not
`(ν−1)/2`); congruence guard `SΩSᵀ = Ω`; symplectic eigenvalues via `|eig(iΩV)|` (not SVD).

---

## B. `mkdocs.yml` nav — none required

§27 is a CONVENTIONS *section* (not a new doc page) and edits no existing section, so **no new
`mkdocs.yml` nav line** is needed here.

---

## C. Seal-time edits — at the v0.6 seal only (maintainer-governed)

§27 is a **post-v0.5 amendment** (§26 sealed at v0.5). Whether the maintainer spins up a
`WP/FREEZE-v0.6.md` side-car or amends in place is the maintainer's call. Either way the
following land **once, in a single seal commit**:

1. **Confirm §27 free** — `grep -nE "^## 2[7-9]\." CONVENTIONS.md` returns nothing (as of the
   v0.5 seal, CONVENTIONS.md ends at §26). If a competing claim appears, renumber §27 → the next
   free integer and update WP-07 in lock-step.
2. **Append §27** (A.1) after §26; drop the `*(staged …)*` tag from its heading and append a
   freeze line: `**§27 freeze.** Sections 27.1–27.4 received a complete read-through for the v0.6
   convention gate. Post-v0.6 additions require a further version bump.`
3. **Bump** `CONVENTION_VERSION` 0.5 → 0.6 in `src/iontrap_dynamics/conventions.py`, **and** update
   the pinned literal in `tests/conventions/test_convention_version.py` (0.5 → 0.6) in the same
   commit (the FREEZE-v0.3 §3 / SQ §C bump-and-guard pairing). Optionally add a one-line inline
   comment recording the date + §27 context.
4. **Header block** → `**Scope:** Conventions covering §1–27`; freeze narrative names v0.6 and the
   added §27.
5. **Endorsement Marker** → restate: §17–18 under v0.2; §19–24 under the v0.3 freeze; §25 under
   v0.4; §26 under v0.5; **§27 under v0.6**; §1–16 carry forward.
6. **Footer** → `**Convention version:** 0.6 · 2026-XX-XX · multimode Gaussian toolbox (§27).`
7. **`WORKPLAN_v0.3.md`** → the WP-07 dispatch-track stub **§5.10** (staged with the WP;
   header/footer/version bumped v0.3.11 → v0.3.12 in lock-step) — a separate maintainer act at
   seal.
8. **Verify after the seal** — `git grep CONVENTION_VERSION` shows no stale `0.5` outside the
   updated `test_convention_version.py`; re-run the conventions tier.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with externally
validated laws. This is a Coastline proposal side-car within the Open-Science Harbour, stewarded
by U. Warring (AG Schätz, Albert-Ludwigs-Universität Freiburg). It stages — does not apply —
edits to the `CONVENTIONS.md` and `src/iontrap_dynamics/conventions.py` locks; the seal and the
`CONVENTION_VERSION` 0.5 → 0.6 bump are maintainer-governed acts. Licensed under **CC BY-SA 4.0**.

**Convention version:** stages `CONVENTIONS.md` §27 for a v0.6 amendment (post-v0.5).
**Workplan reference:** `WP/WP-07-gaussian-toolbox.md` §1 (card linkage), §4 (GT1); conventions
test `tests/conventions/test_gaussian_conventions.py`.
