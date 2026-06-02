# EDF — Proposed CONVENTIONS §19–22 + nav text *(maintainer-applied)*

**Status:** Dispatch **EDF** — proposal only. The text below is **ready to paste**
but edits two governed files (`CONVENTIONS.md`, `mkdocs.yml`); per the WP-01
governance rule those are **maintainer-governed acts**, applied at the v0.3 seal,
not by the implementing agent. Nothing here is auto-applied.

**Licence:** CC BY-SA 4.0 (`WP/` governance material). Permitted side-car, not a WP.

**What this proposal contains**

1. The four **staged** CONVENTIONS sections §19–22 (§A below) — paste verbatim
   after §18, before the `## Endorsement Marker`.
2. The **`mkdocs.yml` nav line** for the review note (§B).
3. The **seal-time edits** to `CONVENTIONS.md`'s header/marker/footer that the
   freeze gate requires (§C) — to be done **only at the seal**, coordinated with
   the single `CONVENTION_VERSION` 0.2 → 0.3 bump owned by
   [`FREEZE-v0.3.md`](FREEZE-v0.3.md).

**What this proposal deliberately does NOT do** (owned elsewhere): the
`CONVENTION_VERSION` bump and `tests/conventions/test_convention_version.py`
(owned by `FREEZE-v0.3.md` §3, executed once for the whole v0.3 freeze); the
`WORKPLAN_v0.3.md` §5.4 paste (WP-01 §13); the FREEZE-v0.3 §4 combined-vs-
WP-01-first timeline decision (taken at WP-02 ratification). The sections below
are **staged**, not sealed — sealing is step §C.

Every section cites [`docs/estimation-darwinism-review.md`](../docs/estimation-darwinism-review.md)
as the authority for its definition, satisfying the WP-01 §5 binding rule.

---

## A. Staged CONVENTIONS sections §19–22 (paste after §18)

### `## 19. Quantum and classical Fisher information *(staged — v0.3 Convention Freeze target)*`

**Status:** opened at Dispatch EDA (`src/iontrap_dynamics/information/fisher.py`).
Rules below are staged, not frozen: they seal at the v0.3 release under the
shared Convention Freeze gate coordinated by `WP/FREEZE-v0.3.md`. Definitions are
fixed by `docs/estimation-darwinism-review.md` §2.

#### 19.1 Quantum Fisher information — SLD convention

The library adopts the **symmetric logarithmic derivative (SLD)** quantum Fisher
information. The SLD L_θ solves ∂_θ ρ_θ = ½(L_θ ρ_θ + ρ_θ L_θ), and the QFI is
F_Q(θ) = Tr(ρ_θ L_θ²) — the Braunstein–Caves convention (four times the squared
Bures metric). The right/left logarithmic-derivative and Kubo–Mori variants are
**not** used. `quantum_fisher_information_trajectory` returns F_Q along a state
trajectory under a supplied Hermitian generator.

#### 19.2 Classical Fisher information

For an outcome distribution p(x; θ), the classical Fisher information is
F_cl(θ) = Σ_x (∂_θ p(x; θ))² / p(x; θ), with zero-probability outcomes
contributing zero (the term is masked, not evaluated). `classical_fisher_information`
takes probabilities and their parameter derivative.

#### 19.3 Cramér–Rao bound

`cramer_rao_bound` returns the single-shot bound 1/F for scalar F (and the matrix
inverse F⁻¹ for the Fisher matrix). The ν-repetition scaling Var(θ̂) ≥ 1/(ν F) is
the caller's responsibility — the library does not assume a repetition count.

#### 19.4 Linear-Gaussian closed form

For a linear model with known design matrix A and known covariance Σ ≻ 0,
`linear_gaussian_fisher` returns F = AᵀΣ⁻¹A (Σ checked positive-definite). The
saturation inequality F_cl ≤ F_Q (Braunstein–Caves) holds for every measurement
and is a built-in numerical guard.

**Convention.** SLD-QFI; Fisher information carries the parameter's
inverse-square unit (1/[θ]²) and is reparameterisation-covariant; CFI ≤ QFI for
every measurement.
**Cross-refs.** §3 (spin basis for the generator J_z); review note §2.
**Test.** `tests/unit/test_fisher.py` (numerics, CFI ≤ QFI);
`tests/regression/analytic/test_qfi_scaling.py` (QFI_GHZ = N², QFI_product = N);
`tests/regression/analytic/test_cfi_linear_gaussian.py` (F = AᵀΣ⁻¹A).

### `## 20. Quantum Darwinism — redundancy and recoverability *(staged — v0.3 Convention Freeze target)*`

**Status:** opened at Dispatch EDB (`src/iontrap_dynamics/information/redundancy.py`,
`src/iontrap_dynamics/information/recoverability.py`). Staged; seals with the v0.3
freeze. Definitions are fixed by `docs/estimation-darwinism-review.md` §3.

#### 20.1 Fragment mutual information

I(S : F) = S(ρ_S) + S(ρ_F) − S(ρ_{S∪F}), with the von Neumann entropy S(·) in
**bits** (base-2 logarithm; the library's `_von_neumann_entropy_bits`).
`fragment_mutual_information` and `partial_information_plot` operate on a state
and an explicit system/environment partition.

#### 20.2 Redundancy — information-deficit convention

The library adopts R_δ = N / f_δ, where N is the environment size and f_δ is the
smallest fragment fraction at which I(S : F) reaches (1 − δ)·H_S; δ is the
caller-supplied information deficit. `redundancy` returns R_δ for a given δ.

#### 20.3 Recoverability — clamped coherent information

recoverability = max(0, S(ρ_A) − S(ρ_{S∪A})) in bits — the Schumacher–Nielsen
coherent information I_c(S⟩A), floored at zero. This form is chosen over a
fidelity- or relative-entropy-of-recovery measure because it is computed from two
reduced-state entropies with no recovery-map optimisation and has exact closed-form
endpoints. `recoverability` takes a state, the system indices, and the accessible
indices.

**Convention.** Entropies in bits; recoverability clamped at zero; redundancy is
the deficit form R_δ = N/f_δ.
**Cross-refs.** §19 (shared entropy/partition machinery); review note §3.
**Test.** `tests/unit/test_redundancy.py`, `tests/unit/test_recoverability.py`
(endpoints + monotonicity);
`tests/regression/analytic/test_darwinism_redundancy.py` (GHZ-cascade plateau,
R_δ = N); `tests/regression/analytic/test_recoverability_channel.py` (Werner
endpoints 0 → H_S).

### `## 21. GHZ and cat state conventions *(staged — v0.3 Convention Freeze target)*`

**Status:** opened at Dispatch EDC (`src/iontrap_dynamics/states.py`). Staged;
seals with the v0.3 freeze. Definitions are fixed by
`docs/estimation-darwinism-review.md` §4.

#### 21.1 GHZ state and parity fringe

`ghz_state(hilbert)` builds |GHZ_N⟩ = (|0…0⟩ + |1…1⟩)/√2 in the §3 spin basis.
Under e^{−iφ J_z} the parity observable oscillates at N times the single-qubit
rate: ⟨X^⊗N⟩ = cos(N φ).

#### 21.2 Cat state parity

`cat_mode(fock_dim, alpha, *, parity)` builds the even/odd cat |α⟩ ± |−α⟩ in a
truncated Fock space; the even (odd) state is the +1 (−1) eigenstate of photon
parity. `fock_dim ≤ 0` and non-finite α raise `ConventionError`.

**Convention.** GHZ in the computational/§3 spin basis; cat parity sign maps
even → +1, odd → −1; Fock truncation per §13 convergence.
**Cross-refs.** §3 (spin basis), §13 (Fock truncation); review note §4.
**Test.** `tests/unit/test_states_ghz_cat.py` (norm, parity, `ConventionError`);
`tests/regression/analytic/test_ghz_cat_properties.py` (⟨X^⊗N⟩ = cos(Nφ); cat
parity ±1).

### `## 22. Common-mode (shared-latent) phase channel *(staged — v0.3 Convention Freeze target)*`

**Status:** opened at Dispatch EDD (`src/iontrap_dynamics/systematics/common_mode.py`).
Staged; seals with the v0.3 freeze. Definitions are fixed by
`docs/estimation-darwinism-review.md` §5.

#### 22.1 Channel definition

`CommonModePhase(sigma_rad, correlation, label)` draws, per subsystem,
offset_i = √c · ξ_shared + √(1 − c) · ξ_i with ξ_shared, ξ_i ~ 𝒩(0, σ²) and
c = correlation ∈ [0, 1]. The marginal per-subsystem variance is σ² at every c.
`perturb_common_mode` applies the shared draw across a list of drives.

#### 22.2 Difference-variance convention and the rejection limit

Var(offset_0 − offset_1) = 2σ²(1 − c). At c = 0 the offsets are independent
(variance 2σ²); at c = 1 the shared latent cancels exactly (variance 0) —
common-mode rejection, the standard differential-measurement limit. The variance
is monotone decreasing in c.

**Convention.** Shared draw is `√c · ξ_shared + √(1−c) · ξ_i` (marginal variance
σ² preserved across c); difference variance 2σ²(1−c); c = 1 is exact rejection,
**measured, not enforced** (a broken draw would surface in the benchmark).
**Cross-refs.** §18 (systematics layer — `PhaseDrift` is the c = 0 reduction);
review note §5.
**Test.** `tests/unit/test_common_mode.py` (c = 0 reduces to independent draw;
`dataclasses.replace` not mutation; `ValueError` on `shots < 1`);
`tests/regression/analytic/test_common_mode_rejection.py` (2σ²(1−c); exact
rejection at c = 1).

---

## B. `mkdocs.yml` nav line (paste between `Benchmarks:` and `Boundary Decision Tree:`)

```yaml
  - "Literature Review — Estimation & Darwinism": estimation-darwinism-review.md
```

Exact placement (top-level nav, per WP-01 §9):

```yaml
  - Benchmarks: benchmarks.md
  - "Literature Review — Estimation & Darwinism": estimation-darwinism-review.md   # <-- add
  - Tutorials:
      ...
  - Boundary Decision Tree: boundary-decision-tree.md
```

(WP-01 §9 names the slot "between `Benchmarks:` and `Boundary Decision Tree:`";
the `Tutorials:` group sits between them in the current file, so the line goes
immediately after `Benchmarks:`.) `mkdocs build --strict` currently warns that
the page is off-nav; this line clears it.

---

## C. Seal-time edits to `CONVENTIONS.md` (at the v0.3 release only)

Done **once**, at the seal, in the same commit as the single `CONVENTION_VERSION`
0.2 → 0.3 bump (`FREEZE-v0.3.md` §3). Following the §17/§18 staged → frozen
pattern (WP-01 §6 steps 3–6):

1. **Seal each section.** Append to §19–22 a closing line, e.g.:
   `**§19 freeze.** Sections 19.1–19.4 received a complete read-through for the
   Convention Freeze gate at the v0.3 release. Post-freeze additions require a
   CONVENTIONS.md version bump.` Drop each `*(staged — v0.3 Convention Freeze
   target)*` tag from the H2 headings.
2. **Header block.** Update the `Status:` line and `**Scope:**` to cover §1–22
   (currently §1–18), and the freeze narrative to name v0.3 and the added §§.
3. **Endorsement Marker.** Restate: §17–18 closed under v0.2; **§19–22 closed
   under the v0.3 Convention Freeze**; §1–16 carry forward unchanged.
4. **Footer.** `**Convention version:** 0.3 · 2026-06-XX · v0.3 Convention Freeze.`
   and refresh `**Workplan reference:**`.

If the FREEZE-v0.3 §4 decision is **combined**, §23–24 (WP-02) seal in the same
commit; if **WP-01-first**, seal §19–22 as v0.3 and §23–24 move to a later v0.4
freeze. That call is taken at WP-02 ratification, not here.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with
externally validated laws. This is a Coastline proposal side-car within the
Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-
Universität Freiburg). It stages — does not apply — edits to the `CONVENTIONS.md`
and `mkdocs.yml` locks; the seal and bump are maintainer-governed acts owned by
`WP/FREEZE-v0.3.md`. Licensed under **CC BY-SA 4.0**.

**Convention version:** stages `CONVENTIONS.md` §19–22 for the v0.3 freeze.
**Workplan reference:** `WP/WP-01-estimation-darwinism.md` §5, §6; `WP/FREEZE-v0.3.md`.
