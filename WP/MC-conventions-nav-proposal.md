# MC — Proposed CONVENTIONS §23–24 + nav text *(maintainer-applied)*

**Status:** WP-02 conventions-before-seal — proposal only. The text below is
**ready to paste** but edits two governed files (`CONVENTIONS.md`, `mkdocs.yml`);
per the WP governance rule those are **maintainer-governed acts**, applied at the
v0.3 seal, not by the implementing agent. Nothing here is auto-applied.

**Licence:** CC BY-SA 4.0 (`WP/` governance material). Permitted side-car, not a WP.

**Combined seal (FREEZE-v0.3 §4 = ratified Combined, bounded to WP-02 P0).** The
WP-02 P0 subset (F1+F2+F3 / WI-1–WI-3) is **complete**, so this proposal and
WP-01's [`EDF-conventions-nav-proposal.md`](EDF-conventions-nav-proposal.md) are
applied **together in one maintainer seal commit**: §19–22 (WP-01) **and**
§23–24 (WP-02), the single `CONVENTION_VERSION` 0.2 → 0.3 bump, both review-note
nav lines, and both WORKPLAN stubs (§5.4 + §5.5). The bump + the
`tests/conventions/test_convention_version.py` guard are owned once by
[`FREEZE-v0.3.md`](FREEZE-v0.3.md) §3.

Every section cites [`docs/two-mode-motional-review.md`](../docs/two-mode-motional-review.md)
as the authority for its definition (WP-02 §7 binding rule).

---

## A. Staged CONVENTIONS sections §23–24 (paste after §22)

### `## 23. Two-mode squeezing / SU(1,1) *(staged — v0.3 Convention Freeze target)*`

**Status:** opened at Dispatch MCA (`src/iontrap_dynamics/hamiltonians.py`) and
MCB (`src/iontrap_dynamics/states.py`). Staged, not frozen; seals at the v0.3
release under the shared Convention Freeze (`WP/FREEZE-v0.3.md`). Definitions are
fixed by `docs/two-mode-motional-review.md` §2.

#### 23.1 Two-mode squeeze operator and squeezed-vacuum factory

The two-mode squeeze operator is ``S₂(z) = exp(z* âb̂ − z â†b̂†)`` with
``z = r·e^{iθ}`` — the two-mode partner of the single-mode squeeze (§6),
**without** the ½ factor and π-period phase (two distinct modes).
`states.two_mode_squeezed_vacuum(fock_dims, z)` returns ``S₂(z)|0,0⟩``: the
Schmidt state ``Σₙ cₙ|n,n⟩`` with ``|cₙ| = tanhⁿr/cosh r``, per-mode
``⟨n̂_a⟩ = ⟨n̂_b⟩ = sinh²|z|``. **This is defined explicitly, not via
`qutip.squeezing`** (whose ½-convention gives ``sinh²(|z|/2)``).

#### 23.2 Hamiltonian convention and phase/sign map

`hamiltonians.two_mode_squeezing_hamiltonian` builds
``H_TMS/ℏ = i g (e^{iφ} â†b̂† − e^{−iφ} âb̂)`` (Hermitian; ``g`` in rad·s⁻¹, ``φ``
the squeezing phase). Evolving the vacuum for a time ``τ`` gives the two-mode
squeezed vacuum with the **signed complex** parameter ``z = −gτ·e^{iφ}``
(magnitude ``r = |z| = gτ``), per-mode ``sinh²(gτ)``. ``g`` may be negative (a
``π`` phase shift); identical mode labels and non-finite ``g``/``φ`` raise.

#### 23.3 su(1,1) algebra and the conserved Casimir

The generators ``K̂₊ = â†b̂†``, ``K̂₋ = âb̂``, ``K̂₀ = ½(n̂_a + n̂_b + 1)`` close
su(1,1). The squeezing creates/annihilates excitations in **pairs**, so it
commutes with the difference number ``n̂_a − n̂_b`` (the conserved Casimir label):
a state seeded from the vacuum keeps ``⟨n̂_a⟩ = ⟨n̂_b⟩``.

#### 23.4 Beamsplitter (SU(2))

`hamiltonians.beamsplitter_hamiltonian` builds
``H_BS/ℏ = J (e^{iφ} â†b̂ + e^{−iφ} âb̂†)`` (the SU(2) partner), which conserves
the **total** occupation ``n̂_a + n̂_b``.

**Convention.** Two-mode squeeze with no ½ factor (per-mode ``sinh²|z|``);
label-based mode embedding (agnostic to tensor ordering); ``z = −gτ·e^{iφ}``;
the two interacting modes must be distinct.
**Cross-refs.** §6 (single-mode squeeze), §3 (spin basis), §11 (mode
eigenvectors); review note §2.
**Test.** `tests/unit/test_two_mode.py` (Hermiticity, conserved-charge
commutators, validation); `tests/regression/analytic/test_two_mode_squeezing.py`
(``sinh²`` occupation, Casimir, factory↔Hamiltonian consistency incl. non-zero
phase, beamsplitter total-number conservation).

### `## 24. Motional CPTP channels *(staged — v0.3 Convention Freeze target)*`

**Status:** opened at Dispatch MCC (`src/iontrap_dynamics/channels.py`,
`src/iontrap_dynamics/sequences.py` `solve`). Staged; seals with the v0.3 freeze.
Definitions are fixed by `docs/two-mode-motional-review.md` §3.

#### 24.1 Lindblad parameterisation

Typed motional channels are dissipators in Lindblad (GKSL) form
``dρ/dt = −(i/ℏ)[H, ρ] + Σ_k (L_k ρ L_k† − ½{L_k†L_k, ρ})``, each contributing
collapse operator(s) ``L_k`` on a **labelled mode** with rates in **s⁻¹**.

#### 24.2 The three dissipators

- `AmplitudeDamping(mode, rate)` — ``L = √κ·â`` (zero-temperature; ``⟨n̂⟩ →
  0`` as ``e^{−κt}``).
- `Heating(mode, rate, n_bar_bath)` — ``L₋ = √(κ(n̄+1))·â``, ``L₊ = √(κn̄)·â†``
  (relaxes to the bath, ``⟨n̂⟩ → n̄``; the anomalous-heating model).
- `Dephasing(mode, rate)` — ``L = √γ·n̂`` (coherence ``ρ_{nm}`` decays as
  ``e^{−(γ/2)(n−m)²t}``; ``⟨n̂⟩`` unchanged).

`Depolarising` is **deferred** — not a canonical single-mode bosonic dissipator.

#### 24.3 Solver routing

`sequences.solve(…, channels=[…])` builds the collapse operators and forces the
master-equation (`mesolve`) path. An **empty** ``channels`` leaves the solver
byte-for-byte unchanged; `backend="jax"` and `solver="sesolve"` with dissipative
channels raise.

#### 24.4 Sequence-aware (time-windowed) application

A channel may carry ``window=(t0, t1)`` (half-open ``[t0, t1)``); the dissipation
is active only inside, via QuTiP's ``[L, coeff]`` time-dependent format (the
``coeff`` obeys the ``QobjEvo`` ``f(t, args)`` contract). What is order-dependent
is the **temporal schedule**, not the `channels`-list order (same-window channels
are simultaneous Lindblad terms) — the library does not assume the dissipators
commute (the R8 boundary). When any channel is windowed, `solve` caps the
integrator ``max_step`` at the smallest gap in the union of the output times and
window endpoints, so a short window cannot be stepped over.

**Convention.** Lindblad rates in s⁻¹; the collapse-operator map above; channels
force `mesolve`; half-open windows; temporal-schedule (not list) order-dependence.
**Cross-refs.** §18 (systematics layer — the common-mode/jitter channels are §22
of the WP-01 set; these are the *motional* dissipators); §13 (Fock truncation —
heating must not saturate the truncation); review note §3.
**Test.** `tests/unit/test_channels.py`;
`tests/regression/analytic/test_motional_channels.py` (decay/heating/dephasing
oracles, R8 non-commuting schedule, short-window-not-skipped).

---

## B. `mkdocs.yml` nav line (paste after the WP-01 review-note line)

```yaml
  - "Literature Review — Two-Mode & Motional": two-mode-motional-review.md
```

Place it immediately after the WP-01 review-note nav line
(`- "Literature Review — Estimation & Darwinism": estimation-darwinism-review.md`,
itself added by the EDF proposal §B), so the two review notes sit together in the
top-level nav. `mkdocs build --strict` warns until both lines are present.

---

## C. Seal-time edits — applied **with** the EDF package, once, at the v0.3 release

Done in the **single combined seal commit** (FREEZE-v0.3 §4 = Combined), together
with WP-01's `EDF-conventions-nav-proposal.md` §C. Following the §17/§18 staged →
frozen pattern:

1. **Seal §19–24 together.** Append the freeze line to each of §19–22 (per the
   EDF proposal) **and** §23–24, e.g.: `**§23 freeze.** Sections 23.1–23.4
   received a complete read-through for the Convention Freeze gate at the v0.3
   release. Post-freeze additions require a CONVENTIONS.md version bump.` Drop the
   `*(staged …)*` tags.
2. **Single bump** `CONVENTION_VERSION` 0.2 → 0.3 + add
   `tests/conventions/test_convention_version.py` (owned by FREEZE-v0.3 §3) —
   once for §19–24.
3. **Header block** → `**Scope:** Conventions covering §1–24`; freeze narrative
   names v0.3 and the added §19–24.
4. **Endorsement Marker** → restate: §17–18 closed under v0.2; **§19–24 closed
   under the v0.3 Convention Freeze**; §1–16 carry forward.
5. **Footer** → `**Convention version:** 0.3 · 2026-06-XX · v0.3 Convention Freeze.`
6. **`mkdocs.yml`** → both review-note nav lines (EDF §B + this §B).
7. **`WORKPLAN_v0.3.md`** → both dispatch-track stubs (WP-01 §5.4 + WP-02 §5.5),
   header/footer/Endorsement bumped in lock-step.

---

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with
externally validated laws. This is a Coastline proposal side-car within the
Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-
Universität Freiburg). It stages — does not apply — edits to the `CONVENTIONS.md`
and `mkdocs.yml` locks; the seal and bump are maintainer-governed acts owned by
`WP/FREEZE-v0.3.md`, applied together with the EDF package. Licensed under
**CC BY-SA 4.0**.

**Convention version:** stages `CONVENTIONS.md` §23–24 for the v0.3 freeze.
**Workplan reference:** `WP/WP-02-two-mode-motional.md` §6, §7; `WP/FREEZE-v0.3.md`.
