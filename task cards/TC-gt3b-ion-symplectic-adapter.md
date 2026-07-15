# TC-GT3b — Ion normal→local symplectic `S`-adapter (cross-repo handshake)

Version 0.1 · Drafted 2026-07-15 · **Status: Ratified** · Dispatch: **WP-07 GT3b** (already minted, open) · Family `GT`

> Propose-don't-apply. This card is a deliberation artifact; nothing here changes sealed conventions or code until ratified. It resolves **how** the ion-specific `S`-adapter is built and, above all, the **handshake protocol** between `uwarring82/iontrap-structure` (the mode producer) and `iontrap-dynamics` (the congruence consumer), so the structure-repo work proceeds **independently**.

## 1. Purpose

GT3a landed the **application-agnostic** congruence `congruence(S, V) = S V Sᵀ` in [gaussian.py](src/iontrap_dynamics/gaussian.py) (`1033f00`), with `S Ω Sᵀ = Ω` validated to a per-entry relative tolerance and ν preserved. It deliberately takes `S` as given and knows **nothing** about ions.

GT3b builds the **ion-specific `S`** — the canonical map `R_local = S R_normal` from normal-mode quadratures to local-ion quadratures — so that a covariance expressed in the normal-mode basis (where squeezing dynamics live) can be re-expressed in the **local-ion cut** and its entanglement read out with GT4 `log_negativity`. This is the dependency the **WP-SQ Phase B** two-ion consumer needs; it is the **first cross-repo dynamics consumer** of `iontrap-structure`.

Per the maintainer boundary (GT3a review): none of this machinery — masses, mode eigenvectors, local frequencies, `ModeConfig` coupling — may enter `gaussian.py`. GT3 stays **partial** until GT3b + its acceptance tests land.

## 2. The producer surface (as it exists today, `iontrap-structure` `main`)

`normal_modes(eq) → ModeResult` ([src/iontrap_structure/modes.py](https://github.com/uwarring82/iontrap-structure/blob/main/src/iontrap_structure/modes.py)) — validated by `tests/test_modeconfig_contract.py`:

| Field | Shape / units | Contract (tested) |
|---|---|---|
| `frequencies_rad_s` | `(n_modes,)`, rad·s⁻¹ | all `> 0` (raises on unstable equilibrium) |
| `eigenvectors` | `(n_modes, n_ions, 3)` | **mass-symmetrised** (eigenvectors of `D = M^{-1/2} H M^{-1/2}`); per-mode `Σ_i ‖b_{i,m}‖² = 1`; cross-mode Gram `= I` |
| `masses` | `(n_ions,)`, kg | species masses |
| `n_modes = 3·n_ions`, `n_ions` | — | properties |
| `to_mode_configs()` | `list[ModeConfig]` | per-mode `frequency_rad_s`, `eigenvector_per_ion`, `label`; **uses the consumer's `ModeConfig` parent class if importable** (already a test-pinned seam) |

**The critical convention:** `eigenvectors` are mass-symmetrised (orthonormal in the mass-metric), **not** raw displacement vectors — they coincide only for equal masses. Physical displacement is `u = M^{-1/2} · (eigenvector · normal-coord)`. Any `S` built from these must fold in `M^{-1/2}` and the per-mode `√(ℏ/2 m ω)` quadrature scale, or `S Ω Sᵀ = Ω` fails.

## 3. The handshake protocol (the core deliverable of this card)

The contract that lets the two repos evolve independently:

1. **Exchange object = an immutable `ModeConfig` record**, not a live import. `iontrap-structure` already emits the consumer's `ModeConfig` class when importable (the parent-class hook) — GT3b formalises that class as the **handshake schema** owned by whichever repo D1 decides. No hard runtime dependency in either direction beyond that record.
2. **Producer guarantees** (pinned by `iontrap-structure`'s own CI, independently): stable-equilibrium modes; the four fields above with their tested normalisation/orthonormality; mass-metric convention documented.
3. **Consumer guarantees** (pinned by `iontrap-dynamics`'s CI): given a conforming record, the adapter produces a **symplectic** `S` (verified by feeding it through GT3a `congruence`, whose `SΩSᵀ=Ω` gate is the contract check) and the resulting local-cut functionals (GT4 `log_negativity`) match the physics oracles (§6).
4. **Versioning:** the `ModeConfig` schema carries a version; a producer/consumer mismatch is a loud error, not a silent misread. (Mirrors the §-seal discipline for the cross-repo boundary.)

## 4. Repo boundary — what is done where (independence)

| Work item | Repo | Rationale |
|---|---|---|
| Guarantee/expose the mode data (complete basis, mass-metric orthonormality, ω, masses) | **`iontrap-structure`** (independent) | already produced + contract-tested; GT3b may request *additional* metadata (see D3) |
| The `ModeConfig` handshake schema (immutable record) | **per D1** | the shared boundary object |
| The normal→local `S` builder (mass + frequency weighting → symplectic `S`) | **`iontrap-dynamics` adapter module** (e.g. `ion_modes.py`, **not** `gaussian.py`) | needs the §27 phase-space convention, which lives here |
| Generic `congruence(S, V)` + `SΩSᵀ=Ω` guard | **`gaussian.py` — DONE (GT3a)** | application-agnostic; unchanged |
| Ion-cut `E_N` read-out | **`gaussian.py` GT4 `log_negativity` — DONE** | consumes the adapter's output |

## 5. Decisions to ratify

- **D1 — where the adapter (and the `ModeConfig` handshake class) lives.**
  - **(a) A new module in `iontrap-dynamics` consuming an immutable `ModeConfig` record (recommended).** The normal→local `S`-builder lives next to the congruence consumer and inside the repo that owns §27; `iontrap-structure` stays free of phase-space conventions and emits the record via its existing parent-class hook. The `ModeConfig` schema itself is the shared record shape — the producer owns the canonical fields it guarantees, the consumer owns the fallback-compatible class used when the producer is not importable.
  - (b) `iontrap-structure` emits `S` directly — rejected: forces §27 phase-space + ℏ/quadrature conventions into a *classical* structural-dynamics repo, breaking its scope and independence.
  - (c) A third bridge package — overkill for one consumer; revisit only if a second dynamics repo appears.
- **D2 — handshake mechanism.** (a) Immutable `ModeConfig` record via the existing importable-parent-class seam, **no hard dependency (recommended)** — matches `iontrap-structure`'s "immutable records / seam" design. (b) Hard `import iontrap_structure` dependency — heavier, couples release cadences; rejected unless the record proves insufficient.
- **D3 — the missing inputs the producer surface does not yet carry.** Building `S` needs (i) **local reference frequencies** (single-ion trap frequencies defining the *local* quadrature scale) and (ii) an explicit statement of the **mass-symmetrised → physical-displacement** weighting. Decide: does `ModeConfig` gain these fields (producer-side, independent work), or does the adapter take them as caller-supplied trap parameters? Recommendation: local reference frequencies are a **trap** property `iontrap-structure` already holds (`HarmonicTrap`) → expose them on the record; the weighting convention is documented in the handshake schema.

## 6. Acceptance (physics oracles)

- `S Ω Sᵀ = Ω` for every produced `S` — **already enforced** by GT3a `congruence` (the adapter's output is validated by construction when consumed).
- **Squeezed** normal-mode product → correct **ion-cut** `E_N` via GT4 `log_negativity`. (Thermal/classical normal modes give **zero** ion-cut entanglement — a 50:50-type mixing needs *squeezed* inputs, Kim 2002; a thermal oracle would falsely read separable.)
- **Non-orthogonal case:** COM vs stretch differ by √3 in participation — the adapter must reproduce the correct local-cut spectrum, not assume equal weighting.
- Mode correctness itself (James 1998, `M_ij`) is **`iontrap-structure`'s** acceptance, verified independently there.

## 7. Governance

- Dispatch **GT3b** (already minted, open). **Observable-only** — consumes sealed §27, **no convention bump** (`CONVENTION_VERSION` stays `0.6`). GT3 marked fully landed only when GT3b + acceptance tests land.
- **Ratification (2026-07-15).** Maintainer review ratified the deliberation with the recommended options: **D1(a)** adapter module in `iontrap-dynamics` consuming an immutable `ModeConfig` record; **D2(a)** handshake via the existing importable-parent-class seam with no hard dependency; **D3** local reference frequencies exposed on the producer record (structure-repo work, tracked independently). One clarifying edit: the schema is co-owned — `iontrap-structure` owns the canonical fields it guarantees, `iontrap-dynamics` owns the fallback-compatible class used when the producer is not importable.
- Both repos **MIT**; the immutable-record handshake keeps CI, licence, and release cadence independent per repo.
- This is the first `iontrap-structure` → `iontrap-dynamics` dynamics consumer; the handshake schema + versioning set the precedent for future exchanges.

## 8. References

- D. F. V. James, *Appl. Phys. B* **66**, 181 (1998) — ion-trap normal modes (`iontrap-structure` validation ref).
- M. S. Kim et al., *Phys. Rev. A* **65**, 032323 (2002) — beamsplitter entanglement requires non-classical (squeezed) inputs (the ion-cut `E_N` oracle caveat).
- Parent card [TC-gaussian-entanglement-toolbox.md](task%20cards/TC-gaussian-entanglement-toolbox.md) §C (normal→local congruence) + the GT3a/GT3b split rationale.
