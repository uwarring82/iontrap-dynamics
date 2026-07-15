# TC-GT3b — Ion normal→local symplectic `S`-adapter (cross-repo handshake)

Version 0.3 · Drafted 2026-07-15 · Ratified 2026-07-15 · **Status: Ratified (v0.3).** D1 ratified; D2 ratified around the versioned `IonModeBasis` payload; D3 open and non-blocking (tagged local-coordinate gauge). · Dispatch: **WP-07 GT3b** (minted, open) · Family `GT`

> Propose-don't-apply. This card is a deliberation artifact; nothing here changes sealed conventions or code until ratified. It resolves **how** the ion-specific `S`-adapter is built and, above all, the **handshake protocol** between `uwarring82/iontrap-structure` (mode producer) and `iontrap-dynamics` (congruence consumer), so the structure-repo work proceeds **independently**.
>
> **Review corrections folded in (v0.2 → v0.3):** exchange object is a **new versioned basis payload** (`IonModeBasis`), not `ModeConfig` (D2); explicit **mass/frequency derivation** with the §27 `√2` quadrature convention and the exact block form of `S` (§3); corrected **COM/stretch** oracle — equal participation `1/√2`, *frequency* ratio `√3`, weights `3^{±1/4}` (§7.5); **thermal⇒separable** demoted to the passive limit only (§7.2); **D3 is a local gauge** that does not change ion-cut `E_N`, so it does not block GT3b (§6); **payload wire invariants** specified and **single schema-version authority** assigned to `iontrap-dynamics` (§4); coupled-ground-state oracle qualified as a *selected* case (§7.3).

## 1. Purpose

GT3a landed the **application-agnostic** congruence `congruence(S, V) = S V Sᵀ` in [gaussian.py](src/iontrap_dynamics/gaussian.py) (`1033f00`), with `S Ω Sᵀ = Ω` validated to a per-entry relative tolerance and ν preserved. It takes `S` as given and knows **nothing** about ions.

GT3b builds the **ion-specific `S`** — the canonical map `R_local = S R_normal` from normal-mode quadratures to local-ion quadratures — so a covariance expressed in the normal-mode basis (where squeezing dynamics live) can be re-expressed in the **local-ion cut** and its entanglement read out with GT4 `log_negativity`. This is the dependency the **WP-SQ Phase B** two-ion consumer needs; the **first cross-repo dynamics consumer** of `iontrap-structure`.

Per the GT3a boundary: none of this machinery — masses, mode eigenvectors, local frequencies, basis coupling — may enter `gaussian.py`. GT3 stays **partial** until GT3b + its acceptance tests land.

## 2. The producer surface (as it exists today, `iontrap-structure` `main`)

`normal_modes(eq) → ModeResult` ([modes.py](https://github.com/uwarring82/iontrap-structure/blob/main/src/iontrap_structure/modes.py)), validated by `tests/test_modeconfig_contract.py`:

| Field | Shape / units | Contract (tested) |
|---|---|---|
| `frequencies_rad_s` | `(3N,)`, rad·s⁻¹ | all `> 0` (raises on unstable equilibrium) — the `ω_m` |
| `eigenvectors` | `(3N, N, 3)` | **mass-symmetrised** `B` (eigenvectors of `D = M^{-1/2} H M^{-1/2}`); per-mode `Σ‖b‖² = 1`; cross-mode Gram `= I` |
| `masses` | `(N,)`, kg | species masses `m_j` |
| `to_mode_configs()` | `list[ModeConfig]` | per-mode record (freq + one eigenvector + label); **importable-parent-class seam** — legacy per-mode consumers only, **not** the GT3b handshake |

**Convention:** `B` is mass-symmetrised (orthonormal in the mass-metric), **not** raw displacement — the two coincide only for equal masses. `B` acts in **mass-weighted** coordinates; the physical-displacement map `u = M^{-1/2}(B q)` contributes the `M^{-1/2}`, and the mass **cancels only after** conversion to local dimensionless quadratures (§3). The adapter therefore builds `S` from the §3 result directly — it must **not** additionally apply `M^{-1/2}`.

## 3. The normal→local map `S` (derivation + block form)

Physical displacement of coordinate `j` (ion, axis) in mass-weighted normal coordinates `q_m` (effective mass 1, frequency `ω_m`), with the physical-displacement map carrying `M^{-1/2}`:

```
u_j = Σ_m B_{jm} q_m / √m_j
```

Convert to **dimensionless §27 quadratures** — §27 fixes **vacuum variance 1**, so `x̂ = √(2ω/ℏ) q̂`, `p̂ = √(2/(ℏω)) π̂`. Thus normal mode `m`: `x_normal,m = √(2ω_m/ℏ) q_m`; local coordinate `j` (reference frequency `ω_local,j`, mass `m_j`): `x_local,j = √(2 m_j ω_local,j/ℏ) u_j`. Substituting, **the `√2`, `√m_j`, and `ℏ` all cancel**:

```
x_local,j = Σ_m B_{jm} √(ω_local,j / ω_m) · x_normal,m
p_local,j = Σ_m B_{jm} √(ω_m / ω_local,j) · p_normal,m
```

So in the block ordering `R = (x_1…x_{3N}, p_1…p_{3N})` the map is `S = diag(X, P)` with

```
X_{jm} = B_{jm} √(ω_local,j / ω_m),   P_{jm} = B_{jm} √(ω_m / ω_local,j).
```

The `√2`/`ℏ`/mass convention factors cancel from `X`, `P`, so the adapter is unchanged — but the derivation now **agrees with the sealed §27 convention**. **Symplecticity** (block `Ω = [[0, I],[-I, 0]]`): `S Ω Sᵀ = Ω ⟺ X Pᵀ = I`, and `(X Pᵀ)_{jk} = √(ω_local,j/ω_local,k) Σ_m B_{jm} B_{km} = √(ω_local,j/ω_local,k) δ_{jk} = δ_{jk}` (rows of `B` orthonormal). ✓ — `S` is symplectic and **mass-free**.

`X = P` (a passive rotation) **only** when `B` mixes solely modes of a **common frequency** matched by `ω_local` — i.e. for every nonzero `B_{jm}`, `ω_local,j = ω_m` (a global common frequency, or `B` confined to equal-frequency/degenerate subspaces with `ω_local` set to that value). Otherwise `X ≠ P` and the map is an **active (squeezing)** transformation (§7.2–7.3). The adapter emits `S` permuted into §27 per-mode ordering `(x̂₁,p̂₁,…)`.

## 4. The handshake protocol (core deliverable)

The exchange object is a **new, versioned, serialization-neutral basis payload** (`IonModeBasis` / `ModeBasisPayload`) — arrays + metadata, no shared runtime class, so neither repo type-couples to the other:

| Field | Shape / units | Meaning |
|---|---|---|
| `schema_version` | int/str | loud mismatch, not silent misread |
| `frequencies_rad_s` | `(3N,)` rad·s⁻¹ | mode frequencies `ω_m` |
| `mass_weighted_eigenvectors` | `(3N, 3N)` | `B`, with **explicit row (coordinate = ion×axis) / column (mode) ordering** |
| `masses_kg` | `(N,)` | `m_j` |
| `coordinate_frame` | tag | trap-frame axis order + handedness + ion index order |
| `local_reference_frequencies_rad_s` | `(3N,)` | `ω_local,j` (tagged gauge, **D3**) |
| `normalization_weighting_tags` | tags | mass-symmetrised / per-mode-unit-norm, etc. |

**Wire invariants** the payload must fix **explicitly** (so producer and consumer agree bit-for-bit): exact **axis ordering** + per-ion coordinate labels; trap-frame **orientation and handedness**; **frequency↔column alignment** (column `m` of `B` ↔ `frequencies_rad_s[m]`); **mode ordering**; the **`schema_version`** identifier; and the treatment of the residual **eigenvector-sign** and **degenerate-subspace rotation** freedom (canonicalised or explicitly tagged).

- **Producer guarantees** (`iontrap-structure` CI, independent): stable-equilibrium modes; the arrays above with their tested normalisation/orthonormality; conventions named in the tags — i.e. the **physical meaning and correctness** of the exported fields.
- **Consumer materializes its own immutable record.** `iontrap-dynamics` **validates** the payload (shapes, `ω>0`, Gram `= I`, tag + wire-invariant match) and builds a local frozen object — **no import of a producer class**. The `to_mode_configs()` / importable-`ModeConfig` seam **remains for legacy per-mode consumers**, not the GT3b handshake.
- **Schema authority (one owner).** `iontrap-dynamics` **owns and versions the consumer contract** (`schema_version`, field shapes/ordering, wire invariants); `iontrap-structure` owns the **physical correctness** of the fields. (Conceptually co-owned, but a single versioning authority avoids drift.) A `schema_version` mismatch is a hard error.
- **Consumer acceptance:** the built `S` passes GT3a `congruence`'s `SΩSᵀ=Ω` gate (the contract check) and the local-cut functionals match the §7 oracles.

## 5. Repo boundary — what is done where (independence)

| Work item | Repo | Rationale |
|---|---|---|
| Mode data (complete `3N` basis, mass-metric orthonormality, `ω`, masses) + `local_reference_frequencies` field | **`iontrap-structure`** (independent) | already produced + contract-tested; adds the payload emitter + local-freq field |
| The `IonModeBasis` **payload contract + `schema_version`** | **`iontrap-dynamics`** (consumer-owned, §4) | single versioning authority; producer owns field *correctness* |
| The normal→local `S` builder (§3) + payload validation/materialization | **`iontrap-dynamics` adapter module** (e.g. `ion_modes.py`, **not** `gaussian.py`) | needs the §27 phase-space convention, which lives here |
| Generic `congruence(S, V)` + `SΩSᵀ=Ω` guard | **`gaussian.py` — DONE (GT3a)** | application-agnostic; unchanged |
| Ion-cut `E_N` read-out | **`gaussian.py` GT4 `log_negativity` — DONE** | consumes the adapter's output |

## 6. Decisions

- **D1 — adapter location. RATIFIED (a):** a new module in `iontrap-dynamics`; the `S`-builder lives next to the congruence consumer, inside the repo that owns §27. `iontrap-structure` stays free of phase-space conventions.
- **D2 — handshake object. RATIFIED (v0.3): a new versioned `IonModeBasis` payload (§4), not `ModeConfig`.** A per-mode `ModeConfig` cannot carry the complete `3N` basis + ordering + masses + local frequencies + version, and emitting the consumer's class is runtime type coupling. The serialization-neutral payload keeps the repos release-independent; the `ModeConfig` seam is retained for legacy per-mode consumers only. Schema/version authority: **`iontrap-dynamics`** owns the consumer contract; `iontrap-structure` owns field correctness.
- **D3 — "local reference frequency". OPEN, but does NOT block GT3b.** `ω_local,j` is an explicit **local-coordinate gauge**: changing the positive `ω_local,j` is a product of **local single-mode squeezings** (per ion-coordinate), so it changes the covariance representation, occupation, and `T_eff` — but **not** entanglement across an ion partition (ion-cut `E_N` is invariant under local symplectic maps). GT3b proceeds with `ω_local` as a **tagged gauge**; a canonical default — candidates **(i)** bare single-ion **trap curvature** (`HarmonicTrap`, no Coulomb); **(ii)** the **diagonal local Hessian** `√(H_jj/m_j)` at equilibrium; **(iii)** another reference — is decided separately and recorded in the payload tag.

## 7. Acceptance (physics oracles)

1. **Canonicality:** every produced `S` satisfies `S Ω Sᵀ = Ω` (GT3a `congruence` gate) — pinned for equal- and **mixed-mass** configs.
2. **Passive limit** — `B` mixing only a **common-frequency (or degenerate) subspace** with `ω_local` set to that frequency, so `X = P = B` (a rotation): a **thermal** normal-mode product maps to a **separable** local state (the regime where Kim 2002's passive-mixing result applies — a rotation cannot entangle classical inputs).
3. **Selected coupled two-ion ground state** — a **specific stable, nonzero-coupling** configuration (not a universal claim): map the normal-mode vacuum through `S`; compare the local-cut covariance / `log_negativity` against an **independent quadratic-Hamiltonian ground-state** calculation. For this case the active map (`X ≠ P`) yields nonzero ion-cut `E_N` from the normal-mode vacuum; this does not contradict the passive-mixing result.
4. **Squeezed normal input:** compare against an **analytic two-mode covariance** propagation.
5. **Equal-mass COM/stretch:** pin `ω_stretch/ω_COM = √3` (participation magnitudes both `1/√2`) and the resulting `3^{±1/4}` quadrature weights in `X`, `P`.
6. **Mixed-mass:** pin canonicality **and** an independent position/momentum covariance oracle (the mass-cancellation of §3 must hold for unequal `m_j`).
7. **Gauge invariance:** rescaling `ω_local` (a local squeeze) leaves ion-cut `log_negativity` unchanged while changing occupation / `T_eff` (pins D3's gauge claim).

Mode correctness itself (James 1998, `M_ij`) is **`iontrap-structure`'s** acceptance, verified independently there.

## 8. Governance

- **Ratification (v0.3, 2026-07-15, over v0.1 `3aee26e`):** **D1** (adapter module in `iontrap-dynamics`) and **D2** (versioned `IonModeBasis` handshake payload, consumer-owned schema) **ratified**; **D3** (local reference frequency) **open, non-blocking** as a tagged local-coordinate gauge. GT3b implementation may proceed against this contract.
- Dispatch **GT3b** (minted, open). **Observable-only** — consumes sealed §27, **no convention bump** (`CONVENTION_VERSION` stays `0.6`). GT3 fully landed only when GT3b + acceptance tests land.
- Both repos **MIT**; the serialization-neutral payload keeps CI, licence, and release cadence independent per repo.
- First `iontrap-structure` → `iontrap-dynamics` dynamics consumer; the payload schema + versioning (owned by `iontrap-dynamics`) set the precedent for future exchanges.

## 9. References

- D. F. V. James, *Appl. Phys. B* **66**, 181 (1998) — ion-trap normal modes (`iontrap-structure` validation ref).
- M. S. Kim et al., *Phys. Rev. A* **65**, 032323 (2002) — beamsplitter entanglement requires non-classical inputs. **Scope:** applies to *passive* mixing under common normalisation (§7.2), **not** the general active GT3b map (§3, §7.3).
- Parent card [TC-gaussian-entanglement-toolbox.md](task%20cards/TC-gaussian-entanglement-toolbox.md) §C + the GT3a/GT3b split.
