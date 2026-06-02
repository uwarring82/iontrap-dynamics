# Estimation & Darwinism — Literature Review and Convention Anchors

This note fixes the canonical definitions behind the estimation and
quantum-Darwinism service surface (`iontrap_dynamics.information`,
`iontrap_dynamics.states`, `iontrap_dynamics.systematics.common_mode`) and
identifies the analytic oracles each generic benchmark reproduces. It is
deliberately **bounded**: it exists only to (a) pin the conventions adopted by
the library to primary sources, and (b) name the closed-form anchor each
benchmark is checked against. It is not a survey of the fields.

!!! note "Convention-anchor contract (binding)"
    Each definition below cites a primary source, and each numbered
    `CONVENTIONS.md` section in the estimation/Darwinism range (§19–22) cites
    this note as authoritative for its definition. The pairing is the audit
    trail: a reader can trace any library symbol → its CONVENTIONS section →
    the convention chosen here → the primary literature.

!!! note "Application-agnostic by construction"
    Every quantity here is defined on generic inputs — a state, a channel, a
    partition, a generator. The library carries no application framing, and
    neither does this note: the benchmarks are textbook-oracle validations, not
    domain demonstrations. See [Project Framework](framework.md) for the
    Coastline/Sail layering and [Benchmarks](benchmarks.md) for the runnable
    artefacts.

## 1. Scope and how to read this note

The four library capabilities and their convention anchors:

| Capability | Library surface | CONVENTIONS § (staged) | Benchmark(s) |
|---|---|---|---|
| Classical / quantum Fisher information + Cramér–Rao | `information.fisher` | §19 | 1 (QFI scaling), 2 (CFI linear-Gaussian) |
| Quantum-Darwinism redundancy + recoverability | `information.redundancy`, `information.recoverability` | §20 | 3 (Darwinism redundancy), 4 (recoverability) |
| GHZ / cat state factories | `states.ghz_state`, `states.cat_mode` | §21 | 5 (GHZ / cat) |
| Correlated common-mode channel | `systematics.common_mode` | §22 | 6 (common-mode) |

Mathematical notation follows `CONVENTIONS.md`: inline Unicode, entropies in
**bits** (base-2 logarithm), density operators ρ, and the partial trace written
ρ_X for the reduced state on subsystem X.

## 2. Quantum and classical Fisher information *(backs CONVENTIONS §19, WI-1)*

**Classical Fisher information (CFI).** For an outcome distribution p(x; θ)
depending on a scalar parameter θ, the classical Fisher information is

  F_cl(θ) = Σ_x p(x; θ) · (∂_θ ln p(x; θ))² = Σ_x (∂_θ p(x; θ))² / p(x; θ),

with the convention that outcomes of zero probability contribute zero (the
0/0 term is masked, not evaluated). This is the textbook score-variance
definition [Helstrom 1976; Paris 2009].

**Quantum Fisher information (QFI), SLD convention.** The library adopts the
**symmetric logarithmic derivative (SLD)** quantum Fisher information. The SLD
L_θ is the Hermitian operator solving the Lyapunov equation

  ∂_θ ρ_θ = ½ (L_θ ρ_θ + ρ_θ L_θ),

and the QFI is F_Q(θ) = Tr(ρ_θ L_θ²). This is the Braunstein–Caves convention,
in which the QFI is four times the squared Bures statistical-distance metric
[Braunstein & Caves 1994]. We adopt the SLD-QFI (not the right/left logarithmic
derivative or the Kubo–Mori variants) because it is the version saturating the
quantum Cramér–Rao bound for a single parameter and is standard across the
metrology literature [Paris 2009; Tóth & Apellaniz 2014].

**Cramér–Rao bound.** For an unbiased estimator from ν repetitions,
Var(θ̂) ≥ 1 / (ν · F), with F the relevant (classical or quantum) Fisher
information. The library exposes the single-shot bound 1/F; the ν-scaling is the
caller's [Helstrom 1976; Paris 2009].

**CFI ≤ QFI (saturation).** For any measurement, F_cl ≤ F_Q, with equality
attainable by the projective measurement in the SLD eigenbasis
[Braunstein & Caves 1994]. This inequality is a built-in numerical guard in the
library and the headline check of benchmark 2.

**Linear-Gaussian closed form.** For a linear model x = A·θ + noise with known
design matrix A and known covariance Σ ≻ 0, the Fisher information matrix is

  F = Aᵀ Σ⁻¹ A,

the standard Gaussian result [Paris 2009; and any estimation-theory text]. This
is the exact oracle of benchmark 2.

**Heisenberg vs standard quantum limit.** Under the collective generator
J_z = ½ Σ_i σ_z^(i), an N-qubit GHZ probe attains QFI_GHZ = N² (Heisenberg
scaling), whereas an uncorrelated product probe attains QFI_product = N (the
standard quantum limit) [Giovannetti, Lloyd & Maccone 2004; Bollinger et al.
1996]. The N² vs N separation is the single decoupling figure of benchmark 1
(the keystone).

!!! tip "Oracles identified for §7"
    - **Benchmark 1 (`qfi_scaling`)** — QFI_GHZ = N², QFI_product = N from
      closed forms under J_z.
    - **Benchmark 2 (`cfi_linear_gaussian`)** — F = AᵀΣ⁻¹A exact; CFI ≤ QFI
      saturation.

## 3. Quantum Darwinism — redundancy and recoverability *(backs CONVENTIONS §20, WI-2)*

**Fragment mutual information.** For a system S and an environment fragment F,
the quantum mutual information is

  I(S : F) = S(ρ_S) + S(ρ_F) − S(ρ_{S∪F}),

with S(·) the von Neumann entropy in bits. This is the quantity whose
fragment-size dependence carries the Darwinism signature
[Ollivier, Poulin & Zurek 2004; Zurek 2009].

**Partial-information plot and the redundancy convention.** Plotting
I(S : F) against fragment size f (averaged over fragments of that size) yields
the partial-information curve. The **classical plateau** sits at the Holevo
bound for the system's pointer information, ≈ H_S. The library adopts the
**information-deficit redundancy** convention

  R_δ = N / f_δ,

where f_δ is the smallest fragment fraction for which I(S : F) reaches
(1 − δ)·H_S, and N is the environment size; δ is the supplied information deficit
[Blume-Kohout & Zurek 2006; Zurek 2009; Riedel, Zurek & Zwolak 2012]. For the
GHZ-cascade model (one system qubit redundantly copied onto N environment qubits) the
mutual information is a step: it jumps to the full H_S as soon as a single
environment qubit is included, so the plateau equals H_S and R_δ = N — the exact
oracle of benchmark 3.

**Recoverability — the coherent-information measure.** The library quantifies
how much of the system is recoverable from an accessible set A by the
(clamped) **Schumacher–Nielsen coherent information**

  recoverability = max(0, S(ρ_A) − S(ρ_{S∪A}))   [bits],

i.e. the coherent information I_c(S⟩A) = S(ρ_A) − S(ρ_{S∪A}), floored at zero
[Schumacher & Nielsen 1996]. Positive coherent information is the standard
signal that quantum information about S survives in, and is recoverable from, A;
its connection to approximate recovery maps is made precise by the Petz recovery
map and the Fawzi–Renner bound [Petz 1986; Fawzi & Renner 2015; Bény & Oreshkov
2010], and it underlies the quantum error-correction conditions of
[Knill & Laflamme 1997]. We adopt the coherent-information form (rather than a
fidelity-of-recovery or a relative-entropy-of-recovery measure) because it is
computable directly from two reduced-state entropies, with no recovery-map
optimisation, and its endpoints are exact closed forms — see [Wilde 2017] for
the textbook treatment.

**Werner endpoint oracle.** On the two-qubit Werner family
ρ(p) = p·|Φ⁺⟩⟨Φ⁺| + (1 − p)·I/4, with S the first qubit and A the second, the
measure is 0 at p = 0 (the maximally mixed product state, full decoherence) and
H_S = 1 bit at p = 1 (the maximally entangled Bell pair, perfect recovery), and
is monotone non-decreasing in p between them — the oracle of benchmark 4.

!!! tip "Oracles identified for §7"
    - **Benchmark 3 (`darwinism_redundancy`)** — GHZ-cascade plateau
      I(S : F) = H_S; R_δ = N.
    - **Benchmark 4 (`recoverability`)** — Werner endpoints (0 at full
      decoherence, H_S at perfect recovery); monotone interior.

## 4. GHZ and cat state conventions *(backs CONVENTIONS §21, WI-3)*

**GHZ state.** The N-qubit Greenberger–Horne–Zeilinger state is
|GHZ_N⟩ = (|0…0⟩ + |1…1⟩)/√2 in the computational basis (the spin convention of
`CONVENTIONS.md` §3). Under the collective phase e^{−iφ J_z} the parity
observable oscillates at N times the single-qubit rate,

  ⟨X^⊗N⟩ = cos(N φ),

the N-fold-enhanced Ramsey fringe that underlies GHZ-based metrology
[Bollinger, Itano, Wineland & Heinzen 1996; Giovannetti, Lloyd & Maccone 2004].
This fringe is the oracle of benchmark 5.

**Cat state.** The even/odd Schrödinger-cat superpositions
|cat_±⟩ ∝ |α⟩ ± |−α⟩ of a single bosonic mode are eigenstates of the photon
parity operator with eigenvalue +1 (even) and −1 (odd) respectively. The library
factory builds the requested parity in a truncated Fock space; the exact ±1
parity is the companion check of benchmark 5.

!!! tip "Oracle identified for §7"
    - **Benchmark 5 (`ghz_cat`)** — GHZ parity fringe ⟨X^⊗N⟩ = cos(N φ); cat
      parity ±1.

## 5. Common-mode (shared-latent) channel *(backs CONVENTIONS §22, WI-4)*

**Channel definition.** The common-mode phase channel draws, for each of M
subsystems, an offset that mixes a shared latent with an independent
per-subsystem term to a tunable degree c = correlation ∈ [0, 1]:

  offset_i = √c · ξ_shared + √(1 − c) · ξ_i,   ξ_shared, ξ_i ~ 𝒩(0, σ²),

independent across the ξ. The marginal per-subsystem variance is σ² at every c,
while the **difference observable** offset_0 − offset_1 has variance

  Var(offset_0 − offset_1) = 2 σ² (1 − c),

by the elementary covariance identity for a difference of jointly Gaussian
variables with correlation coefficient c. At c = 0 the offsets are independent
and the difference jitter is the full 2σ² (incoherent sum of two independent
draws); at c = 1 the shared latent cancels exactly and the difference variance
is 0 — **common-mode rejection**, the standard motivation for differential
measurement in precision spectroscopy. The variance is monotone decreasing in c
between the endpoints. This closed form is the oracle of benchmark 6.

!!! note "Primary source is the statistical identity"
    The channel convention rests on the textbook variance-of-correlated-Gaussians
    identity; the common-mode-rejection limit (c = 1) is the elementary cancellation
    Var(X − X) = 0. It is cited here only to fix the **generic** channel definition —
    no domain-specific noise model is implied.

!!! tip "Oracle identified for §7"
    - **Benchmark 6 (`common_mode`)** — difference variance 2σ²(1 − c); exact
      rejection at c = 1.

## 6. Source matrix

| Convention chosen | Where it lands | Primary source(s) |
|---|---|---|
| SLD-QFI, F_Q = Tr(ρ L²) | CONVENTIONS §19; `information.fisher` | Braunstein & Caves 1994; Paris 2009; Tóth & Apellaniz 2014 |
| CFI score-variance; CFI ≤ QFI | CONVENTIONS §19; benchmark 2 | Helstrom 1976; Braunstein & Caves 1994 |
| Cramér–Rao bound 1/F | CONVENTIONS §19; `cramer_rao_bound` | Helstrom 1976; Paris 2009 |
| Linear-Gaussian F = AᵀΣ⁻¹A | CONVENTIONS §19; benchmark 2 | Paris 2009 (Gaussian estimation) |
| QFI_GHZ = N² / QFI_product = N | CONVENTIONS §19/§21; benchmark 1 | Giovannetti, Lloyd & Maccone 2004; Bollinger et al. 1996 |
| Mutual information I(S:F) | CONVENTIONS §20; `information.redundancy` | Ollivier, Poulin & Zurek 2004; Zurek 2009 |
| Redundancy R_δ = N/f_δ (deficit δ) | CONVENTIONS §20; benchmark 3 | Blume-Kohout & Zurek 2006; Riedel, Zurek & Zwolak 2012 |
| Recoverability = max(0, S(ρ_A) − S(ρ_{S∪A})) | CONVENTIONS §20; `information.recoverability` | Schumacher & Nielsen 1996; Petz 1986; Fawzi & Renner 2015 |
| GHZ parity fringe ⟨X^⊗N⟩ = cos(Nφ) | CONVENTIONS §21; benchmark 5 | Bollinger et al. 1996 |
| Cat parity eigenvalue ±1 | CONVENTIONS §21; `states.cat_mode` | standard bosonic-parity convention (see Wilde 2017) |
| Common-mode Var = 2σ²(1−c) | CONVENTIONS §22; benchmark 6 | textbook covariance identity |
| QuTiP backend | all WIs | Johansson, Nation & Nori 2012 |

## References

1. **Helstrom, C. W.** (1976). *Quantum Detection and Estimation Theory.*
   Academic Press, New York.
2. **Braunstein, S. L. & Caves, C. M.** (1994). Statistical distance and the
   geometry of quantum states. *Phys. Rev. Lett.* **72**, 3439.
   <https://doi.org/10.1103/PhysRevLett.72.3439>
3. **Paris, M. G. A.** (2009). Quantum estimation for quantum technology.
   *Int. J. Quantum Inf.* **7**, 125.
   <https://doi.org/10.1142/S0219749909004839>
4. **Tóth, G. & Apellaniz, I.** (2014). Quantum metrology from a quantum
   information science perspective. *J. Phys. A: Math. Theor.* **47**, 424006.
   <https://doi.org/10.1088/1751-8113/47/42/424006>
5. **Giovannetti, V., Lloyd, S. & Maccone, L.** (2004). Quantum-enhanced
   measurements: beating the standard quantum limit. *Science* **306**, 1330.
   <https://doi.org/10.1126/science.1104149>
6. **Bollinger, J. J., Itano, W. M., Wineland, D. J. & Heinzen, D. J.** (1996).
   Optimal frequency measurements with maximally correlated states.
   *Phys. Rev. A* **54**, R4649.
   <https://doi.org/10.1103/PhysRevA.54.R4649>
7. **Ollivier, H., Poulin, D. & Zurek, W. H.** (2004). Objective properties from
   subjective quantum states: environment as a witness. *Phys. Rev. Lett.*
   **93**, 220401. <https://doi.org/10.1103/PhysRevLett.93.220401>
8. **Zurek, W. H.** (2009). Quantum Darwinism. *Nat. Phys.* **5**, 181.
   <https://doi.org/10.1038/nphys1202>
9. **Blume-Kohout, R. & Zurek, W. H.** (2006). Quantum Darwinism: entanglement,
   branches, and the emergent classicality of redundantly stored quantum
   information. *Phys. Rev. A* **73**, 062310.
   <https://doi.org/10.1103/PhysRevA.73.062310>
10. **Riedel, C. J., Zurek, W. H. & Zwolak, M.** (2012). The rise and fall of
    redundancy in decoherence and quantum Darwinism. *New J. Phys.* **14**,
    083010. <https://doi.org/10.1088/1367-2630/14/8/083010>
11. **Schumacher, B. & Nielsen, M. A.** (1996). Quantum data processing and
    error correction. *Phys. Rev. A* **54**, 2629.
    <https://doi.org/10.1103/PhysRevA.54.2629>
12. **Knill, E. & Laflamme, R.** (1997). Theory of quantum error-correcting
    codes. *Phys. Rev. A* **55**, 900.
    <https://doi.org/10.1103/PhysRevA.55.900>
13. **Petz, D.** (1986). Sufficient subalgebras and the relative entropy of
    states of a von Neumann algebra. *Comm. Math. Phys.* **105**, 123.
    <https://doi.org/10.1007/BF01212345>
14. **Bény, C. & Oreshkov, O.** (2010). General conditions for approximate
    quantum error correction and near-optimal recovery channels.
    *Phys. Rev. Lett.* **104**, 120501.
    <https://doi.org/10.1103/PhysRevLett.104.120501>
15. **Fawzi, O. & Renner, R.** (2015). Quantum conditional mutual information and
    approximate Markov chains. *Comm. Math. Phys.* **340**, 575.
    <https://doi.org/10.1007/s00220-015-2466-x>
16. **Wilde, M. M.** (2017). *Quantum Information Theory* (2nd ed.). Cambridge
    University Press. <https://doi.org/10.1017/9781316809976>
17. **Johansson, J. R., Nation, P. D. & Nori, F.** (2012). QuTiP: an open-source
    Python framework for the dynamics of open quantum systems.
    *Comput. Phys. Commun.* **183**, 1760.
    <https://doi.org/10.1016/j.cpc.2012.02.021>

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with
externally validated laws. This is a Coastline review note within the
Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-
Universität Freiburg). It fixes the conventions binding `iontrap_dynamics` at
this version and is the cited authority for `CONVENTIONS.md` §19–22; it carries
no external endorsement. Licensed under **CC BY-SA 4.0** — Coastline material is
a shareable constraint, not adaptive Sail guidance.

**Convention version:** staged for the v0.3 Convention Freeze (§19–22).
**Workplan reference:** `WP/WP-01-estimation-darwinism.md` §5.
