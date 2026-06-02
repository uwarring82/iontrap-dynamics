# Two-Mode Squeezing & Motional Open-System — Literature Review and Convention Anchors

This note fixes the canonical definitions behind the two-mode SU(1,1) squeezing
surface (`iontrap_dynamics.hamiltonians.two_mode_squeezing_hamiltonian` /
`beamsplitter_hamiltonian`, `iontrap_dynamics.states.two_mode_squeezed_vacuum`)
and the typed motional CPTP channels (`iontrap_dynamics.channels`), and
identifies the analytic oracle each generic benchmark reproduces. It is
deliberately **bounded** (like the WP-01 estimation/Darwinism note): it pins the
conventions the library adopts to their sources in the literature, and names the
closed-form anchors — it is not a survey.

!!! note "Convention-anchor contract (binding)"
    Each definition below is anchored to the literature — a primary source where
    one exists, or the standard textbook treatment / elementary identity for
    standard results — and the staged `CONVENTIONS.md` sections **§23** (two-mode
    squeezing) and **§24** (motional CPTP channels) cite this note as
    authoritative for their definitions. The pairing is the audit trail: a reader
    traces any library symbol → its CONVENTIONS section → the convention here →
    the cited literature.

!!! note "Application-agnostic by construction"
    Every quantity here is defined on generic spin-motion inputs — a labelled
    mode pair, a typed channel, a probe state. The library carries no consuming-
    application framing, and neither does this note. See
    [Project Framework](framework.md) for the Coastline/Sail layering and
    [Benchmarks](benchmarks.md) for the runnable artefacts.

## 1. Scope and how to read this note

| Capability | Library surface | CONVENTIONS § (staged) |
|---|---|---|
| Two-mode parametric (SU(1,1)) Hamiltonian + beamsplitter | `hamiltonians.two_mode_squeezing_hamiltonian`, `beamsplitter_hamiltonian` | §23 |
| Two-mode squeezed-vacuum factory | `states.two_mode_squeezed_vacuum` | §23 |
| Typed motional CPTP channels (damping / heating / dephasing) + windowing | `channels` (`AmplitudeDamping`, `Heating`, `Dephasing`) | §24 |

Mathematical notation follows `CONVENTIONS.md`: inline Unicode, mode operators
â, b̂ on a labelled pair, the number operator n̂ = â†â, and SI seconds / rad·s⁻¹
for times and rates.

## 2. Two-mode squeezing and the SU(1,1) algebra *(backs CONVENTIONS §23)*

**Two-mode squeeze operator and convention.** The library adopts the two-mode
squeeze operator

  S₂(z) = exp(z* âb̂ − z â†b̂†),   z = r·e^{iθ},

the two-mode partner of the single-mode squeeze (§6) — **without** the
single-mode factor of ½ and π-period phase, because the two distinct modes carry
one excitation each per pair. The two-mode squeezed states are introduced in
[Caves & Schumaker 1985] (Part I); the explicit two-mode squeeze operator and the
compact two-mode notation are developed in [Schumaker & Caves 1985] (Part II).
Acting on the two-mode vacuum it gives the **two-mode squeezed vacuum**

  |z⟩ = S₂(z)|0,0⟩ = Σₙ cₙ |n, n⟩,   |cₙ| = tanhⁿr / cosh r,

so only equal-occupation Schmidt components |n,n⟩ are populated and the per-mode
mean occupation is

  ⟨n̂_a⟩ = ⟨n̂_b⟩ = sinh²|z|

[Caves & Schumaker 1985; Barnett & Radmore 2002 (operator ordering); Gerry &
Knight 2005 (Schmidt form)]. We define this **explicitly** rather than delegate
to `qutip.squeezing`, whose ½-convention yields sinh²(|z|/2) instead — a hidden
mismatch with the §6 single-mode parameterisation.

**SU(1,1) algebra and the conserved Casimir.** The pair generators

  K̂₊ = â†b̂†,  K̂₋ = âb̂,  K̂₀ = ½(n̂_a + n̂_b + 1)

close the su(1,1) algebra, [K̂₀, K̂±] = ±K̂±, [K̂₋, K̂₊] = 2K̂₀ [Yurke, McCall &
Klauder 1986]. The squeezing interaction creates and annihilates excitations in
**pairs**, so it commutes with the difference number n̂_a − n̂_b — the conserved
Casimir label; a state seeded from the vacuum keeps ⟨n̂_a⟩ = ⟨n̂_b⟩ at all times.

**Hamiltonian convention and the phase/sign map.** The library's generator is

  H_TMS/ℏ = i g (e^{iφ} â†b̂† − e^{−iφ} âb̂),

Hermitian, with g the parametric coupling (rad·s⁻¹) and φ the squeezing phase.
Evolving the vacuum for a time τ gives the two-mode squeezed vacuum with the
**signed complex** parameter z = −gτ·e^{iφ} (magnitude r = |z| = |g|τ), per-mode
n̂ = sinh²(gτ) = sinh²(|g|τ). g may be negative (a π phase shift). This phase/sign/ordering
choice is exactly what §23 fixes (the convention freeze gate).

**Beamsplitter (SU(2)).** The secondary

  H_BS/ℏ = J (e^{iφ} â†b̂ + e^{−iφ} âb̂†)

is the SU(2) partner [Yurke, McCall & Klauder 1986]: it conserves the total
occupation n̂_a + n̂_b (excitation only rotates between the modes).

!!! tip "Oracles identified (WI-1/WI-2)"
    - Factory per-mode ⟨n̂⟩ = sinh²|z|; Schmidt support on |n,n⟩ only.
    - H_TMS evolves the vacuum to ⟨n̂⟩ = sinh²(gτ) with n̂_a − n̂_b conserved, and
      matches the factory at z = −gτ·e^{iφ}.
    - H_BS conserves n̂_a + n̂_b.

## 3. Motional CPTP channels and the Lindblad master equation *(backs CONVENTIONS §24)*

**Lindblad form.** The library's typed dissipators enter the Markovian
master equation in Lindblad (GKSL) form

  dρ/dt = −(i/ℏ)[H, ρ] + Σ_k ( L_k ρ L_k† − ½{L_k†L_k, ρ} ),

the most general generator of a completely-positive trace-preserving (CPTP)
dynamical semigroup [Lindblad 1976; Gorini, Kossakowski & Sudarshan 1976; Breuer
& Petruccione 2002]. Each channel contributes one or two collapse operators L_k
on a labelled mode, with rates in s⁻¹; routed into `solve()` they are passed to
QuTiP's `mesolve` (no channels → the path is unchanged).

**The three motional dissipators.**

- **Amplitude damping** (zero temperature): L = √κ·â. Drives the mode to the
  ground state, ⟨n̂(t)⟩ = ⟨n̂(0)⟩·e^{−κt} — the bosonic amplitude-damping channel
  [Nielsen & Chuang 2010; Breuer & Petruccione 2002].
- **Heating** (finite bath n̄): L₋ = √(κ(n̄+1))·â and L₊ = √(κn̄)·â†. The mode
  relaxes to the bath, ⟨n̂(t)⟩ = n̄ + (⟨n̂(0)⟩ − n̄)·e^{−κt} → n̄. This is the
  standard model of trapped-ion **anomalous heating** [Turchette et al. 2000;
  Brownnutt, Kumph, Rabl & Blatt 2015; Wineland et al. 1998].
- **Pure dephasing**: L = √γ·n̂. Decoheres off-diagonal Fock coherences while
  leaving ⟨n̂⟩ unchanged; the coherence ρ_{nm} decays as e^{−(γ/2)(n−m)²t}
  [Breuer & Petruccione 2002].

`Depolarising` is **not** provided: it is canonical for finite-dimensional
systems, not for a single bosonic mode, so it is deferred rather than given an
arbitrary truncated-Fock definition.

**Sequence-aware (time-windowed) application.** A channel may carry a window
[t0, t1) over which its dissipation is active, realised with QuTiP's
time-dependent `[L, coeff]` format. What is order-dependent here is the
**temporal schedule** — a dissipator active on [0, T/2] followed by another on
[T/2, T] gives a different state from the reverse schedule, because the Lindblad
generators do not commute in general (the **R8** boundary; the library does not
assume they do). Channels active over the *same* window are simultaneous Lindblad
terms — their order is irrelevant; only *when* each acts matters.

!!! tip "Oracles identified (WI-3)"
    - amplitude damping ⟨n̂⟩ = n₀e^{−κt}; heating ⟨n̂⟩ → n̄; dephasing ⟨n̂⟩ constant
      with coherence ⟨X̂⟩ = X₀e^{−γt/2};
    - CPTP trace preservation; **R8**: a non-commuting two-window schedule is
      order-dependent.

## 4. Source matrix

| Convention chosen | Where it lands | Primary source(s) |
|---|---|---|
| Two-mode squeeze S₂(z) = exp(z*âb̂ − zâ†b̂†); per-mode sinh²\|z\| | §23; `two_mode_squeezed_vacuum` | Caves & Schumaker 1985; Schumaker & Caves 1985 |
| su(1,1) generators; n̂_a − n̂_b conserved (Casimir) | §23; `two_mode_squeezing_hamiltonian` | Yurke, McCall & Klauder 1986 |
| Schmidt form Σ cₙ\|n,n⟩; operator ordering | §23 | Barnett & Radmore 2002; Gerry & Knight 2005 |
| Beamsplitter SU(2); n̂_a + n̂_b conserved | §23; `beamsplitter_hamiltonian` | Yurke, McCall & Klauder 1986 |
| Lindblad (GKSL) CPTP generator | §24; `channels` → `solve` | Lindblad 1976; Gorini, Kossakowski & Sudarshan 1976; Breuer & Petruccione 2002 |
| Amplitude-damping / dephasing collapse operators | §24 | Nielsen & Chuang 2010; Breuer & Petruccione 2002 |
| Thermal-`â`/`â†` anomalous-heating model | §24; `Heating` | Turchette et al. 2000; Brownnutt et al. 2015 |
| QuTiP backend | all | Johansson, Nation & Nori 2012 |

## References

1. **Caves, C. M. & Schumaker, B. L.** (1985). New formalism for two-photon
   quantum optics. I. Quadrature phases and squeezed states. *Phys. Rev. A*
   **31**, 3068. <https://doi.org/10.1103/PhysRevA.31.3068>
2. **Schumaker, B. L. & Caves, C. M.** (1985). New formalism for two-photon
   quantum optics. II. Mathematical foundation and compact notation.
   *Phys. Rev. A* **31**, 3093. <https://doi.org/10.1103/PhysRevA.31.3093>
3. **Yurke, B., McCall, S. L. & Klauder, J. R.** (1986). SU(2) and SU(1,1)
   interferometers. *Phys. Rev. A* **33**, 4033.
   <https://doi.org/10.1103/PhysRevA.33.4033>
4. **Barnett, S. M. & Radmore, P. M.** (2002). *Methods in Theoretical Quantum
   Optics.* Oxford University Press.
   <https://doi.org/10.1093/acprof:oso/9780198563617.001.0001>
5. **Gerry, C. C. & Knight, P. L.** (2005). *Introductory Quantum Optics.*
   Cambridge University Press. <https://doi.org/10.1017/CBO9780511791239>
6. **Lindblad, G.** (1976). On the generators of quantum dynamical semigroups.
   *Comm. Math. Phys.* **48**, 119. <https://doi.org/10.1007/BF01608499>
7. **Gorini, V., Kossakowski, A. & Sudarshan, E. C. G.** (1976). Completely
   positive dynamical semigroups of N-level systems. *J. Math. Phys.* **17**,
   821. <https://doi.org/10.1063/1.522979>
8. **Breuer, H.-P. & Petruccione, F.** (2002). *The Theory of Open Quantum
   Systems.* Oxford University Press.
   <https://doi.org/10.1093/acprof:oso/9780199213900.001.0001>
9. **Nielsen, M. A. & Chuang, I. L.** (2010). *Quantum Computation and Quantum
   Information* (10th anniversary ed.). Cambridge University Press.
   <https://doi.org/10.1017/CBO9780511976667>
10. **Turchette, Q. A. et al.** (2000). Heating of trapped ions from the quantum
    ground state. *Phys. Rev. A* **61**, 063418.
    <https://doi.org/10.1103/PhysRevA.61.063418>
11. **Brownnutt, M., Kumph, M., Rabl, P. & Blatt, R.** (2015). Ion-trap
    measurements of electric-field noise near surfaces. *Rev. Mod. Phys.* **87**,
    1419. <https://doi.org/10.1103/RevModPhys.87.1419>
12. **Wineland, D. J. et al.** (1998). Experimental issues in coherent
    quantum-state manipulation of trapped atomic ions. *J. Res. Natl. Inst.
    Stand. Technol.* **103**, 259. <https://doi.org/10.6028/jres.103.019>
13. **Johansson, J. R., Nation, P. D. & Nori, F.** (2012). QuTiP: an open-source
    Python framework for the dynamics of open quantum systems.
    *Comput. Phys. Commun.* **183**, 1760.
    <https://doi.org/10.1016/j.cpc.2012.02.021>

## Endorsement Marker

**Local candidate framework under active stewardship.** No parity implied with
externally validated laws. This is a Coastline review note within the
Open-Science Harbour, stewarded by U. Warring (AG Schätz, Albert-Ludwigs-
Universität Freiburg). It fixes the conventions binding `iontrap_dynamics` at
this version and is the cited authority for `CONVENTIONS.md` §23–24; it carries
no external endorsement. Licensed under **CC BY-SA 4.0** — Coastline material is
a shareable constraint, not adaptive Sail guidance.

**Convention version:** staged for the v0.3 Convention Freeze (§23–24).
**Workplan reference:** `WP/WP-02-two-mode-motional.md` §7.
