# Task Card — Classical structural-dynamics add-on: foundation survey & scoping

**Authored from:** the `iontrap-dynamics` side, as a scoping / build-vs-adopt decision aid
**Topic:** structural dynamics of large ion ensembles (10²–10⁴ ions) tuned near the
phase transition from ordered **Coulomb crystals** to disordered **non-neutral plasmas**
**Surveyed:** 2026-06-16 — 32 repos `gh api`-verified for license + maintenance and scored
against a 7-criterion rubric (§6), ranked to 29 rows after consolidating duplicate `pylion`
fork entries (multi-agent workflow `coulomb-crystal-code-survey`; ~0.8M subagent tokens —
**do not casually re-run**)
**Status:** v1.0 — **FROZEN 2026-06-17** (deliberation & survey record). **Recommendation: clean-room/self-owned implementation
by default; abstraction kept as an interface *seam* (not a framework) behind concrete ion-trap
presets, with immutable result records and per-interaction-class eigensolve strategy;
neutral-atom backgrounds deferred to a force/energy seam (no collision API yet); name the
concrete deliverable first. v0.5 adds explicit out-of-scope (§2.1), first-slice acceptance
criteria (§9.1), and operational metadata (§9.7). Forwarded to `uwarring82/iontrap-structure` to seed
implementation — see the freeze note below.** Captures the landscape so the build-vs-adopt
question is not re-litigated from scratch.

---

> ## ❄️ Freeze note — v1.0, 2026-06-17
>
> This card is **frozen** and closed as the authoritative **deliberation & survey record**. It is
> being **forwarded to `uwarring82/iontrap-structure`** — a new MIT-licensed sibling repo — to seed
> implementation. **Further design, decisions, and iteration continue there, not by editing this
> card.** This copy stays in `iontrap-dynamics` as the origin / provenance record; if it ever needs
> to change, supersede it with a dated successor rather than reopening the freeze.
>
> **Resolved at freeze:** classical/quantum split (separate MIT sibling producing `ModeConfig`);
> clean-room / self-owned by default; abstraction as an interface *seam* behind ion-trap presets;
> neutral background deferred to a force/energy seam; **host/name = `uwarring82/iontrap-structure`** (§9.6).
>
> **Carried forward as the new repo's opening backlog:**
> - First slice + acceptance criteria (§9.1) → decoupled analysis layer (§9.2) → operational metadata (§9.7).
> - **Still-open decisions, to resolve in the new repo:** the two §7.2 scoping decisions (second
>   abstraction use case; neutral-background goal) and the **§8 large-N eigensolve spike** (the first
>   hard research risk — settle before committing scope).
> - Optional, non-blocking: the §5 relicensing outreach (W. L. Johnson, J. Zaris).

## 1. Verdict

**Build a fresh, MIT-licensed sibling package — the niche is unfilled — and implement the
core ourselves from equations/papers. Let the internals be a small classical many-body
kernel, but keep the first public path concrete: ion-trap equilibrium → normal modes →
`ModeConfig`. Use existing packages as validation/interoperability references by default,
not as code seeds.**

No surveyed project delivers the four target capabilities — (a) equilibrium positions,
(b) normal modes, (c) MD + laser cooling / bath, (d) phase classification / Γ / zigzag→shell
/ melting — at 10²–10⁴ ions across the order→disorder transition under a usable license.
Everything that *fits the contract* is small-N harmonic/linear (tens of ions, dense
O(N²)–O(N³) kernels, no transition physics); everything that *scales* (LAMMPS, ESPResSo,
HOOMD) emits no mode contract and is mostly copyleft. The code that **actually models
the crystal→plasma / shell regime** carries provenance/licensing risk (§5). Hence:
fresh clean-room build, interoperating with existing tools via a small exporter that
emits `ModeConfig` records for modes and separate structural diagnostics for positions/phases.
The abstraction should sit behind focused trap presets; it must not delay a
James-1998-validated linear-chain export.

This **confirms the classical-vs-quantum split** agreed in deliberation: `iontrap-dynamics`
stays the MIT quantum-dynamics *consumer* of mode structure (Design Principle 2,
[`src/iontrap_dynamics/modes.py`](../src/iontrap_dynamics/modes.py) — "external normal-mode
solver"); the new package is the *producer* it currently delegates to.

## 2. Ownership boundary

| Capability | Owner | Rationale |
|---|---|---|
| Equilibrium positions (1D/2D/3D, RF + anharmonic) | **new sibling** | classical structure; not in the quantum library's coastline |
| Normal-mode (Hessian) eigen-solve → `ModeConfig`-compatible output | **new sibling** | the producer side of the delegated boundary |
| MD with laser cooling / heating / thermal bath | **new sibling** | classical dynamics |
| Phase classification / Γ / zigzag→shell / melting | **new sibling** | statistical-mechanical layer |
| Internal abstract kernel (`particles`, `external potential`, `interaction`; `bath` is a **post-MD-slice placeholder**, absent from the equilibrium+modes first slice) | **new sibling** | reusable implementation *seam* behind concrete presets; immutable result records (not a fluent framework); eigensolve strategy chosen per interaction class — not a performance-uniformity claim |
| Neutral atom background / hybrid atom-ion coupling | **future extension or separate package** | useful physics, but a new collision/reaction domain; expose a force/energy seam now, defer the collision API and implementation |
| Quantum spin–motion dynamics that *consume* modes | **iontrap-dynamics** | unchanged; DP2 still bars guessing mode structure |
| The shared mode interface (`ModeConfig`: per-mode eigvec `(N,3)` with ∑ᵢ‖b_{i,m}‖²=1, ω in **rad/s**) | **iontrap-dynamics** (CONVENTIONS §10/§11) | already exists; the sibling exports to it |
| Structural state / diagnostics (`positions: (N,3)`, trap metadata, Γ, order parameters, phase labels) | **new sibling** | not part of `ModeConfig`; carried separately and optionally serialized beside exported modes |

**License rule.** The sibling is MIT and feeds an MIT core → keep owned implementation
MIT by default; never vendor or link copyleft; never copy source from repositories whose
reuse rights are not proven. Reimplement algorithms from underlying papers/spec notes, not
from GPL/NOASSERTION code. If Apache/MIT/BSD code is ever vendored or translated, preserve
the upstream license, copyright notices, change notices, and any `NOTICE` obligations rather
than pretending the result is pure MIT. Use LAMMPS only as an out-of-process subprocess.

**Clean-room discipline.** For core algorithms, first write a short equations/spec note
from citable papers and local convention requirements, then implement against that note.
The person writing owned code reads **papers/specs only**; an encumbered (GPL/NOASSERTION)
repository may be consulted — ideally by someone other than the implementer — only for a
**black-box numerical comparison** after the spec is frozen, never as implementation source.
Patent risk is low: the methods are published academic physics, mostly decades old
(James 1998, Dubin–O'Neil 1999, Wang–Keith–Freericks 2013). Clean-room is slower than porting,
but the slowdown is bounded — the physics is textbook — and it is the price of a clean MIT licence.

### 2.1 Explicitly out of scope (v0.1)

To resist scope creep, the first release deliberately excludes:

- **Neutral-atom backgrounds / hybrid atom-ion traps** and any `CollisionModel` (§7.2 — deferred).
- **Multi-species reaction chemistry** (charge exchange, molecular-ion formation, rate tables).
- **Arbitrary electrode-potential / BEM field solvers** — consume an externally supplied potential
  (or a simple analytic/anharmonic trap); do not build an electrostatics solver.
- **A public DSL / general `System` API** — the §7.1 sketch is internal-shape only, behind presets.
- **Large-N MD and the full eigensolve at N≳10³** — gated behind the §8 spike.
- **Quantum spin–motion dynamics** — stays in `iontrap-dynamics` (the consumer).

## 3. Permissive references — validate against these, do not port by default

| Package | License | Validation value | Still build ourselves |
|---|---|---|---|
| [**oqd-trical**](https://github.com/OpenQuantumDesign/oqd-trical) | Apache-2.0 | equilibrium (SLSQP) + Hessian modes; **ω already rad/s**; useful oracle for linear/harmonic single-species cases | trap-potential API; Coulomb gradient/Hessian; mass-symmetrized eigenproblem; `ModeConfig` exporter (its eigvecs return **axis-blocked `(3N,)`** as `[x₀…x_N, y₀…y_N, z₀…z_N]` → reshape to `(N,3)`); all (c),(d); large-N scaling |
| [**trimos**](https://github.com/carmelom/trimos) | MIT | analytic Coulomb gradient/Hessian behavior; eigvec shape + ∑‖b‖²=1 convention; good small-N cross-check | independent formulas + tests; Hz→rad/s handling; **mass-symmetrized** diagonalization for mixed species; all (c),(d); large-N beyond dense `eig` |
| [**levitated-nanoparticle-…-cooling**](https://github.com/saurabh-gupta97/levitated-nanoparticle-ion-sympathetic-cooling) `legacy/` | MIT | genuine **3D** N-body equilibrium (`root` + Monte-Carlo restart → multiple stable configs) + generalized `eigh(H,M)`, ω in rad/s; useful off-axis oracle | ion-only trap model; M-orthonormal→Euclidean normalization spec; `(N,3)` position/mode serialization; anharmonic/RF equilibrium; all (c),(d) |

Use oqd-trical + trimos as cross-validation targets for the linear/harmonic path.
Use levitated-`legacy/` as a 3D off-axis equilibrium oracle. Do not copy, translate,
or mechanically port code unless a later legal/provenance review deliberately accepts
the upstream license obligations in the new repository.

## 4. Analysis layer (d) — decoupled quick win, ship early

Order parameters don't need the engine; they act on any `(N,3)` configuration:

- [**freud**](https://github.com/glotzerlab/freud) (BSD-3, active): Steinhardt/hexatic order,
  Voronoi, RDF, clustering, solid-liquid. Use as an **optional** dependency.
  **Caveat:** `freud-analysis` 3.5.0 (verified 2026-06-17) requires Python ≥3.12 — gate it
  optional or pin an older 3.x line.
- **Coupling parameter Γ** = q²/(4πε₀ · a · k_BT): inline one-liner; diagnostics structure
  (RDF/S(k)/VACF) patterned from [**Sarkas**](https://github.com/murillo-group/sarkas) (MIT,
  bulk-plasma MD — reference-only; no trap/minimizer/Hessian).
- **Zigzag/kink order parameter + KZM scaling:** pattern from
  [**Yb174-Zigzag**](https://github.com/yyken9299/Yb174-Zigzag-Phase-Transition) (MIT, BAOAB quench).

Ship (d) as a thin MIT module: inline Γ + optional freud delegation. Engine-agnostic.

## 5. Transition physics references (clean-room path)

**The code in this survey that actually models the crystal→plasma / shell-structure
regime is legally/provenance risky as implementation source:**

- [**WesLeeJohnson/mode_analysis**](https://github.com/WesLeeJohnson/mode_analysis) — Penning /
  shell normal modes; **NOASSERTION / no reuse rights proven**.
- [**jzaris/coldatoms_051623**](https://github.com/jzaris/coldatoms_051623) — 3D Penning-trap
  FMM ion-crystal MD (Zaris/Bollinger/Parker, JPP 2025); **NOASSERTION + GPL (cold-atoms)
  ancestry**.

Both implement the canonical **Wang–Keith–Freericks 2013** normal-mode + **Dubin–O'Neil**
shell method, validated against Dubin–O'Neil cloud shapes. The physics is independently
citable from the papers.

→ **Highest-leverage optional move: email W. L. Johnson and J. Zaris requesting an
MIT/BSD relicense and explicit provenance statement.** Securing it would help validation
and comparison. It should not be the implementation path by default: shell-mode physics
should be clean-room from Wang–Keith–Freericks 2013, Dubin–O'Neil, and follow-on papers.

## 6. Full roster (ranked by usefulness as a foundation)

Scoring rubric (the 7 criteria named in the header): scope match · **license / family** ·
health & maintenance · **contract-fit** · dependency footprint · validation pedigree ·
build-on-vs-port. The **Fit** column reports *contract-fit only* — can it emit `(N,3)` modes
normalized per §11 in rad/s: `direct` (already), `adaptable` (needs reshape / Hz→rad/s /
renormalization), `poor` (no mode machinery). Full per-criterion scores live in the workflow
output, not reproduced here.

Verdict legend: **validate** = use as an oracle/cross-check, not a source port by default;
**interop** = out-of-process cross-check; **clean-room ref** = use papers/specs, not source;
**ref** = read for ideas only; **reject** = no value here.

| # | Package | License (family) | Health (last) | Fit | Verdict |
|---|---|---|---|---|---|
| 1 | oqd-trical | Apache-2.0 (perm) | active 2026-06 | adaptable | **validate** (a)+(b) |
| 2 | trimos | MIT (perm) | stale 2024-04 | direct* | **validate** (a)+(b) |
| 3 | levitated-nanoparticle-cooling `legacy/` | MIT (perm) | active 2025-12 | adaptable | validate (a)+(b), 3D, rad/s |
| 4 | WesLeeJohnson/mode_analysis | NOASSERTION | stale 2026-03 | adaptable | clean-room ref — relicense only for validation/source comparison |
| 5 | Sarkas | MIT (perm) | stale 2024-07 | poor | ref (Γ, RDF, S(k)) |
| 6 | freud | BSD-3 (perm) | active 2026-06 | poor | **interop** (layer d) |
| 7 | IonChainTools (SQRLab/ReorderSimulation) | NOASSERTION | stale 2024-09 | adaptable | clean-room ref (1D oracle idea) — relicense for comparison |
| 8 | IonSim.jl | MIT (perm) | active 2025-11 | poor | ref (linear-chain oracle, Julia) |
| 9 | ion-cooling / DopplerCG (jbqubit) | NOASSERTION | active 2026-03 | adaptable | clean-room ref — relicense for comparison; James-1998-validated |
| 10 | Sion (surface-ion) | GPL-3.0 (copyleft) | active 2026-06 | adaptable | ref — copyleft + Windows-only |
| 11 | pylion (all forks) | MIT (perm) | maintained 2025-09 | poor | **interop** (MD cross-check) |
| 12 | jzaris 3D Penning FMM | NOASSERTION | dead 2023-06 | adaptable | clean-room ref — relicense + provenance review |
| 13 | arbitrary-electrostatic-trap | NOASSERTION | dead 2024-05 | adaptable | ref |
| 14 | trapped-ion-numerics (ajrazander) | NOASSERTION | dead 2023-05 | adaptable | ref |
| 15 | LAMMPS | GPL-2.0 (copyleft) | active | adaptable | interop (subprocess) |
| 16 | Yb174-Zigzag | MIT (perm) | maintained 2026-03 | poor | ref (BAOAB quench, kink count) |
| 17 | cold-atoms (d-meiser) | GPL-3.0 (copyleft) | dead 2018 | poor | ref |
| 18 | electrode (NIST) | GPL-3.0 (copyleft) | dead 2017 | poor | ref |
| 19 | Coulomb-crystal-CNN | MIT (perm) | stale 2025-03 | poor | ref (cooling/RF kicks) |
| 20 | ion-trap-qsim (HaeffnerLab) | GPL-2.0 (copyleft) | dead 2016 | poor | ref |
| 21 | HOOMD-blue | BSD-3 (perm) | active | poor | reject (no modes) |
| 22 | diff-evol-ions | NOASSERTION | dead 2018 | adaptable | ref |
| 23 | ESPResSo | GPL-3.0 (copyleft) | active | poor | reject |
| 24 | bem (NIST) | mixed GPL + non-free Triangle | dead 2017 | poor | reject |
| 25 | IonMD | GPL-3.0 (copyleft) | dead 2017 | poor | reject |
| 26 | Penning.jl | MIT (perm) | active 2025-11 | poor | reject (no Hessian/eq, Julia) |
| 27 | Non-Neutral-Plasma-Simulator | NOASSERTION | dead 2022 | poor | reject (electrons) |
| 28 | icc-julia | NOASSERTION | dead 2019 | poor | reject (beam) |
| 29 | iheuze/Coulomb-Crystals | GPL-3.0 (copyleft) | stale 2024 | poor | reject (Mathematica) |

\* trimos is "direct" only on eigvec shape/normalization; still needs Hz→rad/s and a
mass-symmetrized rediagonalization (diagonalize the symmetric `M^{-1/2} H M^{-1/2}`, not the
non-symmetric `M⁻¹H` — equivalent for equal masses, not for mixed species), and does not scale
past ~tens of ions.

**License taxonomy.** Permissive (validation, interop, and deliberate vendoring/porting
possible with attribution/notice obligations): oqd-trical (Apache), trimos,
levitated-cooling, IonSim.jl, Yb174-Zigzag, Coulomb-crystal-CNN, Sarkas, pylion (all forks),
Penning.jl (MIT); freud, HOOMD (BSD-3). Strong copyleft (subprocess/interop only, never link;
clean-room algorithms from papers): Sion, electrode, cold-atoms, ESPResSo, iheuze, IonMD
(GPL-3); LAMMPS, ion-trap-qsim (GPL-2); bem (GPL + non-free Triangle = hardest blocker).
NOASSERTION = no detected/reusable license in GitHub metadata; treat as legally unusable
implementation source until license/provenance is explicit: WesLeeJohnson,
IonChainTools, ion-cooling/DopplerCG, jzaris, arbitrary-electrostatic-trap, trapped-ion-numerics,
diff-evol-ions, Non-Neutral-Plasma-Simulator, icc-julia.

## 7. Scope extensions under consideration

> *This section is a forward-looking design sketch, not part of the survey record. Once the
> two open scoping decisions below settle (second use case; neutral-background goal), §7 should
> graduate into its own design note, leaving this card as the survey it is titled to be.*

### 7.1 Abstract layer: cautious yes

The sibling should not be named or designed as only "Coulomb crystals in ion traps" if a
slightly more general core costs little. The reusable narrow waist is a set of **pure
functions over immutable records** — matching the parent project's house style (frozen
dataclasses + the CONVENTIONS §14 reproducibility ladder), *not* a fluent stateful builder:

*Non-binding internal shape — illustrative, not an API commitment or acceptance scope.*

```python
# First slice = equilibrium + normal_modes ONLY (frozen finite result records).
system = System(particles, confinement_potential, interaction_potential)
eq     = equilibrium(system)        # -> EquilibriumResult (frozen)
modes  = normal_modes(eq)           # -> ModeResult (frozen); .to_modeconfig() exports iontrap_dynamics.ModeConfig
diag   = order_parameters(eq)       # -> StructuralDiagnostics (frozen); also accepts a trajectory

# Post-MD-slice (capability c), shown for shape only — NOT built first:
#   system = System(..., bath=...)      # bath = the deferred force/energy seam (§7.2)
#   traj   = thermalize(system, eq, T)  # a trajectory is NOT one frozen object: immutable run
#                                       # metadata + append-only / chunked samples (repr TBD when MD lands)
```

Internally, this suggests stable primitives: `ParticleSpecies`, `Configuration`,
`ExternalPotential`, `InteractionPotential`, `EnergyModel` with optional gradient/Hessian,
`ForceModel`, `Bath`, `EquilibriumSolver`, `ModeSolver`, and `ModeConfig` adapters. **The core
math is not ion-specific** — the same equilibrium / Hessian / mode machinery applies to other
confined charged-particle systems (Penning traps, RF+static hybrids, dusty plasmas, charged
nanoparticles) — but that breadth is a claim to be *earned per demonstrated case*, not asserted
up front (cf. the project's "document the boundaries of every claim" ethos).

**The abstraction is an interface seam, not a performance-uniformity claim.** The hard part of
the large-N problem (§8) is interaction-specific: bare Coulomb `1/r` is long-range → **dense**
Hessian → dense/FMM eigensolve; screened (Yukawa/Debye) interactions, as in most dusty plasmas,
are short-range → **sparse** Hessian → neighbour-list/cutoff scaling. So `InteractionPotential`
unifies the *energy/gradient interface* only; the eigensolve and scaling strategy are selected
per interaction class, never promised uniform across them.

**Constraint:** the abstraction sits behind focused presets, not in front of the first
milestone. Do not ship a DSL before physics output. The first acceptance test remains:
linear Paul trap preset → James-1998 equilibrium/modes → `ModeConfig` records with
CONVENTIONS §10/§11 normalization and rad/s units. Let the second use case (likely Penning
shell modes or dusty-plasma-style confined charges) pressure-test the abstraction before
freezing it.

### 7.2 Neutral atom background: defer, but leave hooks

Immersing ions in a neutral atom background unlocks real hybrid atom-ion physics:
sympathetic cooling, Langevin/orbiting collisions, micromotion-interruption heating,
charge exchange, reactive collisions, molecular-ion formation, and hybrid traps with
variable neutral density `n_n(r,t)`, temperature `T_n(r,t)`, flow, mass, polarizability,
and internal state.

This is a larger domain than the original structural-dynamics card. It needs either a
collision operator or its own atom dynamics, species-dependent cross sections/rate tables,
reaction channels, and timestep handling for rare but stiff collision events. Treat it as
a future extension or a sibling package that consumes the isolated-ion structural engine.

Recommended staged model:

| Tier | Neutral model | Scope |
|---|---|---|
| A | Parameterized bath field | density/temperature/flow fields; cooling/heating/diffusion/loss rates |
| B | Stochastic binary collisions | Langevin-regime elastic collisions and micromotion-interruption heating |
| C | Explicit neutral atoms | only when back-action, depletion, correlations, or cloud dynamics matter |
| D | Quantum-scattering rate tables | ultracold/reactive channels from MQDT or experiment-specific data |

**Tier B is not an incremental add to the structural engine.** Micromotion-interruption
heating (the effect that sets the ion–atom sympathetic-cooling temperature floor) is an
intrinsically **RF / time-dependent** effect; it requires full time-dependent RF integration,
not the static pseudopotential the equilibrium+modes core uses. Tier A (a parameterized bath
acting on the existing dynamics) is cheap; the RF-dependent Tier B crosses into machinery the
first slices do not build — so **Tier A must precede Tier B**.

For now, leave a **seam, not an API**: ensure the force/energy entry point admits an additive
external contribution (conservative neutral mean-field effects can enter an `EnergyModel`; a
dissipative/stochastic term can enter a `ForceModel`), so a bath *can* plug in later. **Do not
design a `CollisionModel` interface until Tier A is actually built** — the collision domain is
the least understood part and the most likely to bake in wrong assumptions. Stochastic
collisions and reactions belong in MD/bath modules, never in `ModeConfig`.

| Direction | Fit with this card | First-slice impact | Recommendation |
|---|---|---|---|
| Abstract layer | Moderate — same core math, broader names/interfaces | Medium — design care, little extra physics if hidden behind presets | **Do cautiously**; concrete ion-trap output first |
| Neutral atom background | Low-to-moderate — important hybrid physics, but beyond structure/modes | High if implemented now | **Defer**; leave a force/energy seam (not a collision API) and add after the isolated-ion engine works |

Open scoping decisions before implementation freeze:

1. **Second use case for the abstraction:** Penning shell modes, dusty plasmas, charged
   nanoparticles, or another confined-particle system.
2. **Neutral-background goal:** sympathetic cooling/thermalization (bath/thermostat) versus
   reactive collisions/chemistry (cross sections, channels, and rate tables).

## 8. First hard research risk

**Large-N scaling of the *eigensolve*, not the minimization.** FMM (jzaris) accelerates
equilibrium-finding but not diagonalizing a dense 3N×3N Hessian. At N~10³–10⁴ that is the real
wall, and no surveyed code has solved it for *modes*. Spike a sparse/iterative or
structure-exploiting eigensolver **before** committing to scope.

## 9. Recommended first slice & next steps

1. **First slice (smallest thing producing real `ModeConfig` output):** implement from
   scratch an MIT scaffold for the **linear/harmonic equilibrium+modes** path: trap model,
   Coulomb energy/gradient/Hessian, optimizer wrapper, mass-symmetrized eigenproblem,
   Euclidean mode normalization, and `ModeConfig` export. Add finite-difference Hessian
   tests and validate against a **James-1998 linear-chain oracle** plus oqd-trical/trimos
   numerical comparisons. Use minimal generic data primitives internally, but no public DSL
   before this path works. *Defer* large-N and full MD until the eigensolve spike (§8) lands.
   **Acceptance criteria** (initial targets; tighten as the regression ladder matures): compare
   against the James-1998 analytic linear chain and the oqd-trical/trimos oracles on — (i)
   equilibrium positions (≤1e-6 rel. error vs analytic for N≤10); (ii) mode frequencies in
   **rad/s** (≤1e-6 rel.); (iii) eigenvectors up to per-mode sign and rotation within degenerate
   subspaces (subspace overlap ≥1−1e-8). Independently assert the `ModeConfig` contract:
   ∑ᵢ‖b_{i,m}‖²=1 (≤1e-10), cross-mode orthogonality, ω>0, and a finite-difference check of the
   analytic Hessian (≤1e-7).
2. **Decoupled analysis layer:** inline Γ and lightweight order-parameter helpers; keep
   freud optional because `freud-analysis` 3.5.0 targets Python 3.12+ (verified 2026-06-17).
3. **Relicensing outreach (§5), optional but useful:** draft emails to W. L. Johnson and
   J. Zaris requesting MIT/BSD relicensing and provenance statements for validation/reference
   use. Do not block the clean-room implementation on this.
4. **Python-version target:** decide before pinning optional analysis extras.
5. **Neutral-background scope guard:** ensure the force/energy seam admits an external additive
   contribution; do **not** design a `CollisionModel` API or implement `NeutralBath`/hybrid
   atom-ion dynamics until the isolated-ion engine works — and Tier A (parameterized bath) must
   precede any RF-dependent Tier B (§7.2).
6. **Host/name decision — RESOLVED (freeze, 2026-06-17):** `uwarring82/iontrap-structure` — the
   concrete ion-trap-first name. Do **not** adopt a general confined-particles name until a second
   use case earns the generality (§7.1); a package named for breadth it doesn't yet have is a claim
   it can't back. MIT-owned code, interoperating via exported `ModeConfig` objects and a separate
   structural state/diagnostics schema.
7. **Operational metadata (when scaffolding):** Python ≥3.11 to match the parent library (gate
   freud, which needs ≥3.12, as an optional extra); `pytest` with a regression/oracle ladder
   mirroring the parent's reproducibility tiers (CONVENTIONS §14); dependency policy = **no
   copyleft runtime deps**, all heavy/analysis deps optional and gated; MIT `src/` per the
   parent's split-licence model.

## 10. References (clean-room sources)

- D. F. V. James, *Quantum dynamics of cold trapped ions…*, Appl. Phys. B **66**, 181 (1998) — linear-chain equilibrium + modes (validation oracle).
- C.-C. J. Wang, A. C. Keith, J. K. Freericks, PRA **87**, 013422 (2013) — Penning normal modes.
- D. H. E. Dubin, T. M. O'Neil, Rev. Mod. Phys. **71**, 87 (1999) — non-neutral plasma / shell structure / Γ.
- S. Fishman, G. De Chiara, T. Calarco, G. Morigi, PRB **77**, 064111 (2008) — linear→zigzag transition.
- M. Tomza et al., Rev. Mod. Phys. **91**, 035001 (2019) — cold hybrid ion-atom systems review.
- A. T. Grier et al., PRL **102**, 223201 (2009) — cold ion-atom collisions in the semiclassical/Langevin regime.
- R. Côté and A. Dalgarno, PRA **62**, 012709 (2000) — ultracold atom-ion collision theory.
- Z. Idziaszek, T. Calarco, P. S. Julienne, A. Simoni, PRA **79**, 010702(R) (2009) — MQDT/quantum theory for ultracold atom-ion collisions.
