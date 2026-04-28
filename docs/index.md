# Welcome

<section class="hero-panel">
  <p class="hero-kicker">iontrap-dynamics · v0.4.0 — Clos 2016 reproduction landed</p>
  <h1>Open-system quantum dynamics for trapped-ion spin-motion systems</h1>
  <p class="hero-lede">
    A Python library for deterministic, typed, convention-explicit modelling of
    trapped-ion Hamiltonians, motional modes, observables, and cache-verifiable
    results.
  </p>
  <div class="hero-actions">
    <a class="hero-button hero-button-primary" href="getting-started/">Read the quick start</a>
    <a class="hero-button" href="framework/">See the framework</a>
  </div>
  <p class="hero-meta">
    QuTiP reference backend + opt-in JAX / Dynamiqs backend on the same solve() surface.
  </p>
</section>

<div class="grid cards landing-cards" markdown>

-   :material-telescope:{ .lg .middle } __What this is__

    Typed configuration for species, drives, modes, storage policy, result
    metadata, and regression anchors. Public conventions are locked before
    solver breadth.

-   :material-layers-triple:{ .lg .middle } __Three-layer model__

    Physics evolves the quantum state. Apparatus models systematics. Observation
    maps ideal observables to sampled outcomes. Noise does not leak back into
    the Hamiltonian layer.

-   :material-shield-check-outline:{ .lg .middle } __Integrity-first__

    Cache loads are hash-verified. Results carry convention version, backend
    identity, provenance tags, and structured warnings. Silent degradation is
    explicitly forbidden.

-   :material-school-outline:{ .lg .middle } __Pedagogy without hand-waving__

    Clock-School readability is a design constraint. The API is intended to be
    physically legible before it is performance-maximal.

</div>

## Why this project exists

Most open quantum toolkits are intentionally general. `iontrap-dynamics` is not.
Its public identity is domain-specific to trapped-ion spin-motion dynamics while
remaining configuration-general with respect to species, isotope, laser geometry,
mode structure, and measurement protocol.

That means changing from ²⁵Mg⁺ to ⁴⁰Ca⁺, or from an axial single-mode toy model
to an externally supplied normal-mode decomposition, should be a configuration
change rather than a solver rewrite.

## What is already in place

<div class="status-strip" markdown>

`CONVENTIONS.md` v0.2 frozen  
Result schema, cache format, three-layer regression harness  
QuTiP reference backend + JAX / Dynamiqs opt-in backend  
Strict package metadata, CI (lint + mypy strict + tests + pa11y WCAG 2 A)

</div>

At `v0.4.0` the library covers Phase 0, Phase 1, and Phase 2 in full,
plus the Clos 2016 reproduction track:

Foundation (Phase 0):

- Canonical exception hierarchy, convention-version marker, frozen
  result dataclasses with declared storage policy, hash-verified
  cache round-trips, analytic reference formulas, backend-agnostic
  invariant diagnostics.

Configuration & dynamics (Phase 1):

- Atomic-physics Pauli operators (`sigma_z_ion` etc. — sign-flipped
  against QuTiP's native convention).
- `IonSpecies`, `DriveConfig`, `ModeConfig`, `IonSystem` composition
  with cross-validation at construction.
- `HilbertSpace` implementing the §2 tensor ordering, operator
  embedding, motional primitives (a, a†, n̂).
- Full Hamiltonian builder surface: carrier, red / blue sideband,
  MS gate — exact and detuned forms — plus `modulated_carrier`
  (time-dependent envelope primitive) and `two_ion_{red,blue}_sideband`.
- State prep (coherent, squeezed, squeezed-coherent), observable
  registry, entanglement evaluators (concurrence, EoF, log-negativity).

Measurement + systematics (Phase 1, v0.2 frozen):

- Sampling channels (Bernoulli, Binomial, Poisson), detector model
  with efficiency / dark counts / thresholding, protocol composers
  (`SpinReadout`, `ParityScan`, `SidebandInference`), Wilson /
  Clopper–Pearson intervals (§17).
- Jitter and drift primitives (Rabi / detuning / phase), SPAM
  preparation errors (§18).

Performance + JAX backend (Phase 2):

- `sequences.solve(backend="jax", ...)` routes through
  `dynamiqs.sesolve` / `dynamiqs.mesolve` with all three `StorageMode`
  values supported and the same Fock-saturation check (§13) as the
  QuTiP path.
- Every time-dependent Hamiltonian builder accepts `backend="jax"`
  and emits a Dynamiqs `TimeQArray` for the five canonical families
  (carrier, RSB, BSB, MS gate, modulated carrier with user-supplied
  `envelope_jax`).
- Cross-backend numeric equivalence validated at 1e-3 tolerance;
  honest performance characterisation (null result at dim ≥ 100 /
  5000 steps on CPU) in `docs/benchmarks.md`.

## Boundaries

<div class="grid cards landing-cards" markdown>

-   __In scope__

    Carrier and sideband physics, Mølmer-Sørensen style interactions, open-system
    evolution, observables, and later typed measurement/systematics layers.

-   __Out of scope__

    Trap geometry simulation, molecular dynamics of ion crystals, EM field
    modelling, pulse compilers, and full laboratory digital twins.

</div>

## Current posture

This is a planning-to-implementation transition point, not a finished research
package. The repository is already opinionated about conventions, provenance,
and failure semantics, because those are the parts that become expensive to fix
after solver code exists.

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
