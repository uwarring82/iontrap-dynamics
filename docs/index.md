# Welcome

<section class="hero-panel">
  <p class="hero-kicker">iontrap-dynamics · Phase 0 scaffold</p>
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
    QuTiP-first architecture now. JAX/Dynamiqs-ready backend boundary later.
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

`CONVENTIONS.md` v0.1-draft  
Result schema and cache format  
Analytic and invariant regression anchors  
Strict package metadata and CI scaffold

</div>

The current package surface is still early, but it already exposes the core
contracts future solver code will depend on:

- Canonical exception hierarchy
- Convention-version marker
- Frozen result dataclasses and storage policy
- Hash-verified cache round-trips
- Analytic reference formulas
- Backend-agnostic invariant diagnostics

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
